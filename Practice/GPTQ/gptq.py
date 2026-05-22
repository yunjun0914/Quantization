"""
GPTQ Core Algorithm
===================
논문: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"
     Frantar et al., ICLR 2023  (arXiv:2210.17323)

Algorithm 1 (논문 Figure 2) 충실 구현:

  Input : W ∈ R^{d_row × d_col},  H = 2 X X^T,  bits b
  Output: Q ∈ R^{d_row × d_col}  (quantized weights)

  Step 1: Arbitrary Order Insight  ─ left-to-right 고정 순서 사용
  Step 2: Lazy Batch-Updates       ─ blocksize 단위 lazy update
  Step 3: Cholesky Reformulation   ─ H^{-1} 전체를 미리 Cholesky 분해,
                                     반복 Gaussian elimination 회피

Cholesky trick 수식 근거 (논문 Section 4 / Algorithm 1 notation):
  - H^{-1} = U^T U  (upper Cholesky)
  - 열 j 처리 시:
      d_j        = U[j,j]                   # = sqrt([H^{-1}_F]_{jj})
      Q[:,j]     = quant(W[:,j])             # fake-quantize
      err[:,j]   = (W[:,j] - Q[:,j]) / d_j  # scaled error
      W[:,j:]   -= err[:,j:j+1] @ U[j:j+1, j:]  # Schur complement update
  - 블록 완료 후: W[:,i2:] -= Err @ U[i1:i2, i2:]
"""

import math
import torch
import torch.nn as nn
from quantize import Quantizer, quantize, find_params


# ─────────────────────────────────────────────────────────────────────────────
# GPTQ class
# ─────────────────────────────────────────────────────────────────────────────

class GPTQ:
    """
    하나의 Linear layer를 GPTQ 알고리즘으로 양자화.

    사용법:
        gptq = GPTQ(layer)
        # 데이터 forward pass 중 hook으로 Hessian 누적
        gptq.add_batch(inp, out)
        # 양자화 실행
        Q, scale, zero = gptq.quantize(bits=4, blocksize=128)
        # 메모리 해제
        gptq.free()
    """

    def __init__(self, layer: nn.Linear):
        self.layer   = layer
        self.dev     = layer.weight.device
        self.dtype   = layer.weight.dtype

        W = layer.weight.data                 # (d_row, d_col)
        self.d_row, self.d_col = W.shape

        # Hessian H = 2 X X^T  ── (d_col × d_col)
        # 논문 eq.(1): objective = ||W X - Q X||^2_F
        # X: (d_col, n_samples)  →  H = 2 * X @ X^T  (d_col × d_col)
        self.H       = torch.zeros((self.d_col, self.d_col), device=self.dev)
        self.nsamples = 0

    # ── Hessian 누적 ──────────────────────────────────────────────────────────
    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        """
        미니배치 입력으로 Hessian을 온라인 누적.
        inp : (batch, seq_len, d_col)  or  (batch, d_col)

        H = 2 * X X^T  (X: d_col × n_tokens)
        """
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])   # (n_tokens, d_col)
        inp = inp.float().t()                       # (d_col, n_tokens)
        

        n = inp.shape[1]
        # 온라인 평균: H_new = (n_old * H_old + 2 * inp @ inp^T) / (n_old + n)
        # ─ 최종 H = 2 X X^T (전체 평균)
        self.H      = (self.nsamples * self.H + 2 * inp @ inp.t()) / (self.nsamples + n)
        self.nsamples += n

    # ── GPTQ Algorithm 1 ──────────────────────────────────────────────────────
    def quantize(
        self,
        bits:        int   = 4,
        blocksize:   int   = 128,
        percdamp:    float = 0.01,
        groupsize:   int   = -1,
        sym:         bool  = False,
        actorder:    bool  = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        GPTQ Algorithm 1 실행.

        Args:
            bits      : 양자화 비트 수 (2 / 3 / 4 / 8)
            blocksize : lazy batch update 블록 크기 (논문: 128)
            percdamp  : Hessian 대각 damping 비율 (H += percdamp * mean(diag(H)) * I)
            groupsize : weight grouping (-1: per-channel)
            sym       : symmetric quantization 여부
            actorder  : activation magnitude 기준 열 재정렬 (논문 추가 기법)

        Returns:
            Q      : (d_row, d_col) fake-quantized weights
            scale  : (d_row, n_groups) or (d_row, 1)
            zero   : same shape as scale
            losses : (d_row, d_col) per-weight squared error / d^2
        """
        W = self.layer.weight.data.clone().float()   # (d_row, d_col)
        H = self.H.float()                           # (d_col, d_col)

        # ── dead weight 처리 (활성화 0인 열 = 사용 안 된 feature) ──────────────
        dead = (torch.diag(H) == 0)
        H[dead, dead] = 1.0
        W[:, dead]    = 0.0

        # ── Step 3: Cholesky Reformulation ────────────────────────────────────
        # (a) Hessian diagonal damping (수치 안정성)
        damp_val   = percdamp * torch.mean(torch.diag(H))
        diag_idx   = torch.arange(self.d_col, device=self.dev)
        H[diag_idx, diag_idx] += damp_val

        # (b) H^{-1} 계산: H = L L^T → H^{-1} → U^T U = H^{-1}
        #     U[j,j] = sqrt([H^{-1}_F]_{jj})  (Schur complement 대각)
        #     U[j,:] 는 error propagation에 직접 사용
        L    = torch.linalg.cholesky(H)                    # lower triangular L s.t. L L^T = H
        Hinv = torch.cholesky_inverse(L)                   # H^{-1}
        Hinv = torch.linalg.cholesky(Hinv, upper=True)     # upper U s.t. U^T U = H^{-1}
        # 이제 Hinv[j,j] = d_j = [H^{-1}_F]_{jj}^{1/2}

        # ── activation order 재정렬 (optional, 논문 Section 5.1) ──────────────
        if actorder:
            # 활성화 크기 내림차순으로 열 정렬 (큰 활성화 → 먼저 양자화)
            perm   = torch.argsort(torch.diag(H), descending=True)
            W      = W[:, perm]
            Hinv   = Hinv[:, perm][perm, :]   # symmetric reordering
            invperm = torch.argsort(perm)
        else:
            perm    = None
            invperm = None

        # ── groupsize 용 scale/zero 저장소 ────────────────────────────────────
        maxq = 2 ** bits - 1
        if groupsize > 0:
            n_groups = math.ceil(self.d_col / groupsize)
        else:
            n_groups = 1

        # 먼저 per-channel (전체 열 기준) scale/zero 초기화
        # groupsize > 0 이면 블록 내부에서 재계산
        scale_all = torch.zeros((self.d_row, n_groups), device=self.dev)
        zero_all  = torch.zeros((self.d_row, n_groups), device=self.dev)

        if groupsize <= 0:
            # per-channel: 전체 W 기준으로 scale/zero 한 번 계산
            scale, zero = find_params(W, bits, perchannel=True, sym=sym)
            scale_all[:, 0] = scale.squeeze(1)
            zero_all[:, 0]  = zero.squeeze(1)

        # ── Output tensors ────────────────────────────────────────────────────
        Q      = torch.zeros_like(W)
        Losses = torch.zeros_like(W)

        # ── Algorithm 1: Outer loop (lazy batch) ─────────────────────────────
        #
        #  for i1 = 0, blocksize, 2*blocksize, ... :
        #      i2 = min(i1 + blocksize, d_col)
        #      [inner loop over columns i1..i2-1]
        #      W[:, i2:] -= Err @ Hinv[i1:i2, i2:]   # lazy global update
        # ─────────────────────────────────────────────────────────────────────
        for i1 in range(0, self.d_col, blocksize):
            i2    = min(i1 + blocksize, self.d_col)
            count = i2 - i1

            W1    = W[:, i1:i2].clone()     # (d_row, count) – local copy
            Q1    = torch.zeros_like(W1)    # quantized block
            Err1  = torch.zeros_like(W1)    # errors for lazy update
            Loss1 = torch.zeros_like(W1)

            Hinv1 = Hinv[i1:i2, i1:i2]     # (count, count) upper triangular sub-block

            # ── Algorithm 1: Inner loop (column-wise) ────────────────────────
            #
            #  for j = i1, ..., i2-1 :
            #      j_local = j - i1
            #      d        = Hinv1[j_local, j_local]          # = [H^{-1}_F]_{jj}^{1/2}
            #      Q[:,j]   = quant(W[:,j])                    # quantize
            #      err      = (W[:,j] - Q[:,j]) / d            # scaled quantization error
            #      W[:,j:]  -= err[:,None] @ Hinv1[j_local,j_local:]  # Schur update
            # ─────────────────────────────────────────────────────────────────
            for j_loc in range(count):
                j_global = i1 + j_loc
                col      = W1[:, j_loc]      # (d_row,) = W[:, j]
                d        = Hinv1[j_loc, j_loc]  # d = U[j,j] = sqrt([H^{-1}_F]_{jj})

                # ── groupsize: 그룹 경계마다 scale/zero 재계산 ────────────────
                if groupsize > 0 and j_global % groupsize == 0:
                    g_idx  = j_global // groupsize
                    g_end  = min(j_global + groupsize, self.d_col)
                    scale, zero = find_params(
                        W[:, j_global:g_end], bits, perchannel=True, sym=sym
                    )
                    scale_all[:, g_idx] = scale.squeeze(1)
                    zero_all[:, g_idx]  = zero.squeeze(1)
                elif groupsize <= 0:
                    g_idx  = 0
                    scale  = scale_all[:, 0:1]   # (d_row, 1)
                    zero   = zero_all[:, 0:1]

                # ── Q[:, j] = quant(W[:, j])  (fake-quantize) ─────────────
                q = quantize(
                    col.unsqueeze(-1),    # (d_row, 1)
                    scale,                # (d_row, 1)
                    zero,                 # (d_row, 1)
                    maxq,
                ).squeeze(-1)            # (d_row,)

                Q1[:, j_loc]   = q
                # loss = (w - q)^2 / (2 * H^{-1}_{jj})  = (err)^2 / 2
                #  논문 eq.(1): objective = ||WX - QX||^2  ⟹  column 기여 = (w-q)^2 / H^{-1}_{jj}
                #  Loss1 저장은 (w-q)^2 / d^2  (/2 는 나중에 적용)
                err            = (col - q) / d                  # (d_row,)
                Loss1[:, j_loc] = err ** 2

                # ── Weight update (Schur complement): W[:,j:] -= err ⊗ U[j, j:] ──
                # 논문 eq.(2) δF = -(w-q)/H^{-1}_{jj} * H^{-1}_{j,:}
                # H^{-1}_{F,j,j:} = U[j,j] * U[j,j:] = d * Hinv1[j_loc, j_loc:]
                # ∴ δ = (err/d) * d * Hinv1[j_loc, j:] = err * Hinv1[j_loc, j:]
                W1[:, j_loc:] -= torch.ger(err, Hinv1[j_loc, j_loc:])

                Err1[:, j_loc] = err

            # ── Lazy global update: W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:] ──
            # 블록 전체 에러를 한 번에 나머지 열에 반영
            # 논문 Figure 2 (Algorithm 1) 마지막 줄
            Q[:, i1:i2]      = Q1
            Losses[:, i1:i2] = Loss1 / 2     # 논문 eq.(1)에서 factor 2 보정
            W[:, i2:]       -= Err1 @ Hinv[i1:i2, i2:]

        # ── activation order 역정렬 ───────────────────────────────────────────
        if actorder:
            Q          = Q[:, invperm]
            scale_all  = scale_all  # grouping과 actorder 동시 사용 시 별도 처리 필요
            zero_all   = zero_all

        # ── dtype 복원 ────────────────────────────────────────────────────────
        Q         = Q.to(self.dtype)
        scale_all = scale_all.to(self.dtype)
        zero_all  = zero_all.to(self.dtype)

        return Q, scale_all, zero_all, Losses

    # ── 메모리 해제 ──────────────────────────────────────────────────────────
    def free(self):
        del self.H
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
