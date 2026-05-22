"""
Rotated GPTQ Core
==================
기존 gptq.py와 동일하나 Hessian에 V 회전 적용:

    H̃ = V H V^T

weight는 이미 rotate_weight()로 UWV^T 상태로 들어옴.
GPTQ는 회전된 공간에서 동작.
"""

import math
import torch
import torch.nn as nn
from quantize import quantize, find_params


class RotatedGPTQ:
    """
    하나의 Linear layer를 Rotated GPTQ로 양자화.

    사용법:
        handler = RotatedGPTQ(layer, U, V)
        handler.add_batch(inp, out)   # Hessian 누적 (원래 공간 입력)
        Q, scale, zero, loss = handler.quantize(bits=4)
    """

    def __init__(self, layer: nn.Linear, U: torch.Tensor, V: torch.Tensor):
        self.layer = layer
        self.dev   = layer.weight.device
        self.dtype = layer.weight.dtype

        W = layer.weight.data
        self.d_row, self.d_col = W.shape

        self.U = U.to(self.dev).float()   # (d_row,) sign vector
        self.V = V.to(self.dev).float()   # (d_col, d_col) orthogonal

        # Hessian H = 2 X X^T  (원래 공간)
        self.H        = torch.zeros((self.d_col, self.d_col), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        """
        V @ X 로 회전된 입력으로 Hessian 직접 계산.
        H̃ = 2(VX)(VX)^T = V(2XX^T)V^T = VHV^T
        → quantize()에서 별도 VHV^T 변환 불필요.
        """
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = self.V @ inp.float().t()   # (d_col, n_tokens) → V @ X
        n = inp.shape[1]
        self.H = (self.nsamples * self.H + 2 * inp @ inp.t()) / (self.nsamples + n)
        self.nsamples += n

    def quantize(
        self,
        bits:      int   = 4,
        blocksize: int   = 128,
        percdamp:  float = 0.01,
        groupsize: int   = -1,
        sym:       bool  = False,
        actorder:  bool  = False,
    ):
        """
        Rotated GPTQ Algorithm 1:
          1. W_rot = U * W * V^T  (이미 layer.weight에 반영되어 있어야 함)
          2. H̃ = V H V^T
          3. 회전된 공간에서 GPTQ 실행
          4. Q_rot 반환 (layer.weight에 덮어씌움)
        """
        # ── W_rot: 이미 layer.weight = UWV^T 상태라고 가정 ──────────────────
        W = self.layer.weight.data.clone().float()   # (d_row, d_col) = UWV^T

        # H̃ = VHV^T: add_batch에서 이미 V@X로 누적 → 그대로 사용
        H = self.H.float()

        # ── dead weight ────────────────────────────────────────────────────
        dead = (torch.diag(H) == 0)
        H[dead, dead] = 1.0
        W[:, dead]    = 0.0

        # ── Cholesky Reformulation ─────────────────────────────────────────
        damp_val = percdamp * torch.mean(torch.diag(H))
        diag_idx = torch.arange(self.d_col, device=self.dev)
        H[diag_idx, diag_idx] += damp_val

        L    = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

        # ── actorder ───────────────────────────────────────────────────────
        if actorder:
            perm    = torch.argsort(torch.diag(H), descending=True)
            W       = W[:, perm]
            Hinv    = Hinv[:, perm][perm, :]
            invperm = torch.argsort(perm)
        else:
            perm = invperm = None

        # ── groupsize scale/zero 저장 ───────────────────────────────────────
        maxq = 2 ** bits - 1
        n_groups = math.ceil(self.d_col / groupsize) if groupsize > 0 else 1

        scale_all = torch.zeros((self.d_row, n_groups), device=self.dev)
        zero_all  = torch.zeros((self.d_row, n_groups), device=self.dev)

        if groupsize <= 0:
            scale, zero = find_params(W, bits, perchannel=True, sym=sym)
            scale_all[:, 0] = scale.squeeze(1)
            zero_all[:, 0]  = zero.squeeze(1)

        Q      = torch.zeros_like(W)
        Losses = torch.zeros_like(W)

        # ── Algorithm 1 (회전된 공간) ───────────────────────────────────────
        for i1 in range(0, self.d_col, blocksize):
            i2    = min(i1 + blocksize, self.d_col)
            count = i2 - i1

            W1    = W[:, i1:i2].clone()
            Q1    = torch.zeros_like(W1)
            Err1  = torch.zeros_like(W1)
            Loss1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for j_loc in range(count):
                j_global = i1 + j_loc
                col      = W1[:, j_loc]
                d        = Hinv1[j_loc, j_loc]

                if groupsize > 0 and j_global % groupsize == 0:
                    g_idx  = j_global // groupsize
                    g_end  = min(j_global + groupsize, self.d_col)
                    scale, zero = find_params(
                        W[:, j_global:g_end], bits, perchannel=True, sym=sym
                    )
                    scale_all[:, g_idx] = scale.squeeze(1)
                    zero_all[:, g_idx]  = zero.squeeze(1)
                elif groupsize <= 0:
                    g_idx = 0
                    scale = scale_all[:, 0:1]
                    zero  = zero_all[:, 0:1]

                q = quantize(
                    col.unsqueeze(-1), scale, zero, maxq
                ).squeeze(-1)

                Q1[:, j_loc]    = q
                err             = (col - q) / d
                Loss1[:, j_loc] = err ** 2
                W1[:, j_loc:]  -= torch.ger(err, Hinv1[j_loc, j_loc:])
                Err1[:, j_loc]  = err

            Q[:, i1:i2]      = Q1
            Losses[:, i1:i2] = Loss1 / 2
            W[:, i2:]       -= Err1 @ Hinv[i1:i2, i2:]

        if actorder:
            Q = Q[:, invperm]

        Q         = Q.to(self.dtype)
        scale_all = scale_all.to(self.dtype)
        zero_all  = zero_all.to(self.dtype)

        return Q, scale_all, zero_all, Losses

    def free(self):
        del self.H
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
