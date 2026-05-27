"""
Rotated GPTQ Core v2
=====================
목적함수: L = ||[UWV^T - Q(UWV^T)] VX||_F^2

핵심 설계:
  - H̃ = VHV^T: add_batch에서 V@X로 직접 수집
  - W는 W 공간 유지 (WV^T 상태)
  - 매 column 양자화 시에만 U 회전 적용:
      col_rot = U @ col  →  Q(col_rot)  →  q = U^T @ q_rot
  - 오차는 W 공간에서 전파:
      err = (col - q) / d
  - H̃^{-1}은 그대로 사용 (U: d_row 방향, H̃^{-1}: d_col 방향으로 독립)
"""

import math
import torch
import torch.nn as nn
from quantize import quantize, find_params, quantize_nf, find_params_nf, NF2_GRID, get_nf_grid, quantize_e8, quantize_e8_indexed, find_scale_e8, get_e8p_codebook


class RotatedGPTQ:

    def __init__(self, layer: nn.Linear, U: torch.Tensor, V: torch.Tensor, restore_u: bool = True, use_e8: bool = False):
        self.layer = layer
        self.dev   = layer.weight.device
        self.dtype = layer.weight.dtype

        W = layer.weight.data
        self.d_row, self.d_col = W.shape

        self.U = U.to(self.dev).float() if U is not None else None
        self.restore_u = restore_u
        self.use_e8     = use_e8  # U=None이어도 E8 적용 가능
        self.V = V.to(self.dev).float()

        self.H        = torch.zeros((self.d_col, self.d_col), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp: torch.Tensor, out: torch.Tensor):
        """
        H̃ = VHV^T 직접 수집.
        inp = X (원래 공간) → V@X → H̃ = 2(VX)(VX)^T
        """
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        inp = inp.float().t()            # (d_col, n)
        if self.V.dim() == 2:
            inp = self.V @ inp           # V@X: full rotation
        else:
            inp = self.V.unsqueeze(1) * inp  # elementwise: 1D V
        n   = inp.shape[1]
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
        W 공간 유지 + 매 column 양자화 시에만 U 회전.

        layer.weight는 WV^T 상태로 들어와야 함.
        (rotate_weight에서 U@W@V^T로 회전하지 않고 W@V^T만 적용)
        """
        W = self.layer.weight.data.clone().float()   # (d_row, d_col) = WV^T
        H = self.H.float()                           # H̃ = VHV^T

        # ── dead weight ───────────────────────────────────────────────────
        dead = (torch.diag(H) == 0)
        H[dead, dead] = 1.0
        W[:, dead]    = 0.0

        # ── Cholesky ─────────────────────────────────────────────────────
        damp_val = percdamp * torch.mean(torch.diag(H))
        diag_idx = torch.arange(self.d_col, device=self.dev)
        H[diag_idx, diag_idx] += damp_val

        L    = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        Hinv = torch.linalg.cholesky(Hinv, upper=True)

        # ── actorder ─────────────────────────────────────────────────────
        if actorder:
            perm    = torch.argsort(torch.diag(H), descending=True)
            W       = W[:, perm]
            Hinv    = Hinv[:, perm][perm, :]
            invperm = torch.argsort(perm)
        else:
            perm = invperm = None

        # ── groupsize ────────────────────────────────────────────────────
        maxq     = 2 ** bits - 1
        n_groups = math.ceil(self.d_col / groupsize) if groupsize > 0 else 1
        scale_all = torch.zeros((self.d_row, n_groups), device=self.dev)
        zero_all  = torch.zeros((self.d_row, n_groups), device=self.dev)

        nf_grid  = get_nf_grid(bits)   # 2→NF2, 3→NF8, 4→NF16, else None
        use_nf   = nf_grid is not None
        use_e8vq = self.use_e8    and (bits == 2) and (self.d_row % 8 == 0)


        # E8P per-layer scalar scale (QuIP# 방식)
        if use_e8vq:
            W_rot_all = self._apply_U(W)
            scale_e8_layer = (W_rot_all.square().mean().sqrt() / 0.9
                              ).clamp(min=1e-8).to(W.device)

        if groupsize <= 0:
            W_rot_full = self._apply_U(W)
            if use_nf:
                scale = find_params_nf(W_rot_full, perchannel=True, nf_grid=nf_grid)
                zero  = torch.zeros_like(scale)
            else:
                scale, zero = find_params(W_rot_full, bits, perchannel=True, sym=sym)
            scale_all[:, 0] = scale.squeeze(1)
            zero_all[:, 0]  = zero.squeeze(1)

        Q      = torch.zeros_like(W)
        Losses = torch.zeros_like(W)

        # ── Algorithm 1 ──────────────────────────────────────────────────
        for i1 in range(0, self.d_col, blocksize):
            i2    = min(i1 + blocksize, self.d_col)
            count = i2 - i1

            W1    = W[:, i1:i2].clone()    # W 공간
            Q1    = torch.zeros_like(W1)
            Err1  = torch.zeros_like(W1)
            Loss1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]



            for j_loc in range(count):
                j_global = i1 + j_loc
                col      = W1[:, j_loc]    # W 공간 column (d_row,)
                d        = Hinv1[j_loc, j_loc]

                if groupsize > 0 and j_global % groupsize == 0:
                    g_idx = j_global // groupsize
                    g_end = min(j_global + groupsize, self.d_col)
                    W_rot_g = self._apply_U(W[:, j_global:g_end])
                    if use_nf:
                        scale = find_params_nf(W_rot_g, perchannel=True, nf_grid=nf_grid)
                        zero  = torch.zeros_like(scale)
                    else:
                        scale, zero = find_params(W_rot_g, bits, perchannel=True, sym=sym)
                    scale_all[:, g_idx] = scale.squeeze(1)
                    zero_all[:, g_idx]  = zero.squeeze(1)
                elif groupsize <= 0:
                    g_idx = 0
                    scale = scale_all[:, 0:1]
                    zero  = zero_all[:, 0:1]

                # ── 핵심: 양자화 시에만 U 회전 ──────────────────────────
                col_rot = self._apply_U(col.unsqueeze(1))  # (d_row, 1)

                if use_e8vq:
                    # per-layer scalar scale
                    scale_e8_col = scale_e8_layer.expand(self.d_row // 8, 1)
                    q_rot = quantize_e8(col_rot,
                                        scale_e8_col.to(col_rot.dtype))  # (d_row, 1)
                elif use_nf:
                    q_rot = quantize_nf(col_rot, scale, nf_grid)
                else:
                    q_rot = quantize(col_rot, scale, zero, maxq)

                q_rot = q_rot.squeeze(1)
                col_rot = col_rot.squeeze(1)

                if self.restore_u:
                    q = self._apply_Ut(q_rot.unsqueeze(1)).squeeze(1)
                else:
                    q = q_rot

                Q1[:, j_loc]    = q

                # W 공간 순수 오차
                err             = (col - q) / d       # (d_row,)
                Loss1[:, j_loc] = err ** 2

                # W 공간에서 전파, H̃^{-1} 그대로
                # (U: d_row 방향, H̃^{-1}: d_col 방향 → 독립)
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



        # export 모드: E8P index 저장
        if use_e8vq and getattr(self, 'export_mode', False):
            n_blocks = self.d_row // 8
            idx_all  = torch.zeros(n_blocks, self.d_col, dtype=torch.int32)
            for i in range(self.d_col):
                col     = Q[:, i:i+1].float()
                col_rot = self._apply_U(col)
                _, idx  = quantize_e8_indexed(col_rot, scale_e8_layer.expand(n_blocks,1))
                idx_all[:, i] = idx
            self.e8p_idx   = idx_all          # (d_row//8, d_col) int32
            self.e8p_scale = scale_e8_layer   # scalar

        return Q, scale_all, zero_all, Losses

    def _apply_U(self, x: torch.Tensor) -> torch.Tensor:
        """U @ x. U=None이면 identity."""
        if self.U is None:
            return x
        if self.U.dim() == 1:
            return self.U.unsqueeze(1) * x
        else:
            return self.U @ x

    def _apply_Ut(self, x: torch.Tensor) -> torch.Tensor:
        """U^T @ x. U=None이면 identity."""
        if self.U is None:
            return x
        if self.U.dim() == 1:
            return self.U.unsqueeze(1) * x
        else:
            return self.U.t() @ x

    def free(self):
        del self.H
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
