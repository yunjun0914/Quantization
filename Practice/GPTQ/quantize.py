"""
Quantization primitives for GPTQ.

논문 Section 3: layer-wise quantization grid는 사전에 고정됨.
symmetric/asymmetric 양쪽 모두 구현, per-channel (row-wise) 기본.
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Low-level quantize / dequantize
# ─────────────────────────────────────────────────────────────────────────────

# NF grids (Gaussian quantile 기반, [-1, 1] 정규화)
NF2_GRID  = torch.tensor([-1.0, -0.2770, 0.2770, 1.0], dtype=torch.float32)
NF8_GRID  = torch.tensor([-1.0, -0.5783, -0.3186, -0.1025, 0.1025, 0.3186, 0.5783, 1.0], dtype=torch.float32)
NF16_GRID = torch.tensor([-1.0, -0.7076, -0.5422, -0.4168, -0.3109, -0.2159, -0.1273, -0.0421,
                            0.0421,  0.1273,  0.2159,  0.3109,  0.4168,  0.5422,  0.7076,  1.0], dtype=torch.float32)

def get_nf_grid(bits: int) -> torch.Tensor:
    if bits == 2: return NF2_GRID
    if bits == 3: return NF8_GRID
    if bits == 4: return NF16_GRID
    return None
NF4_GRID = torch.tensor([
    -1.0, -0.6962, -0.5251, -0.3949, -0.2767, -0.1691, -0.0626,
     0.0626,  0.1691,  0.2767,  0.3949,  0.5251,  0.6962,  1.0,
    -0.6962, -0.5251  # padding to 16
], dtype=torch.float32)


def quantize_nf(x: torch.Tensor, scale: torch.Tensor, nf_grid: torch.Tensor) -> torch.Tensor:
    """
    NormalFloat quantization.
    scale: (d_row, 1) absmax scale
    nf_grid: 고정 quantization 값들 (k개)

    x를 [-1, 1]로 정규화 후 nearest neighbor lookup, dequantize.
    """
    grid = nf_grid.to(x.device)
    x_norm = x / scale.clamp(min=1e-8)                   # (d_row, d_col)
    x_norm = x_norm.clamp(-1, 1)

    # nearest neighbor: (d_row, d_col, k) → argmin
    dists = (x_norm.unsqueeze(-1) - grid.unsqueeze(0).unsqueeze(0)) ** 2
    idx   = dists.argmin(dim=-1)                          # (d_row, d_col)
    x_q   = grid[idx]                                     # (d_row, d_col)
    return x_q * scale                                    # dequantize


def find_params_nf(x: torch.Tensor, perchannel: bool = True, nf_grid: torch.Tensor = None) -> torch.Tensor:
    """
    NF quantization용 MSE clipping scale 계산 (QuaRot / AWQ 방식).
    ratio를 grid search해서 MSE를 최소화하는 scale을 찾음.
    Returns: scale (d_row, 1)
    """
    if nf_grid is None:
        nf_grid = NF2_GRID

    if perchannel:
        x_flat = x.float().reshape(x.shape[0], -1)
    else:
        x_flat = x.float().flatten().unsqueeze(0)

    # rotation 후 분포가 Gaussian에 가까워지므로 std 기반 scale 사용
    # absmax 기반은 outlier에 민감하지만, rotation 후엔 outlier가 줄어들어
    # std * k 형태로 search하는 게 더 적합
    std    = x_flat.std(dim=1, keepdim=True).clamp(min=1e-8)
    grid   = nf_grid.to(x_flat.device)

    best_scale = std * 2.0  # 초기값
    best_mse   = torch.full((x_flat.shape[0], 1), float('inf'), device=x_flat.device)

    for k in [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0]:
        scale_cand = std * k                                 # (d_row, 1)
        x_norm     = (x_flat / scale_cand).clamp(-1, 1)     # (d_row, d_col)
        dists      = (x_norm.unsqueeze(-1) - grid.unsqueeze(0).unsqueeze(0)) ** 2
        idx        = dists.argmin(dim=-1)                    # (d_row, d_col)
        x_q        = grid[idx] * scale_cand                  # dequantize
        mse        = (x_flat - x_q).pow(2).mean(dim=1, keepdim=True)  # (d_row, 1)

        better     = mse < best_mse
        best_scale = torch.where(better, scale_cand, best_scale)
        best_mse   = torch.where(better, mse, best_mse)

    if perchannel:
        return best_scale             # (d_row, 1)
    return best_scale.reshape(1, 1)


def quantize(x: torch.Tensor, scale: torch.Tensor, zero: torch.Tensor, maxq: int) -> torch.Tensor:
    """
    Uniform quantization:  Q = clamp( round(x / scale) + zero,  0,  maxq )
    그 후 fake-quantize:   Q_fp = (Q - zero) * scale

    Args:
        x     : (...,) fp32/fp16 weight tensor
        scale : broadcast-compatible scale
        zero  : broadcast-compatible zero-point  (asymmetric: real offset / symmetric: 0)
        maxq  : 최대 정수값 = 2^bits - 1

    Returns:
        fake-quantized tensor (same dtype/shape as x)
    """
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


def find_params(
    x: torch.Tensor,
    bits: int,
    perchannel: bool = True,
    sym: bool = False,
    weight: bool = True,
    groupsize: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    x의 값 범위로부터 (scale, zero) 계산.

    논문에서는 quantization grid가 고정됨을 가정.
    여기서는 min-max 기반으로 scale/zero를 결정.

    Returns:
        scale : (rows, 1)  또는  (rows, cols/group, 1)
        zero  : (rows, 1)  or same shape as scale
    """
    maxq = 2 ** bits - 1

    shape = x.shape  # (d_row, d_col)

    if perchannel:
        if weight:
            # 각 output channel(row)마다 독립적으로 계산
            x_flat = x.float().reshape(shape[0], -1)
        else:
            # activation의 경우 column 방향
            x_flat = x.float().reshape(-1, shape[-1])
            x_flat = x_flat.t()
    else:
        x_flat = x.float().flatten().unsqueeze(0)

    # ── min / max ──
    xmin = x_flat.min(dim=1, keepdim=True).values
    xmax = x_flat.max(dim=1, keepdim=True).values

    if sym:
        # Symmetric: zero-point = (maxq+1)/2, scale = max(|xmin|, |xmax|) / (maxq/2)
        xmax = torch.maximum(xmin.abs(), xmax.abs())
        tmp = xmax == 0
        scale = xmax / (maxq / 2)
        scale[tmp] = 1.0
        zero = torch.full_like(scale, (maxq + 1) / 2)
    else:
        # Asymmetric: full range utilization
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1.0
        xmax[tmp] = 1.0
        scale = (xmax - xmin) / maxq
        zero = torch.round(-xmin / scale)

    # reshape back: (d_row, 1)
    if perchannel and weight:
        scale = scale.reshape(shape[0], -1)
        zero  = zero.reshape(shape[0], -1)
    else:
        scale = scale.reshape(1, -1)
        zero  = zero.reshape(1, -1)

    return scale, zero


# ─────────────────────────────────────────────────────────────────────────────
# Quantizer class (GPTQ loop에서 사용)
# ─────────────────────────────────────────────────────────────────────────────

class Quantizer:
    """
    Per-channel (row-wise) quantizer.
    GPTQ Algorithm 1 내부에서 각 column 양자화 시 호출.
    """

    def __init__(
        self,
        bits: int = 4,
        perchannel: bool = True,
        sym: bool = False,
        groupsize: int = -1,
    ):
        self.bits       = bits
        self.maxq       = 2 ** bits - 1
        self.perchannel = perchannel
        self.sym        = sym
        self.groupsize  = groupsize  # -1 means no grouping

        self.scale: torch.Tensor | None = None
        self.zero:  torch.Tensor | None = None
        self.ready  = False

    # ── configure scale/zero from the full weight matrix ──────────────────
    def find_params(self, W: torch.Tensor):
        """
        전체 weight matrix W : (d_row, d_col) 로부터 scale/zero 결정.
        groupsize > 0 이면 각 group 내에서 독립 계산.
        """
        self.scale, self.zero = find_params(
            W, self.bits, perchannel=self.perchannel, sym=self.sym, weight=True
        )
        self.ready = True

    # ── fake-quantize a single column (called inside GPTQ loop) ───────────
    def quantize_col(self, col: torch.Tensor, col_idx: int) -> torch.Tensor:
        """
        col : (d_row,)  - W[:, j]
        col_idx : j     - column index (for groupsize support)
        Returns fake-quantized column (d_row,)
        """
        assert self.ready, "call find_params() first"

        if self.groupsize > 0:
            # group 경계마다 scale/zero를 재계산
            if col_idx % self.groupsize == 0:
                g_start = col_idx
                g_end   = min(col_idx + self.groupsize, col.shape[0])
                # 현재 group 범위의 weight chunk를 사용 (caller가 W 전달하지 않으므로,
                # 여기서는 현재 col만으로 근사. 실제로는 gptq.py에서 직접 처리)
                pass

        # scale/zero: (d_row, 1) 이므로 col과 broadcast 가능
        return quantize(
            col.unsqueeze(-1),   # (d_row, 1)
            self.scale,          # (d_row, 1)
            self.zero,           # (d_row, 1)
            self.maxq,
        ).squeeze(-1)            # (d_row,)
