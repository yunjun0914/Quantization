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

# 2D cross-row Vector Quantization codebook (Gaussian-optimal, 16 centers = 2bit/weight)
# 미리 계산된 codebook (모듈 로드 시 한 번만)
_CODEBOOK_2D_DATA = torch.tensor([
    [ 1.8768, -0.2440], [ 0.0240, -0.0116], [ 1.7147,  1.0772], [-0.7371, -0.4963],
    [ 1.2226, -1.4278], [-1.8262, -0.3406], [-1.1431, -1.4695], [ 0.5959,  1.7583],
    [ 0.0635, -0.9517], [ 0.8230, -0.4106], [ 0.8337,  0.5137], [-0.0190,  0.8710],
    [-0.7123,  1.6927], [-0.8336,  0.4270], [ 0.0342, -1.9977], [-1.7693,  0.9513],
], dtype=torch.float32)

def get_codebook_2d(device='cpu') -> torch.Tensor:
    """Gaussian-optimal 2D codebook (16 centers, 2bit/weight) - 미리 계산됨"""
    return _CODEBOOK_2D_DATA.to(device)


def quantize_2d_vq(x: torch.Tensor, scale: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """
    2D cross-row vector quantization.
    x:        (d_row, 1) - 하나의 column (d_row개 원소)
    scale:    (d_row//2, 1) - 2개 row 쌍마다 scale
    codebook: (16, 2) - 2D codebook

    Returns: (d_row, 1) quantized
    """
    d_row = x.shape[0]
    assert d_row % 2 == 0, "d_row must be even for 2D VQ"

    x_pairs = x.reshape(-1, 2).float()          # (d_row//2, 2)
    x_norm  = x_pairs / scale.clamp(min=1e-8)   # normalize

    # nearest neighbor
    cb = codebook.to(x.device)                  # (16, 2)
    dists = (x_norm.unsqueeze(1) - cb.unsqueeze(0)).pow(2).sum(-1)  # (d_row//2, 16)
    idx   = dists.argmin(dim=1)                 # (d_row//2,)
    x_q   = cb[idx] * scale                     # (d_row//2, 2)

    return x_q.reshape(-1, 1)                   # (d_row, 1)


def find_scale_2d(x: torch.Tensor, codebook: torch.Tensor = None) -> torch.Tensor:
    """
    2D VQ용 per-pair MSE clipping scale 계산.
    pair × 전체 column 기준 k search → optimal scale.
    x: (d_row, d_col) UWV^T 공간
    Returns: scale (d_row//2, 1)
    """
    if codebook is None:
        codebook = get_codebook_2d(x.device)

    x_pairs = x.float().reshape(-1, 2, x.shape[1])        # (d_row//2, 2, d_col)
    x_flat  = x_pairs.reshape(x_pairs.shape[0], -1)        # (d_row//2, 2*d_col)
    std     = x_flat.std(dim=1, keepdim=True).clamp(min=1e-8)
    cb      = codebook.to(x.device)

    best_scale = std * 2.0
    best_mse   = torch.full((x_flat.shape[0], 1), float('inf'), device=x.device)

    x_2d = x_flat.reshape(x_flat.shape[0], -1, 2)          # (d_row//2, d_col, 2)
    for k in [1.5, 1.8, 2.0, 2.2, 2.5, 3.0]:
        scale_cand = std * k                                 # (d_row//2, 1)
        x_norm = (x_2d / scale_cand.unsqueeze(1)).clamp(-3, 3)
        mse_sum = torch.zeros(x_flat.shape[0], device=x.device)
        chunk = 512
        for c in range(0, x_2d.shape[1], chunk):
            xc    = x_norm[:, c:c+chunk, :]
            dists = (xc.unsqueeze(-2) - cb.unsqueeze(0).unsqueeze(0)).pow(2).sum(-1)
            idx   = dists.argmin(dim=-1)
            xc_q  = cb[idx] * scale_cand.unsqueeze(1)
            mse_sum += (x_2d[:, c:c+chunk, :] - xc_q).pow(2).sum(dim=(1,2))
        mse = (mse_sum / (x_2d.shape[1] * 2)).unsqueeze(1)
        better     = mse < best_mse
        best_scale = torch.where(better, scale_cand, best_scale)
        best_mse   = torch.where(better, mse, best_mse)

    return best_scale  # (d_row//2, 1)

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

    std    = x_flat.std(dim=1, keepdim=True).clamp(min=1e-8)
    grid   = nf_grid.to(x_flat.device)

    best_scale = std * 2.0
    best_mse   = torch.full((x_flat.shape[0], 1), float('inf'), device=x_flat.device)

    for ratio in [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0]:
        scale_cand = std * ratio                             # (d_row, 1)
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
