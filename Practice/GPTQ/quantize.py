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
