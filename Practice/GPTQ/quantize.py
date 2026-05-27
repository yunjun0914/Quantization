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

# 2D cross-row Vector Quantization codebook (Gaussian-optimal, 16 centers = 2bit/weight)
# 미리 계산된 codebook (모듈 로드 시 한 번만)

# E8P codebook (QuIP# 방식, 65536개 = 정확히 2bit/weight)
import os as _os
_E8P_CB_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "e8p_codebook.pt")
_E8P_CODEBOOK = None

def _get_norm12():
    return torch.tensor([
        [3,1,1,1,3,3,3,3],[1,3,1,1,3,3,3,3],[1,1,3,1,3,3,3,3],[1,1,1,3,3,3,3,3],
        [3,3,3,1,3,3,1,1],[3,3,3,1,3,1,3,1],[3,3,3,1,1,3,3,1],[3,3,3,1,3,1,1,3],
        [3,3,3,1,1,3,1,3],[3,3,3,1,1,1,3,3],[3,3,1,3,3,3,1,1],[3,3,1,3,3,1,3,1],
        [3,3,1,3,1,3,3,1],[3,3,1,3,3,1,1,3],[3,3,1,3,1,3,1,3],[3,3,1,3,1,1,3,3],
        [3,1,3,3,3,3,1,1],[3,1,3,3,3,1,3,1],[3,1,3,3,1,3,3,1],[3,1,3,3,3,1,1,3],
        [3,1,3,3,1,3,1,3],[1,3,3,3,1,1,3,3],[1,3,3,3,3,3,1,1],[1,3,3,3,3,1,3,1],
        [1,3,3,3,1,3,3,1],[1,3,3,3,3,1,1,3],[1,3,3,3,1,3,1,3],[1,1,3,3,1,3,3,3],
        [3,3,1,1,3,3,3,1],
    ]) / 2

def _build_e8p_grid():
    """QuIP# E8P codebook 생성 (65536개, 2bit/weight)"""
    intr  = torch.arange(-4, 4)
    d8    = torch.cartesian_prod(*[intr]*8).float() + 0.5
    d8m2  = (d8.sum(dim=-1) % 2 == 0)
    d8n   = d8.norm(dim=-1)**2 <= 10
    d8abs = torch.unique(d8[d8m2 & d8n].abs(), dim=0)
    abs_grid = torch.cat([d8abs, _get_norm12()], dim=0)  # (256, 8)

    N      = 1 << 16
    c_idx  = torch.arange(N, dtype=torch.int32)
    signs  = c_idx & 255
    abs_i  = (c_idx >> 8).long()
    par    = torch.zeros(N, dtype=torch.int32)
    for i in range(8): par = par ^ ((signs >> i) & 1)
    signs  = signs ^ par
    shuffle = [0,4,1,5,2,6,3,7]
    base   = abs_grid[abs_i][:, shuffle]
    sign_mat = torch.ones(N, 8)
    for i in range(8):
        sign_mat[:, i] = torch.where(((signs >> i) & 1).bool(),
                                     torch.tensor(-1.0), torch.tensor(1.0))
    grid = base * sign_mat
    grid[par.bool()]  -= 0.25
    grid[~par.bool()] += 0.25
    return grid

def get_e8p_codebook(device='cpu') -> torch.Tensor:
    """E8P codebook (65536×8, 2bit/weight) - lazy load or build"""
    global _E8P_CODEBOOK
    if _E8P_CODEBOOK is None:
        if _os.path.exists(_E8P_CB_PATH):
            _E8P_CODEBOOK = torch.load(_E8P_CB_PATH, weights_only=True)
        else:
            print("[E8P] codebook 생성 중...")
            _E8P_CODEBOOK = _build_e8p_grid()
            torch.save(_E8P_CODEBOOK, _E8P_CB_PATH)
    return _E8P_CODEBOOK.to(device)


# ── E8P fast quantize (QuIP# fast_quantize_part 방식) ─────────────────────
def _build_e8p_fast_components():
    """grid_part (1366, 8) 및 norm 사전 계산"""
    def get_norm12():
        return torch.tensor([
            [3,1,1,1,3,3,3,3],[1,3,1,1,3,3,3,3],[1,1,3,1,3,3,3,3],[1,1,1,3,3,3,3,3],
            [3,3,3,1,3,3,1,1],[3,3,3,1,3,1,3,1],[3,3,3,1,1,3,3,1],[3,3,3,1,3,1,1,3],
            [3,3,3,1,1,3,1,3],[3,3,3,1,1,1,3,3],[3,3,1,3,3,3,1,1],[3,3,1,3,3,1,3,1],
            [3,3,1,3,1,3,3,1],[3,3,1,3,3,1,1,3],[3,3,1,3,1,3,1,3],[3,3,1,3,1,1,3,3],
            [3,1,3,3,3,3,1,1],[3,1,3,3,3,1,3,1],[3,1,3,3,1,3,3,1],[3,1,3,3,3,1,1,3],
            [3,1,3,3,1,3,1,3],[1,3,3,3,1,1,3,3],[1,3,3,3,3,3,1,1],[1,3,3,3,3,1,3,1],
            [1,3,3,3,1,3,3,1],[1,3,3,3,3,1,1,3],[1,3,3,3,1,3,1,3],[1,1,3,3,1,3,3,3],
            [3,3,1,1,3,3,3,1],
        ]) / 2

    intr  = torch.arange(-4, 4)
    d8    = torch.cartesian_prod(*[intr]*8).float() + 0.5
    abs_grid = torch.cat([
        torch.unique(d8[(d8.sum(-1)%2==0) & (d8.norm(dim=-1)**2<=10)].abs(), dim=0),
        get_norm12()
    ], dim=0)  # (256, 8)

    N = 1<<16
    c_idx = torch.arange(N, dtype=torch.int32)
    signs = c_idx & 255; abs_i = (c_idx>>8).long()
    par = torch.zeros(N, dtype=torch.int32)
    for i in range(8): par = par ^ ((signs>>i)&1)
    signs = signs ^ par
    base = abs_grid[abs_i][:, [0,4,1,5,2,6,3,7]]
    sm = torch.stack([(~((signs>>i)&1).bool()).float()*2-1 for i in range(8)], dim=1)
    grid = base * sm
    pb = par.bool()
    grid[pb] -= 0.25; grid[~pb] += 0.25

    gp = grid[pb] + 0.25
    mask = ((gp[:,:7]<0).sum(-1)<=1) & (gp[:,:7].min(-1).values>=-0.5)
    grid_part = gp[mask]
    return grid_part, (grid_part**2).sum(-1)

_E8P_GRID_PART, _E8P_GRID_PART_NORM = _build_e8p_fast_components()


def _e8p_fast_qpart(X: torch.Tensor) -> tuple:
    """QuIP# fast_quantize_part: (N,8) → vals, err"""
    gp   = _E8P_GRID_PART.to(X.device)
    gpn  = _E8P_GRID_PART_NORM.to(X.device)
    Xp   = X.abs().clone()
    odd  = (X < 0).sum(dim=1) % 2 != 0
    Xp[odd, 7] *= -1
    msk  = torch.where(X < 0, torch.full_like(X, -1.0), torch.ones_like(X))
    msk[odd, 7] *= -1
    scores = 2 * Xp @ gp.T - gpn
    ro     = gp[scores.argmax(dim=1)]
    vals   = ro * msk
    return vals, (X - vals).norm(dim=1)


def quantize_e8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    E8P quantization (QuIP# fast_quantize_part 방식).
    x:     (d_row, 1)
    scale: (d_row//8, 1)
    Returns: (d_row, 1)
    """
    x_blocks = x.reshape(-1, 8).float()           # (d_row//8, 8)
    x_norm   = x_blocks / scale.clamp(min=1e-8)   # normalize

    pv, pe = _e8p_fast_qpart(x_norm + 0.25)
    mv, me = _e8p_fast_qpart(x_norm - 0.25)
    which  = (pe < me).unsqueeze(1)
    q_norm = torch.where(which, pv - 0.25, mv + 0.25)
    return (q_norm * scale).reshape(-1, 1)


def quantize_e8_indexed(x: torch.Tensor, scale: torch.Tensor):
    """
    E8P quantization with index.
    x:     (d_row, 1)
    scale: (d_row//8, 1)
    Returns: (q_float, idx)
      q_float: (d_row, 1) dequantized float
      idx:     (d_row//8,) uint16 E8P codeword index
    """
    cb       = get_e8p_codebook(x.device)
    x_blocks = x.reshape(-1, 8).float()
    x_norm   = x_blocks / scale.clamp(min=1e-8)

    pv, pe = _e8p_fast_qpart(x_norm + 0.25)
    mv, me = _e8p_fast_qpart(x_norm - 0.25)
    which  = (pe < me).unsqueeze(1)
    q_norm = torch.where(which, pv - 0.25, mv + 0.25)

    # q_norm은 E8P codebook의 한 점 → 직접 index 계산
    # argmax trick: argmin||q-cb||² = argmax(2q@cb.T - ||cb||²)
    cb_norm = (cb ** 2).sum(dim=-1)
    scores  = 2 * q_norm @ cb.T - cb_norm        # (d_row//8, 65536)
    idx     = scores.argmax(dim=1).to(torch.int32)  # (d_row//8,)

    q_float = (q_norm * scale).reshape(-1, 1)
    return q_float, idx


def dequantize_e8(idx: torch.Tensor, scale: torch.Tensor, device='cpu') -> torch.Tensor:
    """
    E8P dequantization from index.
    idx:   (d_row//8,) int32
    scale: scalar or (d_row//8, 1)
    Returns: (d_row,) float
    """
    cb = get_e8p_codebook(device)
    q  = cb[idx] * scale                    # (d_row//8, 8)
    return q.reshape(-1)                    # (d_row,)


def find_scale_e8(W: torch.Tensor) -> torch.Tensor:
    """
    E8P scale 계산 (QuIP# scale_override=0.9 방식).
    W: (d_row, d_col)
    Returns: scale (d_row//8, 1)
    """
    d_row, d_col = W.shape
    x_blocks = W.float().reshape(-1, 8, d_col)
    x_flat   = x_blocks.reshape(x_blocks.shape[0], -1)
    std      = x_flat.std(dim=1, keepdim=True).clamp(min=1e-8)
    return std * 0.9  # QuIP# scale_override 권장값


def get_nf_grid(bits: int) -> torch.Tensor:
    if bits == 2: return NF2_GRID
    return None


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
