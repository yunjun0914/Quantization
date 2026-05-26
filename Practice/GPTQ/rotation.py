"""
Rotation utilities for Rotated GPTQ
=====================================

U : sign vector ∈ {-1, +1}^{d_out}  →  per-output-channel reflection
V : full orthogonal matrix (d_in × d_in)  →  input space rotation

Absorption:
    V : 이전 레이어 weight에 흡수  (W_prev ← W_prev @ V^T)
    U : 다음 레이어 weight에 흡수  (W_next ← diag(U) @ W_next)
"""

import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────────────────────────────────────
# U: sign vector
# ─────────────────────────────────────────────────────────────────────────────

def get_sign_vector(d_out: int, seed: int = 0, device="cpu") -> torch.Tensor:
    """
    U = random sign vector ∈ {-1, +1}^{d_out}
    orthogonal: U^T U = I  (diag(±1)^T diag(±1) = I)
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return torch.randint(0, 2, (d_out,), generator=g, device=device).float() * 2 - 1


# ─────────────────────────────────────────────────────────────────────────────
# V: full orthogonal rotation
# ─────────────────────────────────────────────────────────────────────────────

def get_random_orthogonal(d: int, seed: int = 0, device="cpu") -> torch.Tensor:
    """
    V : random orthogonal matrix via QR decomposition of random Gaussian matrix.
    V^T V = I
    """
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    A = torch.randn(d, d, generator=g, device=device)
    Q, R = torch.linalg.qr(A)
    # QR의 부호 일관성 보정 (torch convention)
    Q *= R.diag().sign().unsqueeze(0)
    return Q


def _hadamard_pow2(d: int, device="cpu") -> torch.Tensor:
    # d = 2^k 인 Hadamard matrix
    assert d > 0 and (d & (d - 1)) == 0
    H = torch.tensor([[1.0]], device=device)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H,-H], dim=1)], dim=0)
    return H / math.sqrt(d)


def get_hadamard(d: int, device="cpu", seed: int = 0) -> torch.Tensor:
    """
    Randomized Hadamard (QuaRot 방식).
    d가 2의 거듭제곱이면 exact Hadamard.
    아니면 random orthogonal (QR) 사용 - padding submatrix는 orthogonal 보장 불가.
    randomized: diag(±1) @ H @ diag(±1)
    """
    if d > 0 and (d & (d - 1)) == 0:
        H = _hadamard_pow2(d, device=device)
        # randomize
        g = torch.Generator(device=device); g.manual_seed(seed)
        dl = (torch.randint(0, 2, (d,), generator=g, device=device).float() * 2 - 1)
        g.manual_seed(seed + 1)
        dr = (torch.randint(0, 2, (d,), generator=g, device=device).float() * 2 - 1)
        return dl.unsqueeze(1) * H * dr.unsqueeze(0)
    else:
        # d가 2의 거듭제곱 아님
        # d % 172 == 0이면 QuIP# H_64 ⊗ H_172 방식
        if d % 172 == 0:
            from had_utils import matmul_hadU
            # H_d를 명시적 행렬로 생성: H @ I
            eye = torch.eye(d, device=device)
            return matmul_hadU(eye).T  # (d, d) Hadamard matrix
        return get_random_orthogonal(d, seed=seed, device=device)

def get_rotation(d: int, mode: str = "random", seed: int = 0, device="cpu") -> torch.Tensor:
    """
    V 생성.
    mode: "random" | "hadamard"
    hadamard: padding 기반 randomized Hadamard (임의 d 지원)
    """
    if mode == "hadamard":
        return get_hadamard(d, device=device, seed=seed)
    else:
        return get_random_orthogonal(d, seed=seed, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Absorption
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def absorb_V_to_prev(prev_layer: nn.Linear, V: torch.Tensor):
    """
    V를 이전 레이어에 흡수.
    y = W_prev x  →  V y = (V W_prev) x
    W_prev ← V @ W_prev
    """
    prev_layer.weight.data = (V @ prev_layer.weight.data.float()).to(prev_layer.weight.dtype)
    if prev_layer.bias is not None:
        prev_layer.bias.data = (V @ prev_layer.bias.data.float()).to(prev_layer.bias.dtype)


@torch.no_grad()
def absorb_V_to_next(next_layer: nn.Linear, V: torch.Tensor):
    """
    V^T를 다음 레이어에 흡수.
    y = W_next (Vx)  →  W_next ← W_next @ V^T  는 잘못된 방향.

    정확한 absorption:
    이전 레이어 출력이 Vx이면,
    다음 레이어 입력이 Vx → W_next @ V^T 로 원래 공간 복원.

    여기서는 "다음 레이어의 입력이 V 회전된 상태"를 가정하므로:
    W_next ← W_next @ V^T
    """
    next_layer.weight.data = (next_layer.weight.data.float() @ V.t()).to(next_layer.weight.dtype)


@torch.no_grad()
def absorb_U_to_next(next_layer: nn.Linear, U: torch.Tensor):
    """
    U를 다음 레이어에 흡수.
    현재 레이어 출력: U * (Wx)  (element-wise, U는 sign vector)
    다음 레이어: W_next @ (U * out) = (W_next * U.unsqueeze(0)) @ out_before_U

    → W_next의 각 column j에 U[j]를 곱함
    W_next[:, j] *= U[j]
    """
    next_layer.weight.data = (next_layer.weight.data.float() * U.unsqueeze(0)).to(next_layer.weight.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Apply rotation to a single Linear layer weight
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def rotate_weight(W: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    W_rot = U @ W @ V^T
    U: (d_out,) 1D vector 또는 (d_out, d_out) matrix
    V: (d_in,)  1D vector 또는 (d_in, d_in)   matrix
    1D인 경우 elementwise (identity로 쓸 수 있음)
    """
    # V 적용
    if V.dim() == 1:
        WVt = W.float() * V.unsqueeze(0)   # elementwise: W * v^T
    else:
        WVt = W.float() @ V.t()            # matmul: W @ V^T

    # U 적용
    if U.dim() == 1:
        return (U.unsqueeze(1) * WVt).to(W.dtype)
    else:
        return (U.float() @ WVt).to(W.dtype)


@torch.no_grad()
def unrotate_weight(W_rot: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    W = U^T @ W_rot @ V
    U: (d_out,) sign vector (U^T=U) 또는 (d_out, d_out) orthogonal (U^T=U.t())
    """
    W_vabs = W_rot.float() @ V
    if U.dim() == 1:
        return (U.unsqueeze(1) * W_vabs).to(W_rot.dtype)    # U^T = U for ±1
    else:
        return (U.t().float() @ W_vabs).to(W_rot.dtype)     # U^T @ W_rot @ V
