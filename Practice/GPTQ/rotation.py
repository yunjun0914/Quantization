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


def get_hadamard(d: int, device="cpu") -> torch.Tensor:
    """
    Block Hadamard matrix. V^T V = I.
    d가 2의 거듭제곱이면 전체 Hadamard.
    아니면 가장 큰 2^k 블록으로 block-diagonal 구성.
      ex) d=768  = 3 x 256  -> 3개의 256x256 블록
          d=3072 = 3 x 1024 -> 3개의 1024x1024 블록
    """
    if d > 0 and (d & (d - 1)) == 0:
        return _hadamard_pow2(d, device=device)

    # 가장 큰 2^k 블록 탐색 (d를 나누어야 함)
    block = 1
    while block * 2 <= d:
        block *= 2
    while block > 1 and d % block != 0:
        block //= 2

    if block <= 1:
        return get_random_orthogonal(d, seed=0, device=device)

    H_block = _hadamard_pow2(block, device=device)
    n_blocks = d // block
    return torch.block_diag(*[H_block for _ in range(n_blocks)])


def get_rotation(d: int, mode: str = "random", seed: int = 0, device="cpu") -> torch.Tensor:
    """
    V 생성.
    mode: "random" | "hadamard"
    hadamard: d가 2의 거듭제곱 또는 그 배수일 때 block-diagonal Hadamard.
    """
    if mode == "hadamard":
        return get_hadamard(d, device=device)
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
    W_rot = diag(U) @ W @ V^T
    U: (d_out,) sign vector
    V: (d_in, d_in) orthogonal
    """
    return (U.unsqueeze(1) * (W.float() @ V.t())).to(W.dtype)


@torch.no_grad()
def unrotate_weight(W_rot: torch.Tensor, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    W = diag(U)^T @ W_rot @ V  =  diag(U) @ W_rot @ V  (U^T = U since ±1)
    """
    return (U.unsqueeze(1) * (W_rot.float() @ V)).to(W_rot.dtype)
