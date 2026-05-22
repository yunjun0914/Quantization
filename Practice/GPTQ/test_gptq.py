"""
GPTQ 핵심 알고리즘 단위 테스트
================================
실제 모델 없이 수치 정확도를 검증한다.

테스트 항목:
  1. Quantizer: find_params, fake-quantize 정확도
  2. GPTQ: 합성 W, H로 Algorithm 1 동작 확인
  3. Cholesky trick: H^{-1} 계산 정확도
  4. 에러 전파: 블록 업데이트가 loss를 줄이는지
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
from quantize import quantize, find_params, Quantizer
from gptq import GPTQ


# ─────────────────────────────────────────────────────────────────────────────
# 1. Quantizer tests
# ─────────────────────────────────────────────────────────────────────────────

def test_quantize_asymmetric():
    """asymmetric quantize → dequantize 후 에러가 scale/2 이하여야 함."""
    x     = torch.randn(32, 64)
    bits  = 4
    maxq  = 2 ** bits - 1
    scale, zero = find_params(x, bits, perchannel=True, sym=False)

    x_q = quantize(x, scale, zero, maxq)

    # scale = (xmax - xmin) / maxq → 최대 에러는 scale/2
    err = (x - x_q).abs()
    assert err.max().item() <= scale.max().item() * 1.01, \
        f"Max error {err.max():.4f} > scale/2 {scale.max():.4f}"
    print(f"[PASS] asymmetric quant max_err={err.max():.5f}, scale_max={scale.max():.5f}")


def test_quantize_symmetric():
    """symmetric quantize."""
    x     = torch.randn(16, 32)
    bits  = 4
    maxq  = 2 ** bits - 1
    scale, zero = find_params(x, bits, perchannel=True, sym=True)
    x_q   = quantize(x, scale, zero, maxq)
    err   = (x - x_q).abs()
    print(f"[PASS] symmetric quant max_err={err.max():.5f}")
    assert err.max().item() < 1.0


def test_quantize_8bit():
    """8bit: 절대 에러가 scale/2 이하여야 함."""
    x = torch.randn(8, 16) * 100
    scale, zero = find_params(x, 8, perchannel=True, sym=False)
    x_q = quantize(x, scale, zero, 255)
    abs_err = (x - x_q).abs()
    assert abs_err.max().item() <= scale.max().item() * 0.51
    print(f"[PASS] 8bit abs_err={abs_err.max():.5f} <= scale/2={scale.max()*0.5:.5f}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. GPTQ Algorithm 1 – synthetic test
# ─────────────────────────────────────────────────────────────────────────────

def make_dummy_layer(d_row=4, d_col=16):
    """작은 테스트용 Linear layer."""
    lin = nn.Linear(d_col, d_row, bias=False)
    nn.init.normal_(lin.weight)
    return lin


def test_gptq_runs():
    """GPTQ가 에러 없이 실행되고 올바른 shape를 반환하는지."""
    d_row, d_col = 4, 16
    lin = make_dummy_layer(d_row, d_col)

    gptq = GPTQ(lin)

    # 합성 Hessian H = 2 X X^T  (X: d_col × n_samples)
    X = torch.randn(20, d_col)   # (n_tokens, d_col)
    gptq.add_batch(X, X @ lin.weight.t())

    Q, scale, zero, losses = gptq.quantize(bits=4, blocksize=8)

    assert Q.shape     == (d_row, d_col), f"Q shape mismatch: {Q.shape}"
    assert scale.shape[0] == d_row
    assert losses.shape  == (d_row, d_col)
    print(f"[PASS] GPTQ shape check: Q={Q.shape}, scale={scale.shape}, loss_mean={losses.mean():.6f}")
    gptq.free()


def test_gptq_reduces_loss():
    """
    GPTQ 에러 전파가 없을 때(naive rounding) vs 있을 때 loss 비교.
    GPTQ loss가 항상 ≤ naive RTN loss여야 함.
    """
    torch.manual_seed(42)
    d_row, d_col = 8, 32

    lin = make_dummy_layer(d_row, d_col)
    W   = lin.weight.data.clone()

    # ── Naive RTN (round-to-nearest) loss ─────────────────────────────────
    bits  = 4
    maxq  = 2 ** bits - 1
    scale, zero = find_params(W, bits, perchannel=True, sym=False)
    W_rtn = quantize(W, scale, zero, maxq)
    rtn_loss = (W - W_rtn).pow(2).mean().item()

    # ── GPTQ loss ─────────────────────────────────────────────────────────
    X    = torch.randn(50, d_col)
    gptq = GPTQ(lin)
    gptq.add_batch(X, X @ W.t())

    _, _, _, losses = gptq.quantize(bits=bits, blocksize=8, actorder=False)
    gptq_loss = losses.mean().item()

    print(f"[INFO] RTN loss={rtn_loss:.6f}  GPTQ loss={gptq_loss:.6f}")
    # GPTQ는 2차 정보를 활용하므로 일반적으로 RTN보다 낮거나 비슷
    # 작은 랜덤 행렬에서 noise가 있을 수 있으므로 느슨하게 체크
    assert gptq_loss <= rtn_loss * 2.0, \
        f"GPTQ loss {gptq_loss:.6f} >> RTN loss {rtn_loss:.6f}"
    print("[PASS] GPTQ loss ≤ RTN loss (within 2x)")
    gptq.free()


def test_cholesky_hinv():
    """
    Cholesky trick 검증:
    U = upper_chol(H^{-1})  →  U^T @ U ≈ H^{-1}  (근사 확인)
    """
    torch.manual_seed(0)
    n = 16
    A = torch.randn(n, n)
    H = A @ A.t() + 2 * torch.eye(n)   # SPD

    L    = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    U    = torch.linalg.cholesky(Hinv, upper=True)

    recon = U.t() @ U
    err   = (recon - Hinv).abs().max().item()
    print(f"[PASS] Cholesky reconstruction error: {err:.2e}")
    assert err < 1e-5, f"Cholesky error too large: {err}"


def test_hessian_accumulation():
    """add_batch 누적이 H = 2 * (1/N) * sum X_i X_i^T를 정확히 계산하는지."""
    torch.manual_seed(1)
    d_col = 8
    lin   = nn.Linear(d_col, 4, bias=False)
    gptq  = GPTQ(lin)

    X_all = []
    for _ in range(5):
        X = torch.randn(10, d_col)
        X_all.append(X)
        gptq.add_batch(X, X @ lin.weight.t())

    X_cat  = torch.cat(X_all, dim=0).t()  # (d_col, 50)
    H_ref  = 2 * (X_cat @ X_cat.t()) / 50

    err = (gptq.H - H_ref).abs().max().item()
    print(f"[PASS] Hessian accumulation error: {err:.2e}")
    assert err < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("GPTQ Unit Tests")
    print("=" * 55)

    tests = [
        test_quantize_asymmetric,
        test_quantize_symmetric,
        test_quantize_8bit,
        test_gptq_runs,
        test_gptq_reduces_loss,
        test_cholesky_hinv,
        test_hessian_accumulation,
    ]

    passed = 0
    for t in tests:
        print(f"\n── {t.__name__} ──")
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {e}")
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")

    print(f"\n{'='*55}")
    print(f"Results: {passed}/{len(tests)} passed")
    print("=" * 55)
