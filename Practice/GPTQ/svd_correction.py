"""
SVD Residual Correction
========================
Rotated GPTQ 후 순수 양자화 오차를 low-rank SVD로 보정.

W ≈ Q_rot + U_r @ diag(S_r) @ V_r^T  (rank-r)

이점:
  - Rotated GPTQ는 W 공간에서 순수 양자화 오차만 남김
  - 회전 복원 오차 없음 → R = W - Q_rot 이 순수 양자화 오차
  - 양자화 오차는 outlier 방향에 집중 → low-rank 구조 존재
  - rank=1이면 추가 파라미터 (d_row + d_col)개 fp16만 필요

inference:
  output = Q_rot @ x + U_r @ (S_r * (V_r^T @ x))
"""

import torch
import torch.nn as nn


def compute_svd_correction(
    W_orig:   torch.Tensor,   # (d_row, d_col) 원래 weight
    W_quant:  torch.Tensor,   # (d_row, d_col) 양자화된 weight (Q_rot)
    rank:     int = 1,
    use_hessian_weight: bool = False,
    H:        torch.Tensor = None,   # (d_col, d_col) Hessian (optional)
) -> dict:
    """
    R = W_orig - W_quant 의 rank-r SVD 계산.

    Args:
        W_orig:  원래 weight
        W_quant: 양자화된 weight
        rank:    SVD rank
        use_hessian_weight: H로 residual 가중 후 SVD
        H:       Hessian (use_hessian_weight=True일 때 필요)

    Returns:
        {
            "U":     (d_row, rank) fp16
            "S":     (rank,) fp16
            "Vt":    (rank, d_col) fp16
            "error_before": RMSE before correction
            "error_after":  RMSE after correction
            "singular_values": full singular value spectrum
        }
    """
    R = (W_orig.float() - W_quant.float())   # 순수 양자화 오차

    if use_hessian_weight and H is not None:
        # H 대각으로 column 가중: 중요한 column의 오차를 더 잡음
        h_diag = H.diag().clamp(min=1e-6).sqrt()
        R_weighted = R * h_diag.unsqueeze(0)
        U, S, Vt = torch.linalg.svd(R_weighted, full_matrices=False)
        # 가중치 제거
        Vt = Vt / h_diag.unsqueeze(0)
    else:
        U, S, Vt = torch.linalg.svd(R, full_matrices=False)

    # rank-r truncation
    U_r  = U[:, :rank]     # (d_row, rank)
    S_r  = S[:rank]        # (rank,)
    Vt_r = Vt[:rank, :]    # (rank, d_col)

    # 보정된 weight
    R_approx = U_r @ torch.diag(S_r) @ Vt_r
    W_corrected = W_quant.float() + R_approx

    # 오차 측정
    err_before = R.pow(2).mean().sqrt().item()
    err_after  = (W_orig.float() - W_corrected).pow(2).mean().sqrt().item()

    # singular value 스펙트럼 (얼마나 low-rank인지 확인)
    sv_ratio = (S[:rank].sum() / S.sum()).item()

    return {
        "U":               U_r.half(),
        "S":               S_r.half(),
        "Vt":              Vt_r.half(),
        "W_corrected":     W_corrected.half(),
        "error_before":    err_before,
        "error_after":     err_after,
        "sv_ratio":        sv_ratio,      # rank-r가 전체 분산의 몇 %를 설명하는지
        "singular_values": S[:min(10, len(S))].tolist(),
    }


class SVDCorrectedLinear(nn.Module):
    """
    inference용: Q_rot + U_r S_r V_r^T 를 효율적으로 계산.

    W_quant: (d_row, d_col) 2bit quantized (fake-quantized fp16)
    U:       (d_row, rank)
    S:       (rank,)
    Vt:      (rank, d_col)

    output = W_quant @ x + U @ diag(S) @ Vt @ x
           = W_quant @ x + U @ (S * (Vt @ x))
    """

    def __init__(
        self,
        W_quant: torch.Tensor,
        U:       torch.Tensor,
        S:       torch.Tensor,
        Vt:      torch.Tensor,
        bias:    torch.Tensor = None,
    ):
        super().__init__()
        self.register_buffer("W_quant", W_quant)
        self.register_buffer("U",       U)
        self.register_buffer("S",       S)
        self.register_buffer("Vt",      Vt)
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 기본 양자화 출력
        out = x @ self.W_quant.t()
        # low-rank 보정: U @ (S * (Vt @ x^T))
        # x: (batch, seq, d_col) → Vt @ x.t(): (rank, batch*seq)
        x_flat = x.reshape(-1, x.shape[-1])             # (n, d_col)
        svt_x  = (x_flat @ self.Vt.t()) * self.S        # (n, rank)
        out_flat = out.reshape(-1, out.shape[-1])
        out_flat = out_flat + svt_x @ self.U.t()         # (n, d_row)
        out = out_flat.reshape(x.shape[:-1] + (out_flat.shape[-1],))

        if self.bias is not None:
            out = out + self.bias
        return out


@torch.no_grad()
def apply_svd_corrections(
    model,
    results:  dict,
    get_layers_fn,
    find_linears_fn,
    rank:     int   = 1,
    verbose:  bool  = True,
) -> dict:
    """
    모든 layer에 SVD correction 적용.

    results: GPTQ 결과 dict (W_orig 포함 필요)
    rank:    SVD rank
    """
    layers  = get_layers_fn(model)
    sv_stats = {}

    for layer_idx, layer in enumerate(layers):
        linears = find_linears_fn(layer)
        for name, lin in linears.items():
            key = f"layer{layer_idx}.{name}"
            if key not in results or "W_orig" not in results[key]:
                continue

            W_orig  = results[key]["W_orig"].to(lin.weight.device).float()
            W_quant = lin.weight.data.float()

            corr = compute_svd_correction(W_orig, W_quant, rank=rank)

            if verbose:
                print(f"  {key:45s}  "
                      f"RMSE {corr['error_before']:.5f} → {corr['error_after']:.5f}  "
                      f"sv_ratio={corr['sv_ratio']:.3f}  "
                      f"sv={[f'{v:.3f}' for v in corr['singular_values'][:3]]}")

            # weight 업데이트: Q_rot + low-rank
            lin.weight.data = corr["W_corrected"].to(lin.weight.dtype)
            sv_stats[key]   = corr

    return sv_stats
