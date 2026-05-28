"""
LLaMA E8P Real Quantization Inference
======================================

fake quant 구조와 완전히 동일:

  q/k/v/o/gate/up:
    W_stored = U^T @ Q_rot @ V  (WV^T 공간 → V 흡수)
    inference: x_rot = V @ LayerNorm(x)  [hook]
               y = x_rot @ W_stored.t()

  down_proj:
    W_stored = U_d^T @ Q_rot  (V 흡수 없음, fake quant와 동일)
    inference: x_in = U_gu @ h  [R4 hook]
               y = x_in @ W_stored.t()

저장:
  - idx:   (d_row//8, d_col) int32  ← 2bit/weight
  - scale: scalar fp16              ← ~0bpw
  - U:     globally shared 참조     ← 메모리 중복 없음
"""

import torch
import torch.nn as nn
from quantize import get_e8p_codebook


class E8PLinear(nn.Module):
    """
    Real 2bit E8P quantized Linear layer.
    fake quant의 W_stored = U^T @ Q_rot (@ V) 와 동일.
    """
    def __init__(self, d_row, d_col, idx, scale, U=None, V=None, bias=None):
        super().__init__()
        self.register_buffer('idx',   idx.cpu())
        self.register_buffer('scale', scale.cpu())
        self.d_row = d_row
        self.d_col = d_col
        self.U     = U   # globally shared tensor 참조
        self.V     = V   # globally shared tensor 참조

        if bias is not None:
            self.register_buffer('bias', bias.cpu())
        else:
            self.bias = None

    def dequant_weight(self):
        """
        E8P index → U^T @ Q_rot (fake quant의 Q와 동일)
        """
        cb = get_e8p_codebook(self.idx.device)
        q  = cb[self.idx]                               # (d_row//8, d_col, 8)
        q  = q * self.scale
        q  = q.permute(0, 2, 1).reshape(self.d_row, self.d_col)  # U 공간

        # U^T 적용
        if self.U is not None:
            q = self.U.to(q.device).t().float() @ q.float()

        # V 적용 → U^T @ Q_rot @ V = fake quant의 W_stored
        if self.V is not None:
            q = q.float() @ self.V.to(q.device).float()

        return q

    def forward(self, x):
        W   = self.dequant_weight().to(x.dtype)
        out = x @ W.t()

        if self.bias is not None:
            out = out + self.bias
        return out


def build_e8p_model(model, quantized_layers, V, rotations=None):
    """
    fake quant 모델을 real quant E8PLinear로 교체.

    rotations: {layer_name: (U, V)} - 각 layer의 U 정보
    """
    layers = model.model.layers
    hooks  = []

    hooks = []  # V hook 불필요 (dequant_weight에서 V 처리)


    # 2. Linear layer를 E8PLinear로 교체
    for full_name, data in quantized_layers.items():
        parts     = full_name.split('.')
        layer_idx = int(parts[0].replace('layer', ''))
        layer     = layers[layer_idx]
        sub_name  = '.'.join(parts[1:])

        parent    = layer
        sub_parts = sub_name.split('.')
        for p in sub_parts[:-1]:
            parent = getattr(parent, p)
        orig = getattr(parent, sub_parts[-1])

        d_row, d_col = orig.weight.shape

        # U 가져오기
        U = None
        if rotations and sub_name in rotations:
            U, _ = rotations[sub_name]

        # V: rotations에서 가져오기 (down_proj는 V 대신 U_gu)
        V_layer = None
        if rotations and sub_name in rotations:
            _, V_layer = rotations[sub_name]

        e8p_layer = E8PLinear(
            d_row = d_row,
            d_col = d_col,
            idx   = data['idx'],
            scale = data['scale'],
            U     = U,
            V     = V_layer,
            bias  = orig.bias,
        )
        setattr(parent, sub_parts[-1], e8p_layer)
        mem_kb = d_row * d_col * 2 / 8 / 1024
        print(f"  replaced {full_name}: {d_row}×{d_col} → E8P ({mem_kb:.1f}KB)")

    return model, hooks
