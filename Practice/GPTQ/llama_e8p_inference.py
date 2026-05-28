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
    def __init__(self, d_row, d_col, idx, scale, U=None, bias=None):
        super().__init__()
        self.register_buffer('idx',   idx.cpu())
        self.register_buffer('scale', scale.cpu())
        self.d_row = d_row
        self.d_col = d_col
        self.U     = U   # globally shared tensor 참조

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

        # U^T 적용 (fake quant의 q = U^T @ q_rot 와 동일)
        if self.U is not None:
            U = self.U.to(q.device).float()
            q = U.t() @ q.float()

        return q

    def forward(self, x):
        W   = self.dequant_weight().to(x.dtype)
        out = x @ W.t()

        if self.bias is not None:
            out = out + self.bias
        return out


def make_v_hook(V):
    """LayerNorm output에 V rotation 적용.
    fake quant: y = x @ (Q@V).t = x @ V.t @ Q.t
    real quant: y = x_rot @ Q.t, x_rot = x @ V.t
    따라서 x_rot = x @ V.t (= (V @ x.t).t)
    """
    def hook(module, input, output):
        orig_shape = output.shape
        out_flat   = output.reshape(-1, output.shape[-1]).float()
        V_dev      = V.float().to(out_flat.device)
        out_rot    = out_flat @ V_dev.t()   # x @ V.t (fake quant와 동일)
        return out_rot.reshape(orig_shape).to(output.dtype)
    return hook


def build_e8p_model(model, quantized_layers, V, rotations=None):
    """
    fake quant 모델을 real quant E8PLinear로 교체.

    rotations: {layer_name: (U, V)} - 각 layer의 U 정보
    """
    layers = model.model.layers
    hooks  = []

    # 1. LayerNorm에 V hook 등록 (q/k/v/o/gate/up input rotation)
    for layer in layers:
        hooks.append(layer.input_layernorm.register_forward_hook(make_v_hook(V)))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(make_v_hook(V)))

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

        e8p_layer = E8PLinear(
            d_row = d_row,
            d_col = d_col,
            idx   = data['idx'],
            scale = data['scale'],
            U     = U,
            bias  = orig.bias,
        )
        setattr(parent, sub_parts[-1], e8p_layer)
        mem_kb = d_row * d_col * 2 / 8 / 1024
        print(f"  replaced {full_name}: {d_row}×{d_col} → E8P ({mem_kb:.1f}KB)")

    return model, hooks
