"""
LLaMA E8P Real Quantization Inference
======================================

저장 구조:
  - E8P index: (d_row//8, d_col) int32  ← 2bit/weight
  - scale:     scalar fp16              ← ~0bpw
  - V → LayerNorm weight에 흡수         ← 별도 저장 없음
  - U → inference 시 FWHT on-the-fly   ← 별도 저장 없음

Inference 흐름:
  x_rot = LayerNorm_V(x)               # V가 LayerNorm에 흡수
  y_rot = dequant(idx, scale) @ x_rot  # matmul
  y     = FWHT(y_rot)                  # U^T online (O(n log n))
"""

import torch
import torch.nn as nn
from quantize import get_e8p_codebook
from had_utils import matmul_hadUt


class E8PLinear(nn.Module):
    """
    Real 2bit E8P quantized Linear layer.
    저장: E8P index (int32) + per-layer scale
    V: LayerNorm에 흡수 → 여기서 불필요
    U: forward 시 FWHT on-the-fly
    """
    def __init__(self, d_row, d_col, idx, scale, bias=None):
        super().__init__()
        self.register_buffer('idx',   idx.cpu())      # (d_row//8, d_col) int32
        self.register_buffer('scale', scale.cpu())    # scalar
        self.d_row   = d_row
        self.d_col   = d_col

        if bias is not None:
            self.register_buffer('bias', bias.cpu())
        else:
            self.bias = None

    def dequant_weight(self):
        """E8P index → float weight (d_row, d_col), U^T 미적용"""
        cb = get_e8p_codebook(self.idx.device)       # (65536, 8)
        q  = cb[self.idx]                             # (d_row//8, d_col, 8)
        q  = q * self.scale
        q  = q.permute(0, 2, 1).reshape(self.d_row, self.d_col)  # (d_row, d_col)
        return q

    def forward(self, x):
        W   = self.dequant_weight().to(x.dtype)      # (d_row, d_col) dtype 맞춤
        out = x @ W.t()                              # (*, d_row) U 공간

        # U^T online: FWHT (O(n log n))
        # out: (batch, seq, d_row) or (batch, d_row)
        orig_shape = out.shape
        out_flat   = out.reshape(-1, self.d_row).float()
        out_flat   = matmul_hadUt(out_flat)           # U^T 적용
        out        = out_flat.reshape(orig_shape).to(x.dtype)

        if self.bias is not None:
            out = out + self.bias
        return out


def absorb_V_to_layernorm(layernorm, V):
    """
    LayerNorm weight에 V 흡수.
    y = LayerNorm(x) → W @ y  가
    y = LayerNorm_V(x) → W_rot @ y_rot  가 되도록.

    LLaMA RMSNorm: y = x / RMS(x) * weight
    V 흡수 후: weight_new = weight * V (elementwise가 아닌 행렬곱)
    실제로는 V가 rotation이라 weight_new = (V @ diag(weight))의 diagonal
    → 하지만 RMSNorm은 elementwise scaling이라
      V @ (weight * x_norm) ≠ weight_new * x_norm (일반적으로)

    올바른 방법: activation에 V를 hook으로 적용
    """
    # RMSNorm의 특성상 weight에 직접 흡수 불가
    # 대신 forward hook으로 LayerNorm output에 V 적용
    def hook(module, input, output):
        out_flat = output.reshape(-1, output.shape[-1]).float()
        # V는 Hadamard: matmul_hadU 사용
        from had_utils import matmul_hadU
        out_flat = matmul_hadU(out_flat)
        return out_flat.reshape(output.shape).to(output.dtype)
    return layernorm.register_forward_hook(hook)


def build_e8p_model(model, quantized_layers, V, seed=0):
    """
    fake quant 모델을 real quant E8PLinear로 교체.
    V는 LayerNorm forward hook으로 적용.

    quantized_layers: {
        'layer0.self_attn.q_proj': {'idx': ..., 'scale': ...}, ...
    }
    """
    layers = model.model.layers
    hooks  = []

    # 1. 각 layer의 LayerNorm에 V hook 등록
    for layer_idx, layer in enumerate(layers):
        h1 = absorb_V_to_layernorm(layer.input_layernorm, V)
        h2 = absorb_V_to_layernorm(layer.post_attention_layernorm, V)
        hooks.extend([h1, h2])

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
        e8p_layer = E8PLinear(
            d_row = d_row,
            d_col = d_col,
            idx   = data['idx'],
            scale = data['scale'],
            bias  = orig.bias,
        )
        setattr(parent, sub_parts[-1], e8p_layer)
        mem_kb = d_row * d_col * 2 / 8 / 1024
        print(f"  replaced {full_name}: {d_row}×{d_col} → E8P ({mem_kb:.1f}KB)")

    return model, hooks
