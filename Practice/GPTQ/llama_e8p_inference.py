"""
LLaMA E8P Real Quantization Inference
======================================

fake quant (Q @ V fp16)이 아닌 진짜 2bit E8P 저장 + inference.

저장 구조:
  layer_i: {
    'e8p_idx':   (d_row//8, d_col) int32   ← 2bit/weight
    'e8p_scale': scalar fp16               ← ~0bpw
    'U':         (d_row, d_row) Hadamard   ← online U^T
    'V':         (d_col, d_col) Hadamard   ← online V
  }

Inference 흐름 (per linear layer):
  x_rot = V @ x                # input rotation online
  y_rot = dequant(idx, scale) @ x_rot
  y     = U^T @ y_rot          # output unrotation online
"""

import torch
import torch.nn as nn
from quantize import get_e8p_codebook


class E8PLinear(nn.Module):
    """
    Real 2bit E8P quantized Linear layer.
    저장: E8P index (int32) + per-layer scale
    Inference: dequant → U^T online, V online on input
    """
    def __init__(self, d_row, d_col, idx, scale, U=None, V=None, bias=None):
        super().__init__()
        self.register_buffer('idx',   idx.cpu())      # (d_row//8, d_col) int32
        self.register_buffer('scale', scale.cpu())    # scalar or (1,)
        self.d_row = d_row
        self.d_col = d_col

        # U, V: Hadamard matrix (d_row×d_row, d_col×d_col)
        # register_buffer로 device 이동 자동 처리
        if U is not None:
            self.register_buffer('U', U.cpu().float())
        else:
            self.U = None

        if V is not None:
            self.register_buffer('V', V.cpu().float())
        else:
            self.V = None

        if bias is not None:
            self.register_buffer('bias', bias.cpu())
        else:
            self.bias = None

    def dequant_weight(self):
        """
        E8P index → float weight (d_row, d_col)
        U^T 적용 (output rotation 복원)
        """
        cb = get_e8p_codebook(self.idx.device)       # (65536, 8)
        q  = cb[self.idx]                             # (d_row//8, d_col, 8)
        q  = q * self.scale                           # scale broadcast
        # (d_row//8, d_col, 8) → (d_row//8, 8, d_col) → (d_row, d_col)
        q  = q.permute(0, 2, 1).reshape(self.d_row, self.d_col)

        # U^T online (output unrotation)
        if self.U is not None:
            q = self.U.to(q.dtype).t() @ q
        return q

    def forward(self, x):
        # V online (input rotation): x_rot = V @ x
        if self.V is not None:
            x_orig_shape = x.shape
            x_flat = x.reshape(-1, x.shape[-1]).float()
            x_flat = (self.V.to(x_flat.dtype) @ x_flat.t()).t()
            x = x_flat.reshape(x_orig_shape).to(x.dtype)

        W   = self.dequant_weight()
        out = x @ W.t()

        if self.bias is not None:
            out = out + self.bias
        return out


def build_e8p_model(model, quantized_layers, rotations):
    """
    fake quant 모델을 real quant E8PLinear로 교체.

    quantized_layers: {
        'layer0.self_attn.q_proj': {
            'idx': ..., 'scale': ..., 'U': ..., 'V': ...
        }, ...
    }
    rotations: {layer_name: (U, V)} optional
    """
    layers = model.model.layers

    for full_name, data in quantized_layers.items():
        # full_name: 'layer0.self_attn.q_proj'
        parts      = full_name.split('.')
        layer_idx  = int(parts[0].replace('layer', ''))
        layer      = layers[layer_idx]
        sub_name   = '.'.join(parts[1:])   # 'self_attn.q_proj'

        # sub_name으로 layer 안에서 찾기
        parent = layer
        sub_parts = sub_name.split('.')
        for p in sub_parts[:-1]:
            parent = getattr(parent, p)
        orig = getattr(parent, sub_parts[-1])

        d_row, d_col = orig.weight.shape

        e8p_layer = E8PLinear(
            d_row  = d_row,
            d_col  = d_col,
            idx    = data['idx'],
            scale  = data['scale'],
            U      = data.get('U', None),
            V      = data.get('V', None),
            bias   = orig.bias,
        )
        setattr(parent, sub_parts[-1], e8p_layer)
        mem_kb = d_row * d_col * 2 / 8 / 1024
        print(f"  replaced {full_name}: {d_row}×{d_col} → E8P ({mem_kb:.1f}KB)")

    return model
