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
        self.register_buffer('idx',   idx.cpu().to(torch.int16))
        self.register_buffer('scale', scale.cpu())
        self.d_row = d_row
        self.d_col = d_col
        # globally shared 참조 (register_buffer 안 함 → 복사본 생성 방지)
        # build_e8p_model에서 .to(device) 후 참조 전달
        self.U = U
        self.V = V

        if bias is not None:
            self.register_buffer('bias', bias.cpu())
        else:
            self.bias = None

    def dequant_weight(self):
        """
        E8P index → U^T @ Q_rot @ V (fake quant의 W_stored와 동일)

        idx: UWV^T 공간에서 수집된 E8P index
        cb[idx] = Q_rot (UWV^T 공간)
        U^T @ cb[idx] = WV^T 공간 (uwvt_mode에서 최종 Q = U^T @ Q_rot)
        @ V → W 공간

        fake quant W_stored = U^T @ Q_rot @ V 와 동일
        """
        cb = get_e8p_codebook(self.idx.device)
        q  = cb[self.idx.to(torch.int32)]              # (d_row//8, d_col, 8)
        q  = q * self.scale
        q  = q.permute(0, 2, 1).reshape(self.d_row, self.d_col)  # UWV^T 공간

        # U^T 적용 → WV^T 공간
        if self.U is not None:
            U = self.U if self.U.device == q.device else self.U.to(q.device)
            q = U.t().float() @ q.float()

        # V 적용 → W 공간
        if self.V is not None:
            V = self.V if self.V.device == q.device else self.V.to(q.device)
            q = q.float() @ V.float()

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

        # U, V: rotations에서 가져오기
        U_layer, V_layer = None, None
        if rotations and sub_name in rotations:
            U_layer, V_layer = rotations[sub_name]

        e8p_layer = E8PLinear(
            d_row = d_row,
            d_col = d_col,
            idx   = data['idx'],
            scale = data['scale'],
            U     = U_layer,
            V     = V_layer,
            bias  = orig.bias,
        )
        setattr(parent, sub_parts[-1], e8p_layer)
        mem_kb = d_row * d_col * 2 / 8 / 1024
        print(f"  replaced {full_name}: {d_row}×{d_col} → E8P ({mem_kb:.1f}KB)")

    # 실제 저장 크기 출력
    total_bytes = sum(
        p.numel() * p.element_size()
        for m in model.modules() if isinstance(m, E8PLinear)
        for p in m.buffers()
    )
    print(f"  [Real Quant] 총 저장 크기: {total_bytes/1e9:.2f}GB")
    idx_bytes = sum(
        m.idx.numel() * m.idx.element_size()
        for m in model.modules() if isinstance(m, E8PLinear)
    )
    print(f"  [Real Quant] idx (int16): {idx_bytes/1e9:.2f}GB = {idx_bytes*8/sum(m.idx.numel()*8 for m in model.modules() if isinstance(m, E8PLinear)):.2f}bpw")

    # U, V unique tensor들을 GPU로 올리기
    # CPU tensor id 기준으로 중복 제거 → 같은 tensor 공유
    dev = next(model.parameters()).device
    moved = {}  # cpu_id -> gpu_tensor
    for m in model.modules():
        if isinstance(m, E8PLinear):
            if m.U is not None:
                cpu_id = id(m.U)
                if cpu_id not in moved:
                    moved[cpu_id] = m.U.to(dev)
                m.U = moved[cpu_id]   # 같은 GPU tensor 공유
            if m.V is not None:
                cpu_id = id(m.V)
                if cpu_id not in moved:
                    moved[cpu_id] = m.V.to(dev)
                m.V = moved[cpu_id]   # 같은 GPU tensor 공유
    print(f"  U/V unique tensors on GPU: {len(moved)}개")

    return model, hooks
