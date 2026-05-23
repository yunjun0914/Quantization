"""
LLaMA Rotated GPTQ v2 - Proper Absorption
==========================================

기존 문제: unrotate_weight = U^T @ Q @ V 를 fp16으로 저장
          → 자체 정밀도 문제는 없지만, V matmul이 한 번 더 발생

v2 전략: V를 weight에 흡수(Q @ V, fp32), U는 인접 레이어에 흡수
         → inference 시 추가 연산 없음, 오차 원인 제거

U 흡수 규칙 (LLaMA 구조 기준):
  q, k   → 같은 U_qk 공유: (U⊙q)@(U⊙k)^T = q@k^T  (U²=1 소거) ✓
  v      → U_v를 o_proj columns에 흡수:
            o_proj[:,j] *= U_v[j]  →  o_proj @ (U_v⊙v) = (o_proj*U_v) @ v ✓
  gate,up → 같은 U_gu 공유: SiLU(U⊙g)*(U⊙u) → down_proj columns에 흡수:
            down_proj[:,j] *= U_gu[j]  →  down_proj @ (U_gu⊙swiglu) ✓
  o_proj  → output이 residual에 더해짐 → U = +1 (identity)
  down    → output이 residual에 더해짐 → U = +1 (identity)

V 흡수:
  모든 linear: W_stored = Q(UWV^T) @ V  (fp32 계산, fp16 저장)
  → inference 시 W_stored @ x = Q(UWV^T) @ V @ x  (추가 matmul 없음)
"""

import time
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM

from gptq_rot import RotatedGPTQ
from svd_correction import apply_svd_corrections
from rotation import get_rotation, get_sign_vector, rotate_weight
from data import get_loaders


def get_llama_layers(model):
    return model.model.layers


def find_linear_layers(layer):
    return {name: m for name, m in layer.named_modules() if isinstance(m, nn.Linear)}


def get_rotations_v2(layer_idx, hidden, inter, rot_mode, seed, device):
    """
    V 공유 + U 흡수 가능한 구조로 rotation 생성.

    U 규칙:
      q, k  → U_qk (공유, QK에서 소거)
      v     → U_v  (o_proj에 흡수)
      gate, up → U_gu (공유, down_proj에 흡수)
      o_proj, down_proj → U = +1 (identity, residual 때문)
    """
    base = seed + layer_idx * 20

    V_attn    = get_rotation(hidden, mode=rot_mode, seed=base+0, device=device)
    V_o       = get_rotation(hidden, mode=rot_mode, seed=base+1, device=device)
    V_gate_up = get_rotation(hidden, mode=rot_mode, seed=base+2, device=device)
    V_down    = get_rotation(inter,  mode="random", seed=base+3, device=device)

    # U: Hadamard (hidden=4096=2^12) → 진짜 two-sided Hadamard rotation
    # U^T 보정이 inner loop에서 적용되므로 full matrix 가능
    U_qk     = get_rotation(hidden, mode="hadamard", seed=base+10, device=device)  # (hidden, hidden)
    U_v      = get_rotation(hidden, mode="hadamard", seed=base+11, device=device)
    U_gu     = get_rotation(inter,  mode="random",   seed=base+12, device=device)  # inter는 random
    U_ones_h  = torch.eye(hidden, device=device)   # o_proj: identity
    U_ones_h2 = torch.eye(hidden, device=device)   # down_proj: identity

    rotations = {
        "self_attn.q_proj":  (U_qk,      V_attn),
        "self_attn.k_proj":  (U_qk,      V_attn),    # q와 동일 U_qk
        "self_attn.v_proj":  (U_v,       V_attn),
        "self_attn.o_proj":  (U_ones_h,  V_o),       # U = +1
        "mlp.gate_proj":     (U_gu,      V_gate_up),
        "mlp.up_proj":       (U_gu,      V_gate_up), # gate와 동일 U_gu
        "mlp.down_proj":     (U_ones_h2, V_down),    # U = +1
    }

    # 흡수 정보: {layer_name: (target_name, side)}
    absorb_map = {
        "self_attn.v_proj": ("self_attn.o_proj", "col"),   # U_v → o_proj columns
        "mlp.gate_proj":    ("mlp.down_proj",    "col"),   # U_gu → down_proj columns
    }

    return rotations, absorb_map, {"U_v": U_v, "U_gu": U_gu}


@torch.no_grad()
def llama_rot_sequential_v2(
    model, dataloader, dev,
    bits=4, blocksize=128, percdamp=0.01,
    groupsize=-1, sym=False, actorder=False,
    rot_mode="hadamard", seed=0,
):
    print(f"[LLaMA RotatedGPTQ v2] bits={bits}  blocksize={blocksize}  rot={rot_mode}")
    print(f"  V absorbed into weights (fp32), U absorbed into adjacent layers")
    model.eval()

    layers   = get_llama_layers(model)
    dtype    = next(iter(model.parameters())).dtype
    device   = torch.device(dev)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    # ── 입력 캡처 ─────────────────────────────────────────────────────────
    nsamples = len(dataloader)
    seqlen   = dataloader[0].shape[1]
    inps     = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size),
        dtype=dtype, device="cpu",
    )
    cache = {"i": 0, "position_ids": None}

    class CatchInput(nn.Module):
        def __init__(self, m): super().__init__(); self.module = m
        def forward(self, inp, **kw):
            inps[cache["i"]] = inp[0].detach().cpu()
            cache["position_ids"] = kw.get("position_ids", None)
            cache["i"] += 1; raise StopIteration

    model = model.to(device)
    layers[0] = CatchInput(layers[0]); model.model.layers[0] = layers[0]
    for b in dataloader:
        try: model(b.to(device))
        except StopIteration: pass
    layers[0] = layers[0].module; model.model.layers[0] = layers[0]
    model = model.cpu(); torch.cuda.empty_cache()

    results  = {}
    hidden   = model.config.hidden_size
    inter    = model.config.intermediate_size

    for layer_idx, layer in enumerate(layers):
        print(f"\n[Layer {layer_idx:02d}/{len(layers)-1}]")
        t0    = time.time()
        layer = layer.to(device)
        linears = find_linear_layers(layer)

        # ── rotation 생성 ──────────────────────────────────────────────────
        rotations, absorb_map, u_vecs = get_rotations_v2(
            layer_idx, hidden, inter, rot_mode, seed, device
        )

        # ── Step 1: 원래 weight로 Hessian 수집 ───────────────────────────
        handlers = {
            name: RotatedGPTQ(lin, *rotations[name])
            for name, lin in linears.items()
            if name in rotations
        }

        hooks = []
        for name, lin in linears.items():
            if name not in handlers: continue
            def make_hook(h):
                def hook(m, inp, out): h.add_batch(inp[0].data, out.data)
                return hook
            hooks.append(lin.register_forward_hook(make_hook(handlers[name])))

        for i in range(nsamples):
            kw = {}
            if cache["position_ids"] is not None:
                kw["position_ids"] = cache["position_ids"].to(device)
            layer(inps[i].unsqueeze(0).to(device), **kw)
        for h in hooks: h.remove()

        # ── Step 2: weight rotate ─────────────────────────────────────────
        for name, lin in linears.items():
            if name in rotations:
                lin.weight.data = rotate_weight(lin.weight.data, *rotations[name])

        # ── Step 3: GPTQ 실행 ─────────────────────────────────────────────
        if layer_idx == 0:
            print("  [H diag stats]")
            for name, handler in handlers.items():
                Ht = handler.H.float()
                Ho = (handler.V.t() @ Ht @ handler.V).diag()
                print(f"    {name:25s}  H std={Ho.std():.2f} max={Ho.max():.2f} | "
                      f"H̃ std={Ht.diag().std():.2f} max={Ht.diag().max():.2f}")

        Q_dict = {}
        for name, handler in handlers.items():
            print(f"  quantizing {name:30s} ... ", end="", flush=True)
            Q, scale, zero, loss = handler.quantize(
                bits=bits, blocksize=blocksize, percdamp=percdamp,
                groupsize=groupsize, sym=sym, actorder=actorder,
            )
            Q_dict[name] = Q
            results[f"layer{layer_idx}.{name}"] = {
                "Q": Q.cpu(), "scale": scale.cpu(), "zero": zero.cpu(),
                "loss": loss.mean().item(),
                "W_orig": (linears[name].weight.data.float() @ rotations[name][1].float()).cpu().half(),
            }
            print(f"loss={loss.mean().item():.6f}")
            handler.free()

        # ── Step 4: V 흡수 (핵심) ─────────────────────────────────────────
        # W_stored = Q @ V  (fp32 계산, fp16 저장)
        # inference: W_stored @ x = Q(UWV^T) @ V @ x  ✓  (추가 matmul 없음)
        for name, lin in linears.items():
            if name not in Q_dict: continue
            Q = Q_dict[name]
            U, V = rotations[name]
            # Q @ V: V 흡수 (fp32)
            W_v_abs = (Q.float() @ V.float()).to(dtype)
            lin.weight.data = W_v_abs

        # ── Step 5: U 흡수 ─────────────────────────────────────────────────
        # U_v → o_proj columns
        # U_v가 full matrix이면: o_proj_new = o_proj @ U_v^T
        # U_v가 vector(±1)이면: o_proj[:,j] *= U_v[j]
        if "self_attn.o_proj" in linears and "self_attn.v_proj" in linears:
            U_v = u_vecs["U_v"]
            W_o = linears["self_attn.o_proj"].weight.data.float()
            if U_v.dim() == 2:
                W_o_new = W_o @ U_v.t()   # o_proj @ U_v^T
            else:
                W_o_new = W_o * U_v.unsqueeze(0)
            linears["self_attn.o_proj"].weight.data = W_o_new.to(dtype)

        # U_gu → down_proj columns
        if "mlp.down_proj" in linears and "mlp.gate_proj" in linears:
            U_gu = u_vecs["U_gu"]
            W_d = linears["mlp.down_proj"].weight.data.float()
            if U_gu.dim() == 2:
                W_d_new = W_d @ U_gu.t()
            else:
                W_d_new = W_d * U_gu.unsqueeze(0)
            linears["mlp.down_proj"].weight.data = W_d_new.to(dtype)

        # ── 다음 layer 입력 업데이트 ──────────────────────────────────────
        for i in range(nsamples):
            kw = {}
            if cache["position_ids"] is not None:
                kw["position_ids"] = cache["position_ids"].to(device)
            out   = layer(inps[i].unsqueeze(0).to(device), **kw)
            inps[i] = out[0].detach().cpu()

        layer = layer.cpu(); torch.cuda.empty_cache()
        print(f"  ↳ done in {time.time()-t0:.1f}s")

    model.config.use_cache = use_cache
    return results


@torch.no_grad()
def eval_ppl(model, testenc, dev, seqlen=2048):
    model.eval()
    device    = torch.device(dev)
    input_ids = (testenc.input_ids if hasattr(testenc, "input_ids") else testenc).to(device)
    nsamples  = input_ids.shape[1] // seqlen
    total_nll = 0.0
    for i in range(nsamples):
        chunk = input_ids[:, i*seqlen:(i+1)*seqlen]
        out   = model(chunk, labels=chunk.clone())
        total_nll += out.loss.item() * seqlen
    return torch.exp(torch.tensor(total_nll / (nsamples * seqlen))).item()


def run_llama_rot_v2(
    model_name="meta-llama/Llama-2-7b-hf", bits=4, dataset="wikitext2",
    nsamples=128, seqlen=2048, seed=0, blocksize=128, percdamp=0.01,
    groupsize=-1, sym=False, actorder=False, rot_mode="hadamard",
    dev="cuda:0", eval_before=True, svd_rank=0,
):
    print(f"Loading model: {model_name}")
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    trainloader, _ = get_loaders(dataset, nsamples=nsamples, seed=seed, seqlen=seqlen, model=model_name)
    _, testenc     = get_loaders("wikitext2", nsamples=nsamples, seed=seed, seqlen=seqlen, model=model_name)

    ppl_fp16 = None
    if eval_before:
        model = model.to(dev)
        ppl_fp16 = eval_ppl(model, testenc, dev, seqlen)
        print(f"\n[FP16 baseline] PPL = {ppl_fp16:.2f}")
        model = model.cpu()

    t0      = time.time()
    results = llama_rot_sequential_v2(
        model, trainloader, dev,
        bits=bits, blocksize=blocksize, percdamp=percdamp,
        groupsize=groupsize, sym=sym, actorder=actorder,
        rot_mode=rot_mode, seed=seed,
    )
    print(f"\n[RotatedGPTQ v2] Total time: {time.time()-t0:.1f}s")

    if svd_rank > 0:
        print(f"\n[SVD correction] rank={svd_rank}")
        apply_svd_corrections(
            model, results,
            get_layers_fn=get_llama_layers,
            find_linears_fn=find_linear_layers,
            rank=svd_rank,
            verbose=True,
        )

    model   = model.to(dev)
    ppl_q   = eval_ppl(model, testenc, dev, seqlen)
    tag     = f"RotatedGPTQ v2 + SVD(r={svd_rank})" if svd_rank > 0 else "RotatedGPTQ v2"
    print(f"[{bits}bit {tag}] PPL = {ppl_q:.2f}")

    return {"ppl_fp16": ppl_fp16, "ppl_quant": ppl_q, "results": results}
