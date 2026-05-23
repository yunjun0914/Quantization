"""
LLaMA Rotated GPTQ v2
======================
QuaRot 구조 기반 (R1=V_global, R2=U_v, R3=U_qk, R4=U_gu)

weight 흡수:
  q_proj:    U_qk @ W @ V^T   (U=R3, V=R1^-1)
  k_proj:    U_qk @ W @ V^T
  v_proj:    U_v  @ W @ V^T   (U=R2, V=R1^-1)
  o_proj:    W @ V @ U_v^T    (V=R1, U_v^T=R2^-1 columns 흡수)
  gate_proj: W @ V^T          (V=R1^-1)
  up_proj:   W @ V^T
  down_proj: W @ V @ U_gu^T   (V=R1, U_gu^T=R4^-1 columns 흡수)

복원:
  U_qk: QK^T에서 U_qk U_qk^T = I 자동 소거
  U_v:  o_proj에서 U_v^T U_v = I 소거
  U_gu: down_proj에서 U_gu^T U_gu = I 소거
  V:    chain으로 V V^T = I 소거

추가 inference 비용 없음.
"""

import time
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM

from gptq_rot import RotatedGPTQ
from rotation import get_rotation, get_sign_vector
from data import get_loaders


def get_llama_layers(model):
    return model.model.layers


def find_linear_layers(layer):
    return {name: m for name, m in layer.named_modules() if isinstance(m, nn.Linear)}


@torch.no_grad()
def llama_rot_sequential_v2(
    model, dataloader, dev,
    bits=4, blocksize=128, percdamp=0.01,
    groupsize=-1, sym=False, actorder=False,
    rot_mode="hadamard", seed=0,
):
    print(f"[LLaMA RotatedGPTQ v2] bits={bits}  blocksize={blocksize}  rot={rot_mode}")
    model.eval()

    layers = get_llama_layers(model)
    dtype  = next(iter(model.parameters())).dtype
    device = torch.device(dev)
    hidden = model.config.hidden_size        # 4096
    inter  = model.config.intermediate_size  # 11008

    use_cache = model.config.use_cache
    model.config.use_cache = False

    # ── Global V, U (한 번만 생성) ────────────────────────────────────────
    V    = get_rotation(hidden, mode=rot_mode, seed=seed,   device=device)  # R1
    U_qk = get_rotation(hidden, mode=rot_mode, seed=seed+1, device=device)  # R3
    U_v  = get_rotation(hidden, mode=rot_mode, seed=seed+2, device=device)  # R2
    U_gu = get_rotation(inter, mode=rot_mode, seed=seed+3, device=device)   # R4 (full)

    print(f"  V({hidden},{hidden})  U_qk({hidden},{hidden})  U_v({hidden},{hidden})  U_gu({inter},)")

    # GPTQ inner loop용: (U, V) 쌍
    # q/k/v: U@W@V^T 형태로 양자화
    # o/gate/up/down: U=None (inner loop에서 회전 없음, absorption만)
    rotations = {
        "self_attn.q_proj": (U_qk, V),
        "self_attn.k_proj": (U_qk, V),
        "self_attn.v_proj": (U_v,  V),
        "self_attn.o_proj": (None, V),
        "mlp.gate_proj":    (None, V),
        "mlp.up_proj":      (None, V),
        "mlp.down_proj":    (None, U_gu),  # V=U_gu (input 공간 rotation)
    }

    # ── 입력 캡처 ─────────────────────────────────────────────────────────
    nsamples = len(dataloader)
    seqlen   = dataloader[0].shape[1]
    inps     = torch.zeros((nsamples, seqlen, hidden), dtype=dtype, device="cpu")
    cache    = {"i": 0, "position_ids": None}

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

    results = {}

    for layer_idx, layer in enumerate(layers):
        print(f"\n[Layer {layer_idx:02d}/{len(layers)-1}]")
        t0    = time.time()
        layer = layer.to(device)
        linears = find_linear_layers(layer)

        # ── Step 1: Hessian 수집 ──────────────────────────────────────────
        # U=None인 layer도 V rotation은 적용 → add_batch에서 V@X
        handlers = {}
        for name, lin in linears.items():
            if name not in rotations: continue
            U, Vr = rotations[name]
            if Vr is None: continue  # down_proj: GPTQ 없이 absorption만
            U_eff = U if U is not None else torch.ones(lin.weight.shape[0], device=device)
            handlers[name] = RotatedGPTQ(lin, U_eff, Vr)

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

        # ── H diag stats (layer 0) ────────────────────────────────────────
        if layer_idx == 0:
            print("  [H diag stats]")
            for name, handler in handlers.items():
                Ht = handler.H.float()
                d  = Ht.diag()
                print(f"    {name:25s}  H̃ std={d.std():.4f}  max={d.max():.4f}  min={d.min():.6f}")

        # ── Step 2: W @ V^T ───────────────────────────────────────────────
        for name, lin in linears.items():
            if name not in rotations: continue
            _, Vr = rotations[name]
            if Vr is None: continue
            lin.weight.data = (lin.weight.data.float() @ Vr.t()).to(dtype)

        # ── Step 3: GPTQ 실행 ─────────────────────────────────────────────
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
            }
            print(f"loss={loss.mean().item():.6f}")
            handler.free()

        # ── Step 4: V 흡수 (Q @ V) ───────────────────────────────────────
        for name, lin in linears.items():
            if name not in Q_dict: continue
            _, Vr = rotations[name]
            if Vr is None:
                lin.weight.data = Q_dict[name]
            else:
                lin.weight.data = (Q_dict[name].float() @ Vr.float()).to(dtype)

        # ── Step 5: U absorption ──────────────────────────────────────────
        # U_v^T → o_proj columns: W_o @ U_v^T
        if "self_attn.o_proj" in linears:
            W_o = linears["self_attn.o_proj"].weight.data.float()
            linears["self_attn.o_proj"].weight.data = (W_o @ U_v.t()).to(dtype)

        # down_proj: V@W@U_gu^T
        #   output방향: V (row) → V @ W_down
        #   input방향:  U_gu^T (column, sign) → * U_gu elementwise
        if "mlp.down_proj" in linears:
            W_d = linears["mlp.down_proj"].weight.data.float()
            W_d = V.float() @ W_d           # V @ W: (4096,4096)@(4096,11008)
            W_d = W_d * U_gu.unsqueeze(0)   # U_gu^T: (4096,11008)*(1,11008)
            linears["mlp.down_proj"].weight.data = W_d.to(dtype)

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
    dev="cuda:0", eval_before=True,
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

    model   = model.to(dev)
    ppl_q   = eval_ppl(model, testenc, dev, seqlen)
    print(f"[{bits}bit RotatedGPTQ v2] PPL = {ppl_q:.2f}")

    return {"ppl_fp16": ppl_fp16, "ppl_quant": ppl_q, "results": results}
