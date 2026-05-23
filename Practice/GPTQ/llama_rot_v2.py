"""
LLaMA Rotated GPTQ v2
======================
QuaRot 구조 기반:
  - V_global (Hadamard, hidden×hidden): 모든 linear에 동일하게 적용
  - U_qk: q,k output rotation (QK에서 자동 소거)
  - U_v:  v output rotation   (o_proj columns에 흡수)
  - U_gu: gate,up output rotation (down_proj columns에 흡수)

모든 rotation은 weight에 흡수 → inference 추가 비용 없음
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
    hidden = model.config.hidden_size       # 4096

    use_cache = model.config.use_cache
    model.config.use_cache = False

    # ── Global V, U (한 번만 생성) ────────────────────────────────────────
    V = get_rotation(hidden, mode=rot_mode, seed=seed, device=device)   # (4096, 4096)
    U_qk = get_rotation(hidden, mode=rot_mode, seed=seed+1, device=device)
    U_v  = get_rotation(hidden, mode=rot_mode, seed=seed+2, device=device)
    U_gu = get_sign_vector(hidden, seed=seed+3, device=device)

    # 모든 linear: V 동일하게 적용
    # U는 absorption 가능한 것만
    rotations = {
        "self_attn.q_proj": (U_qk, V),
        "self_attn.k_proj": (U_qk, V),
        "self_attn.v_proj": (U_v,  V),
        "self_attn.o_proj": (U_qk, V),  # U_qk: QK 이후 출력 공간
        "mlp.gate_proj":    (U_gu, V),
        "mlp.up_proj":      (U_gu, V),
        "mlp.down_proj":    (U_gu, V),  # input이 gate/up output 공간
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
        handlers = {
            name: RotatedGPTQ(lin, *rotations[name])
            for name, lin in linears.items() if name in rotations
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

        # ── H diag stats (layer 0) ────────────────────────────────────────
        if layer_idx == 0:
            print("  [H diag stats]")
            for name, handler in handlers.items():
                Ht = handler.H.float()
                Ht_diag = Ht.diag()
                print(f"    {name:25s}  "
                      f"H̃ std={Ht_diag.std():.4f} max={Ht_diag.max():.4f} "
                      f"min={Ht_diag.min():.6f}")

        # ── Step 2: V 적용 (W @ V^T) ─────────────────────────────────────
        for name, lin in linears.items():
            if name not in rotations: continue
            _, Vr = rotations[name]
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
            lin.weight.data = (Q_dict[name].float() @ Vr.float()).to(dtype)

        # ── Step 5: U 흡수 ────────────────────────────────────────────────
        # U_v → o_proj columns
        if "self_attn.o_proj" in linears:
            W_o = linears["self_attn.o_proj"].weight.data.float()
            linears["self_attn.o_proj"].weight.data = (W_o @ U_v.t()).to(dtype)

        # U_gu → down_proj columns
        if "mlp.down_proj" in linears:
            W_d = linears["mlp.down_proj"].weight.data.float()
            linears["mlp.down_proj"].weight.data = (W_d @ U_gu.t() if U_gu.dim()==2
                                                    else W_d * U_gu.unsqueeze(0)).to(dtype)

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
