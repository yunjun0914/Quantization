"""
LLaMA Rotated GPTQ v2 - Globally Shared V & U
===============================================

핵심 설계:
  - V_global: 전체 모델 공유 (hidden × hidden, Hadamard)
  - V_inter:  전체 모델 공유 (inter × inter, random)
  - U: 6종 globally shared (layer마다 동일)

globally shared 이유:
  - V: hidden state가 항상 같은 회전 공간 → residual 문제 해결
  - U: 랜덤 회전인데 layer마다 다르게 뽑을 이유 없음

U 구조:
  q, k    → U_qk  (QK에서 U_qk² = I 자동 소거)
  v       → U_v   (o_proj columns에 흡수)
  o_proj  → U_o   (다음 layer input_layernorm weight에 흡수)
  gate,up → U_gu  (down_proj columns에 흡수)
  down    → U_d   (다음 layer post_attention_layernorm weight에 흡수)

V 흡수:
  모든 linear: W_stored = Q @ V  (inference 시 추가 matmul 없음)
"""

import time
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM

from gptq_rot import RotatedGPTQ
from rotation import get_rotation, get_sign_vector, rotate_weight
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
    print(f"  Globally shared V & U")
    model.eval()

    layers = get_llama_layers(model)
    dtype  = next(iter(model.parameters())).dtype
    device = torch.device(dev)
    hidden = model.config.hidden_size       # 4096
    inter  = model.config.intermediate_size  # 11008

    use_cache = model.config.use_cache
    model.config.use_cache = False

    # ── Global V, U 생성 (한 번만) ───────────────────────────────────────
    V_global = get_rotation(hidden, mode=rot_mode, seed=seed,   device=device)
    V_inter  = get_rotation(inter,  mode="random", seed=seed+1, device=device)

    U_qk = get_rotation(hidden, mode=rot_mode, seed=seed+10, device=device)
    U_v  = get_rotation(hidden, mode=rot_mode, seed=seed+11, device=device)
    U_o  = get_sign_vector(hidden, seed=seed+12, device=device)
    U_gu = get_sign_vector(inter,  seed=seed+13, device=device)
    U_d  = get_sign_vector(hidden, seed=seed+14, device=device)

    print(f"  V_global ({hidden},{hidden})  V_inter ({inter},{inter})")

    rotations = {
        "self_attn.q_proj": (U_qk, V_global),
        "self_attn.k_proj": (U_qk, V_global),
        "self_attn.v_proj": (U_v,  V_global),
        "self_attn.o_proj": (U_o,  V_global),
        "mlp.gate_proj":    (U_gu, V_global),
        "mlp.up_proj":      (U_gu, V_global),
        "mlp.down_proj":    (U_d,  V_inter),
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

        # ── Step 1: Hessian 수집 (원래 weight) ───────────────────────────
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
                if handler.V.dim() == 2:
                    Ho = (handler.V.t() @ Ht @ handler.V).diag()
                    print(f"    {name:25s}  H std={Ho.std():.2f} max={Ho.max():.2f} | "
                          f"H̃ std={Ht.diag().std():.2f} max={Ht.diag().max():.2f}")

        # ── Step 2: V만 적용 ──────────────────────────────────────────────
        for name, lin in linears.items():
            if name not in rotations: continue
            _, V = rotations[name]
            lin.weight.data = (lin.weight.data.float() @ V.t()).to(dtype)

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

        # ── Step 4: V 흡수 ────────────────────────────────────────────────
        for name, lin in linears.items():
            if name not in Q_dict: continue
            Q   = Q_dict[name]
            _, V = rotations[name]
            W_v = (Q.float() @ V.float()).to(dtype)
            lin.weight.data = W_v

        # ── Step 5: U 흡수 ────────────────────────────────────────────────
        # U_v → o_proj columns
        if "self_attn.o_proj" in linears:
            W_o = linears["self_attn.o_proj"].weight.data.float()
            if U_v.dim() == 2:
                linears["self_attn.o_proj"].weight.data = (W_o @ U_v.t()).to(dtype)
            else:
                linears["self_attn.o_proj"].weight.data = (W_o * U_v.unsqueeze(0)).to(dtype)

        # U_gu → down_proj columns (sign vector)
        if "mlp.down_proj" in linears:
            W_d = linears["mlp.down_proj"].weight.data.float()
            linears["mlp.down_proj"].weight.data = (W_d * U_gu.unsqueeze(0)).to(dtype)

        # U_o → 다음 layer input_layernorm (globally shared V 덕분에 가능)
        if hasattr(layer, 'post_attention_layernorm'):
            layer.post_attention_layernorm.weight.data = (
                layer.post_attention_layernorm.weight.data.float() * U_o.float()
            ).to(dtype)

        # U_d → 다음 layer input_layernorm
        if layer_idx < len(layers) - 1:
            next_layer = layers[layer_idx + 1].to(device)
            if hasattr(next_layer, 'input_layernorm'):
                next_layer.input_layernorm.weight.data = (
                    next_layer.input_layernorm.weight.data.float() * U_d.float()
                ).to(dtype)
            next_layer = next_layer.cpu()

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
    tag     = "RotatedGPTQ v2"
    print(f"[{bits}bit {tag}] PPL = {ppl_q:.2f}")

    return {"ppl_fp16": ppl_fp16, "ppl_quant": ppl_q, "results": results}
