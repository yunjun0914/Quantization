"""
LLaMA Rotated GPTQ 파이프라인
==============================
LLaMA-7B 기준:
  hidden_size  = 4096 = 2^12  → Hadamard 완벽 적용 가능
  intermediate = 11008         → random orthogonal 사용

V 공유 기준:
  q/k/v_proj   → V_attn  (input_layernorm 출력 공유)
  o_proj       → V_o     (attention 출력)
  gate/up_proj → V_gate_up (post_attention_layernorm 출력 공유)
  down_proj    → V_down  (SwiGLU 출력)
"""

import time
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, AutoTokenizer

from gptq_rot import RotatedGPTQ
from rotation import get_rotation, get_sign_vector, rotate_weight, unrotate_weight
from data import get_loaders


def get_llama_layers(model):
    return model.model.layers


def find_linear_layers(layer):
    return {
        name: module
        for name, module in layer.named_modules()
        if isinstance(module, nn.Linear)
    }


@torch.no_grad()
def llama_rot_sequential(
    model,
    dataloader,
    dev,
    bits        = 4,
    blocksize   = 128,
    percdamp    = 0.01,
    groupsize   = -1,
    sym         = False,
    actorder    = False,
    rot_mode    = "hadamard",   # LLaMA hidden=4096=2^12 → hadamard 가능
    seed        = 0,
):
    print(f"[LLaMA RotatedGPTQ] bits={bits}  blocksize={blocksize}  rot={rot_mode}")
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
    cache = {"i": 0, "position_ids": None, "attention_mask": None}

    class CatchInput(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp[0].detach().cpu()
            cache["position_ids"]  = kwargs.get("position_ids", None)
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            cache["i"] += 1
            raise StopIteration

    model = model.to(device)
    layers[0] = CatchInput(layers[0])
    model.model.layers[0] = layers[0]

    for batch in dataloader:
        try:
            model(batch.to(device))
        except StopIteration:
            pass

    layers[0] = layers[0].module
    model.model.layers[0] = layers[0]
    model = model.cpu()
    torch.cuda.empty_cache()

    # ── Layer-wise Rotated GPTQ ────────────────────────────────────────────
    results  = {}
    hidden   = model.config.hidden_size
    inter    = model.config.intermediate_size

    for layer_idx, layer in enumerate(layers):
        print(f"\n[Layer {layer_idx:02d}/{len(layers)-1}]")
        t0    = time.time()
        layer = layer.to(device)

        linears = find_linear_layers(layer)

        # ── V 공유 생성 ───────────────────────────────────────────────────
        V_attn    = get_rotation(hidden, mode=rot_mode, seed=seed + layer_idx*10+0, device=device)
        V_o       = get_rotation(hidden, mode=rot_mode, seed=seed + layer_idx*10+1, device=device)
        V_gate_up = get_rotation(hidden, mode=rot_mode, seed=seed + layer_idx*10+2, device=device)
        V_down    = get_rotation(inter,  mode="random", seed=seed + layer_idx*10+3, device=device)

        def pick_V(name):
            if name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
                return V_attn
            elif name in ("self_attn.o_proj",):
                return V_o
            elif name in ("mlp.gate_proj", "mlp.up_proj"):
                return V_gate_up
            elif name in ("mlp.down_proj",):
                return V_down
            else:
                d_col = linears[name].weight.shape[1]
                return get_rotation(d_col, mode="random", seed=seed+layer_idx+hash(name), device=device)

        rotations = {}
        for name, lin in linears.items():
            V = pick_V(name)
            U = get_sign_vector(lin.weight.shape[0],
                                seed=seed + layer_idx*10 + hash(name), device=device)
            rotations[name] = (U, V)

        # ── Step 1: 원래 weight로 Hessian 수집 ───────────────────────────
        handlers = {
            name: RotatedGPTQ(lin, *rotations[name])
            for name, lin in linears.items()
        }

        hooks = []
        for name, lin in linears.items():
            def make_hook(h):
                def hook(m, inp, out):
                    h.add_batch(inp[0].data, out.data)
                return hook
            hooks.append(lin.register_forward_hook(make_hook(handlers[name])))

        for i in range(nsamples):
            inp_i = inps[i].unsqueeze(0).to(device)
            kw = {}
            if cache["position_ids"] is not None:
                kw["position_ids"] = cache["position_ids"].to(device)
            layer(inp_i, **kw)

        for h in hooks:
            h.remove()

        # ── Step 2: weight rotate ─────────────────────────────────────────
        for name, lin in linears.items():
            lin.weight.data = rotate_weight(lin.weight.data, *rotations[name])

        # ── RotatedGPTQ 실행 ──────────────────────────────────────────────
        Q_store = {}

        # H diag stats (layer 0만)
        if layer_idx == 0:
            print(f"  [H diag stats]")
            for name, handler in handlers.items():
                Ht = handler.H.float()
                H_orig_diag = (handler.V.t() @ Ht @ handler.V).diag()
                print(f"    {name:25s}  "
                      f"H  std={H_orig_diag.std():.2f} max={H_orig_diag.max():.2f} | "
                      f"H̃ std={Ht.diag().std():.2f} max={Ht.diag().max():.2f}")

        for name, handler in handlers.items():
            print(f"  quantizing {name:30s} ... ", end="", flush=True)
            Q, scale, zero, loss = handler.quantize(
                bits=bits, blocksize=blocksize, percdamp=percdamp,
                groupsize=groupsize, sym=sym, actorder=actorder,
            )
            Q_store[name] = Q
            linears[name].weight.data = unrotate_weight(Q, *rotations[name])
            results[f"layer{layer_idx}.{name}"] = {
                "Q": Q.cpu(), "scale": scale.cpu(), "zero": zero.cpu(),
                "loss": loss.mean().item(),
                "U": rotations[name][0].cpu(), "V": rotations[name][1].cpu(),
            }
            print(f"loss={loss.mean().item():.6f}")
            handler.free()

        # 다음 layer 입력 업데이트
        for i in range(nsamples):
            inp_i = inps[i].unsqueeze(0).to(device)
            kw = {}
            if cache["position_ids"] is not None:
                kw["position_ids"] = cache["position_ids"].to(device)
            out   = layer(inp_i, **kw)
            inps[i] = out[0].detach().cpu()

        # inference용 weight 복원
        for name, lin in linears.items():
            lin.weight.data = Q_store[name]

        layer = layer.cpu()
        torch.cuda.empty_cache()
        print(f"  ↳ done in {time.time()-t0:.1f}s")

    model.config.use_cache = use_cache
    return results


@torch.no_grad()
def apply_corrections(model, results):
    layers = get_llama_layers(model)
    for layer_idx, layer in enumerate(layers):
        linears = find_linear_layers(layer)
        for name, lin in linears.items():
            key = f"layer{layer_idx}.{name}"
            if key not in results: continue
            Q = results[key]["Q"].to(lin.weight.device).to(lin.weight.dtype)
            U = results[key]["U"].to(lin.weight.device).float()
            V = results[key]["V"].to(lin.weight.device).float()
            lin.weight.data = unrotate_weight(Q, U, V)
    print("[absorption] W_corrected = U^T Q(UWV^T) V 적용 완료")


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


def run_llama_rot(
    model_name  = "meta-llama/Llama-2-7b-hf",
    bits        = 4,
    dataset     = "wikitext2",
    nsamples    = 128,
    seqlen      = 2048,
    seed        = 0,
    blocksize   = 128,
    percdamp    = 0.01,
    groupsize   = -1,
    sym         = False,
    actorder    = False,
    rot_mode    = "hadamard",
    dev         = "cuda:0",
    eval_before = True,
):
    print(f"Loading model: {model_name}")
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    print(f"Loading calibration data: {dataset}  nsamples={nsamples}  seqlen={seqlen}")
    trainloader, _ = get_loaders(
        dataset, nsamples=nsamples, seed=seed, seqlen=seqlen, model=model_name
    )
    _, testenc = get_loaders(
        "wikitext2", nsamples=nsamples, seed=seed, seqlen=seqlen, model=model_name
    )

    ppl_fp16 = None
    if eval_before:
        model = model.to(dev)
        ppl_fp16 = eval_ppl(model, testenc, dev, seqlen)
        print(f"\n[FP16 baseline] PPL = {ppl_fp16:.2f}")
        model = model.cpu()

    t0      = time.time()
    results = llama_rot_sequential(
        model, trainloader, dev,
        bits=bits, blocksize=blocksize, percdamp=percdamp,
        groupsize=groupsize, sym=sym, actorder=actorder,
        rot_mode=rot_mode, seed=seed,
    )
    print(f"\n[RotatedGPTQ] Total time: {time.time()-t0:.1f}s")

    apply_corrections(model, results)
    model   = model.to(dev)
    ppl_q   = eval_ppl(model, testenc, dev, seqlen)
    print(f"[{bits}bit LLaMA RotatedGPTQ] PPL = {ppl_q:.2f}")

    return {"ppl_fp16": ppl_fp16, "ppl_quant": ppl_q, "results": results}
