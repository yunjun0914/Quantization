"""
LLaMA Rotated GPTQ v2
======================
QuaRot 구조 기반 (R1=V, R2=U_v, R3=U_qk, R4=U_gu, R5=U_d)

weight 흡수:
  q_proj:    U_qk @ W @ V^T
  k_proj:    U_qk @ W @ V^T
  v_proj:    U_v  @ W @ V^T
  o_proj:    U_qk @ W @ V^T  → U_v^T absorbed into columns
  gate_proj: U_gu @ W @ V^T
  up_proj:   U_gu @ W @ V^T
  down_proj: U_d  @ W @ U_gu^T → U_gu^T absorbed into columns, U_d^T into next layernorm

복원:
  U_qk: QK^T에서 자동 소거
  U_v:  o_proj columns에 U_v^T 흡수
  U_gu: down_proj columns에 U_gu^T 흡수
  U_d:  다음 layer input_layernorm weight에 U_d^T 흡수
  V:    chain으로 V V^T = I 소거

R1(V) absorbable → hidden state 항상 원래 공간 → residual 문제없음
"""

import time
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM

from gptq_rot import RotatedGPTQ
from rotation import get_rotation
from data import get_loaders


def get_llama_layers(model):
    return model.model.layers


@torch.no_grad()
def absorb_rmsnorm(model, dtype):
    """
    RMSNorm α를 인접 input-side weight에 흡수 (QuaRot 논문 Stage 1a).

    LLaMA 구조:
      input_layernorm(α1) → q/k/v_proj (input-side)
      post_attention_layernorm(α2) → gate/up_proj (input-side)
      o_proj, down_proj는 output-side → 흡수 안 함

    W_new = W @ diag(α)  (column-wise 곱)
    RMSNorm weight → 1로 초기화 (순수 normalize만 남김)
    """
    layers = get_llama_layers(model)
    for layer in layers:
        linears = find_linear_layers(layer)

        # input_layernorm α1 → q/k/v_proj columns에만 흡수
        if hasattr(layer, 'input_layernorm'):
            alpha = layer.input_layernorm.weight.data.float()
            for name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
                if name in linears:
                    linears[name].weight.data = (
                        linears[name].weight.data.float() * alpha.unsqueeze(0)
                    ).to(dtype)
            layer.input_layernorm.weight.data = torch.ones_like(
                layer.input_layernorm.weight.data)

        # post_attention_layernorm α2 → gate/up_proj columns에만 흡수
        if hasattr(layer, 'post_attention_layernorm'):
            alpha = layer.post_attention_layernorm.weight.data.float()
            for name in ("mlp.gate_proj", "mlp.up_proj"):
                if name in linears:
                    linears[name].weight.data = (
                        linears[name].weight.data.float() * alpha.unsqueeze(0)
                    ).to(dtype)
            layer.post_attention_layernorm.weight.data = torch.ones_like(
                layer.post_attention_layernorm.weight.data)

    print("[absorb_rmsnorm] RMSNorm α absorbed into q/k/v and gate/up weights")


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
    U_gu = get_rotation(inter,  mode=rot_mode, seed=seed+3, device=device)  # R4
    U_d  = get_rotation(hidden, mode=rot_mode, seed=seed+4, device=device)  # R5

    print(f"  V({hidden},{hidden})  U_qk({hidden},{hidden})  U_v({hidden},{hidden})")
    print(f"  U_gu({inter},{inter})  U_d({hidden},{hidden})")

    rotations = {
        "self_attn.q_proj": (U_qk, V),
        "self_attn.k_proj": (U_qk, V),
        "self_attn.v_proj": (U_v,  V),
        "self_attn.o_proj": (U_qk, V),
        "mlp.gate_proj":    (U_gu, V),
        "mlp.up_proj":      (U_gu, V),
        "mlp.down_proj":    (U_d,  U_gu),
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
                d  = Ht.diag()
                print(f"    {name:25s}  H̃ std={d.std():.4f}  max={d.max():.4f}  min={d.min():.6f}")

        # ── Step 2: W @ V^T ───────────────────────────────────────────────
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
        # down_proj: R4(U_gu)는 online → Q만 저장 (@ U_gu 안 함)
        # 나머지: Q @ V 저장
        for name, lin in linears.items():
            if name not in Q_dict: continue
            _, Vr = rotations[name]
            if name == "mlp.down_proj":
                lin.weight.data = Q_dict[name]  # Q(W_down @ U_gu^T) 그대로
            else:
                lin.weight.data = (Q_dict[name].float() @ Vr.float()).to(dtype)

        # Step 5 불필요: inner loop q = U^T @ Q(U @ WV^T) 이미 복원
        # Step 4: Q @ V = U^T @ Q(U @ WV^T) @ V → inference x만 넣으면 ≈ Wx

        # ── 다음 layer 입력 업데이트 ──────────────────────────────────────
        # R4(U_gu) online: down_proj 앞에 U_gu를 fp32로 적용
        # hook으로 down_proj input에 U_gu 적용
        U_gu_dev = U_gu.to(device)
        def make_r4_hook(U):
            def hook(m, inp):
                x = inp[0].float()
                # x: (batch, seq, inter) or (batch, inter)
                if x.dim() == 3:
                    x = (U @ x.reshape(-1, x.shape[-1]).t()).t().reshape(*x.shape)
                else:
                    x = (U @ x.t()).t()
                return (x.to(inp[0].dtype),)
            return hook

        if "mlp.down_proj" in linears:
            r4_hook = linears["mlp.down_proj"].register_forward_pre_hook(make_r4_hook(U_gu_dev))

        for i in range(nsamples):
            kw = {}
            if cache["position_ids"] is not None:
                kw["position_ids"] = cache["position_ids"].to(device)
            out   = layer(inps[i].unsqueeze(0).to(device), **kw)
            inps[i] = out[0].detach().cpu()

        if "mlp.down_proj" in linears:
            r4_hook.remove()

        layer = layer.cpu(); torch.cuda.empty_cache()
        print(f"  ↳ done in {time.time()-t0:.1f}s")

    model.config.use_cache = use_cache
    return results, U_gu


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

    dtype = next(iter(model.parameters())).dtype

    # FP16 baseline: absorb 이전에 평가 (원래 모델 기준)
    ppl_fp16 = None
    if eval_before:
        model = model.to(dev)
        ppl_fp16 = eval_ppl(model, testenc, dev, seqlen)
        print(f"\n[FP16 baseline] PPL = {ppl_fp16:.2f}")
        model = model.cpu()

    # RMSNorm α 흡수 (FP16 eval 이후, rotation 전에 수행)
    absorb_rmsnorm(model, dtype)

    t0      = time.time()
    results, U_gu = llama_rot_sequential_v2(
        model, trainloader, dev,
        bits=bits, blocksize=blocksize, percdamp=percdamp,
        groupsize=groupsize, sym=sym, actorder=actorder,
        rot_mode=rot_mode, seed=seed,
    )
    print(f"\n[RotatedGPTQ v2] Total time: {time.time()-t0:.1f}s")

    # R4 online: 모든 layer down_proj 앞에 U_gu hook 등록
    U_gu_cpu = U_gu.cpu()
    r4_hooks = []
    for layer in get_llama_layers(model):
        linears = find_linear_layers(layer)
        if "mlp.down_proj" in linears:
            def make_r4_hook(U):
                def hook(m, inp):
                    x = inp[0].float()
                    if x.dim() == 3:
                        x = (U.to(x.device) @ x.reshape(-1, x.shape[-1]).t()).t().reshape(*x.shape)
                    else:
                        x = (U.to(x.device) @ x.t()).t()
                    return (x.to(inp[0].dtype),)
                return hook
            r4_hooks.append(
                linears["mlp.down_proj"].register_forward_pre_hook(make_r4_hook(U_gu_cpu))
            )

    model   = model.to(dev)
    ppl_q   = eval_ppl(model, testenc, dev, seqlen)
    print(f"[{bits}bit RotatedGPTQ v2] PPL = {ppl_q:.2f}")

    for h in r4_hooks:
        h.remove()

    return {"ppl_fp16": ppl_fp16, "ppl_quant": ppl_q, "results": results}
