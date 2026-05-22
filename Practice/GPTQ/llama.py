"""
LLaMA GPTQ baseline 파이프라인 (비교용)
"""

import time
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM

from gptq import GPTQ
from data import get_loaders


def get_llama_layers(model):
    return model.model.layers

def find_linear_layers(layer):
    return {name: m for name, m in layer.named_modules() if isinstance(m, nn.Linear)}


@torch.no_grad()
def llama_sequential(model, dataloader, dev, bits=4, blocksize=128,
                     percdamp=0.01, groupsize=-1, sym=False, actorder=False):
    print(f"[GPTQ] bits={bits}  blocksize={blocksize}  groupsize={groupsize}")
    model.eval()
    layers  = get_llama_layers(model)
    dtype   = next(iter(model.parameters())).dtype
    device  = torch.device(dev)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    nsamples = len(dataloader)
    seqlen   = dataloader[0].shape[1]
    inps     = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device="cpu")
    cache    = {"i": 0, "position_ids": None}

    class Catch(nn.Module):
        def __init__(self, m): super().__init__(); self.module = m
        def forward(self, inp, **kw):
            inps[cache["i"]] = inp[0].detach().cpu()
            cache["position_ids"] = kw.get("position_ids", None)
            cache["i"] += 1; raise StopIteration

    model = model.to(device)
    layers[0] = Catch(layers[0]); model.model.layers[0] = layers[0]
    for b in dataloader:
        try: model(b.to(device))
        except StopIteration: pass
    layers[0] = layers[0].module; model.model.layers[0] = layers[0]
    model = model.cpu(); torch.cuda.empty_cache()

    results = {}
    for layer_idx, layer in enumerate(layers):
        print(f"\n[Layer {layer_idx:02d}/{len(layers)-1}]")
        t0 = time.time(); layer = layer.to(device)
        linears = find_linear_layers(layer)
        handlers = {name: GPTQ(lin) for name, lin in linears.items()}

        hooks = []
        for name, lin in linears.items():
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

        for name, handler in handlers.items():
            print(f"  quantizing {name:30s} ... ", end="", flush=True)
            Q, scale, zero, loss = handler.quantize(
                bits=bits, blocksize=blocksize, percdamp=percdamp,
                groupsize=groupsize, sym=sym, actorder=actorder)
            linears[name].weight.data = Q
            results[f"layer{layer_idx}.{name}"] = {"loss": loss.mean().item()}
            print(f"loss={loss.mean().item():.6f}")
            handler.free()

        for i in range(nsamples):
            kw = {}
            if cache["position_ids"] is not None:
                kw["position_ids"] = cache["position_ids"].to(device)
            out = layer(inps[i].unsqueeze(0).to(device), **kw)
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


def run_llama(model_name="meta-llama/Llama-2-7b-hf", bits=4, dataset="wikitext2",
              nsamples=128, seqlen=2048, seed=0, blocksize=128, percdamp=0.01,
              groupsize=-1, sym=False, actorder=False, dev="cuda:0", eval_before=True):
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
    results = llama_sequential(model, trainloader, dev, bits=bits, blocksize=blocksize,
                                percdamp=percdamp, groupsize=groupsize, sym=sym, actorder=actorder)
    print(f"\n[GPTQ] Total time: {time.time()-t0:.1f}s")

    model   = model.to(dev)
    ppl_q   = eval_ppl(model, testenc, dev, seqlen)
    print(f"[{bits}bit GPTQ] PPL = {ppl_q:.2f}")

    return {"ppl_fp16": ppl_fp16, "ppl_quant": ppl_q, "results": results}
