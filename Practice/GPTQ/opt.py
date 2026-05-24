"""
OPT 모델 GPTQ 양자화 파이프라인
================================
논문 Table 3: OPT-125m ~ OPT-175B 결과 재현 대상.
  - OPT-125m  4bit: PPL ≈ 31.12  (WikiText2)

구현:
  1. Sequential layer-wise quantization
  2. 각 layer forward hook으로 Hessian 수집
  3. GPTQ 실행 후 layer weight 교체
  4. PPL (perplexity) 평가
"""

import time
import torch
import torch.nn as nn
from transformers import OPTForCausalLM, AutoTokenizer

from gptq import GPTQ
from data import get_loaders


# ─────────────────────────────────────────────────────────────────────────────
# OPT decoder layer 내의 Linear 레이어 목록
# ─────────────────────────────────────────────────────────────────────────────

def get_opt_layers(model: OPTForCausalLM) -> list:
    """각 OPT decoder layer (OPTDecoderLayer) 반환."""
    return model.model.decoder.layers


def find_linear_layers(layer: nn.Module) -> dict:
    """
    하나의 decoder layer 내 모든 Linear 레이어를 {name: module} 로 반환.
    OPT decoder layer 내부:
      self_attn.q_proj, k_proj, v_proj, out_proj
      fc1, fc2
    """
    layers = {}
    for name, module in layer.named_modules():
        if isinstance(module, nn.Linear):
            layers[name] = module
    return layers


# ─────────────────────────────────────────────────────────────────────────────
# Calibration forward pass: 모든 decoder layer에 대해 순차 실행
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def opt_sequential(
    model,
    dataloader,
    dev,
    bits        = 4,
    blocksize   = 128,
    percdamp    = 0.01,
    groupsize   = -1,
    sym         = False,
    actorder    = False,
):
    """
    논문 Algorithm: 레이어별 순차 양자화.

    핵심 아이디어:
      - decoder.layers[0] 앞에 CatchInput wrapper를 끼워 layer 0 입력(= embed 출력)을 저장
      - 이후 각 layer를 device로 이동 → GPTQ → 다음 layer 입력 업데이트 → CPU로 이동
      - 전체 모델을 GPU에 올리지 않아도 되어 메모리 효율적

    Returns: {layer_name: {"Q", "scale", "zero", "loss"}}
    """
    print(f"[GPTQ] Quantizing OPT  bits={bits}  blocksize={blocksize}  groupsize={groupsize}")
    model.eval()

    layers  = get_opt_layers(model)
    dtype   = next(iter(model.parameters())).dtype
    device  = torch.device(dev)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    # ── 입력 캡처 버퍼 ───────────────────────────────────────────────────────
    nsamples = len(dataloader)
    seqlen   = dataloader[0].shape[1]
    inps     = torch.zeros(
        (nsamples, seqlen, model.config.hidden_size),
        dtype=dtype, device="cpu",
    )
    # attention_mask: OPT는 causal mask를 내부에서 생성하므로 None 전달 가능.
    # 여기서는 padding mask(= 1D input attention_mask)를 None으로 처리.
    extra_kwargs = {}

    # ── CatchInput: layer 0 직전에 끼워 embed 출력 캡처 ──────────────────────
    # StopIteration을 throw해서 decoder 진입 즉시 중단.
    cache = {"i": 0}

    class CatchInput(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            # inp : (batch=1, seqlen, hidden)  hidden on device
            inps[cache["i"]] = inp[0].detach().cpu()
            # position_ids, attention_mask 등 저장
            extra_kwargs.update(
                {k: v for k, v in kwargs.items() if v is not None}
            )
            cache["i"] += 1
            raise StopIteration

    # ── Step 1: embedding 부분만 device에 올려서 입력 캡처 ───────────────────
    # model 전체를 GPU에 올리는 대신 embed 관련 서브모듈만 이동.
    embed_module = model.model.decoder   # embed_tokens, embed_positions, final_layer_norm 포함

    # decoder.layers는 잠시 제거하고 embed 부분만 device로
    # → 가장 간단한 방법: model 전체를 device로 이동 후 캡처, 그 후 다시 cpu
    #   (OPT-125m 정도는 메모리 부담 없음; 대형 모델은 offload 필요)
    model = model.to(device)

    layers[0] = CatchInput(layers[0])
    model.model.decoder.layers[0] = layers[0]

    for batch in dataloader:
        try:
            model(batch.to(device))
        except StopIteration:
            pass

    layers[0] = layers[0].module
    model.model.decoder.layers[0] = layers[0]

    # 캡처 완료 → 모델 다시 CPU (layer-wise로 device 이동할 것)
    model = model.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Step 2: Layer-wise GPTQ ───────────────────────────────────────────────
    results = {}

    for layer_idx, layer in enumerate(layers):
        print(f"\n[Layer {layer_idx:02d}/{len(layers)-1}]")
        t0    = time.time()
        layer = layer.to(device)

        linears = find_linear_layers(layer)

        # GPTQ 핸들러 생성
        gptq_handlers = {name: GPTQ(lin) for name, lin in linears.items()}

        # forward hook: Hessian 누적
        hooks = []
        for name, lin in linears.items():
            def make_hook(handler):
                def hook(module, inp, out):
                    handler.add_batch(inp[0].data, out.data)
                return hook
            hooks.append(lin.register_forward_hook(make_hook(gptq_handlers[name])))

        # 저장된 layer 입력으로 Hessian 누적
        for i in range(nsamples):
            inp_i = inps[i].unsqueeze(0).to(device)   # (1, seqlen, hidden)
            # extra_kwargs에 position_ids, attention_mask 등이 있을 수 있으나,
            # OPT는 position_ids를 자동 계산하고 causal mask도 내부 생성 → 빈 dict OK
            layer(inp_i)

        for h in hooks:
            h.remove()

        # ── GPTQ 실행 및 weight 교체 ─────────────────────────────────────────
        for name, handler in gptq_handlers.items():
            print(f"  quantizing {name:30s} ... ", end="", flush=True)
            Q, scale, zero, loss = handler.quantize(
                bits       = bits,
                blocksize  = blocksize,
                percdamp   = percdamp,
                groupsize  = groupsize,
                sym        = sym,
                actorder   = actorder,
            )
            linears[name].weight.data = Q
            results[f"layer{layer_idx}.{name}"] = {
                "Q":     Q.cpu(),
                "scale": scale.cpu(),
                "zero":  zero.cpu(),
                "loss":  loss.mean().item(),
            }
            print(f"loss={loss.mean().item():.6f}")
            handler.free()

        # 다음 layer 입력 업데이트 (양자화된 layer로 forward)
        for i in range(nsamples):
            inp_i = inps[i].unsqueeze(0).to(device)
            out   = layer(inp_i)
            inps[i] = out[0].detach().cpu()

        layer = layer.cpu()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"  ↳ done in {time.time()-t0:.1f}s")

    model.config.use_cache = use_cache
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity 평가
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_ppl(model, testenc, dev, seqlen=2048):
    """
    WikiText-2 / C4 perplexity 평가.
    논문 Table 2/3: seqlen=2048, non-overlapping window.
    """
    model.eval()
    device    = torch.device(dev)
    input_ids = (testenc.input_ids if hasattr(testenc, 'input_ids') else testenc).to(device)
    nsamples  = input_ids.shape[1] // seqlen

    total_nll = 0.0
    for i in range(nsamples):
        chunk  = input_ids[:, i * seqlen : (i + 1) * seqlen]
        out    = model(chunk, labels=chunk.clone())
        total_nll += out.loss.item() * seqlen

    ppl = torch.exp(torch.tensor(total_nll / (nsamples * seqlen))).item()
    return ppl


# ─────────────────────────────────────────────────────────────────────────────
# 메인 실행 함수
# ─────────────────────────────────────────────────────────────────────────────

def run_opt(
    model_name  = "facebook/opt-125m",
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
    dev         = "cpu",
    eval_before = True,
):
    print(f"Loading model: {model_name}")
    model = OPTForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    print(f"Loading calibration data: {dataset}  nsamples={nsamples}  seqlen={seqlen}")
    trainloader, _ = get_loaders(
        dataset, nsamples=nsamples, seed=seed, seqlen=seqlen, model=model_name
    )
    # 평가는 항상 WikiText2 고정 (논문 설정)
    _, testenc = get_loaders(
        "wikitext2", nsamples=nsamples, seed=seed, seqlen=seqlen, model=model_name
    )

    # FP16 baseline PPL
    ppl_fp16 = None
    if eval_before:
        model = model.to(dev)
        ppl_fp16 = eval_ppl(model, testenc, dev, seqlen)
        print(f"\n[FP16 baseline] PPL = {ppl_fp16:.2f}")
        model = model.cpu()

    # GPTQ 양자화
    t0      = time.time()
    results = opt_sequential(
        model, trainloader, dev,
        bits=bits, blocksize=blocksize, percdamp=percdamp,
        groupsize=groupsize, sym=sym, actorder=actorder,
    )
    print(f"\n[GPTQ] Total time: {time.time()-t0:.1f}s")

    # 양자화 후 PPL
    model   = model.to(dev)
    ppl_q   = eval_ppl(model, testenc, dev, seqlen)
    print(f"[{bits}bit GPTQ] PPL = {ppl_q:.2f}")

    return {"ppl_fp16": ppl_fp16, "ppl_quant": ppl_q, "results": results}
