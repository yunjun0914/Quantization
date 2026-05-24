"""
OPT Rotated GPTQ 파이프라인
============================

각 decoder layer마다:
  1. V 생성 (d_in × d_in orthogonal)
  2. U 생성 (d_out sign vector, per-linear-layer)
  3. W_rot = U * W * V^T  (rotate_weight)
  4. H̃ = V H V^T 로 RotatedGPTQ 실행
  5. activation 업데이트용으로 weight를 U^T Q(UWV^T) V 로 복원
  6. 최종 inference용으로 layer.weight = Q(UWV^T) 저장
     (U, V는 인접 레이어에 흡수 → 추가 비용 없음)
"""

import time
import torch
import torch.nn as nn
from transformers import OPTForCausalLM

from gptq_rot import RotatedGPTQ
from rotation import get_rotation, get_sign_vector, rotate_weight, unrotate_weight
from svd_correction import apply_svd_corrections
from data import get_loaders


def get_opt_layers(model):
    return model.model.decoder.layers


def find_linear_layers(layer):
    return {
        name: module
        for name, module in layer.named_modules()
        if isinstance(module, nn.Linear)
    }


@torch.no_grad()
def opt_rot_sequential(
    model,
    dataloader,
    dev,
    bits        = 4,
    blocksize   = 128,
    percdamp    = 0.01,
    groupsize   = -1,
    sym         = False,
    actorder    = False,
    rot_mode    = "random",
    seed        = 0,
):
    print(f"[RotatedGPTQ] bits={bits}  blocksize={blocksize}  rot={rot_mode}")
    model.eval()

    layers   = get_opt_layers(model)
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
    cache = {"i": 0}

    class CatchInput(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp[0].detach().cpu()
            cache["i"] += 1
            raise StopIteration

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
    model = model.cpu()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Layer-wise Rotated GPTQ ────────────────────────────────────────────
    results = {}

    for layer_idx, layer in enumerate(layers):
        print(f"\n[Layer {layer_idx:02d}/{len(layers)-1}]")
        t0    = time.time()
        layer = layer.to(device)

        linears = find_linear_layers(layer)

        # ── U, V 생성 ──────────────────────────────────────────────────────
        # V는 같은 입력을 공유하는 layer끼리 공유:
        #   q/k/v_proj  : 모두 self_attn_layer_norm 출력(hidden_size) 공유 → V_attn
        #   out_proj    : attention 출력(hidden_size) → V_out
        #   fc1         : final_layer_norm 출력(hidden_size) → V_ffn
        #   fc2         : fc1 출력(ffn_dim=3072) → V_fc2
        # U는 output channel마다 독립적으로 유지.
        hidden  = model.config.hidden_size
        ffn_dim = model.config.ffn_dim if hasattr(model.config, "ffn_dim") else hidden * 4

        V_attn = get_rotation(hidden,  mode=rot_mode, seed=seed + layer_idx * 10 + 0, device=device)
        V_out  = get_rotation(hidden,  mode=rot_mode, seed=seed + layer_idx * 10 + 1, device=device)
        V_ffn  = get_rotation(hidden,  mode=rot_mode, seed=seed + layer_idx * 10 + 2, device=device)
        V_fc2  = get_rotation(ffn_dim, mode=rot_mode, seed=seed + layer_idx * 10 + 3, device=device)

        def pick_V(name):
            if name in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"):
                return V_attn
            elif name == "self_attn.out_proj":
                return V_out
            elif name == "fc1":
                return V_ffn
            elif name == "fc2":
                return V_fc2
            else:
                # fallback: d_in 기준 새 rotation
                d_col = linears[name].weight.shape[1]
                return get_rotation(d_col, mode=rot_mode, seed=seed + layer_idx + hash(name), device=device)

        rotations = {}
        for name, lin in linears.items():
            V = pick_V(name)
            U = get_sign_vector(lin.weight.shape[0],
                                seed=seed + layer_idx * 10 + hash(name), device=device)
            rotations[name] = (U, V)

        # ── Step 1: 원래 weight로 Hessian 수집 (rotation 전) ─────────────
        # add_batch 내부에서 V@X로 H̃ = VHV^T 직접 계산.
        # weight는 아직 원래 W → forward가 오염되지 않음.
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
            layer(inps[i].unsqueeze(0).to(device))

        for h in hooks:
            h.remove()

        # ── Step 2: V만 적용 (U는 inner loop에서 매 column마다 적용)
        # W_stored = W @ V^T  (V만 흡수, U는 RotatedGPTQ 내부에서 처리)
        for name, lin in linears.items():
            U, V = rotations[name]
            lin.weight.data = (lin.weight.data.float() @ V.t()).to(lin.weight.dtype)

        # ── RotatedGPTQ 실행 ──────────────────────────────────────────────
        Q_store = {}   # name → Q(UWV^T)  (최종 inference용)

        # H 대각 통계 출력 (첫 번째 layer만)
        if layer_idx == 0:
            print(f"  [H diag stats]")
            for name, handler in handlers.items():
                Ht  = handler.H.float()
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
            Q_store[name] = Q   # Q(UWV^T) 저장

            U, V = rotations[name]

            # ── 핵심 수정 ──────────────────────────────────────────────────
            # 다음 layer 입력 업데이트용: U^T Q(UWV^T) V ≈ W  (원래 공간 복원)
            # → forward pass가 올바른 activation을 다음 layer에 전달
            # Q는 이미 W 공간 (U^T @ Q_rot 복원됨)
            # V 흡수: W_final = Q @ V
            linears[name].weight.data = (Q.float() @ rotations[name][1].float()).to(Q.dtype)

            # W_orig: V 흡수 전 원래 weight (SVD correction용)
            W_orig_v = (linears[name].weight.data.float() @ rotations[name][1].float()).cpu().half()

            results[f"layer{layer_idx}.{name}"] = {
                "Q":     Q.cpu(),
                "scale": scale.cpu(),
                "zero":  zero.cpu(),
                "loss":  loss.mean().item(),
                "U":     U.cpu(),
                "V":     V.cpu(),
                "W_orig": W_orig_v,
            }
            print(f"loss={loss.mean().item():.6f}")
            handler.free()

        # ── 다음 layer 입력 업데이트 (복원된 weight로 forward) ───────────
        for i in range(nsamples):
            out     = layer(inps[i].unsqueeze(0).to(device))
            inps[i] = out[0].detach().cpu()

        # ── 최종 inference용: layer.weight = Q(UWV^T) 로 복원 ─────────────
        # (U, V absorption은 별도. 여기서는 Q(UWV^T) 그대로 저장)
        for name, lin in linears.items():
            lin.weight.data = Q_store[name]

        layer = layer.cpu()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"  ↳ done in {time.time()-t0:.1f}s")

    model.config.use_cache = use_cache
    return results


@torch.no_grad()
def eval_ppl(model, testenc, dev, seqlen=2048):
    model.eval()
    device    = torch.device(dev)
    input_ids = (testenc.input_ids if hasattr(testenc, 'input_ids') else testenc).to(device)
    nsamples  = input_ids.shape[1] // seqlen
    total_nll = 0.0
    for i in range(nsamples):
        chunk = input_ids[:, i * seqlen : (i + 1) * seqlen]
        out   = model(chunk, labels=chunk.clone())
        total_nll += out.loss.item() * seqlen
    return torch.exp(torch.tensor(total_nll / (nsamples * seqlen))).item()


def run_opt_rot(
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
    rot_mode    = "random",
    dev         = "cpu",
    eval_before = True,
    svd_rank    = 0,    # 0: SVD correction 없음, >0: rank-r correction
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

    ppl_fp16 = None
    if eval_before:
        model = model.to(dev)
        ppl_fp16 = eval_ppl(model, testenc, dev, seqlen)
        print(f"\n[FP16 baseline] PPL = {ppl_fp16:.2f}")
        model = model.cpu()

    t0      = time.time()
    results = opt_rot_sequential(
        model, trainloader, dev,
        bits=bits, blocksize=blocksize, percdamp=percdamp,
        groupsize=groupsize, sym=sym, actorder=actorder,
        rot_mode=rot_mode, seed=seed,
    )
    print(f"\n[RotatedGPTQ] Total time: {time.time()-t0:.1f}s")

    # PPL 평가: W_corrected = U^T Q(UWV^T) V 적용 후 평가
    apply_corrections(model, results)

    if svd_rank > 0:
        print(f"\n[SVD correction] rank={svd_rank}")
        apply_svd_corrections(
            model, results,
            get_layers_fn=get_opt_layers,
            find_linears_fn=find_linear_layers,
            rank=svd_rank,
            verbose=True,
        )

    model   = model.to(dev)
    ppl_q   = eval_ppl(model, testenc, dev, seqlen)
    print(f"[{bits}bit RotatedGPTQ + SVD(rank={svd_rank})] PPL = {ppl_q:.2f}")

    return {"ppl_fp16": ppl_fp16, "ppl_quant": ppl_q, "results": results}


# ─────────────────────────────────────────────────────────────────────────────
# Absorption
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def apply_corrections(model, results: dict):
    """
    V 공유 기반 absorption:

    q/k/v_proj → V_attn 공유 → self_attn_layer_norm에 흡수 가능
    fc1        → V_ffn  공유 → final_layer_norm에 흡수 가능

    absorption 방식:
      - LayerNorm output y = γ ⊙ norm(x) + β
      - W_q @ (V @ y) = (W_q @ V) @ y  → W_q_eff = Q(UWV^T) @ V ≈ UW
      - U^T 보정 추가:  W_corrected = U^T @ Q(UWV^T) @ V = unrotate(Q, U, V)

    V 공유 덕분에 q/k/v가 동일한 회전 공간 사용 → Hessian 일관성 ✅
    U absorption (출력 방향)은 sign flip이라 다음 layer에 흡수 가능하나
    attention 연산 중간에 있어 현재는 unrotate로 처리.
    """
    layers = get_opt_layers(model)

    for layer_idx, layer in enumerate(layers):
        linears = find_linear_layers(layer)
        for name, lin in linears.items():
            key = f"layer{layer_idx}.{name}"
            if key not in results:
                continue
            Q = results[key]["Q"].to(lin.weight.device).to(lin.weight.dtype)
            U = results[key]["U"].to(lin.weight.device).float()
            V = results[key]["V"].to(lin.weight.device).float()

            # Q는 W 공간, V 흡수
            lin.weight.data = (Q.float() @ V.float()).to(lin.weight.dtype)

    print("[absorption] V absorbed: W_final = Q @ V")
