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
from rotation import get_rotation, get_block_rotation
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
    rot_mode="hadamard", seed=0, use_u=True, v1_mode=False, use_e8=False, uwvt_mode=False, block_u=False,
    export=False,
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

    # ablation: use_u=False이면 모든 U=None (V만 사용)
    rotations = {
        "self_attn.q_proj": (U_qk if use_u else None, V),
        "self_attn.k_proj": (U_qk if use_u else None, V),
        "self_attn.v_proj": (U_v  if use_u else None, V),
        "self_attn.o_proj": (U_qk if use_u else None, V),
        "mlp.gate_proj":    (U_gu if use_u else None, V),
        "mlp.up_proj":      (U_gu if use_u else None, V),
        "mlp.down_proj":    (U_d  if use_u else None, U_gu),
    }
    # CPU 복사본 (export/inference용, GPU 메모리 분리)
    def _cpu(t): return t.cpu() if t is not None else None
    rotations_cpu = {k: (_cpu(u), _cpu(v)) for k, (u, v) in rotations.items()}

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
        # v1_mode=True: V1 방식 (회전 오차 포함 전파)
        # v1_mode=False: our method (순수 양자화 오차만 전파)
        handlers = {
            name: RotatedGPTQ(lin, *rotations[name], restore_u=(not v1_mode), use_e8=use_e8)
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
            if export and use_e8:
                handler.export_mode = True
            Q, scale, zero, loss = handler.quantize(
                bits=bits, blocksize=blocksize, percdamp=percdamp,
                groupsize=groupsize, sym=sym, actorder=actorder,
                uwvt_mode=uwvt_mode,
            )
            Q_dict[name] = Q
            res_entry = {
                "Q": None if export else Q.cpu(),  # export 모드: Q CPU 저장 안 함
                "scale": scale.cpu(), "zero": zero.cpu(),
                "loss": loss.mean().item(),
            }
            # export 모드: E8P index만 저장 (U, V는 rotations에서 직접 전달)
            if export and use_e8 and hasattr(handler, 'e8p_idx'):
                res_entry['e8p_idx']   = handler.e8p_idx.cpu().to(torch.int16)
                res_entry['e8p_scale'] = handler.e8p_scale.cpu()
            results[f"layer{layer_idx}.{name}"] = res_entry
            print(f"loss={loss.mean().item():.6f}")
            handler.free()
        handlers.clear()  # handler GPU tensor 해제

        # ── Step 4: V 흡수 (Q @ V) ───────────────────────────────────────
        # down_proj: R4(U_gu)는 online → Q만 저장 (@ U_gu 안 함)
        # Q @ V: weight 복원 (next layer input 계산을 위해 항상 필요)
        for name, lin in linears.items():
                if name not in Q_dict: continue
                _, Vr = rotations[name]
                if name == "mlp.down_proj":
                    lin.weight.data = Q_dict[name]
                else:
                    lin.weight.data = (Q_dict[name].float() @ Vr.float()).to(dtype)
        # Q_dict 즉시 해제 (idx로 저장 완료, 더 이상 불필요)
        Q_dict.clear()

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
            del U_gu_dev  # GPU tensor 해제

        layer = layer.cpu(); torch.cuda.empty_cache()
        print(f"  ↳ done in {time.time()-t0:.1f}s")

    model.config.use_cache = use_cache
    del inps
    import gc; gc.collect()
    torch.cuda.empty_cache()

    return results, U_gu, V, rotations, rotations_cpu


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
    model_name="meta-llama/Llama-2-7b-hf", bits=4, dataset="c4",
    nsamples=128, seqlen=2048, seed=0, blocksize=128, percdamp=0.01,
    groupsize=-1, sym=False, actorder=False, rot_mode="hadamard",
    dev="cuda:0", eval_before=True, use_u=True, v1_mode=False, use_e8=False, uwvt_mode=False, block_u=False,
    export=False,
):
    print(f"Loading model: {model_name}")
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

    trainloader, _ = get_loaders(dataset, nsamples=nsamples, seed=seed, seqlen=seqlen, model=model_name)
    # GPTQ 논문 방식: wikitext2 + c4 둘 다 평가
    _, testenc_wiki = get_loaders("wikitext2", nsamples=nsamples, seed=seed, seqlen=seqlen, model=model_name)
    _, testenc_c4   = get_loaders("c4",        nsamples=nsamples, seed=seed, seqlen=seqlen, model=model_name)

    ppl_fp16_wiki = None
    ppl_fp16_c4   = None
    if eval_before:
        model = model.to(dev)
        ppl_fp16_wiki = eval_ppl(model, testenc_wiki, dev, seqlen)
        ppl_fp16_c4   = eval_ppl(model, testenc_c4,   dev, seqlen)
        print(f"\n[FP16 baseline] WikiText2={ppl_fp16_wiki:.2f}  C4={ppl_fp16_c4:.2f}")
        model = model.cpu()

    t0      = time.time()
    results, U_gu, V, rotations, rotations_cpu = llama_rot_sequential_v2(
        model, trainloader, dev,
        bits=bits, blocksize=blocksize, percdamp=percdamp,
        groupsize=groupsize, sym=sym, actorder=actorder,
        rot_mode=rot_mode, seed=seed, use_u=use_u, v1_mode=v1_mode, use_e8=use_e8, uwvt_mode=uwvt_mode,
        export=export,
    )
    print(f"\n[RotatedGPTQ v2] Total time: {time.time()-t0:.1f}s")

    # R4 online: 모든 layer down_proj 앞에 U_gu hook 등록
    # export 모드에서도 동일하게 적용 (E8PLinear의 V가 down_proj input을 처리)
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
    if not export:
        ppl_q_wiki = eval_ppl(model, testenc_wiki, dev, seqlen)
        ppl_q_c4   = eval_ppl(model, testenc_c4,   dev, seqlen)
        ppl_q = ppl_q_wiki
        print(f"[{bits}bit RotatedGPTQ v2] WikiText2={ppl_q_wiki:.2f}  C4={ppl_q_c4:.2f}")
    else:
        ppl_q = ppl_q_wiki = ppl_q_c4 = None
        print(f"[Export mode] fake quant PPL 생략")

    for h in r4_hooks:
        h.remove()

    # export 모드: E8P handler에서 index 수집 → real quant PPL
    ppl_real = None
    if export and use_e8:
        print("\n[Real Quant] E8P index 기반 PPL 측정 중...")
        from llama_e8p_inference import E8PLinear, build_e8p_model

        # quantized_layers 추출
        quantized_layers = {}
        for layer_name, res in results.items():
            if 'e8p_idx' in res:
                sub_name = '.'.join(layer_name.split('.')[1:])
                Ur, Vr = rotations_cpu.get(sub_name, (None, None))
                quantized_layers[layer_name] = {
                    'idx':   res['e8p_idx'],
                    'scale': res['e8p_scale'],
                    'U':     Ur,
                    'V':     Vr,
                }
        del results
        import gc; gc.collect()

        # fake quant 모델을 그대로 E8PLinear로 교체 (새로 로드 안 함)
        model = model.cpu()
        model_real = model
        if quantized_layers:
            # rotations: layer별 U 정보 전달
            V_cpu = rotations_cpu['self_attn.q_proj'][1]  # globally shared V cpu
            model_real, r4_hooks_real = build_e8p_model(
                model_real, quantized_layers, V_cpu, rotations=rotations_cpu
            )
            model_real = model_real.to(dev)
            # R4 hook: down_proj input에 U_gu 적용 (fake quant와 동일)
            U_gu_cpu = U_gu.cpu()
            for layer in get_llama_layers(model_real):
                linears = find_linear_layers(layer)
                if "mlp.down_proj" in linears:
                    def make_r4_hook(U_g):
                        def hook(m, inp):
                            x  = inp[0].float()
                            sh = x.shape
                            x  = (U_g.to(x.device).float() @ x.reshape(-1, sh[-1]).t()).t()
                            return (x.reshape(sh).to(inp[0].dtype),)
                        return hook
                    r4_hooks_real.append(
                        linears["mlp.down_proj"].register_forward_pre_hook(
                            make_r4_hook(U_gu_cpu))
                    )
            ppl_real_wiki = eval_ppl(model_real, testenc_wiki, dev, seqlen)
            ppl_real_c4   = eval_ppl(model_real, testenc_c4,   dev, seqlen)
            ppl_real = ppl_real_wiki
            ppl_real_c4_val = ppl_real_c4
            print(f"[Real Quant E8P] WikiText2={ppl_real_wiki:.2f}  C4={ppl_real_c4:.2f}")
            for h in r4_hooks_real: h.remove()

            # 모델 저장
            if export:
                save_path = "e8p_2bit.pt"
                print(f"[Real Quant] 모델 저장 중... {save_path}")
                torch.save({
                    'quantized_layers': {
                        k: {
                            'idx':   v['idx'].to(torch.int16),  # int32 → int16 (2bpw)
                            'scale': v['scale'],
                        } for k, v in quantized_layers.items()
                    },
                    'config': {
                        'model_name': model_name,
                        'seed':       seed,
                        'rot_mode':   rot_mode,
                        'bits':       bits,
                        'ppl':        ppl_real,
                        'bpw':        2.0,
                    },
                    # rotations은 저장 안 함 → inference 시 seed로 재생성
                }, save_path)
                save_size = sum(
                    v['idx'].numel() * 2  # int16 = 2bytes
                    for v in quantized_layers.values()
                ) / 1e9
                total_weights = sum(v['idx'].numel() * 8 for v in quantized_layers.values())
                bpw = save_size * 8e9 / total_weights
                print(f"[Real Quant] 저장 완료: {save_path}")
                print(f"[Real Quant] idx (int16): {save_size:.2f}GB = {bpw:.2f}bpw")

            del model_real
            torch.cuda.empty_cache()

    return {
        "ppl_fp16_wiki":  ppl_fp16_wiki,
        "ppl_fp16_c4":    ppl_fp16_c4,
        "ppl_quant_wiki": ppl_q_wiki if not export else None,
        "ppl_quant_c4":   ppl_q_c4  if not export else None,
        "ppl_real_wiki":  ppl_real if export else None,
        "ppl_real_c4":    ppl_real_c4_val if export else None,
        "results":        results,
    }
