"""
LLaMA GPTQ / RotatedGPTQ entry point

사용 예시:
  # 기존 GPTQ
  python main_llama.py --model meta-llama/Llama-2-7b-hf --bits 4 --dev cuda:0

  # RotatedGPTQ v2 (globally shared V & U)
  python main_llama.py --model meta-llama/Llama-2-7b-hf --bits 4 --v2 --compare --dev cuda:0

  # 3bit
  python main_llama.py --model meta-llama/Llama-2-7b-hf --bits 3 --v2 --compare --dev cuda:0
"""

import argparse
from llama_rot_v2 import run_llama_rot_v2
from llama import run_llama


def parse_args():
    p = argparse.ArgumentParser(description="LLaMA Rotated GPTQ  |  김윤준")
    p.add_argument("--model",     type=str,   default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--dataset",   type=str,   default="wikitext2", choices=["wikitext2","c4"])
    p.add_argument("--nsamples",  type=int,   default=128)
    p.add_argument("--seqlen",    type=int,   default=2048)
    p.add_argument("--seed",      type=int,   default=0)
    p.add_argument("--bits",      type=int,   default=4)
    p.add_argument("--blocksize", type=int,   default=128)
    p.add_argument("--percdamp",  type=float, default=0.01)
    p.add_argument("--groupsize", type=int,   default=-1)
    p.add_argument("--sym",       action="store_true")
    p.add_argument("--actorder",  action="store_true")
    p.add_argument("--rot",       type=str,   default="hadamard", choices=["random","hadamard"])
    p.add_argument("--v2",        action="store_true", help="RotatedGPTQ v2 (globally shared V & U)")
    p.add_argument("--no_u",      action="store_true", help="ablation: U rotation 제거")
    p.add_argument("--v1_mode",    action="store_true", help="ablation: V1 방식 (U^T 복원 없음, 회전 오차 포함)")
    p.add_argument("--vq2d",       action="store_true", help="2D cross-row vector quantization")
    p.add_argument("--e8",         action="store_true", help="E8 lattice quantization (8D cross-row)")
    p.add_argument("--dev",       type=str,   default="cuda:0")
    p.add_argument("--compare",   action="store_true", help="기존 GPTQ와 PPL 비교")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    ver = "RotatedGPTQ v2" if args.v2 else "GPTQ baseline"
    print(f"LLaMA {ver}  |  김윤준")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k:12s}: {v}")
    print("=" * 60)

    if args.v2:
        out_rot = run_llama_rot_v2(
            model_name  = args.model,
            bits        = args.bits,
            dataset     = args.dataset,
            nsamples    = args.nsamples,
            seqlen      = args.seqlen,
            seed        = args.seed,
            blocksize   = args.blocksize,
            percdamp    = args.percdamp,
            groupsize   = args.groupsize,
            sym         = args.sym,
            actorder    = args.actorder,
            rot_mode    = args.rot,
            dev         = args.dev,
            eval_before = True,
            use_u       = not args.no_u,
            v1_mode     = args.v1_mode,
            use_2d_vq   = args.vq2d,
            use_e8      = args.e8,
        )

        if args.compare:
            print("\n" + "=" * 60)
            print("[비교] 기존 GPTQ")
            out_base = run_llama(
                model_name  = args.model,
                bits        = args.bits,
                dataset     = args.dataset,
                nsamples    = args.nsamples,
                seqlen      = args.seqlen,
                seed        = args.seed,
                blocksize   = args.blocksize,
                percdamp    = args.percdamp,
                groupsize   = args.groupsize,
                sym         = args.sym,
                actorder    = args.actorder,
                dev         = args.dev,
                eval_before = False,
            )

        print("\n" + "=" * 60)
        print(f"  FP16 baseline              : {out_rot['ppl_fp16']:.2f}")
        print(f"  {args.bits}bit RotatedGPTQ v2       : {out_rot['ppl_quant']:.2f}")
        if args.compare:
            print(f"  {args.bits}bit GPTQ (baseline)    : {out_base['ppl_quant']:.2f}")
            print(f"  차이 (rot - base)          : {out_rot['ppl_quant'] - out_base['ppl_quant']:+.2f}")
        print("=" * 60)

    else:
        out = run_llama(
            model_name  = args.model,
            bits        = args.bits,
            dataset     = args.dataset,
            nsamples    = args.nsamples,
            seqlen      = args.seqlen,
            seed        = args.seed,
            blocksize   = args.blocksize,
            percdamp    = args.percdamp,
            groupsize   = args.groupsize,
            sym         = args.sym,
            actorder    = args.actorder,
            dev         = args.dev,
            eval_before = True,
            use_u       = not args.no_u,
            v1_mode     = args.v1_mode,
            use_2d_vq   = args.vq2d,
            use_e8      = args.e8,
        )
        print("\n" + "=" * 60)
        print(f"  FP16 baseline : {out['ppl_fp16']:.2f}")
        print(f"  {args.bits}bit GPTQ      : {out['ppl_quant']:.2f}")
        print("=" * 60)
