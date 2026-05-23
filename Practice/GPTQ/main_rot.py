"""
Rotated GPTQ – main entry point

사용 예시:
  python main_rot.py --model facebook/opt-125m --bits 4 --rot random
  python main_rot.py --model facebook/opt-125m --bits 4 --rot hadamard --dev cuda:0
"""

import argparse
import torch
from opt_rot import run_opt_rot
from opt import run_opt   # 기존 GPTQ (비교용)


def parse_args():
    p = argparse.ArgumentParser(description="Rotated GPTQ")
    p.add_argument("--model",     type=str,   default="facebook/opt-125m")
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
    p.add_argument("--rot",       type=str,   default="random", choices=["random","hadamard"])
    p.add_argument("--dev",       type=str,   default="cpu")
    p.add_argument("--compare",   action="store_true", help="기존 GPTQ와 PPL 비교")
    p.add_argument("--svd_rank",  type=int, default=0, help="SVD residual correction rank")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("Rotated GPTQ  |  김윤준 아이디어")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k:12s}: {v}")
    print("=" * 60)

    # ── Rotated GPTQ ──────────────────────────────────────────────────────
    out_rot = run_opt_rot(
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
        svd_rank    = args.svd_rank,
    )

    # ── 기존 GPTQ 비교 (선택) ────────────────────────────────────────────
    if args.compare:
        print("\n" + "=" * 60)
        print("[비교] 기존 GPTQ 실행")
        out_base = run_opt(
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
    print(f"  FP16 baseline         : {out_rot['ppl_fp16']:.2f}")
    print(f"  {args.bits}bit RotatedGPTQ ({args.rot}): {out_rot['ppl_quant']:.2f}")
    if args.compare:
        print(f"  {args.bits}bit GPTQ (baseline) : {out_base['ppl_quant']:.2f}")
        diff = out_rot['ppl_quant'] - out_base['ppl_quant']
        print(f"  차이 (rot - base)     : {diff:+.2f}")
    print("=" * 60)
