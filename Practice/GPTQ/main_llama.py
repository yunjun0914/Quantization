"""
LLaMA GPTQ / RotatedGPTQ entry point

사용 예시:
  # v1 (unrotate 방식)
  python main_llama.py --model meta-llama/Llama-2-7b-hf --bits 4 --rot hadamard --compare --dev cuda:0

  # v2 (proper absorption)
  python main_llama.py --model meta-llama/Llama-2-7b-hf --bits 4 --rot hadamard --v2 --compare --dev cuda:0
"""

import argparse, torch
from llama_rot import run_llama_rot
from llama_rot_v2 import run_llama_rot_v2
from llama import run_llama


def parse_args():
    p = argparse.ArgumentParser(description="LLaMA Rotated GPTQ")
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
    p.add_argument("--v2",        action="store_true", help="Proper absorption (v2)")
    p.add_argument("--svd_rank",  type=int, default=0, help="SVD residual correction rank")
    p.add_argument("--dev",       type=str,   default="cuda:0")
    p.add_argument("--compare",   action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print(f"LLaMA Rotated GPTQ {'v2 (proper absorption)' if args.v2 else 'v1'}  |  김윤준")
    print("=" * 60)
    for k, v in vars(args).items():
        print(f"  {k:12s}: {v}")
    print("=" * 60)

    runner = run_llama_rot_v2 if args.v2 else run_llama_rot
    out_rot = runner(
        model_name=args.model, bits=args.bits, dataset=args.dataset,
        nsamples=args.nsamples, seqlen=args.seqlen, seed=args.seed,
        blocksize=args.blocksize, percdamp=args.percdamp, groupsize=args.groupsize,
        sym=args.sym, actorder=args.actorder, rot_mode=args.rot,
        dev=args.dev, eval_before=True, svd_rank=args.svd_rank,
    )

    if args.compare:
        print("\n" + "=" * 60)
        print("[비교] 기존 GPTQ")
        out_base = run_llama(
            model_name=args.model, bits=args.bits, dataset=args.dataset,
            nsamples=args.nsamples, seqlen=args.seqlen, seed=args.seed,
            blocksize=args.blocksize, percdamp=args.percdamp, groupsize=args.groupsize,
            sym=args.sym, actorder=args.actorder, dev=args.dev, eval_before=False,
        )

    tag = "v2" if args.v2 else args.rot
    print("\n" + "=" * 60)
    print(f"  FP16 baseline              : {out_rot['ppl_fp16']:.2f}")
    print(f"  {args.bits}bit RotatedGPTQ ({tag:8s}): {out_rot['ppl_quant']:.2f}")
    if args.compare:
        print(f"  {args.bits}bit GPTQ (baseline)    : {out_base['ppl_quant']:.2f}")
        print(f"  차이 (rot - base)          : {out_rot['ppl_quant'] - out_base['ppl_quant']:+.2f}")
    print("=" * 60)
