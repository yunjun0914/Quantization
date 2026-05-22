"""
GPTQ from scratch – main entry point
=====================================

사용 예시:

  # OPT-125m, 4-bit, WikiText2 캘리브레이션
  python main.py --model facebook/opt-125m --bits 4 --dataset wikitext2

  # 3-bit + group quantization (group=128)
  python main.py --model facebook/opt-125m --bits 3 --groupsize 128

  # activation order + GPU
  python main.py --model facebook/opt-125m --bits 4 --actorder --dev cuda:0
"""

import argparse
import torch
from opt import run_opt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPTQ from scratch (Frantar et al., ICLR 2023)")

    # ── 모델 / 데이터 ──────────────────────────────────────────────────────
    p.add_argument("--model",    type=str,  default="facebook/opt-125m",
                   help="HuggingFace model name or local path")
    p.add_argument("--dataset",  type=str,  default="wikitext2",
                   choices=["wikitext2", "c4"],
                   help="Calibration dataset (논문: C4 / WikiText2)")
    p.add_argument("--nsamples", type=int,  default=128,
                   help="Number of calibration samples (논문: 128)")
    p.add_argument("--seqlen",   type=int,  default=2048,
                   help="Sequence length for calibration and PPL eval (논문: 2048)")
    p.add_argument("--seed",     type=int,  default=0,
                   help="Random seed for calibration data sampling")

    # ── 양자화 설정 ────────────────────────────────────────────────────────
    p.add_argument("--bits",      type=int,   default=4,
                   help="Quantization bit-width (2 / 3 / 4 / 8)")
    p.add_argument("--blocksize", type=int,   default=128,
                   help="GPTQ lazy batch blocksize (논문: 128)")
    p.add_argument("--percdamp",  type=float, default=0.01,
                   help="Hessian diagonal damping %%  (논문: 1%%)")
    p.add_argument("--groupsize", type=int,   default=-1,
                   help="Weight grouping size (-1: per-channel, 128: typical group)")
    p.add_argument("--sym",       action="store_true",
                   help="Use symmetric quantization")
    p.add_argument("--actorder",  action="store_true",
                   help="Activation-order column sorting (논문 Section 5.1)")

    # ── 실행 환경 ──────────────────────────────────────────────────────────
    p.add_argument("--dev",          type=str,  default="cpu",
                   help="Device (cpu / cuda:0 / etc.)")
    p.add_argument("--no-eval-before", action="store_true",
                   help="Skip FP16 baseline PPL evaluation")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("=" * 60)
    print("GPTQ from scratch  |  Frantar et al., ICLR 2023")
    print("=" * 60)
    print(f"  model     : {args.model}")
    print(f"  bits      : {args.bits}")
    print(f"  dataset   : {args.dataset}")
    print(f"  nsamples  : {args.nsamples}")
    print(f"  seqlen    : {args.seqlen}")
    print(f"  blocksize : {args.blocksize}")
    print(f"  percdamp  : {args.percdamp}")
    print(f"  groupsize : {args.groupsize}")
    print(f"  sym       : {args.sym}")
    print(f"  actorder  : {args.actorder}")
    print(f"  device    : {args.dev}")
    print("=" * 60)

    out = run_opt(
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
        eval_before = not args.no_eval_before,
    )

    print("\n" + "=" * 60)
    if out["ppl_fp16"] is not None:
        print(f"  FP16 baseline PPL : {out['ppl_fp16']:.2f}")
    print(f"  {args.bits}bit GPTQ PPL  : {out['ppl_quant']:.2f}")
    print("=" * 60)
