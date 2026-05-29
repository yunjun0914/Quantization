"""
Calibration dataset loading.

논문: nsamples=128, seqlen=2048 캘리브레이션 데이터 (C4 / WikiText2).
"""

import random
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def get_wikitext2(
    nsamples: int,
    seed:     int,
    seqlen:   int,
    tokenizer_path: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    WikiText-2 calibration / test data 로딩.

    Returns:
        trainloader : list of (1, seqlen) token id tensors  (캘리브레이션용)
        testenc     : (1, total_tokens) token id tensor      (PPL 평가용)
    """
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata  = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc  = tokenizer("\n\n".join(testdata["text"]),  return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        start = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        chunk = trainenc.input_ids[:, start : start + seqlen]
        trainloader.append(chunk)

    return trainloader, testenc


def get_c4(
    nsamples: int,
    seed:     int,
    seqlen:   int,
    tokenizer_path: str,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    C4 calibration / validation data 로딩.
    논문에서 OPT 계열은 C4 캘리브레이션 사용.
    """
    # datasets 버전에 따라 config name이 다를 수 있어 fallback 처리
    try:
        traindata = load_dataset(
            "allenai/c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        valdata = load_dataset(
            "allenai/c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )
    except Exception:
        traindata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
            split="train",
        )
        valdata = load_dataset(
            "allenai/c4",
            "allenai--c4",
            data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
            split="validation",
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            idx    = random.randint(0, len(traindata) - 1)
            tokens = tokenizer(traindata[idx]["text"], return_tensors="pt")
            if tokens.input_ids.shape[1] >= seqlen:
                break
        start = random.randint(0, tokens.input_ids.shape[1] - seqlen - 1)
        trainloader.append(tokens.input_ids[:, start : start + seqlen])

    # validation: GPTQ 논문 방식
    # 문서를 이어붙여 연속 텍스트로 만든 후 stride PPL 측정
    valenc = tokenizer(
        " ".join(valdata[i]["text"] for i in range(1100)),
        return_tensors="pt"
    )

    return trainloader, valenc


def get_loaders(
    dataset:        str,
    nsamples:       int   = 128,
    seed:           int   = 0,
    seqlen:         int   = 2048,
    model:          str   = "facebook/opt-125m",
) -> tuple:
    """
    dataset 이름으로 적절한 loader 선택.
    """
    if dataset == "wikitext2":
        return get_wikitext2(nsamples, seed, seqlen, model)
    elif dataset == "c4":
        return get_c4(nsamples, seed, seqlen, model)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Choose 'wikitext2' or 'c4'.")
