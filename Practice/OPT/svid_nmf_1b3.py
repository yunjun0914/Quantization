import torch
import torch.nn as nn
import math
import random
import time
import sys
from tqdm import tqdm


# ============================================================
# Config
# ============================================================
MODEL_NAME    = "facebook/opt-1.3b"
NSAMPLES      = 128
CALIB_SEQ_LEN = 2048
SEED          = 0
NMF_STEPS     = 2000
ALPHAS        = [0.5]   # sweet spot 고정

# (r_nmf, r_corr, corr_int4, nmf_int4)
# 1.3B: dim=2048라 125m(768) 대비 rank 2~3배 필요
# r_nmf/dim 비율을 125m과 비슷하게 맞춤
CONFIGS = [
    (32, 256, True, False),   # bpw_attn≈1.50
    (32, 384, True, False),   # bpw_attn≈1.75
    (64, 256, True, False),   # bpw_attn≈1.75
    (64, 384, True, False),   # bpw_attn≈2.00
    (64, 512, True, False),   # bpw_attn≈2.25
]


# ============================================================
# Utils
# ============================================================
def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_linear_modules(model):
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and "lm_head" not in name
    }


def restore_weights(model, W_dict):
    for name, module in get_linear_modules(model).items():
        if name in W_dict:
            module.weight.data.copy_(
                W_dict[name].to(module.weight.device, module.weight.dtype)
            )


def calc_bpw(shape, r_nmf, r_corr, corr_int4=False, nmf_int4=False):
    m, n = shape
    bits_sign  = m * n * 1
    bits_alpha = n * 16 if r_nmf == 0 else 0   # mean alpha per-column
    bits_nmf   = (m + n) * r_nmf * (4 if nmf_int4 else 16)
    bits_corr  = (m + n) * r_corr * (4 if corr_int4 else 16)
    return (bits_sign + bits_alpha + bits_nmf + bits_corr) / (m * n)


# ============================================================
# Data
# ============================================================
def make_calib_batches(tokenizer, nsamples=128, seq_len=2048, seed=0, device="cuda"):
    from datasets import load_dataset
    raw       = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    full_text = "\n\n".join([t for t in raw["text"] if t.strip()])
    enc       = tokenizer(full_text, return_tensors="pt").input_ids[0]
    gen       = torch.Generator()
    gen.manual_seed(seed)
    max_start = enc.numel() - seq_len - 1
    batches   = []
    for _ in range(nsamples):
        start = torch.randint(0, max_start, (1,), generator=gen).item()
        seg   = enc[start:start + seq_len].unsqueeze(0).to(device)
        batches.append({"input_ids": seg, "attention_mask": torch.ones_like(seg)})
    return batches


def load_test_ids(tokenizer):
    from datasets import load_dataset
    raw       = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    full_text = "\n\n".join([t for t in raw["text"] if t.strip()])
    return tokenizer(full_text, return_tensors="pt").input_ids


# ============================================================
# Calibration
# ============================================================
class Calibrator:
    def __init__(self, model, batches, device):
        self.model   = model
        self.batches = batches
        self.device  = device

    @torch.no_grad()
    def collect(self):
        Hs, counts, weights, handles = {}, {}, {}, []
        modules = get_linear_modules(self.model)
        for name, module in modules.items():
            n             = module.in_features
            Hs[name]      = torch.zeros(n, n, dtype=torch.float32, device="cpu")
            counts[name]  = 0
            weights[name] = module.weight.detach().float().cpu()

            def make_hook(nm):
                def fn(mod, inp, out):
                    x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                    Hs[nm]     += (2.0 * x.T @ x).cpu()
                    counts[nm] += x.shape[0]
                return fn
            handles.append(module.register_forward_hook(make_hook(name)))

        self.model.eval()
        self.model.config.use_cache = False
        with torch.inference_mode():
            for batch in self.batches:
                self.model(**{k: v.to(self.device) for k, v in batch.items()},
                           use_cache=False)
        for h in handles:
            h.remove()
        for name in Hs:
            n        = max(counts[name], 1)
            Hs[name] = 0.5 * (Hs[name] + Hs[name].T) / n
        return Hs, weights


# ============================================================
# NMF (Lee & Seung multiplicative update)
# ============================================================
def nmf_solve(V, rank, n_steps=200, H=None):
    """
    H=None: Frobenius NMF
    H 주어지면: H-weighted NMF
      A ← A ⊙ (V H B) / (A B^T H B)
      B ← B ⊙ (V^T H A) / (B A^T H A)
    """
    eps = 1e-8
    U, S, Vh = torch.linalg.svd(V, full_matrices=False)
    A = (U[:, :rank] * S[:rank].sqrt().unsqueeze(0)).abs().clamp(min=eps)
    B = (Vh[:rank, :].T * S[:rank].sqrt().unsqueeze(0)).abs().clamp(min=eps)

    if H is None:
        for _ in range(n_steps):
            A = (A * (V @ B)   / (A @ (B.T @ B) + eps)).clamp(min=eps)
            B = (B * (V.T @ A) / (B @ (A.T @ A) + eps)).clamp(min=eps)
    else:
        # H: [in, in], V: [out, in], A: [out, r], B: [in, r]
        for _ in range(n_steps):
            # A update
            HB    = H @ B                   # [in, r]
            VHB   = V @ HB                  # [out, r]  numerator A
            ABtHB = A @ (B.T @ HB)          # [out, r]  denominator A
            A = (A * VHB / (ABtHB + eps)).clamp(min=eps)

            # B update: gradient = -2 H(V^T A - B A^T A)
            HVtA  = H @ (V.T @ A)           # [in, r]  numerator B
            HBAtA = H @ (B @ (A.T @ A))     # [in, r]  denominator B
            B = (B * HVtA / (HBAtA + eps)).clamp(min=eps)
    return A, B


# ============================================================
# SmoothQuant: H outlier → W로 흡수
# ============================================================
def smooth(W, H, alpha):
    """
    s     = diag(H)^alpha    [in]
    W'    = W * s            outlier → W로 흡수
    H'    = S^{-1} H S^{-1} H flat해짐
    복원: hat_W = hat_W' * s_inv
    """
    if alpha == 0.0:
        return W, H, None

    h_diag = H.diagonal().clamp(min=1e-6)
    s      = h_diag.pow(alpha)
    s_inv  = 1.0 / s

    Wp = W * s.unsqueeze(0)
    Hp = H * s_inv.unsqueeze(0) * s_inv.unsqueeze(1)
    return Wp, Hp, s_inv



# ============================================================
# Factor quantization utilities
# ============================================================
def rebalance_factors(A, B, eps=1e-8):
    na = A.norm(dim=0).clamp(min=eps)
    nb = B.norm(dim=0).clamp(min=eps)
    s  = (nb / na).sqrt()
    return A * s.unsqueeze(0), B / s.unsqueeze(0)


def fake_quant_int4_per_col(X):
    """signed int4 per-column: SVD correction factors"""
    scale = X.abs().max(dim=0).values.clamp(min=1e-8) / 7.0
    X_q   = (X / scale.unsqueeze(0)).round().clamp(-8, 7)
    return X_q * scale.unsqueeze(0)


def fake_quant_uint4_per_col(X):
    """unsigned int4 per-column: NMF factors (non-negative)"""
    scale = X.max(dim=0).values.clamp(min=1e-8) / 15.0
    X_q   = (X / scale.unsqueeze(0)).round().clamp(0, 15)
    return X_q * scale.unsqueeze(0)


# ============================================================
# Core: SmoothQuant + NMF + correction
# ============================================================
def svid_nmf_correction(W, H, r_nmf, r_corr, nmf_steps=200, alpha=0.5, corr_int4=False, nmf_int4=False):
    dev = W.device
    out_dim, in_dim = W.shape

    # Step 0: SmoothQuant
    Wp, Hp, s_inv = smooth(W, H, alpha)

    # Step 0.5: Cholesky (Step 1, 2 모두 사용)
    diag_p   = Hp.diagonal()
    mean_d   = diag_p.mean().clamp(min=1e-8)
    median_d = diag_p.median().clamp(min=1e-8)
    ratio    = (mean_d / median_d).clamp(min=1.0)
    damp     = 0.01 * ratio * mean_d
    H_damp   = Hp + damp * torch.eye(in_dim, device=dev, dtype=Hp.dtype)
    try:
        L = torch.linalg.cholesky(H_damp)
    except Exception:
        damp   = 0.1 * mean_d
        H_damp = Hp + damp * torch.eye(in_dim, device=dev, dtype=Hp.dtype)
        L      = torch.linalg.cholesky(H_damp)

    # Step 1: sign x H-aware SVD on |W'| → int4 factors
    S_sign = Wp.sign()
    absWp  = Wp.abs()

    if r_nmf > 0:
        # H diagonal weighting: V' = |W'| * sqrt(h), NMF(V') → A, B'
        # B = B' / sqrt(h) 로 복원 → non-negative 유지하면서 H-aware
        sqrt_h  = Hp.diagonal().clamp(min=1e-8).sqrt()
        absWp_h = absWp * sqrt_h.unsqueeze(0)

        A, Bp   = nmf_solve(absWp_h, rank=r_nmf, n_steps=nmf_steps, H=None)
        B       = Bp / sqrt_h.unsqueeze(1)

        if nmf_int4:
            A, B = rebalance_factors(A, B)
            A = fake_quant_uint4_per_col(A)
            B = fake_quant_uint4_per_col(B)
        W_hat1p = S_sign * (A @ B.T)
    else:
        # mean alpha per-column: bpw overhead ≈ 0
        alpha   = absWp.mean(dim=0).clamp(min=1e-8)   # [in]
        W_hat1p = S_sign * alpha.unsqueeze(0)

    # Step 2: Hessian-aware SVD correction
    if r_corr > 0:
        Ep  = Wp - W_hat1p

        EL  = Ep @ L
        U_, Sv, Vh_ = torch.linalg.svd(EL, full_matrices=False)
        Ur  = U_[:, :r_corr]
        Sr  = Sv[:r_corr]
        Vhr = Vh_[:r_corr, :]
        # A_c = U_r * S_r^{1/2},  B_c = L^{-T} (S_r^{1/2} V_r^T)^T
        sqrt_Sr = Sr.sqrt()
        A_c = Ur * sqrt_Sr.unsqueeze(0)                        # [out, r]
        B_c = torch.linalg.solve_triangular(
                  L.T,
                  (Vhr * sqrt_Sr.unsqueeze(1)).T,
                  upper=True
              )                                                  # [in, r]

        A_c, B_c = rebalance_factors(A_c, B_c)

        if corr_int4:
            A_c = fake_quant_int4_per_col(A_c)
            B_c = fake_quant_int4_per_col(B_c)

        corr   = A_c @ B_c.T
        hat_Wp = W_hat1p + corr
    else:
        hat_Wp = W_hat1p

    # Step 3: 복원
    if s_inv is not None:
        hat_W = hat_Wp * s_inv.unsqueeze(0)
    else:
        hat_W = hat_Wp

    # rel_error (원본 H 기준)
    damp  = 0.01 * H.diagonal().median()
    H_e   = H + damp * torch.eye(in_dim, device=dev, dtype=H.dtype)
    Err   = W - hat_W
    rel_err = (
        torch.trace(Err @ H_e @ Err.T) /
        torch.trace(W @ H_e @ W.T).clamp(min=1e-8)
    ).sqrt().item()

    return hat_W, rel_err


# ============================================================
# Perplexity
# ============================================================
@torch.no_grad()
def compute_perplexity(model, input_ids, device, seq_len=2048):
    model.eval()
    ids = input_ids.to(device)
    total_nll, total_tokens = 0.0, 0
    for start in range(0, ids.shape[1] - 1, seq_len):
        seg = ids[:, start:start + seq_len]
        if seg.shape[1] < 2:
            continue
        out = model(seg, labels=seg)
        n = seg.shape[1] - 1
        total_nll    += out.loss.item() * n
        total_tokens += n
    return math.exp(total_nll / total_tokens)


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoConfig, OPTForCausalLM

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    cfg.tie_word_embeddings = False
    tok   = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = OPTForCausalLM.from_pretrained(
                MODEL_NAME, config=cfg, torch_dtype=torch.float32).to(device)
    model.eval()

    calib    = make_calib_batches(tok, NSAMPLES, CALIB_SEQ_LEN, SEED, device)
    test_ids = load_test_ids(tok)

    ppl_before = compute_perplexity(model, test_ids, device)
    print(f"Before PPL : {ppl_before:.4f}\n")

    H_dict, W_dict = Calibrator(model, calib, device).collect()
    print("Calibration done.\n")

    modules = get_linear_modules(model)

    hdr = (f"{'alpha':>6} {'r_nmf':>6} {'r_corr':>7} {'q':>6} {'bpw':>8} "
           f"{'err':>8} {'PPL':>12} {'ΔPPL':>10}")
    print("=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))

    for alpha in ALPHAS:
        for r_nmf, r_corr, corr_int4, nmf_int4 in CONFIGS:
            rel_errs, bpws = [], []
            t0 = time.time()

            desc = f"a={alpha:.2f} nmf={r_nmf} corr={r_corr}"
            for name in tqdm(list(W_dict.keys()), desc=desc, ncols=110, file=sys.stdout):
                H = H_dict[name].to(device)
                W = W_dict[name].to(device)

                hat_W, rel_err = svid_nmf_correction(
                    W, H, r_nmf, r_corr, nmf_steps=NMF_STEPS,
                    alpha=alpha, corr_int4=corr_int4, nmf_int4=nmf_int4
                )
                bpw = calc_bpw(W.shape, r_nmf, r_corr,
                               corr_int4=corr_int4, nmf_int4=nmf_int4)

                modules[name].weight.data.copy_(
                    hat_W.to(modules[name].weight.device, modules[name].weight.dtype)
                )
                rel_errs.append(rel_err)
                bpws.append(bpw)
                del H, W, hat_W
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            ppl = compute_perplexity(model, test_ids, device)
            restore_weights(model, W_dict)

            elapsed = time.time() - t0
            mean_err = sum(rel_errs) / len(rel_errs)
            mean_bpw = sum(bpws) / len(bpws)

            q_str = ('i4/i4' if nmf_int4 else 'i4') if corr_int4 else 'fp16'
            print(f"{alpha:>6.2f} {r_nmf:>6} {r_corr:>7} {q_str:>6} {mean_bpw:>8.4f} "
                  f"{mean_err:>8.4f} {ppl:>12.4f} {ppl-ppl_before:>+10.4f}  [{elapsed:.0f}s]")
            sys.stdout.flush()

            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print()

    print("=" * len(hdr))