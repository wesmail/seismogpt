#!/usr/bin/env python3
"""
Efficient autoregressive rollout evaluation — metrics-only (no plotting).

Key optimizations over testset_rollout.py:
  1. KV-cache: each autoregressive step processes only the NEW token,
     reusing cached K/V from all prior positions. The original script
     re-encodes the entire (growing) sequence every step — O(N*T^2)
     attention.  With KV-cache the prefill is O(T_ctx^2) and each
     decode step is O(T), giving overall O(T_ctx^2 + N*T) — typically
     10-100x faster per item.
  2. DataLoader with prefetch workers: overlaps disk I/O with GPU compute.
  3. cudnn.benchmark = True (auto-tunes kernels; no deterministic penalty).
  4. Optional --skip-psd to drop expensive Welch / robust-SNR metrics
     (~2-3x faster metric computation per item).
  5. Optional --half for float16 inference on GPU.

Produces the same CSV as testset_rollout.py (same columns, same semantics)
and is compatible with all downstream plotting scripts.

Usage:
  python efficient_rollout.py --checkpoint ckpt.ckpt --future_secs 120 \\
      --context_ratio 1 -n 2000 --csv metrics.csv

  python efficient_rollout.py --checkpoint ckpt.ckpt --future_secs 120 \\
      --context_ratio 1 -n 2000 --csv metrics.csv --half --skip-psd
"""

import argparse
import csv
import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_handling import SeismicWaveformDataset
from models.lightning_module import GPTLightning
from utils import compute_robust_snr
from scipy.signal import welch

# ---------------------------------------------------------------------------
# Metrics (identical to testset_rollout.py)
# ---------------------------------------------------------------------------
CHANNEL_LABELS = ["Z", "N", "E"]


def compute_skill(y_true, y_pred, eps=1e-12):
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    err = np.sum((y_true - y_pred) ** 2)
    ref = np.sum(y_true ** 2) + eps
    return (1.0 - err / ref).item()


def compute_snr_db(y_true, y_pred, eps=1e-10):
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    signal_power = (y_true ** 2).mean().item()
    noise_power = ((y_pred - y_true) ** 2).mean().item()
    return 10.0 * np.log10(signal_power / (noise_power + eps))


def normalized_xcorr(y_true, y_pred, eps=1e-8):
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    num = np.sum(y_true * y_pred)
    den = np.sqrt(np.sum(y_true ** 2) * np.sum(y_pred ** 2)) + eps
    return (num / den).item()


def compute_psd_metrics(y_true, y_pred, fs=1.0, eps=1e-12):
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    L = len(y_true)
    nperseg = min(1024, L)
    if nperseg < 8:
        nperseg = L
    noverlap = nperseg // 2
    f, Pt = welch(y_true, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling="density")
    _, Pp = welch(y_pred, fs=fs, nperseg=nperseg, noverlap=noverlap, scaling="density")
    log_Pt = np.log10(Pt + eps)
    log_Pp = np.log10(Pp + eps)
    psd_logl2 = np.mean((log_Pt - log_Pp) ** 2).item()
    psd_rel = np.mean(np.abs(Pt - Pp) / (Pt + eps)).item()
    return {"psd_logl2": psd_logl2, "psd_rel": psd_rel}


def evaluate_prediction_horizon(y_true_future, y_pred_future, fs=1.0, skip_psd=False):
    out = {}
    out["snr_db_global"] = compute_snr_db(y_true_future, y_pred_future)
    out["ncc_global"] = normalized_xcorr(y_true_future, y_pred_future)
    out["skill_global"] = compute_skill(y_true_future, y_pred_future)

    if not skip_psd:
        psd_g = compute_psd_metrics(y_true_future, y_pred_future, fs=fs)
        out["psd_logl2_global"] = psd_g["psd_logl2"]
        out["psd_rel_global"] = psd_g["psd_rel"]
        rob_g = compute_robust_snr(y_true_future, y_pred_future, fs)
        out["robust_snr_db_global"] = rob_g.get("snr_scaled_shifted_db", np.nan)
    else:
        out["psd_logl2_global"] = np.nan
        out["psd_rel_global"] = np.nan
        out["robust_snr_db_global"] = np.nan

    C = y_true_future.shape[1]
    for ch in range(C):
        label = CHANNEL_LABELS[ch] if ch < len(CHANNEL_LABELS) else str(ch)
        out[f"snr_db_{label}"] = compute_snr_db(y_true_future[:, ch], y_pred_future[:, ch])
        out[f"ncc_{label}"] = normalized_xcorr(y_true_future[:, ch], y_pred_future[:, ch])
        out[f"skill_{label}"] = compute_skill(y_true_future[:, ch], y_pred_future[:, ch])
        if not skip_psd:
            psd_ch = compute_psd_metrics(y_true_future[:, ch], y_pred_future[:, ch], fs=fs)
            out[f"psd_logl2_{label}"] = psd_ch["psd_logl2"]
            out[f"psd_rel_{label}"] = psd_ch["psd_rel"]
            rob_ch = compute_robust_snr(
                y_true_future[:, ch: ch + 1], y_pred_future[:, ch: ch + 1], fs
            )
            out[f"robust_snr_db_{label}"] = rob_ch.get("snr_scaled_shifted_db", np.nan)
        else:
            out[f"psd_logl2_{label}"] = np.nan
            out[f"psd_rel_{label}"] = np.nan
            out[f"robust_snr_db_{label}"] = np.nan
    return out


# ---------------------------------------------------------------------------
# KV-cache wrapper for the encoder stack
# ---------------------------------------------------------------------------

class KVCacheEncoderLayer:
    """
    Wraps a single RoPETransformerEncoderLayer for incremental decoding.

    On the first call (prefill) the full context is processed and K/V are
    stored.  On every subsequent call only the new token(s) are projected;
    the cached K/V are extended and the full K/V set is used for attention.
    """

    __slots__ = ("layer", "k_cache", "v_cache")

    def __init__(self, layer):
        self.layer = layer
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def reset(self):
        self.k_cache = None
        self.v_cache = None

    @torch.no_grad()
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        src: [B, T_new, d_model]
        Returns: [B, T_new, d_model]
        """
        layer = self.layer
        x = layer.self_attn_norm(src)
        B, T_new, _ = x.shape

        q = layer.w_q(x).view(B, T_new, layer.nhead, layer.head_dim).transpose(1, 2)
        k = layer.w_k(x).view(B, T_new, layer.nhead, layer.head_dim).transpose(1, 2)
        v = layer.w_v(x).view(B, T_new, layer.nhead, layer.head_dim).transpose(1, 2)

        if self.k_cache is not None:
            # Incremental step — apply RoPE at the correct absolute positions
            T_prev = self.k_cache.shape[2]
            cos = layer.rope.cos_cache[:, :, T_prev: T_prev + T_new, :].to(q.dtype)
            sin = layer.rope.sin_cache[:, :, T_prev: T_prev + T_new, :].to(q.dtype)
            q = layer.rope._apply_rotary(q, cos, sin)
            k = layer.rope._apply_rotary(k, cos, sin)
            self.k_cache = torch.cat([self.k_cache, k], dim=2)
            self.v_cache = torch.cat([self.v_cache, v], dim=2)
        else:
            # Prefill — RoPE applied from position 0
            q, k = layer.rope(q, k)
            self.k_cache = k
            self.v_cache = v

        # is_causal only on prefill (all new tokens); for incremental steps
        # the query attends to the full cached key set without causal restriction
        # because each new query position is after all cached positions.
        is_causal_flag = (self.k_cache.shape[2] == T_new)

        attn_out = F.scaled_dot_product_attention(
            q, self.k_cache, self.v_cache,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal_flag,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T_new, layer.d_model)
        out = src + layer.w_o(attn_out)
        out = out + layer.ffn(layer.ffn_norm(out))
        return out


class CachedGPT:
    """
    Wraps a GPT model for efficient autoregressive inference with KV-cache.

    Usage::

        cached = CachedGPT(model)
        cached.reset()
        out = cached.forward_prefill(x_context)   # [B, T_ctx, C, K]
        out = cached.forward_one_token(x_token)    # [B, 1, C, K]
    """

    def __init__(self, model):
        self.model = model
        self.cached_layers = [KVCacheEncoderLayer(l) for l in model.encoder_layers]

    def reset(self):
        for cl in self.cached_layers:
            cl.reset()

    def _run_encoder(self, h: torch.Tensor) -> torch.Tensor:
        for cl in self.cached_layers:
            h = cl.forward(h)
        return h

    def _decode_head(self, h: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """Run the horizon-0 prediction head. Returns [B, T*K, C_out]."""
        m = self.model
        hz = m.horizon_embed.weight[0].view(1, 1, -1)
        h_cond = h + hz
        mu_h = m.shared_mu_head(h_cond)
        K, C = m.kernel_size, m.in_channels
        mu_h = mu_h.view(B, T, K, C).contiguous()

        if getattr(m, "probabilistic_output", False) and m.shared_sigma_head is not None:
            raw_sig = m.shared_sigma_head(h_cond).view(B, T, K, C)
            sigma_h = F.softplus(raw_sig) + 1e-4
            mu_flat = mu_h.reshape(B, T * K, C)
            sig_flat = sigma_h.reshape(B, T * K, C)
            return torch.cat([mu_flat, sig_flat], dim=-1)

        return mu_h.reshape(B, T * K, C)

    @torch.no_grad()
    def forward_prefill(self, x_context: torch.Tensor) -> torch.Tensor:
        """Process full context, cache K/V.  Returns [B, T_ctx*K, C_out]."""
        B, T, C, K = x_context.shape
        h = self.model.time_token_embed(x_context)
        h = self.model.dropout_layer(h)
        h = self._run_encoder(h)
        return self._decode_head(h, B, T)

    @torch.no_grad()
    def forward_one_token(self, x_token: torch.Tensor) -> torch.Tensor:
        """Decode one new token, appending to KV cache.  Returns [B, K, C_out]."""
        B = x_token.shape[0]
        h = self.model.time_token_embed(x_token)
        h = self.model.dropout_layer(h)
        h = self._run_encoder(h)
        return self._decode_head(h, B, 1)


# ---------------------------------------------------------------------------
# Efficient rollout (KV-cache based)
# ---------------------------------------------------------------------------

def rollout_kvcache(
    cached_model: CachedGPT,
    x_full: torch.Tensor,
    T_context: int,
    n_future_tokens: int,
    kernel_size: int,
    mode: str = "free",
) -> tuple:
    """
    Autoregressive rollout with KV-cache.

    Prefill processes tokens [0 .. T_context-1] in a single forward pass and
    caches K/V.  The last-position output is the first future prediction.
    Each subsequent step feeds one token (predicted or teacher-forced) and
    appends to the cache — O(1) per-step projection instead of O(T).

    Returns
    -------
    true_future : ndarray [L_future, C]
    pred_future : ndarray [L_future, C]
    L_context   : int   (samples)
    L_future    : int   (samples)
    """
    B, T_full, C, K = x_full.shape
    T_context = min(T_context, T_full - 1)
    n_future_tokens = min(n_future_tokens, T_full - T_context)
    L_context = T_context * K

    cached_model.reset()

    x_ctx = x_full[:, :T_context]
    gt_future = x_full[:, T_context: T_context + n_future_tokens]

    # Prefill: one forward pass over the full context
    prefill_out = cached_model.forward_prefill(x_ctx)       # [1, T_ctx*K, C]
    C_out = prefill_out.shape[-1]
    # For probabilistic models C_out = 2*C; we only need the mu (first C channels)
    if C_out > C:
        prefill_out = prefill_out[:, :, :C]

    first_pred = prefill_out[:, -K:, :]                     # [1, K, C]
    pred_list = [first_pred.squeeze(0).cpu()]

    # Autoregressive decode
    for i in range(1, n_future_tokens):
        if mode == "teacher":
            inp_token = gt_future[:, i - 1: i]
        else:
            prev_pred = pred_list[-1].unsqueeze(0).to(x_full.device, x_full.dtype)
            inp_token = prev_pred.permute(0, 2, 1).unsqueeze(1)    # [1, 1, C, K]

        out = cached_model.forward_one_token(inp_token)     # [1, K, C_out]
        if out.shape[-1] > C:
            out = out[:, :, :C]
        pred_list.append(out[:, -K:, :].squeeze(0).cpu())

    pred_future = torch.cat(pred_list, dim=0).float().numpy()
    true_future = (
        gt_future.squeeze(0)
        .permute(0, 2, 1)
        .reshape(n_future_tokens * K, C)
        .cpu().float().numpy()
    )

    L = min(len(pred_future), len(true_future))
    return true_future[:L], pred_future[:L], L_context, L


# ---------------------------------------------------------------------------
# CSV columns (identical to testset_rollout.py)
# ---------------------------------------------------------------------------
METRICS_CSV_COLUMNS = [
    "item",
    "snr_db_global", "ncc_global", "skill_global", "psd_logl2_global", "psd_rel_global",
    "robust_snr_db_global",
    "snr_db_Z", "snr_db_N", "snr_db_E",
    "ncc_Z", "ncc_N", "ncc_E",
    "skill_Z", "skill_N", "skill_E",
    "robust_snr_db_Z", "robust_snr_db_N", "robust_snr_db_E",
    "psd_logl2_Z", "psd_logl2_N", "psd_logl2_E",
    "psd_rel_Z", "psd_rel_N", "psd_rel_E",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Efficient rollout evaluation (KV-cache, metrics-only)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to GPTLightning checkpoint (.ckpt)")
    p.add_argument("--data_dir", type=str,
                   default="/mnt/d/waleed/Seismology/simulation/clean/SobolDataset/PaperTestSet/",
                   help="Path to seisbench WaveformDataset directory")
    p.add_argument("--kernel_size", type=int, default=16, help="Samples per token (K)")
    p.add_argument("--num_tokens", type=int, default=320,
                   help="Number of tokens per sample in the dataset")
    p.add_argument("--context_ratio", type=float, default=1.0,
                   help="Context length = ratio x (S_arrival - P_arrival) in seconds")
    p.add_argument("--future_secs", type=float, required=True,
                   help="Future prediction length in seconds")
    p.add_argument("--mode", type=str, default="free", choices=["free", "teacher"],
                   help="Rollout mode: free (autoregressive) or teacher forcing")
    p.add_argument("--num_samples", "-n", type=int, default=1,
                   help="Number of random events to evaluate")
    p.add_argument("--csv", type=str, required=True, help="Output CSV path")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    g = p.add_argument_group("performance")
    g.add_argument("--skip-psd", action="store_true",
                   help="Skip PSD and robust-SNR metrics (~2-3x faster metric computation)")
    g.add_argument("--half", action="store_true",
                   help="Run inference in float16 (GPU only)")
    g.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader workers for I/O prefetch (0 = main-thread)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Prefetch helper — wraps dataset + item list into a DataLoader
# ---------------------------------------------------------------------------

class _ItemSubset(torch.utils.data.Dataset):
    """Thin wrapper that indexes a base dataset by a pre-selected item array."""
    def __init__(self, dataset, items):
        self.dataset = dataset
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        real_idx = int(self.items[idx])
        sample = self.dataset[real_idx]
        return real_idx, sample["x"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    t_start = time.time()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_half = args.half and device.type == "cuda"
    dtype = torch.float16 if use_half else torch.float32
    print(f"Device: {device}  |  dtype: {dtype}  |  Seed: {args.seed}")

    # ---- Data ----
    dataset = SeismicWaveformDataset(
        data_dir=args.data_dir,
        kernel_size=args.kernel_size,
        stride=args.kernel_size,
        num_tokens=args.num_tokens,
        training=False,
        normalize=True,
    )
    sample_rate = dataset.sr
    meta_data = dataset.metadata
    print(f"Sample rate: {sample_rate:.6f} Hz  |  Dataset size: {len(dataset)}")

    # ---- Model ----
    lightning_model = GPTLightning.load_from_checkpoint(args.checkpoint)
    model = lightning_model.gpt.to(device)
    if use_half:
        model = model.half()
    model.eval()
    cached_model = CachedGPT(model)

    # ---- Pick items (same RNG logic as testset_rollout.py) ----
    N = len(dataset)
    n = min(args.num_samples, N)
    rng = np.random.default_rng(args.seed)
    items = np.sort(rng.choice(N, size=n, replace=False))
    print(f"Evaluating {len(items)} samples  |  Mode: {args.mode}")
    print(f"Context: {args.context_ratio} x (S-P)  |  Future: {args.future_secs}s")
    if args.skip_psd:
        print("PSD / robust-SNR metrics SKIPPED (--skip-psd)")

    # ---- DataLoader for I/O prefetch ----
    subset = _ItemSubset(dataset, items)
    loader = DataLoader(
        subset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # ---- Rollout loop ----
    K = args.kernel_size
    metrics_rows = []
    pbar = tqdm(loader, total=len(items), desc="Rollout", unit="item")

    for real_idx_batch, x_batch in pbar:
        item = int(real_idx_batch[0])
        x_full = x_batch.to(device, dtype=dtype)   # [1, T, C, K]

        p_sec = meta_data.iloc[item].get("trace_p_arrival_s", np.nan)
        s_sec = meta_data.iloc[item].get("trace_s_arrival_s", np.nan)
        ps_sec = s_sec - p_sec

        if np.isfinite(ps_sec) and ps_sec > 0:
            context_secs = args.context_ratio * ps_sec
        else:
            context_secs = 60.0

        T_context = max(1, int(context_secs * sample_rate) // K)
        n_future_tokens = max(1, int(args.future_secs * sample_rate) // K)

        y_true_future, y_pred_rollout, L_ctx, L_fut = rollout_kvcache(
            cached_model, x_full, T_context, n_future_tokens, K, args.mode,
        )

        metrics = evaluate_prediction_horizon(
            y_true_future, y_pred_rollout, fs=sample_rate, skip_psd=args.skip_psd,
        )
        row = {"item": item}
        row.update(metrics)
        metrics_rows.append(row)
        pbar.set_postfix(
            SNR=f"{metrics['snr_db_global']:+.1f}dB",
            NCC=f"{metrics['ncc_global']:.3f}",
        )

    # ---- Save CSV ----
    out_dir = os.path.dirname(os.path.abspath(args.csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(metrics_rows)

    elapsed = time.time() - t_start
    print(f"\nMetrics CSV saved to {args.csv}")
    print(f"Done — {n} events in {elapsed:.1f}s  ({elapsed / max(n, 1):.2f} s/item)")


if __name__ == "__main__":
    main()
