#!/usr/bin/env python3
"""
Autoregressive rollout evaluation script.

Usage examples:
  # Free rollout, 360s context, 120s future, random item from test set
  python rollout_eval.py --checkpoint logs/version_0/checkpoints/epoch=14-step=70320.ckpt \
                         --context_secs 360 --future_secs 120 --mode free

  # Teacher-forced rollout on a specific item
  python rollout_eval.py --checkpoint logs/version_0/checkpoints/epoch=14-step=70320.ckpt \
                         --context_secs 360 --future_secs 120 --mode teacher --item 42

  # Custom data directory and kernel size
  python rollout_eval.py --checkpoint my_ckpt.ckpt --data_dir /path/to/data \
                         --context_secs 180 --future_secs 60 --kernel_size 16 --num_tokens 128

  # Save to a specific file
  python rollout_eval.py --checkpoint my_ckpt.ckpt --context_secs 360 --future_secs 120 \
                         --output my_rollout.png
"""

import argparse
import csv
import os
import numpy as np
import torch
import matplotlib
from tqdm import tqdm

matplotlib.use("Agg")  # non-interactive backend for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import matplotlib.patches as mpatches
from scipy.signal import welch

from data_handling.data_handling import (
    SeismicWaveformDataset,
    freq_features_from_tokens,
)
from models.lightning_module import GPTLightning
from utils import compute_robust_snr

# ---------------------------------------------------------------------------
# Global rcParams
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "mathtext.fontset":   "dejavuserif",
    "font.size":          7,
    "axes.titlesize":     7,
    "axes.labelsize":     7,
    "legend.fontsize":    6.5,
    "xtick.labelsize":    6.5,
    "ytick.labelsize":    6.5,
    "lines.linewidth":    0.75,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.5,
    "axes.grid":          True,
    "grid.color":         "#d0d0d0",
    "grid.linewidth":     0.3,
    "grid.alpha":         0.6,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.width":  0.5,
    "ytick.major.width":  0.5,
    "xtick.major.size":   2,
    "ytick.major.size":   2,
    "xtick.minor.visible": False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "savefig.dpi":        300,
    "figure.dpi":         150,
})
# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
C_TRUE   = "#2c3e7a"   # slate blue   – ground truth (full waveform)
C_FUTURE = "#b5320a"   # crimson      – true future (visible segment in future window)
C_PRED   = "#e07b00"   # amber/burnt  – model prediction
C_SHADE  = "#d0e4f7"   # pale blue    – context window shade
C_VLINE  = "#444444"   # dark grey    – context end marker

CHANNEL_LABELS = ["Z", "N", "E"]

# ---------------------------------------------------------------------------
# Metrics (prediction horizon): SNR, NCC, PSD, Forecast Skill
# ---------------------------------------------------------------------------
def compute_skill(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    Forecast skill score: Skill = 1 - ||y - y_hat||^2 / (||y||^2 + eps).
    Interpretation: 1 = perfect; 0 = equivalent to predicting zero; < 0 = worse than zero baseline.
    """
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    err = np.sum((y_true - y_pred) ** 2)
    ref = np.sum(y_true ** 2) + eps
    return (1.0 - err / ref).item()


def compute_snr_db(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10) -> float:
    """Signal-to-noise ratio in dB. Inputs are 1-D or 2-D (flattened internally)."""
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    signal_power = (y_true ** 2).mean().item()
    noise_power = ((y_pred - y_true) ** 2).mean().item()
    return 10.0 * np.log10(signal_power / (noise_power + eps))


def normalized_xcorr(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Normalized cross-correlation (scalar cosine similarity). Inputs flattened."""
    y_true, y_pred = y_true.ravel(), y_pred.ravel()
    num = np.sum(y_true * y_pred)
    den = np.sqrt(np.sum(y_true ** 2) * np.sum(y_pred ** 2)) + eps
    return (num / den).item()


def compute_psd_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fs: float = 1.0,
    eps: float = 1e-12,
) -> dict:
    """
    Welch PSD comparison for a single 1-D signal pair.
    Returns dict: psd_logl2, psd_rel.
    """
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


def evaluate_prediction_horizon(
    y_true_future: np.ndarray,
    y_pred_future: np.ndarray,
    fs: float = 1.0,
) -> dict:
    """Compute SNR, NCC, PSD, robust SNR, skill (global + per channel) over the prediction horizon. Returns flat dict for CSV."""
    out = {}
    # Global (all channels flattened)
    out["snr_db_global"] = compute_snr_db(y_true_future, y_pred_future)
    out["ncc_global"] = normalized_xcorr(y_true_future, y_pred_future)
    out["skill_global"] = compute_skill(y_true_future, y_pred_future)
    psd_g = compute_psd_metrics(y_true_future, y_pred_future, fs=fs)
    out["psd_logl2_global"] = psd_g["psd_logl2"]
    out["psd_rel_global"] = psd_g["psd_rel"]
    rob_g = compute_robust_snr(y_true_future, y_pred_future, fs)
    out["robust_snr_db_global"] = rob_g.get("snr_scaled_shifted_db", np.nan)
    # Per channel
    C = y_true_future.shape[1]
    for ch in range(C):
        label = CHANNEL_LABELS[ch] if ch < len(CHANNEL_LABELS) else str(ch)
        out[f"snr_db_{label}"] = compute_snr_db(y_true_future[:, ch], y_pred_future[:, ch])
        out[f"ncc_{label}"] = normalized_xcorr(y_true_future[:, ch], y_pred_future[:, ch])
        out[f"skill_{label}"] = compute_skill(y_true_future[:, ch], y_pred_future[:, ch])
        psd_ch = compute_psd_metrics(y_true_future[:, ch], y_pred_future[:, ch], fs=fs)
        out[f"psd_logl2_{label}"] = psd_ch["psd_logl2"]
        out[f"psd_rel_{label}"] = psd_ch["psd_rel"]
        rob_ch = compute_robust_snr(y_true_future[:, ch : ch + 1], y_pred_future[:, ch : ch + 1], fs)
        out[f"robust_snr_db_{label}"] = rob_ch.get("snr_scaled_shifted_db", np.nan)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def flatten_tokens_to_samples(tokens_1B_TCK: torch.Tensor) -> torch.Tensor:
    """[1, T, C, K] -> [T*K, C]"""
    assert tokens_1B_TCK.ndim == 4 and tokens_1B_TCK.shape[0] == 1
    T, C, K = tokens_1B_TCK.shape[1], tokens_1B_TCK.shape[2], tokens_1B_TCK.shape[3]
    return tokens_1B_TCK.squeeze(0).permute(0, 2, 1).reshape(T * K, C)


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def rollout(
    model,
    data: dict,
    item: int,
    context_secs: float,
    future_secs: float,
    sample_rate: float,
    device: torch.device,
    kernel_size: int = 16,
    mode: str = "free",
    max_context_tokens: int | None = None,
    *,
    freq_keep_bins: int = 8,
    freq_log1p: bool = True,
    freq_norm: str = "none",
):
    """
    Autoregressive rollout on token space.

    Args:
        model:              trained GPT model
        data:               batch dict with "x" : [B, T_full, C, K]
        item:               index inside the batch
        context_secs:       context length in seconds
        future_secs:        future (prediction) length in seconds
        sample_rate:        samples per second (from the dataset)
        device:             torch device
        kernel_size:        samples per token (K)
        mode:               "free" (autoregressive) or "teacher" (teacher forcing)
        max_context_tokens: optional sliding-window cap

    Returns:
        true_full_samples:  [T_full*K, C]  (numpy)
        true_future_samples:[L_future, C]  (numpy)
        pred_future_samples:[L_future, C]  (numpy)
        L_context:          int, context length in samples
        L_future:           int, future length in samples
    """
    x_full = data["x"].to(device)  # [B, T_full, C, K]
    B, T_full, C, K = x_full.shape

    assert K == kernel_size, f"Expected K=={kernel_size}, got K={K}"
    assert mode in ("free", "teacher"), "mode must be 'free' or 'teacher'"

    # Convert seconds → tokens
    T_context = max(1, int(context_secs * sample_rate) // K)
    n_future_tokens = max(1, int(future_secs * sample_rate) // K)

    # Clamp to available data
    T_context = min(T_context, T_full - 1)
    T_remain = T_full - T_context
    n_future_tokens = min(n_future_tokens, T_remain)

    L_context = T_context * K
    L_future = n_future_tokens * K

    # Seed context (true tokens)
    x = x_full[item : item + 1, :T_context].clone()  # [1, T_context, C, K]

    # Ground-truth future tokens
    gt_future_tokens = x_full[
        item : item + 1, T_context : T_context + n_future_tokens
    ].clone()

    pred_list = []
    for i in range(n_future_tokens):
        # Optional sliding window
        if max_context_tokens is not None and x.shape[1] > max_context_tokens:
            x = x[:, -max_context_tokens:, :, :].contiguous()

        # Frequency features: per-token rFFT magnitude (same as training / dataset)
        x_freq = freq_features_from_tokens(
            x,
            freq_keep_bins=freq_keep_bins,
            freq_log1p=freq_log1p,
            freq_norm=freq_norm,
        )

        with torch.no_grad():
            out = model(x, x_freq, is_causal=True)  # [1, T_in*K, C]

        next_samples = out[:, -K:, :]  # [1, K, C]
        pred_list.append(next_samples.squeeze(0).cpu())

        if mode == "teacher":
            x = torch.cat([x, gt_future_tokens[:, i : i + 1]], dim=1)
        else:
            next_token = next_samples.permute(0, 2, 1).unsqueeze(1)  # [1, 1, C, K]
            x = torch.cat([x, next_token], dim=1)

    pred_future_samples = torch.cat(pred_list, dim=0).numpy()
    true_future_samples = flatten_tokens_to_samples(gt_future_tokens).cpu().numpy()
    true_full_samples = flatten_tokens_to_samples(x_full[item : item + 1]).cpu().numpy()

    # Safety trim
    L = min(len(pred_future_samples), len(true_future_samples))
    pred_future_samples = pred_future_samples[:L]
    true_future_samples = true_future_samples[:L]

    return true_full_samples, true_future_samples, pred_future_samples, L_context, L


# ---------------------------------------------------------------------------
# Main function  (same signature as the original)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Colour palette  (identical to the standard pubready script)
# ---------------------------------------------------------------------------
C_TRUE   = "#2c3e7a"   # slate blue  – observation
C_FUTURE = "#b5320a"   # crimson     – true future
C_PRED   = "#e07b00"   # amber       – model prediction
C_SHADE  = "#d0e4f7"   # pale blue   – context window
C_VLINE  = "#555555"   # grey        – context end marker

CHANNEL_LABELS = ["Z", "N", "E"]

# ---------------------------------------------------------------------------
# Layout presets
# ---------------------------------------------------------------------------
_LAYOUTS = {
    #            figsize        hspace  lw_true  lw_pred  ch_fs  meta_fs  leg_fs
    "compact":  ((7.0, 2.0),   0.04,   0.65,    0.85,    7.5,   6.5,     6.0),
    "standard": ((7.0, 3.2),   0.10,   0.80,    1.00,    8.5,   7.5,     7.0),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sec_to_mmss(x, _):
    s = int(max(x, 0))
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"


def _smart_xticks(total_s: float, n_target: int = 10) -> np.ndarray:
    """Pick round tick intervals that yield ~n_target ticks."""
    candidates = [10, 20, 30, 60, 90, 120, 180, 300, 600]
    step = min(candidates, key=lambda c: abs(total_s / c - n_target))
    return np.arange(0, total_s + step * 0.01, step)


def _per_channel_ncc(
    y_true_future: np.ndarray,
    y_pred_rollout: np.ndarray,
    eps: float = 1e-8,
) -> list[float]:
    """Return NCC for each channel."""
    nccs = []
    for ch in range(y_true_future.shape[1]):
        t = y_true_future[:, ch].ravel()
        p = y_pred_rollout[:, ch].ravel()
        num = np.sum(t * p)
        den = np.sqrt(np.sum(t ** 2) * np.sum(p ** 2)) + eps
        nccs.append(float(num / den))
    return nccs


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def plot_rollout(
    y_true_full: np.ndarray,
    y_true_future: np.ndarray,
    y_pred_rollout: np.ndarray,
    L_ctx: int,
    sample_rate: float,
    mode: str,
    context_secs: float,
    future_secs: float,
    item: int,
    output_path: str,
    *,
    metadata: dict | None = None,
    metrics:  dict | None = None,
    layout:   str = "compact",
):
    """
    Compact 3-channel rollout figure.

    Parameters
    ----------
    layout : "compact" (default, ~2 in tall) or "standard" (~3.2 in tall).
    All other parameters are identical to the original plot_rollout signature.
    """
    if layout not in _LAYOUTS:
        raise ValueError(f"layout must be one of {list(_LAYOUTS)}; got '{layout}'")

    figsize, hspace, lw_true, lw_pred, ch_fs, meta_fs, leg_fs = _LAYOUTS[layout]

    # ---- Time axes ----------------------------------------------------------
    N_full  = len(y_true_full)
    t_full  = np.arange(N_full) / sample_rate
    t_fut   = np.arange(L_ctx, L_ctx + len(y_pred_rollout)) / sample_rate
    ctx_end = L_ctx / sample_rate
    total_s = N_full / sample_rate

    # Per-channel NCC (computed here so it is always available even if the
    # caller did not pass a `metrics` dict)
    ch_ncc = _per_channel_ncc(y_true_future, y_pred_rollout)

    # ---- Figure layout ------------------------------------------------------
    fig, axes = plt.subplots(
        3, 1, figsize=figsize, sharex=True,
        gridspec_kw={"hspace": hspace},
    )

    # Reserve left margin for shared y-label + right margin
    fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.18)

    # ---- Draw each channel --------------------------------------------------
    for ch, ax in enumerate(axes):
        # Context shade
        ax.axvspan(0, ctx_end, color=C_SHADE, alpha=0.22, lw=0, zorder=0)

        # Observation (full)
        ax.plot(t_full, y_true_full[:, ch],
                color=C_TRUE, lw=lw_true, alpha=0.85, zorder=2)

        # True future
        ax.plot(t_fut, y_true_future[:, ch],
                color=C_FUTURE, lw=lw_true + 0.1, alpha=0.70, zorder=3)

        # Prediction
        ax.plot(t_fut, y_pred_rollout[:, ch],
                color=C_PRED, lw=lw_pred, ls="--", alpha=0.95, zorder=4)

        # Context-end marker
        ax.axvline(ctx_end, color=C_VLINE, ls=(0, (4, 3)), lw=0.6, zorder=5)

        # Channel label — top-left corner, bold
        lbl = CHANNEL_LABELS[ch] if ch < 3 else str(ch)
        ax.text(0.005, 0.90, lbl, transform=ax.transAxes,
                fontsize=ch_fs, fontweight="bold", va="top", ha="left",
                color="#111111")

        # Per-channel NCC — top-right corner, italic
        ax.text(0.995, 0.90, f"NCC={ch_ncc[ch]:.3f}",
                transform=ax.transAxes,
                fontsize=ch_fs - 0.5, style="italic", va="top", ha="right",
                color="#333333")

        # Y-axis: symmetric limits, minimal ticks
        peak = float(np.abs(y_true_full[:, ch]).max())
        if peak > 0:
            ax.set_ylim(-(peak * 1.12), peak * 1.12)
        ax.yaxis.set_major_locator(mpl_ticker.MaxNLocator(nbins=2, symmetric=True))
        ax.tick_params(axis="y", pad=1)

        # Remove x-tick labels on upper panels (shared axis)
        if ch < 2:
            ax.tick_params(axis="x", length=0)

    # ---- Shared y-label (rotated, left of panels) ---------------------------
    fig.text(0.012, 0.50, "Amplitude", va="center", ha="center",
             rotation="vertical", fontsize=7, color="#333333")

    # ---- X-axis (bottom panel only) -----------------------------------------
    ticks = _smart_xticks(total_s)
    axes[-1].set_xticks(ticks)
    axes[-1].xaxis.set_major_formatter(mpl_ticker.FuncFormatter(_sec_to_mmss))
    axes[-1].set_xlim(0, total_s)
    axes[-1].set_xlabel("Time (MM:SS)", labelpad=2, fontsize=7)
    axes[-1].tick_params(axis="x", labelrotation=0, pad=1)

    # ---- Header: metadata + global NCC on a single compact line -------------
    header_parts = []
    if metadata:
        depth = metadata.get("src_depth_km", np.nan)
        Mw    = metadata.get("Mw",           np.nan)
        dist  = metadata.get("distance_deg", np.nan)
        if not np.isnan(depth): header_parts.append(f"depth={depth:.1f} km")
        if not np.isnan(Mw):    header_parts.append(fr"$M_w$={Mw:.2f}")
        if not np.isnan(dist):  header_parts.append(fr"$\Delta$={dist:.1f}°")

    if metrics:
        ncc_g = metrics.get("ncc_global", np.nan)
        skl   = metrics.get("skill_global", np.nan)
        if not np.isnan(ncc_g): header_parts.append(f"NCC={ncc_g:.4f}")
        if not np.isnan(skl):   header_parts.append(f"Skill={skl:.4f}")

    header_str = "    ".join(header_parts)
    fig.text(0.50, 0.97, header_str,
             ha="center", va="top",
             fontsize=meta_fs, color="#444444", style="italic")

    # ---- Compact horizontal legend above the panels -------------------------
    legend_handles = [
        mpatches.Patch(color=C_SHADE,  alpha=0.4,  label="Context"),
        plt.Line2D([0], [0], color=C_TRUE,   lw=0.9,              label="Observation"),
        plt.Line2D([0], [0], color=C_FUTURE, lw=0.9, alpha=0.7,   label="True future"),
        plt.Line2D([0], [0], color=C_PRED,   lw=1.0, ls="--",     label=f"Pred ({mode})"),
        plt.Line2D([0], [0], color=C_VLINE,  lw=0.7, ls=(0,(4,3)),label="Context end"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.92),     # just below the metadata line
        ncol=5,
        frameon=True, framealpha=0.95,
        edgecolor="#cccccc",
        fontsize=leg_fs,
        handlelength=1.4,
        handletextpad=0.4,
        columnspacing=0.8,
        borderpad=0.3,
    )

    # ---- Save ---------------------------------------------------------------
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {output_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Autoregressive rollout evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to GPTLightning checkpoint (.ckpt)")

    # Data
    p.add_argument("--data_dir", type=str,
                    default="/mnt/d/waleed/Seismology/simulation/clean/SobolDataset/Regional/testset",
                    help="Path to seisbench WaveformDataset directory")
    p.add_argument("--kernel_size", type=int, default=16, help="Samples per token (K)")
    p.add_argument("--num_tokens", type=int, default=128, help="Number of tokens per sample in the dataset")

    # Rollout
    p.add_argument("--context_secs", type=float, required=True, help="Context length in seconds")
    p.add_argument("--future_secs", type=float, required=True, help="Future prediction length in seconds")
    p.add_argument("--mode", type=str, default="free", choices=["free", "teacher"],
                    help="Rollout mode: 'free' (autoregressive) or 'teacher' (teacher forcing)")
    p.add_argument("--num_samples", "-n", type=int, default=1,
                    help="Number of random events to evaluate")
    p.add_argument("--max_context_tokens", type=int, default=None,
                    help="Optional sliding-window cap on context tokens")

    # Output
    p.add_argument("--output", type=str, default="rollout.png", help="Output image path")
    p.add_argument("--csv", type=str, default=None,
                    help="Path to save metrics CSV (item, snr_db_global, ncc_global, psd_*, per-channel). If not set, no CSV is written.")
    p.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducible item selection (same seed => same items every run)")

    return p.parse_args()


# CSV column order for metrics
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


def main():
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
    print(f"Sample rate: {sample_rate:.6f} Hz")

    # ---- Model ----
    lightning_model = GPTLightning.load_from_checkpoint(args.checkpoint)
    model = lightning_model.gpt.to(device)
    model.eval()
    # Freq feature config (must match training for rollout parity)
    h = getattr(lightning_model, "hparams", None)
    freq_keep_bins = int(getattr(h, "freq_keep_bins", 8)) if h is not None else 8
    freq_log1p = bool(getattr(h, "freq_log1p", True)) if h is not None else True
    freq_norm = str(getattr(h, "freq_norm", "none")) if h is not None else "none"

    # ---- Pick items ----
    N = len(dataset)
    n = min(args.num_samples, N)
    items = np.atleast_1d(np.random.choice(N, size=n, replace=False))
    print(f"Evaluating {len(items)} samples  |  Mode: {args.mode}")
    print(f"Context: {args.context_secs}s  |  Future: {args.future_secs}s")

    # ---- Rollout loop ----
    stem, ext = os.path.splitext(args.output)
    ext = ext or ".png"
    metrics_rows = []
    n_plot = max(1, int(0.01 * len(items)))
    n_plotted = 0

    pbar = tqdm(list(items), desc="Rollout", unit="item")
    item_num = 0
    for item in pbar:
        item_num += 1
        if item_num > 500:
            break
        sample = dataset[int(item)]
        p_arrival_sec = meta_data.iloc[item].get("trace_p_arrival_s", np.nan)
        s_arrival_sec = meta_data.iloc[item].get("trace_s_arrival_s", np.nan)
        offset_sec = s_arrival_sec - p_arrival_sec

        data = {"x": sample["x"].unsqueeze(0)}
        y_true_full, y_true_future, y_pred_rollout, L_ctx, L_fut = rollout(
            model=model,
            data=data,
            item=0,
            context_secs=args.context_secs + offset_sec,
            future_secs=args.future_secs,
            sample_rate=sample_rate,
            device=device,
            kernel_size=args.kernel_size,
            mode=args.mode,
            max_context_tokens=args.max_context_tokens,
            freq_keep_bins=freq_keep_bins,
            freq_log1p=freq_log1p,
            freq_norm=freq_norm,
        )

        # Metrics over prediction horizon
        metrics = evaluate_prediction_horizon(y_true_future, y_pred_rollout, fs=sample_rate)
        row = {"item": int(item)}
        row.update(metrics)
        metrics_rows.append(row)
        pbar.set_postfix(SNR=f"{metrics['snr_db_global']:+.1f}dB", NCC=f"{metrics['ncc_global']:.3f}")

        # Plot 1% of samples (at least one)
        if n_plotted < n_plot:
            meta = dataset.metadata.iloc[item]
            metadata_dict = {
                "trace_p_arrival_s": meta.get("trace_p_arrival_s", np.nan),
                "trace_s_arrival_s": meta.get("trace_s_arrival_s", np.nan),
                "src_depth_km": meta.get("src_depth_km", np.nan),
                "Mw": meta.get("Mw", np.nan),
                "distance_deg": meta.get("distance_deg", np.nan),
            }
            out_path = f"{stem}_item_{item:05d}{ext}" if n > 1 else f"{stem}{ext}"
            plot_rollout(
                y_true_full, y_true_future, y_pred_rollout,
                L_ctx, sample_rate, args.mode,
                args.context_secs, args.future_secs,
                int(item), out_path,
                metadata=metadata_dict,
                metrics=metrics,
            )
            n_plotted += 1

    if args.csv and metrics_rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)) or ".", exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=METRICS_CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(metrics_rows)
        print(f"\nMetrics CSV saved to {args.csv}")

    print(f"\nDone — {n} event(s) evaluated.")


if __name__ == "__main__":
    main()