#!/usr/bin/env python3
"""
Publication figure generator for SeismoGPT paper.

Subcommands:

  horizon-series : same event at 60 / 120 / 240 s horizons  (Fig 7)
  contrast       : successful forecasts (stacked examples)    (Fig 8)
  multi-events   : filter metrics by Δ / Mw / depth (and optional NCC),
                   run up to N rollouts, one Z–N–E row per event (batch QC)

Reuses model loading, rollout, and rcParams from testset_rollout.py.

Examples:
  python make_paper_figures.py horizon-series \\
      --checkpoint best.ckpt --metadata-csv meta.csv \\
      --metrics-csv metrics.csv --output fig7.png

  python make_paper_figures.py contrast \\
      --checkpoint best.ckpt --metadata-csv meta.csv \\
      --metrics-csv metrics.csv --output fig8.png --n-examples 2

  python make_paper_figures.py multi-events \\
      --checkpoint best.ckpt --metadata-csv meta.csv \\
      --metrics-csv metrics.csv --output figures/batch.pdf \\
      --dist-lo 30 --dist-hi 50 --mag-lo 5 --mag-hi 6 -n 5
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import testset_rollout as tsr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker

from data.data_handling import SeismicWaveformDataset
from models.lightning_module import GPTLightning

# ---------------------------------------------------------------------------
# Publication rcParams (identical to plot_horizon_overlay.py)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "mathtext.fontset":   "dejavuserif",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "legend.fontsize":    8.5,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.4,
    "axes.grid":          False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

# ---------------------------------------------------------------------------
# Shared visual constants (matching plot_horizon_overlay.py)
# ---------------------------------------------------------------------------
CH_LABELS = tsr.CHANNEL_LABELS          # ["Z", "N", "E"]

C_TRUTH = "#333333"
C_PRED  = "#e07b00"
C_SHADE = "#d6e6f5"
C_P     = "#d62728"
C_S     = "#2ca02c"

HORIZON_COLORS = [
    "#2c8c6e",   # teal
    "#4c72b0",   # steel blue
    "#8c564b",   # brown
    "#9467bd",   # purple
]


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — model / data loading (thin wrappers around testset_rollout)
# ═══════════════════════════════════════════════════════════════════════════

def _load_model_and_data(args):
    """Return (model, dataset, sample_rate, metadata, device)."""
    tsr._set_global_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SeismicWaveformDataset(
        data_dir=args.data_dir,
        kernel_size=args.kernel_size,
        stride=args.kernel_size,
        num_tokens=args.num_tokens,
        training=False,
        normalize=True,
    )
    sr = dataset.sr
    meta = dataset.metadata
    lightning = GPTLightning.load_from_checkpoint(args.checkpoint)
    model = lightning.gpt.to(device)
    model.eval()
    return model, dataset, sr, meta, device


def _run_rollout(model, dataset, meta, item, context_ratio, future_secs,
                 sr, device, kernel_size, mode="free"):
    """Run a single rollout.

    Returns (y_true_full, y_true_future, y_pred, L_ctx, L_fut, context_secs, ps_sec).
    ps_sec is the S-P travel time in seconds (S-arrival relative to t=0 since the
    waveform segment starts at the P-arrival).
    """
    sample = dataset[int(item)]
    p_sec = meta.iloc[item].get("trace_p_arrival_s", np.nan)
    s_sec = meta.iloc[item].get("trace_s_arrival_s", np.nan)
    ps_sec = s_sec - p_sec
    context_secs = context_ratio * ps_sec if (np.isfinite(ps_sec) and ps_sec > 0) else 60.0
    data = {"x": sample["x"].unsqueeze(0)}
    y_true_full, y_true_future, y_pred, L_ctx, L_fut = tsr.rollout(
        model=model, data=data, item=0,
        context_secs=context_secs, future_secs=future_secs,
        sample_rate=sr, device=device, kernel_size=kernel_size, mode=mode,
    )
    return y_true_full, y_true_future, y_pred, L_ctx, L_fut, context_secs, ps_sec


def _get_event_meta(metadata_csv, item, dist_col, depth_col, mag_col):
    """Return (distance, depth, Mw) for a single item from the external metadata CSV."""
    row = metadata_csv.iloc[int(item)]
    return (float(row.get(dist_col, np.nan)),
            float(row.get(depth_col, np.nan)),
            float(row.get(mag_col,   np.nan)))


def _ncc_on_window(y_true, y_pred, start, end):
    """NCC computed on samples [start:end] across all channels."""
    t = y_true[start:end]
    p = y_pred[start:end]
    return tsr.normalized_xcorr(t, p)


def _sec_to_mmss(x, _):
    s = int(max(x, 0))
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"


def save_contrast_style_figure(
    events: list[dict],
    output_path: str | Path,
    *,
    save_dpi: int = 600,
    fig_width: float = 4.6,
    row_height: float = 1.1,
    top_pad: float = 0.5,
    hspace: float = 0.25,
    show_arrival_labels: bool = False,
) -> None:
    """
    Save a Fig-8-style contrast PDF/PNG: 3 rows per event (Z, N, E), one column.

    Each entry in *events* must provide:
      y_full, y_pred, L_ctx, L_fut, sr, ps_sec (float or nan),
      title (optional str for the Z-panel axes title).

    Defaults (4.6 × 3.8 in for one event) match the paper contrast figure.
    Catalog rollouts pass a wider, shorter size via *fig_width* / *row_height*.
    """
    n_ex = len(events)
    if n_ex < 1:
        raise ValueError("events must be non-empty")

    n_rows = 3 * n_ex
    fig_h = row_height * n_rows + top_pad
    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_width, fig_h), sharex=True,
                             squeeze=False)
    fig.subplots_adjust(hspace=hspace)

    for ex_idx, ev in enumerate(events):
        y_full = ev["y_full"]
        y_pred = ev["y_pred"]
        L_ctx = int(ev["L_ctx"])
        L_fut = int(ev["L_fut"])
        sr = float(ev["sr"])
        ps_sec = ev.get("ps_sec", np.nan)
        s_arr = ps_sec if np.isfinite(ps_sec) else None

        ctx_end_s = L_ctx / sr
        pred_end_s = ctx_end_s + L_fut / sr
        t_full = np.arange(len(y_full)) / sr
        t_pred = np.arange(L_ctx, L_ctx + L_fut) / sr

        base_row = ex_idx * 3
        title = ev.get("title")
        if title:
            axes[base_row, 0].set_title(title, fontsize=8, pad=4)

        for ch_idx, ch_label in enumerate(CH_LABELS):
            ax = axes[base_row + ch_idx, 0]

            ax.axvspan(0, ctx_end_s, color=C_SHADE, alpha=0.45, lw=0, zorder=0)

            full_end = min(int(pred_end_s * sr), len(y_full))
            ax.plot(t_full[:full_end], y_full[:full_end, ch_idx],
                    color=C_TRUTH, lw=0.7, alpha=0.85, zorder=2)
            ax.plot(t_pred, y_pred[:L_fut, ch_idx],
                    color=C_PRED, lw=0.9, ls="--", alpha=0.88, zorder=3)

            ax.axvline(0, color=C_P, lw=0.8, alpha=0.7, zorder=5)
            if s_arr is not None and s_arr > 0:
                ax.axvline(s_arr, color=C_S, lw=0.8, alpha=0.7, zorder=5)

            if show_arrival_labels and ch_idx == 0:
                label_y = 0.93
                x_off = max(0.5, 0.008 * pred_end_s)
                ax.text(
                    x_off, label_y, "P",
                    transform=ax.get_xaxis_transform(),
                    color=C_P, fontsize=8, fontweight="bold",
                    va="top", ha="left", clip_on=False, zorder=7,
                )
                if s_arr is not None and s_arr > 0:
                    ax.text(
                        s_arr + x_off, label_y, "S",
                        transform=ax.get_xaxis_transform(),
                        color=C_S, fontsize=8, fontweight="bold",
                        va="top", ha="left", clip_on=False, zorder=7,
                    )

            peak = float(np.abs(y_full[:full_end, ch_idx]).max()) or 1.0
            ax.set_ylim(-peak * 1.15, peak * 1.15)

            ax.set_ylabel(ch_label, fontsize=11, fontweight="bold",
                          rotation=0, labelpad=12, va="center")
            ax.xaxis.set_major_formatter(mpl_ticker.FuncFormatter(_sec_to_mmss))

            is_last_row = (base_row + ch_idx == n_rows - 1)
            if is_last_row:
                ax.set_xlabel("Time (mm:ss)", fontsize=10)

        if ex_idx < n_ex - 1:
            sep_ax = axes[ex_idx * 3 + 2, 0]
            sep_ax.spines["bottom"].set_visible(True)
            sep_ax.spines["bottom"].set_linewidth(0.8)
            sep_ax.spines["bottom"].set_color("#999999")

    top_rect = 0.97 if show_arrival_labels else 0.95
    fig.tight_layout(rect=[0, 0, 1, top_rect])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Selection helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_metrics_and_meta(args):
    """Load and merge the metrics CSV and external metadata CSV."""
    mdf = pd.read_csv(args.metrics_csv)
    for col in ("item", "ncc_global"):
        if col not in mdf.columns:
            raise KeyError(f"'{col}' missing in {args.metrics_csv}. "
                           f"Available: {mdf.columns.tolist()}")
    ext = pd.read_csv(args.metadata_csv)
    dist_col = args.distance_col
    depth_col = args.depth_col
    mag_col = args.mag_col
    for col in (dist_col, depth_col, mag_col):
        if col not in ext.columns:
            raise KeyError(f"'{col}' missing in {args.metadata_csv}. "
                           f"Available: {ext.columns.tolist()}")
    mdf["_dist"] = mdf["item"].apply(
        lambda i: ext.iloc[int(i)][dist_col] if int(i) < len(ext) else np.nan)
    mdf["_depth"] = mdf["item"].apply(
        lambda i: ext.iloc[int(i)][depth_col] if int(i) < len(ext) else np.nan)
    mdf["_mag"] = mdf["item"].apply(
        lambda i: ext.iloc[int(i)][mag_col] if int(i) < len(ext) else np.nan)
    return mdf, ext


def _pick_median(sub: pd.DataFrame) -> int:
    """Return item index of the row closest to the median ncc_global."""
    if sub.empty:
        return -1
    med = sub["ncc_global"].median()
    idx = (sub["ncc_global"] - med).abs().idxmin()
    return int(sub.loc[idx, "item"])


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand 1: horizon-series
# ═══════════════════════════════════════════════════════════════════════════

def _add_horizon_args(sub):
    sub.add_argument("--item", type=int, default=None,
                     help="Force a specific item index (skip auto-selection)")
    sub.add_argument("--dist-lo", type=float, default=50.0)
    sub.add_argument("--dist-hi", type=float, default=60.0)
    sub.add_argument("--mag-lo", type=float, default=6.0)
    sub.add_argument("--mag-hi", type=float, default=7.0)
    sub.add_argument("--depth-lo", type=float, default=50.0)
    sub.add_argument("--depth-hi", type=float, default=60.0)
    sub.add_argument("--ncc-min", type=float, default=0.9)


def cmd_horizon_series(args):
    mdf, ext_meta = _load_metrics_and_meta(args)
    horizons = [120, 240, 300, 600]

    # ---- Select event ----
    if args.item is not None:
        item = args.item
    else:
        filt = mdf[
            (mdf["_dist"]  >= args.dist_lo)  & (mdf["_dist"]  <= args.dist_hi) &
            (mdf["_mag"]   >= args.mag_lo)   & (mdf["_mag"]   <= args.mag_hi) &
            (mdf["_depth"] >= args.depth_lo) & (mdf["_depth"] <= args.depth_hi) &
            (mdf["ncc_global"] >= args.ncc_min)
        ]
        if filt.empty:
            print("ERROR: no items match the selection filters:", file=sys.stderr)
            print(f"  distance ∈ [{args.dist_lo}, {args.dist_hi}]°", file=sys.stderr)
            print(f"  magnitude ∈ [{args.mag_lo}, {args.mag_hi}]", file=sys.stderr)
            print(f"  depth ∈ [{args.depth_lo}, {args.depth_hi}] km", file=sys.stderr)
            print(f"  ncc_global ≥ {args.ncc_min}", file=sys.stderr)
            sys.exit(1)
        item = _pick_median(filt)

    dist, depth, mw = _get_event_meta(ext_meta, item, args.distance_col,
                                       args.depth_col, args.mag_col)
    ncc_val = float(mdf.loc[mdf["item"] == item, "ncc_global"].iloc[0])
    print(f"Selected item {item}: Mw={mw:.1f}, Δ={dist:.0f}°, "
          f"depth={depth:.0f} km, ncc_global={ncc_val:.4f}")

    # ---- Model + single long rollout (longest horizon only) ----
    model, dataset, sr, ds_meta, device = _load_model_and_data(args)
    longest = float(max(horizons))
    y_full, y_fut, y_pred, L_ctx, L_fut, ctx_secs, ps_sec = _run_rollout(
        model, dataset, ds_meta, item, args.context_ratio,
        longest, sr, device, args.kernel_size,
    )
    ctx_end_s = L_ctx / sr
    t_full = np.arange(len(y_full)) / sr
    max_sample = min(L_ctx + L_fut, len(y_full))

    s_arrival_s = ps_sec if np.isfinite(ps_sec) else None

    # Pre-compute per-horizon mean NCC (sub-windows of the single rollout)
    n_channels = 3
    sorted_h = sorted(horizons)
    horizon_ncc = {}
    for h_sec in sorted_h:
        h_samples = min(int(h_sec * sr), L_fut)
        nccs = [tsr.normalized_xcorr(y_fut[:h_samples, ch], y_pred[:h_samples, ch])
                for ch in range(n_channels)]
        horizon_ncc[h_sec] = nccs

    # ---- Figure: 3 rows (Z / N / E), shared x-axis ----
    fig, axes = plt.subplots(n_channels, 1,
                             figsize=(7.2, 1.1 * n_channels + 0.5),
                             sharex=True)
    if n_channels == 1:
        axes = [axes]

    t_pred = np.arange(L_ctx, L_ctx + L_fut) / sr

    for ch_idx, (ax, ch_label) in enumerate(zip(axes, CH_LABELS)):
        ax.axvspan(0, ctx_end_s, color=C_SHADE, alpha=0.45, lw=0, zorder=0)

        ax.plot(t_full[:max_sample], y_full[:max_sample, ch_idx],
                color=C_TRUTH, lw=0.7, alpha=0.85, zorder=2)

        ax.plot(t_pred, y_pred[:L_fut, ch_idx],
                color=C_PRED, lw=0.9, ls="--", alpha=0.88, zorder=3)

        ax.axvline(0, color=C_P, lw=0.8, alpha=0.7, zorder=5)
        if s_arrival_s is not None and s_arrival_s > 0:
            ax.axvline(s_arrival_s, color=C_S, lw=0.8, alpha=0.7, zorder=5)
        if ch_idx == 0:
            ax.text(0 + 0.5, 0.90, "P",
                    transform=ax.get_xaxis_transform(),
                    color=C_P, fontsize=8, fontweight="bold", va="top")
            if s_arrival_s is not None:
                ax.text(s_arrival_s + 0.5, 0.90, " S",
                        transform=ax.get_xaxis_transform(),
                        color=C_S, fontsize=8, fontweight="bold", va="top")

        peak = float(np.abs(y_full[:max_sample, ch_idx]).max()) or 1.0
        ax.set_ylim(-peak * 1.15, peak * 1.15)

        for h_idx, h_sec in enumerate(sorted_h):
            h_samples = min(int(h_sec * sr), L_fut)
            ckpt_s = ctx_end_s + h_samples / sr
            is_last = (h_sec == longest)
            col = C_PRED if is_last else HORIZON_COLORS[h_idx % len(HORIZON_COLORS)]
            ax.axvline(ckpt_s, color=col, ls="--", lw=1.2, alpha=0.7, zorder=4)

            if ch_idx == 0:
                ncc_mean = np.mean(horizon_ncc[h_sec])
                y_stagger = 1.12 if (h_idx % 2 == 0) else 1.30
                ax.text(ckpt_s, y_stagger,
                        f"{int(h_sec)} s\nNCC {ncc_mean:.2f}",
                        transform=ax.get_xaxis_transform(),
                        fontsize=6.5, color=col, ha="center", va="bottom",
                        fontweight="bold")

        ax.set_ylabel(ch_label, fontsize=11, fontweight="bold", rotation=0,
                      labelpad=12, va="center")

    axes[-1].set_xlabel("Time (mm:ss)", fontsize=10)
    axes[-1].xaxis.set_major_formatter(mpl_ticker.FuncFormatter(_sec_to_mmss))

    title_parts = []
    if np.isfinite(mw):
        title_parts.append(f"$M_w$ = {mw:.1f}")
    if np.isfinite(dist):
        title_parts.append(f"$\\Delta$ = {dist:.0f}°")
    if np.isfinite(depth):
        title_parts.append(f"depth = {depth:.0f} km")
    if title_parts:
        fig.suptitle(", ".join(title_parts), fontsize=10, y=1.01)

    fig.tight_layout(rect=[0, 0, 1, 0.88])

    out = Path(args.output).with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand 2: contrast (success cases only; low-NCC figures use
# rollout_filtered_catalog_plots.py)
# ═══════════════════════════════════════════════════════════════════════════

def _add_contrast_args(sub):
    sub.add_argument("--item-good", type=int, nargs="+", default=None,
                     help="Specific high-NCC item index(es)")
    sub.add_argument("--n-examples", type=int, default=1,
                     help="Number of success examples to show (rows = 3×N)")
    sub.add_argument("--dist-center", type=float, default=35.0)
    sub.add_argument("--dist-band-width", type=float, default=5.0)
    sub.add_argument("--mag-center", type=float, default=5.5)
    sub.add_argument("--mag-band-width", type=float, default=0.3)
    sub.add_argument("--depth-center", type=float, default=40.0)
    sub.add_argument("--depth-band-width", type=float, default=20.0)
    sub.add_argument("--future-secs", type=float, default=240.0)


def _find_success_items(mdf, args, n):
    """Auto-select *n* high-NCC events from a matched parameter regime."""
    dc, dw = args.dist_center, args.dist_band_width
    mc, mw_bw = args.mag_center, args.mag_band_width
    zc, zw = args.depth_center, args.depth_band_width

    good_items: list = []
    for attempt in range(4):
        scale = 1.0 + 0.5 * attempt
        regime = mdf[
            (mdf["_dist"]  >= dc - dw * scale) & (mdf["_dist"]  <= dc + dw * scale) &
            (mdf["_mag"]   >= mc - mw_bw * scale) & (mdf["_mag"] <= mc + mw_bw * scale) &
            (mdf["_depth"] >= zc - zw * scale)  & (mdf["_depth"] <= zc + zw * scale)
        ]
        if attempt > 0 and not regime.empty:
            print(f"  Warning: widened regime by {scale:.1f}x to find candidates "
                  f"({len(regime)} items)")

        if len(good_items) < n:
            good_cand = regime[regime["ncc_global"] >= 0.95].sort_values(
                "ncc_global", ascending=False)
            exclude = set(good_items)
            for _, row in good_cand.iterrows():
                it = int(row["item"])
                if it not in exclude:
                    good_items.append(it)
                    exclude.add(it)
                    if len(good_items) >= n:
                        break

        if len(good_items) >= n:
            break

    if len(good_items) < n:
        print(f"ERROR: only found {len(good_items)}/{n} 'success' events (ncc >= 0.95). "
              "Try widening bands or use --item-good.", file=sys.stderr)
        sys.exit(1)
    return good_items


def cmd_contrast(args):
    mdf, ext_meta = _load_metrics_and_meta(args)
    n_ex = args.n_examples

    good_items = list(args.item_good) if args.item_good else []

    need_good = max(0, n_ex - len(good_items))
    if need_good > 0:
        auto_g = _find_success_items(mdf, args, n_ex)
        if need_good > 0:
            existing = set(good_items)
            for g in auto_g:
                if g not in existing and len(good_items) < n_ex:
                    good_items.append(g)

    good_items = good_items[:n_ex]

    future_secs = args.future_secs

    for ex_idx in range(n_ex):
        it = good_items[ex_idx]
        d, z, m = _get_event_meta(ext_meta, it, args.distance_col,
                                   args.depth_col, args.mag_col)
        ncc = float(mdf.loc[mdf["item"] == it, "ncc_global"].iloc[0])
        print(f"  example {ex_idx+1}: item {it}  Mw={m:.1f}, "
              f"Δ={d:.0f}°, depth={z:.0f} km, NCC={ncc:.4f}")

    # ---- Load model, run rollouts ----
    model, dataset, sr, ds_meta, device = _load_model_and_data(args)

    results = []
    for ex_idx in range(n_ex):
        it = good_items[ex_idx]
        results.append(
            _run_rollout(
                model, dataset, ds_meta, it, args.context_ratio,
                future_secs, sr, device, args.kernel_size,
            )
        )

    events = []
    for ex_idx in range(n_ex):
        it = good_items[ex_idx]
        y_full, y_fut, y_pred, L_ctx, L_fut, ctx_s, ps = results[ex_idx]
        d, z, m = _get_event_meta(ext_meta, it, args.distance_col,
                                   args.depth_col, args.mag_col)
        ncc = float(mdf.loc[mdf["item"] == it, "ncc_global"].iloc[0])
        events.append({
            "y_full": y_full,
            "y_pred": y_pred,
            "L_ctx": L_ctx,
            "L_fut": L_fut,
            "sr": sr,
            "ps_sec": ps,
            "title": (
                f"Successful  |  NCC = {ncc:.2f}\n"
                f"$M_w$ {m:.1f}, $\\Delta$ {d:.0f}°, depth {z:.0f} km"
            ),
        })

    out = Path(args.output).with_suffix(".pdf")
    save_contrast_style_figure(events, out, save_dpi=600)
    print(f"Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# Subcommand 3: multi-events (geometry + optional NCC filter, batch rollouts)
# ═══════════════════════════════════════════════════════════════════════════


def _add_multi_events_args(sub):
    sub.add_argument(
        "--items",
        type=str,
        default=None,
        help="Comma-separated item indices (skips auto filter; truncated to -n)",
    )
    sub.add_argument(
        "-n", "--max-events",
        type=int,
        default=6,
        help="Maximum number of events to plot",
    )
    sub.add_argument("--dist-lo", type=float, default=10.0)
    sub.add_argument("--dist-hi", type=float, default=90.0)
    sub.add_argument("--mag-lo", type=float, default=3.0)
    sub.add_argument("--mag-hi", type=float, default=7.0)
    sub.add_argument("--depth-lo", type=float, default=5.0)
    sub.add_argument("--depth-hi", type=float, default=150.0)
    sub.add_argument(
        "--ncc-min",
        type=float,
        default=None,
        help="If set, require ncc_global >= this value",
    )
    sub.add_argument(
        "--ncc-max",
        type=float,
        default=None,
        help="If set, require ncc_global <= this value",
    )
    sub.add_argument(
        "--sort",
        choices=("ncc_asc", "ncc_desc", "item"),
        default="ncc_asc",
        help="Order candidates before picking (ignored with --items)",
    )
    sub.add_argument(
        "--pick",
        choices=("head", "spread"),
        default="spread",
        help="head: first N after sort; spread: evenly spaced indices (diverse NCC)",
    )
    sub.add_argument("--future-secs", type=float, default=240.0)


def _select_multi_event_items(mdf: pd.DataFrame, args) -> list:
    """Return list of item ids (length <= args.max_events)."""
    nmax = args.max_events
    if args.items:
        raw = [int(x.strip()) for x in args.items.split(",") if x.strip()]
        if not raw:
            print("ERROR: --items is empty", file=sys.stderr)
            sys.exit(1)
        for it in raw:
            if mdf.loc[mdf["item"] == it].empty:
                print(f"ERROR: item {it} not found in metrics CSV", file=sys.stderr)
                sys.exit(1)
        if len(raw) > nmax:
            print(f"  Warning: truncating --items from {len(raw)} to {nmax}", file=sys.stderr)
        return raw[:nmax]

    filt = mdf.dropna(subset=["_dist", "_depth", "_mag", "item", "ncc_global"])
    filt = filt[
        (filt["_dist"] >= args.dist_lo) & (filt["_dist"] <= args.dist_hi) &
        (filt["_mag"] >= args.mag_lo) & (filt["_mag"] <= args.mag_hi) &
        (filt["_depth"] >= args.depth_lo) & (filt["_depth"] <= args.depth_hi)
    ]
    if args.ncc_min is not None:
        filt = filt[filt["ncc_global"] >= args.ncc_min]
    if args.ncc_max is not None:
        filt = filt[filt["ncc_global"] <= args.ncc_max]
    if filt.empty:
        print("ERROR: no items match geometry / NCC filters:", file=sys.stderr)
        print(f"  distance ∈ [{args.dist_lo}, {args.dist_hi}]°", file=sys.stderr)
        print(f"  magnitude ∈ [{args.mag_lo}, {args.mag_hi}]", file=sys.stderr)
        print(f"  depth ∈ [{args.depth_lo}, {args.depth_hi}] km", file=sys.stderr)
        if args.ncc_min is not None:
            print(f"  ncc_global >= {args.ncc_min}", file=sys.stderr)
        if args.ncc_max is not None:
            print(f"  ncc_global <= {args.ncc_max}", file=sys.stderr)
        sys.exit(1)

    if args.sort == "item":
        filt = filt.sort_values("item", ascending=True)
    else:
        asc = args.sort == "ncc_asc"
        filt = filt.sort_values("ncc_global", ascending=asc)

    n_take = min(nmax, len(filt))
    if args.pick == "head":
        sub = filt.head(n_take)
    else:
        if len(filt) <= n_take:
            sub = filt
        else:
            idx = np.linspace(0, len(filt) - 1, n_take, dtype=int)
            sub = filt.iloc[idx]
    return [int(x) for x in sub["item"].tolist()]


def cmd_multi_events(args):
    mdf, ext_meta = _load_metrics_and_meta(args)
    items = _select_multi_event_items(mdf, args)
    future_secs = args.future_secs

    for k, it in enumerate(items):
        d, z, m = _get_event_meta(ext_meta, it, args.distance_col,
                                   args.depth_col, args.mag_col)
        ncc = float(mdf.loc[mdf["item"] == it, "ncc_global"].iloc[0])
        print(f"  [{k+1}/{len(items)}] item {it}  Mw={m:.1f}, Δ={d:.0f}°, "
              f"depth={z:.0f} km, NCC={ncc:.4f}")

    model, dataset, sr, ds_meta, device = _load_model_and_data(args)
    rollouts = []
    for it in items:
        rollouts.append(
            _run_rollout(
                model, dataset, ds_meta, it, args.context_ratio,
                future_secs, sr, device, args.kernel_size,
            )
        )

    n_rows = len(items)
    fig_h = 1.05 * n_rows + 0.45
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, fig_h), sharex="row",
                             squeeze=False)
    fig.subplots_adjust(hspace=0.32, wspace=0.20)

    for row, it in enumerate(items):
        y_full, y_fut, y_pred, L_ctx, L_fut, ctx_s, ps = rollouts[row]
        ctx_end_s = L_ctx / sr
        pred_end_s = ctx_end_s + L_fut / sr
        t_full = np.arange(len(y_full)) / sr
        t_pred = np.arange(L_ctx, L_ctx + L_fut) / sr
        s_arr = ps if np.isfinite(ps) else None

        d, z, m = _get_event_meta(ext_meta, it, args.distance_col,
                                   args.depth_col, args.mag_col)
        ncc = float(mdf.loc[mdf["item"] == it, "ncc_global"].iloc[0])

        for ch_idx, ch_label in enumerate(CH_LABELS):
            ax = axes[row, ch_idx]

            ax.axvspan(0, ctx_end_s, color=C_SHADE, alpha=0.45, lw=0, zorder=0)

            full_end = min(int(pred_end_s * sr), len(y_full))
            ax.plot(t_full[:full_end], y_full[:full_end, ch_idx],
                    color=C_TRUTH, lw=0.7, alpha=0.85, zorder=2)
            ax.plot(t_pred, y_pred[:L_fut, ch_idx],
                    color=C_PRED, lw=0.9, ls="--", alpha=0.88, zorder=3)

            ax.axvline(0, color=C_P, lw=0.8, alpha=0.7, zorder=5)
            if s_arr is not None and s_arr > 0:
                ax.axvline(s_arr, color=C_S, lw=0.8, alpha=0.7, zorder=5)

            peak = float(np.abs(y_full[:full_end, ch_idx]).max()) or 1.0
            ax.set_ylim(-peak * 1.15, peak * 1.15)

            ax.xaxis.set_major_formatter(mpl_ticker.FuncFormatter(_sec_to_mmss))
            if row == 0:
                ax.set_title(ch_label, fontsize=11, fontweight="bold")
            if row == n_rows - 1:
                ax.set_xlabel("Time (mm:ss)", fontsize=10)

        axes[row, 0].text(
            0.02, 0.95, f"item {it}", transform=axes[row, 0].transAxes,
            fontsize=8.5, fontweight="bold", va="top", ha="left",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5),
        )
        meta_str = (
            f"$M_w$ {m:.1f}, $\\Delta$ {d:.0f}°\n"
            f"depth {z:.0f} km, NCC {ncc:.2f}"
        )
        axes[row, 0].text(
            0.98, 0.95, meta_str, transform=axes[row, 0].transAxes,
            fontsize=7, va="top", ha="right", color="#555555",
            bbox=dict(facecolor="white", edgecolor="#cccccc", alpha=0.85,
                      pad=1.5, boxstyle="round,pad=0.25"),
        )

        if row < n_rows - 1:
            for ch_idx in range(3):
                sep_ax = axes[row, ch_idx]
                sep_ax.spines["bottom"].set_visible(True)
                sep_ax.spines["bottom"].set_linewidth(0.8)
                sep_ax.spines["bottom"].set_color("#999999")

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = Path(args.output).with_suffix(".pdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser():
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--checkpoint", type=str, required=True,
                        help="Path to GPTLightning checkpoint")
    parent.add_argument("--metadata-csv", type=str, required=True,
                        help="Test-set metadata CSV")
    parent.add_argument("--metrics-csv", type=str, required=True,
                        help="Per-item metrics CSV from testset_rollout.py")
    parent.add_argument("--output", type=str, required=True,
                        help="Output figure path (PNG/PDF)")
    parent.add_argument("--data-dir", dest="data_dir", type=str,
                        default="/mnt/d/waleed/Seismology/simulation/clean/SobolDataset/PaperTestSet/")
    parent.add_argument("--kernel-size", dest="kernel_size", type=int, default=16)
    parent.add_argument("--num-tokens", dest="num_tokens", type=int, default=320)
    parent.add_argument("--context-ratio", dest="context_ratio", type=float, default=2.0)
    parent.add_argument("--max-future-secs", type=float, default=240.0)
    parent.add_argument("--seed", type=int, default=42)
    parent.add_argument("--distance-col", default="distance_deg")
    parent.add_argument("--depth-col", default="src_depth_km")
    parent.add_argument("--mag-col", default="Mw")

    top = argparse.ArgumentParser(
        description="Generate publication figures for the SeismoGPT paper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subs = top.add_subparsers(dest="command", required=True)

    s1 = subs.add_parser("horizon-series", parents=[parent],
                         help="Same event at 60/120/240 s horizons (Fig 7)",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_horizon_args(s1)

    s2 = subs.add_parser("contrast", parents=[parent],
                         help="Successful forecasts only, stacked (Fig 8)",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_contrast_args(s2)

    s3 = subs.add_parser("multi-events", parents=[parent],
                         help="Batch Z/N/E rollouts filtered by Δ, Mw, depth (+ optional NCC)",
                         formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_multi_events_args(s3)

    return top


def main():
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "horizon-series": cmd_horizon_series,
        "contrast":       cmd_contrast,
        "multi-events":   cmd_multi_events,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

# How to run:
# Fig 7 — Horizon series (auto-selects a representative event):
# python make_paper_figures.py horizon-series \
#    --checkpoint phase1/epoch\=12-step\=633750.ckpt \
#    --metadata-csv /mnt/d/waleed/Seismology/simulation/clean/SobolDataset/PaperTestSet/metadata_data_3.csv \
#    --metrics-csv determinstic_b.csv \
#    --output figures/fig7_horizons.pdf

# Fig 8 — Contrast (high-NCC successes only; failure panels: rollout_filtered_catalog_plots.py):
# python make_paper_figures.py contrast \
#    --checkpoint phase1/epoch\=12-step\=633750.ckpt \
#    --metadata-csv /mnt/d/waleed/Seismology/simulation/clean/SobolDataset/PaperTestSet/metadata_data_3.csv \
#    --metrics-csv determinstic_b.csv \
#    --output figures/fig8_contrast.png

# Batch rollouts (filter by geometry / NCC, same metrics CSV as efficient_rollout):
# python make_paper_figures.py multi-events \
#    --checkpoint phase1/epoch\=12-step\=633750.ckpt \
#    --metadata-csv /mnt/d/waleed/Seismology/simulation/clean/SobolDataset/PaperTestSet/metadata_data_3.csv \
#    --metrics-csv determinstic_b.csv \
#    --output figures/multi_events.pdf \
#    --dist-lo 30 --dist-hi 55 -n 5 --pick spread