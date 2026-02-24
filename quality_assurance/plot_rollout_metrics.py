#!/usr/bin/env python3
"""
Publication-ready rollout metrics plots.

Every figure is saved as its own file.  No multi-panel canvases.

Usage:
  python plot_rollout_metrics.py --csv rollout_metrics.csv --output-dir metrics_plots
  python plot_rollout_metrics.py --csv rollout_metrics.csv \
      --metadata-csv testset/metadata.csv --output-dir metrics_plots
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
from scipy.stats import gaussian_kde

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "mathtext.fontset":     "dejavuserif",
    "font.size":            9,
    "axes.titlesize":       10,
    "axes.labelsize":       9,
    "legend.fontsize":      8,
    "xtick.labelsize":      8,
    "ytick.labelsize":      8,
    "figure.titlesize":     10,
    "lines.linewidth":      1.0,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.7,
    "axes.grid":            True,
    "grid.color":           "#cccccc",
    "grid.linewidth":       0.4,
    "grid.alpha":           0.5,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.width":    0.7,
    "ytick.major.width":    0.7,
    "xtick.major.size":     3,
    "ytick.major.size":     3,
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "savefig.dpi":          300,
    "figure.dpi":           150,
})

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
# Qualitative palette that prints well in greyscale and is colourblind-safe
CH_COLORS = {
    "Z": "#2c3e7a",   # slate blue
    "N": "#b5320a",   # crimson
    "E": "#2a7a3b",   # forest green
}
HIST_COLORS = [
    "#2c3e7a",   # blue
    "#b5320a",   # crimson
    "#2a7a3b",   # green
    "#7a5c2c",   # brown
    "#5c2c7a",   # purple
    "#2c7a6e",   # teal
]
HEXBIN_CMAP = "Blues"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHANNEL_LABELS = ["Z", "N", "E"]

GLOBAL_METRICS = [
    ("snr_db_global",       "SNR (dB)",            "Global SNR"),
    ("ncc_global",          "NCC",                  "Global NCC"),
    ("skill_global",        "Skill score",          "Global Forecast Skill"),
    ("robust_snr_db_global","Robust SNR (dB)",      "Global Robust SNR"),
    ("psd_logl2_global",    "PSD log-L² error",     "Global PSD log-L²"),
    ("psd_rel_global",      "PSD relative error",   "Global PSD Relative Error"),
]

PER_CH_GROUPS = [
    ("snr_db",       "SNR (dB)",         "SNR"),
    ("ncc",          "NCC",              "NCC"),
    ("skill",        "Skill score",      "Forecast Skill"),
    ("robust_snr_db","Robust SNR (dB)",  "Robust SNR"),
    ("psd_logl2",    "PSD log-L² error", "PSD log-L²"),
    ("psd_rel",      "PSD relative error","PSD Relative Error"),
]

META_X_COLS = [
    ("distance_deg",  "Epicentral distance (°)"),
    ("src_depth_km",  "Source depth (km)"),
    ("Mw",            r"Moment magnitude $M_w$"),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stat_box(ax, x: np.ndarray) -> None:
    """Add a small stats annotation inside the axes (top-right corner)."""
    txt = (
        f"$n$ = {len(x)}\n"
        f"$\\mu$ = {x.mean():.3f}\n"
        f"$\\tilde{{x}}$ = {np.median(x):.3f}\n"
        f"$\\sigma$ = {x.std():.3f}"
    )
    ax.text(
        0.97, 0.97, txt,
        transform=ax.transAxes,
        fontsize=7.5, va="top", ha="right",
        linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#aaaaaa", alpha=0.9, linewidth=0.6),
    )


def _kde_overlay(ax, x: np.ndarray, color: str, bins: int) -> None:
    """Overlay a smooth KDE curve scaled to histogram counts."""
    if len(x) < 5:
        return
    try:
        kde = gaussian_kde(x, bw_method="scott")
        xgrid = np.linspace(x.min(), x.max(), 300)
        # Scale KDE to match histogram (density=False)
        bin_width = (x.max() - x.min()) / bins
        ax.plot(xgrid, kde(xgrid) * len(x) * bin_width,
                color=color, lw=1.4, zorder=5)
    except Exception:
        pass


def _histogram(
    ax,
    series: pd.Series,
    xlabel: str,
    title: str,
    color: str,
    bins: int,
) -> bool:
    """Draw a single publication-ready histogram with KDE, median, mean lines."""
    x = series.dropna().values
    if len(x) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888888")
        ax.set_title(title)
        return False

    ax.hist(x, bins=bins, color=color, alpha=0.55,
            edgecolor="white", linewidth=0.4, zorder=3)
    _kde_overlay(ax, x, color, bins)

    med  = float(np.median(x))
    mean = float(np.mean(x))
    ax.axvline(med,  color="#222222", ls="--",  lw=1.1, zorder=6,
               label=f"Median = {med:.3f}")
    ax.axvline(mean, color="#888888", ls=":",   lw=1.0, zorder=6,
               label=f"Mean = {mean:.3f}")

    ax.set_xlabel(xlabel, labelpad=3)
    ax.set_ylabel("Count", labelpad=3)
    ax.set_title(title, pad=5)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#aaaaaa",
              fontsize=7.5, handlelength=1.5)
    ax.yaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True, nbins=5))
    _stat_box(ax, x)
    return True


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  {path}")


# ---------------------------------------------------------------------------
# Plot generators — each returns one figure → one file
# ---------------------------------------------------------------------------

def plot_global_metric(
    df: pd.DataFrame,
    col: str,
    xlabel: str,
    title: str,
    out_path: Path,
    bins: int,
    dpi: int,
    color: str,
) -> None:
    if col not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    _histogram(ax, df[col], xlabel, title, color, bins)
    fig.tight_layout()
    _save(fig, out_path, dpi)


def plot_per_channel_overlay(
    df: pd.DataFrame,
    base_col: str,
    xlabel: str,
    title: str,
    out_path: Path,
    bins: int,
    dpi: int,
) -> None:
    """Three channels overlaid on a single histogram axes — one file."""
    cols = {c: f"{base_col}_{c}" for c in CHANNEL_LABELS}
    available = {c: col for c, col in cols.items() if col in df.columns}
    if not available:
        return

    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for ch, col in available.items():
        x = df[col].dropna().values
        if len(x) == 0:
            continue
        color = CH_COLORS[ch]
        ax.hist(x, bins=bins, color=color, alpha=0.40,
                edgecolor="white", linewidth=0.3, label=ch, zorder=3)
        _kde_overlay(ax, x, color, bins)
        med = float(np.median(x))
        ax.axvline(med, color=color, ls="--", lw=1.1, zorder=6,
                   label=f"{ch} median = {med:.3f}")

    ax.set_xlabel(xlabel, labelpad=3)
    ax.set_ylabel("Count", labelpad=3)
    ax.set_title(title, pad=5)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="#aaaaaa",
              fontsize=7, handlelength=1.5, ncol=2)
    ax.yaxis.set_major_locator(mpl_ticker.MaxNLocator(integer=True, nbins=5))

    fig.tight_layout()
    _save(fig, out_path, dpi)


def plot_per_channel_individual(
    df: pd.DataFrame,
    base_col: str,
    xlabel: str,
    title_base: str,
    out_dir: Path,
    bins: int,
    dpi: int,
) -> None:
    """One histogram file per channel (Z, N, E)."""
    for ch in CHANNEL_LABELS:
        col = f"{base_col}_{ch}"
        if col not in df.columns:
            continue
        color = CH_COLORS[ch]
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        _histogram(ax, df[col], xlabel, f"{title_base} — {ch}", color, bins)
        fig.tight_layout()
        fname = out_dir / f"metric_{base_col}_{ch}.png"
        _save(fig, fname, dpi)


def plot_metric_vs_meta_single(
    df: pd.DataFrame,
    metric_col: str,
    ylabel: str,
    x_col: str,
    xlabel: str,
    title: str,
    out_path: Path,
    dpi: int,
) -> None:
    """One hexbin scatter — one file."""
    if metric_col not in df.columns or x_col not in df.columns:
        return
    valid = df[[metric_col, x_col]].dropna()
    if len(valid) < 5:
        return

    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    hb = ax.hexbin(
        valid[x_col], valid[metric_col],
        gridsize=35, cmap=HEXBIN_CMAP,
        mincnt=1, linewidths=0.2,
    )
    cb = fig.colorbar(hb, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label("Count", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # Thin running-median line
    try:
        xv = valid[x_col].values
        yv = valid[metric_col].values
        order = np.argsort(xv)
        xsort, ysort = xv[order], yv[order]
        w = max(1, len(xsort) // 15)
        from numpy.lib.stride_tricks import sliding_window_view
        ymed = np.array([
            np.median(ysort[max(0, i - w // 2): i + w // 2 + 1])
            for i in range(len(ysort))
        ])
        ax.plot(xsort, ymed, color="#b5320a", lw=1.2, ls="--",
                zorder=5, label="Running median")
        ax.legend(frameon=True, framealpha=0.9, edgecolor="#aaaaaa",
                  fontsize=7.5, handlelength=1.5)
    except Exception:
        pass

    ax.set_xlabel(xlabel, labelpad=3)
    ax.set_ylabel(ylabel, labelpad=3)
    ax.set_title(title, pad=5)
    #ax.text(0.02, 0.97, f"$n$ = {len(valid)}", transform=ax.transAxes, fontsize=7.5, va="top")

    fig.tight_layout()
    _save(fig, out_path, dpi)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Rollout metrics plots (one figure per file)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", required=True, help="Metrics CSV from testset_rollout.py")
    p.add_argument("--metadata-csv", default=None,
                   help="Test-set metadata CSV; enables metric vs distance/depth/Mw plots")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: same folder as --csv)")
    p.add_argument("--bins",   type=int, default=30,  help="Histogram bins")
    p.add_argument("--dpi",    type=int, default=300, help="Figure DPI")
    p.add_argument("--no-per-channel-individual", action="store_true",
                   help="Skip individual per-channel files (only save overlay)")
    return p.parse_args()


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows  ·  {len(df.columns)} columns")

    # ---- Merge metadata if given -------------------------------------------
    if args.metadata_csv:
        meta_path = Path(args.metadata_csv)
        if not meta_path.exists():
            print(f"  Warning: metadata CSV not found: {meta_path}, skipping.")
        elif "item" not in df.columns:
            print("  Warning: no 'item' column; cannot merge metadata.")
        else:
            metadata = pd.read_csv(meta_path)
            for col, _ in META_X_COLS:
                df[col] = df["item"].apply(
                    lambda i: metadata.iloc[int(i)][col]
                    if col in metadata.columns and int(i) < len(metadata)
                    else np.nan
                )
            print(f"  Merged metadata: {meta_path.name}")

    out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    bins = args.bins
    dpi  = args.dpi

    # =========================================================================
    # 1. Global metric histograms  (one file each)
    # =========================================================================
    print("\n── Global metrics ──────────────────────────────────────────────")
    for i, (col, xlabel, title) in enumerate(GLOBAL_METRICS):
        color = HIST_COLORS[i % len(HIST_COLORS)]
        out_path = out_dir / f"global_{col}.png"
        plot_global_metric(df, col, xlabel, title, out_path, bins, dpi, color)

    # =========================================================================
    # 2. Per-channel histograms
    #    (a) overlay: all three channels on one axes → one file per metric
    #    (b) individual: one file per channel per metric  (unless --no-per-channel-individual)
    # =========================================================================
    print("\n── Per-channel metrics ─────────────────────────────────────────")
    for base_col, xlabel, title_base in PER_CH_GROUPS:
        # Overlay
        out_overlay = out_dir / f"perchannel_overlay_{base_col}.png"
        plot_per_channel_overlay(df, base_col, xlabel, title_base, out_overlay, bins, dpi)
        # Individual
        if not args.no_per_channel_individual:
            plot_per_channel_individual(df, base_col, xlabel, title_base,
                                        out_dir, bins, dpi)

    # =========================================================================
    # 3. Metric vs metadata scatter/hexbin  (one file per metric × meta variable)
    # =========================================================================
    meta_x_present = [c for c, _ in META_X_COLS if c in df.columns]
    if meta_x_present:
        print("\n── Metric vs metadata ──────────────────────────────────────────")
        scatter_metrics = [
            ("snr_db_global",       "SNR (dB)",         "SNR"),
            ("ncc_global",          "NCC",              "NCC"),
            ("skill_global",        "Skill score",      "Skill"),
            ("robust_snr_db_global","Robust SNR (dB)",  "Robust SNR"),
        ]
        for m_col, ylabel, m_short in scatter_metrics:
            for x_col, xlabel in META_X_COLS:
                if x_col not in df.columns:
                    continue
                title  = f"{m_short} vs {xlabel.split('(')[0].strip()}"
                fname  = out_dir / f"scatter_{m_col}_vs_{x_col}.png"
                plot_metric_vs_meta_single(
                    df, m_col, ylabel, x_col, xlabel, title, fname, dpi
                )

    print(f"\nDone — all figures saved in  {out_dir}")


if __name__ == "__main__":
    main()