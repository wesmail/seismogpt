#!/usr/bin/env python3
"""
Publication-ready global metrics violin plots from rollout CSV files.

Same file-discovery and config-parsing logic as plot_global_metrics_from_csv.py,
but uses violin plots instead of box plots.

Each violin:
  • KDE body (full distribution shape)
  • Inner IQR bar (thick, white)
  • Median tick (white)
  • Mean diamond (amber)
  • Clipped-outlier count annotations (+k / −k) when y-limits hide extreme tails
  • n= sample count below each violin

One figure per (metric, channel):
  NCC_{Z,N,E}  and  PSD_logl2_{Z,N,E}  — 6 individual files.

Usage:
  python plot_violin_metrics.py --csv-dir . --output-dir violin_plots
  python plot_violin_metrics.py --csv a.csv b.csv --output-dir violin_plots
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import numpy as np
import pandas as pd
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
    "xtick.labelsize":      20,
    "ytick.labelsize":      20,
    "figure.titlesize":     10,
    "lines.linewidth":      0.9,
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
METRIC_COLORS = {
    "ncc":       "#2c3e7a",   # slate blue
    "psd_logl2": "#b5320a",   # crimson
}
MEAN_COLOR   = "#f0a500"      # amber diamond
INNER_COLOR  = "#ffffff"      # white IQR bar / median tick
FLIER_COLOR  = "#888888"      # grey outlier count text

CHANNELS = ["Z", "N", "E"]
METRICS  = ["ncc", "psd_logl2"]

METRIC_META = {
    "ncc": {
        "ylabel":    "NCC",
        "title_fmt": "Channel {ch}",
        "ref_line":  0.0,
        "ref_label": "NCC = 0",
    },
    "psd_logl2": {
        "ylabel":    "PSD log-L² error",
        "title_fmt": "Channel {ch}",
        "ref_line":  None,
        "ref_label": None,
    },
}

# ---------------------------------------------------------------------------
# Filename parsing  (identical to box-plot script)
# ---------------------------------------------------------------------------

def parse_context_future(csv_path: Path) -> Tuple[Optional[int], Optional[int]]:
    m = re.search(r"_(\d+)_(\d+)$", csv_path.stem)
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def collect_configs(csv_paths: List[Path]):
    parsed = []
    for p in csv_paths:
        ctx, fut = parse_context_future(p)
        sort_key = (ctx if ctx is not None else 9999,
                    fut if fut is not None else 9999)
        label = f"{ctx} s / {fut} s" if ctx is not None else p.stem
        parsed.append((sort_key, label, p))
    parsed.sort(key=lambda x: x[0])
    return [
        (chr(ord("A") + i) if i < 26 else f"A{i}", label, path)
        for i, (_, label, path) in enumerate(parsed)
    ]


# ---------------------------------------------------------------------------
# Robust y-limits  (same logic as box-plot script)
# ---------------------------------------------------------------------------

def _robust_ylim(
    data_list: List[np.ndarray],
    ref_line:  Optional[float],
    p_lo: float = 2,
    p_hi: float = 98,
    pad_frac: float = 0.15,
) -> Tuple[float, float]:
    all_vals = np.concatenate([v for v in data_list if len(v) > 0])
    if len(all_vals) == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(all_vals, p_lo), np.percentile(all_vals, p_hi)
    span   = hi - lo if hi > lo else 1.0
    pad    = pad_frac * span
    y_lo   = lo - pad
    y_hi   = hi + pad
    if ref_line is not None:
        y_lo = min(y_lo, ref_line - 0.05)
    return float(y_lo), float(y_hi)


# ---------------------------------------------------------------------------
# Core violin drawing  (manual KDE so we can clip to y-limits)
# ---------------------------------------------------------------------------

def _draw_violin(
    ax,
    vals: np.ndarray,
    pos: float,
    y_lo: float,
    y_hi: float,
    color: str,
    width: float = 0.7,
    bw_method: str = "scott",
) -> None:
    """
    Draw a single violin at x=pos clipped to [y_lo, y_hi].
    The KDE is evaluated only within the visible range so the body never
    bleeds outside the axes boundaries.
    """
    if len(vals) < 4:
        return

    # KDE evaluated on a fine grid clipped to the axes range
    y_grid = np.linspace(y_lo, y_hi, 400)
    try:
        kde  = gaussian_kde(vals, bw_method=bw_method)
        dens = kde(y_grid)
    except Exception:
        return

    dens = np.maximum(dens, 0)
    max_dens = dens.max()
    if max_dens == 0:
        return

    # Normalise density so half-width == width/2
    half_w = (dens / max_dens) * (width / 2)

    x_left  = pos - half_w
    x_right = pos + half_w

    # Filled body
    ax.fill_betweenx(y_grid, x_left, x_right,
                     color=color, alpha=0.35, linewidth=0, zorder=2)
    # Outline
    ax.plot(x_left,  y_grid, color=color, lw=0.7, alpha=0.80, zorder=3)
    ax.plot(x_right, y_grid, color=color, lw=0.7, alpha=0.80, zorder=3)


def _draw_inner_stats(
    ax,
    vals: np.ndarray,
    pos: float,
    y_lo: float,
    y_hi: float,
) -> None:
    """
    Overlay the inner statistics on a violin:
      • Thick white bar spanning the IQR
      • White tick for the median
      • Amber diamond for the mean (clipped to visible range)
    """
    if len(vals) == 0:
        return

    q1, med, q3 = np.percentile(vals, [25, 50, 75])
    mean_val     = float(np.mean(vals))

    # Clip to visible range
    q1_c    = np.clip(q1,       y_lo, y_hi)
    q3_c    = np.clip(q3,       y_lo, y_hi)
    med_c   = np.clip(med,      y_lo, y_hi)
    mean_c  = np.clip(mean_val, y_lo, y_hi)

    # IQR bar — thin vertical white line
    ax.plot([pos, pos], [q1_c, q3_c],
            color=INNER_COLOR, lw=3.5, solid_capstyle="round", zorder=5)

    # Median tick — short horizontal white bar
    tick_hw = 0.06
    ax.plot([pos - tick_hw, pos + tick_hw], [med_c, med_c],
            color=INNER_COLOR, lw=2.0, solid_capstyle="round", zorder=6)

    # Mean diamond
    ax.scatter([pos], [mean_c],
               marker="D", s=22, color=MEAN_COLOR,
               zorder=7, linewidths=0.4, edgecolors="white")


def _draw_mean_median_labels(
    ax,
    vals: np.ndarray,
    pos: float,
    y_lo: float,
    y_hi: float,
    decimals: int = 3,
) -> None:
    """Draw mean (μ) on first line, median (x̄) on second line below with spacing."""
    if len(vals) == 0:
        return
    mean_val = float(np.mean(vals))
    med_val = float(np.median(vals))
    dy = y_hi - y_lo
    line_height = 0.07 * dy if dy > 0 else 0.0  # separation between mean and median
    y_mean = y_lo + 0.09 * dy if dy > 0 else y_lo
    y_median = y_mean - line_height
    ax.text(pos, y_mean, r"$\mu$ = " + f"{mean_val:.{decimals}f}",
            ha="center", va="bottom", fontsize=18, color="#333333", zorder=8)
    ax.text(pos, y_median, r"$\bar{x}$ = " + f"{med_val:.{decimals}f}",
            ha="center", va="bottom", fontsize=18, color="#333333", zorder=8)


# ---------------------------------------------------------------------------
# Annotation helpers  (n=, clipped-outlier counts)
# ---------------------------------------------------------------------------

def _annotate_violin(
    ax,
    vals: np.ndarray,
    pos: float,
    y_lo: float,
    y_hi: float,
) -> None:
    """n= below axis; +k / −k for values outside the visible y-range."""
    if len(vals) == 0:
        return

    # n= — uses get_xaxis_transform() so y is in axes fraction
    #ax.text(
    #    pos, -0.08,
    #    f"$n$={len(vals):,}",
    #    ha="center", va="top",
    #    fontsize=6.5, color="#555555",
    #    transform=ax.get_xaxis_transform(),
    #)

    # Values outside visible range
    n_above = int(np.sum(vals > y_hi))
    n_below = int(np.sum(vals < y_lo))
    if n_above > 0:
        ax.text(pos, y_hi, f"+{n_above}",
                ha="center", va="bottom",
                fontsize=6.5, color=FLIER_COLOR, style="italic",
                clip_on=False, zorder=8)
    if n_below > 0:
        ax.text(pos, y_lo, f"−{n_below}",
                ha="center", va="top",
                fontsize=6.5, color=FLIER_COLOR, style="italic",
                clip_on=False, zorder=8)


# ---------------------------------------------------------------------------
# Config legend table  (same as box-plot script)
# ---------------------------------------------------------------------------

def _add_legend_table(fig, configs: List[Tuple[str, str, Path]]) -> None:
    parts = [f"$\\mathbf{{{letter}}}$: {label}" for letter, label, _ in configs]
    mid   = (len(parts) + 1) // 2
    row1  = "    ".join(parts[:mid])
    row2  = "    ".join(parts[mid:])
    text  = row1 + ("\n" + row2 if row2 else "")
    fig.text(0.5, 0.04, text,
             ha="center", va="bottom",
             fontsize=7.5, color="#333333", linespacing=1.6)


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------

def plot_violin_channel(
    config_dfs: List[Tuple[str, str, pd.DataFrame]],
    col: str,
    metric_key: str,
    channel: str,
    out_path: Path,
) -> None:
    """One publication-ready violin figure for (metric, channel)."""
    meta      = METRIC_META[metric_key]
    color     = METRIC_COLORS[metric_key]
    positions = np.arange(len(config_dfs))
    letters   = [c[0] for c in config_dfs]

    data_list = [df[col].dropna().values for _, _, df in config_dfs]

    # Figure sizing — generous so violins never crowd the legend
    fig_w = max(4.5, 1.6 * len(config_dfs) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, 3.5))

    # Robust y-limits computed before drawing
    y_lo, y_hi = _robust_ylim(data_list, meta["ref_line"])
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-0.6, len(config_dfs) - 0.4)

    # Reference line
    if meta["ref_line"] is not None:
        ax.axhline(meta["ref_line"], color="#666666", ls=":", lw=0.9,
                   zorder=1, label=meta["ref_label"])

    # Draw violins + inner stats + mean/median numbers
    VIOLIN_WIDTH = min(0.7, 0.9 / max(len(config_dfs), 1))
    decimals = 2 if metric_key == "psd_logl2" else 3
    for pos, vals in zip(positions, data_list):
        _draw_violin(ax, vals, pos, y_lo, y_hi, color, width=VIOLIN_WIDTH)
        _draw_inner_stats(ax, vals, pos, y_lo, y_hi)
        _draw_mean_median_labels(ax, vals, pos, y_lo, y_hi, decimals=decimals)

    # Axes formatting (no legend)
    ax.set_xticks(positions)
    ax.set_xticklabels(letters, fontsize=20, fontweight="bold")
    ax.set_xlabel("Configuration", labelpad=12, fontsize=18)
    ax.set_ylabel(meta["ylabel"], labelpad=3, fontsize=18)
    ax.set_title(meta["title_fmt"].format(ch=channel), pad=5, loc="left", fontsize=18)
    ax.yaxis.set_major_locator(mpl_ticker.MaxNLocator(nbins=6))

    # No config legend on figure; x-axis label "Configuration" only
    fig.subplots_adjust(bottom=0.12, top=0.92, left=0.14, right=0.97)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Violin plots from rollout CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv-dir",    default=None,
                   help="Directory to glob rollout_metrics*.csv from")
    p.add_argument("--csv",        nargs="*", default=None,
                   help="Explicit CSV paths (overrides --csv-dir)")
    p.add_argument("--output-dir", default="violin_plots")
    p.add_argument("--pattern",    default="rollout_metrics*.csv",
                   help="Glob pattern when using --csv-dir")
    p.add_argument("--no-summary", action="store_true",
                   help="Skip writing the summary CSV")
    return p.parse_args()


def main():
    args = parse_args()

    if args.csv:
        csv_paths = [Path(f) for f in args.csv]
    elif args.csv_dir:
        csv_paths = sorted(Path(args.csv_dir).glob(args.pattern))
    else:
        csv_paths = sorted(Path(".").glob(args.pattern))

    if not csv_paths:
        print("No CSV files found. Use --csv-dir or --csv.")
        return

    configs = collect_configs(csv_paths)
    print("Configurations found:")
    for letter, label, path in configs:
        print(f"  {letter}: {label:20s}  ←  {path.name}")

    config_dfs = [(letter, label, pd.read_csv(path))
                  for letter, label, path in configs]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n── Violin plots ({len(METRICS)} metrics × {len(CHANNELS)} channels) ──────────")
    for metric_key in METRICS:
        for ch in CHANNELS:
            col = f"{metric_key}_{ch}"
            if not any(col in df.columns for _, _, df in config_dfs):
                print(f"  [skip] {col} — column not found in any CSV")
                continue
            plot_violin_channel(
                config_dfs=config_dfs,
                col=col,
                metric_key=metric_key,
                channel=ch,
                out_path=out_dir / f"violin_{col}.png",
            )

    if not args.no_summary:
        rows = []
        for letter, label, df in config_dfs:
            row = {"config": letter, "label": label, "n": len(df)}
            for mk in METRICS:
                for ch in CHANNELS:
                    c = f"{mk}_{ch}"
                    if c in df.columns:
                        row[f"{c}_mean"]   = df[c].mean()
                        row[f"{c}_median"] = df[c].median()
                        row[f"{c}_std"]    = df[c].std()
            rows.append(row)
        summary_path = out_dir / "summary.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        print(f"\n  Summary table → {summary_path}")

    print(f"\nDone — all figures in  {out_dir.absolute()}")


if __name__ == "__main__":
    main()