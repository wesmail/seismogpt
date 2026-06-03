#!/usr/bin/env python3
"""
Publication-ready grouped box-plot figure from rollout CSV files.

Produces a single 1x3 panel figure (NCC | SNR | PSD) with grouped box plots:
  - X-axis per panel: channels Z, N, E
  - Within each channel group: one box per configuration (A, B, C, …), color-coded
  - Shared legend at top

Usage:
  python plot_global_metrics_from_csv.py --csv-dir . --output-dir metrics_plots
  python plot_global_metrics_from_csv.py --csv a.csv b.csv --output metrics_combined.pdf
"""

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family":          "serif",
    "font.serif":           ["DejaVu Serif", "Times New Roman", "Times", "serif"],
    "mathtext.fontset":     "dejavuserif",
    "font.size":            8,
    "axes.titlesize":       9,
    "axes.labelsize":       8,
    "legend.fontsize":      7.5,
    "xtick.labelsize":      8,
    "ytick.labelsize":      7,
    "figure.titlesize":     9,
    "lines.linewidth":      0.9,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.linewidth":       0.5,
    "axes.grid":            True,
    "grid.color":           "#d5d5d5",
    "grid.linewidth":       0.25,
    "grid.alpha":           0.6,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.major.width":    0.5,
    "ytick.major.width":    0.5,
    "xtick.major.size":     2.5,
    "ytick.major.size":     2.5,
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "savefig.dpi":          600,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.02,
    "figure.dpi":           150,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
})

# ---------------------------------------------------------------------------
# Colour palette — colorblind-friendly, 3-config default
# ---------------------------------------------------------------------------
CONFIG_COLORS = [
    "#4477AA",   # blue
    "#EE6677",   # rose/red # "#EE7733",   # orange (replaces rose)
    "#228833",   # green    # "#66CCEE",   # cyan (replaces green)
    "#CCBB44",   # yellow
    "#66CCEE",   # cyan
    "#AA3377",   # purple
    "#BBBBBB",   # grey
]


MEAN_COLOR = "#222222"

CHANNELS = ["Z", "N", "E"]
METRICS  = ["ncc", "snr_db", "psd_logl2"]

METRIC_META = {
    "ncc": {
        "ylabel":   "NCC",
        "ref_line": None,
        "y_max":    1.02,
    },
    "snr_db": {
        "ylabel":   "SRR",
        "ref_line": 0.0,
    },
    "psd_logl2": {
        "ylabel":   r"PSD log-$L^2$ error",
        "ref_line": None,
    },
}

# ---------------------------------------------------------------------------
# Filename parsing
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
        label = f"{ctx}s / {fut}s" if ctx is not None else p.stem
        parsed.append((sort_key, label, p))
    parsed.sort(key=lambda x: x[0])
    return [
        (chr(ord("A") + i) if i < 26 else f"A{i}", label, path)
        for i, (_, label, path) in enumerate(parsed)
    ]


# ---------------------------------------------------------------------------
# Core: draw one group of boxes at a channel position
# ---------------------------------------------------------------------------

def _draw_grouped_boxes(
    ax,
    data_per_config: List[np.ndarray],
    group_center: float,
    colors: List[str],
    box_width: float = 0.22,
    gap: float = 0.02,
) -> None:
    """
    Draw N side-by-side box plots centred at *group_center*.

    Each box:  IQR body, median line, mean diamond, 1.5*IQR whiskers, no fliers.
    """
    n = len(data_per_config)
    total_w = n * box_width + (n - 1) * gap
    x_start = group_center - total_w / 2 + box_width / 2

    for i, vals in enumerate(data_per_config):
        if len(vals) == 0:
            continue
        x = x_start + i * (box_width + gap)
        color = colors[i % len(colors)]

        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        iqr = q3 - q1
        whisker_lo = max(vals.min(), q1 - 1.5 * iqr)
        whisker_hi = min(vals.max(), q3 + 1.5 * iqr)
        mean_val = float(np.mean(vals))

        hw = box_width / 2

        # In _draw_grouped_boxes, add hatch patterns:
        HATCHES = ["", "//", "\\\\", "xx", "..", "oo"]        
        # Box (IQR)
        rect = plt.Rectangle(
            (x - hw, q1), box_width, iqr,
            facecolor=color, edgecolor=color,
            alpha=0.40, linewidth=0.6, zorder=3,
            hatch=HATCHES[i % len(HATCHES)],
        )
        ax.add_patch(rect)
        # Box outline
        ax.plot([x - hw, x + hw, x + hw, x - hw, x - hw],
                [q1, q1, q3, q3, q1],
                color=color, lw=0.6, zorder=4)

        # Median line
        ax.plot([x - hw, x + hw], [med, med],
                color=color, lw=1.2, solid_capstyle="butt", zorder=5)

        # Whiskers
        ax.plot([x, x], [whisker_lo, q1], color=color, lw=0.6, zorder=3)
        ax.plot([x, x], [q3, whisker_hi], color=color, lw=0.6, zorder=3)
        cap_hw = hw * 0.5
        ax.plot([x - cap_hw, x + cap_hw], [whisker_lo, whisker_lo],
                color=color, lw=0.6, zorder=3)
        ax.plot([x - cap_hw, x + cap_hw], [whisker_hi, whisker_hi],
                color=color, lw=0.6, zorder=3)

        # Mean diamond
        ax.scatter([x], [mean_val], marker="D", s=10,
                   color=MEAN_COLOR, zorder=6, linewidths=0.3, edgecolors="white")


# ---------------------------------------------------------------------------
# Combined 1x3 panel figure
# ---------------------------------------------------------------------------

def plot_combined(
    config_dfs: List[Tuple[str, str, pd.DataFrame]],
    out_path: Path,
) -> None:
    """Single 1×3 figure: one panel per metric, grouped boxes per channel."""
    n_configs = len(config_dfs)
    colors = CONFIG_COLORS[:n_configs]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))

    for ax, metric_key in zip(axes, METRICS):
        meta = METRIC_META[metric_key]

        all_data = []
        for ch in CHANNELS:
            col = f"{metric_key}_{ch}"
            group = [df[col].dropna().values if col in df.columns else np.array([])
                     for _, _, df in config_dfs]
            all_data.append(group)

        # Y-limits from all data across channels for this metric
        flat = np.concatenate([v for grp in all_data for v in grp if len(v) > 0])
        if len(flat) > 0:
            q1, q3 = np.percentile(flat, [25, 75])
            iqr = q3 - q1 if q3 > q1 else 1.0
            y_lo = q1 - 1.8 * iqr
            y_hi = q3 + 1.8 * iqr
            if meta["ref_line"] is not None:
                y_lo = min(y_lo, meta["ref_line"] - 0.05 * iqr)
            y_max_cap = meta.get("y_max")
            if y_max_cap is not None:
                y_hi = min(float(y_hi), float(y_max_cap))
                if y_lo >= y_hi:
                    y_lo = y_hi - 0.05
            ax.set_ylim(y_lo, y_hi)

        # Reference line
        if meta["ref_line"] is not None:
            ax.axhline(meta["ref_line"], color="#999999", ls=":", lw=0.5, zorder=1)

        # Draw grouped boxes
        group_positions = np.arange(len(CHANNELS))
        for gi, (ch, group) in enumerate(zip(CHANNELS, all_data)):
            _draw_grouped_boxes(ax, group, group_positions[gi], colors)

        ax.set_xticks(group_positions)
        ax.set_xticklabels(CHANNELS, fontweight="bold")
        ax.set_xlim(-0.5, len(CHANNELS) - 0.5)
        ax.set_ylabel(meta["ylabel"], fontsize=9)
        ax.yaxis.set_major_locator(mpl_ticker.MaxNLocator(nbins=8))

    # Shared legend at top
    handles = [mpatches.Patch(facecolor=colors[i], edgecolor=colors[i],
                              alpha=0.55, label=config_dfs[i][0])
               for i in range(n_configs)]
    handles.append(plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=MEAN_COLOR,
                              markersize=4, label="Mean", linewidth=0))
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.0), ncol=n_configs + 1,
               frameon=True, framealpha=0.95, edgecolor="#cccccc",
               fontsize=9, handlelength=1.4, handletextpad=0.4,
               columnspacing=1.2, borderpad=0.3)

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.10, top=0.85, wspace=0.30)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Grouped box-plot figure from rollout CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv-dir",    default=None,
                   help="Directory to glob rollout_metrics*.csv from")
    p.add_argument("--csv",        nargs="*", default=None,
                   help="Explicit CSV paths (overrides --csv-dir)")
    p.add_argument("--output-dir", default="metrics_plots")
    p.add_argument("--output",     default=None,
                   help="Output path for combined figure (default: <output-dir>/metrics_boxplot.pdf)")
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
        print(f"  {letter}: {label:20s}  <-  {path.name}")

    config_dfs = [(letter, label, pd.read_csv(path))
                  for letter, label, path in configs]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = Path(args.output) if args.output else out_dir / "metrics_boxplot.pdf"
    print(f"\n-- Combined grouped box-plot figure --")
    plot_combined(config_dfs, out_path)

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
        print(f"\n  Summary table -> {summary_path}")

    print(f"\nDone -- figure saved to {out_path}")


if __name__ == "__main__":
    main()
