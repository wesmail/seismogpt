#!/usr/bin/env python3
"""
2-D (distance × depth) NCC heatmap diagnostic for SeismoGPT test-set rollouts.

Produces two separate figures (saved as <stem>_ncc.<ext> and <stem>_count.<ext>):
  1. Median NCC per (distance, depth) cell  (viridis, annotated with NCC value)
  2. Event count per cell                   (Greys,   annotated with count)

Usage:
  python plot_ncc_heatmap.py --csv mc_metrics.csv \
      --metadata-csv path/to/metadata.csv --output ncc_heatmap.pdf

  python plot_ncc_heatmap.py --csv mc_metrics.csv \
      --metadata-csv path/to/metadata.csv --channel Z --min-count 10
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
from scipy.stats import binned_statistic_2d

# ---------------------------------------------------------------------------
# Style (matches plot_rollout_metrics.py)
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
    "savefig.dpi":          600,
    "figure.dpi":           150,
    "pdf.fonttype":         42,
    "ps.fonttype":          42,
})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="2-D NCC heatmap: median NCC per (distance, depth) cell",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", required=True, help="Per-item metrics CSV from testset_rollout.py")
    p.add_argument("--metadata-csv", required=True, help="Dataset metadata CSV (one row per item)")
    p.add_argument("--output", default="ncc_heatmap_dist_depth.pdf", help="Output figure path")
    p.add_argument("--channel", default="global", choices=["global", "Z", "N", "E"],
                   help="Which NCC column to plot")
    p.add_argument("--min-count", type=int, default=20,
                   help="Mask cells with fewer than this many events")

    p.add_argument("--distance-col", default="distance_deg", help="Distance column in metadata")
    p.add_argument("--depth-col", default="src_depth_km", help="Depth column in metadata")

    p.add_argument("--dist-min", type=float, default=10.0)
    p.add_argument("--dist-max", type=float, default=82.0)
    p.add_argument("--dist-bin-width", type=float, default=6.0)
    p.add_argument("--depth-min", type=float, default=5.0)
    p.add_argument("--depth-max", type=float, default=105.0)
    p.add_argument("--depth-bin-width", type=float, default=10.0)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _outlined_text(ax, x, y, txt, fontsize=7, color="white", **kwargs):
    """Place text with a thin black outline for readability on any background."""
    t = ax.text(x, y, txt, fontsize=fontsize, color=color,
                ha="center", va="center", **kwargs)
    t.set_path_effects([
        PathEffects.withStroke(linewidth=1.5, foreground="black"),
    ])
    return t


def _annotate_heatmap(ax, matrix, counts, dist_edges, depth_edges,
                      min_count, mode="ncc"):
    """Annotate cells of an imshow heatmap with values and counts."""
    n_dist = len(dist_edges) - 1
    n_depth = len(depth_edges) - 1
    for di in range(n_dist):
        for dj in range(n_depth):
            cnt = int(counts[dj, di])
            val = matrix[dj, di]
            cx = 0.5 * (dist_edges[di] + dist_edges[di + 1])
            cy = 0.5 * (depth_edges[dj] + depth_edges[dj + 1])
            if np.isnan(val):
                continue
            if mode == "ncc":
                _outlined_text(ax, cx, cy, f"{val:.2f}", fontsize=6.5)
            else:
                _outlined_text(ax, cx, cy, f"{cnt}", fontsize=6.5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Load CSVs ----------------------------------------------------------
    csv_path = Path(args.csv)
    meta_path = Path(args.metadata_csv)
    if not csv_path.exists():
        print(f"Error: metrics CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)
    if not meta_path.exists():
        print(f"Error: metadata CSV not found: {meta_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    metadata = pd.read_csv(meta_path)

    # ---- Determine NCC column -----------------------------------------------
    if args.channel == "global":
        ncc_col = "ncc_global"
    else:
        ncc_col = f"ncc_{args.channel}"
    if ncc_col not in df.columns:
        raise KeyError(
            f"Column '{ncc_col}' not found in {csv_path.name}. "
            f"Available: {df.columns.tolist()}"
        )

    # ---- Merge metadata -----------------------------------------------------
    if "item" not in df.columns:
        raise KeyError(f"'item' column missing in {csv_path.name}. "
                       f"Available: {df.columns.tolist()}")
    dist_col = args.distance_col
    depth_col = args.depth_col
    for col in [dist_col, depth_col]:
        if col not in metadata.columns:
            raise KeyError(
                f"Column '{col}' not found in {meta_path.name}. "
                f"Available: {metadata.columns.tolist()}"
            )
    df[dist_col] = df["item"].apply(
        lambda i: metadata.iloc[int(i)][dist_col] if int(i) < len(metadata) else np.nan
    )
    df[depth_col] = df["item"].apply(
        lambda i: metadata.iloc[int(i)][depth_col] if int(i) < len(metadata) else np.nan
    )

    valid = df[[ncc_col, dist_col, depth_col]].dropna()
    print(f"Loaded {len(df)} items, {len(valid)} with valid NCC + distance + depth")

    # ---- Bin edges ----------------------------------------------------------
    dist_edges = np.arange(args.dist_min,
                           args.dist_max + args.dist_bin_width * 0.01,
                           args.dist_bin_width)
    depth_edges = np.arange(args.depth_min,
                            args.depth_max + args.depth_bin_width * 0.01,
                            args.depth_bin_width)

    x = valid[dist_col].values
    y = valid[depth_col].values
    z = valid[ncc_col].values

    # ---- Binned statistics --------------------------------------------------
    med_stat = binned_statistic_2d(
        x, y, z, statistic="median", bins=[dist_edges, depth_edges],
    )
    cnt_stat = binned_statistic_2d(
        x, y, z, statistic="count", bins=[dist_edges, depth_edges],
    )
    median_grid = med_stat.statistic.T   # shape (n_depth, n_dist)
    count_grid = cnt_stat.statistic.T

    # Mask sparse cells
    median_masked = np.where(count_grid < args.min_count, np.nan, median_grid)
    count_display = np.where(count_grid < args.min_count, np.nan, count_grid)

    n_cells = median_masked.size
    n_masked = int(np.isnan(median_masked).sum())
    frac_masked = n_masked / n_cells

    if frac_masked > 0.5:
        print(f"WARNING: {frac_masked:.0%} of cells masked (count < {args.min_count}). "
              "Consider lowering --min-count or widening bins.")

    # ---- Summary stats ------------------------------------------------------
    all_median = np.nanmedian(median_masked)
    print(f"Grid median NCC:          {all_median:.4f}")
    print(f"Cells masked:             {n_masked}/{n_cells} ({frac_masked:.0%})")

    # ---- Figures (separate files) --------------------------------------------
    extent = [dist_edges[0], dist_edges[-1], depth_edges[0], depth_edges[-1]]
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem = out_path.stem
    suffix = out_path.suffix or ".pdf"

    # -- Figure 1: NCC heatmap --
    fig_ncc, ax_ncc = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    im_ncc = ax_ncc.imshow(
        np.ma.masked_invalid(median_masked),
        origin="lower", aspect="auto",
        cmap="viridis", vmin=0.5, vmax=1.0,
        extent=extent,
    )
    _annotate_heatmap(ax_ncc, median_masked, count_grid,
                      dist_edges, depth_edges, args.min_count, mode="ncc")

    cb_ncc = fig_ncc.colorbar(im_ncc, ax=ax_ncc, fraction=0.046, pad=0.03)
    cb_ncc.set_label("Median NCC", fontsize=8)
    cb_ncc.ax.tick_params(labelsize=7)

    ax_ncc.set_xlabel(r"Epicentral distance ($\degree$)")
    ax_ncc.set_ylabel("Source depth (km)")
    ax_ncc.set_title("Median NCC per cell", pad=5)

    ncc_path = out_path.parent / f"{stem}_ncc{suffix}"
    fig_ncc.savefig(ncc_path, dpi=600, bbox_inches="tight")
    plt.close(fig_ncc)
    print(f"Saved -> {ncc_path}")

    # -- Figure 2: count heatmap --
    fig_cnt, ax_cnt = plt.subplots(figsize=(4.5, 3.5), constrained_layout=True)
    im_cnt = ax_cnt.imshow(
        np.ma.masked_invalid(count_display),
        origin="lower", aspect="auto",
        cmap="Greys",
        extent=extent,
    )
    _annotate_heatmap(ax_cnt, count_display, count_grid,
                      dist_edges, depth_edges, args.min_count, mode="count")

    cb_cnt = fig_cnt.colorbar(im_cnt, ax=ax_cnt, fraction=0.046, pad=0.03)
    cb_cnt.set_label("Events", fontsize=8)
    cb_cnt.ax.tick_params(labelsize=7)

    ax_cnt.set_xlabel(r"Epicentral distance ($\degree$)")
    ax_cnt.set_ylabel("Source depth (km)")
    ax_cnt.set_title("Event count per cell", pad=5)

    cnt_path = out_path.parent / f"{stem}_count{suffix}"
    fig_cnt.savefig(cnt_path, dpi=600, bbox_inches="tight")
    plt.close(fig_cnt)
    print(f"Saved -> {cnt_path}")


if __name__ == "__main__":
    main()
