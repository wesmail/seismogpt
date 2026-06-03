#!/usr/bin/env python3
"""
3×3 physics–metric corner for SeismoGPT test-set rollouts.

**Layout (always three columns, one row per requested metric):**

* **Rows** — one per metric in ``--metrics`` (default **NCC, SNR, PSD log-L²**).
  Use exactly three metrics for a **3×3** figure (recommended).

* **Columns** — three pairwise **physical** planes; each panel title states
  **$x$** and **$y$** explicitly (distance / depth / $M_w$).

Each panel is a **2-D heatmap** of the **median** of that row’s metric in each
bin. This is the seismology analogue of a GW “corner” over source parameters,
with **forecast quality** (NCC / SNR / PSD) shown instead of a posterior.

Color limits: **NCC** keeps fixed defaults; **SNR / PSD / …** use robust
percentiles of the plotted data unless you set ``--vrange``.

Companion to plot_ncc_heatmap.py and plot_ncc_facet.py.

Usage:
  python plot_ncc_corner.py --csv mc_metrics.csv \\
      --metadata-csv path/to/metadata.csv --output physics_corner.pdf

  # Z-channel metrics (still 3×3 if --metrics has three entries)
  python plot_ncc_corner.py --csv mc_metrics.csv \\
      --metadata-csv path/to/metadata.csv \\
      --metrics ncc,snr,psd_logl2 --channel Z --output ncc_corner_Z.pdf
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
# Style (matches plot_ncc_heatmap.py / plot_ncc_facet.py)
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
# Metric registry
# ---------------------------------------------------------------------------
# Each metric key resolves to a CSV column "<col_prefix>_<channel>"
# (channel is "global", "Z", "N", or "E"). cmap is chosen so green = good:
# viridis    for "higher-is-better" metrics, viridis_r for error metrics.
# (vmin, vmax) defaults are reasonable starting points; override via CLI.

METRICS = {
    "ncc": {
        "label":      "NCC",
        "cb_label":   "Median NCC",
        "col_prefix": "ncc",
        "cmap":       "viridis",
        "vmin":       0.5,
        "vmax":       1.0,
        "fmt":        "{:.2f}",
        "higher_is_better": True,
    },
    "snr": {
        "label":      "SRR",
        "cb_label":   "Median SRR",
        "col_prefix": "snr_db",
        "cmap":       "viridis",
        "vmin":       None,
        "vmax":       None,
        "fmt":        "{:.0f}",
        "higher_is_better": True,
    },
    "robust_snr": {
        "label":      "Robust SRR",
        "cb_label":   "Median robust SRR",
        "col_prefix": "robust_snr_db",
        "cmap":       "viridis",
        "vmin":       None,
        "vmax":       None,
        "fmt":        "{:.0f}",
        "higher_is_better": True,
    },
    "psd_logl2": {
        "label":      "PSD log-L2",
        "cb_label":   "Median PSD log-L2 error",
        "col_prefix": "psd_logl2",
        "cmap":       "viridis_r",   # reversed: low error -> green
        "vmin":       None,          # filled from data percentiles in main
        "vmax":       None,
        "fmt":        "{:.2f}",
        "higher_is_better": False,
    },
    "psd_rel": {
        "label":      "PSD rel.",
        "cb_label":   "Median PSD relative error",
        "col_prefix": "psd_rel",
        "cmap":       "viridis_r",
        "vmin":       None,
        "vmax":       None,
        "fmt":        "{:.2f}",
        "higher_is_better": False,
    },
    "skill": {
        "label":      "Forecast skill",
        "cb_label":   "Median forecast skill",
        "col_prefix": "skill",
        "cmap":       "viridis",
        "vmin":       None,
        "vmax":       None,
        "fmt":        "{:.2f}",
        "higher_is_better": True,
    },
}

DEFAULT_METRICS = ["ncc", "snr", "psd_logl2"]


def _resolve_metric_column(metric_key: str, channel: str) -> str:
    """Map (metric_key, channel) -> CSV column name."""
    prefix = METRICS[metric_key]["col_prefix"]
    suffix = "global" if channel == "global" else channel
    return f"{prefix}_{suffix}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "3-column physics corner: one row per metric × "
            "(distance×depth, distance×Mw, depth×Mw) heatmaps"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", required=True, help="Per-item metrics CSV")
    p.add_argument("--metadata-csv", required=True, help="Dataset metadata CSV")
    p.add_argument("--output", default="physics_grid.pdf",
                   help="Output figure path")
    p.add_argument("--channel", default="global", choices=["global", "Z", "N", "E"],
                   help="Which channel suffix to use for every metric")
    p.add_argument("--min-count", type=int, default=20,
                   help="Mask cells with fewer than this many events")

    p.add_argument("--metrics", default=",".join(DEFAULT_METRICS),
                   help=("Comma-separated list of metrics to plot, one per row. "
                         f"Options: {', '.join(METRICS.keys())}."))

    p.add_argument("--distance-col", default="distance_deg",
                   help="Distance column in metadata")
    p.add_argument("--depth-col", default="src_depth_km",
                   help="Depth column in metadata")
    p.add_argument("--mag-col", default="Mw",
                   help="Magnitude column in metadata (PaperTestSet uses Mw)")

    p.add_argument("--dist-min", type=float, default=10.0)
    p.add_argument("--dist-max", type=float, default=82.0)
    p.add_argument("--dist-bin-width", type=float, default=6.0)
    p.add_argument("--depth-min", type=float, default=5.0)
    p.add_argument("--depth-max", type=float, default=105.0)
    p.add_argument("--depth-bin-width", type=float, default=10.0)
    p.add_argument("--mag-min", type=float, default=3.0,
                   help="Magnitude bin lower edge (PaperTestSet Mw ~3–7)")
    p.add_argument("--mag-max", type=float, default=7.0,
                   help="Magnitude bin upper edge")
    p.add_argument("--mag-bin-width", type=float, default=0.5)

    p.add_argument("--no-annotate", action="store_true",
                   help="Disable per-cell numeric annotations "
                        "(useful when cells are very small)")

    # Per-metric v-range overrides (optional). Format: "ncc=0.6:1.0,snr=0:20"
    p.add_argument("--vrange",
                   default=None,
                   help=("Optional per-metric v-range overrides, e.g. "
                         "'ncc=0.6:1.0,snr=0:25'."))

    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _outlined_text(ax, x, y, txt, fontsize=6, color="white", **kwargs):
    """Place text with a thin black outline for readability on any background."""
    t = ax.text(x, y, txt, fontsize=fontsize, color=color,
                ha="center", va="center", **kwargs)
    t.set_path_effects([
        PathEffects.withStroke(linewidth=1.5, foreground="black"),
    ])
    return t


def _heatmap_panel(ax, x, y, z, x_edges, y_edges, min_count,
                   vmin, vmax, cmap, fmt, annotate=True, annot_fontsize=5.0):
    """Plot median-of-z heatmap on (x_edges, y_edges); returns AxesImage."""
    med = binned_statistic_2d(
        x, y, z, statistic="median", bins=[x_edges, y_edges],
    ).statistic.T
    cnt = binned_statistic_2d(
        x, y, z, statistic="count", bins=[x_edges, y_edges],
    ).statistic.T
    med_masked = np.where(cnt < min_count, np.nan, med)

    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    im = ax.imshow(
        np.ma.masked_invalid(med_masked),
        origin="lower", aspect="auto",
        cmap=cmap, vmin=vmin, vmax=vmax, extent=extent,
    )

    if annotate:
        for i in range(len(x_edges) - 1):
            for j in range(len(y_edges) - 1):
                v = med_masked[j, i]
                if np.isnan(v):
                    continue
                cx = 0.5 * (x_edges[i] + x_edges[i + 1])
                cy = 0.5 * (y_edges[j] + y_edges[j + 1])
                _outlined_text(ax, cx, cy, fmt.format(v),
                               fontsize=annot_fontsize)

    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    return im


def _parse_vrange_overrides(s):
    """Parse '--vrange ncc=0.6:1.0,snr=0:25' into {metric: (vmin, vmax)}."""
    if s is None:
        return {}
    overrides = {}
    for chunk in s.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk or ":" not in chunk:
            raise SystemExit(
                f"Error: bad --vrange element '{chunk}'. "
                "Use 'metric=vmin:vmax'."
            )
        m, rng = chunk.split("=", 1)
        lo, hi = rng.split(":", 1)
        overrides[m.strip()] = (float(lo), float(hi))
    return overrides


def _metric_color_limits(metric_key, vals, vrange_overrides, m_info):
    """
    Return (vmin, vmax) for a metric row.

    Order: ``--vrange`` override > fixed registry (both set) >
    robust 2–98% range of ``vals``.
    """
    if metric_key in vrange_overrides:
        return vrange_overrides[metric_key]
    vmin_def, vmax_def = m_info["vmin"], m_info["vmax"]
    if vmin_def is not None and vmax_def is not None:
        return float(vmin_def), float(vmax_def)
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(v, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        if lo >= hi:
            lo, hi = lo - 1e-6, hi + 1e-6
    return lo, hi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- Parse metric list --------------------------------------------------
    metric_keys = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if len(metric_keys) == 0:
        raise SystemExit("Error: need at least one metric in --metrics")
    for m in metric_keys:
        if m not in METRICS:
            raise SystemExit(
                f"Error: unknown metric '{m}'. "
                f"Options: {', '.join(METRICS.keys())}"
            )

    if len(metric_keys) != 3:
        print(
            f"Note: {len(metric_keys)} metric row(s); for a 3×3 grid use "
            f"exactly three metrics, e.g. --metrics ncc,snr,psd_logl2",
            file=sys.stderr,
        )

    vrange_overrides = _parse_vrange_overrides(args.vrange)

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

    if "item" not in df.columns:
        raise KeyError(
            f"'item' column missing in {csv_path.name}. "
            f"Available: {df.columns.tolist()}"
        )

    # ---- Resolve & validate metric columns ---------------------------------
    metric_cols = {m: _resolve_metric_column(m, args.channel) for m in metric_keys}
    for m, col in metric_cols.items():
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' (for metric '{m}', channel '{args.channel}') "
                f"not found in {csv_path.name}. "
                f"Available: {df.columns.tolist()}"
            )

    # ---- Merge metadata -----------------------------------------------------
    meta_cols = [args.distance_col, args.depth_col, args.mag_col]
    for col in meta_cols:
        if col not in metadata.columns:
            raise KeyError(
                f"Column '{col}' not found in {meta_path.name}. "
                f"Available: {metadata.columns.tolist()}"
            )
    for col in meta_cols:
        df[col] = df["item"].apply(
            lambda i, c=col: (metadata.iloc[int(i)][c]
                              if int(i) < len(metadata) else np.nan)
        )

    needed_cols = list(metric_cols.values()) + meta_cols
    valid = df[needed_cols].dropna()
    print(f"Loaded {len(df)} items, {len(valid)} valid for "
          f"{len(metric_keys)} metric(s) + dist + depth + mag")

    # ---- Bin edges ----------------------------------------------------------
    dist_edges = np.arange(
        args.dist_min, args.dist_max + args.dist_bin_width * 0.01,
        args.dist_bin_width,
    )
    depth_edges = np.arange(
        args.depth_min, args.depth_max + args.depth_bin_width * 0.01,
        args.depth_bin_width,
    )
    mag_edges = np.arange(
        args.mag_min, args.mag_max + args.mag_bin_width * 0.01,
        args.mag_bin_width,
    )

    # Column definitions:
    # (x_col, x_edges, x_label, y_col, y_edges, y_label, column_title)
    pairs = [
        (args.distance_col, dist_edges,  r"Distance ($\degree$)",
         args.depth_col,    depth_edges, r"Depth (km)",
         r"$x$ = distance ($\degree$)" "\n" r"$y$ = depth (km)"),
        (args.distance_col, dist_edges,  r"Distance ($\degree$)",
         args.mag_col,      mag_edges,   r"$M_w$",
         r"$x$ = distance ($\degree$)" "\n" r"$y$ = $M_w$"),
        (args.depth_col,    depth_edges, r"Depth (km)",
         args.mag_col,      mag_edges,   r"$M_w$",
         r"$x$ = depth (km)" "\n" r"$y$ = $M_w$"),
    ]

    n_rows = len(metric_keys)
    n_cols = 3

    # ---- Figure -------------------------------------------------------------
    panel_w = 2.7
    panel_h = 2.3
    fig_w = panel_w * n_cols + 1.55   # colorbars + row labels + y labels on all cols
    fig_h = panel_h * n_rows + 0.9   # extra for title + bottom xlabels

    fig = plt.figure(figsize=(fig_w, fig_h))
    # An extra narrow column on the right for per-row colorbars.
    gs = fig.add_gridspec(
        n_rows, n_cols + 1,
        width_ratios=[1.0, 1.0, 1.0, 0.06],
        hspace=0.18, wspace=0.18,
        left=0.085, right=0.92,
        top=1.0 - 0.6 / fig_h,
        bottom=0.55 / fig_h,
    )

    annotate = not args.no_annotate

    for r, metric_key in enumerate(metric_keys):
        m_info = METRICS[metric_key]
        m_col = metric_cols[metric_key]
        z = valid[m_col].values

        vmin, vmax = _metric_color_limits(
            metric_key, z, vrange_overrides, m_info,
        )

        im_last = None
        for c, (cx_col, cx_edges, cx_lbl,
                cy_col, cy_edges, cy_lbl, col_title) in enumerate(pairs):
            ax = fig.add_subplot(gs[r, c])
            x = valid[cx_col].values
            y = valid[cy_col].values

            im = _heatmap_panel(
                ax, x, y, z, cx_edges, cy_edges, args.min_count,
                vmin=vmin, vmax=vmax,
                cmap=m_info["cmap"], fmt=m_info["fmt"],
                annotate=annotate,
            )
            im_last = im

            if r == 0:
                ax.set_title(col_title, fontsize=8.5, pad=8)

            # X labels only on the bottom row
            if r == n_rows - 1:
                ax.set_xlabel(cx_lbl, fontsize=8)
            else:
                ax.tick_params(axis="x", labelbottom=False)

            # Y axis: label + tick values on every column (right column is also
            # $M_w$ vs depth; it should not rely on the middle panel alone).
            ax.set_ylabel(cy_lbl, fontsize=8)

            # Row label on the leftmost panel (metric name to the left of y-label)
            if c == 0:
                ax.text(
                    -0.36, 0.5, m_info["label"],
                    transform=ax.transAxes,
                    fontsize=9, fontweight="bold", color="#222222",
                    ha="center", va="center", rotation=90,
                )

        # ---- Per-row colorbar ------------------------------------------------
        cax = fig.add_subplot(gs[r, n_cols])
        cax.grid(False)
        cb = fig.colorbar(im_last, cax=cax)
        cb.set_label(m_info["cb_label"], fontsize=7.5)
        cb.ax.tick_params(labelsize=7)

    #fig.suptitle(
    #    f"Metrics × parameter planes — channel: {args.channel}  "
    #    f"(N={len(valid)}, min count per cell={args.min_count})",
    #    fontsize=10, y=0.99,
    #)

    # ---- Summary stats to stdout -------------------------------------------
    print("\nPer-metric overall summary:")
    for m_key in metric_keys:
        m_col = metric_cols[m_key]
        col_vals = valid[m_col].values
        print(f"  {m_key:11s} ({m_col:24s}): "
              f"median={np.nanmedian(col_vals):.4f}, "
              f"IQR=[{np.nanpercentile(col_vals, 25):.4f}, "
              f"{np.nanpercentile(col_vals, 75):.4f}]")

    # ---- Save ---------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()