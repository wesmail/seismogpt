#!/usr/bin/env python3
"""
Filtered catalog rollouts → one PDF per selected event.

1. Scan dataset metadata for **distance**, **Mw**, and **depth** inside user ranges.
2. Run the same autoregressive **rollout** as ``testset_rollout.py`` (cap how many
   geometry-qualified items are tried with ``--max-scan``).
3. Keep events whose **global** NCC, SNR (dB), and PSD log-L² sit in metric
   bands (defaults are wide; override flags to narrow).
4. From that pool, pick **n** items (``head`` / ``spread`` / ``random``) and save
   one **PDF** per item (Z/N/E; header with geometry + NCC + SRR + PSD;
   publication style matching ``make_paper_figures.py``).

Run from the ``Paper`` directory (or pass an absolute path to this file)::

  cd /path/to/Paper
  python rollout_filtered_catalog_plots.py --checkpoint ... \\
      --future_secs 240 --context_ratio 2 --mode free \\
      --dist-min 30 --dist-max 70 --mw-min 5 --mw-max 6.5 \\
      --depth-min 10 --depth-max 90 \\
      --ncc-min 0.75 --snr-min 0 --psd-l2-max 6 \\
      -n 5 --pick spread \\
      --output-dir figures/catalog_rollouts --stem demo \\
      --csv figures/catalog_rollouts/selected.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")

from data.data_handling import SeismicWaveformDataset
from models.lightning_module import GPTLightning

import make_paper_figures as mpf
import testset_rollout as tsr
from testset_rollout import (
    METRICS_CSV_COLUMNS,
    evaluate_prediction_horizon,
    rollout,
)


def _catalog_axes_title(
    metadata: dict | None,
    metrics: dict | None,
) -> str:
    """Single-line axes title: geometry and metrics separated by |."""
    parts: List[str] = []
    if metadata:
        m = float(metadata.get("source_magnitude", np.nan))
        d = float(metadata.get("path_ep_distance_deg", np.nan))
        z = float(metadata.get("source_depth_km", np.nan))
        if np.isfinite(m) and np.isfinite(d) and np.isfinite(z):
            parts.append(f"$M_w$ {m:.1f}, $\\Delta$ {d:.0f}°, depth {z:.0f} km")
    if metrics:
        ncc_g = float(metrics.get("ncc_global", np.nan))
        srr_g = float(metrics.get("snr_db_global", np.nan))
        psd_g = float(metrics.get("psd_logl2_global", np.nan))
        if np.isfinite(ncc_g):
            parts.append(f"NCC = {ncc_g:.2f}")
        if np.isfinite(srr_g):
            parts.append(f"SRR = {srr_g:+.1f} dB")
        if np.isfinite(psd_g):
            parts.append(f"PSD log-L$^2$ = {psd_g:.3f}")
    return "  |  ".join(parts)


def plot_catalog_rollout(
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
    metrics: dict | None = None,
    layout: str = "compact",
    save_dpi: int = 600,
    kernel_size: int = 16,
):
    """One-event PDF using the same renderer as ``make_paper_figures.py contrast``."""
    del y_true_future, mode, context_secs, future_secs, item, layout, kernel_size

    ps_sec = np.nan
    if metadata:
        p_m = float(metadata.get("trace_p_arrival_s", np.nan))
        s_m = float(metadata.get("trace_s_arrival_s", np.nan))
        if np.isfinite(p_m) and np.isfinite(s_m) and (s_m - p_m) > 0:
            ps_sec = float(s_m - p_m)

    L_fut = len(y_pred_rollout)
    mpf.save_contrast_style_figure(
        [{
            "y_full": y_true_full,
            "y_pred": y_pred_rollout,
            "L_ctx": L_ctx,
            "L_fut": L_fut,
            "sr": sample_rate,
            "ps_sec": ps_sec,
            "title": _catalog_axes_title(metadata, metrics),
        }],
        output_path,
        save_dpi=save_dpi,
        fig_width=9.0,
        row_height=0.78,
        top_pad=0.40,
        hspace=0.18,
        show_arrival_labels=True,
    )


def _in_range(v: float, lo: Optional[float], hi: Optional[float]) -> bool:
    if not np.isfinite(v):
        return False
    if lo is not None and v < lo:
        return False
    if hi is not None and v > hi:
        return False
    return True


def _row_float(row, key: str) -> float:
    return float(row.get(key, np.nan))


def _geometry_candidates(
    meta,
    *,
    dist_col: str,
    mag_col: str,
    depth_col: str,
    dist_lo: Optional[float],
    dist_hi: Optional[float],
    mag_lo: Optional[float],
    mag_hi: Optional[float],
    depth_lo: Optional[float],
    depth_hi: Optional[float],
) -> List[int]:
    out: List[int] = []
    for i in range(len(meta)):
        row = meta.iloc[i]
        d = _row_float(row, dist_col)
        m = _row_float(row, mag_col)
        z = _row_float(row, depth_col)
        if not (
            _in_range(d, dist_lo, dist_hi)
            and _in_range(m, mag_lo, mag_hi)
            and _in_range(z, depth_lo, depth_hi)
        ):
            continue
        out.append(i)
    return out


def _passes_metric_filters(m: Dict[str, float], args) -> bool:
    ncc = float(m.get("ncc_global", np.nan))
    snr = float(m.get("snr_db_global", np.nan))
    psd = float(m.get("psd_logl2_global", np.nan))
    if not np.isfinite(ncc) or not np.isfinite(snr) or not np.isfinite(psd):
        return False
    if not _in_range(ncc, args.ncc_min, args.ncc_max):
        return False
    if not _in_range(snr, args.snr_min, args.snr_max):
        return False
    if not _in_range(psd, args.psd_l2_min, args.psd_l2_max):
        return False
    return True


def _select_indices(n_pool: int, n_take: int, pick: str, rng: np.random.Generator) -> np.ndarray:
    if n_pool < n_take:
        raise ValueError("pool smaller than n_take")
    if pick == "head":
        return np.arange(n_take)
    if pick == "random":
        return np.sort(rng.choice(n_pool, size=n_take, replace=False))
    return np.linspace(0, n_pool - 1, n_take, dtype=int)


def _metadata_plot_dict(
    meta_row, dist_col: str, mag_col: str, depth_col: str,
) -> Dict[str, float]:
    return {
        "trace_p_arrival_s":    meta_row.get("trace_p_arrival_s", np.nan),
        "trace_s_arrival_s":    meta_row.get("trace_s_arrival_s", np.nan),
        "source_depth_km":      meta_row.get(depth_col, np.nan),
        "source_magnitude":     meta_row.get(mag_col, np.nan),
        "path_ep_distance_deg": meta_row.get(dist_col, np.nan),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Rollout events in Mw/Δ/depth ranges; filter by NCC/SNR/PSD; save n PDFs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument(
        "--data-dir",
        type=str,
        default="/mnt/d/waleed/Seismology/simulation/clean/SobolDataset/PaperTestSet/",
    )
    p.add_argument("--kernel-size", type=int, default=16)
    p.add_argument("--num-tokens", type=int, default=320)
    p.add_argument("--context-ratio", "--context_ratio", type=float, default=2.0)
    p.add_argument("--future-secs", "--future_secs", type=float, default=240.0)
    p.add_argument("--mode", choices=("free", "teacher"), default="free")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-context-tokens", type=int, default=None)

    p.add_argument("--distance-col", default="distance_deg")
    p.add_argument("--mag-col", default="Mw")
    p.add_argument("--depth-col", default="src_depth_km")

    # Geometry (defaults match plot_ncc_heatmap.py / PaperTestSet-style ranges)
    p.add_argument(
        "--dist-min", type=float, default=25.0,
        help="Minimum epicentral distance (°)",
    )
    p.add_argument(
        "--dist-max", type=float, default=40.0,
        help="Maximum epicentral distance (°)",
    )
    p.add_argument(
        "--mw-min", type=float, default=3.0,
        help="Minimum magnitude Mw",
    )
    p.add_argument(
        "--mw-max", type=float, default=7.0,
        help="Maximum magnitude Mw",
    )
    p.add_argument(
        "--depth-min", type=float, default=5.0,
        help="Minimum source depth (km)",
    )
    p.add_argument(
        "--depth-max", type=float, default=105.0,
        help="Maximum source depth (km)",
    )

    # Post-rollout global metrics (wide defaults ≈ no practical cut)
    p.add_argument(
        "--ncc-min", type=float, default=0.0,
        help="Minimum ncc_global",
    )
    p.add_argument(
        "--ncc-max", type=float, default=0.4,
        help="Maximum ncc_global",
    )
    p.add_argument(
        "--snr-min", type=float, default=-20.0,
        help="Minimum snr_db_global (dB)",
    )
    p.add_argument(
        "--snr-max", type=float, default=2.0,
        help="Maximum snr_db_global (dB)",
    )
    p.add_argument(
        "--psd-l2-min", type=float, default=10.0,
        help="Minimum psd_logl2_global (log-spectral L²)",
    )
    p.add_argument(
        "--psd-l2-max", type=float, default=1.0e5,
        help="Maximum psd_logl2_global (log-spectral L²)",
    )

    p.add_argument(
        "-n",
        "--num-examples",
        type=int,
        default=5,
        help="How many events to plot (one PDF each)",
    )
    p.add_argument(
        "--max-scan",
        type=int,
        default=1000,
        help="Max geometry-qualified items to rollout-scan (shuffled order)",
    )
    p.add_argument(
        "--max-pool",
        type=int,
        default=400,
        help="Stop scanning once this many items pass metric filters",
    )
    p.add_argument(
        "--pick",
        choices=("spread", "head", "random"),
        default="spread",
        help="How to choose n from the metric pool (spread uses NCC order)",
    )
    p.add_argument(
        "--sort-for-spread",
        choices=("ncc_asc", "ncc_desc"),
        default="ncc_asc",
        help="Sort key before spread pick",
    )

    p.add_argument(
        "--output-dir",
        type=str,
        default="figures/catalog_rollout_pdfs",
    )
    p.add_argument(
        "--stem",
        type=str,
        default="rollout_filtered",
        help="Filename stem: {stem}_item_XXXXX.pdf",
    )
    p.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional CSV path for the n selected rows",
    )
    p.add_argument(
        "--csv-pool",
        type=str,
        default=None,
        help="Optional CSV for every metric-passing candidate before sub-selection",
    )
    p.add_argument("--layout", choices=("compact", "standard"), default="compact")
    p.add_argument("--save-dpi", type=int, default=600)
    return p.parse_args()


def main():
    args = parse_args()
    tsr._set_global_seeds(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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
    print(f"Dataset: {len(dataset)} items @ {sr:.3f} Hz")

    geom = _geometry_candidates(
        meta,
        dist_col=args.distance_col,
        mag_col=args.mag_col,
        depth_col=args.depth_col,
        dist_lo=args.dist_min,
        dist_hi=args.dist_max,
        mag_lo=args.mw_min,
        mag_hi=args.mw_max,
        depth_lo=args.depth_min,
        depth_hi=args.depth_max,
    )
    if not geom:
        print("ERROR: no items match geometry filters.", file=sys.stderr)
        sys.exit(1)
    print(f"Geometry filter: {len(geom)} items")

    rng.shuffle(geom)
    scan_list = geom[: max(1, args.max_scan)]

    lightning = GPTLightning.load_from_checkpoint(args.checkpoint)
    model = lightning.gpt.to(device)
    model.eval()

    pool: List[Dict[str, object]] = []
    pool_rows_csv: List[Dict[str, object]] = []

    pbar = tqdm(scan_list, desc="Rollout scan", unit="item")
    for item in pbar:
        row = meta.iloc[item]
        p_sec = row.get("trace_p_arrival_s", np.nan)
        s_sec = row.get("trace_s_arrival_s", np.nan)
        ps_sec = float(s_sec) - float(p_sec) if np.isfinite(p_sec) and np.isfinite(s_sec) else np.nan
        if np.isfinite(ps_sec) and ps_sec > 0:
            context_secs = float(args.context_ratio) * ps_sec
        else:
            context_secs = 60.0

        sample = dataset[int(item)]
        data = {"x": sample["x"].unsqueeze(0)}

        try:
            y_full, y_fut, y_pred, L_ctx, L_fut = rollout(
                model=model,
                data=data,
                item=0,
                context_secs=context_secs,
                future_secs=args.future_secs,
                sample_rate=sr,
                device=device,
                kernel_size=args.kernel_size,
                mode=args.mode,
                max_context_tokens=args.max_context_tokens,
            )
        except Exception as e:
            tqdm.write(f"  skip item {item}: rollout failed ({e})")
            continue

        metrics = evaluate_prediction_horizon(y_fut, y_pred, fs=sr)
        if not _passes_metric_filters(metrics, args):
            continue

        pool.append(
            {
                "item": int(item),
                "context_secs": float(context_secs),
                "y_true_full": y_full,
                "y_true_future": y_fut,
                "y_pred": y_pred,
                "L_ctx": int(L_ctx),
                "metrics": metrics,
                "meta_plot": _metadata_plot_dict(
                    row, args.distance_col, args.mag_col, args.depth_col,
                ),
            }
        )
        row_csv = {"item": int(item)}
        row_csv.update(metrics)
        pool_rows_csv.append(row_csv)
        pbar.set_postfix(pool=len(pool))

        if len(pool) >= args.max_pool:
            break

    if len(pool) < args.num_examples:
        print(
            f"ERROR: only {len(pool)} item(s) passed metric filters; need "
            f"{args.num_examples}. Loosen filters, raise --max-scan, or lower -n.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.csv_pool and pool_rows_csv:
        outp = Path(args.csv_pool)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=METRICS_CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(pool_rows_csv)
        print(f"Pool CSV ({len(pool_rows_csv)} rows) → {outp}")

    asc = args.sort_for_spread == "ncc_asc"
    pool.sort(
        key=lambda r: float(r["metrics"]["ncc_global"]),  # type: ignore[index]
        reverse=not asc,
    )

    idx_take = _select_indices(len(pool), args.num_examples, args.pick, rng)
    chosen = [pool[int(i)] for i in idx_take]

    written_rows = []
    for rank, rec in enumerate(chosen):
        it = int(rec["item"])
        pdf_path = out_dir / f"{args.stem}_item_{it:05d}.pdf"
        plot_catalog_rollout(
            rec["y_true_full"],  # type: ignore[arg-type]
            rec["y_true_future"],  # type: ignore[arg-type]
            rec["y_pred"],  # type: ignore[arg-type]
            int(rec["L_ctx"]),
            sr,
            args.mode,
            float(rec["context_secs"]),
            args.future_secs,
            it,
            str(pdf_path),
            metadata=rec["meta_plot"],  # type: ignore[arg-type]
            metrics=rec["metrics"],  # type: ignore[arg-type]
            layout=args.layout,
            save_dpi=args.save_dpi,
            kernel_size=args.kernel_size,
        )
        row = {"item": it}
        row.update(rec["metrics"])  # type: ignore[arg-type]
        written_rows.append(row)
        print(f"  [{rank+1}/{len(chosen)}] item {it} → {pdf_path}")

    if args.csv and written_rows:
        outp = Path(args.csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=METRICS_CSV_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(written_rows)
        print(f"Selected CSV ({len(written_rows)} rows) → {outp}")


if __name__ == "__main__":
    main()

# Configuration A
# python rollout_filtered_catalog_plots.py   --checkpoint phase1/epoch=12-step=633750.ckpt   --future_secs 120 --context_ratio 1 --mode free  --output-dir figures/catalog_rollouts --stem demo   --csv figures/catalog_rollouts/selected.csv -n 10 --depth-max 20 --dist-max 30

# Configuration B
# python rollout_filtered_catalog_plots.py   --checkpoint phase1/epoch=12-step=633750.ckpt   --future_secs 240 --context_ratio 1 --mode free  --output-dir figures/catalog_rollouts --stem demo   --csv figures/catalog_rollouts/selected.csv -n 10 --depth-min 80 --dist-max 30

# Configuration C
# python rollout_filtered_catalog_plots.py   --checkpoint phase1/epoch=12-step=633750.ckpt   --future_secs 240 --context_ratio 2 --mode free  --output-dir figures/catalog_rollouts --stem demo   --csv figures/catalog_rollouts/selected.csv -n 10 --depth-min 70 --dist-max 65 --dist-min 5