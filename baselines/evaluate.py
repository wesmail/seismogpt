#!/usr/bin/env python3
"""Evaluate any forecaster (SeismoGPT or a baseline) on the test set.

Same items, context lengths, forecast lengths, metrics and CSV columns as
quality_assurance/efficient_rollout.py, so every downstream plotting script works on
the output and the numbers of all models are directly comparable (paired by item).

  python baselines/evaluate.py --model knn --library knn_lib \
      --data_dir /path/test --context_ratio 1 --future_secs 120 -n 2000 \
      --csv results/eval/knn/rollout_metrics_1_120.csv

Models and what they need:
  zero, persistence      nothing
  ar_burg                nothing (--ar_order, default 32)
  linear                 --weights linear_ar.npz      (baselines/train_linear.py)
  gru                    --checkpoint gru.ckpt        (configs/baseline_gru.yaml)
  knn, knn_params        --library knn_lib/           (baselines/build_knn_library.py)
  seismogpt              --checkpoint seismogpt.ckpt

Besides the CSV, writes <csv>.json: model, settings, and the forecast time per item.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "quality_assurance")]

from baselines.forecasters import MODELS, Forecaster, build                      # noqa: E402
from data.data_handling import SeismicWaveformDataset                             # noqa: E402
from efficient_rollout import (METRICS_CSV_COLUMNS, _ItemSubset,                 # noqa: E402
                               evaluate_prediction_horizon)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--data_dir", required=True, help="test set (SeisBench folder)")
    p.add_argument("--context_ratio", type=float, default=1.0, help="context = ratio x (S - P)")
    p.add_argument("--future_secs", type=float, required=True)
    p.add_argument("--num_samples", "-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42, help="item selection (same as efficient_rollout.py)")
    p.add_argument("--csv", required=True)
    p.add_argument("--kernel_size", type=int, default=16)
    p.add_argument("--num_tokens", type=int, default=320)
    p.add_argument("--skip-psd", action="store_true")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g = p.add_argument_group("model files and options")
    g.add_argument("--checkpoint", help="gru or seismogpt")
    g.add_argument("--weights", help="linear: .npz from train_linear.py")
    g.add_argument("--library", help="knn: folder from build_knn_library.py")
    g.add_argument("--ar_order", type=int, default=32)
    g.add_argument("--knn_k", type=int, default=1)
    g.add_argument("--knn_no_flip", action="store_true", help="knn: do not use reversed polarity")
    return p.parse_args()


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    K = a.kernel_size

    # ---- data and items: exactly as efficient_rollout.py ----------------------------
    ds = SeismicWaveformDataset(data_dir=a.data_dir, kernel_size=K, stride=K,
                                num_tokens=a.num_tokens, training=False, normalize=True)
    fs, meta = ds.sr, ds.metadata
    n = min(a.num_samples, len(ds))
    items = np.sort(np.random.default_rng(a.seed).choice(len(ds), size=n, replace=False))

    loader = DataLoader(_ItemSubset(ds, items), batch_size=1, shuffle=False,
                        num_workers=a.num_workers)
    rows, contexts, ns, truths, metas = [], [], [], [], []
    for idx, x in tqdm(loader, total=n, desc="load", unit="item"):
        item, x = int(idx[0]), x[0]                                  # x [T, C, K]
        ps = meta.iloc[item].get("trace_s_arrival_s", np.nan) - meta.iloc[item].get("trace_p_arrival_s", np.nan)
        ctx_secs = a.context_ratio * ps if np.isfinite(ps) and ps > 0 else 60.0
        T_ctx = max(1, int(ctx_secs * fs) // K)
        n_fut = max(1, int(a.future_secs * fs) // K)
        T_ctx = min(T_ctx, x.shape[0] - 1)                           # same clamps as rollout_kvcache
        n_fut = min(n_fut, x.shape[0] - T_ctx)
        rows.append(item)
        contexts.append(x[:T_ctx].clone())                           # the forecaster sees ONLY this
        truths.append(x[T_ctx:T_ctx + n_fut].clone())
        ns.append(n_fut)
        metas.append(meta.iloc[item].to_dict())                     # only knn_params reads it

    # ---- forecast ---------------------------------------------------------------------
    model = build(a.model, a)
    print(f"model: {a.model}  |  device: {a.device}  |  {n} items  |  "
          f"context {a.context_ratio} x (S-P), future {a.future_secs} s")
    t0 = time.time()
    if type(model).forecast_many is not Forecaster.forecast_many:   # batched search (knn)
        preds = model.forecast_many(contexts, ns, metas)
    else:
        preds = [model.forecast(c, m) for c, m in tqdm(zip(contexts, ns), total=n, desc=a.model)]
    secs = time.time() - t0

    # ---- metrics: same function and columns as efficient_rollout.py -------------------
    out = []
    for item, y, p in tqdm(zip(rows, truths, preds), total=n, desc="metrics"):
        C = y.shape[1]
        y_true = y.permute(0, 2, 1).reshape(-1, C).float().numpy()   # [n*K, C], time order
        y_pred = p.permute(0, 2, 1).reshape(-1, C).float().numpy()
        out.append({"item": item, **evaluate_prediction_horizon(y_true, y_pred, fs=fs,
                                                                skip_psd=a.skip_psd)})

    Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=METRICS_CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    info = {"model": a.model, "seconds_per_item": secs / n, "items": n,
            "args": {k: v for k, v in vars(a).items()}}
    if getattr(model, "last_neighbours", None) is not None:          # knn: which training window
        info["knn_neighbours"] = {int(i): model.last_neighbours[b].tolist() for b, i in enumerate(rows)}
    Path(a.csv + ".json").write_text(json.dumps(info, indent=1))
    ncc = np.array([r["ncc_global"] for r in out])
    print(f"NCC median {np.median(ncc):.3f}, mean {ncc.mean():.3f}, NCC < 0.5: {np.mean(ncc < 0.5):.1%}"
          f"  |  {secs / n * 1e3:.1f} ms per item  |  wrote {a.csv}")


if __name__ == "__main__":
    main()
