#!/usr/bin/env python3
"""Compare all evaluated models in one table per configuration.

Reads results/eval/<model>/rollout_metrics_<ratio>_<future>.csv (from evaluate.py),
keeps the items that every model was evaluated on, and reports for each model:

  typical     median of the metric
  average     mean and standard deviation
  tail        fraction of items with NCC < 0.5              (Reviewer 2, comment 7)
  paired      difference to the reference model, item by item (same items, same
              context): median difference with a 95 % bootstrap interval, and the
              win rate = fraction of items where the reference is better

With --data_dir and --band, the same table is also made for the items inside and
outside the held-out distance band (interpolation test, Reviewer 1).

  python baselines/compare.py --results results/eval --reference seismogpt \
      --data_dir /path/test --band 50 60 --out results/eval/comparison
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines.forecasters import MODELS                 # noqa: E402

CONFIGS = {"A": "1_120", "B": "1_240", "C": "2_240"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--results", default="results/eval", help="folder with one subfolder per model")
ap.add_argument("--reference", default="seismogpt")
ap.add_argument("--metric", default="ncc_global", help="any CSV column, e.g. ncc_Z, skill_global")
ap.add_argument("--tail", type=float, default=0.5, help="tail threshold for NCC")
ap.add_argument("--data_dir", help="test set, to split by distance (needs --band)")
ap.add_argument("--band", type=float, nargs=2, help="held-out distance band in degrees, e.g. 50 60")
ap.add_argument("--n_boot", type=int, default=2000)
ap.add_argument("--out", help="prefix for the summary CSV and LaTeX table")
a = ap.parse_args()
rng = np.random.default_rng(0)


def load(config):
    """{model: Series of the metric indexed by item}, only items common to all models."""
    found = {}
    for m in sorted(p.name for p in Path(a.results).iterdir() if p.is_dir()):
        f = Path(a.results) / m / f"rollout_metrics_{CONFIGS[config]}.csv"
        if f.exists():
            found[m] = pd.read_csv(f).set_index("item")[a.metric]
    common = sorted(set.intersection(*(set(s.index) for s in found.values()))) if found else []
    order = [m for m in MODELS if m in found] + [m for m in found if m not in MODELS]
    return {m: found[m].loc[common] for m in order}, common


def boot_median(d):
    """95 % bootstrap interval of the median of d (resampling items)."""
    i = rng.integers(0, len(d), size=(a.n_boot, len(d)))
    return np.percentile(np.median(d[i], axis=1), [2.5, 97.5])


def table(res, items):
    ref = res.get(a.reference)
    rows = []
    for m, s in res.items():
        v = s.loc[items].to_numpy()
        r = {"model": m, "n": len(v), "median": np.median(v), "mean": v.mean(), "std": v.std(),
             f"frac<{a.tail}": np.mean(v < a.tail) if a.metric.startswith("ncc") else np.nan}
        if ref is not None and m != a.reference:
            d = ref.loc[items].to_numpy() - v                  # > 0: reference better
            lo, hi = boot_median(d)
            r.update({"ref-model median": np.median(d), "CI low": lo, "CI high": hi,
                      "ref wins": np.mean(d > 0)})
        rows.append(r)
    return pd.DataFrame(rows)


subsets = {"all": None}
if a.data_dir and a.band:
    import seisbench.data as sbd
    dist = sbd.WaveformDataset(a.data_dir).metadata["distance_deg"].to_numpy()
    lo, hi = a.band
    subsets = {"all": None, f"inside {lo:g}-{hi:g} deg": (dist >= lo) & (dist < hi),
               f"outside {lo:g}-{hi:g} deg": ~((dist >= lo) & (dist < hi))}

pd.set_option("display.float_format", lambda x: f"{x:.3f}")
pd.set_option("display.width", 200)
summary = []
for config in CONFIGS:
    res, common = load(config)
    if not res:
        continue
    for name, mask in subsets.items():
        items = common if mask is None else [i for i in common if mask[i]]
        if not items:
            continue
        t = table(res, items)
        print(f"\n=== configuration {config} ({CONFIGS[config]}), {a.metric}, {name}: {len(items)} items ===")
        print(t.to_string(index=False))
        summary.append(t.assign(config=config, subset=name))

if a.out and summary:
    s = pd.concat(summary)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(f"{a.out}.csv", index=False)
    with open(f"{a.out}.tex", "w") as f:                     # compact table for the paper
        f.write("% config & subset & model & median & mean $\\pm$ std & tail & win rate\n")
        for _, r in s.iterrows():
            win = "" if pd.isna(r.get("ref wins", np.nan)) else f"{r['ref wins']:.0%}".replace("%", "\\%")
            tail = "" if pd.isna(r[f"frac<{a.tail}"]) else f"{r[f'frac<{a.tail}']:.0%}".replace("%", "\\%")
            f.write(f"{r.config} & {r.subset} & {r.model} & {r['median']:.2f} & "
                    f"{r['mean']:.2f} $\\pm$ {r['std']:.2f} & {tail} & {win} \\\\\n")
    print(f"\nwrote {a.out}.csv and {a.out}.tex")
