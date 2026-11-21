#!/usr/bin/env python3
"""Checks for data made by waveform_generator.py v2 (JGR revision).

Needs a dataset written with --save_raw: <out>_<postfix> (processed) and
<out>_<postfix>_raw (unprocessed). Run from the repository root:

  python check_generator.py --data_dir SmallV2_1 --exclude_dist 50 60

Checks
  1. metadata has the columns the dataset class and evaluation scripts use
  2. no event inside the excluded distance band (if --exclude_dist given)
  3. processing the stored unprocessed trace reproduces the stored processed one
  4. LEAK TEST of the generator: change only the samples after the S time
     (x3 plus a large extra pulse), process again, compare everything before
     the S time. A causal generator gives exactly 0.
  5. magnitude computed from the stored moment tensor equals the label Mw
  6. (optional) the repository's dataset class loads the data
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import obspy
import pandas as pd

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data_dir", required=True, help="processed folder, e.g. SmallV2_1")
ap.add_argument("--raw_dir", default=None, help="default: <data_dir>_raw")
ap.add_argument("--gen_dir", default="data_generation", help="folder containing waveform_generator.py")
ap.add_argument("--exclude_dist", type=float, nargs=2, default=None)
a = ap.parse_args()

sys.path.insert(0, str(Path(a.gen_dir).resolve()))
sys.path.insert(0, str(Path(".").resolve()))
import seisbench.data as sbd                      # noqa: E402
from waveform_generator import process_stream     # noqa: E402  (the generator's own processing)

raw_dir = a.raw_dir or f"{a.data_dir.rstrip('/')}_raw"
meta = pd.read_csv(Path(a.data_dir) / "metadata.csv")
proc = sbd.WaveformDataset(a.data_dir)
raw = sbd.WaveformDataset(raw_dir)
fs = float(meta.trace_sampling_rate_hz.iloc[0])
fmin, fmax = float(meta.band_min_Hz.iloc[0]), float(meta.band_max_Hz.iloc[0])
print(f"{len(meta)} traces, fs = {fs:.4f} Hz, band {fmin}-{fmax} Hz\n")
results = []


def report(name, ok, detail):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def run(x):
    st = obspy.Stream([obspy.Trace(np.asarray(c, dtype=np.float64).copy(),
                                   header={"sampling_rate": fs}) for c in x])
    return np.stack([tr.data for tr in process_stream(st, fmin, fmax)])


# 1. columns
need = ["trace_p_arrival_s", "trace_s_arrival_s", "trace_sampling_rate_hz", "distance_deg",
        "src_depth_km", "Mw", "azimuth_deg", "m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp",
        "band_min_Hz", "band_max_Hz", "event_id"]
missing = [c for c in need if c not in meta.columns]
report("1 metadata columns", not missing, "none missing" if not missing else f"missing {missing}")

# 2. excluded band
if a.exclude_dist:
    lo, hi = a.exclude_dist
    n_in = int(((meta.distance_deg >= lo) & (meta.distance_deg < hi)).sum())
    report("2 excluded distance band", n_in == 0, f"{n_in} events in {lo}-{hi} deg")

# 3. reproduce processed from raw
d3 = 0.0
for i in range(len(meta)):
    p = proc.get_waveforms(i).astype(np.float64)
    d3 = max(d3, float(np.max(np.abs(run(raw.get_waveforms(i)) - p)) / np.max(np.abs(p))))
report("3 processed = process(raw)", d3 < 1e-9, f"max relative difference {d3:.1e}")

# 4. leak test of the generator's processing
worst, n_tested = 0.0, 0
for i in range(len(meta)):
    s_time = meta.trace_s_arrival_s.iloc[i]
    if not np.isfinite(s_time):
        continue
    x = raw.get_waveforms(i).astype(np.float64)
    t0 = int(s_time * fs)
    y = x.copy()
    y[:, t0:] *= 3.0
    y[:, t0 + 50:] += 5.0 * np.max(np.abs(x))
    a0, b0 = run(x)[:, :t0], run(y)[:, :t0]
    worst = max(worst, float(np.max(np.abs(b0 - a0))))
    n_tested += 1
report("4 leak test (future changed, past compared)", worst == 0.0,
       f"largest change before the S time: {worst} over {n_tested} traces (must be exactly 0)")

# 5. magnitude label vs tensor
M = np.stack([[meta.m_rr, meta.m_rt, meta.m_rp],
              [meta.m_rt, meta.m_tt, meta.m_tp],
              [meta.m_rp, meta.m_tp, meta.m_pp]]).transpose(2, 0, 1)
mw = (2.0 / 3.0) * np.log10(np.linalg.norm(M, axis=(1, 2)) / np.sqrt(2.0) * 1e7) - 10.7
d5 = float(np.max(np.abs(mw - meta.Mw)))
report("5 Mw label = Mw of the tensor", d5 < 1e-6, f"max difference {d5:.1e}")

# 6. dataset class (optional)
try:
    from data.data_handling import SeismicWaveformDataset
    item = SeismicWaveformDataset(data_dir=a.data_dir, kernel_size=16, stride=16, num_tokens=320,
                                  normalize=True, num_pred_horizons=4)[0]
    report("6 dataset class loads", True, str({k: tuple(v.shape) for k, v in item.items()}))
except ImportError as e:
    print(f"[skip] 6 dataset class: {e}")

print(f"\n{sum(results)} of {len(results)} checks passed")
print("Note: leak_test.py --data_dir on these files will still show ~0.66; that tests the "
      "normaliser in data_handling.py, which is fixed in the next step.")
