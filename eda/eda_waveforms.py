#!/usr/bin/env python3
"""Look at waveforms from the new generator (JGR revision).

For a few traces spread over distance, one figure each:
  left   the whole model window (from P), as the model would see it with the
         anchor normaliser (scale = peak of the first token, shared by Z, N, E,
         times 10), with P, S and the forecast starts of configurations A/B and C
  right  zoom on the P arrival: processed trace, and the unprocessed one
         (if <data_dir>_raw exists), with the anchor token shaded
plus one summary figure for the whole folder:
  (a) window peak / anchor peak against distance
  (b) anchor peak (metres) against magnitude, with eps = 1e-8 for reference

  python eda/eda_waveforms.py --data_dir SmallV2_1 --n 6 --out eda_plots
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seisbench.data as sbd

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data_dir", required=True)
ap.add_argument("--raw_dir", default=None, help="default: <data_dir>_raw, if it exists")
ap.add_argument("--n", type=int, default=6, help="number of example traces, spread over distance")
ap.add_argument("--ids", type=int, nargs="*", default=None, help="plot these trace indices instead")
ap.add_argument("--out", default="eda_plots")
a = ap.parse_args()

K, WIN = 16, (320 + 4) * 16                      # token and window length, as in the dataset class
BLUE, ORANGE, GREEN, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
                     "legend.frameon": False, "figure.constrained_layout.use": True})

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
ds = sbd.WaveformDataset(a.data_dir)
m = ds.metadata.reset_index(drop=True)
raw_dir = a.raw_dir or f"{a.data_dir.rstrip('/')}_raw"
raw = sbd.WaveformDataset(raw_dir) if Path(raw_dir).exists() else None
fs = float(m.trace_sampling_rate_hz.iloc[0])
ok = np.isfinite(m.trace_p_arrival_s) & np.isfinite(m.trace_s_arrival_s)
print(f"{len(m)} traces ({int(ok.sum())} with P and S), fs = {fs:.4f} Hz, raw copy: {'yes' if raw else 'no'}")


def window(x, p):
    w = x[:, p:p + WIN].astype(np.float64)
    return np.pad(w, ((0, 0), (0, WIN - w.shape[1])))


# ------------------------------------------------------------------ summary over all traces
ratio, anchor = np.full(len(m), np.nan), np.full(len(m), np.nan)
for i in np.nonzero(ok.to_numpy())[0]:
    w = window(ds.get_waveforms(int(i)), int(m.trace_p_arrival_s[i] * fs))
    anchor[i] = np.abs(w[:, :K]).max()
    ratio[i] = np.abs(w).max() / anchor[i] if anchor[i] > 0 else np.nan
fig, axs = plt.subplots(1, 2, figsize=(8, 3))
axs[0].scatter(m.distance_deg, ratio, s=10, color=BLUE)
axs[0].set_yscale("log"); axs[0].set_xlabel("distance (deg)"); axs[0].set_ylabel("window peak / anchor peak")
axs[0].set_title("(a) range of values after anchor scaling")
axs[1].scatter(m.Mw, anchor, s=10, color=BLUE)
axs[1].axhline(1e-8, color=ORANGE, ls="--", lw=1); axs[1].text(m.Mw.min(), 1.4e-8, "eps = 1e-8", color=INK2)
axs[1].set_yscale("log"); axs[1].set_xlabel("Mw"); axs[1].set_ylabel("anchor peak (m)")
axs[1].set_title("(b) size of the P onset")
fig.savefig(out / "summary.png", dpi=150); plt.close(fig)
print(f"window peak / anchor peak: median {np.nanmedian(ratio):.0f}, 90% {np.nanpercentile(ratio, 90):.0f}, "
      f"max {np.nanmax(ratio):.0f}; anchors below 1e-8 m: {np.mean(anchor[ok] < 1e-8):.0%}")

# ------------------------------------------------------------------ example traces
idx = np.nonzero(ok.to_numpy())[0]
if a.ids:
    picks = a.ids
else:
    order = idx[np.argsort(m.distance_deg.to_numpy()[idx])]
    picks = order[np.linspace(0, len(order) - 1, min(a.n, len(order))).round().astype(int)]

for i in picks:
    r = m.iloc[i]
    tP, tS = float(r.trace_p_arrival_s), float(r.trace_s_arrival_s)
    p = int(tP * fs)
    x = ds.get_waveforms(int(i)).astype(np.float64)
    w = window(x, p)
    scale = np.abs(w[:, :K]).max()
    z = 10.0 * w / scale                                   # what the model would see
    t = np.arange(WIN) / fs                                # seconds after P
    starts = {"A/B start (r=1)": tS - tP, "C start (r=2)": 2 * (tS - tP)}

    fig, axs = plt.subplots(3, 2, figsize=(10, 5.6), sharex="col",
                            gridspec_kw={"width_ratios": [2.3, 1]})
    for c, comp in enumerate("ZNE"):
        ax = axs[c, 0]
        ax.plot(t, z[c], color=INK, lw=0.6)
        ax.axvspan(0, K / fs, color=GREEN, alpha=0.25, lw=0)
        ax.axvline(0, color=BLUE, lw=1)
        for (lab, ts), ls in zip(starts.items(), ["--", ":"]):
            ax.axvline(ts, color=ORANGE, lw=1, ls=ls, label=lab)
        ax.set_ylabel(f"{comp}  (x10 / anchor)")
        if c == 0:
            ax.legend(loc="upper right", fontsize=7)
            ax.set_title("model window from P, anchor-scaled (green: anchor token)")

        ax = axs[c, 1]
        zt = np.arange(x.shape[1]) / fs - tP
        sel = (zt > -30) & (zt < 40)
        ax.axvspan(0, K / fs, color=GREEN, alpha=0.25, lw=0)
        if raw is not None:
            xr = raw.get_waveforms(int(i)).astype(np.float64)
            ax.plot(zt[sel], xr[c, sel] / scale, color=INK2, lw=0.8, ls="--", label="unprocessed")
        ax.plot(zt[sel], x[c, sel] / scale, color=BLUE, lw=1.0, label="processed")
        ax.axvline(0, color=BLUE, lw=1)
        if c == 0:
            ax.legend(loc="upper left", fontsize=7)
            ax.set_title("zoom on P (blue line: stored P time)")
    axs[2, 0].set_xlabel("time after P (s)")
    axs[2, 1].set_xlabel("time relative to P (s)")
    fig.suptitle(f"trace {i}: distance {r.distance_deg:.1f} deg, depth {r.src_depth_km:.0f} km, "
                 f"Mw {r.Mw:.2f}, azimuth {r.azimuth_deg:.0f} deg, anchor {scale:.1e} m")
    name = out / f"trace_{i:05d}_d{r.distance_deg:04.1f}.png"
    fig.savefig(name, dpi=150); plt.close(fig)
    print("wrote", name)
