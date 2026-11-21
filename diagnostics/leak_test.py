#!/usr/bin/env python3
"""
Future-leakage test for the SeismoGPT preprocessing
(JGR-MLC revision 2026JH001579, Reviewer 2, comment 5).

The question: does the model input BEFORE the forecast start t0 depend on
samples AFTER t0?  Test: change only the samples after t0, rerun the
preprocessing, and measure how much the input before t0 moves.
A causal pipeline gives exactly zero.

Two stages are tested, exactly as they are in the repository:

  Stage 1  generator  (data_generation/waveform_generator.py, L277-286)
           whole record: demean, linear detrend, 5 % Hann taper,
           4-corner Butterworth band-pass with zerophase=True
  Stage 2  dataset    (data/data_handling.py, L236-244)
           whole (T+H)*K window, context AND future together:
           per-channel demean, linear detrend, divide by per-channel peak, x10

Modes
  python leak_test.py --demo
      Stand-in record, no data needed.  Shows the size of each leak and that
      a causal chain gives exactly zero.
  python leak_test.py --data_dir /path/PaperTestSet -n 1000 [--norm_mode legacy_peak]
      Stage 2 on the stored test traces (they already went through stage 1),
      with the new anchor normaliser (default; must give exactly 0) or the
      submitted one (legacy_peak), picking the same items as
      efficient_rollout.py (same seed).  Also
      reports where the window peak sits, raw amplitudes vs the eps in the
      normaliser, and the sampled parameter ranges (for Table 1).
  python leak_test.py --data_dir /path/PaperTestSet --db /path/ak135f_2s -n 20
      ...
  python leak_test.py --raw_dir /path/SmallV2_1_raw
      Unprocessed traces stored by the new generator (--save_raw). Measures the
      leak of the SUBMITTED stage 1 (whole-record demean/detrend, 5 % taper,
      zero-phase band-pass, SeisBench's mean subtraction) on real waveforms, the
      whole submitted pipeline (stage 1 + 2), and, for contrast, the new
      generator's processing (must be exactly 0).
      Regenerates the first n traces with Instaseis from the stored metadata,
      checks which settings reproduce the stored data (depth units,
      displacement vs velocity), then runs stages 1+2 end to end on raw data.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.signal as scipy_signal

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ----------------------------------------------------------------------------
# Stage 2: the dataset normaliser. Imported from the repo when possible.
# ----------------------------------------------------------------------------
try:
    from data.data_handling import SeismicWaveformDataset

    _normalize = SeismicWaveformDataset._normalize_strain_max_per_channel
    _anchor = getattr(SeismicWaveformDataset, "_normalize_anchor", None)
    NORM_SOURCE = "imported from data/data_handling.py"
except Exception as exc:  # torch / lightning / seisbench missing
    _anchor = None

    def _normalize(strain, demean=True, detrend=False, amp_norm="peak", eps=1e-8):
        # verbatim copy of SeismicWaveformDataset._normalize_strain_max_per_channel
        if demean:
            strain = strain - np.mean(strain, axis=1, keepdims=True)
        if detrend:
            strain = scipy_signal.detrend(strain, axis=1)
        if amp_norm == "peak":
            denom = np.max(np.abs(strain), axis=1, keepdims=True) + eps
            strain = strain / denom
        elif amp_norm == "std":
            denom = np.std(strain, axis=1, keepdims=True) + eps
            strain = strain / denom
        elif amp_norm is not None:
            raise ValueError("amp_norm must be one of: 'peak', 'std', 'rms', None")
        return strain

    NORM_SOURCE = f"verbatim copy (repo import failed: {type(exc).__name__})"

EPS_NORM = 1e-8          # eps inside the normaliser
K = 16                   # samples per token
T_TOKENS, H = 320, 4     # window = (T + H) * K samples, as in __getitem__
WIN = (T_TOKENS + H) * K


def dataset_stage(segment):
    """The SUBMITTED __getitem__ normalisation (norm_mode="legacy_peak")."""
    return _normalize(segment, demean=True, detrend=True, amp_norm="peak") * 10.0


def dataset_stage_anchor(segment):
    """The v2 __getitem__ normalisation (norm_mode="anchor"), from the repo."""
    if _anchor is None:
        raise RuntimeError("data_handling.py has no _normalize_anchor: update it, or use "
                           "--norm_mode legacy_peak")
    return _anchor(segment, K)[0] * 10.0


def dataset_stage_causal(segment, n_anchor=K):
    """Leak-free reference: one scale for all channels, from the first token
    only (keeps component ratios).  For comparison, not a recommendation."""
    s = np.max(np.abs(segment[:, :n_anchor])) + EPS_NORM
    return segment / s * 10.0


# ----------------------------------------------------------------------------
# Stage 1: the generator's processing chain, reproduced with ObsPy.
# ----------------------------------------------------------------------------
FMAX_DATA = 1.0 / 2.1      # band edge of the data: 1 / database period (2.1 s)


def generator_stage(data, fs, fmin=0.02, fmax=FMAX_DATA):
    import obspy

    st = obspy.Stream([obspy.Trace(data=np.asarray(c, dtype=np.float64).copy(),
                                   header={"sampling_rate": fs}) for c in data])
    st.detrend("demean")
    st.detrend("linear")
    st.taper(max_percentage=0.05, type="hann")
    st.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=True)
    return np.stack([tr.data for tr in st])


def generator_stage_causal(data, fs, fmin=0.02, fmax=FMAX_DATA, n_pre=None):
    """Leak-free reference: demean with the pre-P part only, taper only the
    start (a fixed window), one forward pass of the band-pass."""
    n_pre = n_pre or int(60 * fs)
    x = np.asarray(data, dtype=np.float64)
    x = x - x[:, :n_pre].mean(axis=1, keepdims=True)
    n_tap = max(1, int(0.05 * x.shape[1]))
    ramp = np.ones(x.shape[1])
    ramp[:n_tap] = 0.5 * (1 - np.cos(np.pi * np.arange(n_tap) / n_tap))
    sos = scipy_signal.butter(4, [fmin, fmax], btype="band", fs=fs, output="sos")
    return scipy_signal.sosfilt(sos, x * ramp, axis=1)


# ----------------------------------------------------------------------------
# Perturbations of the future and the leak measure
# ----------------------------------------------------------------------------
def perturb_scale(x, t0, factor=3.0):
    """A larger future: every sample from t0 on, times `factor`."""
    y = np.array(x, dtype=np.float64, copy=True)
    y[:, t0:] *= factor
    return y


def perturb_arrival(x, t0, fs, delay_s=None, f0=0.1, amp=5.0):
    """An extra arrival after t0 (Ricker, centre frequency f0), `amp` times
    the context peak.  Default delay 1.5/f0 (at least 30 s), so the wavelet
    is negligible before t0."""
    delay_s = max(30.0, 1.5 / f0) if delay_s is None else delay_s
    y = np.array(x, dtype=np.float64, copy=True)
    n = y.shape[1]
    t = (np.arange(n) - (t0 + int(delay_s * fs))) / fs
    a = (np.pi * f0 * t) ** 2
    ricker = (1 - 2 * a) * np.exp(-a)
    ricker[: t0] = 0.0                      # strictly after t0
    peak = np.max(np.abs(y[:, :t0])) + 1e-30
    return y + amp * peak * ricker[None, :]


def leak(stage, x, x_pert, t0, fs, tol=1e-3):
    """Relative change of the input before t0, and how far back it reaches.

    Returns (rel, reach_s):
      rel      max |stage(x_pert) - stage(x)| / max |stage(x)| over [0, t0)
      reach_s  seconds before t0 of the earliest sample that moved by more
               than tol * context peak (0 if nothing moved)
    """
    a = stage(x)[:, :t0]
    b = stage(x_pert)[:, :t0]
    diff = np.max(np.abs(b - a), axis=0)
    ref = np.max(np.abs(a)) + 1e-30
    rel = float(diff.max() / ref)
    moved = np.nonzero(diff > tol * ref)[0]
    reach_s = float((t0 - moved[0]) / fs) if moved.size else 0.0
    return rel, reach_s


def fmt(rel, reach):
    return "exactly 0" if rel == 0 else f"{rel:9.3g}   (reaches {reach:6.0f} s back)"


# ----------------------------------------------------------------------------
# Mode 1: stand-in record
# ----------------------------------------------------------------------------
def demo_record(fs, dur=3600.0, tP=400.0, tS=700.0, tR=950.0):
    """Noise-free stand-in: small P, larger S, much larger surface waves."""
    t = np.arange(int(dur * fs)) / fs

    def packet(t_on, f, w):
        u = np.clip(t - t_on, 0, None)
        return (u / w) * np.exp(-u / w) * np.sin(2 * np.pi * f * u)

    z = 1.0 * packet(tP, 0.20, 10) + 3.0 * packet(tS, 0.12, 20) + 20 * packet(tR, 0.05, 120)
    n = 0.4 * packet(tP, 0.20, 10) + 5.0 * packet(tS, 0.10, 20) + 15 * packet(tR + 40, 0.04, 150)
    e = 0.3 * packet(tP, 0.20, 10) + 4.0 * packet(tS, 0.11, 20) + 25 * packet(tR + 60, 0.045, 150)
    return np.stack([z, n, e]), tP, tS


def run_demo(fs=1.9, ratio=1.0):
    raw, tP, tS = demo_record(fs)
    p = int(tP * fs)
    t_ctx = int(ratio * (tS - tP) * fs) // K          # as efficient_rollout.py
    t0_win = t_ctx * K                               # forecast start in window
    t0_raw = p + t0_win                              # same instant in record
    print(f"stand-in record, fs = {fs} Hz, context = {ratio} x (S-P) = "
          f"{t0_win / fs:.0f} s, forecast starts {t0_win} samples into the window\n")

    def window(rec):
        seg = rec[:, p: p + WIN]
        return np.pad(seg, ((0, 0), (0, WIN - seg.shape[1])))

    print("relative change of the input before t0 when only the future changes")
    print("-" * 74)
    for name, pert in [("future x3", lambda x, t0: perturb_scale(x, t0)),
                       ("extra arrival 30 s after t0",
                        lambda x, t0: perturb_arrival(x, t0, fs))]:
        print(f"[{name}]")
        filt = generator_stage(raw, fs)
        r = leak(lambda x: generator_stage(x, fs), raw, pert(raw, t0_raw), t0_raw, fs)
        print(f"  stage 1  generator, as in repo     : {fmt(*r)}")
        r = leak(lambda x: generator_stage_causal(x, fs, n_pre=int(300 * fs)),
                 raw, pert(raw, t0_raw), t0_raw, fs)
        print(f"  stage 1  causal reference          : {fmt(*r)}")
        seg = window(filt)
        r = leak(dataset_stage, seg, pert(seg, t0_win), t0_win, fs)
        print(f"  stage 2  dataset, as in repo       : {fmt(*r)}")
        r = leak(dataset_stage_causal, seg, pert(seg, t0_win), t0_win, fs)
        print(f"  stage 2  causal reference          : {fmt(*r)}")
        end2end = lambda x: dataset_stage(window(generator_stage(x, fs)))
        a, b = end2end(raw)[:, :t0_win], end2end(pert(raw, t0_raw))[:, :t0_win]
        rel = float(np.max(np.abs(b - a)) / np.max(np.abs(a)))
        print(f"  stages 1+2, what the model sees    : {rel:9.3g}\n")

    seg = window(generator_stage(raw, fs))
    in_future = np.argmax(np.abs(seg), axis=1) >= t0_win
    print(f"window peak lies after t0 on channels Z,N,E: {in_future.tolist()}")
    print(f"normaliser: {NORM_SOURCE}")


# ----------------------------------------------------------------------------
# Mode 2: stored test traces (stage 2) + amplitude and parameter checks
# ----------------------------------------------------------------------------
def load_items(data_dir, n, seed):
    import seisbench.data as sbd

    ds = sbd.WaveformDataset(Path(data_dir))
    meta = ds.metadata
    rng = np.random.default_rng(seed)                # same as efficient_rollout.py
    items = np.sort(rng.choice(len(meta), size=min(n, len(meta)), replace=False))
    return ds, meta, items


def window_from_trace(wf, p_sec, sr):
    p = int(p_sec * sr) if np.isfinite(p_sec) else 0
    seg = wf[:, p: p + WIN].astype(np.float64)
    return np.pad(seg, ((0, 0), (0, WIN - seg.shape[1])))


def forecast_start(row, sr, ratio):
    ps = row.get("trace_s_arrival_s", np.nan) - row.get("trace_p_arrival_s", np.nan)
    secs = ratio * ps if np.isfinite(ps) and ps > 0 else 60.0
    return max(1, int(secs * sr) // K) * K


def run_data(data_dir, n, seed, ratio, norm_mode="anchor"):
    ds, meta, items = load_items(data_dir, n, seed)
    sr = float(meta["trace_sampling_rate_hz"].iloc[0])
    stage = dataset_stage_anchor if norm_mode == "anchor" else dataset_stage
    print(f"{len(items)} items from {data_dir}, fs = {sr:.4f} Hz, "
          f"context = {ratio} x (S-P), normaliser {NORM_SOURCE}, norm_mode = {norm_mode}\n")

    rels, peak_future, raw_peak, mws = [], [], [], []
    for i in items:
        row = meta.iloc[i]
        seg = window_from_trace(ds.get_waveforms(int(i)), row["trace_p_arrival_s"], sr)
        t0 = forecast_start(row, sr, ratio)
        rels.append(leak(stage, seg, perturb_scale(seg, t0), t0, sr)[0])
        x = seg - seg.mean(axis=1, keepdims=True)
        x = scipy_signal.detrend(x, axis=1)
        peak_future.append(np.argmax(np.abs(x), axis=1) >= t0)
        raw_peak.append(np.max(np.abs(x), axis=1))
        mws.append(row.get("Mw", np.nan))
    rels, peak_future = np.array(rels), np.array(peak_future)
    raw_peak, mws = np.array(raw_peak), np.array(mws)

    print(f"stage 2 leak (future x3, norm_mode {norm_mode}), relative change of the context")
    print(f"  median {np.median(rels):.3g}   90th pct {np.percentile(rels, 90):.3g}   "
          f"max {np.max(rels):.3g}   exactly zero in {np.mean(rels == 0):.1%} of items")
    if norm_mode == "anchor":
        anchors = np.array([np.max(np.abs(window_from_trace(ds.get_waveforms(int(i)),
                            meta.iloc[i]["trace_p_arrival_s"], sr)[:, :K])) for i in items])
        print(f"  anchor (peak of the first token): min {anchors.min():.2e}   median {np.median(anchors):.2e}"
              f"   zero in {np.mean(anchors == 0):.1%} of items")
    print("  window peak after t0 (what set the scale in the legacy normaliser):  "
          + "  ".join(f"{c} {m:.1%}" for c, m in zip("ZNE", peak_future.mean(0))))

    print("\nlegacy normaliser only: raw window peak vs eps = 1e-8 in the normaliser "
          "(eps > 1 % of the peak means the scaled trace no longer peaks at 1)")
    eps_share = EPS_NORM / (raw_peak + EPS_NORM)
    edges = np.arange(np.floor(np.nanmin(mws)), np.ceil(np.nanmax(mws)) + 0.5, 0.5)
    print("  Mw bin      n   median raw peak (Z)   median scaled peak   eps > 1 %")
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (mws >= lo) & (mws < hi)
        if m.sum():
            print(f"  {lo:.1f}-{hi:.1f}  {m.sum():5d}   {np.median(raw_peak[m, 0]):16.3e}   "
                  f"{np.median(1 - eps_share[m, 0]):17.3f}   {np.mean(eps_share[m] > 0.01):8.1%}")

    print("\nsampled parameters in this set (min / median / max)")
    for col in ["distance_deg", "src_depth_km", "Mw", "azimuth_deg"]:
        if col in meta:
            v = meta[col].to_numpy(dtype=float)
            print(f"  {col:13s} {np.nanmin(v):9.2f} {np.nanmedian(v):9.2f} {np.nanmax(v):9.2f}")
    if "event_id" in meta:
        print(f"  traces {len(meta)}, events {meta['event_id'].nunique()}")
    return ds, meta, items, sr


# ----------------------------------------------------------------------------
# Mode 3: regenerate with Instaseis, check settings, end-to-end leak on raw data
# ----------------------------------------------------------------------------
def run_db(db_path, ds, meta, items, sr, n_db, ratio):
    import instaseis
    import obspy

    db = instaseis.open_db(db_path)
    period = float(db.info.period)
    row0 = meta.iloc[items[0]]
    fmin = float(row0.get("band_min_Hz", 0.02))
    fmax = float(row0.get("band_max_Hz", min(1.0, 1.0 / period)))
    f0 = float(np.sqrt(fmin * fmax))                 # perturbation inside the band
    print(f"\nInstaseis database {db_path}, period {period} s, band {fmin}-{fmax} Hz "
          f"(from metadata), output resampled to {sr:.4f} Hz")

    def ncc(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

    variants = [("depth km as metres, displacement", 1.0, "displacement"),
                ("depth km as metres, velocity", 1.0, "velocity"),
                ("depth in metres (x1000), displacement", 1000.0, "displacement"),
                ("depth in metres (x1000), velocity", 1000.0, "velocity")]
    scores = {v[0]: [] for v in variants}
    e2e = []
    ok = [i for i in items if np.isfinite(meta.iloc[i].get("trace_p_arrival_s", np.nan))
          and np.isfinite(meta.iloc[i].get("trace_s_arrival_s", np.nan))]
    print(f"{len(items) - len(ok)} of {len(items)} items have no P or S time and are skipped here")
    import warnings
    warnings.filterwarnings("ignore", message="Source depth cannot be changed")
    if not getattr(db.info, "is_reciprocal", True):
        print("NOTE: forward database with one fixed source depth; Instaseis ignores the "
              "depth passed, so the depth-units rows below cannot differ")
    for i in ok[:n_db]:
        row = meta.iloc[i]
        mt = {k: float(row[k]) for k in ["m_rr", "m_tt", "m_pp", "m_rt", "m_rp", "m_tp"]}
        rec = instaseis.Receiver(latitude=float(row["recv_lat"]), longitude=float(row["recv_lon"]),
                                 network="IV", station="S0000")
        stored = ds.get_waveforms(int(i)).astype(np.float64)
        raws = {}
        for name, mult, kind in variants:
            src = instaseis.Source(latitude=float(row["src_lat"]), longitude=float(row["src_lon"]),
                                   depth_in_m=float(row["src_depth_km"]) * mult,
                                   origin_time=obspy.UTCDateTime(0), **mt)
            st = db.get_seismograms(source=src, receiver=rec, components="ZNE", kind=kind,
                                    dt=1.0 / sr)
            raw = np.stack([tr.data for tr in st])
            raws[name] = raw
            proc = generator_stage(raw, sr, fmin, fmax)
            L = min(proc.shape[1], stored.shape[1])
            scores[name].append(np.mean([ncc(proc[c, :L], stored[c, :L]) for c in range(3)]))

        # end-to-end leak on the raw record of the best-matching variant so far
        best = max(scores, key=lambda k: np.mean(scores[k]))
        raw = raws[best]
        p = int(float(row["trace_p_arrival_s"]) * sr)
        t0w = forecast_start(row, sr, ratio)

        def model_input(x):
            proc = generator_stage(x, sr, fmin, fmax)
            return dataset_stage(window_from_trace(proc, float(row["trace_p_arrival_s"]), sr))

        a = model_input(raw)[:, :t0w]
        b = model_input(perturb_arrival(raw, p + t0w, sr, f0=f0))[:, :t0w]
        e2e.append(float(np.max(np.abs(b - a)) / np.max(np.abs(a))))

    print("which generation settings reproduce the stored traces (mean NCC):")
    for name in scores:
        print(f"  {name:40s} {np.mean(scores[name]):.4f}")
    print(f"\nend to end (stages 1+2, extra arrival at {f0:.3g} Hz after t0), "
          f"relative change of the model input before t0:")
    print(f"  median {np.median(e2e):.3g}   max {np.max(e2e):.3g}   over {len(e2e)} items")


def old_stage1(data, fs, fmin, fmax):
    """Stage 1 exactly as submitted: generator processing, then the mean
    subtraction at the end of seisbench.util.stream_to_array."""
    x = generator_stage(data, fs, fmin, fmax)
    return x - x.mean(axis=1, keepdims=True)


def run_raw(raw_dir, ratio, gen_dir="data_generation"):
    import seisbench.data as sbd

    ds = sbd.WaveformDataset(Path(raw_dir))
    meta = ds.metadata
    fs = float(meta["trace_sampling_rate_hz"].iloc[0])
    fmin, fmax = float(meta["band_min_Hz"].iloc[0]), float(meta["band_max_Hz"].iloc[0])
    f0 = float(np.sqrt(fmin * fmax))
    try:
        sys.path.insert(0, str(Path(gen_dir).resolve()))
        from waveform_generator import process_stream
        import obspy

        def new_stage1(x):
            st = obspy.Stream([obspy.Trace(np.asarray(c, np.float64).copy(),
                                           header={"sampling_rate": fs}) for c in x])
            return np.stack([tr.data for tr in process_stream(st, fmin, fmax)])
    except ImportError as exc:
        new_stage1 = None
        print(f"NOTE: new generator not tested: cannot import waveform_generator from "
              f"'{gen_dir}' ({exc}). Install the missing package (e.g. pip install instaseis) "
              f"or pass --gen_dir.\n")
    print(f"{len(meta)} unprocessed traces from {raw_dir}, fs = {fs:.4f} Hz, band {fmin:.3g}-{fmax:.3g} Hz, "
          f"context = {ratio} x (S-P)\n")

    res = {k: [] for k in ["s1_x3", "s1_arr", "e2e_arr", "new", "new_e2e"]}
    for i in range(len(meta)):
        row = meta.iloc[i]
        p_sec = row.get("trace_p_arrival_s", np.nan)
        if not (np.isfinite(p_sec) and np.isfinite(row.get("trace_s_arrival_s", np.nan))):
            continue
        x = ds.get_waveforms(i).astype(np.float64)
        t0w = forecast_start(row, fs, ratio)           # forecast start inside the window
        t0 = int(p_sec * fs) + t0w                      # the same instant in the record
        s1 = lambda y: old_stage1(y, fs, fmin, fmax)
        res["s1_x3"].append(leak(s1, x, perturb_scale(x, t0), t0, fs))
        xa = perturb_arrival(x, t0, fs, f0=f0)
        res["s1_arr"].append(leak(s1, x, xa, t0, fs))
        model_input = lambda y: dataset_stage(window_from_trace(s1(y), p_sec, fs))
        a0, b0 = model_input(x)[:, :t0w], model_input(xa)[:, :t0w]
        res["e2e_arr"].append(float(np.max(np.abs(b0 - a0)) / np.max(np.abs(a0))))
        if new_stage1 is not None:
            res["new"].append(leak(new_stage1, x, xa, t0, fs)[0])
            if _anchor is not None:      # new generator + new normaliser, end to end
                new_input = lambda y: dataset_stage_anchor(window_from_trace(new_stage1(y), p_sec, fs))
                a1, b1 = new_input(x)[:, :t0w], new_input(perturb_scale(xa, t0))[:, :t0w]
                res["new_e2e"].append(float(np.max(np.abs(b1 - a1))))

    def summ(v):
        v = np.asarray(v, dtype=float)
        return f"median {np.median(v):.3g}   90th pct {np.percentile(v, 90):.3g}   max {np.max(v):.3g}"

    for key, label in [("s1_x3", "future x3"), ("s1_arr", f"extra arrival ({f0:.3g} Hz) after t0")]:
        rel = [r[0] for r in res[key]]
        reach = [r[1] for r in res[key]]
        print(f"submitted stage 1 (zero-phase filter etc.), {label}:")
        print(f"  relative change of the context:  {summ(rel)}")
        print(f"  reaches back (s, change > 0.1 %): {summ(reach)}")
    print("submitted pipeline end to end (stage 1 + normaliser), extra arrival after t0:")
    print(f"  relative change of the model input before t0: {summ(res['e2e_arr'])}")
    if res["new"]:
        print("new generator processing, extra arrival after t0:")
        print(f"  largest relative change: {np.max(res['new'])}   "
              f"exactly zero in {np.mean(np.asarray(res['new']) == 0):.0%} of traces")
    if res["new_e2e"]:
        print("NEW pipeline end to end (new generator + anchor normaliser), future x3 + extra arrival:")
        print(f"  largest change of the model input before t0: {np.max(res['new_e2e'])}   "
              f"exactly zero in {np.mean(np.asarray(res['new_e2e']) == 0):.0%} of traces")
    print(f"\n{len(res['s1_x3'])} traces tested")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--data_dir")
    ap.add_argument("--db", help="Instaseis database (needs --data_dir)")
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("--n_db", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--context_ratio", type=float, default=1.0)
    ap.add_argument("--fs", type=float, default=1.9, help="sampling rate for --demo")
    ap.add_argument("--raw_dir", help="unprocessed traces from the new generator (--save_raw)")
    ap.add_argument("--gen_dir", default="data_generation", help="folder of waveform_generator.py")
    ap.add_argument("--norm_mode", default="anchor", choices=["anchor", "legacy_peak"],
                    help="which dataset normaliser --data_dir tests (default: the new one)")
    a = ap.parse_args()

    if a.raw_dir:
        run_raw(a.raw_dir, a.context_ratio, a.gen_dir)
        return
    if a.demo or not a.data_dir:
        run_demo(a.fs, a.context_ratio)
        return
    ds, meta, items, sr = run_data(a.data_dir, a.n, a.seed, a.context_ratio, a.norm_mode)
    if a.db:
        run_db(a.db, ds, meta, items, sr, a.n_db, a.context_ratio)


if __name__ == "__main__":
    main()
