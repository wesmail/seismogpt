# waveform_generator.py
# ------------------------------------------------------------
# Wisely-sampled synthetic seismograms for foundation models
# - Sobol sampling over (depth, distance, azimuth, Mw)
# - Multiple receivers per source
# - SeisBench writer with rich, interpolation-friendly metadata
#
# v2 (JGR revision, Nov 2026): causal preprocessing, depth in metres,
# explicit displacement, reproducible moment tensors, unique event ids,
# optional held-out distance band and optional unprocessed copy.
# ------------------------------------------------------------

import argparse
import contextlib
from math import ceil, log2
from pathlib import Path

import numpy as np
import obspy
import instaseis
from obspy.taup import TauPyModel
from obspy.geodetics.base import degrees2kilometers

from tqdm.auto import tqdm

# SeisBench
import seisbench.data as sbd
import seisbench.util as sbu

# CMT helper (GCMT-like stats, Mw control)
from cmt import CMT

# Low-discrepancy sampler
from scipy.stats import qmc


def process_stream(stream, fmin, fmax, taper_s=10.0):
    """Causal preprocessing (v2).

    Every output sample depends only on input samples at the same time or
    earlier, so the record can later be cut into context and future without
    information leaking backwards (Reviewer 2, comment 5).

    v1 did: demean, linear detrend, 5 % taper at both ends, band-pass with
    zerophase=True.
    - demean / linear detrend removed: both are statistics of the whole record,
      future included; Instaseis synthetics start at rest, so they are not needed.
    - taper: now 10 s at the start only. The v1 taper (5 % of the record at each
      end, minutes long) depended on the record length and could reach the P
      wave at short distances; a forward-only filter needs no taper at the end.
    - band-pass: one forward pass (zerophase=False) instead of forward + backward.
    """
    stream.taper(max_percentage=None, max_length=taper_s, type="hann", side="left")
    stream.filter("bandpass", freqmin=fmin, freqmax=fmax, corners=4, zerophase=False)
    return stream


def stream_to_array(stream, component_order, dtype="float32"):
    """v2: like seisbench.util.stream_to_array, but WITHOUT its last step,
    which subtracts the mean of the whole record (a statistic that includes
    the future). Instaseis returns all components on the same time grid.
    v2: processing runs in float64; only the stored copy is cast to `dtype`
    (float32 halves the disk use and is what training uses anyway)."""
    return np.stack([stream.select(channel=f"??{c}")[0].data for c in component_order]).astype(dtype)


class SeismicSyntheticDataGenerator:
    def __init__(
        self,
        db_name: str,
        seimic_model: str = "prem",
        output_dir: str = "seisbench_data",
        output_postfix: str = "",
        # phase calculation
        phase_list=("P", "S"),
        # geometry envelope (local/regional default)
        target_depth_km=(0.0, 100.0),
        target_dist_deg=(0.5, 15.0),
        # magnitude range
        mw_range=(3.0, 7.0),
        # receivers per source (promotes interpolation)
        num_receivers_per_event: int = 4,
        # processing band (Hz)
        freq_band=(0.02, 1.0),
        # v2: distance band (deg) left out of this dataset, e.g. (50, 60)
        exclude_dist_deg=None,
        # v2: also write the unprocessed Instaseis traces to <output>_raw
        save_raw=False,
        dtype="float32",
    ):
        """
        A coverage-aware synthetic generator that produces datasets
        where models can *interpolate* across geometry and magnitude.
        """
        self.db = instaseis.open_db(db_name)
        self.seimic_model = seimic_model
        self.phase_list = list(phase_list)

        self.target_depth_km = target_depth_km
        self.target_dist_deg = target_dist_deg
        self.mw_range = mw_range
        self.num_receivers_per_event = int(num_receivers_per_event)
        self.freq_band = freq_band
        self.exclude_dist_deg = exclude_dist_deg
        self.save_raw = bool(save_raw)
        self.dtype = dtype

        self.output_dir = output_dir
        self.output_postfix = str(output_postfix)

    # --------------------
    # Geometry helpers
    # --------------------
    def dist_to_geo(self, azimuth_deg: float, distance_deg: float, start_lat: float, start_lon: float):
        """
        Great-circle forward problem on a spherical Earth (radius 6371 km).
        Returns dict with latitude, longitude (deg).
        """
        az = np.radians(azimuth_deg)
        dist_km = degrees2kilometers(distance_deg)
        R = 6371.0

        lat0 = np.radians(start_lat)
        lon0 = np.radians(start_lon)

        lat1 = np.arcsin(
            np.sin(lat0) * np.cos(dist_km / R) + np.cos(lat0) * np.sin(dist_km / R) * np.cos(az)
        )
        lon1 = lon0 + np.arctan2(
            np.sin(az) * np.sin(dist_km / R) * np.cos(lat0),
            np.cos(dist_km / R) - np.sin(lat0) * np.sin(lat1),
        )

        lat_deg = np.degrees(np.unwrap([0.0, lat1])[-1])
        lon_deg = np.degrees(np.unwrap([0.0, lon1])[-1])
        return {"latitude": lat_deg, "longitude": lon_deg}

    # --------------------
    # Core simulation
    # --------------------
    def simulate_event(
        self,
        num_events: int = 1000,
        components: str = "ZNE",
        random_seed: int = 42,
    ):
        """
        Generate events with Sobol coverage. Each event has 1 source and
        K receivers at evenly spaced azimuths (K = self.num_receivers_per_event).
        Data are written in SeisBench format (metadata.csv + waveforms.hdf5).
        """
        rng = np.random.default_rng(random_seed)
        np.random.seed(random_seed)  # v2: cmt.py draws moment tensors with np.random

        # Output paths
        base = Path(self.output_dir if self.output_postfix == "" else f"{self.output_dir}_{self.output_postfix}")
        base.mkdir(parents=True, exist_ok=True)
        metadata_path = base / "metadata.csv"
        waveforms_path = base / "waveforms.hdf5"

        # Sobol over: depth, distance, azimuth, Mw, src_lat, src_lon
        dmin, dmax = self.target_depth_km
        rmin, rmax = self.target_dist_deg
        mw_min, mw_max = self.mw_range

        # v2: uniform Sobol sampling over the requested ranges. This is the sampling
        # that produced the data of the submitted paper (in v1 it was commented out
        # in favour of a banded scheme with hard-coded ranges).
        sampler = qmc.Sobol(d=6, scramble=True, seed=random_seed)
        m = int(ceil(log2(num_events)))              # 2^m points avoids SciPy's warning
        U = sampler.random_base2(m=m)[:num_events, :]

        depths = dmin + (dmax - dmin) * U[:, 0]
        dist_deg = rmin + (rmax - rmin) * U[:, 1]
        az0_deg = 360.0 * U[:, 2]  # base azimuth for the event
        Mw_vals = mw_min + (mw_max - mw_min) * U[:, 3]
        src_lat = -90.0 + 180.0 * U[:, 4]
        src_lon = -180.0 + 360.0 * U[:, 5]

        # v2: optional held-out distance band (Reviewer 1): drop events inside it
        if self.exclude_dist_deg is not None:
            lo, hi = self.exclude_dist_deg
            keep = ~((dist_deg >= lo) & (dist_deg < hi))
            depths, dist_deg, az0_deg = depths[keep], dist_deg[keep], az0_deg[keep]
            Mw_vals, src_lat, src_lon = Mw_vals[keep], src_lat[keep], src_lon[keep]
            num_events = int(keep.sum())

        # SeisBench writer
        with contextlib.ExitStack() as stack:
            writer = stack.enter_context(sbd.WaveformDataWriter(metadata_path, waveforms_path))
            raw_writer = None
            if self.save_raw:  # v2: unprocessed copy, same metadata
                raw_base = Path(f"{base}_raw")
                raw_base.mkdir(parents=True, exist_ok=True)
                raw_writer = stack.enter_context(
                    sbd.WaveformDataWriter(raw_base / "metadata.csv", raw_base / "waveforms.hdf5"))
            data_format = {
                "dimension_order": "CW",
                "component_order": components,
                "measurement": "displacement",  # v2: was "velocity" (Instaseis default is displacement)
                "unit": "m",                    # v2: was "counts"
                "instrument_response": "not restituted",
            }
            writer.data_format = data_format
            if raw_writer is not None:
                raw_writer.data_format = data_format

            taup = TauPyModel(self.seimic_model)
            period = float(self.db.info.period)
            src_shift = float(self.db.info.src_shift)

            for i in tqdm(range(num_events), desc="Simulating events"):
                event_id = f"{random_seed}_{i:06d}"  # v2: unique across shards (seed differs)

                # Source (CMT with fixed Mw for coverage control)
                cmt = CMT(Mw=float(Mw_vals[i]))
                # v2: take the components from cmt.mt, which cmt.py rescales to the
                # requested Mw; cmt.mrr etc. are the unscaled draws (off by ~0.1 Mw)
                mrr, mtt, mpp = (float(cmt.mt[0, 0]), float(cmt.mt[1, 1]), float(cmt.mt[2, 2]))
                mrt, mrp, mtp = (float(cmt.mt[0, 1]), float(cmt.mt[0, 2]), float(cmt.mt[1, 2]))
                source = instaseis.Source(
                    origin_time=obspy.UTCDateTime(),
                    latitude=float(src_lat[i]),
                    longitude=float(src_lon[i]),
                    depth_in_m=float(depths[i]) * 1000.0,  # v2: was km passed as metres
                    m_rr=mrr,
                    m_tt=mtt,
                    m_pp=mpp,
                    m_rt=mrt,
                    m_rp=mrp,
                    m_tp=mtp,
                )

                # Evenly spaced receivers around azimuth base (optionally jitter distance slightly)
                for k in range(self.num_receivers_per_event):
                    azi_k = (az0_deg[i] + 360.0 * k / self.num_receivers_per_event) % 360.0
                    dist_k = float(dist_deg[i])  # could add tiny jitter if desired

                    recv_geo = self.dist_to_geo(
                        azimuth_deg=float(azi_k),
                        distance_deg=dist_k,
                        start_lat=float(src_lat[i]),
                        start_lon=float(src_lon[i]),
                    )

                    # Short station for Instaseis: max 5 chars, e.g. S0000, S1234, etc.
                    # Here: S + (event_id mod 1000 as 3 digits) + receiver index (0–9) = 5 chars
                    station_code = f"S{(i % 1000):03d}{k}"

                    receiver = instaseis.Receiver(
                        latitude=recv_geo["latitude"],
                        longitude=recv_geo["longitude"],
                        network="IV",
                        station=station_code,
                    )

                    # Generate seismograms
                    stream = self.db.get_seismograms(source=source, receiver=receiver,
                                                     components=components, kind="displacement")

                    # --- Pre-processing (v2: causal, see process_stream) ---
                    raw_stream = stream.copy() if raw_writer is not None else None
                    process_stream(stream, self.freq_band[0], min(self.freq_band[1], 1.0 / period))

                    # Travel times (guard lengths)
                    arrivals = taup.get_travel_times(
                        source_depth_in_km=float(depths[i]),
                        distance_in_degree=dist_k,
                        phase_list=self.phase_list,
                    )
                    # Extract first P and S if present
                    p_time_s = np.nan
                    s_time_s = np.nan
                    for arr in arrivals:
                        # TauP returns multiple phases; pick first P and first S seen
                        if np.isnan(p_time_s) and arr.name.upper().startswith("P"):
                            p_time_s = float(arr.time)  # v2: no + src_shift; Instaseis already removes it
                        if np.isnan(s_time_s) and arr.name.upper().startswith("S"):
                            s_time_s = float(arr.time)  # v2: no + src_shift
                        if not np.isnan(p_time_s) and not np.isnan(s_time_s):
                            break

                    # Convert to array for SeisBench
                    # component_order is respected by stream_to_array
                    data = stream_to_array(stream, components, self.dtype)  # v2: no whole-record demean

                    # Compute back-azimuth at the RECEIVER (direction receiver→source).
                    # If azimuth_deg is defined as source→receiver, back-azimuth = azimuth + 180 (mod 360).
                    back_azimuth_deg = (float(azi_k) + 180.0) % 360.0

                    # Boolean availability flags for masking multitask losses etc.
                    has_P = int(np.isfinite(p_time_s))
                    has_S = int(np.isfinite(s_time_s))

                    # Metadata (interpolation-friendly)
                    # Use *seconds* suffix for times to avoid confusion with "samples".
                    trace_meta = {
                            # IDs
                            "source_id": "synthetic",
                            "event_id": event_id,
                            "trace_id": f"SY.{event_id}..MX",
                            "station": station_code,
                            # Timing / sampling
                            "trace_start_time": stream[0].stats.starttime,
                            "trace_end_time": stream[0].stats.endtime,
                            "trace_sampling_rate_hz": float(stream[0].stats.sampling_rate),
                            "trace_npts": int(stream[0].stats.npts),
                            "trace_component_order": components,
                            # Phase arrivals (s)
                            "trace_p_arrival_s": p_time_s,
                            "trace_s_arrival_s": s_time_s,
                            # Phase availability flags (for masked auxiliary losses / evaluation)
                            "has_P": has_P,                       # 0/1 int
                            "has_S": has_S,                       # 0/1 int                            
                            # Geometry
                            "src_lat": float(src_lat[i]),
                            "src_lon": float(src_lon[i]),
                            "src_depth_km": float(depths[i]),
                            "recv_lat": float(receiver.latitude),
                            "recv_lon": float(receiver.longitude),
                            "distance_deg": dist_k,
                            "azimuth_deg": float(azi_k),
                            "back_azimuth_deg": back_azimuth_deg, # receiver -> source (added)
                            # Magnitude & MT
                            "Mw": float(cmt.Mw),
                            "m_rr": mrr,
                            "m_tt": mtt,
                            "m_pp": mpp,
                            "m_rt": mrt,
                            "m_rp": mrp,
                            "m_tp": mtp,
                            # Provenance / processing
                            "earth_model": self.seimic_model,
                            "band_min_Hz": float(self.freq_band[0]),
                            "band_max_Hz": float(min(self.freq_band[1], 1.0 / period)),
                            "filter_causal": 1,              # v2
                            "generator_seed": int(random_seed),  # v2
                    }
                    writer.add_trace(trace_meta, data)
                    if raw_writer is not None:  # v2
                        raw_data = stream_to_array(raw_stream, components, self.dtype)
                        raw_writer.add_trace(trace_meta, raw_data)


def main():
    parser = argparse.ArgumentParser(description="[Generator] Sobol-sampled seismic events for SeisBench.")
    parser.add_argument("--db", type=str, required=False,
                        default="/scratch/tmp/wesmail/ak135f_2s/", help="Instaseis DB path or syngine URI")
    parser.add_argument("--seimic_model", type=str, default="ak135",
                        help="TauP model (v2 default ak135, to match the ak135f database; v1: prem)")
    parser.add_argument("--num_events", type=int, default=5000, help="Number of sources to simulate")
    parser.add_argument("--receivers_per_event", type=int, default=1, help="Receivers per source (paper: 1)")
    parser.add_argument("--out", type=str, default="seisbench_data", help="Output dir base name")
    parser.add_argument("--postfix", type=str, default="", help="Output dir postfix (e.g., 01)")
    parser.add_argument("--min_depth", type=float, default=5.0)
    parser.add_argument("--max_depth", type=float, default=100.0)  # v2: paper range
    parser.add_argument("--min_dist", type=float, default=10.0)
    parser.add_argument("--max_dist", type=float, default=90.0)   # v2: paper range
    parser.add_argument("--mw_min", type=float, default=3.0)
    parser.add_argument("--mw_max", type=float, default=7.0)
    parser.add_argument("--components", type=str, default="ZNE")
    parser.add_argument("--fmin", type=float, default=0.02)
    parser.add_argument("--fmax", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42,
                        help="use a different seed for every shard, and separate seeds for test sets")
    parser.add_argument("--exclude_dist", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                        help="v2: leave out this distance band (deg), e.g. 50 60 for training")
    parser.add_argument("--save_raw", action="store_true",
                        help="v2: also write the unprocessed traces to <out>_<postfix>_raw")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                        help="v2: storage precision (processing is always float64)")
    args = parser.parse_args()

    gen = SeismicSyntheticDataGenerator(
        db_name=args.db,
        seimic_model=args.seimic_model,
        output_dir=args.out,
        output_postfix=args.postfix,
        phase_list=("P", "S"),
        target_depth_km=(args.min_depth, args.max_depth),
        target_dist_deg=(args.min_dist, args.max_dist),
        mw_range=(args.mw_min, args.mw_max),
        num_receivers_per_event=args.receivers_per_event,
        freq_band=(args.fmin, args.fmax),
        exclude_dist_deg=args.exclude_dist,
        save_raw=args.save_raw,
        dtype=args.dtype,
    )

    gen.simulate_event(
        num_events=args.num_events,
        components=args.components,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
