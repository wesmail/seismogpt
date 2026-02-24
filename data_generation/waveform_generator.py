# waveform_generator.py
# ------------------------------------------------------------
# Wisely-sampled synthetic seismograms for foundation models
# - Sobol sampling over (depth, distance, azimuth, Mw)
# - Multiple receivers per source
# - SeisBench writer with rich, interpolation-friendly metadata
# ------------------------------------------------------------

import argparse
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

        # Output paths
        base = Path(self.output_dir if self.output_postfix == "" else f"{self.output_dir}_{self.output_postfix}")
        base.mkdir(parents=True, exist_ok=True)
        metadata_path = base / "metadata.csv"
        waveforms_path = base / "waveforms.hdf5"

        # Sobol over: depth, distance, azimuth, Mw, src_lat, src_lon
        dmin, dmax = self.target_depth_km
        rmin, rmax = self.target_dist_deg
        mw_min, mw_max = self.mw_range

        #sampler = qmc.Sobol(d=6, scramble=True, seed=random_seed)
        #U = sampler.random(num_events)

        #depths = dmin + (dmax - dmin) * U[:, 0]
        #dist_deg = rmin + (rmax - rmin) * U[:, 1]
        #az0_deg = 360.0 * U[:, 2]  # base azimuth for the event
        #Mw_vals = mw_min + (mw_max - mw_min) * U[:, 3]
        #src_lat = -90.0 + 180.0 * U[:, 4]
        #src_lon = -180.0 + 360.0 * U[:, 5]

        # -----------------------------------------------------------------------------------------------------
        # More controlled Sobol sampling
        # Distances: overall 0.2°–25°, split as
        # 40% in 0.2–3° (local)
        # 40% in 3–10° (regional)
        # 20% in 10–25° (long regional / surface-wave rich)
        # Depths (km):
        # 70% in 0–30 km (shallow crust)
        # 20% in 30–70 km (intermediate)
        # 10% in 70–150 km (deep-ish)
        # Azimuths: 360° uniform
        # Magnitudes: Mw range
        # Source latitudes: -90°–90° uniform
        # Source longitudes: -180°–180° uniform
        # -----------------------------------------------------------------------------------------------------
        # --- Sobol engine ---
        sampler = qmc.Sobol(d=6, scramble=True, seed=random_seed)

        # Use 2^m samples to avoid the SciPy "power of 2" warning
        m = int(ceil(log2(num_events)))
        U = sampler.random_base2(m=m)[:num_events, :]

        u_depth = U[:, 0]
        u_dist  = U[:, 1]
        u_az    = U[:, 2]
        u_Mw    = U[:, 3]
        u_lat   = U[:, 4]
        u_lon   = U[:, 5]

        # ------------------------------------------------------------
        # 1) Distance [deg] with 3 bands: 0.2–3, 3–10, 10–25
        #    - 40% local       (0.2–3)
        #    - 40% regional    (3–10)
        #    - 20% long-regional (10–25)
        # ------------------------------------------------------------
        dist_deg = np.empty(num_events, dtype=float)

        # Band 1: u_dist in [0, 0.4) -> [0.2, 3]
        mask1 = u_dist < 0.4
        u1 = u_dist[mask1] / 0.4
        dist_deg[mask1] = 0.2 + (3.0 - 0.2) * u1

        # Band 2: u_dist in [0.4, 0.8) -> [3, 10]
        mask2 = (u_dist >= 0.4) & (u_dist < 0.8)
        u2 = (u_dist[mask2] - 0.4) / 0.4
        dist_deg[mask2] = 3.0 + (10.0 - 3.0) * u2

        # Band 3: u_dist in [0.8, 1.0] -> [10, 25]
        mask3 = u_dist >= 0.8
        u3 = (u_dist[mask3] - 0.8) / 0.2
        dist_deg[mask3] = 10.0 + (25.0 - 10.0) * u3

        # ------------------------------------------------------------
        # 2) Depth [km] with 3 bands: 0–30, 30–70, 70–150
        #    - 70% shallow (0–30)
        #    - 20% mid     (30–70)
        #    - 10% deep    (70–150)
        # ------------------------------------------------------------
        depths = np.empty(num_events, dtype=float)

        # Shallow: u_depth in [0, 0.7) -> [0, 30]
        mask1 = u_depth < 0.7
        u1 = u_depth[mask1] / 0.7
        depths[mask1] = 0.0 + (30.0 - 0.0) * u1

        # Mid: u_depth in [0.7, 0.9) -> [30, 70]
        mask2 = (u_depth >= 0.7) & (u_depth < 0.9)
        u2 = (u_depth[mask2] - 0.7) / 0.2
        depths[mask2] = 30.0 + (70.0 - 30.0) * u2

        # Deep: u_depth in [0.9, 1.0] -> [70, 150]
        mask3 = u_depth >= 0.9
        u3 = (u_depth[mask3] - 0.9) / 0.1
        depths[mask3] = 70.0 + (150.0 - 70.0) * u3

        # ------------------------------------------------------------
        # 3) Other parameters as before
        # ------------------------------------------------------------
        mw_min, mw_max = self.mw_range

        az0_deg = 360.0 * u_az
        Mw_vals = mw_min + (mw_max - mw_min) * u_Mw
        src_lat = -90.0 + 180.0 * u_lat
        src_lon = -180.0 + 360.0 * u_lon        
        # -----------------------------------------------------------------------------------------------------

        # SeisBench writer
        with sbd.WaveformDataWriter(metadata_path, waveforms_path) as writer:
            writer.data_format = {
                "dimension_order": "CW",
                "component_order": components,
                "measurement": "velocity",
                "unit": "counts",
                "instrument_response": "not restituted",
            }

            taup = TauPyModel(self.seimic_model)
            period = float(self.db.info.period)
            src_shift = float(self.db.info.src_shift)

            for i in tqdm(range(num_events), desc="Simulating events"):
                event_id = f"{i:06d}"

                # Source (CMT with fixed Mw for coverage control)
                cmt = CMT(Mw=float(Mw_vals[i]))
                source = instaseis.Source(
                    origin_time=obspy.UTCDateTime(),
                    latitude=float(src_lat[i]),
                    longitude=float(src_lon[i]),
                    depth_in_m=float(depths[i]),
                    m_rr=cmt.mrr,
                    m_tt=cmt.mtt,
                    m_pp=cmt.mpp,
                    m_rt=cmt.mrt,
                    m_rp=cmt.mrp,
                    m_tp=cmt.mtp,
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
                    stream = self.db.get_seismograms(source=source, receiver=receiver, components=components)

                    # --- Pre-processing (standardized) ---
                    stream.detrend("demean")
                    stream.detrend("linear")
                    stream.taper(max_percentage=0.05, type="hann")
                    stream.filter(
                        "bandpass",
                        freqmin=self.freq_band[0],
                        freqmax=min(self.freq_band[1], 1.0 / period),
                        corners=4,
                        zerophase=True,
                    )

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
                            p_time_s = float(arr.time + src_shift)
                        if np.isnan(s_time_s) and arr.name.upper().startswith("S"):
                            s_time_s = float(arr.time + src_shift)
                        if not np.isnan(p_time_s) and not np.isnan(s_time_s):
                            break

                    # Convert to array for SeisBench
                    # component_order is respected by stream_to_array
                    _, data, _ = sbu.stream_to_array(stream, component_order=components)

                    # Compute back-azimuth at the RECEIVER (direction receiver→source).
                    # If azimuth_deg is defined as source→receiver, back-azimuth = azimuth + 180 (mod 360).
                    back_azimuth_deg = (float(azi_k) + 180.0) % 360.0

                    # Boolean availability flags for masking multitask losses etc.
                    has_P = int(np.isfinite(p_time_s))
                    has_S = int(np.isfinite(s_time_s))

                    # Metadata (interpolation-friendly)
                    # Use *seconds* suffix for times to avoid confusion with "samples".
                    writer.add_trace(
                        {
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
                            "m_rr": float(cmt.mrr),
                            "m_tt": float(cmt.mtt),
                            "m_pp": float(cmt.mpp),
                            "m_rt": float(cmt.mrt),
                            "m_rp": float(cmt.mrp),
                            "m_tp": float(cmt.mtp),
                            # Provenance / processing
                            "earth_model": self.seimic_model,
                            "band_min_Hz": float(self.freq_band[0]),
                            "band_max_Hz": float(min(self.freq_band[1], 1.0 / period)),
                        },
                        data,
                    )


def main():
    parser = argparse.ArgumentParser(description="[Generator] Sobol-sampled seismic events for SeisBench.")
    parser.add_argument("--db", type=str, required=False,
                        default="/scratch/tmp/wesmail/ak135f_2s/", help="Instaseis DB path or syngine URI")
    parser.add_argument("--seimic_model", type=str, default="prem", help="TauP model (prem/ak135/iasp91)")
    parser.add_argument("--num_events", type=int, default=5000, help="Number of sources to simulate")
    parser.add_argument("--receivers_per_event", type=int, default=2, help="Receivers per source")
    parser.add_argument("--out", type=str, default="seisbench_data", help="Output dir base name")
    parser.add_argument("--postfix", type=str, default="", help="Output dir postfix (e.g., 01)")
    parser.add_argument("--min_depth", type=float, default=5.0)
    parser.add_argument("--max_depth", type=float, default=20.0)
    parser.add_argument("--min_dist", type=float, default=10.0)
    parser.add_argument("--max_dist", type=float, default=40.0)
    parser.add_argument("--mw_min", type=float, default=3.0)
    parser.add_argument("--mw_max", type=float, default=7.0)
    parser.add_argument("--components", type=str, default="ZNE")
    parser.add_argument("--fmin", type=float, default=0.02)
    parser.add_argument("--fmax", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
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
    )

    gen.simulate_event(
        num_events=args.num_events,
        components=args.components,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
