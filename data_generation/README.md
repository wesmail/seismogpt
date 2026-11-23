# data_generation

Synthetic three-component seismograms for SeismoGPT, computed with
[Instaseis](https://instaseis.net) from a pre-computed AxiSEM database and
written in the [SeisBench](https://github.com/seisbench/seisbench) format.

| file | purpose |
|---|---|
| `waveform_generator.py` | samples sources and receivers, computes, processes and stores the seismograms |
| `cmt.py` | random moment tensor (CMT) rescaled to a requested magnitude Mw |
| `check_generator.py` | checks a generated file before it is used (see below) |

## How one trace is made

1. **Sample the event.** A scrambled Sobol sequence (seeded with `--seed`) gives
   depth, epicentral distance, azimuth, Mw and source position, spread evenly over
   the ranges given on the command line. Default ranges: depth 5–100 km,
   distance 10–90°, Mw 3–7.
2. **Source.** `cmt.py` draws a random moment tensor and scales it to the requested Mw.
3. **Seismogram.** Instaseis returns ground displacement (m) at the receiver in Z, N, E.
4. **Processing (causal).** A 10 s taper at the start of the record, then a one-pass
   (causal) 4-corner Butterworth band-pass from `--fmin` (0.02 Hz) to
   min(`--fmax`, 1/database period). That is 0.476 Hz for `ak135f_2s`.
   Each processed sample depends only on earlier samples.
5. **Arrival times.** TauP gives the first P and S times in the `--seimic_model`
   velocity model (default `ak135`).
6. **Store.** One SeisBench file per run: `<out>_<postfix>/metadata.csv` and
   `waveforms.hdf5`. With `--save_raw`, the unprocessed traces also go to
   `<out>_<postfix>_raw/`, with the same metadata.

## Usage

```bash
# small local test with the online database (no download needed)
python data_generation/waveform_generator.py --db syngine://ak135f_2s \
    --num_events 64 --out SmallV2 --postfix 1 --seed 1 --save_raw

# one training file as on the cluster (see slurm/)
python data_generation/waveform_generator.py --db /path/ak135f_2s/ \
    --num_events 34300 --out train --postfix 1 --seed 1 --exclude_dist 50 60 --save_raw
```

| option | default | meaning |
|---|---|---|
| `--db` | – | Instaseis database: a local folder or `syngine://<model>` |
| `--num_events` | 5000 | events to sample (before `--exclude_dist`) |
| `--out`, `--postfix` | `seisbench_data`, empty | output folder `<out>_<postfix>` |
| `--seed` | 42 | seeds the Sobol sequence and NumPy; use a different seed for every file |
| `--min_depth`, `--max_depth` | 5, 100 | source depth (km) |
| `--min_dist`, `--max_dist` | 10, 90 | epicentral distance (degrees) |
| `--mw_min`, `--mw_max` | 3, 7 | moment magnitude |
| `--exclude_dist LO HI` | off | drop events with LO ≤ distance < HI (held-out band) |
| `--fmin`, `--fmax` | 0.02, 1.0 | band-pass corners (Hz); `fmax` is capped at 1/database period |
| `--receivers_per_event` | 1 | receivers per event (one, as in the paper) |
| `--save_raw` | off | also store the unprocessed traces |
| `--dtype` | float32 | storage precision; processing always runs in float64 |

Event IDs are `<seed>_<index>`, so they are unique across files made with different seeds.

## Changes for the revision (v2)

| | v1 (submitted) | v2 |
|---|---|---|
| processing | whole-record demean and detrend, 5 % taper at both ends, zero-phase (forward and backward) band-pass | 10 s taper at the start, one-pass (causal) band-pass |
| array conversion | SeisBench `stream_to_array`, which also subtracts the whole-record mean | own conversion, no mean subtraction |
| P and S times | TauP time + 3.15 s source shift (counted twice: Instaseis already removes it) | TauP time |
| moment tensor | unscaled components stored (label and tensor differ by about 0.1 Mw) | components rescaled to the label Mw |
| source depth | km passed where Instaseis expects m | km × 1000 |
| sampling | uniform Sobol for the paper data, but the code had switched to a banded scheme with hard-coded ranges | uniform Sobol over the command-line ranges, as in the paper |
| held-out band | – | `--exclude_dist` |
| unprocessed copy | – | `--save_raw` |
| stored precision | float64 | float32 (`--dtype`) |
| metadata | labelled velocity, counts | labelled displacement, m; plus `filter_causal`, `generator_seed` |

The first four rows fix leaks of future information or wrong labels. The whole-record
statistics and the zero-phase filter made samples before the forecast start depend on
samples after it. See `diagnostics/` for the test that measures this.

## check_generator.py

Run it on every new file before using it. It needs the `--save_raw` copy.

```bash
python data_generation/check_generator.py --data_dir train_1 --exclude_dist 50 60 --n 3000
```

| check | passes when |
|---|---|
| 1 metadata columns | every column the dataset class and evaluation scripts read is present |
| 2 excluded band | no trace lies inside `--exclude_dist` (only if given) |
| 3 processed = process(raw) | processing the stored raw copy reproduces the stored processed copy (relative difference < 1e-4 for float32, which leaves room for rounding; a wrong processing step gives 0.1 or more) |
| 4 leak test | the samples after the S time are changed (×3 plus a large pulse) and processed again; everything before the S time is **exactly** unchanged |
| 5 Mw label | Mw computed from the stored moment tensor equals the label |
| 6 dataset class | `data/data_handling.py` loads the file (skipped if torch is not installed) |

`--n` runs checks 3 and 4 on that many random traces (default: all). The other checks
always use every trace. The exit code is 0 only if every check passes, so a batch job
can refuse a bad file (see `slurm/`).
