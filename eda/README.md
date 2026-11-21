# eda

## eda_waveforms.py: look at generated waveforms

A quick visual check of a generated file, as the model will see it.

```bash
python eda/eda_waveforms.py --data_dir /path/train_1 --n 6 --out eda_plots
python eda/eda_waveforms.py --data_dir /path/train_1 --ids 3 17 --out eda_plots   # given traces
```

| option | default | meaning |
|---|---|---|
| `--data_dir` | – | processed SeisBench folder |
| `--raw_dir` | `<data_dir>_raw` if it exists | unprocessed copy, drawn for comparison |
| `--n` | 6 | number of example traces, spread evenly over distance |
| `--ids` | – | plot these trace indices instead |
| `--out` | `eda_plots` | output folder (not tracked by git) |

### Figures

**`summary.png`** covers all traces in the folder:
- (a) window peak / anchor peak against distance: how large values get after anchor
  scaling. S and surface waves are often tens to hundreds of times larger than the
  P onset.
- (b) anchor peak (m) against Mw, with the eps = 1e-8 of the submitted normaliser
  drawn for reference. Anchors below that line were dominated by eps in v1.

**`trace_<index>_d<distance>.png`**, one per example trace:
- left: the model window from P (5,184 samples = 324 tokens of 16), scaled as the
  model sees it (× 10 / anchor peak; the anchor is the first token, shared by Z, N, E).
  The anchor token is shaded, and the forecast starts of configurations A/B
  (r = 1) and C (r = 2) are marked.
- right: zoom on the P arrival, processed trace, and unprocessed trace if available.
  The causal filter delays and reshapes the onset slightly; this panel shows by how much.

Prints the median, 90th percentile and maximum of the ratio in (a), and the fraction
of anchors below 1e-8 m.
