# baselines

Baselines for the revision (Reviewer 1, general comment 2; Reviewer 2, comment 2):
is SeismoGPT better than simpler ways of forecasting the same waveforms?

## Fair by construction

Every model, SeismoGPT included, goes through one evaluation script,
`evaluate.py`. It uses the same:

- test items (same seed as `quality_assurance/efficient_rollout.py`),
- window (from P, anchor-normalised, × 10),
- context length (r × (S − P)) and forecast length,
- metrics, the functions of `efficient_rollout.py`, with the same CSV columns.

The forecaster receives **only the context tokens**, so no model can see the future
it predicts. Models trained on data use the training part of the same training set,
the 80/10/10 split of `SeismicDataModule`. For SeismoGPT, `evaluate.py` gives exactly
the numbers of `efficient_rollout.py` (tested: difference 0).

| model | trained on | idea | file needed |
|---|---|---|---|
| `zero` | – | the ground returns to rest; reference with skill 0 | – |
| `persistence` | – | repeat the last context token | – |
| `ar_burg` | the context itself | linear prediction fitted to each trace (Burg's method, always stable) | – |
| `linear` | training set | one linear map: last 16 tokens (134 s) → next token | `linear_ar.npz` |
| `gru` | training set | recurrent network on the same tokens, same training setup | `gru.ckpt` |
| `knn` | training set | copy the continuation of the most similar training context (look-up) | `knn_lib/` |
| `knn_params` | training set | copy the continuation of the training trace with the most similar source and receiver (distance, depth, azimuth, mechanism) | `knn_lib/` |
| `seismogpt` | training set | the transformer of the paper | `seismogpt.ckpt` |

`knn` answers "does the model do more than look up a similar training trace?" Its
gap to SeismoGPT inside the held-out distance band (50–60°, not in training) is the
interpolation test.

`knn_params` is the retrieval exactly as Reviewer 1 phrased it ("most similar
training trace in parameter space"). It is an **oracle**: it uses the true source
parameters of the test event, which SeismoGPT never sees. Magnitude is left out
(after normalisation it only sets the amplitude). Distance, depth, azimuth and the
shape of the moment tensor are standardised and weighted equally, one weight per group.
Both look-ups share one library.

## Files

| file | purpose |
|---|---|
| `forecasters.py` | all models behind one interface: `forecast(context [T, C, K], n) → [n, C, K]` |
| `evaluate.py` | one model, one configuration → CSV (+ `.json` with settings and time per item) |
| `compare.py` | all models → one table per configuration (also inside/outside the held-out band) |
| `run_abc.sh` | evaluate all models on A, B, C, then compare |
| `train_linear.py` | closed-form least-squares fit of the linear model |
| `gru.py` | GRU model and its Lightning module (trained with `main.py`) |
| `build_knn_library.py` | stores training windows for the look-up |
| `common.py` | picks training items from the same split as the data module |

## Workflow

**1. Train or build** (once, on the v2 training set):
```bash
python main.py fit --config configs/baseline_gru.yaml                    # GPU, like SeismoGPT
python baselines/train_linear.py --data_dir /path/train --n_items 50000 --out linear_ar.npz
python baselines/build_knn_library.py --data_dir /path/train --n_items 200000 --out knn_lib
```
Set `data_dir` in `configs/baseline_gru.yaml` to the same folder as in `configs/train.yaml`.

**2. Evaluate and compare** (on the v2 test set):
```bash
DATA_DIR=/path/test CKPT=seismogpt.ckpt GRU_CKPT=gru.ckpt \
LINEAR=linear_ar.npz KNN_LIB=knn_lib ./baselines/run_abc.sh
```
Models without a file are skipped, and existing CSVs are kept (`FORCE=1` redoes them).
Settings: `N` (items, default 2000), `SEED`, `OUT` (default `results/eval`),
`BAND` (default `50 60`), `EXTRA` (more options for `evaluate.py`, e.g. `--skip-psd`).

Output:
```
results/eval/<model>/rollout_metrics_1_120.csv     configuration A  (same names as run_evaluation_abc.sh,
results/eval/<model>/rollout_metrics_1_240.csv     configuration B   so the plotting scripts in
results/eval/<model>/rollout_metrics_2_240.csv     configuration C   quality_assurance/ work per model)
results/eval/comparison.csv, comparison.tex        the table
```

**3. Read the table.** For each model it gives:
- typical: the median;
- average: mean ± std;
- tail: the fraction with NCC < 0.5 (Reviewer 2, comment 7);
- paired difference to SeismoGPT on the same items: the median of (SeismoGPT − model)
  with a 95 % bootstrap interval, and the win rate, the fraction of items where
  SeismoGPT is better.

Use another metric with `--metric`, e.g. `skill_global` or `ncc_Z`:
```bash
python baselines/compare.py --results results/eval --metric skill_global --out results/eval/skill
```

## Notes

- **GRU size.** The default (hidden 512, 3 layers) has about 5 M parameters; SeismoGPT
  has more. To compare at equal size, raise `hidden` in `configs/baseline_gru.yaml`.
  The parameter count is printed at the start of training.
- **GRU loss.** Log-cosh between the predicted and true next token, the main term of
  SeismoGPT's loss, computed without overflow. The multi-horizon, STFT and coherence
  terms are left out; say so in the paper.
- **kNN cost.** The search compares each test context with every library window in one
  matrix product. Use a GPU for libraries above about 50,000 windows. The library
  uses N × 192 tokens × 48 × 4 bytes (200,000 windows = 7.4 GB). `.json` lists the
  neighbour of every test item; `knn_lib/index.csv` gives its distance, depth and Mw.
- **Timing.** `seconds_per_item` in each `.json` is the forecast time per item, which
  is useful for the cost comparison with Instaseis (Reviewer 1, general comment 3).
- **Adding a model.** Write a class with `forecast(ctx, n)` in `forecasters.py`, and
  add it to `build()` and `MODELS`. Everything else works unchanged.
