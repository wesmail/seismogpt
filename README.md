# SeismoGPT

**Autoregressive seismic waveform prediction** — a foundation-style model for generating or predicting seismic waveforms in the time domain.

---

## Repository structure

```
.
├── configs
│   └── train.yaml
├── data_generation
│   ├── cmt.py
│   └── waveform_generator.py
├── data_handling
│   └── data_handling.py
├── main.py
├── models
│   ├── lightning_module.py
│   └── models.py
└── quality_assurance
    ├── plot_global_metrics_from_csv.py
    ├── plot_rollout_metrics.py
    ├── testset_rollout.py
    └── utils.py
```

---

## Data generation: install and usage

The `data_generation` module builds Sobol-sampled synthetic seismograms for training SeismoGPT. It uses **Instaseis** (Green’s function DB) and **ObsPy** for propagation, plus **SeisBench** for writing datasets.

### Installing dependencies

From the project root, use a dedicated environment (recommended):

**Option A — Conda (recommended for ObsPy + Instaseis)**

```bash
conda create -n seismogpt python=3.10
conda activate seismogpt
conda install -c conda-forge obspy instaseis
pip install numpy scipy tqdm seisbench
```

**Option B — pip only**

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install numpy scipy tqdm obspy instaseis seisbench
```

You also need an **Instaseis database** (precomputed Green’s functions). The script’s default path is `--db /scratch/tmp/wesmail/ak135f_2s/`. Use your own DB path or a [Syngine](https://docs.obspy.org/packages/obspy.clients.syngine.html) URI. Public DBs can be built with Instaseis or downloaded from community sources.

### Running the waveform generator

Run from the **`data_generation`** directory so that the local `cmt` module is importable:

```bash
cd data_generation
python waveform_generator.py --db /path/to/your/instaseis_db --num_events 5000 --out seisbench_data
```

**Main arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--db` | `/scratch/tmp/wesmail/ak135f_2s/` | Instaseis DB path or Syngine URI |
| `--seimic_model` | `prem` | TauP model: `prem`, `ak135`, or `iasp91` |
| `--num_events` | `5000` | Number of sources to simulate |
| `--receivers_per_event` | `2` | Receivers per source |
| `--out` | `seisbench_data` | Output directory (SeisBench-style dataset) |
| `--postfix` | `""` | Optional postfix for the output dir (e.g. `01`) |
| `--min_depth`, `--max_depth` | `5.0`, `20.0` | Source depth range (km) |
| `--min_dist`, `--max_dist` | `10.0`, `40.0` | Epicentral distance range (degrees) |
| `--mw_min`, `--mw_max` | `3.0`, `7.0` | Moment magnitude range |
| `--components` | `ZNE` | Components to export (e.g. `ZNE`, `Z`) |
| `--fmin`, `--fmax` | `0.02`, `1.0` | Bandpass band (Hz) |
| `--seed` | `42` | Random seed for Sobol sampling |

**Example (custom geometry and output):**

```bash
cd data_generation
python waveform_generator.py \
  --db /data/instaseis/ak135f_2s \
  --num_events 10000 \
  --receivers_per_event 4 \
  --min_depth 0 --max_depth 50 \
  --min_dist 1 --max_dist 15 \
  --out my_dataset \
  --postfix regional
```

Output is written under `my_dataset_regional/` (or `seisbench_data` if no `--out`/`--postfix`) in SeisBench format, ready for use with the training data module (e.g. `data_handling.SeismicDataModule`).

---

## Training a model

Training uses **PyTorch** and **PyTorch Lightning** (Lightning CLIs for config-driven runs). Data is expected in SeisBench format (e.g. from the data generation step above).

### Installing dependencies

From the project root, in a dedicated environment:

**If you already have the data-generation env**, add the training stack:

```bash
conda activate seismogpt   # or your venv
pip install torch lightning h5py
```

**New env for training only:**

```bash
conda create -n seismogpt-train python=3.10
conda activate seismogpt-train
pip install torch lightning numpy scipy h5py seisbench
```

For GPU training, install a CUDA-enabled PyTorch build (see [pytorch.org](https://pytorch.org/get-started/locally/) for your CUDA version).

### Configuring the run

All training and data options are set in **`configs/train.yaml`**.

1. **Set the data path**  
   Under `data.init_args`, set `data_dir` to the **absolute path** of your SeisBench-style dataset (the directory that contains the waveform data and metadata):

   ```yaml
   data:
     class_path: data_handling.data_handling.SeismicDataModule
     init_args:
       data_dir: /path/to/your/seisbench_dataset   # <-- change this
       kernel_size: 16
       stride: 16
       batch_size: 96
       # ...
   ```

2. **Adjust other parameters (optional)**  
   In the same file you can change:
   - **trainer**: `max_epochs`, `accelerator`, `devices`, `precision`, `accumulate_grad_batches`, etc.
   - **model**: architecture (`d_model`, `num_layers`, `num_heads`, `kernel_size`, `num_tokens`, …), loss (`time_loss`), learning rate (`lr`), scheduler, scheduled-sampling options.
   - **data**: `batch_size`, `num_workers`, `kernel_size`, `stride`, `num_tokens`, `normalize`, `freq_norm`.

   Keep `kernel_size`, `num_tokens`, and `freq_norm` (and related model options) consistent between data and model.

### Running training

From the **project root** (so `models` and `data_handling` are on the Python path):

```bash
python main.py fit --config configs/train.yaml
```

Checkpoints and TensorBoard logs are written according to `trainer.callbacks` and `trainer.logger` in the config (default: checkpoints and `./logs/`).

---

## Quality assurance

The **`quality_assurance`** folder contains scripts to evaluate a trained model with autoregressive rollouts and to produce publication-ready metric plots.

| File | Purpose |
|------|--------|
| **`testset_rollout.py`** | Runs autoregressive (or teacher-forced) rollout on a test set: loads a checkpoint, runs the model on context windows, and predicts the future. Writes per-sample metrics (SNR, NCC, PSD, forecast skill) to a CSV. |
| **`plot_rollout_metrics.py`** | Reads the rollout CSV and builds plots of metrics vs. metadata (e.g. distance, depth, magnitude): histograms, hexbins, and summary figures. |
| **`plot_global_metrics_from_csv.py`** | Builds global metric summaries (e.g. violin/box plots) from one or more rollout CSV files for cross-run comparison. |
| **`utils.py`** | Shared metric helpers (SNR, normalized cross-correlation, PSD metrics, best-shift alignment). |
| **`run_rollout_and_plot.sh`** | One-shot script: runs `testset_rollout.py` then `plot_rollout_metrics.py` with the same CSV. |

We provide a trained checkpoint at **`quality_assurance/best.ckpt`** so you can run evaluation without training first.

### Running evaluation with the trained model

From the **`quality_assurance`** directory, use the bash script. You must pass the dataset path, the checkpoint path, and a metadata CSV (used for plotting). The script runs the rollout and then generates the rollout metrics plots.

**Minimal run** (only required arguments; other options use defaults):

```bash
cd quality_assurance
./run_rollout_and_plot.sh \
  --data_dir /path/to/dataset \
  --checkpoint "logs/version_0/checkpoints/epoch=25-step=121888.ckpt" \
  --metadata-csv /path/to/metadata_data_50.csv
```

**Using the provided checkpoint:**

```bash
cd quality_assurance
./run_rollout_and_plot.sh \
  --data_dir /path/to/dataset \
  --checkpoint best.ckpt \
  --metadata-csv /path/to/metadata_data_50.csv
```

**Full example** (override rollout length, sample count, and output paths):

```bash
./run_rollout_and_plot.sh \
  --data_dir /path/to/dataset \
  --checkpoint my.ckpt \
  --metadata-csv /path/to/metadata.csv \
  --context_secs 60 \
  --future_secs 120 \
  --num_samples 5000 \
  --csv rollout_metrics_earlycoda.csv \
  --output-dir metrics_plots_earlycoda/
```

**Required arguments:** `--data_dir` (SeisBench dataset root), `--checkpoint` (`.ckpt` file), `--metadata-csv` (CSV used by the plotting step).

**Optional (rollout):** `--kernel_size`, `--num_tokens`, `--context_secs`, `--future_secs`, `--num_samples`, `--csv`, `--mode` (`free` or `teacher`), `--output` (rollout figure), `--seed`.  
**Optional (plots):** `--output-dir` (defaults to `metrics_plots_<csv_stem>/`).

The script writes the metrics CSV (e.g. `rollout_metrics_longhorizon.csv`) and saves figures into the chosen output directory.