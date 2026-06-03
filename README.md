# SeismoGPT

Official code repository for the paper:

**[Data-Driven Forecasting of three-Component Seismograms Using Transformer Architectures](https://arxiv.org/abs/2606.02912v1)**  
Waleed Esmail, Stuart Russell, Jana Klinge, Alexander Kappes, Christine Thomas — [arXiv:2606.02912v1](https://arxiv.org/abs/2606.02912v1) (2026)

**Autoregressive seismic waveform prediction** — a causal RoPE transformer over Z/N/E waveform tokens, trained with PyTorch Lightning. The model forecasts three-component seismograms in the time domain from context starting at the P-wave arrival through a distance-normalized window beyond the S-wave arrival, then autoregressively continues the motion (evaluation configurations **A**, **B**, and **C** in the paper).

**Pre-trained Phase 1 weights:** [wesmail/SeismoGPT on Hugging Face](https://huggingface.co/wesmail/SeismoGPT)

---

## Repository layout

```
.
├── configs/
│   └── train.yaml              # LightningCLI training config
├── data/
│   └── data_handling.py        # SeisBench dataset + SeismicDataModule
├── data_generation/
│   ├── cmt.py
│   └── waveform_generator.py   # Synthetic training data (Instaseis + ObsPy)
├── models/
│   ├── lightning_module.py
│   └── models.py
├── quality_assurance/          # Test-set rollouts and metric figures
│   ├── efficient_rollout.py    # Fast metrics-only rollout → CSV
│   ├── plot_global_metrics_from_csv.py
│   ├── plot_ncc_corner.py      # NCC / SRR / PSD vs Δ, depth, Mw
│   ├── plot_ncc_heatmap.py
│   ├── make_paper_figures.py
│   ├── rollout_filtered_catalog_plots.py
│   └── utils.py
├── scripts/
│   └── download_seismogpt.sh   # Fetch checkpoint from Hugging Face
├── main.py
└── run_evaluation_abc.sh       # Configs A/B/C rollout + plots (one command)
```

---

## Quick start: model + evaluation

### 1. Install dependencies

```bash
conda create -n seismogpt python=3.10
conda activate seismogpt
pip install torch lightning seisbench pyyaml matplotlib scipy tqdm h5py
pip install huggingface_hub   # optional, for downloading weights
```

### 2. Get the trained model

**Option A — Hugging Face (recommended)**

```bash
# from repository root
./scripts/download_seismogpt.sh
# → phase1/epoch=12-step=633750.ckpt
```

Or:

```bash
pip install huggingface_hub
huggingface-cli download wesmail/SeismoGPT \
  epoch=12-step=633750.ckpt train_phase1_logcosh.yaml \
  --local-dir phase1
```

**Option B — local checkpoint**

Place your `.ckpt` anywhere and pass `--checkpoint /path/to/file.ckpt` to the scripts below.

**Load in Python** (from repository root):

```python
from models.lightning_module import GPTLightning

model = GPTLightning.load_from_checkpoint("phase1/epoch=12-step=633750.ckpt")
model.eval()
```

Hub page: [https://huggingface.co/wesmail/SeismoGPT](https://huggingface.co/wesmail/SeismoGPT)

### 3. Run paper configurations A, B, C

You need a **SeisBench-style test dataset** (`--data-dir`) and a **metadata CSV** (`--metadata-csv`) with columns used for plotting (e.g. `distance_deg`, `src_depth_km`, `Mw` — see plotting scripts for defaults).

| Config | Context ratio | Prediction horizon | Output CSV |
|--------|---------------|--------------------|------------|
| **A** | 1 × (S−P) | 120 s | `rollout_metrics_1_120.csv` |
| **B** | 1 × (S−P) | 240 s | `rollout_metrics_1_240.csv` |
| **C** | 2 × (S−P) | 240 s | `rollout_metrics_2_240.csv` |

**Full pipeline** (rollouts + global box plots + physics corner plots):

```bash
./run_evaluation_abc.sh \
  --download-hf \
  --data-dir /path/to/PaperTestSet \
  --metadata-csv /path/to/metadata.csv \
  --output-dir results/evaluation_abc
```

With a local checkpoint:

```bash
./run_evaluation_abc.sh \
  --data-dir /path/to/PaperTestSet \
  --metadata-csv /path/to/metadata.csv \
  --checkpoint phase1/epoch=12-step=633750.ckpt \
  --output-dir results/evaluation_abc
```

By default, each configuration evaluates **1000** events (`-n 1000`). Pass a larger `-n` or set `N_SAMPLES` to cover more of the test set.

**Outputs:**

| Path | Content |
|------|---------|
| `results/evaluation_abc/csv/rollout_metrics_*.csv` | Per-item NCC, SRR (`snr_db_*`), PSD log-L², … |
| `results/evaluation_abc/figures/global/metrics_boxplot_ABC.pdf` | Grouped NCC / SRR / PSD by channel (A vs B vs C) |
| `results/evaluation_abc/figures/corner/*_corner_Z.pdf` | Median NCC / SRR / PSD on Δ–depth, Δ–Mw, depth–Mw planes |

Useful flags: `--half` (GPU fp16), `--skip-psd` (faster rollouts), `--skip-rollout` / `--skip-plots`, `--corner-channel N|E|global`.

---

## Quality assurance (manual steps)

### Rollout → CSV (single configuration)

From repository root:

```bash
python quality_assurance/efficient_rollout.py \
  --checkpoint phase1/epoch=12-step=633750.ckpt \
  --data_dir /path/to/PaperTestSet \
  --context_ratio 1 \
  --future_secs 120 \
  --mode free \
  -n 2000 \
  --csv rollout_metrics_1_120.csv
```

Use `--context_ratio` and `--future_secs` for B (1, 240) and C (2, 240). Name CSVs `rollout_metrics_<ratio>_<future>.csv` so `plot_global_metrics_from_csv.py` labels configs A/B/C automatically.

### Global metrics figure

```bash
python quality_assurance/plot_global_metrics_from_csv.py \
  --csv results/evaluation_abc/csv/rollout_metrics_1_120.csv \
        results/evaluation_abc/csv/rollout_metrics_1_240.csv \
        results/evaluation_abc/csv/rollout_metrics_2_240.csv \
  --output results/evaluation_abc/figures/global/metrics_boxplot_ABC.pdf
```

### Physics corner (NCC, SRR, PSD vs geometry)

```bash
python quality_assurance/plot_ncc_corner.py \
  --csv results/evaluation_abc/csv/rollout_metrics_1_120.csv \
  --metadata-csv /path/to/metadata.csv \
  --metrics ncc,snr,psd_logl2 \
  --channel Z \
  --output results/evaluation_abc/figures/corner/rollout_metrics_1_120_corner_Z.pdf
```

Repeat per CSV or use `run_evaluation_abc.sh`.

### Other QA scripts

| Script | Role |
|--------|------|
| `plot_ncc_heatmap.py` | Median NCC heatmap on distance × depth |
| `make_paper_figures.py` | Fig. 7 horizon series, contrast successes, multi-event panels |
| `rollout_filtered_catalog_plots.py` | Filtered low/high-NCC catalog PDFs |

---

## Data generation

Synthetic training data via **Instaseis** + **ObsPy** + **SeisBench** export.

```bash
cd data_generation
conda install -c conda-forge obspy instaseis
pip install numpy scipy tqdm seisbench

python waveform_generator.py \
  --db /path/to/instaseis_db \
  --num_events 5000 \
  --out seisbench_data
```

See `data_generation/waveform_generator.py --help` for depth, distance, and magnitude ranges.

---

## Training

Edit `configs/train.yaml` — set `data.init_args.data_dir` to your SeisBench dataset.

```bash
python main.py fit --config configs/train.yaml
```

Training uses `data.data_handling.SeismicDataModule` (not a separate top-level `data_handling` package). Run from **repository root** so `models` and `data` import correctly.

Phase 1 hyperparameters for the published checkpoint are also on Hugging Face as `train_phase1_logcosh.yaml`.

---

## Citation

If you use this code or the published weights, please cite:

```bibtex
@article{esmail2026seismogpt,
  title   = {Data-Driven Forecasting of three-Component Seismograms Using Transformer Architectures},
  author  = {Esmail, Waleed and Russell, Stuart and Klinge, Jana and Kappes, Alexander and Thomas, Christine},
  journal = {arXiv preprint arXiv:2606.02912},
  year    = {2026},
  eprint  = {2606.02912},
  archivePrefix = {arXiv},
  primaryClass  = {astro-ph.IM},
  url     = {https://arxiv.org/abs/2606.02912}
}
```

Paper: [https://arxiv.org/abs/2606.02912v1](https://arxiv.org/abs/2606.02912v1)

## License

This repository is released under the [MIT License](LICENSE). The Hugging Face model [wesmail/SeismoGPT](https://huggingface.co/wesmail/SeismoGPT) is also distributed under MIT.
