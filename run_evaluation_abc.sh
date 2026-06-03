#!/usr/bin/env bash
#
# Paper evaluation pipeline: configurations A, B, C
#
#   A — context_ratio 1, future 120 s  → rollout_metrics_1_120.csv
#   B — context_ratio 1, future 240 s  → rollout_metrics_1_240.csv
#   C — context_ratio 2, future 240 s  → rollout_metrics_2_240.csv
#
# Then:
#   • Global NCC / SRR / PSD log-L² box plots (all configs)
#   • Physics corner plots (NCC, SRR, PSD vs Δ, depth, Mw) per config
#
# Run from the repository root (this script's directory):
#
#   ./run_evaluation_abc.sh \
#     --data-dir /path/to/PaperTestSet \
#     --metadata-csv /path/to/metadata.csv \
#     --checkpoint phase1/epoch=12-step=633750.ckpt
#
# Or download weights from Hugging Face first:
#
#   ./run_evaluation_abc.sh --download-hf --data-dir ... --metadata-csv ...
#
# Environment overrides: CKPT, DATA_DIR, METADATA_CSV, OUTPUT_DIR, N_SAMPLES,
#   SEED, KERNEL_SIZE, NUM_TOKENS, MODE, HF_REPO, SKIP_ROLLOUT, SKIP_PLOTS

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

CKPT="${CKPT:-}"
DATA_DIR="${DATA_DIR:-}"
METADATA_CSV="${METADATA_CSV:-}"
OUTPUT_DIR="${OUTPUT_DIR:-results/evaluation_abc}"
N_SAMPLES="${N_SAMPLES:-1000}"
SEED="${SEED:-42}"
KERNEL_SIZE="${KERNEL_SIZE:-16}"
NUM_TOKENS="${NUM_TOKENS:-320}"
MODE="${MODE:-free}"
HF_REPO="${HF_REPO:-wesmail/SeismoGPT}"
WEIGHTS_DIR="${WEIGHTS_DIR:-phase1}"
DOWNLOAD_HF=false
SKIP_ROLLOUT=false
SKIP_PLOTS=false
HALF=false
SKIP_PSD=false
NUM_WORKERS="${NUM_WORKERS:-4}"
CORNER_CHANNEL="${CORNER_CHANNEL:-Z}"

usage() {
  cat <<EOF
Usage: $0 --data-dir PATH --metadata-csv PATH [OPTIONS]

Required:
  --data-dir PATH         SeisBench test dataset root (PaperTestSet/)
  --metadata-csv PATH     Metadata CSV (distance, depth, Mw, … for plotting)

Model (one of):
  --checkpoint PATH       Local Lightning .ckpt
  --download-hf           Download Phase 1 from Hugging Face ($HF_REPO) into WEIGHTS_DIR

Rollout:
  -n, --num-samples N     Items to evaluate per config (default: $N_SAMPLES)
  --seed N                RNG seed (default: $SEED)
  --kernel-size N         (default: $KERNEL_SIZE)
  --num-tokens N          (default: $NUM_TOKENS)
  --mode free|teacher     (default: $MODE)
  --half                  float16 inference (GPU)
  --skip-psd              Faster metrics (no Welch PSD / robust SRR in CSV)
  --num-workers N         DataLoader workers (default: $NUM_WORKERS)

Output:
  --output-dir PATH       Root for CSVs and figures (default: $OUTPUT_DIR)

Pipeline control:
  --skip-rollout          Only run plotting (CSVs must already exist)
  --skip-plots            Only run rollouts (write CSVs)
  --corner-channel CH     Z, N, E, or global for corner plots (default: $CORNER_CHANNEL)

Configurations (fixed):
  A  context_ratio=1, future_secs=120
  B  context_ratio=1, future_secs=240
  C  context_ratio=2, future_secs=240
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --metadata-csv) METADATA_CSV="$2"; shift 2 ;;
    --checkpoint) CKPT="$2"; shift 2 ;;
    --download-hf) DOWNLOAD_HF=true; shift ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    -n|--num-samples) N_SAMPLES="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --kernel-size) KERNEL_SIZE="$2"; shift 2 ;;
    --num-tokens) NUM_TOKENS="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --weights-dir) WEIGHTS_DIR="$2"; shift 2 ;;
    --hf-repo) HF_REPO="$2"; shift 2 ;;
    --half) HALF=true; shift ;;
    --skip-psd) SKIP_PSD=true; shift ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --skip-rollout) SKIP_ROLLOUT=true; shift ;;
    --skip-plots) SKIP_PLOTS=true; shift ;;
    --corner-channel) CORNER_CHANNEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -z "$DATA_DIR" || -z "$METADATA_CSV" ]]; then
  echo "ERROR: --data-dir and --metadata-csv are required." >&2
  usage
  exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: data directory not found: $DATA_DIR" >&2
  exit 1
fi

if [[ ! -f "$METADATA_CSV" ]]; then
  echo "ERROR: metadata CSV not found: $METADATA_CSV" >&2
  exit 1
fi

if $DOWNLOAD_HF; then
  echo "==> Downloading SeismoGPT from Hugging Face ($HF_REPO)"
  HF_REPO="$HF_REPO" "$SCRIPT_DIR/scripts/download_seismogpt.sh" "$WEIGHTS_DIR"
  CKPT="${WEIGHTS_DIR}/epoch=12-step=633750.ckpt"
fi

if [[ -z "$CKPT" ]]; then
  echo "ERROR: set --checkpoint PATH or use --download-hf" >&2
  exit 1
fi

if [[ ! -f "$CKPT" ]]; then
  echo "ERROR: checkpoint not found: $CKPT" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR/csv" "$OUTPUT_DIR/figures/global" "$OUTPUT_DIR/figures/corner"

CSV_A="${OUTPUT_DIR}/csv/rollout_metrics_1_120.csv"
CSV_B="${OUTPUT_DIR}/csv/rollout_metrics_1_240.csv"
CSV_C="${OUTPUT_DIR}/csv/rollout_metrics_2_240.csv"

run_rollout() {
  local ratio="$1"
  local future="$2"
  local csv_out="$3"
  local tag="$4"

  if [[ -f "$csv_out" ]] && [[ "${FORCE_ROLLOUT:-0}" != "1" ]]; then
    echo "==> [$tag] CSV exists, skipping rollout: $csv_out"
    return 0
  fi

  echo "==> [$tag] Rollout: context_ratio=$ratio future_secs=${future}s → $csv_out"
  local -a cmd=(
    python3 quality_assurance/efficient_rollout.py
    --checkpoint "$CKPT"
    --data_dir "$DATA_DIR"
    --kernel_size "$KERNEL_SIZE"
    --num_tokens "$NUM_TOKENS"
    --context_ratio "$ratio"
    --future_secs "$future"
    --mode "$MODE"
    --csv "$csv_out"
    --seed "$SEED"
    --num-workers "$NUM_WORKERS"
  )
  cmd+=( -n "$N_SAMPLES" )
  if $HALF; then cmd+=( --half ); fi
  if $SKIP_PSD; then cmd+=( --skip-psd ); fi

  "${cmd[@]}"
}

if ! $SKIP_ROLLOUT; then
  run_rollout 1 120 "$CSV_A" "A"
  run_rollout 1 240 "$CSV_B" "B"
  run_rollout 2 240 "$CSV_C" "C"
else
  echo "==> Skipping rollouts (--skip-rollout)"
fi

for f in "$CSV_A" "$CSV_B" "$CSV_C"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing CSV (needed for plots): $f" >&2
    exit 1
  fi
done

if $SKIP_PLOTS; then
  echo "==> Skipping plots (--skip-plots)"
  echo "CSVs: $CSV_A $CSV_B $CSV_C"
  exit 0
fi

echo "==> Global metric box plots (NCC, SRR, PSD log-L²)"
python3 quality_assurance/plot_global_metrics_from_csv.py \
  --csv "$CSV_A" "$CSV_B" "$CSV_C" \
  --output-dir "$OUTPUT_DIR/figures/global" \
  --output "$OUTPUT_DIR/figures/global/metrics_boxplot_ABC.pdf"

run_corner() {
  local csv="$1"
  local stem
  stem="$(basename "$csv" .csv)"
  echo "==> Corner plot: $stem (channel $CORNER_CHANNEL)"
  python3 quality_assurance/plot_ncc_corner.py \
    --csv "$csv" \
    --metadata-csv "$METADATA_CSV" \
    --metrics ncc,snr,psd_logl2 \
    --channel "$CORNER_CHANNEL" \
    --output "$OUTPUT_DIR/figures/corner/${stem}_corner_${CORNER_CHANNEL}.pdf"
}

run_corner "$CSV_A"
run_corner "$CSV_B"
run_corner "$CSV_C"

echo ""
echo "Done."
echo "  Checkpoint:  $CKPT"
echo "  CSVs:        $OUTPUT_DIR/csv/"
echo "  Global plot: $OUTPUT_DIR/figures/global/metrics_boxplot_ABC.pdf"
echo "  Corner plots: $OUTPUT_DIR/figures/corner/"
