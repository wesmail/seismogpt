#!/usr/bin/env bash
#
# Run autoregressive rollout (testset_rollout.py) then plot metrics (plot_rollout_metrics.py).
#
# Mandatory:
#   --data_dir PATH          Dataset directory (seisbench WaveformDataset)
#   --checkpoint PATH        Model checkpoint (.ckpt)
#   --metadata-csv PATH      Metadata CSV for plot_rollout_metrics.py
#
# Optional (rollout defaults in parentheses):
#   --kernel_size N          (16)
#   --num_tokens N           (128)
#   --context_secs S         (120)
#   --future_secs S          (240)
#   --num_samples N          (8000)
#   --csv PATH               Output CSV path (default: rollout_metrics_longhorizon.csv)
#   --mode free|teacher      (free)
#   --output PATH            Rollout image path (default: rollout.png)
#   --seed N                 (42)
#   --output-dir PATH        Plot output directory (default: metrics_plots_<csv_stem>)
#
# Example:
#   ./run_rollout_and_plot.sh --data_dir /path/to/dataset --checkpoint logs/version_0/checkpoints/epoch=25-step=121888.ckpt --metadata-csv /path/to/metadata_data_50.csv
#   ./run_rollout_and_plot.sh --data_dir /path/to/dataset --checkpoint my.ckpt --metadata-csv meta.csv --context_secs 60 --future_secs 120 --csv rollout_earlycoda.csv
#

set -e

# Defaults (optional args)
KERNEL_SIZE=16
NUM_TOKENS=128
CONTEXT_SECS=120
FUTURE_SECS=240
NUM_SAMPLES=8000
CSV_OUT="rollout_metrics_longhorizon.csv"
MODE="free"
ROLLOUT_OUTPUT="rollout.png"
SEED=42
PLOT_OUTPUT_DIR=""

# Mandatory (empty until set)
DATA_DIR=""
CHECKPOINT=""
METADATA_CSV=""

usage() {
  echo "Usage: $0 --data_dir PATH --checkpoint PATH --metadata-csv PATH [OPTIONS]"
  echo ""
  echo "Mandatory:"
  echo "  --data_dir PATH       Dataset directory"
  echo "  --checkpoint PATH     Model checkpoint (.ckpt)"
  echo "  --metadata-csv PATH   Metadata CSV for plotting"
  echo ""
  echo "Optional (rollout): --kernel_size, --num_tokens, --context_secs, --future_secs,"
  echo "  --num_samples, --csv, --mode, --output, --seed"
  echo "Optional (plot):   --output-dir"
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir)
      DATA_DIR="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --metadata-csv)
      METADATA_CSV="$2"
      shift 2
      ;;
    --kernel_size)
      KERNEL_SIZE="$2"
      shift 2
      ;;
    --num_tokens)
      NUM_TOKENS="$2"
      shift 2
      ;;
    --context_secs)
      CONTEXT_SECS="$2"
      shift 2
      ;;
    --future_secs)
      FUTURE_SECS="$2"
      shift 2
      ;;
    --num_samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    -n)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --csv)
      CSV_OUT="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    --output)
      ROLLOUT_OUTPUT="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --output-dir)
      PLOT_OUTPUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

if [[ -z "$DATA_DIR" || -z "$CHECKPOINT" || -z "$METADATA_CSV" ]]; then
  echo "Error: --data_dir, --checkpoint, and --metadata-csv are required."
  usage
fi

# Default plot output dir from CSV stem if not set (e.g. rollout_metrics_longhorizon.csv -> metrics_plots_longhorizon)
if [[ -z "$PLOT_OUTPUT_DIR" ]]; then
  CSV_STEM=$(basename "$CSV_OUT" .csv)
  CSV_STEM=${CSV_STEM#rollout_metrics_}   # optional: strip prefix
  PLOT_OUTPUT_DIR="metrics_plots_${CSV_STEM}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "========== 1. Running autoregressive rollout =========="
python testset_rollout.py \
  --data_dir "$DATA_DIR" \
  --checkpoint "$CHECKPOINT" \
  --kernel_size "$KERNEL_SIZE" \
  --num_tokens "$NUM_TOKENS" \
  --context_secs "$CONTEXT_SECS" \
  --future_secs "$FUTURE_SECS" \
  --num_samples "$NUM_SAMPLES" \
  --mode "$MODE" \
  --output "$ROLLOUT_OUTPUT" \
  --csv "$CSV_OUT" \
  --seed "$SEED"

echo ""
echo "========== 2. Plotting rollout metrics =========="
python plot_rollout_metrics.py \
  --csv "$CSV_OUT" \
  --metadata-csv "$METADATA_CSV" \
  --output-dir "$PLOT_OUTPUT_DIR"

echo ""
echo "Done. CSV: $CSV_OUT  Plots: $PLOT_OUTPUT_DIR/"
