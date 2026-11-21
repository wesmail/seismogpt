#!/usr/bin/env bash
# Evaluate every model on configurations A, B, C, then compare them.
#
#   DATA_DIR=/path/test CKPT=seismogpt.ckpt GRU_CKPT=gru.ckpt \
#   LINEAR=linear_ar.npz KNN_LIB=knn_lib ./baselines/run_abc.sh
#
# Models whose file is not set are skipped. Existing CSVs are kept (FORCE=1 redoes them).
# Output: $OUT/<model>/rollout_metrics_<ratio>_<future>.csv (+ .json), and
#         $OUT/comparison.csv / .tex from compare.py.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR=${DATA_DIR:?set DATA_DIR to the test set}
OUT=${OUT:-results/eval}
N=${N:-2000}
SEED=${SEED:-42}
BAND=${BAND:-50 60}                   # held-out distance band; empty to skip the split
EXTRA=${EXTRA:-}                      # e.g. "--skip-psd"

declare -A FILES=(                    # model -> option with its file ("" = needs none)
  [zero]=""  [persistence]=""  [ar_burg]=""
  [linear]="${LINEAR:+--weights $LINEAR}"
  [gru]="${GRU_CKPT:+--checkpoint $GRU_CKPT}"
  [knn]="${KNN_LIB:+--library $KNN_LIB}"
  [knn_params]="${KNN_LIB:+--library $KNN_LIB}"
  [seismogpt]="${CKPT:+--checkpoint $CKPT}"
)
NEEDS_FILE="linear gru knn knn_params seismogpt"

for model in zero persistence ar_burg linear gru knn knn_params seismogpt; do
  if [[ " $NEEDS_FILE " == *" $model "* && -z "${FILES[$model]}" ]]; then
    echo "--- $model: no file given, skipped"; continue
  fi
  for cfg in "1 120" "1 240" "2 240"; do
    read -r ratio future <<< "$cfg"
    csv="$OUT/$model/rollout_metrics_${ratio}_${future}.csv"
    if [[ -f $csv && "${FORCE:-0}" != 1 ]]; then echo "--- exists: $csv"; continue; fi
    # shellcheck disable=SC2086
    python baselines/evaluate.py --model "$model" ${FILES[$model]} --data_dir "$DATA_DIR" \
      --context_ratio "$ratio" --future_secs "$future" -n "$N" --seed "$SEED" --csv "$csv" $EXTRA
  done
done

# shellcheck disable=SC2086
python baselines/compare.py --results "$OUT" --reference seismogpt --data_dir "$DATA_DIR" \
  ${BAND:+--band $BAND} --out "$OUT/comparison"
