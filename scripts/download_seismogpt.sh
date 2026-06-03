#!/usr/bin/env bash
# Download SeismoGPT Phase 1 weights from Hugging Face into ./phase1/
#
# Usage:
#   ./scripts/download_seismogpt.sh
#   HF_REPO=yourusername/SeismoGPT ./scripts/download_seismogpt.sh

set -euo pipefail

REPO_ID="${HF_REPO:-wesmail/SeismoGPT}"
OUT_DIR="${1:-phase1}"

if ! command -v huggingface-cli &>/dev/null; then
  if [[ -x "${HOME}/.local/bin/huggingface-cli" ]]; then
    export PATH="${HOME}/.local/bin:${PATH}"
  else
    echo "Install: pip install huggingface_hub" >&2
    exit 1
  fi
fi

mkdir -p "$OUT_DIR"
echo "Downloading from ${REPO_ID} → ${OUT_DIR}/"
huggingface-cli download "$REPO_ID" \
  epoch=12-step=633750.ckpt \
  train_phase1_logcosh.yaml \
  --local-dir "$OUT_DIR"

echo "Done. Use: export CKPT=${OUT_DIR}/epoch=12-step=633750.ckpt"
