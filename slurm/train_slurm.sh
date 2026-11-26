#!/bin/bash
#SBATCH --job-name=seismogpt_train
#SBATCH --partition=gpu4090,gpu3090   # not gpua100: its nodes fail with CUDA "busy or unavailable"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1             # ONE PROCESS PER GPU: must equal the number of GPUs
#SBATCH --cpus-per-task=9               # 8 data-loading workers + the training process
#SBATCH --gres=gpu:1                    # 1 GPU, batch 32 (about 1 h per epoch on an RTX 4090)
#SBATCH --mem=48G                       # measured need: see the "[mem ...]" lines in the log
#SBATCH --time=7-00:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#
# Train SeismoGPT or a baseline. Submit from the repository root. Every argument ending
# in .yaml becomes a --config (later files override earlier ones); anything else is
# passed to main.py unchanged.
#
#   SMOKE=1 sbatch --time=00:20:00 slurm/train_slurm.sh configs/train.yaml   # 3-minute test FIRST
#   sbatch slurm/train_slurm.sh configs/train.yaml                           # full training
#   sbatch slurm/train_slurm.sh configs/train.yaml --ckpt_path logs/version_N/checkpoints/latest-....ckpt
#   sbatch --job-name=gru slurm/train_slurm.sh configs/baseline_gru.yaml
#
# SMOKE=1 runs two tiny epochs (train -> validation -> train -> validation) without
# logging, so that everything that can go wrong at an epoch boundary shows up in minutes,
# and prints the memory it needed.
#
# Lightning starts one training process per SLURM task: --ntasks-per-node must equal the
# number of GPUs. For two GPUs: --gres=gpu:2 --ntasks-per-node=2 (memory and CPUs double).

set -eo pipefail

# ---- repository: the folder you submit from ----
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ ! -f $REPO/main.py ]]; then
    echo "REPO=$REPO is not the repository root; submit from the root or set REPO"; exit 1
fi
cd "$REPO"

# ---- environment ----
# Same activation as the training script used for the paper: the micromamba on PATH,
# with its default root (where the "pytorch" environment lives).
if [[ -z "${SKIP_ENV:-}" ]]; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate "${ENV_NAME:-pytorch}"
    export LD_LIBRARY_PATH="$MAMBA_ROOT_PREFIX/envs/${ENV_NAME:-pytorch}/lib:${LD_LIBRARY_PATH:-}"
fi
# stop at once, with a clear message, if the environment is not the right one
python -c "import torch, lightning" 2>/dev/null || {
    echo "torch/lightning not found in $(command -v python): environment not activated"; exit 1; }

# ---- plain, unbuffered log output (no terminal in a batch job) ----
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export COLUMNS=120 TQDM_DYNAMIC_NCOLS=0 TERM=dumb NO_COLOR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# W&B: key kept outside the repository (one line in ~/.wandb_key, chmod 600)
[[ -f ~/.wandb_key ]] && export WANDB_API_KEY=$(< ~/.wandb_key)
# Never let W&B stop a training that waited in the queue: without a key, log offline
# (upload later with: wandb sync wandb/offline-run-*)
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    echo "W&B: key found, logging online"
else
    export WANDB_MODE=offline
    echo "W&B: no key found (~/.wandb_key), logging OFFLINE; sync later with: wandb sync wandb/offline-run-*"
fi

# ---- arguments: *.yaml -> --config, the rest unchanged ----
ARGS=()
for a in "$@"; do
    if [[ $a == *.yaml ]]; then ARGS+=(--config "$a"); else ARGS+=("$a"); fi
done
if [[ ${#ARGS[@]} -eq 0 ]]; then echo "give at least one config file"; exit 1; fi

# stop if the number of processes does not match the number of GPUs
N_GPU=$( (nvidia-smi -L 2>/dev/null || true) | wc -l)
if [[ -n "${SLURM_NTASKS_PER_NODE:-}" && $N_GPU -gt 0 && $SLURM_NTASKS_PER_NODE -ne $N_GPU ]]; then
    echo "--ntasks-per-node=$SLURM_NTASKS_PER_NODE but $N_GPU GPUs: set them equal"; exit 1
fi
echo "repo $REPO, commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown), host $(hostname)"
echo "python main.py fit ${ARGS[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# ---- smoke test: two tiny epochs, nothing logged or kept ----
if [[ -n "${SMOKE:-}" ]]; then
    SMOKE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/smoke_XXXXXX")
    export WANDB_MODE=disabled
    ARGS+=(--trainer.max_epochs 2 --trainer.limit_train_batches "${SMOKE_TRAIN:-400}"
           --trainer.limit_val_batches "${SMOKE_VAL:-100}"
           --trainer.logger false --trainer.default_root_dir "$SMOKE_DIR"
           --trainer.callbacks "[]")       # default checkpoint only, inside $SMOKE_DIR
    echo "SMOKE TEST: 2 epochs x (${SMOKE_TRAIN:-400} training + ${SMOKE_VAL:-100} validation batches), output in $SMOKE_DIR"
fi

# ---- memory used by this job (program memory, without the file cache), logged regularly ----
mem_gb() {
    local rel f
    rel=$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup 2>/dev/null)                 # cgroup v2
    f=/sys/fs/cgroup${rel%/step_*}/memory.stat
    if [[ -z $rel || ! -r $f ]]; then                                                 # cgroup v1
        rel=$(awk -F: '$2 ~ /(^|,)memory(,|$)/{print $3}' /proc/self/cgroup 2>/dev/null)
        f=/sys/fs/cgroup/memory${rel%/step_*}/memory.stat
    fi
    if [[ -r $f ]]; then
        awk '$1=="anon"||$1=="rss"||$1=="shmem"{s+=$2} END{printf "%.1f", s/2^30}' "$f"
    else                                                                              # fallback
        ps -u "$USER" -o rss= | awk '{s+=$1} END{printf "%.1f", s/2^20}'
    fi
}
PEAK_FILE=$(mktemp)
echo 0 > "$PEAK_FILE"
(
    set +e
    every=${MEM_EVERY:-$([[ -n "${SMOKE:-}" ]] && echo 15 || echo 600)}
    first=1
    while sleep "$([[ $first == 1 && $every -gt 120 ]] && echo 120 || echo "$every")"; do
        first=0
        m=$(mem_gb)
        awk -v m="$m" -v p="$(cat "$PEAK_FILE")" 'BEGIN{exit !(m>p)}' && echo "$m" > "$PEAK_FILE"
        echo "[mem $(date +%H:%M:%S)] ${m} GB used (limit ${SLURM_MEM_PER_NODE:-?} MB)"
    done
) &
MON=$!
cleanup() {                              # runs when the script ends, however it ends
    local rc=$?
    set +e
    pkill -P "$MON" 2>/dev/null; kill "$MON" 2>/dev/null; wait "$MON" 2>/dev/null
    echo "[mem] peak $(cat "$PEAK_FILE") GB"
    rm -rf "$PEAK_FILE" ${SMOKE_DIR:-}
    exit $rc
}
trap cleanup EXIT

# srun so that Lightning's multi-GPU start-up works on this cluster
${LAUNCH-srun --unbuffered} python main.py fit "${ARGS[@]}"
if [[ -n "${SMOKE:-}" ]]; then echo "SMOKE TEST PASSED"; fi
