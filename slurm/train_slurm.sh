#!/bin/bash
#SBATCH --job-name=seismogpt_train
#SBATCH --partition=gpu4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16              # 2 GPUs x 8 data-loading workers
#SBATCH --gres=gpu:2                    # 2 GPUs x batch 32 = effective batch 64, as in the paper
#SBATCH --mem=64G
#SBATCH --time=12:00:00                 # pilot: hours; full training: raise (partition allows 7 days)
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append
#
# Train SeismoGPT or a baseline. Submit from the repository root. Every argument ending
# in .yaml becomes a --config (later files override earlier ones); anything else is
# passed to main.py unchanged.
#
#   sbatch slurm/train_slurm.sh configs/train.yaml configs/pilot_a.yaml
#   sbatch slurm/train_slurm.sh configs/train.yaml configs/pilot_b.yaml
#   sbatch --time=7-00:00:00 slurm/train_slurm.sh configs/train.yaml       # full training
#   sbatch --job-name=gru slurm/train_slurm.sh configs/baseline_gru.yaml
#   ... configs/train.yaml --ckpt_path logs/version_0/checkpoints/last.ckpt  # resume

set -eo pipefail

# ---- repository: the folder you submit from ----
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
if [[ ! -f $REPO/main.py ]]; then
    echo "REPO=$REPO is not the repository root; submit from the root or set REPO"; exit 1
fi
cd "$REPO"

# ---- environment ----
if [[ -z "${SKIP_ENV:-}" ]]; then
    export MAMBA_ROOT_PREFIX=/scratch/tmp/wesmail/micromamba_root
    eval "$(/scratch/tmp/wesmail/micromamba/bin/micromamba shell hook -s bash -r $MAMBA_ROOT_PREFIX)"
    micromamba activate pytorch
    export LD_LIBRARY_PATH="$MAMBA_ROOT_PREFIX/envs/pytorch/lib:${LD_LIBRARY_PATH:-}"
fi

# ---- plain, unbuffered log output (no terminal in a batch job) ----
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export COLUMNS=120 TQDM_DYNAMIC_NCOLS=0 TERM=dumb NO_COLOR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- arguments: *.yaml -> --config, the rest unchanged ----
ARGS=()
for a in "$@"; do
    if [[ $a == *.yaml ]]; then ARGS+=(--config "$a"); else ARGS+=("$a"); fi
done
if [[ ${#ARGS[@]} -eq 0 ]]; then echo "give at least one config file"; exit 1; fi

echo "repo $REPO, commit $(git rev-parse --short HEAD 2>/dev/null || echo unknown), host $(hostname)"
echo "python main.py fit ${ARGS[*]}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

# srun so that Lightning's multi-GPU start-up works on this cluster
${LAUNCH-srun --unbuffered} python main.py fit "${ARGS[@]}"
