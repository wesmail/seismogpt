#!/bin/bash
#SBATCH --job-name=seismogpt_v2_data
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4               # the generator runs on one core; 4 leaves headroom
#SBATCH --mem=32G
#SBATCH --time=06:00:00                 # adjust after the first task (see sacct)
#SBATCH --partition=requeue
#SBATCH --array=1-132                   # 1-130 training files, 131-132 test files
#
# SeismoGPT revision (JGR-MLC 2026JH001579): generate the v2 dataset.
#
#   task   1-130 : training file  i, seed i,             34,300 events, 50-60 deg left out
#                  (about 30,000 kept per file -> about 3.9 million training traces)
#   task 131-132 : test file      j, seed 100000 + j,    30,000 events, full range
#                  (60,000 test traces; those at 50-60 deg are the held-out test)
#
# Each task generates into a work folder, runs check_generator.py on it, and only
# if every check passes moves it into the chunked SeisBench layout:
#   $OUT/train/metadata_data_<i>.csv  waveforms_data_<i>.hdf5
#   $OUT/test/metadata_data_<j>.csv   waveforms_data_<j>.hdf5
# The unprocessed copy (needed by the checks) is kept for both test files and for
# the first KEEP_RAW_TRAIN training files ($OUT/test_raw, $OUT/train_raw); for the
# other training files it is deleted after the checks, to save disk.
# A finished file is skipped, so a requeued or resubmitted task does no double work.
#
# First ONE task, check it, then the rest:
#   sbatch --array=1 slurm/generate_slurm.sh
#   sbatch --array=2-132 slurm/generate_slurm.sh

set -eo pipefail

# ------------------------------------------------------------------ settings
REPO=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}                  # default: the folder you submit from
if [[ ! -f $REPO/data_generation/waveform_generator.py ]]; then
    echo "REPO=$REPO is not the repository root; submit from the root or set REPO"; exit 1
fi
DB=${DB:-/scratch/tmp/wesmail/ak135f_2s/}
OUT=${OUT:-/scratch/tmp/wesmail/ErUM-WAVE/SingleStation/SobolSamplingV2}
N_TRAIN_TASKS=${N_TRAIN_TASKS:-130}
EVENTS_TRAIN=${EVENTS_TRAIN:-34300}     # x 7/8 (50-60 deg left out) = about 30,000 kept
EVENTS_TEST=${EVENTS_TEST:-30000}
CHECK=${CHECK:-1}                       # 1: run check_generator.py before accepting a file
CHECK_N=${CHECK_N:-3000}                # traces per file for the reprocess and leak checks
KEEP_RAW_TRAIN=${KEEP_RAW_TRAIN:-2}     # keep the unprocessed copy of training files 1..this

# ------------------------------------------------------------------ environment
if [[ -z "${SKIP_ENV:-}" ]]; then
    export MAMBA_ROOT_PREFIX=/scratch/tmp/wesmail/micromamba_root
    eval "$(/scratch/tmp/wesmail/micromamba/bin/micromamba shell hook -s bash -r $MAMBA_ROOT_PREFIX)"
    micromamba activate seismo
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
# Instaseis compiles its inner functions with numba and caches the result next to its
# code in the shared environment. Many tasks writing that cache at once corrupt it
# ("ReferenceError: underlying object has vanished"). Each task gets its own cache
# on the node instead; compiling costs a few seconds per task.
export NUMBA_CACHE_DIR=${TMPDIR:-/tmp}/numba_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}
mkdir -p "$NUMBA_CACHE_DIR"
trap 'rm -rf "$NUMBA_CACHE_DIR"' EXIT
set -u

# ------------------------------------------------------------------ what this task does
ID=${SLURM_ARRAY_TASK_ID:?submit with sbatch --array=...}
if (( ID <= N_TRAIN_TASKS )); then
    SPLIT=train; CHUNK=$ID; SEED=$ID; EVENTS=$EVENTS_TRAIN
    EXTRA=(--exclude_dist 50 60); CHECK_EXTRA=(--exclude_dist 50 60)
else
    SPLIT=test; CHUNK=$(( ID - N_TRAIN_TASKS )); SEED=$(( 100000 + CHUNK )); EVENTS=$EVENTS_TEST
    EXTRA=(); CHECK_EXTRA=()
fi
if [[ $SPLIT == test ]] || (( CHUNK <= KEEP_RAW_TRAIN )); then KEEP_RAW=1; else KEEP_RAW=0; fi
if [[ $CHECK == 1 || $KEEP_RAW == 1 ]]; then SAVE_RAW=1; EXTRA+=(--save_raw); else SAVE_RAW=0; fi

FINAL=$OUT/$SPLIT
FINAL_RAW=$OUT/${SPLIT}_raw
WORK=$OUT/work/${SPLIT}_${CHUNK}                         # the generator writes <out>_<postfix>
mkdir -p "$FINAL" "$OUT/work"
if [[ $KEEP_RAW == 1 ]]; then mkdir -p "$FINAL_RAW"; fi

if [[ -f $FINAL/metadata_data_${CHUNK}.csv ]]; then
    echo "$SPLIT file $CHUNK already done, nothing to do"
    exit 0
fi

COMMIT=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
if [[ -n "$(git -C "$REPO" status --porcelain data_generation 2>/dev/null)" ]]; then
    echo "WARNING: uncommitted changes in $REPO/data_generation; provenance is unclear"
fi
echo "task $ID: $SPLIT file $CHUNK, seed $SEED, $EVENTS events, repo $REPO, commit $COMMIT, host $(hostname)"

# ------------------------------------------------------------------ generate
rm -rf "$WORK" "${WORK}_raw"                             # leftovers of an interrupted run
cd "$REPO/data_generation"
python waveform_generator.py --db "$DB" --num_events "$EVENTS" \
    --out "$OUT/work/$SPLIT" --postfix "$CHUNK" --seed "$SEED" "${EXTRA[@]}"

# ------------------------------------------------------------------ check
# a failed check stops the task here (exit code 1): nothing is moved into place
if [[ $CHECK == 1 ]]; then
    cd "$REPO"
    python data_generation/check_generator.py --data_dir "$WORK" \
        --n "$CHECK_N" "${CHECK_EXTRA[@]}"
fi

# ------------------------------------------------------------------ move into place
# the processed metadata file goes last: its presence marks the file as complete
if [[ $KEEP_RAW == 1 ]]; then
    mv "${WORK}_raw/waveforms.hdf5" "$FINAL_RAW/waveforms_data_${CHUNK}.hdf5"
    mv "${WORK}_raw/metadata.csv"   "$FINAL_RAW/metadata_data_${CHUNK}.csv"
fi
rm -rf "${WORK}_raw"
mv "$WORK/waveforms.hdf5" "$FINAL/waveforms_data_${CHUNK}.hdf5"
mv "$WORK/metadata.csv"   "$FINAL/metadata_data_${CHUNK}.csv"
rmdir "$WORK"

N=$(( $(wc -l < "$FINAL/metadata_data_${CHUNK}.csv") - 1 ))
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -Iseconds)" "$SPLIT" "$CHUNK" "$SEED" "$EVENTS" "$N" "$COMMIT" \
    >> "$OUT/provenance.tsv"
echo "done: $SPLIT file $CHUNK, $N traces"
