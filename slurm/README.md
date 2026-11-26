# slurm

## generate_slurm.sh: the v2 dataset on the cluster

One SLURM array job makes the whole dataset. Each array task writes one SeisBench file.

| tasks | split | file | seed | events | distance |
|---|---|---|---|---|---|
| 1–130 | train | 1–130 | 1–130 | 34,300 (≈ 30,000 kept) | 10–90°, without 50–60° |
| 131–132 | test | 1–2 | 100001–100002 | 30,000 | 10–90° |

That gives about 3.9 million training traces, as in the paper, and 60,000 test traces.
About 7,500 test traces lie in 50–60°, a band the model never saw in training
(interpolation test).

### What each task does

1. Skips the task if its file already exists, so a requeued or resubmitted task does
   no double work.
2. Runs `data_generation/waveform_generator.py` into `$OUT/work/`.
3. Runs `data_generation/check_generator.py` on the result. If a check fails, the
   task stops with an error and the file stays in `$OUT/work/` for inspection.
4. Moves the file into place. The metadata goes last, so its presence marks a
   complete file.
5. Adds a line to `$OUT/provenance.tsv`: date, split, file, seed, events, traces,
   git commit.

### Output

```
$OUT/train/metadata_data_<i>.csv, waveforms_data_<i>.hdf5     training (SeisBench chunks)
$OUT/test/ metadata_data_<j>.csv, waveforms_data_<j>.hdf5     test
$OUT/train_raw/, $OUT/test_raw/                               unprocessed copies (test files + first training files)
$OUT/provenance.tsv                                           which seed and commit made which file
```

In the training config, set `data_dir: $OUT/train` and `norm_mode: anchor`.
Evaluate on `$OUT/test`.

### Usage

Edit the settings at the top of the script, or set them on the command line
(`REPO=... OUT=... sbatch ...`):

| setting | default | meaning |
|---|---|---|
| `REPO` | the folder you submit from | repository root (branch `paper-review`); submit from the root |
| `DB` | `/scratch/tmp/wesmail/ak135f_2s/` | Instaseis database |
| `OUT` | `.../SobolSamplingV2` | output folder |
| `N_TRAIN_TASKS` | 130 | tasks above this number make test files |
| `EVENTS_TRAIN`, `EVENTS_TEST` | 34300, 30000 | events per file |
| `CHECK` | 1 | run `check_generator.py` before accepting a file |
| `CHECK_N` | 3000 | traces per file for checks 3 and 4 |
| `KEEP_RAW_TRAIN` | 2 | keep the unprocessed copy of training files 1 to this number |
| `SKIP_ENV` | unset | set to 1 to skip micromamba activation (local test) |

Run one file first. It tells you whether everything runs, how long a task takes,
and how much disk the dataset needs:

```bash
sbatch --array=1 slurm/generate_slurm.sh
sacct -j <jobid> --format=JobID,Elapsed,MaxRSS,State   # adjust --time and --mem
du -sh $OUT/train/waveforms_data_1.hdf5                # x 132 = total disk
sbatch --array=2-132 slurm/generate_slurm.sh
```

To start the rest automatically, but only if file 1 passed every check:

```bash
JOB=$(sbatch --parsable --array=1 slurm/generate_slurm.sh)
sbatch --dependency=afterok:$JOB --array=2-132 slurm/generate_slurm.sh
```

When all tasks are done, `train/` should hold 130 files, `test/` 2, and
`provenance.tsv` 132 lines. Resubmit any missing task number. Logs
(`seismogpt_v2_data_<job>_<task>.out/.err`) are written to the folder you submit from.

## train_slurm.sh: training on the cluster

One script for SeismoGPT and the GRU baseline. Submit from the repository root.

```bash
SMOKE=1 sbatch --time=00:20:00 slurm/train_slurm.sh configs/train.yaml   # 1. three-minute test
sbatch slurm/train_slurm.sh configs/train.yaml                           # 2. the training
```

**Always run the smoke test first** after changing code, configs or the environment. It
runs two tiny epochs (training → validation → training → validation) without logging and
prints the memory it needed. Everything that can fail at an epoch boundary fails here in
minutes instead of after an hour in the queue and an hour of training. The log must end with

```
SMOKE TEST PASSED
[mem] peak N GB
```

Defaults: 1 GPU, 9 CPUs, 48 GB, 7 days, on whichever of `gpu4090`, `gpu3090`, `gpua100`
has a free GPU first. One epoch of the full dataset takes about 1 hour on an RTX 4090.

### What the log tells you

| line | meaning |
|---|---|
| `repo ..., commit ...` | the code that ran |
| `W&B: key found, logging online` | the key in `~/.wandb_key` was read (otherwise: offline, sync later) |
| `metadata slimmed: 1.74 GB -> 0.22 GB per copy` | the metadata table kept for loading |
| `[mem 12:00:00] 7.3 GB used (limit 49152 MB)` | program memory of the whole job, every 10 minutes |
| `[mem] peak 7.9 GB` | the largest value seen, printed when the job ends |

### Checkpoints and resuming

Both files are in `checkpoints/seismogpt_v2/` (GRU: `checkpoints/gru_v2/`):

| file | what | use |
|---|---|---|
| `best-epoch=E-step=S.ckpt` | lowest validation loss so far | evaluation |
| `latest-epoch=E-step=S.ckpt` | the most recent epoch | resuming |

```bash
sbatch slurm/train_slurm.sh configs/train.yaml \
       --ckpt_path "$(ls -t checkpoints/seismogpt_v2/latest-*.ckpt | head -1)"
```

### Memory: why the data loader uses `fork`

Python 3.14 starts data-loading workers as fresh interpreters (`forkserver`). Each one then
imports everything again and gets its own copy of the dataset: 1–2 GB per worker, and
during validation the training and validation workers exist at the same time. With `fork`
(`mp_context: fork` in the config) the workers share the memory of the training process;
all of them together need well under 1 GB. If a job ever stalls with `fork`, fall back to
`--data.mp_context forkserver --data.num_workers 4 --data.val_num_workers 2 --mem=64G`.

### Two GPUs

`--gres=gpu:2 --ntasks-per-node=2` (Lightning starts one process per SLURM task; the script
stops if the two numbers differ). The effective batch size doubles, so results are not
comparable with single-GPU runs at the same settings.
