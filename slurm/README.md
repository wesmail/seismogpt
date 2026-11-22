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
