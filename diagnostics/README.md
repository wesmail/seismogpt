# diagnostics

## leak_test.py: does the model input see the future?

SeismoGPT reads a context window and forecasts what comes after it. The preprocessing
must not let the context depend on samples after the forecast start t0. Otherwise the
model gets a hint about the future it is asked to predict (Reviewer 2, comment 5).

**The test.** Take a trace, change only the samples after t0, run the preprocessing
again, and compare the part before t0:

```
leak = max |x_before_t0(changed future) - x_before_t0(original)| / max |x_before_t0(original)|
```

A causal pipeline gives **exactly 0**, not just a small number, because nothing
before t0 is computed from samples after t0. Two kinds of change are used:
the future multiplied by 3 (a larger event), and an extra arrival (a Ricker pulse)
after t0. The script also reports how far back the change reaches, in seconds.

**Two stages are tested.**
- Stage 1: the generator's processing (`data_generation/waveform_generator.py`).
- Stage 2: the normalisation in the dataset class (`data/data_handling.py`).

The submitted version leaked in both stages: a zero-phase filter and whole-record
demean in stage 1, and a per-channel peak over the whole window, future included,
in stage 2. The v2 pipeline uses a causal filter and the anchor normaliser, which
scales by the peak of the first token only.

### Usage (from the repository root)

```bash
# 1. no data needed: a stand-in record, shows each leak and the causal reference
python diagnostics/leak_test.py --demo

# 2. stage 2 on stored processed traces (same items as the evaluation, same seed)
python diagnostics/leak_test.py --data_dir /path/test -n 2000                       # anchor (v2)
python diagnostics/leak_test.py --data_dir /path/test -n 2000 --norm_mode legacy_peak

# 3. both stages on unprocessed traces stored with --save_raw: submitted vs v2 pipeline
python diagnostics/leak_test.py --raw_dir /path/test_raw
python diagnostics/leak_test.py --raw_dir /path/test_raw --context_ratio 2          # configuration C

# 4. regenerate traces from the stored metadata with Instaseis (settings check)
python diagnostics/leak_test.py --data_dir /path/test --db /path/ak135f_2s -n 20
```

`--context_ratio r` puts t0 at r × (S − P) after P, rounded down to whole tokens:
r = 1 for configurations A and B, r = 2 for C.

### Expected results

| what | submitted pipeline | v2 pipeline |
|---|---|---|
| stage 1 (generator) | non-zero | exactly 0 |
| stage 2 (normaliser) | non-zero (about 0.67 in the demo) | exactly 0 |
| end to end, what the model sees | non-zero | **exactly 0** |

Any non-zero value for the v2 rows is a bug.

### Environment

Stage 2 imports `SeismicWaveformDataset` from `data/data_handling.py`, which needs
torch. Modes 3 and 4 also need Instaseis and ObsPy. If an import fails, the script
says so in a `NOTE:` line (and for the normaliser, falls back to a copy of the
submitted code). Read the first lines of the output to make sure it tested the
repository code.
