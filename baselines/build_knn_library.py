#!/usr/bin/env python3
"""Build the look-up library for the nearest-neighbour baseline.

Stores training windows exactly as the model sees them (window from P,
anchor-normalised, x 10), cut to the first --max_tokens tokens, plus the cumulative
energy per token, which lets the search compare contexts of any length.

  python baselines/build_knn_library.py --data_dir /path/train --n_items 200000 --out knn_lib

Output (float32):
  knn_lib/tokens.npy       [N, max_tokens, C, K]
  knn_lib/cum_energy.npy   [N, max_tokens]   sum of squares of the first t+1 tokens
  knn_lib/index.csv        library row -> dataset item and its parameters (knn_params)

Size: N x max_tokens x 48 x 4 bytes, e.g. 200,000 x 192 tokens = 7.4 GB.
--max_tokens must cover the longest context plus forecast: configuration C at 90 deg
needs about 2 x 640 s of context + 240 s of forecast = 1520 s = 180 tokens.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines.common import training_split            # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data_dir", required=True, help="training set (same as for SeismoGPT)")
ap.add_argument("--n_items", type=int, default=200000, help="library size (0 = whole training split)")
ap.add_argument("--max_tokens", type=int, default=192)
ap.add_argument("--out", default="knn_lib")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--num_workers", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

ds, idx = training_split(a.data_dir, a.n_items, a.seed)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
C, K = ds[int(idx[0])]["x"].shape[1:]
N, T = len(idx), a.max_tokens
tokens = open_memmap(out / "tokens.npy", mode="w+", dtype=np.float32, shape=(N, T, C, K))
cum = open_memmap(out / "cum_energy.npy", mode="w+", dtype=np.float32, shape=(N, T))
print(f"library: {N:,} windows x {T} tokens = {tokens.nbytes / 1e9:.1f} GB")

row = 0
for batch in tqdm(DataLoader(Subset(ds, idx), batch_size=a.batch, num_workers=a.num_workers),
                  desc="library"):
    x = batch["x"][:, :T].numpy()                           # [B, T, C, K]
    tokens[row:row + len(x)] = x
    cum[row:row + len(x)] = np.cumsum((x.astype(np.float64) ** 2).sum(axis=(2, 3)), axis=1)
    row += len(x)
tokens.flush()
cum.flush()

cols = [c for c in ("distance_deg", "src_depth_km", "azimuth_deg", "Mw", "m_rr", "m_tt", "m_pp",
                    "m_rt", "m_rp", "m_tp", "event_id") if c in ds.metadata.columns]   # knn_params needs these
index = ds.metadata.iloc[idx][cols].reset_index(drop=True)
index.insert(0, "item", idx)
index.to_csv(out / "index.csv", index_label="row")
print(f"wrote {out}/")
