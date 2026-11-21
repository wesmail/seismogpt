#!/usr/bin/env python3
"""Fit the linear autoregressive baseline by least squares (no SGD, no GPU needed).

Model: next token (C*K values) = W^T [last P tokens of all channels, 1].
Ridge least squares has a closed-form solution from two sums over all training rows,

    A = sum f f^T  ([D, D])        b = sum f y^T  ([D, C*K])        W = (A + lam I)^-1 b,

so the data are read once, in batches, and never held in memory.

  python baselines/train_linear.py --data_dir /path/train --n_items 50000 --out linear_ar.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baselines.common import training_split            # noqa: E402
from baselines.forecasters import linear_features      # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--data_dir", required=True, help="training set (same as for SeismoGPT)")
ap.add_argument("--n_items", type=int, default=50000, help="training windows used (0 = all)")
ap.add_argument("--P", type=int, default=16, help="tokens of history (16 tokens = 134 s)")
ap.add_argument("--alpha", type=float, default=1e-4, help="ridge strength, relative to mean diag(A)")
ap.add_argument("--out", default="linear_ar.npz")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--num_workers", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

ds, idx = training_split(a.data_dir, a.n_items, a.seed)
loader = DataLoader(Subset(ds, idx), batch_size=a.batch, num_workers=a.num_workers)
dev = "cuda" if torch.cuda.is_available() else "cpu"

A = b = yy = None
rows = 0
for batch in tqdm(loader, desc="accumulate"):
    x = batch["x"].to(dev)                                  # [B, T, C, K]
    B, T, C, K = x.shape
    f = linear_features(x, a.P)[:, :-1].reshape(-1, a.P * C * K + 1)   # history up to token t
    y = x[:, 1:].reshape(-1, C * K)                                    # token t+1
    if A is None:
        A = torch.zeros(f.shape[1], f.shape[1], dtype=torch.float64, device=dev)
        b = torch.zeros(f.shape[1], C * K, dtype=torch.float64, device=dev)
        yy = torch.zeros((), dtype=torch.float64, device=dev)
    A += (f.T @ f).double()                                 # batch sums in float32, totals in float64
    b += (f.T @ y).double()
    yy += (y.double() ** 2).sum()
    rows += f.shape[0]

lam = a.alpha * torch.diagonal(A).mean()
W = torch.linalg.solve(A + lam * torch.eye(A.shape[0], dtype=A.dtype, device=dev), b)

# one-step fit on the training rows, from the same sums: SSE = yy - 2 tr(W^T b) + tr(W^T A W)
sse = yy - 2 * (W * b).sum() + (W * (A @ W)).sum()
print(f"{rows:,} rows, {A.shape[0]} features, one-step skill on training rows: {1 - (sse / yy).item():.4f}")

np.savez(a.out, W=W.float().cpu().numpy(), P=a.P, C=C, K=K, alpha=a.alpha, n_items=len(idx))
print(f"wrote {a.out}")
