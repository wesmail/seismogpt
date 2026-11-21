"""Helpers shared by the baseline training scripts."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.data_handling import SeismicDataModule        # noqa: E402


def training_split(data_dir, n_items, seed=0, kernel_size=16, num_tokens=320):
    """The dataset and n_items random indices from the TRAINING part of data_dir.

    Uses SeismicDataModule's own 80/10/10 split (fixed seed 42), so the linear model
    and the kNN library see exactly the traces SeismoGPT and the GRU were trained on,
    never their validation part. Windows are anchor-normalised as in evaluation:
    no random shift, no augmentation.
    """
    dm = SeismicDataModule(data_dir=data_dir, kernel_size=kernel_size, stride=kernel_size,
                           num_tokens=num_tokens, normalize=True, random_shift=False,
                           num_pred_horizons=1, norm_mode="anchor")
    dm.setup()
    train_idx = np.asarray(dm.train_dataset.indices)
    if n_items and n_items < len(train_idx):
        train_idx = np.sort(np.random.default_rng(seed).choice(train_idx, n_items, replace=False))
    return dm.full_dataset, train_idx
