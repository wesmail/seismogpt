# Generic imports
import math
import h5py
import functools
import numpy as np
from pathlib import Path
from scipy import signal
import scipy.signal as scipy_signal
import os
import random
import logging

log = logging.getLogger(__name__)

# Torch imports
import torch
from torch.utils.data import Dataset, DataLoader, random_split

# PyTorch Lightning
from lightning.pytorch import LightningDataModule

import warnings

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

# Seisbench (for seismic dataset)
try:
    import seisbench.data as sbd
except ImportError:
    sbd = None

import numpy as np
from pathlib import Path
import scipy.signal as scipy_signal
import logging

import torch
from torch.utils.data import Dataset

try:
    import seisbench.data as sbd
except ImportError:
    sbd = None

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Collate function for waveform batches
# --------------------------------------------------------------------------------------
def waveform_collate_fn(batch, num_tokens: int, min_tokens: int = 24, bias_high_tokens: bool = True, kernel_size: int = 16):
    """
    Custom collate function for waveform batches without station dimension.

    This function ensures all samples in a batch share the same number of valid tokens.
    It randomly selects a `real_tokens` count between `min_tokens` and `num_tokens`,
    pads the remaining tokens in each sample with zeros, and constructs a mask indicating
    which tokens are padded (1 = pad, 0 = valid). This mask is useful for attention mechanisms
    and loss functions that should ignore padded regions.

    Args:
        batch (list of dict): Each dict has keys "x", "y", "padding_mask".
        num_tokens (int): Total number of tokens per sample.
        min_tokens (int): Minimum number of valid (unpadded) tokens.
        bias_high_tokens (bool): If True, sample real_tokens with higher weight for larger
            values (linear weights: k has weight proportional to k), so more batches use
            longer context. If False, uniform over [min_tokens, num_tokens].

    Returns:
        dict:
            x (Tensor): [B, T, K, C] - input waveform tokens
            y (Tensor): [B, T*K, C] - target waveform tokens
            padding_mask (Tensor): [B, T] - mask indicating padded tokens
    """
    if bias_high_tokens:
        # Weights: higher token count => higher weight (linear: 1, 2, ..., n)
        population = list(range(min_tokens, num_tokens + 1))
        weights = [k - min_tokens + 1 for k in population]
        real_tokens = random.choices(population, weights=weights, k=1)[0]
    else:
        real_tokens = random.randint(min_tokens, num_tokens)
    pad_mask = torch.zeros(num_tokens)
    pad_mask[real_tokens:] = float("-inf")

    xs, ys, x_freqs, masks = [], [], [], []
    for item in batch:
        x = item["x"].clone()  # [T, K, C]
        y = item["y"].clone()  # [T*K, C] = [L, C]
        x_freq = item["x_freq"].clone()  # [T, C, F]

        x[real_tokens:] = 0
        x_freq[real_tokens:] = 0
        y[real_tokens * kernel_size:] = 0  # correct padding

        xs.append(x)
        ys.append(y)
        x_freqs.append(x_freq)
        masks.append(pad_mask.clone())

    return {
        "x": torch.stack(xs),  # [B, T, K, C]
        "y": torch.stack(ys),  # [B, T*K, C] = [B, L, C]
        "x_freq": torch.stack(x_freqs),  # [B, T, C, F]
        "padding_mask": torch.stack(masks)  # [B, T]
    }
# --------------------------------------------------------------------------------------



# =============================================================================
# Utility: compute per-token FFT features (used by dataset AND rollout/inference)
# =============================================================================
_FREQ_EPS = 1e-6

def freq_features_from_tokens(
    x_time: torch.Tensor,
    freq_keep_bins: int = 8,
    freq_log1p: bool = True,
    freq_norm: str = "none",
) -> torch.Tensor:
    """
    Compute per-token FFT magnitude features from time tokens (NO leakage).

    Shared between dataset preprocessing and autoregressive rollout so that
    frequency features are always computed identically (rollout parity).

    Args:
        x_time: [..., C, K]  — last two dims are channels and within-token samples.
        freq_keep_bins: F, number of low-frequency rFFT bins to keep per channel.
        freq_log1p: apply log(1 + |FFT|) compression.
        freq_norm: "none" | "mean" | "l2" — per-token spectrum normalization.
                   Normalizing reduces sensitivity to amplitude shifts in predicted
                   tokens during autoregressive rollout.

    Returns:
        mag: [..., C, F]  same leading dims as input.
    """
    X = torch.fft.rfft(x_time, dim=-1)          # [..., C, K//2+1] complex
    mag = X.abs()                                # [..., C, K//2+1]

    if freq_log1p:
        mag = torch.log1p(mag)

    Fkeep = min(freq_keep_bins, mag.shape[-1])
    mag = mag[..., :Fkeep]                       # [..., C, F]

    # Optional per-token normalization (applied over F dimension)
    if freq_norm == "mean":
        mag = mag / (mag.mean(dim=-1, keepdim=True) + _FREQ_EPS)
    elif freq_norm == "l2":
        mag = mag / (mag.pow(2).sum(dim=-1, keepdim=True).sqrt() + _FREQ_EPS)
    elif freq_norm != "none":
        raise ValueError(f"Unknown freq_norm='{freq_norm}', expected 'none'|'mean'|'l2'")

    return mag


class SeismicWaveformDataset(Dataset):
    """
    PyTorch Dataset for seismic waveform data.

    UPDATED:
    - Returns multi-modal inputs per token:
        x_time: [T, C, K]  time-domain tokens
        x_freq: [T, C*F]   per-token FFT magnitude features (log1p), keeping F low-freq bins
    - Target remains time-only next token:
        y: [T*K, C]  (token-shifted, flattened exactly as before)

    This implementation avoids STFT/FFT leakage by computing frequency features
    strictly from each time token window (length K).
    """

    def __init__(
        self,
        data_dir: str = "/mnt/d/waleed/Seismology/simulation/clean/SobolDataset/Regional",
        kernel_size: int = 16,
        stride: int = 16,
        component_order: str = "ZNE",
        num_tokens: int = 256,
        training: bool = False,
        normalize: bool = True,
        # frequency feature config
        freq_keep_bins: int = 8,     # F: number of low-frequency rFFT bins per channel to keep
        freq_log1p: bool = True,     # apply log(1 + |FFT|) compression
        # Per-token spectrum normalization (rollout stability: makes freq features
        # scale-invariant so predicted-token FFTs match training distribution better)
        freq_norm: str = "none",     # "none" | "mean" | "l2"
    ):
        if sbd is None:
            raise ImportError("seisbench is required for SeismicWaveformDataset. Install with: pip install seisbench")

        self.data_dir = Path(data_dir)
        self.kernel_size = int(kernel_size)   # K
        self.stride = int(stride)
        self.component_order = component_order
        self.num_tokens = int(num_tokens)
        self.training = bool(training)
        self.normalize = bool(normalize)

        # Frequency features
        self.freq_keep_bins = int(freq_keep_bins)
        self.freq_log1p = bool(freq_log1p)
        self.freq_norm = str(freq_norm)
        assert self.freq_norm in ("none", "mean", "l2"), \
            f"freq_norm must be 'none', 'mean', or 'l2', got '{self.freq_norm}'"

        self.sb_dataset = sbd.WaveformDataset(self.data_dir)
        self.metadata = self.sb_dataset.metadata

        # context_size is for non-overlapping tokenization (stride==kernel_size typical)
        self.context_size = self.kernel_size * self.num_tokens
        self.eps = 1e-10
        self.sr = self.get_sample_rate()

        print(f"SeismicWaveformDataset: loaded {len(self.metadata)} samples from {self.data_dir}")
        print(f"Waveform shape example: {self.sb_dataset.get_waveforms(0).shape}")

    def set_training(self, training: bool):
        self.training = training

    def get_sample_rate(self) -> float:
        return float(self.metadata["trace_sampling_rate_hz"].iloc[0])

    def __len__(self) -> int:
        return len(self.metadata)

    @staticmethod
    def _normalize_strain_max_per_channel(
        strain: np.ndarray,
        demean: bool = True,
        detrend: bool = False,
        amp_norm: str = "peak",
        eps: float = 1e-8,
    ) -> np.ndarray:
        """
        Normalize the strain by the maximum absolute value per channel.
        """
        if demean:
            strain = strain - np.mean(strain, axis=1, keepdims=True)
        if detrend:
            strain = scipy_signal.detrend(strain, axis=1)
        if amp_norm == "peak":
            denom = np.max(np.abs(strain), axis=1, keepdims=True) + eps
            strain = strain / denom
        elif amp_norm == "std":
            denom = np.std(strain, axis=1, keepdims=True) + eps
            strain = strain / denom
        elif amp_norm is not None:
            raise ValueError("amp_norm must be one of: 'peak', 'std', None")
        return strain

    def _freq_features_from_time_tokens(self, x_time: torch.Tensor) -> torch.Tensor:
        """
        Compute per-token FFT magnitude features from time tokens only (NO leakage).

        Args:
            x_time: [T, C, K]  (K=kernel_size)

        Returns:
            x_freq: [T, C, F] where F=min(freq_keep_bins, K//2+1)
        """
        return freq_features_from_tokens(
            x_time,
            freq_keep_bins=self.freq_keep_bins,
            freq_log1p=self.freq_log1p,
            freq_norm=self.freq_norm,
        )

    def __getitem__(self, idx: int):
        # Load full waveform [C, T_total]
        waveform = self.sb_dataset.get_waveforms(idx)
        n_channels, n_samples_total = waveform.shape

        # Find P arrival (if available)
        meta = self.metadata.iloc[idx]
        p_sec = meta.get("trace_p_arrival_s", np.nan)
        if np.isfinite(p_sec):
            p_sample = int(p_sec * self.sr)
        else:
            p_sample = 0

        # Slice and pad the waveform
        required_len = self.context_size + self.kernel_size   # (num_tokens*K + K) => T+1 tokens total
        max_len = waveform.shape[1]
        end_idx = min(p_sample + required_len, max_len)
        actual_len = end_idx - p_sample

        segment = waveform[:, p_sample:end_idx]  # [C, L_actual]
        if actual_len < required_len:
            pad_width = required_len - actual_len
            pad_tensor = np.zeros((n_channels, pad_width), dtype=waveform.dtype)
            segment = np.concatenate([segment, pad_tensor], axis=1)
            log.info(f"Segment padded from {actual_len} to {required_len} samples")

        # Normalize
        if self.normalize:
            segment = self._normalize_strain_max_per_channel(
                segment, demean=True, detrend=True, amp_norm="std"
            )

        # Convert to torch: [L, C]
        segment = segment.T
        strain = torch.from_numpy(segment).float()

        # Time tokens: unfold [L, C] -> [T+1, C, K]
        tokens = strain.unfold(0, self.kernel_size, self.stride)  # [T+1, C, K]

        # AR shift
        x_time = tokens[:-1].clone()   # [T, C, K]
        y_time = tokens[1:].clone()    # [T, C, K]

        # Frequency features computed strictly from x_time tokens (NO leakage)
        x_freq = self._freq_features_from_time_tokens(x_time)  # [T, C, F]

        # y formatted exactly as before: [T*K, C]
        y = y_time.permute(0, 2, 1).contiguous().view(-1, y_time.shape[1])

        return {
            "x": x_time,   # [T, C, K]
            "x_freq": x_freq,   # [T, C, F]
            "y": y,             # [T*K, C]
        }


class SeismicDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for seismic waveform data with time modalities.
    Uses SeismicWaveformDataset (seisbench); no conditioning.
    """

    def __init__(
        self,
        data_dir: str = "/mnt/d/waleed/Seismology/simulation/clean/SobolDataset/Regional",
        kernel_size: int = 64,
        stride: int = 64,
        batch_size: int = 32,
        num_workers: int = 8,
        num_tokens: int = 256,
        normalize: bool = True,
        freq_norm: str = "none",     # per-token spectrum normalization (rollout parity)
    ):
        super().__init__()
        self.data_dir = data_dir
        self.kernel_size = kernel_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_tokens = num_tokens
        self.normalize = normalize
        self.freq_norm = freq_norm
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.full_dataset = SeismicWaveformDataset(
            data_dir=self.data_dir,
            kernel_size=self.kernel_size,
            stride=self.stride,
            num_tokens=self.num_tokens,
            training=False,
            normalize=self.normalize,
            freq_norm=self.freq_norm,
        )
        train_len = int(0.8 * len(self.full_dataset))
        val_len = int(0.1 * len(self.full_dataset))
        test_len = len(self.full_dataset) - train_len - val_len
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.full_dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42), # for reproducible train/val/test split
        )

    def _make_collate_fn(self):
        """
        Return a picklable collate function for use with num_workers > 0.

        Lambda functions cannot be pickled by Python's multiprocessing module,
        which DataLoader uses to spawn worker processes.  functools.partial
        references a module-level function and is fully picklable.
        """
        return functools.partial(
            waveform_collate_fn,
            num_tokens=self.hparams.num_tokens,
            kernel_size=self.hparams.kernel_size,
        )


    def train_dataloader(self):
        self.train_dataset.dataset.set_training(True)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            collate_fn=self._make_collate_fn(),
        )

    def val_dataloader(self):
        self.val_dataset.dataset.set_training(False)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self._make_collate_fn(),
        )

    def test_dataloader(self):
        self.test_dataset.dataset.set_training(False)
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )