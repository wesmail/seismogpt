# Generic imports
import functools
import numpy as np
from pathlib import Path
import scipy.signal as scipy_signal
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

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Collate function for waveform batches
# --------------------------------------------------------------------------------------
def waveform_collate_fn(batch, num_tokens: int, min_tokens: int = 24,
                        bias_high_tokens: bool = True):
    """
    Collate function for variable-length waveform token batches (no padding).

    This function constructs a batch where all samples share the same number of
    valid tokens `real_tokens`, but *without zero-padding*. Instead, each sample
    is truncated to `real_tokens` along the token axis. This avoids padded tokens
    entering the transformer and eliminates the need for attention masking,
    preserving compatibility with Flash Attention (no explicit attn_mask).

    A single `real_tokens` value is sampled per batch in the range
    [min_tokens, num_tokens]. If `bias_high_tokens=True`, longer contexts are
    sampled with linearly increasing probability, encouraging the model to
    train more frequently on longer sequences while still seeing shorter ones.

    Args:
        batch (list of dict):
            Each item must contain:
                - "x":      Tensor [T, C, K]     (waveform tokens)
                - "y":      Tensor [T*K, C]      (flattened next-sample targets)
        num_tokens (int):
            Maximum available tokens per sample.
        min_tokens (int, optional):
            Minimum number of tokens to use in a batch.
        bias_high_tokens (bool, optional):
            If True, sample `real_tokens` with linearly increasing probability
            favoring longer contexts. If False, sample uniformly.

    Returns:
        dict:
            {
                "x": Tensor [B, T', C, K],
                "y": Tensor [B, T'*K, C],
                "real_tokens": int
            }
            (+ "y_horizon" if present in batch items)

        where:
            - B  = batch size
            - T' = sampled real_tokens
            - K  = samples per token (kernel size)
            - C  = number of channels

    Notes:
        - No padding is applied. All samples are truncated to `real_tokens`.
        - This reduces unnecessary computation compared to zero-padding.
        - The returned sequences are suitable for causal transformer training
          without attention masks.
        - Loss functions should operate only on the returned (non-truncated)
          target region.
    """
    if bias_high_tokens:
        population = list(range(min_tokens, num_tokens + 1))
        weights = [k - min_tokens + 1 for k in population]
        real_tokens = random.choices(population, weights=weights, k=1)[0]
    else:
        real_tokens = random.randint(min_tokens, num_tokens)

    xs, ys, y_horizons = [], [], []
    have_y_horizon = "y_horizon" in batch[0]

    for item in batch:
        x = item["x"]   # [T, C, K]
        y = item["y"]   # [T*K, C]

        K = x.shape[2]
        xs.append(x[:real_tokens].clone())       # [real_tokens, C, K]
        ys.append(y[:real_tokens * K].clone())   # [real_tokens*K, C]

        if have_y_horizon:
            yh = item["y_horizon"]   # [H, T*K, C]
            y_horizons.append(yh[:, :real_tokens * K, :].clone())

    out = {
        "x": torch.stack(xs),   # [B, real_tokens, C, K]
        "y": torch.stack(ys),   # [B, real_tokens*K, C]
    }
    if have_y_horizon:
        out["y_horizon"] = torch.stack(y_horizons)  # [B, H, real_tokens*K, C]
    return out
# --------------------------------------------------------------------------------------



class SeismicWaveformDataset(Dataset):
    """
    PyTorch Dataset for seismic waveform data (time-only).

    Returns per sample:
        x: [T, C, K]  time-domain tokens
        y: [T*K, C]   horizon-1 target (flattened next-sample)
        y_horizon: [H, T*K, C]  multi-horizon targets (if num_pred_horizons > 1)
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
        random_shift: bool = False,
        num_pred_horizons: int = 1,
        aug_polarity_flip: bool = False,
        aug_channel_swap: bool = False,
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
        self.random_shift = bool(random_shift)
        self.num_pred_horizons = max(1, int(num_pred_horizons))
        self.aug_polarity_flip = bool(aug_polarity_flip)
        self.aug_channel_swap = bool(aug_channel_swap)
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
            raise ValueError("amp_norm must be one of: 'peak', 'std', 'rms', None")
        return strain

    def __getitem__(self, idx: int):
        # Load full waveform [C, T_total]
        waveform = self.sb_dataset.get_waveforms(idx)
        n_channels, n_samples_total = waveform.shape

        # Find P arrival (if available)
        meta = self.metadata.iloc[idx]
        p_sec = meta.get("trace_p_arrival_s", np.nan)
        if np.isfinite(p_sec):
            if self.random_shift:
                p_sec += random.randint(-10, 10)
            p_sample = int(p_sec * self.sr)
        else:
            p_sample = 0

        # For H prediction horizons we need T+H total tokens so that every
        # horizon h ∈ {1..H} has T complete target tokens.
        # required_len = (T + H) * K  samples
        H = self.num_pred_horizons
        required_len = self.context_size + self.kernel_size * H
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
                segment, demean=True, detrend=True, amp_norm="peak"
            )

        # Convert to torch: [L, C]
        segment = segment.T
        strain = torch.from_numpy(segment).float()*10

        # ── Training-only augmentations ───────────────────────────────────────
        if self.training:
            # 1. Polarity flip — independently invert sign of each channel.
            #    Each channel is flipped with p=0.5, independently of the others.
            if self.aug_polarity_flip:
                # signs: [1, C]  — each entry is +1 or -1
                signs = torch.randint(0, 2, (1, strain.shape[1])).float() * 2 - 1
                strain = strain * signs  # broadcast over time axis

            # 2. Horizontal channel swap (N ↔ E) — swap channels 1 and 2.
            #    Valid only when the waveform has at least 3 channels (ZNE order).
            #    The Z component (channel 0) is never touched.
            if self.aug_channel_swap and strain.shape[1] >= 3:
                if random.random() < 0.5:
                    strain[:, [1, 2]] = strain[:, [2, 1]]
        # ─────────────────────────────────────────────────────────────────────

        # Time tokens: unfold [L, C] -> [T+H, C, K]
        tokens = strain.unfold(0, self.kernel_size, self.stride)  # [T+H, C, K]
        T = self.num_tokens

        # Input context: first T tokens
        x_time = tokens[:T].clone()   # [T, C, K]

        # Horizon-1 target (backward-compatible "y" key): [T*K, C]
        y_h1 = tokens[1:T + 1].clone()   # [T, C, K]
        y = y_h1.permute(0, 2, 1).contiguous().view(-1, y_h1.shape[1])  # [T*K, C]

        # Multi-horizon targets stacked as [H, T*K, C].
        # horizon h target  =  tokens[h : T+h]  (shape [T, C, K])
        y_horizons = []
        for h in range(1, H + 1):
            y_h = tokens[h:T + h].clone()   # [T, C, K]
            y_horizons.append(
                y_h.permute(0, 2, 1).contiguous().view(-1, y_h.shape[1])
            )                               # each [T*K, C]
        y_horizon = torch.stack(y_horizons, dim=0)  # [H, T*K, C]

        out = {
            "x":         x_time,    # [T, C, K]
            "y":         y,         # [T*K, C]     horizon-1
            "y_horizon": y_horizon, # [H, T*K, C]  all H horizons
        }

        return out


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
        random_shift: bool = False,
        num_pred_horizons: int = 1,  # must match model num_pred_horizons
        aug_polarity_flip: bool = False,
        aug_channel_swap: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.kernel_size = kernel_size
        self.stride = stride
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_tokens = num_tokens
        self.normalize = normalize
        self.random_shift = random_shift
        self.num_pred_horizons = max(1, int(num_pred_horizons))
        self.aug_polarity_flip = aug_polarity_flip
        self.aug_channel_swap = aug_channel_swap
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.full_dataset = SeismicWaveformDataset(
            data_dir=self.data_dir,
            kernel_size=self.kernel_size,
            stride=self.stride,
            num_tokens=self.num_tokens,
            training=False,
            normalize=self.normalize,
            random_shift=self.random_shift,
            num_pred_horizons=self.num_pred_horizons,
            aug_polarity_flip=self.aug_polarity_flip,
            aug_channel_swap=self.aug_channel_swap,
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
            persistent_workers=self.num_workers > 0,
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