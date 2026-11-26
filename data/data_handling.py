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
        norm_mode: str = "anchor",
        anchor_len: int | None = None,
        # v2: keep only the metadata columns needed to load a window. Each data-loading
        # worker holds its own copy of the metadata; with 3.9 M traces and all columns
        # that is ~3.5 GB per worker, enough to exhaust the job's memory.
        slim_metadata: bool = False,
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
        # v2 (JGR revision): "anchor" = causal normalisation (default);
        # "legacy_peak" = the submitted, non-causal one (whole-window demean,
        # detrend and per-channel peak), kept only to reproduce old numbers.
        if norm_mode not in ("anchor", "legacy_peak"):
            raise ValueError("norm_mode must be 'anchor' or 'legacy_peak'")
        self.norm_mode = norm_mode
        self.anchor_len = int(anchor_len) if anchor_len else self.kernel_size
        self._warned_zero_anchor = False
        self.sb_dataset = sbd.WaveformDataset(self.data_dir)
        if slim_metadata:
            self._slim_metadata()
        self.metadata = self.sb_dataset.metadata

        # context_size is for non-overlapping tokenization (stride==kernel_size typical)
        self.context_size = self.kernel_size * self.num_tokens
        self.eps = 1e-10
        self.sr = self.get_sample_rate()

        print(f"SeismicWaveformDataset: loaded {len(self.metadata)} samples from {self.data_dir}")
        print(f"Waveform shape example: {self.sb_dataset.get_waveforms(0).shape}")

    # columns read by SeisBench's get_waveforms and by __getitem__
    _NEEDED = ("trace_name", "trace_chunk", "trace_sampling_rate_hz",
               "trace_source_sampling_rate_hz", "trace_component_order",
               "trace_p_arrival_s", "trace_s_arrival_s")

    def _slim_metadata(self):
        m = self.sb_dataset._metadata
        before = m.memory_usage(deep=True).sum()
        m = m[[c for c in self._NEEDED if c in m.columns]].copy()
        for c in ("trace_chunk", "trace_component_order"):     # few distinct values
            if c in m.columns:
                m[c] = m[c].astype("category")
        self.sb_dataset._metadata = m
        after = m.memory_usage(deep=True).sum()
        print(f"metadata slimmed: {before / 1e9:.2f} GB -> {after / 1e9:.2f} GB per copy")

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

    @staticmethod
    def _normalize_anchor(segment: np.ndarray, anchor_len: int):
        """v2: causal normalisation (Reviewer 2, comment 5).

        Divide the whole window by the peak absolute value of its first
        ``anchor_len`` samples (default: the first token), one scale shared by
        all channels. The window starts at P, so the scale comes from the P
        onset only: every model input depends on the past alone, and the
        ratios between Z, N and E are kept. No mean, trend or epsilon: the
        synthetics start at rest, and a fixed epsilon is comparable to the P
        onset of small events (displacement in metres). Returns (scaled, scale).
        """
        peak = float(np.max(np.abs(segment[:, :anchor_len])))
        if peak > 0.0:
            return segment / peak, peak
        return segment, 1.0          # no signal in the anchor: leave unscaled

    def __getitem__(self, idx: int):
        return self.get_item(idx, self.training)

    def get_item(self, idx: int, training: bool):
        """One window. `training` switches the training-only augmentations; it is an
        argument (not only an attribute) so that the training and validation splits,
        which share this object, cannot change each other's setting."""
        # Load full waveform [C, T_total]
        waveform = self.sb_dataset.get_waveforms(idx)
        n_channels, n_samples_total = waveform.shape

        # Find P arrival (if available)
        meta = self.metadata.iloc[idx]
        p_sec = meta.get("trace_p_arrival_s", np.nan)
        if np.isfinite(p_sec):
            if self.random_shift:
                # v2: with the anchor normaliser the window must not start before P,
                # or the first token (the anchor) may contain no signal at all
                lo = 0 if self.norm_mode == "anchor" else -10
                p_sec += random.randint(lo, 10)
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
        if self.normalize and self.norm_mode == "anchor":
            segment, scale = self._normalize_anchor(segment, self.anchor_len)
            if scale == 1.0 and not self._warned_zero_anchor:
                log.warning(f"item {idx}: first token has no signal; left unscaled")
                self._warned_zero_anchor = True
        elif self.normalize:                                   # legacy_peak (submitted)
            segment = self._normalize_strain_max_per_channel(
                segment, demean=True, detrend=True, amp_norm="peak"
            )

        # Convert to torch: [L, C]
        segment = segment.T
        strain = torch.from_numpy(segment).float()*10

        # ── Training-only augmentations ───────────────────────────────────────
        if training:
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


class SplitView(Dataset):
    """One split (train / val / test) of a SeismicWaveformDataset, with its own
    training flag. Before v2 the splits shared one flag on the dataset object, so
    whichever dataloader was created last decided whether a worker augmented."""

    def __init__(self, dataset: SeismicWaveformDataset, indices, training: bool):
        self.dataset = dataset
        self.indices = np.asarray(indices, dtype=np.int64)     # compact, cheap to share
        self.training = bool(training)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.dataset.get_item(int(self.indices[i]), self.training)


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
        norm_mode: str = "anchor",
        anchor_len: int | None = None,
        # Keep data-loading workers alive between epochs? Default False since v2: the
        # full dataset (3.9 M traces) made worker memory grow until the job was killed;
        # fresh workers each epoch start clean, at the cost of a few seconds per epoch.
        persistent_workers: bool = False,
        slim_metadata: bool = True,       # training needs only a few columns (see dataset)
        # Workers for validation/test (None: min(num_workers, 4)). During validation the
        # training workers are still alive, so these come on top of them.
        val_num_workers: int | None = None,
        # How worker processes are started. "fork": workers share the parent's memory
        # (imports, metadata) and start instantly. Python 3.14 changed the default to
        # "forkserver": every worker is a fresh interpreter with its own copy of
        # everything (1-2 GB each), which exhausted the job's memory. None = Python default.
        mp_context: str | None = "fork",
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
        self.norm_mode = norm_mode
        self.anchor_len = anchor_len
        self.persistent_workers = persistent_workers
        self.slim_metadata = slim_metadata
        self.val_num_workers = min(num_workers, 4) if val_num_workers is None else val_num_workers
        self.mp_context = mp_context
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
            norm_mode=self.norm_mode,
            anchor_len=self.anchor_len,
            slim_metadata=self.slim_metadata,
        )
        train_len = int(0.8 * len(self.full_dataset))
        val_len = int(0.1 * len(self.full_dataset))
        test_len = len(self.full_dataset) - train_len - val_len
        train, val, test = random_split(
            self.full_dataset, [train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(42), # for reproducible train/val/test split
        )
        # same indices as before; each split now carries its own training flag
        self.train_dataset = SplitView(self.full_dataset, train.indices, training=True)
        self.val_dataset = SplitView(self.full_dataset, val.indices, training=False)
        self.test_dataset = SplitView(self.full_dataset, test.indices, training=False)

    def _loader(self, dataset, shuffle: bool, num_workers: int, collate: bool = True):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=self.persistent_workers and num_workers > 0,
            multiprocessing_context=self.mp_context if num_workers > 0 else None,
            collate_fn=self._make_collate_fn() if collate else None,
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
        return self._loader(self.train_dataset, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False, num_workers=self.val_num_workers)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False, num_workers=self.val_num_workers,
                            collate=False)
