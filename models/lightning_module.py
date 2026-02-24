"""
PyTorch Lightning module for training the GPT model (time-domain only) with
time-domain loss (MSE/L1) and multi-resolution STFT loss.

OPTIMIZATIONS:
- Flash Attention enabled (via is_causal flag in model)
- Optional torch.compile() for additional speedup
- Flash Attention status logging at init
"""
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from models.models import GPT, MultiResSTFTLoss, print_flash_attention_status, compile_model
from data_handling.data_handling import freq_features_from_tokens

# =============================================================================
# Log-Cosh Loss
# =============================================================================
import torch.nn as nn
class LogCoshLoss(nn.Module):
    def forward(self, x, y):  # x, y: [B, T, F]
        diff = x - y
        return torch.mean(torch.log(torch.cosh(diff + 1e-12)))
    

class GPTLightning(LightningModule):
    """
    Lightning module for time-only GPT: takes x (time tokens) and y (time target),
    returns predictions and computes time + STFT loss.
    """

    def __init__(
        self,
        # Model (GPT) args — must match data kernel_size
        in_channels: int = 3,
        kernel_size: int = 16,
        num_tokens: int = 256,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 2,
        num_enc_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 5000,
        dim_feedforward_multiplier: int = 4,
        # Token causal CNN embedding
        token_cnn_kernel: int = 7,
        token_cnn_layers: int = 4,
        token_cnn_dilation_growth: int = 2,
        token_cnn_dropout: float = 0.0,
        # Post-head stitcher (smooths predictions over sample axis)
        use_stitcher: bool = True,
        stitcher_hidden: int = 64,
        stitcher_kernel: int = 9,
        stitcher_layers: int = 4,
        stitcher_dropout: float = 0.0,
        # Loss
        time_loss: str = "l1",
        fusion_type: str = "cross_attention",
        lr: float = 1e-4,
        # --- NEW: frequency embedding / gate config ---
        freq_embed_type: str = "mlp",       # "mlp" or "conv" (legacy)
        freq_keep_bins: int = 8,            # F bins per channel
        freq_gate: bool = True,             # learned gate on freq branch
        # --- NEW: per-token spectrum normalization (must match dataset) ---
        freq_norm: str = "none",            # "none" | "mean" | "l2"
        freq_log1p: bool = True,            # log(1+|FFT|) — must match dataset
        # --- NEW: scheduled sampling for rollout stability ---
        ss_enable: bool = False,            # enable scheduled sampling unroll
        ss_unroll_tokens: int = 8,          # U: number of AR unroll steps
        ss_start_prob: float = 0.0,         # initial probability of using predicted token
        ss_end_prob: float = 0.5,           # final probability
        ss_warmup_steps: int = 5000,        # steps to linearly ramp from start to end
        ss_future_only_loss: bool = True,   # only compute loss on unrolled predictions
        ss_bias_window_frac: float = 0.4,   # fraction of T to bias start toward end
        # Optional LR scheduler (flat args for CLI; omit scheduler or set null = no scheduler)
        scheduler: Optional[str] = None,
        scheduler_T_0: int = 5,
        scheduler_T_mult: int = 2,
        scheduler_eta_min: float = 1e-6,
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_frequency: int = 1,
        lr_scheduler_monitor: Optional[str] = None,
        # NEW: torch.compile() options
        use_torch_compile: bool = False,
        compile_mode: str = "reduce-overhead",
    ):
        super().__init__()
        self.save_hyperparameters()
        
        if time_loss not in ("mse", "l1", "log_cosh"):
            raise ValueError("time_loss must be 'mse', 'l1', or 'log_cosh'")
        self.time_loss = time_loss

        # Print Flash Attention status at initialization
        if torch.cuda.is_available():
            print_flash_attention_status()

        self.gpt = GPT(
            in_channels=in_channels,
            kernel_size=kernel_size,
            num_tokens=num_tokens,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            num_enc_layers=num_enc_layers,
            dropout=dropout,
            max_len=max_len,
            dim_feedforward_multiplier=dim_feedforward_multiplier,
            token_cnn_kernel=token_cnn_kernel,
            token_cnn_layers=token_cnn_layers,
            token_cnn_dilation_growth=token_cnn_dilation_growth,
            token_cnn_dropout=token_cnn_dropout,
            use_stitcher=use_stitcher,
            stitcher_hidden=stitcher_hidden,
            stitcher_kernel=stitcher_kernel,
            stitcher_layers=stitcher_layers,
            stitcher_dropout=stitcher_dropout,
            fusion_type=fusion_type,
            # --- NEW ---
            freq_embed_type=freq_embed_type,
            freq_keep_bins=freq_keep_bins,
            freq_gate=freq_gate,
        )
        
        # Optional: torch.compile() for additional speedup (PyTorch 2.0+)
        self.use_torch_compile = use_torch_compile
        if use_torch_compile:
            self.gpt = compile_model(self.gpt, mode=compile_mode)

        self.lr = lr
        self.scheduler = scheduler
        self.scheduler_T_0 = scheduler_T_0
        self.scheduler_T_mult = scheduler_T_mult
        self.scheduler_eta_min = scheduler_eta_min
        self.lr_scheduler_interval = lr_scheduler_interval
        self.lr_scheduler_frequency = lr_scheduler_frequency
        self.lr_scheduler_monitor = lr_scheduler_monitor

        # Log-Cosh Loss
        self.log_cosh_loss = LogCoshLoss()

        # Scheduled sampling config (stored for use in training_step)
        self.ss_enable = ss_enable
        self.ss_unroll_tokens = ss_unroll_tokens
        self.ss_start_prob = ss_start_prob
        self.ss_end_prob = ss_end_prob
        self.ss_warmup_steps = ss_warmup_steps
        self.ss_future_only_loss = ss_future_only_loss
        self.ss_bias_window_frac = ss_bias_window_frac
        # Freq feature config needed to recompute x_freq during unroll
        self.freq_norm = freq_norm
        self.freq_log1p = freq_log1p
        self.freq_keep_bins = freq_keep_bins

    def forward(self, x: torch.Tensor, x_freq: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        return self.gpt(x, x_freq, is_causal=is_causal)

    def _shared_step(self, batch: dict, prefix: str) -> torch.Tensor:
        x = batch["x"]            # [B, T, C, K]
        x_freq = batch["x_freq"]  # [B, T, ...]
        y = batch["y"]            # [B, T*K, C]

        out = self(x, x_freq, is_causal=True)  # [B, T*K, C]

        # ---------------------------------------------------------
        # Reshape to tokens
        # ---------------------------------------------------------
        B, L, C = out.shape
        K = self.hparams.kernel_size
        T = L // K
        if T * K != L:
            raise ValueError("Output length must be multiple of kernel_size")

        out_tok = out.view(B, T, K, C)
        y_tok   = y.view(B, T, K, C)

        # ---------------------------------------------------------
        # Per-token energy (RMS)
        # ---------------------------------------------------------
        eps = 1e-6
        energy = y_tok.pow(2).mean(dim=(2, 3)).sqrt()  # [B, T]

        # ---------------------------------------------------------
        # Soft energy weighting
        # ---------------------------------------------------------
        alpha = 0.5   # <--- IMPORTANT HYPERPARAMETER
        weights = (energy + eps) ** alpha              # [B, T]
        weights = weights / (weights.mean() + eps)     # normalize for stability

        # ---------------------------------------------------------
        # Base per-sample loss (no normalization!)
        # ---------------------------------------------------------
        residual = out_tok - y_tok

        if self.time_loss == "mse":
            base_loss = residual.pow(2)
        elif self.time_loss == "l1":
            base_loss = residual.abs()
        elif self.time_loss == "log_cosh":
            base_loss = torch.log(torch.cosh(residual + 1e-12))
        else:
            raise ValueError("time_loss must be 'mse', 'l1', or 'log_cosh'")

        # base_loss: [B, T, K, C] → reduce over samples
        base_loss = base_loss.mean(dim=(2, 3))         # [B, T]

        # ---------------------------------------------------------
        # Energy-weighted loss
        # ---------------------------------------------------------
        loss = (weights * base_loss).mean()

        # ---------------------------------------------------------
        # Logging
        # ---------------------------------------------------------
        plain_mse = F.mse_loss(out, y)

        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}_mse", plain_mse, sync_dist=True)
        self.log(f"{prefix}_energy_mean", energy.mean(), sync_dist=True)
        self.log(f"{prefix}_energy_min", energy.min(), sync_dist=True)
        self.log(f"{prefix}_energy_max", energy.max(), sync_dist=True)

        with torch.no_grad():
            self.log(f"{prefix}_pred_std", out.std(), sync_dist=True)
            self.log(f"{prefix}_target_std", y.std(), sync_dist=True)

        return loss


    def _ss_prob(self) -> float:
        """Linearly ramp scheduled sampling probability from start to end over warmup."""
        if not self.ss_enable or self.ss_warmup_steps <= 0:
            return 0.0
        t = min(self.global_step / self.ss_warmup_steps, 1.0)
        return self.ss_start_prob + t * (self.ss_end_prob - self.ss_start_prob)

    def _compute_freq(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """Recompute freq features from time tokens (matches dataset pipeline)."""
        return freq_features_from_tokens(
            x_tokens,
            freq_keep_bins=self.freq_keep_bins,
            freq_log1p=self.freq_log1p,
            freq_norm=self.freq_norm,
        )

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        # Default path: no scheduled sampling (preserves original behavior exactly)
        if not self.ss_enable or self.ss_unroll_tokens <= 0:
            return self._shared_step(batch, "train")

        # =====================================================================
        # Scheduled sampling: token-level closed-loop unroll (exposure bias reduction)
        # Train with U steps of autoregressive rollout so the model sees its own
        # predictions (and their FFT features) during training, closing the
        # train/inference gap.
        # =====================================================================
        x = batch["x"]            # [B, T, C, K]
        y = batch["y"]            # [B, T*K, C]
        B, T, C, K = x.shape
        U = min(self.ss_unroll_tokens, T - 1)  # clamp to sequence length
        if U <= 0:
            return self._shared_step(batch, "train")

        p = self._ss_prob()
        self.log("ss_prob", p, prog_bar=False)

        # Pick start index biased toward later tokens (harder merger region)
        bias_window = max(1, int(self.ss_bias_window_frac * T))
        earliest = max(0, T - U - bias_window)
        latest = T - U
        start = torch.randint(earliest, latest + 1, (1,)).item()

        # Build initial context from ground truth
        x_ctx = x[:, :start, :, :].clone()  # [B, start, C, K]

        # Collect predictions for loss
        pred_tokens_list = []    # each [B, K, C]
        gt_tokens_list = []      # each [B, K, C]

        for t_offset in range(U):
            t_idx = start + t_offset
            # Recompute freq features from current context
            x_freq_ctx = self._compute_freq(x_ctx)  # [B, ctx_len, C, F]

            # Forward through model
            out = self(x_ctx, x_freq_ctx, is_causal=True)  # [B, ctx_len*K, C]

            # Extract last-token prediction (K samples)
            pred_samples = out[:, -K:, :]  # [B, K, C]
            pred_tokens_list.append(pred_samples)

            # Ground truth for this token
            gt_token = x[:, t_idx, :, :]              # [B, C, K]
            gt_samples = gt_token.permute(0, 2, 1)    # [B, K, C]
            gt_tokens_list.append(gt_samples)

            # Scheduled sampling: with prob p use prediction, else ground truth
            pred_as_token = pred_samples.detach().permute(0, 2, 1).unsqueeze(1)  # [B,1,C,K]
            gt_as_token = x[:, t_idx:t_idx+1, :, :]    # [B,1,C,K]

            with torch.no_grad():
                use_pred = (torch.rand(1, device=x.device) < p).item()
            next_token = pred_as_token if use_pred else gt_as_token

            x_ctx = torch.cat([x_ctx, next_token], dim=1)  # [B, ctx_len+1, C, K]

        # Stack predictions: [B, U*K, C]
        pred_all = torch.cat(pred_tokens_list, dim=1)  # [B, U*K, C]
        gt_all = torch.cat(gt_tokens_list, dim=1)      # [B, U*K, C]

        # Compute loss on unrolled tokens
        if self.ss_future_only_loss:
            out_flat = pred_all
            y_flat = gt_all
        else:
            # Fall back to full-sequence loss via _shared_step
            return self._shared_step(batch, "train")

        # Reshape to tokens for energy-weighted loss (same as _shared_step)
        B2, L2, C2 = out_flat.shape
        T2 = L2 // K
        out_tok = out_flat.view(B2, T2, K, C2)
        y_tok = y_flat.view(B2, T2, K, C2)

        eps = 1e-6
        energy = y_tok.pow(2).mean(dim=(2, 3)).sqrt()
        alpha = 0.5
        weights = (energy + eps) ** alpha
        weights = weights / (weights.mean() + eps)

        residual = out_tok - y_tok
        if self.time_loss == "mse":
            base_loss = residual.pow(2)
        elif self.time_loss == "l1":
            base_loss = residual.abs()
        elif self.time_loss == "log_cosh":
            base_loss = torch.log(torch.cosh(residual + 1e-12))
        else:
            raise ValueError(f"Unknown time_loss='{self.time_loss}'")

        base_loss = base_loss.mean(dim=(2, 3))
        loss = (weights * base_loss).mean()

        # Logging
        plain_mse = F.mse_loss(out_flat, y_flat)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_mse", plain_mse, sync_dist=True)
        self.log("train_ss_unroll_start", float(start), sync_dist=True)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if not self.scheduler:
            return optimizer

        sched_class = getattr(torch.optim.lr_scheduler, self.scheduler, None)
        if sched_class is None:
            raise ValueError(f"Unknown scheduler: {self.scheduler}. Use a PyTorch name, e.g. ReduceLROnPlateau, CosineAnnealingLR.")

        if self.scheduler == "CosineAnnealingLR":
            kwargs = {"T_max": getattr(self.trainer, "max_epochs", 10)}
        elif self.scheduler == "CosineAnnealingWarmRestarts":
            kwargs = {"T_0": self.scheduler_T_0, "T_mult": self.scheduler_T_mult, "eta_min": self.scheduler_eta_min}
        else:
            kwargs = {}

        scheduler = sched_class(optimizer, **kwargs)
        lr_config = {
            "scheduler": scheduler,
            "interval": self.lr_scheduler_interval,
            "frequency": self.lr_scheduler_frequency,
        }
        if self.lr_scheduler_monitor:
            lr_config["monitor"] = self.lr_scheduler_monitor
        return {"optimizer": optimizer, "lr_scheduler": lr_config}