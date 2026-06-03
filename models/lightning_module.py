"""
PyTorch Lightning module for training the GPT model (time-domain only) with
time-domain loss (MSE/L1/log-cosh/Gaussian or Laplace or Student-t NLL) and
optional multi-resolution STFT loss.

Flash Attention via is_causal; optional torch.compile.

"""
import math
from typing import Any, Dict, FrozenSet, Optional

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from lightning.pytorch.callbacks import Callback

from models.models import GPT, MultiResSTFTLoss, print_flash_attention_status, compile_model

# Gaussian ``nll``, Laplace ``nll_laplace``, and Student-t ``nll_studentt`` share 2*C outputs.
_PROBABILISTIC_TIME_LOSSES: FrozenSet[str] = frozenset(
    {"nll", "nll_laplace", "nll_studentt"}
)


def _is_sigma_param(name: str) -> bool:
    return "shared_sigma_head" in name


class TrunkDriftMonitor(Callback):
    """Logs trunk drift from a reference checkpoint (excluding sigma-head parameters)."""

    def __init__(self, reference_ckpt_path: str):
        super().__init__()
        self.reference_ckpt_path = reference_ckpt_path
        self._reference_trunk: dict[str, torch.Tensor] = {}
        self._warned = False

    def on_fit_start(self, trainer, pl_module: LightningModule) -> None:
        ckpt = torch.load(self.reference_ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        ref = {}
        missing = 0
        for name, _p in pl_module.named_parameters():
            if _is_sigma_param(name):
                continue
            if name in state:
                ref[name] = state[name].detach().cpu().float()
            else:
                missing += 1
        self._reference_trunk = ref
        if trainer.is_global_zero:
            print(
                f"[TrunkDriftMonitor] loaded {len(ref)} trunk tensors from reference ckpt; "
                f"missing={missing}"
            )

    def on_train_epoch_end(self, trainer, pl_module: LightningModule) -> None:
        if not self._reference_trunk:
            return
        diff_sq = 0.0
        ref_sq = 0.0
        with torch.no_grad():
            for name, p in pl_module.named_parameters():
                if _is_sigma_param(name):
                    continue
                ref = self._reference_trunk.get(name, None)
                if ref is None:
                    continue
                cur = p.detach().cpu().float()
                d = (cur - ref).pow(2).sum().item()
                r = ref.pow(2).sum().item()
                diff_sq += d
                ref_sq += r
        l2 = float(math.sqrt(max(diff_sq, 0.0)))
        rel = float(l2 / (math.sqrt(max(ref_sq, 0.0)) + 1e-12))
        pl_module.log("trunk_l2_drift", l2, on_epoch=True, sync_dist=True)
        pl_module.log("trunk_rel_drift", rel, on_epoch=True, sync_dist=True)
        if trainer.is_global_zero and trainer.current_epoch <= 1 and rel > 0.05 and not self._warned:
            print(
                f"[TrunkDriftMonitor][warn] trunk_rel_drift={rel:.4f} > 0.05 within first 2 epochs."
            )
            self._warned = True

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

    ``time_loss`` options:
        ``mse``, ``l1``, ``log_cosh`` — deterministic 1×C output.
        ``nll`` — Gaussian negative log-likelihood (backward-compatible name for Gaussian).
        ``nll_laplace`` — Laplace NLL; try first if residuals look heavy-tailed vs Gaussian.
        ``nll_studentt`` — Student-t NLL with fixed ``studentt_df``.

    The head still outputs 2×C channels when probabilistic; ``nll_logvar_min`` /
    ``nll_logvar_max`` clamp the learned scale (σ, b, or s) via
    ``exp(0.5 * min)`` … ``exp(0.5 * max)``. Recommended settings when retuning:
    ``nll_logvar_min: -12.0``, ``nll_logvar_max: 3.0``.
    """

    def __init__(
        self,
        # Model (GPT) args — must match data kernel_size
        in_channels: int = 3,
        kernel_size: int = 16,
        num_tokens: int = 256,
        d_model: int = 128,
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
        # Loss — see class docstring for ``time_loss`` options.
        time_loss: str = "l1",
        # Optional token-to-token temporal consistency (default OFF).
        temporal_delta_weight: float = 0.0,
        # Scale-parameter bounds (interpretation of kwarg names unchanged): for all
        # probabilistic losses (Gaussian σ, Laplace scale b, Student-t scale s),
        #   s_min = exp(0.5 * nll_logvar_min),  s_max = exp(0.5 * nll_logvar_max).
        # Defaults preserve backward compatibility; for a lower floor on scale (reducing
        # σ-collapse / stuck-at-min behavior), try nll_logvar_min: -12.0 and nll_logvar_max: 3.0.
        nll_logvar_min: float = -8.0,
        nll_logvar_max: float = 4.0,
        # Fixed ν for Student-t NLL (``time_loss == "nll_studentt"`` only); must be > 2.
        studentt_df: float = 4.0,
        lr: float = 1e-4,
        # Optional LR scheduler (flat args for CLI; omit scheduler or set null = no scheduler)
        scheduler: Optional[str] = None,
        scheduler_T_0: int = 5,
        scheduler_T_mult: int = 2,
        scheduler_eta_min: float = 1e-6,
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_frequency: int = 1,
        lr_scheduler_monitor: Optional[str] = None,
        # Optional parameter-group LR split for stage-2 probabilistic fine-tuning.
        # trunk LR = lr * lr_trunk_multiplier (all params except shared_sigma_head).
        lr_sigma_head: float = 3e-4,
        lr_trunk_multiplier: float = 1.0,
        # Optional warm-start protection: freeze trunk for first N epochs, train sigma head only.
        freeze_trunk_epochs: int = 0,
        # Linear LR warmup in **optimizer steps** before ``CosineAnnealingWarmRestarts``; 0 = off.
        # Uses ``interval: step``; ``scheduler_T_0`` / ``T_mult`` apply in **optimizer steps**.
        lr_warmup_steps: int = 0,
        # NEW: torch.compile() options
        use_torch_compile: bool = False,
        compile_mode: str = "reduce-overhead",
        # ── Multi-step prediction (MTP) ───────────────────────────────────
        # H prediction heads trained simultaneously on the same encoder output.
        # Head 0 (horizon-1) is the only head used at inference.
        # Auxiliary heads (h=2..H) force the encoder to represent long-range
        # structure — especially surface wave dispersion at large distances.
        # Weight for horizon h = mtp_weight_decay^(h-1): 1.0, 0.5, 0.25, ...
        num_pred_horizons: int = 1,     # H; set 1 to disable MTP
        mtp_weight_decay:  float = 0.5, # exponential decay of horizon weights
        # ── Multi-resolution STFT magnitude loss ───────────────────────────
        stft_enable: bool = False,
        stft_weight: float = 0.1,
        stft_n_ffts: tuple[int, ...] = (256, 1024, 4096),
        # MTP only: STFT magnitude loss on concat(pred_h, pred_{h+1}) vs concat(y_h, y_{h+1}).
        lambda_coherence: float = 0.1,
        # When False, load_state_dict from --ckpt_path is non-strict (needed for
        # deterministic pretrain → NLL finetune: new ``shared_sigma_head`` weights missing in ckpt).
        checkpoint_load_strict: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=("kwargs",))
        if not checkpoint_load_strict:
            self.strict_loading = False

        allowed = ("mse", "l1", "log_cosh", "nll", "nll_laplace", "nll_studentt")
        if time_loss not in allowed:
            raise ValueError(
                "time_loss must be one of 'mse', 'l1', 'log_cosh', 'nll', "
                "'nll_laplace', 'nll_studentt'"
            )
        # Same 2×C output head fits Gaussian, Laplace, and Student-t; only the loss formula changes.
        # Loading a Gaussian ``nll`` checkpoint into ``nll_laplace`` / ``nll_studentt`` is shape-compatible.
        # Loading a deterministic (1×C) checkpoint into any probabilistic mode still needs
        # ``checkpoint_load_strict=False`` (missing sigma head keys).
        self.time_loss = time_loss
        self.temporal_delta_weight = float(temporal_delta_weight)
        self.nll_logvar_min = float(nll_logvar_min)
        self.nll_logvar_max = float(nll_logvar_max)
        self.studentt_df = float(studentt_df)
        if self.studentt_df <= 2.0:
            raise ValueError(f"studentt_df must be > 2.0, got {self.studentt_df}")

        # Print Flash Attention status at initialization
        if torch.cuda.is_available():
            print_flash_attention_status()

        self.gpt = GPT(
            in_channels=in_channels,
            kernel_size=kernel_size,
            num_tokens=num_tokens,
            d_model=d_model,
            num_heads=num_heads,
            num_enc_layers=num_enc_layers,
            dropout=dropout,
            max_len=max_len,
            dim_feedforward_multiplier=dim_feedforward_multiplier,
            token_cnn_kernel=token_cnn_kernel,
            token_cnn_layers=token_cnn_layers,
            token_cnn_dilation_growth=token_cnn_dilation_growth,
            token_cnn_dropout=token_cnn_dropout,
            num_pred_horizons=num_pred_horizons,
            probabilistic_output=(self.time_loss in _PROBABILISTIC_TIME_LOSSES),
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
        self.lr_sigma_head = float(lr_sigma_head)
        self.lr_trunk_multiplier = float(lr_trunk_multiplier)
        self.freeze_trunk_epochs = int(freeze_trunk_epochs)
        self._trunk_frozen = False

        # Log-Cosh Loss
        self.log_cosh_loss = LogCoshLoss()

        # Multi-resolution STFT magnitude loss (optional)
        self.stft_enable = bool(stft_enable)
        self.stft_weight = float(stft_weight)
        self.stft_n_ffts = tuple(int(n) for n in stft_n_ffts)
        self.stft_loss = MultiResSTFTLoss(n_ffts=self.stft_n_ffts) if self.stft_enable else None
        self.lambda_coherence = float(lambda_coherence)
        # Coherence can use STFT even when per-sample stft_enable is False
        self._coherence_stft: Optional[MultiResSTFTLoss]
        if self.lambda_coherence > 0 and self.stft_loss is None:
            self._coherence_stft = MultiResSTFTLoss(n_ffts=self.stft_n_ffts)
        else:
            self._coherence_stft = None

        # MTP
        self.num_pred_horizons = max(1, int(num_pred_horizons))
        self.mtp_weight_decay  = float(mtp_weight_decay)

        if self.freeze_trunk_epochs > 0:
            frozen, trainable = 0, 0
            for n, p in self.named_parameters():
                if _is_sigma_param(n):
                    trainable += p.numel()
                    continue
                p.requires_grad = False
                frozen += p.numel()
            self._trunk_frozen = True
            print(
                f"[stage2] froze trunk for first {self.freeze_trunk_epochs} epoch(s): "
                f"frozen_params={frozen:,}, sigma_trainable={trainable:,}"
            )

    def _active_stft_loss(self) -> Optional[MultiResSTFTLoss]:
        return self.stft_loss if self.stft_loss is not None else self._coherence_stft

    def _cross_horizon_coherence_loss(
        self,
        predictions: list,
        y_horizon: torch.Tensor,
    ) -> torch.Tensor:
        """Sum of MultiResSTFTLoss over concatenated consecutive horizon pairs."""
        stft = self._active_stft_loss()
        if stft is None:
            return predictions[0].new_zeros(())
        H = len(predictions)
        total = predictions[0].new_zeros(())
        for h in range(H - 1):
            pred_h = self._prediction_mean(predictions[h])
            pred_hp = self._prediction_mean(predictions[h + 1])
            y_h = y_horizon[:, h, :, :]
            y_hp = y_horizon[:, h + 1, :, :]
            L = min(pred_h.shape[1], pred_hp.shape[1], y_h.shape[1], y_hp.shape[1])
            if L <= 0:
                continue
            pred_cat = torch.cat([pred_h[:, :L], pred_hp[:, :L]], dim=1)
            y_cat = torch.cat([y_h[:, :L], y_hp[:, :L]], dim=1)
            total = total + stft(
                pred_cat.transpose(1, 2).contiguous(),
                y_cat.transpose(1, 2).contiguous(),
            )
        return total

    def forward(self, x: torch.Tensor, is_causal: bool = True):
        """Thin wrapper around GPT (time-only). Returns List[Tensor]."""
        return self.gpt(x, is_causal=is_causal)

    def _prediction_mean(self, out: torch.Tensor) -> torch.Tensor:
        """Return mean prediction μ for deterministic and probabilistic (2×C) heads."""
        if self.time_loss in _PROBABILISTIC_TIME_LOSSES:
            return out[..., : self.hparams.in_channels]
        return out

    # =========================================================================
    # Multi-step prediction loss
    # =========================================================================
    def _multistep_loss(
        self,
        predictions: list,        # List[Tensor[B, L, C]]  length H
        y_horizon:   torch.Tensor, # [B, H, L, C]
    ):
        """
        Weighted sum of energy-weighted losses across H horizons.

        Weight for horizon h (1-indexed) = mtp_weight_decay^(h-1)
          h=1 → 1.00  (full weight — inference head, no down-weighting)
          h=2 → 0.50
          h=3 → 0.25  …

        Down-weighting distant horizons is important because:
          1. Far-horizon targets are harder and noisier (stochastic coda).
          2. We do not want auxiliary heads to dominate the h=1 gradient.

        Returns:
            total_loss    : weighted sum (scalar, differentiable)
            horizon_losses: list of (h_idx, weight, loss_h.detach()) for logging
        """
        total_loss    = predictions[0].new_zeros(())
        horizon_losses = []
        loss_delta_0 = None
        for idx, pred_h in enumerate(predictions):
            h_idx  = idx + 1                              # 1-indexed
            y_h    = y_horizon[:, idx, :, :]              # [B, L, C]
            if y_h.shape[1] != pred_h.shape[1]:
                continue  # length mismatch — skip (should not happen)

            loss_time_h, loss_delta_h = self._time_domain_loss(pred_h, y_h)
            if loss_delta_0 is None:
                loss_delta_0 = loss_delta_h
            loss_h = loss_time_h

            # Optional STFT loss (computed on waveforms [B, C, L])
            if self.stft_enable and self.stft_loss is not None and self.stft_weight > 0:
                pred_mu_h = self._prediction_mean(pred_h)
                loss_stft_h = self.stft_loss(
                    pred_mu_h.transpose(1, 2).contiguous(),
                    y_h.transpose(1, 2).contiguous(),
                )
                loss_h = loss_h + self.stft_weight * loss_stft_h
            else:
                loss_stft_h = None

            weight = self.mtp_weight_decay ** (h_idx - 1) # 1.0, 0.5, 0.25, …
            total_loss = total_loss + weight * loss_h
            # Store detached total loss for logging (includes STFT if enabled)
            horizon_losses.append((h_idx, weight, loss_h.detach()))
        loss_delta_0 = loss_delta_0 if loss_delta_0 is not None else predictions[0].new_zeros(())
        return total_loss, horizon_losses, loss_delta_0

    # =========================================================================
    # Time-domain loss
    # =========================================================================
    def _time_domain_loss(
        self,
        out: torch.Tensor,  # [B, L, C_out]  predictions
        y:   torch.Tensor,  # [B, L, C]      targets   (L must be divisible by K)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute time-domain loss (MSE, L1, log-cosh, Gaussian/Laplace/Student-t NLL) with
        uniform weighting over tokens. Optionally adds token-to-token temporal
        consistency term on the mean prediction.
        Returns (combined_loss, loss_delta) for logging.
        """
        K        = self.hparams.kernel_size
        B, L, _  = out.shape
        _, Ly, Cy = y.shape
        if Ly != L:
            raise ValueError(f"Prediction/target length mismatch: L_pred={L}, L_tgt={Ly}")
        T        = L // K
        if T * K != L:
            raise ValueError(
                f"Output length {L} must be divisible by kernel_size {K}"
            )
        y_tok   = y.view(B, T, K, Cy)

        if self.time_loss in _PROBABILISTIC_TIME_LOSSES:
            if out.shape[-1] != 2 * Cy:
                raise ValueError(
                    f"Probabilistic modes expect out channels = 2*C_target ({2 * Cy}), got {out.shape[-1]}"
                )
            mu, scale = out.split(Cy, dim=-1)
            out_tok_mu = mu.view(B, T, K, Cy)
            sig_tok = scale.view(B, T, K, Cy)
            smin = math.exp(0.5 * self.nll_logvar_min)
            smax = math.exp(0.5 * self.nll_logvar_max)
            sig_tok = sig_tok.clamp(min=smin, max=smax)

            if self.time_loss == "nll":
                residual = out_tok_mu - y_tok
                # Gaussian NLL: (y-mu)^2/(2*sigma^2) + log(sigma)
                base = (
                    0.5 * residual.pow(2) / (sig_tok.pow(2) + 1e-8)
                    + torch.log(sig_tok)
                ).mean()
            elif self.time_loss == "nll_laplace":
                # Laplace NLL: |y-mu|/b + log(b); ``sig_tok`` is scale b (same clamp as σ).
                diff = y_tok - out_tok_mu
                base = (torch.abs(diff) / sig_tok + torch.log(sig_tok)).mean()
            elif self.time_loss == "nll_studentt":
                nu = float(self.studentt_df)
                z2 = ((y_tok - out_tok_mu) / sig_tok).pow(2)
                # Student-t NLL up to additive constants that depend only on fixed ν (omit lgamma).
                # If ν is ever learned, reintroduce those constant terms.
                base = (
                    0.5 * (nu + 1.0) * torch.log1p(z2 / nu) + torch.log(sig_tok)
                ).mean()
            else:
                raise AssertionError(f"Unhandled probabilistic time_loss: {self.time_loss}")
        else:
            out_tok_mu = out.view(B, T, K, Cy)
            residual = out_tok_mu - y_tok
            if self.time_loss == "mse":
                base = residual.pow(2).mean()
            elif self.time_loss == "l1":
                base = residual.abs().mean()
            elif self.time_loss == "log_cosh":
                base = self.log_cosh_loss(out_tok_mu, y_tok)
            else:
                raise ValueError(
                    f"time_loss must be 'mse', 'l1', 'log_cosh', 'nll', "
                    f"'nll_laplace', or 'nll_studentt', got '{self.time_loss}'"
                )

        # Optional temporal-delta: encourages temporal consistency across neighboring
        # tokens to improve AR rollout stability. Only computed when weight > 0.
        if self.temporal_delta_weight != 0.0:
            d_out = out_tok_mu[:, 1:] - out_tok_mu[:, :-1]   # [B, T-1, K, C]
            d_y   = y_tok[:, 1:] - y_tok[:, :-1]
            loss_delta = F.mse_loss(d_out, d_y)
            combined = base + self.temporal_delta_weight * loss_delta
        else:
            loss_delta = out.new_zeros(())
            combined = base

        return combined, loss_delta

    def _shared_step(self, batch: dict, prefix: str) -> torch.Tensor:
        """
        Shared forward + loss for train and validation.

        Optional batch key: y_horizon [B, H, T*K, C] for multi-horizon targets (MTP).
        """
        x      = batch["x"]       # [B, T, C, K]
        y      = batch["y"]       # [B, T*K, C]  (horizon-1, always present)

        y_horizon = batch.get("y_horizon", None)   # [B, H, T*K, C] or None

        # Forward (time-only)
        predictions = self(x, is_causal=True)

        # predictions[0] is always the horizon-1 (inference) head
        out = predictions[0]   # [B, T*K, C]
        out_mu = self._prediction_mean(out)

        # ── Loss ─────────────────────────────────────────────────────────────
        if y_horizon is not None and len(predictions) > 1:
            loss_time, horizon_losses, loss_delta = self._multistep_loss(predictions, y_horizon)
        else:
            loss_time, loss_delta = self._time_domain_loss(out, y)
            horizon_losses = [(1, 1.0, loss_time.detach())]

        loss = loss_time

        # Optional STFT loss for the inference head (horizon-1) when MTP is off
        # or when y_horizon is not provided. When MTP is active, STFT is already
        # included inside _multistep_loss for every head.
        if (y_horizon is None or len(predictions) <= 1) and self.stft_enable and self.stft_loss is not None and self.stft_weight > 0:
            loss_stft = self.stft_loss(
                out_mu.transpose(1, 2).contiguous(),
                y.transpose(1, 2).contiguous(),
            )
            loss = loss + self.stft_weight * loss_stft
        else:
            loss_stft = None

        loss_coherence = None
        if (
            self.training
            and self.lambda_coherence > 0
            and self.num_pred_horizons > 1
            and y_horizon is not None
            and len(predictions) > 1
        ):
            stft_c = self._active_stft_loss()
            if stft_c is not None:
                loss_coherence = self._cross_horizon_coherence_loss(predictions, y_horizon)
                loss = loss + self.lambda_coherence * loss_coherence

        # ── Logging (minimal: reduces TensorBoard/sync overhead) ─────────────
        # Total `{prefix}_loss` keeps EarlyStopping / ModelCheckpoint on ``val_loss`` working.
        self.log(f"{prefix}_loss_time", loss_time, sync_dist=True)
        if loss_stft is not None:
            self.log(f"{prefix}_stft", loss_stft, sync_dist=True)
        if prefix == "val":
            self.log(f"{prefix}_mse", F.mse_loss(out_mu, y), sync_dist=True)
        # MTP: unweighted per-horizon loss (time + per-head STFT if any) for depth diagnostics
        if len(horizon_losses) > 1:
            for h_idx, _w, l_h in horizon_losses:
                self.log(f"{prefix}_mtp_h{h_idx}_raw", l_h, sync_dist=True)
        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def on_train_epoch_start(self) -> None:
        if (
            self.freeze_trunk_epochs > 0
            and self._trunk_frozen
            and self.current_epoch >= self.freeze_trunk_epochs
        ):
            count = 0
            for n, p in self.named_parameters():
                if _is_sigma_param(n):
                    continue
                p.requires_grad = True
                count += p.numel()
            self._trunk_frozen = False
            print(
                f"[stage2] Unfroze trunk at epoch {self.current_epoch}; "
                f"trainable trunk params={count:,}."
            )

    def configure_optimizers(self):
        sigma_params = []
        trunk_params = []
        for n, p in self.named_parameters():
            if _is_sigma_param(n):
                sigma_params.append(p)
            else:
                trunk_params.append(p)

        param_groups = []
        if trunk_params:
            param_groups.append(
                {
                    "params": trunk_params,
                    "lr": self.lr * self.lr_trunk_multiplier,
                    "name": "trunk",
                }
            )
        if sigma_params:
            param_groups.append(
                {"params": sigma_params, "lr": self.lr_sigma_head, "name": "sigma"}
            )
        if not param_groups:
            raise RuntimeError("No trainable parameters found for optimizer setup.")

        optimizer = torch.optim.AdamW(param_groups, lr=self.lr)
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

        main_scheduler = sched_class(optimizer, **kwargs)
        warmup_steps = int(self.hparams.lr_warmup_steps)
        use_warmup = warmup_steps > 0 and self.scheduler == "CosineAnnealingWarmRestarts"
        if warmup_steps > 0 and not use_warmup:
            import warnings

            warnings.warn(
                "lr_warmup_steps > 0 only chains with scheduler CosineAnnealingWarmRestarts; "
                "warmup ignored for other schedulers.",
                UserWarning,
                stacklevel=2,
            )

        if use_warmup:
            from torch.optim.lr_scheduler import LinearLR, SequentialLR

            warmup_sched = LinearLR(
                optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_sched, main_scheduler],
                milestones=[warmup_steps],
            )
            interval = "step"
        else:
            scheduler = main_scheduler
            interval = self.lr_scheduler_interval

        if len(param_groups) == 2:
            lrs_now = scheduler.get_last_lr() if hasattr(scheduler, "get_last_lr") else []
            if len(lrs_now) != 2:
                raise RuntimeError(
                    f"Expected 2 scheduler LRs for trunk/sigma groups, got {len(lrs_now)}."
                )

        lr_config = {
            "scheduler": scheduler,
            "interval": interval,
            "frequency": self.lr_scheduler_frequency,
        }
        if self.lr_scheduler_monitor:
            lr_config["monitor"] = self.lr_scheduler_monitor
        return {"optimizer": optimizer, "lr_scheduler": lr_config}