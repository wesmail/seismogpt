"""GRU baseline: a recurrent network on the same tokens as SeismoGPT.

Trained like SeismoGPT, with the same data module, data, splits, augmentations,
precision and trainer (see configs/baseline_gru.yaml):

    python main.py fit --config configs/baseline_gru.yaml

    token [C, K] --flatten--> Linear --> GRU (num_layers) --> Linear --> next token [C, K]

Loss: log-cosh between the predicted and the true next token (teacher forcing), the
main time-domain term of SeismoGPT's loss. SeismoGPT's extra terms (multi-horizon,
STFT, coherence) are left out to keep the baseline simple; say so in the paper.
"""
import math

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule


def log_cosh(d: torch.Tensor) -> torch.Tensor:
    """log(cosh(d)) without overflow: |d| + log(1 + exp(-2|d|)) - log 2.
    The plain formula gives inf for |d| > ~89, and anchor-normalised windows reach
    values of 1e3-1e4 (S and surface waves are much larger than the P onset)."""
    a = d.abs()
    return a + nn.functional.softplus(-2.0 * a) - math.log(2.0)


class TokenGRU(nn.Module):
    def __init__(self, in_channels=3, kernel_size=16, hidden=512, num_layers=3, dropout=0.1):
        super().__init__()
        self.C, self.K = in_channels, kernel_size
        d = in_channels * kernel_size
        self.embed = nn.Linear(d, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, d)

    def forward(self, x, h=None):
        """x [B, T, C, K] (+ state h) -> next-token predictions [B, T, C, K], new state."""
        B, T = x.shape[:2]
        z, h = self.gru(self.embed(x.reshape(B, T, -1)), h)
        return self.head(z).reshape(B, T, self.C, self.K), h


class GRULightning(LightningModule):
    def __init__(self, in_channels: int = 3, kernel_size: int = 16, hidden: int = 512,
                 num_layers: int = 3, dropout: float = 0.1, lr: float = 1e-3,
                 weight_decay: float = 0.01, warmup_steps: int = 1000,
                 amp_compress: bool = False):
        super().__init__()
        self.save_hyperparameters()
        self.net = TokenGRU(in_channels, kernel_size, hidden, num_layers, dropout)

    def _step(self, batch, prefix):
        x = batch["x"]                                            # [B, T, C, K]
        B, T, C, K = x.shape
        y = batch["y"].reshape(B, T, K, C).permute(0, 1, 3, 2)    # next tokens, as [B, T, C, K]
        if self.hparams.amp_compress:                             # same option as SeismoGPT
            from models.lightning_module import compress_amplitude
            x, y = compress_amplitude(x), compress_amplitude(y)
        pred, _ = self.net(x)
        loss = log_cosh(pred.float() - y.float()).mean()
        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        if prefix == "val":
            self.log("val_mse", torch.mean((pred.float() - y.float()) ** 2), sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr,
                                weight_decay=self.hparams.weight_decay)
        total = max(1, int(self.trainer.estimated_stepping_batches))
        warm = min(self.hparams.warmup_steps, total // 10)

        def lr_factor(step):                                       # linear warm-up, then cosine
            if step < warm:
                return (step + 1) / warm
            return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def on_fit_start(self):
        n = sum(p.numel() for p in self.parameters())
        print(f"GRU baseline: {n / 1e6:.2f} M parameters")
