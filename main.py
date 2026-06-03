# PyTorch imports
import torch

# PyTorch Lightning imports
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback


class PrintResolvedSaveConfig(SaveConfigCallback):
    """Save merged CLI config to the run log dir (default Lightning behavior) and print it on rank 0."""

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        if trainer.is_global_zero and stage == "fit" and not self.already_saved:
            try:
                text = self.parser.dump(self.config, format="yaml", skip_none=False)
            except Exception:
                text = str(self.config)
            print("======== Resolved training config (rank 0) ========")
            print(text)
            print("====================================================")
        super().setup(trainer, pl_module, stage)


def cli_main():
    LightningCLI(
        save_config_callback=PrintResolvedSaveConfig,
        save_config_kwargs={
            "config_filename": "resolved_config.yaml",
            "overwrite": True,
        },
    )


if __name__ == "__main__":
    # comment if you don't have a GPU with tensor cores
    torch.set_float32_matmul_precision("high")
    cli_main()
