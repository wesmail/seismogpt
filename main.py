# PyTorch imports
import torch

# PyTorch Lightning imports
from lightning.pytorch.cli import LightningCLI


def cli_main():
    cli = LightningCLI()
    # note: don't call fit!!

if __name__ == "__main__":
    #comment if you don't have a GPU with tensor cores
    torch.set_float32_matmul_precision("high")
    cli_main()
