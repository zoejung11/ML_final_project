"""Train and test the custom PyTorch 2D U-Net on fixed Mindboggle splits."""

import argparse
from pathlib import Path

from Mindboggle.models.training import train_and_test
from Mindboggle.models.unet import UNet2D


def main():
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=project_root / "Mindboggle" / "data_processed")
    parser.add_argument("--output-dir", type=Path, default=project_root / "Mindboggle" / "model_outputs" / "unet")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_and_test(UNet2D(), "UNet2D", "3-level 2D U-Net; channels 32, 64, 128, 256; skip connections", **vars(args))


if __name__ == "__main__":
    main()
