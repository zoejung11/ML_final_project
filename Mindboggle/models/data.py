"""Read the already-preprocessed Mindboggle PNGs for model training.

This module is read-only with respect to ``data_processed``.  It does not
rerun preprocessing, alter PNGs, or create new masks.
"""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class CerebellumSliceDataset(Dataset):
    """One existing train, validation, or test split of paired PNG slices."""

    def __init__(self, data_root, split):
        self.data_root = Path(data_root)
        self.split = split
        self.images_dir = self.data_root / "images" / split
        self.masks_dir = self.data_root / "masks" / split
        self.image_paths = sorted(self.images_dir.glob("*.png"))

        if not self.image_paths:
            raise FileNotFoundError(f"No PNG images found in {self.images_dir}")
        missing_masks = [path.name for path in self.image_paths if not (self.masks_dir / path.name).is_file()]
        if missing_masks:
            raise FileNotFoundError(f"Missing masks for {len(missing_masks)} image(s), e.g. {missing_masks[:3]}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.masks_dir / image_path.name
        # PNGs have already been standardized by preprocessing. Dividing by 255
        # changes only their numeric scale for the network, not their content.
        image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32) / 255.0
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) > 127
        if image.shape != mask.shape:
            raise ValueError(f"Image/mask shape mismatch: {image_path.name}")
        return torch.from_numpy(image[None]), torch.from_numpy(mask.astype(np.float32)[None]), image_path.name
