import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


def get_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_dice_score(predictions, targets, threshold=0.5, smooth=1e-6):
    binary_predictions = (predictions > threshold).float()

    flattened_predictions = binary_predictions.view(binary_predictions.size(0), -1)
    flattened_targets = targets.view(targets.size(0), -1)

    intersection = torch.sum(flattened_predictions * flattened_targets, dim=1)
    cardinality = torch.sum(flattened_predictions, dim=1) + torch.sum(
        flattened_targets, dim=1
    )

    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return dice.mean().item()