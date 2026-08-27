import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        # Flatten spatial dimensions per batch sample
        flat_predictions = predictions.view(predictions.size(0), -1)
        flat_targets = targets.view(targets.size(0), -1)

        intersection = (flat_predictions * flat_targets).sum(dim=1)
        cardinality = flat_predictions.sum(dim=1) + flat_targets.sum(dim=1)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice_score.mean()


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, predictions, targets):
        # Compute individual loss components
        bce_loss = self.bce(predictions, targets)
        dice_loss = self.dice(predictions, targets)
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)
