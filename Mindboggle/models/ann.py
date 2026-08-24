"""A non-convolutional baseline for binary cerebellum segmentation."""

import torch
from torch import nn


class PixelANN(nn.Module):
    """Classify every pixel from intensity plus its normalized x/y location.

    The same small multilayer perceptron is applied independently to each
    pixel. This deliberately gives the baseline no neighboring-pixel context,
    unlike the CNNs and U-Nets used later in the project.
    """

    def __init__(self, hidden_sizes=(32, 16)):
        super().__init__()
        layers = []
        features = 3  # MRI intensity, x coordinate, y coordinate
        for hidden_size in hidden_sizes:
            layers.extend((nn.Linear(features, hidden_size), nn.ReLU()))
            features = hidden_size
        layers.append(nn.Linear(features, 1))  # one unthresholded logit per pixel
        self.network = nn.Sequential(*layers)

    def forward(self, images):
        batch_size, _, height, width = images.shape
        y_coordinates, x_coordinates = torch.meshgrid(
            torch.linspace(-1, 1, height, device=images.device),
            torch.linspace(-1, 1, width, device=images.device),
            indexing="ij",
        )
        coordinates = torch.stack((x_coordinates, y_coordinates)).expand(batch_size, -1, -1, -1)
        features = torch.cat((images, coordinates), dim=1).permute(0, 2, 3, 1)
        return self.network(features).permute(0, 3, 1, 2)
