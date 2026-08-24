"""A standard 2D U-Net implemented with native PyTorch layers."""

import torch
from torch import nn


class DoubleConvolution(nn.Module):
    """The two 3x3 convolution + ReLU operations used at each U-Net stage."""

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs):
        return self.layers(inputs)


class UNet2D(nn.Module):
    """Three-level encoder-decoder U-Net for one-channel binary segmentation.

    The input height and width must be divisible by 8, because the encoder
    downsamples three times. Mindboggle's 256 x 160 processed slices satisfy
    that requirement.
    """

    def __init__(self, base_channels=32):
        super().__init__()
        channels = base_channels

        # Encoder: retain features before each max-pooling operation for skips.
        self.encoder1 = DoubleConvolution(1, channels)
        self.encoder2 = DoubleConvolution(channels, channels * 2)
        self.encoder3 = DoubleConvolution(channels * 2, channels * 4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck: lowest-resolution, widest feature representation.
        self.bottleneck = DoubleConvolution(channels * 4, channels * 8)

        # Decoder: upsample, concatenate the matching encoder features, refine.
        self.up3 = nn.ConvTranspose2d(channels * 8, channels * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConvolution(channels * 8, channels * 4)
        self.up2 = nn.ConvTranspose2d(channels * 4, channels * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConvolution(channels * 4, channels * 2)
        self.up1 = nn.ConvTranspose2d(channels * 2, channels, kernel_size=2, stride=2)
        self.decoder1 = DoubleConvolution(channels * 2, channels)

        # One unthresholded cerebellum/background logit per input pixel.
        self.output_layer = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, images):
        if images.shape[-2] % 8 != 0 or images.shape[-1] % 8 != 0:
            raise ValueError("U-Net input height and width must both be divisible by 8.")
        encoder1 = self.encoder1(images)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        bottleneck = self.bottleneck(self.pool(encoder3))

        decoder3 = self.decoder3(torch.cat((self.up3(bottleneck), encoder3), dim=1))
        decoder2 = self.decoder2(torch.cat((self.up2(decoder3), encoder2), dim=1))
        decoder1 = self.decoder1(torch.cat((self.up1(decoder2), encoder1), dim=1))
        return self.output_layer(decoder1)
