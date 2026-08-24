"""MONAI-configured 2D U-Net for binary cerebellum segmentation."""

from monai.networks.nets import UNet


def build_monai_unet():
    """Return a 2D medical-imaging U-Net for one grayscale input and one mask.

    The channel progression mirrors the custom U-Net comparison: 32, 64, 128,
    and 256 feature maps. MONAI residual units provide a short path for feature
    information and gradients, which can make optimization more stable.
    """
    return UNet(
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
    )
