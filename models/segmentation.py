"""
3D U-Net segmentation model. [PROPOSAL Stage 3]
Baseline backbone; nnU-Net is the target but requires its own setup.
[MONAI] uses MONAI UNet.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.inferers import SlidingWindowInferer


class TumorSegmenter(nn.Module):
    """
    3D U-Net segmenter. [PROPOSAL Stage 3]
    Trained on synthetic (volume, mask) pairs from Stage 3.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 2,          # background + tumor
        channels: tuple[int, ...] = (16, 32, 64, 128, 256),
        strides: tuple[int, ...] = (2, 2, 2, 2),
        num_res_units: int = 2,
        norm: str = "batch",
    ):
        super().__init__()
        # [MONAI]
        self.unet = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=norm,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(x)


def build_segmenter(cfg: dict) -> TumorSegmenter:
    m = cfg["model"]
    return TumorSegmenter(
        spatial_dims=m.get("spatial_dims", 3),
        in_channels=m.get("in_channels", 1),
        out_channels=m.get("out_channels", 2),
        channels=tuple(m.get("channels", [16, 32, 64, 128, 256])),
        strides=tuple(m.get("strides", [2, 2, 2, 2])),
        num_res_units=m.get("num_res_units", 2),
        norm=m.get("norm", "batch"),
    )


def get_loss():
    """Dice + Cross-Entropy loss, standard for medical segmentation. [MONAI]"""
    return DiceCELoss(to_onehot_y=True, softmax=True)


def get_sliding_window_inferer(patch_size: list[int], overlap: float = 0.25):
    """[MONAI] sliding window inference for full-volume prediction."""
    return SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=2,
        overlap=overlap,
        mode="gaussian",
    )
