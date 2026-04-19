"""
CT preprocessing transforms using MONAI. [MONAI]
All transforms work on dict keys 'image' and optionally 'label'.
"""
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandCropByLabelClassesd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandFlipd,
    EnsureTyped,
    SpatialPadd,
    NormalizeIntensityd,
    RandSpatialCropd,
)
from monai.data import MetaTensor
import torch


# [PROPOSAL] CT window: [-150, 250] HU for abdominal soft tissue
CT_WIN_MIN = -150.0
CT_WIN_MAX = 250.0


def get_autoencoder_transforms(
    patch_size: list[int] = [96, 96, 96],
    pixdim: tuple[float, ...] = (1.5, 1.5, 1.5),
    is_train: bool = True,
) -> Compose:
    """
    Stage 1 transforms — unlabeled CT only (no label key required). [PROPOSAL Stage 1]
    Resamples to isotropic 1.5mm, windows HU, crops foreground, extracts random patch.
    """
    base = [
        LoadImaged(keys=["image"], image_only=True),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=pixdim, mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=CT_WIN_MIN,
            a_max=CT_WIN_MAX,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > 0.05),
        SpatialPadd(keys=["image"], spatial_size=patch_size),
        RandSpatialCropd(keys=["image"], roi_size=patch_size, random_size=False),
    ]

    if is_train:
        aug = [
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            RandGaussianNoised(keys=["image"], prob=0.15, std=0.02),
            RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.5, 1.0)),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.2),
            RandShiftIntensityd(keys=["image"], offsets=0.05, prob=0.2),
        ]
        base.extend(aug)

    base.append(EnsureTyped(keys=["image"], dtype=torch.float32))
    return Compose(base)


def get_diffusion_transforms(
    patch_size: list[int] = [96, 96, 96],
    pixdim: tuple[float, ...] = (1.5, 1.5, 1.5),
    is_train: bool = True,
    num_samples: int = 2,
) -> Compose:
    """
    Stage 2 transforms — labeled CT with tumor mask. [PROPOSAL Stage 2]
    Crops patches centered on tumor region for efficient conditioning training.
    """
    base = [
        LoadImaged(keys=["image", "label"], image_only=False),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=pixdim, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=CT_WIN_MIN,
            a_max=CT_WIN_MAX,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", select_fn=lambda x: x > 0.05),
        SpatialPadd(keys=["image", "label"], spatial_size=patch_size),
        # [PROPOSAL] crop patches with positive tumor samples guaranteed
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=patch_size,
            pos=3,
            neg=1,
            num_samples=num_samples,
            image_key="image",
            image_threshold=0.05,
        ),
    ]

    if is_train:
        aug = [
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandGaussianNoised(keys=["image"], prob=0.1, std=0.02),
            RandScaleIntensityd(keys=["image"], factors=0.1, prob=0.2),
        ]
        base.extend(aug)

    base.append(EnsureTyped(keys=["image", "label"], dtype=torch.float32))
    return Compose(base)


def get_segmentation_transforms(
    patch_size: list[int] = [96, 96, 96],
    pixdim: tuple[float, ...] = (1.5, 1.5, 1.5),
    is_train: bool = True,
    num_samples: int = 4,
) -> Compose:
    """Stage 3 segmentation training transforms. [PROPOSAL Stage 3]"""
    return get_diffusion_transforms(patch_size, pixdim, is_train, num_samples)
