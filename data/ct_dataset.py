"""
CT dataset loaders for MSD (pancreas), LiTS (liver), KiTS (kidney). [PROPOSAL data]
All three datasets store NIfTI files; directory layout documented below.

Expected raw_dir layout (set in configs/base.yaml → data.raw_dir):

  data/raw/
    pancreas/
      images/   *.nii.gz   (MSD Task07_Pancreas imagesTr/)
      labels/   *.nii.gz   (MSD labelsTr/ — label 1=pancreas, 2=tumor; we use 2)
    liver/
      images/   *.nii.gz   (LiTS volume-*.nii)
      labels/   *.nii.gz   (LiTS segmentation-*.nii)
    kidney/
      images/   *.nii.gz   (KiTS case_XXXX/imaging.nii.gz)
      labels/   *.nii.gz   (KiTS case_XXXX/segmentation.nii.gz — label 2=tumor)
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Literal

from monai.data import CacheDataset, Dataset


OrganType = Literal["pancreas", "liver", "kidney"]


def _glob_paired(image_dir: Path, label_dir: Path, ext: str = "*.nii.gz") -> list[dict]:
    images = sorted(image_dir.glob(ext))
    labels = sorted(label_dir.glob(ext))
    if len(images) != len(labels):
        raise ValueError(
            f"Image/label count mismatch in {image_dir}: {len(images)} vs {len(labels)}"
        )
    return [{"image": str(im), "label": str(lb)} for im, lb in zip(images, labels)]


def build_datalist(
    raw_dir: str | Path,
    organ: OrganType,
    labeled_only: bool = False,
    max_labeled: int | None = None,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Returns (train_files, val_files) as lists of dicts with 'image' (and 'label') keys.

    labeled_only=True  → only returns volumes that have a label file (for diffusion training).
    max_labeled        → [PROPOSAL Exp B] limit number of annotated volumes.
    """
    root = Path(raw_dir) / organ
    image_dir = root / "images"
    label_dir = root / "labels"

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Expected image directory at {image_dir}. "
            "See data/ct_dataset.py docstring for expected layout."
        )

    all_files = _glob_paired(image_dir, label_dir) if label_dir.exists() else [
        {"image": str(p)} for p in sorted(image_dir.glob("*.nii.gz"))
    ]

    rng = random.Random(seed)
    rng.shuffle(all_files)

    if max_labeled is not None:
        all_files = all_files[:max_labeled]

    n_val = max(1, int(len(all_files) * val_fraction))
    val_files = all_files[:n_val]
    train_files = all_files[n_val:]

    if labeled_only:
        train_files = [f for f in train_files if "label" in f]
        val_files = [f for f in val_files if "label" in f]

    return train_files, val_files


def build_unlabeled_datalist(
    raw_dir: str | Path,
    organ: OrganType,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    [PROPOSAL Stage 1] Unlabeled CT volumes for autoencoder training.
    Includes all images regardless of whether labels exist.
    """
    root = Path(raw_dir) / organ
    image_dir = root / "images"

    if not image_dir.exists():
        raise FileNotFoundError(f"Expected {image_dir}")

    all_files = [{"image": str(p)} for p in sorted(image_dir.glob("*.nii.gz"))]
    rng = random.Random(seed)
    rng.shuffle(all_files)

    n_val = max(1, int(len(all_files) * val_fraction))
    return all_files[n_val:], all_files[:n_val]


def make_dataset(
    data_files: list[dict],
    transforms,
    cache_rate: float = 0.0,
    num_workers: int = 4,
) -> Dataset:
    """
    Wraps data_files in a MONAI Dataset or CacheDataset. [MONAI]
    cache_rate=0 uses lazy loading (safe for large datasets on limited RAM).
    """
    if cache_rate > 0:
        return CacheDataset(
            data=data_files,
            transform=transforms,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
    return Dataset(data=data_files, transform=transforms)
