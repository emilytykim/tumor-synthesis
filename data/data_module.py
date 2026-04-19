"""DataLoader factory for all three pipeline stages. [ENG]"""
from __future__ import annotations
from torch.utils.data import DataLoader
from monai.data import pad_list_data_collate  # [DiffTumor] pads variable-size CT volumes

from data.ct_dataset import (
    build_datalist,
    build_unlabeled_datalist,
    make_dataset,
    OrganType,
)
from data.transforms import (
    get_autoencoder_transforms,
    get_diffusion_transforms,
    get_segmentation_transforms,
)


def get_autoencoder_loaders(
    raw_dir: str,
    organs: list[OrganType],
    patch_size: list[int],
    batch_size: int = 1,
    num_workers: int = 4,
    cache_rate: float = 0.0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """[PROPOSAL Stage 1] Unlabeled CT from all organs combined."""
    train_files, val_files = [], []
    for organ in organs:
        tr, vl = build_unlabeled_datalist(raw_dir, organ, seed=seed)
        train_files.extend(tr)
        val_files.extend(vl)

    train_ds = make_dataset(
        train_files,
        get_autoencoder_transforms(patch_size, is_train=True),
        cache_rate=cache_rate,
        num_workers=num_workers,
    )
    val_ds = make_dataset(
        val_files,
        get_autoencoder_transforms(patch_size, is_train=False),
        cache_rate=cache_rate,
        num_workers=num_workers,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=pad_list_data_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=pad_list_data_collate,
    )
    return train_loader, val_loader


def get_diffusion_loaders(
    raw_dir: str,
    organs: list[OrganType],
    patch_size: list[int],
    batch_size: int = 1,
    num_workers: int = 4,
    max_labeled: int | None = None,  # [PROPOSAL Exp B] low-annotation
    num_samples_per_volume: int = 2,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """[PROPOSAL Stage 2] Labeled CT with tumor masks."""
    train_files, val_files = [], []
    for organ in organs:
        tr, vl = build_datalist(
            raw_dir, organ,
            labeled_only=True,
            max_labeled=max_labeled,
            seed=seed,
        )
        train_files.extend(tr)
        val_files.extend(vl)

    train_ds = make_dataset(
        train_files,
        get_diffusion_transforms(patch_size, is_train=True, num_samples=num_samples_per_volume),
        cache_rate=0.0,
        num_workers=num_workers,
    )
    val_ds = make_dataset(
        val_files,
        get_diffusion_transforms(patch_size, is_train=False, num_samples=1),
        cache_rate=0.0,
        num_workers=num_workers,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=pad_list_data_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=pad_list_data_collate,
    )
    return train_loader, val_loader


def get_segmentation_loaders(
    raw_dir: str,
    organ: OrganType,
    synth_files: list[dict] | None,
    patch_size: list[int],
    batch_size: int = 2,
    num_workers: int = 4,
    hybrid_real_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """
    [PROPOSAL Stage 3] Mix synthetic + optional real data.
    synth_files: list of {'image': path, 'label': path} for synthetic pairs.
    hybrid_real_fraction: proportion of real data to mix in (0 = synthetic only).
    """
    train_real, val_real = build_datalist(raw_dir, organ, seed=seed)

    # Mix synthetic + real [PROPOSAL]
    train_files = []
    if synth_files:
        train_files.extend(synth_files)
    if hybrid_real_fraction > 0 and train_real:
        n_real = max(1, int(len(synth_files or []) * hybrid_real_fraction))
        train_files.extend(train_real[:n_real])

    transforms_train = get_segmentation_transforms(patch_size, is_train=True)
    transforms_val = get_segmentation_transforms(patch_size, is_train=False)

    train_ds = make_dataset(train_files, transforms_train)
    val_ds = make_dataset(val_real, transforms_val)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=pad_list_data_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=pad_list_data_collate,
    )
    return train_loader, val_loader
