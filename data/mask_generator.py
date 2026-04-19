"""
Procedural tumor mask generator. [PROPOSAL Stage 3] + [HU fallback]

Generates synthetic tumor masks with controllable size and blob shape
to be placed inside healthy CT volumes for synthesis.

Design follows Hu et al. (label-free liver tumor) approach as the
fallback/simple baseline, adapted to 3D. [HU]
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, label as nd_label
from typing import Literal


TumorSizeCategory = Literal["small", "medium", "large", "random"]


def _ellipsoid_mask(shape: tuple[int, int, int], radii: tuple[float, float, float]) -> np.ndarray:
    """Binary ellipsoid centered in a volume of given shape."""
    z, y, x = [np.arange(s) - s // 2 for s in shape]
    Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
    return ((Z / radii[0]) ** 2 + (Y / radii[1]) ** 2 + (X / radii[2]) ** 2) <= 1.0


def generate_tumor_mask(
    volume_shape: tuple[int, int, int],
    organ_mask: np.ndarray | None = None,
    size_category: TumorSizeCategory = "random",
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate a single binary tumor mask placed within the volume.

    [PROPOSAL] Masks have controllable size and shape.
    [HU] Blob is created by thresholding smoothed random noise for irregular shape.

    Args:
        volume_shape: (D, H, W) target volume shape.
        organ_mask:   if provided, tumor is constrained to organ region.
        size_category: controls approximate tumor radius.
        seed:         RNG seed for reproducibility.

    Returns:
        Binary mask array of shape volume_shape.
    """
    rng = np.random.default_rng(seed)
    D, H, W = volume_shape

    # [PROPOSAL] size categories by voxel radius at 1.5mm isotropic spacing
    size_map = {
        "small":  (4, 8),    # ~6-12mm diameter
        "medium": (8, 16),   # ~12-24mm
        "large":  (16, 28),  # ~24-42mm
    }
    if size_category == "random":
        size_category = rng.choice(["small", "medium", "large"], p=[0.4, 0.4, 0.2])

    r_min, r_max = size_map[size_category]
    r = rng.uniform(r_min, r_max)

    # Random anisotropic radii (tumors are not perfect spheres) [HU]
    radii = (
        r * rng.uniform(0.7, 1.3),
        r * rng.uniform(0.7, 1.3),
        r * rng.uniform(0.7, 1.3),
    )

    # Place center inside organ mask if provided, else random interior point
    if organ_mask is not None and organ_mask.any():
        pts = np.argwhere(organ_mask)
        center = pts[rng.integers(len(pts))]
    else:
        # Clamp margin so center is always inside volume [ENG]
        margin = min(int(max(radii) * 1.5) + 1, min(D, H, W) // 2 - 1)
        margin = max(margin, 1)
        center = np.array([
            rng.integers(margin, max(margin + 1, D - margin)),
            rng.integers(margin, max(margin + 1, H - margin)),
            rng.integers(margin, max(margin + 1, W - margin)),
        ])

    # Build ellipsoid patch and paste into full volume
    patch_r = int(max(radii) * 1.5) + 2
    patch_shape = (2 * patch_r + 1,) * 3
    blob = _ellipsoid_mask(patch_shape, radii).astype(np.float32)

    # [HU] Smooth random noise + threshold for irregular boundary
    noise = rng.uniform(0.0, 1.0, patch_shape).astype(np.float32)
    noise = gaussian_filter(noise, sigma=patch_r * 0.3)
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    blob = ((blob.astype(float) + 0.6 * noise) > 1.0).astype(np.uint8)

    # Paste blob into full-volume mask
    mask = np.zeros(volume_shape, dtype=np.uint8)
    z0, y0, x0 = [max(0, int(c) - patch_r) for c in center]
    z1, y1, x1 = [min(s, int(c) + patch_r + 1) for c, s in zip(center, volume_shape)]

    bz = slice(z0 - (int(center[0]) - patch_r), patch_shape[0] - ((int(center[0]) + patch_r + 1) - z1))
    by = slice(y0 - (int(center[1]) - patch_r), patch_shape[1] - ((int(center[1]) + patch_r + 1) - y1))
    bx = slice(x0 - (int(center[2]) - patch_r), patch_shape[2] - ((int(center[2]) + patch_r + 1) - x1))

    try:
        mask[z0:z1, y0:y1, x0:x1] = blob[bz, by, bx]
    except ValueError:
        # shape mismatch near boundary — fall back to center ellipsoid [ENG]
        mask[z0:z1, y0:y1, x0:x1] = _ellipsoid_mask(
            (z1 - z0, y1 - y0, x1 - x0), radii
        ).astype(np.uint8)

    if organ_mask is not None:
        mask = mask & organ_mask.astype(np.uint8)

    return mask


def generate_mask_batch(
    volume_shape: tuple[int, int, int],
    n_masks: int = 1,
    organ_mask: np.ndarray | None = None,
    size_category: TumorSizeCategory = "random",
    seed: int | None = None,
) -> list[np.ndarray]:
    """Generate a batch of tumor masks."""
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=n_masks)
    return [
        generate_tumor_mask(volume_shape, organ_mask, size_category, int(s))
        for s in seeds
    ]
