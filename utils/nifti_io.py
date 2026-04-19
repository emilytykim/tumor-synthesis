"""NIfTI load/save helpers. [ENG]"""
import numpy as np
import nibabel as nib
from pathlib import Path


def load_nifti(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (volume_float32, affine). Volume shape: (D, H, W)."""
    img = nib.load(str(path))
    vol = img.get_fdata(dtype=np.float32)
    return vol, img.affine


def save_nifti(volume: np.ndarray, affine: np.ndarray, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(volume.astype(np.float32), affine)
    nib.save(img, str(path))


def window_ct(volume: np.ndarray, win_min: float = -150.0, win_max: float = 250.0) -> np.ndarray:
    """Clip + normalize CT to [0, 1]. [ENG] abdominal soft tissue window."""
    vol = np.clip(volume, win_min, win_max)
    return (vol - win_min) / (win_max - win_min)
