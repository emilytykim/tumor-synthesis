"""Slice-grid visualization for CT volumes and masks. [ENG]"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def show_ct_slices(
    volume: np.ndarray,
    mask: np.ndarray | None = None,
    n_slices: int = 6,
    title: str = "",
    save_path: str | Path | None = None,
):
    """
    Display n_slices evenly spaced axial slices. Overlays mask in red if provided.
    volume shape: (D, H, W) normalized to [0, 1].
    """
    D = volume.shape[0]
    indices = np.linspace(0, D - 1, n_slices, dtype=int)

    fig, axes = plt.subplots(1, n_slices, figsize=(3 * n_slices, 3))
    if n_slices == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        ax.imshow(volume[idx], cmap="gray", vmin=0, vmax=1)
        if mask is not None:
            masked = np.ma.masked_where(mask[idx] == 0, mask[idx])
            ax.imshow(masked, cmap="Reds", alpha=0.4, vmin=0, vmax=1)
        ax.set_title(f"z={idx}")
        ax.axis("off")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def compare_recon(
    original: np.ndarray,
    reconstructed: np.ndarray,
    n_slices: int = 4,
    save_path: str | Path | None = None,
):
    """Side-by-side original vs autoencoder reconstruction. [ENG] used for Stage 1 QC."""
    D = original.shape[0]
    indices = np.linspace(0, D - 1, n_slices, dtype=int)
    fig, axes = plt.subplots(2, n_slices, figsize=(3 * n_slices, 6))
    for col, idx in enumerate(indices):
        axes[0, col].imshow(original[idx], cmap="gray", vmin=0, vmax=1)
        axes[0, col].set_title(f"orig z={idx}")
        axes[0, col].axis("off")
        axes[1, col].imshow(reconstructed[idx], cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"recon z={idx}")
        axes[1, col].axis("off")
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
