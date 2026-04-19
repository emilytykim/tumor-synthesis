"""
Stage 3 (synthesis step): Generate synthetic (CT volume, tumor mask) pairs.
[PROPOSAL Stage 3] Use trained autoencoder + diffusion model to synthesize
tumors into healthy CT volumes, then save as NIfTI pairs for segmentation training.

Usage:
  python scripts/synthesize_tumors.py \\
      --config configs/diffusion.yaml \\
      --healthy-dir data/raw/liver/images \\
      --organ liver \\
      --n-samples 200 \\
      --out-dir outputs/samples/liver
"""
import argparse
import sys
import random
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import nibabel as nib
from tqdm import trange, tqdm

from utils.config import load_config, get_device
from utils.nifti_io import load_nifti, save_nifti, window_ct
from utils.visualization import show_ct_slices
from data.mask_generator import generate_tumor_mask
from data.transforms import CT_WIN_MIN, CT_WIN_MAX
from models.autoencoder import build_autoencoder
from models.diffusion import build_diffusion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/diffusion.yaml")
    p.add_argument("--healthy-dir", required=True,
                   help="Directory of healthy CT NIfTI files")
    p.add_argument("--organ", default="liver", choices=["liver", "pancreas", "kidney"])
    p.add_argument("--n-samples", type=int, default=50)
    p.add_argument("--out-dir", default="outputs/samples/liver")
    p.add_argument("--inference-steps", type=int, default=50,
                   help="[PROPOSAL challenge 3] fewer steps = cheaper")
    p.add_argument("--size-category", default="random",
                   choices=["small", "medium", "large", "random"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patch-size", type=int, nargs=3, default=[96, 96, 96])
    return p.parse_args()


def load_models(cfg, device):
    ae = build_autoencoder(cfg).to(device)
    ae_ckpt = cfg.get("autoencoder_ckpt", "outputs/checkpoints/autoencoder_best.pth")
    ae.load_state_dict(torch.load(ae_ckpt, map_location=device)["model"])
    ae.eval()

    diff = build_diffusion(cfg).to(device)
    diff_ckpt = str(Path(cfg["output"]["checkpoint_dir"]) / "diffusion_best.pth")
    diff.load_state_dict(torch.load(diff_ckpt, map_location=device)["model"])
    diff.eval()

    return ae, diff


def preprocess_patch(patch: np.ndarray) -> torch.Tensor:
    """CT HU → [0,1] → tensor. [ENG]"""
    windowed = window_ct(patch, CT_WIN_MIN, CT_WIN_MAX)
    return torch.from_numpy(windowed).float().unsqueeze(0).unsqueeze(0)  # (1,1,D,H,W)


def extract_center_patch(volume: np.ndarray, patch_size: list[int]) -> tuple[np.ndarray, tuple]:
    """Extract center crop from volume. [ENG]"""
    D, H, W = volume.shape
    pd, ph, pw = patch_size
    z0 = max(0, (D - pd) // 2)
    y0 = max(0, (H - ph) // 2)
    x0 = max(0, (W - pw) // 2)
    patch = volume[z0:z0+pd, y0:y0+ph, x0:x0+pw]
    # Pad if smaller than patch_size
    pad = [(0, max(0, pd - patch.shape[0])),
           (0, max(0, ph - patch.shape[1])),
           (0, max(0, pw - patch.shape[2]))]
    patch = np.pad(patch, pad, mode="constant", constant_values=CT_WIN_MIN)
    return patch[:pd, :ph, :pw], (z0, y0, x0)


@torch.no_grad()
def synthesize_one(
    ae, diff_model,
    healthy_patch: np.ndarray,
    patch_size: list[int],
    size_category: str,
    num_inference_steps: int,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Full synthesis pipeline for one patch. [PROPOSAL Stage 3]
    1. Generate tumor mask
    2. Build healthy context (zero tumor region)
    3. Encode healthy context → latent
    4. DDIM sample conditioned on mask + context → tumor latent
    5. Decode → synthetic CT patch with tumor
    Returns: (synthetic_volume_patch, tumor_mask_patch) both in original HU range.
    """
    # 1. Generate tumor mask [PROPOSAL]
    tumor_mask = generate_tumor_mask(
        tuple(patch_size), size_category=size_category, seed=seed
    )

    # 2. Preprocess healthy patch
    healthy_t = preprocess_patch(healthy_patch).to(device)  # (1,1,D,H,W) in [0,1]

    # 3. Build healthy context (mask region zeroed) [PROPOSAL Stage 2]
    mask_t = torch.from_numpy(tumor_mask).float().unsqueeze(0).unsqueeze(0).to(device)
    healthy_context = healthy_t * (1.0 - mask_t)

    # 4. Encode healthy context
    z_context = ae.encode_deterministic(healthy_context)   # (1,C,d,h,w)

    # 5. DDIM sampling [LDM / PROPOSAL challenge 3: fewer steps]
    import torch.nn.functional as F
    mask_ds = F.interpolate(mask_t, size=z_context.shape[2:], mode="nearest")
    z_synth = diff_model.sample(
        tumor_mask=mask_ds,
        healthy_context=z_context,
        num_inference_steps=num_inference_steps,
    )

    # 6. Decode to image space
    synth_patch = ae.decode(z_synth)                      # (1,1,D,H,W) in [0,1]
    synth_patch = synth_patch.squeeze().cpu().numpy()
    synth_patch = np.clip(synth_patch, 0, 1)

    # Convert back to HU range [ENG]
    synth_hu = synth_patch * (CT_WIN_MAX - CT_WIN_MIN) + CT_WIN_MIN

    return synth_hu, tumor_mask


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg)
    rng = random.Random(args.seed)

    out_dir = Path(args.out_dir)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    (out_dir / "previews").mkdir(parents=True, exist_ok=True)

    print("[Stage 3 Synthesis] Loading models...")
    ae, diff_model = load_models(cfg, device)

    healthy_paths = sorted(Path(args.healthy_dir).glob("*.nii.gz"))
    if not healthy_paths:
        raise FileNotFoundError(f"No .nii.gz files found in {args.healthy_dir}")
    print(f"  Found {len(healthy_paths)} healthy CT volumes")

    # Track generated files for segmentation training
    synth_file_list = []

    for i in trange(args.n_samples, desc=f"Synthesizing {args.organ} tumors"):
        seed_i = args.seed + i
        # Pick a random healthy volume
        vol_path = rng.choice(healthy_paths)
        volume, affine = load_nifti(vol_path)

        # Extract patch
        patch, origin = extract_center_patch(volume, args.patch_size)

        # Synthesize
        synth_hu, mask = synthesize_one(
            ae, diff_model, patch, args.patch_size,
            args.size_category, args.inference_steps, device, seed_i,
        )

        # Save NIfTI pair [PROPOSAL]
        fname = f"{args.organ}_synth_{i:05d}"
        img_path = out_dir / "images" / f"{fname}.nii.gz"
        lbl_path = out_dir / "labels" / f"{fname}.nii.gz"
        save_nifti(synth_hu, affine, img_path)
        save_nifti(mask.astype(np.float32), affine, lbl_path)
        synth_file_list.append({"image": str(img_path), "label": str(lbl_path)})

        # Save visual preview every 10 samples [ENG]
        if i % 10 == 0:
            norm = np.clip((synth_hu - CT_WIN_MIN) / (CT_WIN_MAX - CT_WIN_MIN), 0, 1)
            show_ct_slices(
                norm, mask.astype(float), title=f"{args.organ} synth {i}",
                save_path=out_dir / "previews" / f"{fname}_preview.png"
            )

    # Save file list JSON for segmentation training [ENG]
    import json
    list_path = out_dir / "synth_filelist.json"
    with open(list_path, "w") as f:
        json.dump(synth_file_list, f, indent=2)
    print(f"\nDone. {args.n_samples} samples saved to {out_dir}")
    print(f"File list: {list_path}")


if __name__ == "__main__":
    main()
