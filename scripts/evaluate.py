"""
Evaluation script: Dice, tumor-wise sensitivity (stratified by size), Hausdorff.
[PROPOSAL metrics] Cross-organ transfer matrix + domain shift.

Usage:
  python scripts/evaluate.py \\
      --config configs/segmentation.yaml \\
      --checkpoint outputs/checkpoints/segmenter_liver_best.pth \\
      --test-organ liver \\
      --out results/liver_eval.json
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from tqdm import tqdm

from utils.config import load_config, get_device
from utils.metrics import dice_score, tumor_sensitivity, hausdorff_distance, stratified_sensitivity
from data.data_module import get_segmentation_loaders
from models.segmentation import build_segmenter, get_sliding_window_inferer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/segmentation.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--test-organ", required=True, choices=["liver", "pancreas", "kidney"])
    p.add_argument("--out", default="results/eval.json")
    p.add_argument("--compute-hausdorff", action="store_true",
                   help="[PROPOSAL optional] compute Hausdorff (slow)")
    p.add_argument("--domain-corrupt", default=None,
                   choices=["gaussian_noise", "blur", "contrast", "bias_field"],
                   help="[PROPOSAL Exp C] simulate domain shift")
    return p.parse_args()


def apply_corruption(volume: torch.Tensor, corruption: str) -> torch.Tensor:
    """
    [PROPOSAL Exp C] Simulate domain shift via controlled corruptions.
    Inspired by Hendrycks & Dietterich corruption benchmark [9].
    """
    if corruption == "gaussian_noise":
        return (volume + torch.randn_like(volume) * 0.05).clamp(0, 1)
    if corruption == "blur":
        from scipy.ndimage import gaussian_filter
        v = volume.cpu().numpy()
        v = gaussian_filter(v, sigma=1.5)
        return torch.from_numpy(v).to(volume.device)
    if corruption == "contrast":
        mean = volume.mean()
        return ((volume - mean) * 0.7 + mean).clamp(0, 1)
    if corruption == "bias_field":
        # Simple multiplicative bias field [ENG]
        D, H, W = volume.shape[-3:]
        bias = torch.linspace(0.85, 1.15, W, device=volume.device)
        return (volume * bias).clamp(0, 1)
    return volume


@torch.no_grad()
def run_evaluation(model, val_loader, inferer, device, compute_hd, corruption):
    from monai.transforms import AsDiscrete
    post = AsDiscrete(argmax=True)

    per_case = []
    for batch in tqdm(val_loader, desc="Evaluating"):
        x = batch["image"].to(device)
        if corruption:
            x = apply_corruption(x, corruption)

        y = batch["label"][0, 0].cpu().numpy().astype(np.uint8)  # (D,H,W)

        logits = inferer(x, model)
        pred_tensor = post(logits[0])                             # (1,D,H,W)
        pred = pred_tensor[0].cpu().numpy().astype(np.uint8)      # (D,H,W)

        # Binarize to tumor class (label=1)
        pred_bin = (pred == 1).astype(np.uint8)
        gt_bin = (y == 1).astype(np.uint8)

        metrics = {
            "dice": dice_score(pred_bin, gt_bin),
            "sensitivity": tumor_sensitivity(pred_bin, gt_bin),
            "stratified_sensitivity": stratified_sensitivity(pred_bin, gt_bin),
        }
        if compute_hd:
            metrics["hausdorff"] = hausdorff_distance(pred_bin, gt_bin)

        per_case.append(metrics)

    # Aggregate [PROPOSAL metrics]
    dices = [c["dice"] for c in per_case]
    sens = [c["sensitivity"] for c in per_case if not np.isnan(c["sensitivity"])]
    summary = {
        "n_cases": len(per_case),
        "dice_mean": float(np.mean(dices)),
        "dice_std": float(np.std(dices)),
        "sensitivity_mean": float(np.mean(sens)) if sens else float("nan"),
        "stratified_sensitivity": {
            k: float(np.nanmean([c["stratified_sensitivity"][k] for c in per_case]))
            for k in ["small", "medium", "large"]
        },
        "per_case": per_case,
    }
    if compute_hd:
        hds = [c["hausdorff"] for c in per_case if c["hausdorff"] != float("inf")]
        summary["hausdorff_mean"] = float(np.mean(hds)) if hds else float("inf")

    return summary


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg)

    _, val_loader = get_segmentation_loaders(
        raw_dir=cfg["data"]["raw_dir"],
        organ=args.test_organ,
        synth_files=None,
        patch_size=cfg["data"]["patch_size"],
        batch_size=1,
        num_workers=cfg["data"]["num_workers"],
    )

    model = build_segmenter(cfg).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint from {args.checkpoint}")
    print(f"Evaluating on {args.test_organ} ({len(val_loader)} cases)"
          + (f" with corruption={args.domain_corrupt}" if args.domain_corrupt else ""))

    inferer = get_sliding_window_inferer(cfg["data"]["patch_size"], cfg["data"]["overlap"])
    summary = run_evaluation(
        model, val_loader, inferer, device,
        args.compute_hausdorff, args.domain_corrupt,
    )

    print(f"\nDice: {summary['dice_mean']:.4f} ± {summary['dice_std']:.4f}")
    print(f"Sensitivity: {summary['sensitivity_mean']:.4f}")
    print(f"Stratified sensitivity: {summary['stratified_sensitivity']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out}")


if __name__ == "__main__":
    main()
