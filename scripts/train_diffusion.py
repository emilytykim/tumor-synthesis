"""
Stage 2: Train conditional latent diffusion model.
[PROPOSAL Stage 2] Conditioned on tumor_mask + healthy_context.
[LDM] Rombach et al. latent diffusion.
[MONAI] DiffusionModelUNet denoising backbone.

Usage:
  python scripts/train_diffusion.py --config configs/diffusion.yaml
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from utils.config import load_config, get_device
from data.data_module import get_diffusion_loaders
from models.autoencoder import build_autoencoder
from models.diffusion import build_diffusion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/diffusion.yaml")
    p.add_argument("--resume", default=None)
    p.add_argument("--max-labeled", type=int, default=None,
                   help="[PROPOSAL Exp B] limit annotated volumes")
    p.add_argument("--smoke-test", action="store_true")
    return p.parse_args()


def load_frozen_autoencoder(cfg, device):
    """Load trained autoencoder and freeze it — only used for encoding. [LDM]"""
    ae = build_autoencoder(cfg).to(device)
    ckpt_path = cfg.get("autoencoder_ckpt", "outputs/checkpoints/autoencoder_best.pth")
    ckpt = torch.load(ckpt_path, map_location=device)
    ae.load_state_dict(ckpt["model"])
    ae.eval()
    for p in ae.parameters():
        p.requires_grad_(False)
    print(f"  Autoencoder loaded from {ckpt_path} (frozen)")
    return ae


def _make_healthy_context(ae, image, label, device):
    """
    [PROPOSAL Stage 2] Healthy context = image with tumor region zeroed out.
    Encode both healthy context and original image into latent space.
    Returns (tumor_latent, mask_latent, healthy_context_latent).
    """
    mask = (label > 0).float()                   # (B, 1, D, H, W) binary tumor mask
    healthy = image * (1.0 - mask)               # zero out tumor region
    with torch.no_grad():
        z_tumor = ae.encode_deterministic(image)       # target latent
        z_context = ae.encode_deterministic(healthy)   # conditioning context
    # Downsample mask to latent resolution [ENG]
    mask_ds = F.interpolate(mask, size=z_tumor.shape[2:], mode="nearest")
    return z_tumor, mask_ds, z_context


def train_one_epoch(model, ae, loader, optimizer, scaler, device, cfg, epoch):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc=f"[LDM Train] Epoch {epoch}"):
        images = batch["image"].to(device)     # (B, 1, D, H, W)
        labels = batch["label"].to(device)     # (B, 1, D, H, W)

        z_tumor, mask_ds, z_context = _make_healthy_context(ae, images, labels, device)

        optimizer.zero_grad()
        with autocast(enabled=cfg["training"].get("amp", True)):
            noise_pred, noise = model(z_tumor, mask_ds, z_context)
            loss = F.mse_loss(noise_pred, noise)   # [LDM] simple MSE noise prediction

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["training"].get("grad_clip", 1.0)
        )
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, ae, loader, device, cfg, epoch):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(loader, desc=f"[LDM Val] Epoch {epoch}"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        z_tumor, mask_ds, z_context = _make_healthy_context(ae, images, labels, device)
        noise_pred, noise = model(z_tumor, mask_ds, z_context)
        total_loss += F.mse_loss(noise_pred, noise).item()
    return total_loss / max(len(loader), 1)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg)
    torch.manual_seed(cfg["training"].get("seed", 42))

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ae = load_frozen_autoencoder(cfg, device)

    # [PROPOSAL Exp B] override max_labeled from CLI
    max_labeled = args.max_labeled or cfg["training"].get("annotation_subsets", [None])[-1]

    print("[Stage 2] Building data loaders...")
    train_loader, val_loader = get_diffusion_loaders(
        raw_dir=cfg["data"]["raw_dir"],
        organs=cfg["data"]["organs"],
        patch_size=cfg["data"]["patch_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        max_labeled=max_labeled,
    )
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = build_diffusion(cfg).to(device)
    print(f"  Diffusion parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    scaler = GradScaler(enabled=cfg["training"].get("amp", True))
    max_epochs = cfg["training"]["max_epochs"]
    warmup = cfg["training"].get("warmup_epochs", 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs - warmup, eta_min=1e-6
    )

    start_epoch = 1
    best_val = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", float("inf"))

    val_interval = cfg["training"].get("val_interval", 25)

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, ae, train_loader, optimizer, scaler, device, cfg, epoch)
        if epoch > warmup:
            scheduler.step()

        log = f"Epoch {epoch:04d}/{max_epochs} | train_loss={train_loss:.6f}"

        if epoch % val_interval == 0:
            val_loss = validate(model, ae, val_loader, device, cfg, epoch)
            log += f" | val_loss={val_loss:.6f}"
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "epoch": epoch, "best_val": best_val},
                    ckpt_dir / "diffusion_best.pth",
                )
                log += " [saved best]"

        log += f" | {time.time() - t0:.1f}s"
        print(log)

        if epoch % 100 == 0:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "best_val": best_val},
                ckpt_dir / f"diffusion_epoch{epoch:04d}.pth",
            )

        if args.smoke_test and epoch >= 2:
            print("[smoke-test] Exiting.")
            break


if __name__ == "__main__":
    main()
