"""
Stage 1: Train 3D VQGAN-style autoencoder on unlabeled CT volumes.
[PROPOSAL Stage 1] [MONAI]

Usage:
  python scripts/train_autoencoder.py --config configs/autoencoder.yaml

Source tags:
  [PROPOSAL] = from proposal spec
  [MONAI]    = MONAI Generative tutorial pattern
  [LDM]      = Rombach et al. LDM paper
  [ENG]      = engineering simplification
"""
import argparse
import os
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
from utils.visualization import compare_recon
from data.data_module import get_autoencoder_loaders
from models.autoencoder import build_autoencoder, kl_loss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/autoencoder.yaml")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--smoke-test", action="store_true", help="Run 2 batches then exit [ENG]")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, epoch):
    model.train()
    kl_w = cfg["model"].get("kl_weight", 1e-6)  # [LDM]
    total_loss = 0.0

    for batch in tqdm(loader, desc=f"[AE Train] Epoch {epoch}"):
        x = batch["image"].to(device)          # (B, 1, D, H, W)

        optimizer.zero_grad()
        with autocast(enabled=cfg["training"].get("amp", True)):
            recon, mu, sigma = model(x)
            l1 = F.l1_loss(recon, x)           # [PROPOSAL] L1 reconstruction
            kl = kl_loss(mu, sigma)            # [LDM]
            loss = l1 + kl_w * kl

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
def validate(model, loader, device, cfg, epoch, sample_dir):
    model.eval()
    kl_w = cfg["model"].get("kl_weight", 1e-6)
    total_loss = 0.0

    for i, batch in enumerate(tqdm(loader, desc=f"[AE Val] Epoch {epoch}")):
        x = batch["image"].to(device)
        recon, mu, sigma = model(x)
        l1 = F.l1_loss(recon, x)
        kl = kl_loss(mu, sigma)
        total_loss += (l1 + kl_w * kl).item()

        # Save reconstruction sample every val check [ENG]
        if i == 0:
            orig = x[0, 0].cpu().numpy()
            rec = recon[0, 0].clamp(0, 1).cpu().numpy()
            compare_recon(
                orig, rec,
                save_path=Path(sample_dir) / f"recon_epoch{epoch:04d}.png"
            )

    return total_loss / max(len(loader), 1)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg)

    torch.manual_seed(cfg["training"].get("seed", 42))

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    sample_dir = Path(cfg["output"]["sample_dir"]) / "autoencoder"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    # Data [PROPOSAL Stage 1: unlabeled CT from all organs]
    print("[Stage 1] Building data loaders...")
    train_loader, val_loader = get_autoencoder_loaders(
        raw_dir=cfg["data"]["raw_dir"],
        organs=cfg["data"]["organs"],
        patch_size=cfg["data"]["patch_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = build_autoencoder(cfg).to(device)
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    scaler = GradScaler(enabled=cfg["training"].get("amp", True))

    # LR scheduler [ENG]
    warmup = cfg["training"].get("warmup_epochs", 5)
    max_epochs = cfg["training"]["max_epochs"]
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
        print(f"  Resumed from epoch {ckpt['epoch']}")

    val_interval = cfg["training"].get("val_interval", 10)

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, cfg, epoch)

        if epoch > warmup:
            scheduler.step()

        log = f"Epoch {epoch:04d}/{max_epochs} | train_loss={train_loss:.4f}"

        if epoch % val_interval == 0:
            val_loss = validate(model, val_loader, device, cfg, epoch, sample_dir)
            log += f" | val_loss={val_loss:.4f}"
            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "epoch": epoch, "best_val": best_val},
                    ckpt_dir / "autoencoder_best.pth",
                )
                log += " [saved best]"

        log += f" | {time.time() - t0:.1f}s"
        print(log)

        # Periodic checkpoint [ENG]
        if epoch % 50 == 0:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "epoch": epoch, "best_val": best_val},
                ckpt_dir / f"autoencoder_epoch{epoch:04d}.pth",
            )

        if args.smoke_test and epoch >= 2:
            print("[smoke-test] Exiting after 2 epochs.")
            break

    print(f"Done. Best val loss: {best_val:.4f}")


if __name__ == "__main__":
    main()
