"""
Stage 3 (segmentation step): Train 3D U-Net on synthetic (+ optional real) data.
[PROPOSAL Stage 3] [MONAI]

Usage:
  python scripts/train_segmentation.py \\
      --config configs/segmentation.yaml \\
      --organ liver \\
      --synth-dir outputs/samples/liver
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
from torch.cuda.amp import GradScaler, autocast
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from tqdm import tqdm

from utils.config import load_config, get_device
from data.data_module import get_segmentation_loaders
from models.segmentation import build_segmenter, get_loss, get_sliding_window_inferer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/segmentation.yaml")
    p.add_argument("--organ", default="liver", choices=["liver", "pancreas", "kidney"])
    p.add_argument("--synth-dir", default=None, help="Dir with synth_filelist.json")
    p.add_argument("--max-labeled", type=int, default=None, help="[PROPOSAL Exp B]")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--resume", default=None)
    return p.parse_args()


def load_synth_files(synth_dir: str | None) -> list[dict] | None:
    if synth_dir is None:
        return None
    list_path = Path(synth_dir) / "synth_filelist.json"
    if not list_path.exists():
        print(f"  [warn] No synth_filelist.json at {list_path}, skipping synthetic data")
        return None
    with open(list_path) as f:
        return json.load(f)


def train_one_epoch(model, loader, optimizer, scaler, loss_fn, device, cfg, epoch):
    model.train()
    total = 0.0
    for batch in tqdm(loader, desc=f"[Seg Train] Epoch {epoch}"):
        x = batch["image"].to(device)
        y = batch["label"].long().to(device)
        optimizer.zero_grad()
        with autocast(enabled=cfg["training"].get("amp", True)):
            logits = model(x)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"].get("grad_clip", 1.0))
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, inferer, device, cfg, epoch):
    model.eval()
    post_pred = AsDiscrete(argmax=True, to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)
    dice_metric = DiceMetric(include_background=False, reduction="mean")

    for batch in tqdm(loader, desc=f"[Seg Val] Epoch {epoch}"):
        x = batch["image"].to(device)
        y = batch["label"].long().to(device)
        logits = inferer(x, model)
        pred = [post_pred(p) for p in logits]
        lbl = [post_label(l) for l in y]
        dice_metric(y_pred=pred, y=lbl)

    dice = dice_metric.aggregate().item()
    dice_metric.reset()
    return dice


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(cfg)
    torch.manual_seed(cfg["training"].get("seed", 42))

    ckpt_dir = Path(cfg["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    synth_files = load_synth_files(args.synth_dir)

    print(f"[Stage 3] Segmentation training: organ={args.organ}")
    train_loader, val_loader = get_segmentation_loaders(
        raw_dir=cfg["data"]["raw_dir"],
        organ=args.organ,
        synth_files=synth_files,
        patch_size=cfg["data"]["patch_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        hybrid_real_fraction=cfg["training"].get("hybrid_real_fraction", 0.1),
    )
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = build_segmenter(cfg).to(device)
    loss_fn = get_loss()
    inferer = get_sliding_window_inferer(cfg["data"]["patch_size"], cfg["data"]["overlap"])

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr"])
    scaler = GradScaler(enabled=cfg["training"].get("amp", True))
    max_epochs = cfg["training"]["max_epochs"]
    warmup = cfg["training"].get("warmup_epochs", 10)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs - warmup, eta_min=1e-6
    )

    start_epoch = 1
    best_dice = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch = ckpt["epoch"] + 1
        best_dice = ckpt.get("best_dice", 0.0)

    val_interval = cfg["training"].get("val_interval", 10)

    for epoch in range(start_epoch, max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, loss_fn, device, cfg, epoch)
        if epoch > warmup:
            scheduler.step()

        log = f"Epoch {epoch:04d}/{max_epochs} | train_loss={train_loss:.4f}"

        if epoch % val_interval == 0:
            val_dice = validate(model, val_loader, inferer, device, cfg, epoch)
            log += f" | val_dice={val_dice:.4f}"
            if val_dice > best_dice:
                best_dice = val_dice
                torch.save(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "epoch": epoch, "best_dice": best_dice, "organ": args.organ},
                    ckpt_dir / f"segmenter_{args.organ}_best.pth",
                )
                log += " [saved best]"

        log += f" | {time.time() - t0:.1f}s"
        print(log)

        if args.smoke_test and epoch >= 2:
            print("[smoke-test] Exiting.")
            break

    print(f"Done. Best val Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()
