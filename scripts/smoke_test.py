"""
Smoke test — verifies model forward/backward passes without real data.
Run this first to confirm the environment works. [ENG]

Usage:
  python scripts/smoke_test.py
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import numpy as np


def test_autoencoder():
    print("[1/4] Testing CTAutoencoder forward pass...")
    from models.autoencoder import CTAutoencoder, kl_loss
    model = CTAutoencoder(
        num_channels=(16, 32, 32),
        attention_levels=(False, False, True),
        norm_num_groups=8,
    )
    x = torch.randn(1, 1, 32, 32, 32)
    recon, mu, logvar = model(x)
    assert recon.shape == x.shape, f"Shape mismatch: {recon.shape} vs {x.shape}"
    loss = F.l1_loss(recon, x) + 1e-6 * kl_loss(mu, logvar)
    loss.backward()
    print(f"   AE OK | recon shape={recon.shape} | loss={loss.item():.4f}")


def test_diffusion():
    print("[2/4] Testing ConditionalLDM forward pass...")
    from models.diffusion import ConditionalLDM
    model = ConditionalLDM(
        latent_channels=4,
        num_channels=(32, 64, 64),
        attention_levels=(False, True, True),
        norm_num_groups=8,
        num_train_timesteps=100,
    )
    latent = torch.randn(1, 4, 8, 8, 8)
    mask = torch.zeros(1, 1, 8, 8, 8)
    mask[:, :, 2:6, 2:6, 2:6] = 1.0
    ctx = torch.randn(1, 4, 8, 8, 8)
    noise_pred, noise = model(latent, mask, ctx)
    assert noise_pred.shape == latent.shape
    loss = F.mse_loss(noise_pred, noise)
    loss.backward()
    print(f"   LDM OK | noise_pred shape={noise_pred.shape} | loss={loss.item():.6f}")


def test_segmenter():
    print("[3/4] Testing TumorSegmenter forward pass...")
    from models.segmentation import TumorSegmenter, get_loss
    model = TumorSegmenter(channels=(8, 16, 32, 64, 64), strides=(2, 2, 2, 2))
    x = torch.randn(1, 1, 32, 32, 32)
    y = torch.zeros(1, 1, 32, 32, 32, dtype=torch.long)
    y[:, :, 10:20, 10:20, 10:20] = 1
    logits = model(x)
    assert logits.shape[1] == 2
    loss = get_loss()(logits, y)
    loss.backward()
    print(f"   Seg OK | logits shape={logits.shape} | loss={loss.item():.4f}")


def test_mask_generator():
    print("[4/4] Testing mask generator...")
    from data.mask_generator import generate_tumor_mask, generate_mask_batch
    for cat in ["small", "medium", "large", "random"]:
        mask = generate_tumor_mask((32, 32, 32), size_category=cat, seed=0)
        assert mask.shape == (32, 32, 32)
        assert mask.sum() > 0, f"Empty mask for category={cat}"
    masks = generate_mask_batch((32, 32, 32), n_masks=3, seed=1)
    assert len(masks) == 3
    print(f"   MaskGen OK | small voxels={generate_tumor_mask((64,64,64), size_category='small', seed=0).sum()}")


def test_metrics():
    print("[BONUS] Testing metrics...")
    from utils.metrics import dice_score, tumor_sensitivity, stratified_sensitivity
    pred = np.zeros((32, 32, 32), dtype=np.uint8)
    gt   = np.zeros((32, 32, 32), dtype=np.uint8)
    pred[5:15, 5:15, 5:15] = 1
    gt[5:15, 5:15, 5:15] = 1
    d = dice_score(pred, gt)
    assert abs(d - 1.0) < 1e-4, f"Expected Dice=1, got {d}"
    s = tumor_sensitivity(pred, gt)
    assert s == 1.0
    strat = stratified_sensitivity(pred, gt)
    print(f"   Metrics OK | dice={d:.4f} sensitivity={s:.4f} strat={strat}")


if __name__ == "__main__":
    test_mask_generator()
    test_autoencoder()
    test_diffusion()
    test_segmenter()
    test_metrics()
    print("\nAll smoke tests passed.")
