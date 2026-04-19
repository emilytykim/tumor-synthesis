"""
3D VQGAN-style Autoencoder for CT latent space learning. [PROPOSAL Stage 1]
Uses MONAI GenerativeModels AutoencoderKL (3D). [MONAI]

Architecture notes:
- Encoder: 3D conv downsampling stack → mean/logvar heads
- Decoder: 3D conv upsampling stack → image reconstruction
- Loss: L1 reconstruction + KL divergence [LDM]
- Optional: PatchGAN discriminator for adversarial sharpening [PROPOSAL VQGAN ref]
"""
from __future__ import annotations
import torch
import torch.nn as nn
from monai.networks.nets import AutoencoderKL  # [MONAI] 3D-capable


class CTAutoencoder(nn.Module):
    """
    Thin wrapper around MONAI AutoencoderKL.
    Exposes encode() → (z, mu, logvar) and decode() → recon separately
    for use by the diffusion training loop.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 1,
        latent_channels: int = 4,
        num_channels: tuple[int, ...] = (32, 64, 64),
        attention_levels: tuple[bool, ...] = (False, False, True),
        num_res_blocks: int = 2,
        norm_num_groups: int = 16,
    ):
        super().__init__()
        # [MONAI] AutoencoderKL supports spatial_dims=3
        # Note: MONAI uses 'channels' (not 'num_channels') and returns sigma (not logvar)
        self.ae = AutoencoderKL(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            latent_channels=latent_channels,
            channels=num_channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        )
        self.latent_channels = latent_channels

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (z_sample, mu, sigma). z_sample is the reparameterized latent.
        Note: MONAI returns sigma (std), not logvar. kl_loss() accounts for this."""
        z_mu, z_sigma = self.ae.encode(x)
        z = self.ae.sampling(z_mu, z_sigma)
        return z, z_mu, z_sigma

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.ae.decode(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, mu, sigma)."""
        z, mu, sigma = self.encode(x)
        recon = self.decode(z)
        return recon, mu, sigma

    @torch.no_grad()
    def encode_deterministic(self, x: torch.Tensor) -> torch.Tensor:
        """Use mu (no sampling) for stable latent extraction during diffusion training."""
        mu, _ = self.ae.encode(x)
        return mu


def kl_loss(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """VAE KL divergence to unit Gaussian. MONAI returns sigma (std), not logvar. [LDM]"""
    # KL = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
    return 0.5 * torch.mean(sigma.pow(2) + mu.pow(2) - 1.0 - 2.0 * sigma.log())


def build_autoencoder(cfg: dict) -> CTAutoencoder:
    """Construct from config dict."""
    m = cfg["model"]
    return CTAutoencoder(
        spatial_dims=m.get("spatial_dims", 3),
        in_channels=m.get("in_channels", 1),
        out_channels=m.get("out_channels", 1),
        latent_channels=m.get("latent_channels", 4),
        num_channels=tuple(m.get("num_channels", [32, 64, 64])),
        attention_levels=tuple(m.get("attention_levels", [False, False, True])),
        num_res_blocks=m.get("num_res_blocks", 2),
        norm_num_groups=m.get("norm_num_groups", 16),
    )
