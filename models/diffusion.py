"""
Conditional Latent Diffusion Model for 3D tumor synthesis. [PROPOSAL Stage 2]
[LDM] Rombach et al. latent diffusion adapted to 3D medical volumes.
[MONAI] Uses DiffusionModelUNet denoising backbone + DDPMScheduler.

Conditioning strategy [PROPOSAL]:
  Input to denoising UNet = concat(noisy_latent, tumor_mask_latent, healthy_context_latent)
  - tumor_mask_latent: downsampled binary tumor mask → tells model WHERE to synthesize
  - healthy_context_latent: encoded healthy CT patch → tells model WHAT surrounds the tumor
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from generative.networks.nets import DiffusionModelUNet   # [MONAI generative]
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler


class ConditionalLDM(nn.Module):
    """
    Latent diffusion model conditioned on tumor mask + healthy CT context.

    in_channels to the UNet = latent_channels + 1 (mask) + latent_channels (context)
    [PROPOSAL Stage 2]: concat conditioning channels to noisy latent before UNet.
    """

    def __init__(
        self,
        spatial_dims: int = 3,
        latent_channels: int = 4,
        num_channels: tuple[int, ...] = (64, 128, 128),
        attention_levels: tuple[bool, ...] = (False, True, True),
        num_res_blocks: int = 2,
        num_head_channels: int = 32,
        norm_num_groups: int = 16,
        num_train_timesteps: int = 1000,
        beta_schedule: str = "scaled_linear",
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
    ):
        super().__init__()
        # [PROPOSAL] mask channel (1) + healthy context (latent_channels)
        cond_channels = 1 + latent_channels
        unet_in = latent_channels + cond_channels

        # [MONAI] 3D denoising UNet
        self.unet = DiffusionModelUNet(
            spatial_dims=spatial_dims,
            in_channels=unet_in,
            out_channels=latent_channels,
            num_channels=num_channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            num_head_channels=num_head_channels,
            norm_num_groups=norm_num_groups,
            resblock_updown=True,
            with_conditioning=False,  # we use channel concat, not cross-attention [ENG]
        )

        # [LDM] DDPM for training, DDIM for fast inference
        # MONAI generative schedule names: linear_beta, scaled_linear_beta, cosine [MONAI]
        _sched_map = {
            "linear": "linear_beta",
            "linear_beta": "linear_beta",
            "scaled_linear": "scaled_linear_beta",
            "scaled_linear_beta": "scaled_linear_beta",
            "cosine": "cosine",
        }
        _sched_name = _sched_map.get(beta_schedule, "linear_beta")
        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            schedule=_sched_name,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        self.infer_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            schedule=_sched_name,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        self.latent_channels = latent_channels
        self.num_train_timesteps = num_train_timesteps

    def _prepare_input(
        self,
        noisy_latent: torch.Tensor,
        tumor_mask: torch.Tensor,
        healthy_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate conditioning channels to noisy latent along channel dim.
        [PROPOSAL Stage 2] mask + healthy_context conditioning.

        noisy_latent:    (B, C, D, H, W)
        tumor_mask:      (B, 1, D, H, W) — at latent spatial resolution
        healthy_context: (B, C, D, H, W) — encoded healthy CT latent
        """
        # Resize mask to match latent spatial dims if needed [ENG]
        if tumor_mask.shape[2:] != noisy_latent.shape[2:]:
            tumor_mask = F.interpolate(
                tumor_mask.float(), size=noisy_latent.shape[2:], mode="nearest"
            )
        return torch.cat([noisy_latent, tumor_mask, healthy_context], dim=1)

    def forward(
        self,
        latent: torch.Tensor,
        tumor_mask: torch.Tensor,
        healthy_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training forward: add noise at random timestep, predict noise.
        Returns noise prediction for loss computation.
        """
        B = latent.shape[0]
        device = latent.device

        # Sample random timesteps [LDM]
        timesteps = torch.randint(
            0, self.num_train_timesteps, (B,), device=device, dtype=torch.long
        )

        # Add noise according to DDPM forward process [LDM]
        noise = torch.randn_like(latent)
        noisy_latent = self.train_scheduler.add_noise(latent, noise, timesteps)

        # Concatenate conditioning [PROPOSAL]
        model_input = self._prepare_input(noisy_latent, tumor_mask, healthy_context)

        # Predict noise [LDM]
        noise_pred = self.unet(model_input, timesteps)
        return noise_pred, noise

    @torch.no_grad()
    def sample(
        self,
        tumor_mask: torch.Tensor,
        healthy_context: torch.Tensor,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        [PROPOSAL Stage 3] Ancestral DDIM sampling to generate tumor latent.
        Returns denoised latent z same shape as healthy_context.
        """
        B, C, *spatial = healthy_context.shape
        device = healthy_context.device

        # Start from pure noise [LDM]
        z = torch.randn(B, self.latent_channels, *spatial, device=device)

        self.infer_scheduler.set_timesteps(num_inference_steps)

        for t in self.infer_scheduler.timesteps:
            t_batch = torch.full((B,), int(t), device=device, dtype=torch.long)
            model_input = self._prepare_input(z, tumor_mask, healthy_context)
            noise_pred = self.unet(model_input, t_batch)
            # MONAI DDIM step returns (prev_sample, pred_original) tuple [MONAI]
            z, _ = self.infer_scheduler.step(noise_pred, int(t), z)

        return z


def build_diffusion(cfg: dict) -> ConditionalLDM:
    m = cfg["model"]
    s = cfg["scheduler"]
    return ConditionalLDM(
        spatial_dims=m.get("spatial_dims", 3),
        latent_channels=cfg["model"].get("out_channels", 4),
        num_channels=tuple(m.get("num_channels", [64, 128, 128])),
        attention_levels=tuple(m.get("attention_levels", [False, True, True])),
        num_res_blocks=m.get("num_res_blocks", 2),
        num_head_channels=m.get("num_head_channels", 32),
        norm_num_groups=m.get("norm_num_groups", 16),
        num_train_timesteps=s.get("num_train_timesteps", 1000),
        beta_schedule=s.get("beta_schedule", "scaled_linear"),
        beta_start=s.get("beta_start", 0.00085),
        beta_end=s.get("beta_end", 0.012),
    )
