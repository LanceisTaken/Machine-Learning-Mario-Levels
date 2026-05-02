"""TOAD-GAN configuration dataclass."""

from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class TOADGANConfig:
    """Centralised hyperparameters for TOAD-GAN training and generation."""

    # ── Pyramid ──────────────────────────────────────────────────────
    num_scales: int = 5
    scale_factor: float = 0.67

    # ── Noise ────────────────────────────────────────────────────────
    noise_amp: float = 0.15

    # ── Training ─────────────────────────────────────────────────────
    num_epochs: int = 4000
    lr_g: float = 5e-4
    lr_d: float = 5e-4
    beta1: float = 0.5
    lambda_grad: float = 0.1       # gradient-penalty coefficient (WGAN-GP)
    alpha_recon: float = 10.0      # reconstruction-loss weight
    d_steps: int = 3               # discriminator updates per generator update

    # ── Architecture ─────────────────────────────────────────────────
    num_conv_layers: int = 5
    base_channels: int = 64
    kernel_size: int = 5

    # ── I/O ──────────────────────────────────────────────────────────
    level_path: str = ""
    vocab_path: str = ""
    out_dir: str = "TOAD-GAN/output"

    # ── Device ───────────────────────────────────────────────────────
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
