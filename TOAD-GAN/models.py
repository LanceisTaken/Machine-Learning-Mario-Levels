"""TOAD-GAN Generator and PatchGAN Discriminator."""

from typing import Optional

import torch
import torch.nn as nn


# ── helpers ──────────────────────────────────────────────────────────────────

def _same_padding(kernel_size: int) -> int:
    """Return the padding that preserves spatial dimensions."""
    return kernel_size // 2


def _weights_init(m: nn.Module) -> None:
    """Initialize Conv2d and BatchNorm2d weights (Gaussian, 0.02 std)."""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# ── Generator ────────────────────────────────────────────────────────────────

class GeneratorBlock(nn.Module):
    """Fully-convolutional residual generator for one TOAD-GAN scale.

    Architecture (5 conv layers):
        head   : 1×1 Conv2d (input_channels → base_channels)  — merges noise + prev
        body   : 4 × [Conv2d → BatchNorm2d → LeakyReLU(0.2)]
        tail   : Conv2d → Tanh  (base_channels → num_tokens)

    A skip connection adds the (optional) upsampled previous-scale output
    to the generator body output, letting the network learn a *residual*.
    """

    def __init__(
        self,
        num_tokens: int,
        base_channels: int = 32,
        kernel_size: int = 3,
        num_layers: int = 5,
    ) -> None:
        super().__init__()
        pad = _same_padding(kernel_size)

        # Head: merge the two input sources (noise + upsampled prev) into
        # feature space.  At the coarsest scale, prev is all zeros, but the
        # channel count is the same so the head still works.
        layers = [
            nn.Conv2d(num_tokens, base_channels, kernel_size=kernel_size,
                      padding=pad, padding_mode="zeros"),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        # Body
        for _ in range(num_layers - 2):  # -2 because head & tail count
            layers.extend([
                nn.Conv2d(base_channels, base_channels,
                          kernel_size=kernel_size, padding=pad,
                          padding_mode="zeros"),
                nn.BatchNorm2d(base_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        # Tail — project back to token space
        layers.append(
            nn.Conv2d(base_channels, num_tokens, kernel_size=kernel_size,
                      padding=pad, padding_mode="zeros")
        )
        layers.append(nn.Tanh())

        self.body = nn.Sequential(*layers)
        self.apply(_weights_init)

    def forward(
        self,
        noise: torch.Tensor,
        prev_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            noise:       (B, C, H, W)  random noise (scaled by noise_amp)
            prev_output: (B, C, H, W)  upsampled output from previous scale,
                         or ``None`` / zeros for the coarsest scale.
        """
        if prev_output is not None:
            x = noise + prev_output
        else:
            x = noise

        residual = self.body(x)

        # Skip connection: add the upsampled previous output (or zero)
        if prev_output is not None:
            return residual + prev_output
        return residual


# ── Discriminator ────────────────────────────────────────────────────────────

class DiscriminatorBlock(nn.Module):
    """PatchGAN discriminator for one TOAD-GAN scale.

    Outputs a spatial grid of WGAN scores (no sigmoid).

    Architecture (5 conv layers):
        layer 0: Conv2d(num_tokens, base_channels) → LeakyReLU  (no BN)
        layers 1-3: Conv2d → BatchNorm2d → LeakyReLU
        layer 4: Conv2d(base_channels, 1)          (raw score)
    """

    def __init__(
        self,
        num_tokens: int,
        base_channels: int = 32,
        kernel_size: int = 3,
        num_layers: int = 5,
    ) -> None:
        super().__init__()
        pad = _same_padding(kernel_size)

        layers = []

        # First layer — no BatchNorm
        layers.extend([
            nn.Conv2d(num_tokens, base_channels, kernel_size=kernel_size,
                      padding=pad, padding_mode="zeros"),
            nn.LeakyReLU(0.2, inplace=True),
        ])

        # Middle layers — BatchNorm + LeakyReLU
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Conv2d(base_channels, base_channels,
                          kernel_size=kernel_size, padding=pad,
                          padding_mode="zeros"),
                nn.BatchNorm2d(base_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ])

        # Final layer — 1-channel score map, no activation
        layers.append(
            nn.Conv2d(base_channels, 1, kernel_size=kernel_size,
                      padding=pad, padding_mode="zeros")
        )

        self.body = nn.Sequential(*layers)
        self.apply(_weights_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a spatial map of WGAN scores.  Shape: (B, 1, H, W)."""
        return self.body(x)
