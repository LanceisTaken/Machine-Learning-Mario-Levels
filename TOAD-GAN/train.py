"""Multi-scale TOAD-GAN training loop (WGAN-GP)."""

import argparse
import os
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.autograd as autograd

from config import TOADGANConfig
from models import GeneratorBlock, DiscriminatorBlock
from level_utils import (
    load_level,
    load_vocab,
    level_to_one_hot,
    create_gaussian_pyramid,
    upsample_to,
    one_hot_to_level,
    render_level_to_png,
)


# ── Utilities ────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    device: str,
) -> torch.Tensor:
    """Compute WGAN-GP gradient penalty."""
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    d_interp = discriminator(interpolated)

    grads = autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    grads = grads.view(grads.size(0), -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# ── Fixed noise for reconstruction ───────────────────────────────────────────

def make_fixed_noise(shape: Tuple[int, ...], device: str) -> torch.Tensor:
    """Create a deterministic (zero) noise tensor used for reconstruction."""
    return torch.zeros(shape, device=device)


def make_random_noise(shape: Tuple[int, ...], device: str) -> torch.Tensor:
    """Create a random noise tensor sampled from N(0, 1)."""
    return torch.randn(shape, device=device)


# ── Generate through the trained scales ─────────────────────────────────────

def forward_through_generators(
    generators: List[GeneratorBlock],
    noise_amps: List[float],
    pyramid_shapes: List[Tuple[int, ...]],
    device: str,
    fixed: bool = False,
    fixed_noises: List[torch.Tensor] = None,
) -> torch.Tensor:
    """Run noise through all trained generators in sequence."""
    prev_output = None
    for s, gen in enumerate(generators):
        shape = pyramid_shapes[s]
        if fixed and fixed_noises is not None:
            noise = fixed_noises[s]
        else:
            noise = make_random_noise(shape, device)

        noise = noise * noise_amps[s]

        if prev_output is not None:
            # Upsample to current scale's spatial size
            prev_up = torch.nn.functional.interpolate(
                prev_output, size=(shape[2], shape[3]),
                mode="bilinear", align_corners=False,
            )
        else:
            prev_up = None

        prev_output = gen(noise, prev_up)
    return prev_output


# ── Main training function ──────────────────────────────────────────────────

def train(cfg: TOADGANConfig) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    device = cfg.device

    # Load level and vocab
    grid = load_level(cfg.level_path)
    stoi, itos = load_vocab(cfg.vocab_path)
    num_tokens = len(stoi)

    # One-hot tensor → Gaussian pyramid (coarsest-first)
    real_tensor = level_to_one_hot(grid, stoi).to(device)
    pyramid = create_gaussian_pyramid(real_tensor, cfg.num_scales,
                                      cfg.scale_factor)

    print(f"Level shape : {real_tensor.shape[2]}h x {real_tensor.shape[3]}w")
    print(f"Vocab size  : {num_tokens}")
    print(f"Scales      : {cfg.num_scales}")
    for i, t in enumerate(pyramid):
        print(f"  scale {i}: {t.shape[2]}h x {t.shape[3]}w")
    print()

    generators: List[GeneratorBlock] = []
    noise_amps: List[float] = []
    fixed_noises: List[torch.Tensor] = []
    pyramid_shapes: List[Tuple[int, ...]] = []

    for scale_idx in range(cfg.num_scales):
        real = pyramid[scale_idx]
        cur_shape = real.shape  # (1, C, H, W)
        pyramid_shapes.append(tuple(cur_shape))

        print(f"=== Scale {scale_idx} ({cur_shape[2]}h x {cur_shape[3]}w) ===")

        # Create G and D for this scale
        gen = GeneratorBlock(
            num_tokens=num_tokens,
            base_channels=cfg.base_channels,
            kernel_size=cfg.kernel_size,
            num_layers=cfg.num_conv_layers,
        ).to(device)

        disc = DiscriminatorBlock(
            num_tokens=num_tokens,
            base_channels=cfg.base_channels,
            kernel_size=cfg.kernel_size,
            num_layers=cfg.num_conv_layers,
        ).to(device)

        opt_g = torch.optim.Adam(gen.parameters(), lr=cfg.lr_g,
                                  betas=(cfg.beta1, 0.999))
        opt_d = torch.optim.Adam(disc.parameters(), lr=cfg.lr_d,
                                  betas=(cfg.beta1, 0.999))

        # Fixed noise for reconstruction
        z_fixed = make_fixed_noise(cur_shape, device)
        fixed_noises.append(z_fixed)

        # Determine noise amplitude for this scale
        if scale_idx == 0:
            noise_amp = 1.0
        else:
            # Compute reconstruction through all previous generators
            with torch.no_grad():
                recon = forward_through_generators(
                    generators, noise_amps, pyramid_shapes[:-1],
                    device, fixed=True, fixed_noises=fixed_noises[:-1],
                )
                # Upsample recon to current scale
                recon_up = torch.nn.functional.interpolate(
                    recon, size=(cur_shape[2], cur_shape[3]),
                    mode="bilinear", align_corners=False,
                )
                rmse = torch.sqrt(torch.mean((real - recon_up) ** 2))
                noise_amp = cfg.noise_amp * rmse.item()
                noise_amp = max(noise_amp, 1e-6)

        noise_amps.append(noise_amp)
        print(f"  noise_amp = {noise_amp:.6f}")

        # ── Epoch loop ──────────────────────────────────────────────
        for epoch in range(1, cfg.num_epochs + 1):
            # ── Train Discriminator ──
            for _ in range(cfg.d_steps):
                disc.zero_grad()

                # Real
                d_real = disc(real)
                loss_d_real = -d_real.mean()

                # Fake
                noise = make_random_noise(cur_shape, device) * noise_amp
                if scale_idx > 0:
                    with torch.no_grad():
                        prev = forward_through_generators(
                            generators, noise_amps[:-1],
                            pyramid_shapes[:-1], device,
                        )
                        prev_up = torch.nn.functional.interpolate(
                            prev, size=(cur_shape[2], cur_shape[3]),
                            mode="bilinear", align_corners=False,
                        )
                else:
                    prev_up = None

                fake = gen(noise, prev_up).detach()
                d_fake = disc(fake)
                loss_d_fake = d_fake.mean()

                gp = gradient_penalty(disc, real, fake, device)
                loss_d = loss_d_real + loss_d_fake + cfg.lambda_grad * gp
                loss_d.backward()
                opt_d.step()

            # ── Train Generator ──
            gen.zero_grad()

            noise = make_random_noise(cur_shape, device) * noise_amp
            if scale_idx > 0:
                with torch.no_grad():
                    prev = forward_through_generators(
                        generators, noise_amps[:-1],
                        pyramid_shapes[:-1], device,
                    )
                    prev_up = torch.nn.functional.interpolate(
                        prev, size=(cur_shape[2], cur_shape[3]),
                        mode="bilinear", align_corners=False,
                    )
            else:
                prev_up = None

            fake = gen(noise, prev_up)
            loss_g_adv = -disc(fake).mean()

            # Reconstruction loss
            if scale_idx > 0:
                with torch.no_grad():
                    recon_prev = forward_through_generators(
                        generators, noise_amps[:-1],
                        pyramid_shapes[:-1], device,
                        fixed=True, fixed_noises=fixed_noises[:-1],
                    )
                    recon_prev_up = torch.nn.functional.interpolate(
                        recon_prev, size=(cur_shape[2], cur_shape[3]),
                        mode="bilinear", align_corners=False,
                    )
            else:
                recon_prev_up = None

            recon = gen(z_fixed, recon_prev_up)
            loss_recon = torch.nn.functional.mse_loss(recon, real)

            loss_g = loss_g_adv + cfg.alpha_recon * loss_recon
            loss_g.backward()
            opt_g.step()

            if epoch % max(1, cfg.num_epochs // 10) == 0 or epoch == 1:
                print(f"  epoch {epoch:5d}/{cfg.num_epochs}  "
                      f"D={loss_d.item():+.4f}  "
                      f"G_adv={loss_g_adv.item():+.4f}  "
                      f"recon={loss_recon.item():.4f}")

        # Freeze this scale's generator
        gen.eval()
        for p in gen.parameters():
            p.requires_grad_(False)
        generators.append(gen)

        # Save checkpoint after this scale completes
        scale_ckpt_path = os.path.join(
            cfg.out_dir, f"toadgan_scale_{scale_idx}.pt")
        scale_checkpoint = {
            "num_scales": cfg.num_scales,
            "scales_trained": scale_idx + 1,
            "num_tokens": num_tokens,
            "base_channels": cfg.base_channels,
            "kernel_size": cfg.kernel_size,
            "num_conv_layers": cfg.num_conv_layers,
            "noise_amps": list(noise_amps),
            "pyramid_shapes": list(pyramid_shapes),
            "generators": [g.state_dict() for g in generators],
        }
        torch.save(scale_checkpoint, scale_ckpt_path)
        print(f"  Saved scale {scale_idx} checkpoint -> {scale_ckpt_path}")
        print()

    # ── Save checkpoint ─────────────────────────────────────────────
    save_path = os.path.join(cfg.out_dir, "toadgan_checkpoint.pt")
    checkpoint = {
        "num_scales": cfg.num_scales,
        "num_tokens": num_tokens,
        "base_channels": cfg.base_channels,
        "kernel_size": cfg.kernel_size,
        "num_conv_layers": cfg.num_conv_layers,
        "noise_amps": noise_amps,
        "pyramid_shapes": pyramid_shapes,
        "generators": [g.state_dict() for g in generators],
    }
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint -> {save_path}")

    # Generate a sample level from the trained model
    with torch.no_grad():
        sample = forward_through_generators(
            generators, noise_amps, pyramid_shapes, device,
        )
        level_text = one_hot_to_level(sample.cpu(), itos)

    sample_txt = os.path.join(cfg.out_dir, "sample_level.txt")
    with open(sample_txt, "w", encoding="utf-8") as f:
        f.write(level_text)
    print(f"Saved sample level -> {sample_txt}")

    try:
        sample_png = os.path.join(cfg.out_dir, "sample_level.png")
        render_level_to_png(level_text, sample_png)
        print(f"Saved sample PNG  -> {sample_png}")
    except Exception as exc:
        print(f"PNG render failed: {exc}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train TOAD-GAN on a Mario level")
    parser.add_argument("--level", required=True, help="Path to .txt level file")
    parser.add_argument("--vocab", required=True, help="Path to vocab.json")
    parser.add_argument("--out_dir", default="TOAD-GAN/output",
                        help="Output directory for checkpoints")
    parser.add_argument("--num_scales", type=int, default=3)
    parser.add_argument("--scale_factor", type=float, default=0.75)
    parser.add_argument("--num_epochs", type=int, default=2000)
    parser.add_argument("--lr_g", type=float, default=5e-4)
    parser.add_argument("--lr_d", type=float, default=5e-4)
    parser.add_argument("--lambda_grad", type=float, default=0.1)
    parser.add_argument("--alpha_recon", type=float, default=10.0)
    parser.add_argument("--d_steps", type=int, default=3)
    parser.add_argument("--noise_amp", type=float, default=0.1)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--num_conv_layers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TOADGANConfig(
        level_path=args.level,
        vocab_path=args.vocab,
        out_dir=args.out_dir,
        num_scales=args.num_scales,
        scale_factor=args.scale_factor,
        num_epochs=args.num_epochs,
        lr_g=args.lr_g,
        lr_d=args.lr_d,
        lambda_grad=args.lambda_grad,
        alpha_recon=args.alpha_recon,
        d_steps=args.d_steps,
        noise_amp=args.noise_amp,
        base_channels=args.base_channels,
        kernel_size=args.kernel_size,
        num_conv_layers=args.num_conv_layers,
        seed=args.seed,
    )
    train(cfg)


if __name__ == "__main__":
    main()
