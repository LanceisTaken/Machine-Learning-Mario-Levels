"""Inference script for TOAD-GAN: load trained checkpoints and generate new
Mario levels.

Outputs
-------
- ``.txt``  -- human-readable tile-character grid (same format as training levels)
- ``.json`` -- 2-D array of integer tile IDs, ready for Unity consumption
- ``.png``  -- colour-coded tile visualisation
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from models import GeneratorBlock
from level_utils import load_vocab, one_hot_to_level, render_level_to_png


# ── Checkpoint loading ──────────────────────────────────────────────────────

def load_checkpoint(
    model_dir: str,
    device: str,
    checkpoint_name: str = "toadgan_checkpoint.pt",
) -> Tuple[List[GeneratorBlock], List[float], List[Tuple[int, ...]], int]:
    """Load generators, noise amplitudes, and pyramid shapes from a checkpoint.

    Args:
        model_dir:       Directory that contains the ``.pt`` checkpoint file.
        device:          ``'cuda'`` or ``'cpu'``.
        checkpoint_name: Filename of the checkpoint inside *model_dir*.
                         Use ``toadgan_scale_N.pt`` to load a partial model
                         trained only up to scale *N*.

    Returns:
        generators, noise_amps, pyramid_shapes, num_tokens
    """
    ckpt_path = os.path.join(model_dir, checkpoint_name)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    num_tokens: int = ckpt["num_tokens"]
    base_channels: int = ckpt["base_channels"]
    kernel_size: int = ckpt["kernel_size"]
    num_conv_layers: int = ckpt["num_conv_layers"]
    noise_amps: List[float] = ckpt["noise_amps"]
    pyramid_shapes = [tuple(s) for s in ckpt["pyramid_shapes"]]

    # ``scales_trained`` may differ from ``num_scales`` for partial checkpoints
    scales_trained = ckpt.get("scales_trained", ckpt["num_scales"])

    generators: List[GeneratorBlock] = []
    for s in range(scales_trained):
        gen = GeneratorBlock(
            num_tokens=num_tokens,
            base_channels=base_channels,
            kernel_size=kernel_size,
            num_layers=num_conv_layers,
        ).to(device)
        gen.load_state_dict(ckpt["generators"][s])
        gen.eval()
        generators.append(gen)

    return generators, noise_amps, pyramid_shapes, num_tokens


# ── Multi-scale generation pipeline ────────────────────────────────────────

@torch.inference_mode()
def generate_tensor(
    generators: List[GeneratorBlock],
    noise_amps: List[float],
    pyramid_shapes: List[Tuple[int, ...]],
    device: str,
    temperature: float = 1.0,
    scale_h: float = 1.0,
    scale_w: float = 1.0,
) -> torch.Tensor:
    """Run the full multi-scale generation pipeline and return the raw output
    tensor of shape ``(1, C, H, W)`` (logits over token channels).

    Pipeline
    --------
    For the **coarsest scale** (scale 0):
        ``output_0 = G_0(z_0 * noise_amp_0)``

    For each **finer scale** s > 0:
        ``upsampled = upsample(output_{s-1})``
        ``output_s  = G_s(z_s * noise_amp_s + upsampled)``

    Args:
        temperature:  Multiplier on noise amplitude (>1 = more variation).
        scale_h/w:    Spatial multipliers relative to training-level size.
    """
    prev_output: torch.Tensor | None = None

    for s, gen in enumerate(generators):
        # Original shape at this pyramid scale: (1, C, H, W)
        shape = list(pyramid_shapes[s])

        # Apply optional spatial scaling for variable-size output
        h = max(1, round(shape[2] * scale_h))
        w = max(1, round(shape[3] * scale_w))
        cur_shape = (shape[0], shape[1], h, w)

        # ── Noise input ──
        noise = torch.randn(cur_shape, device=device) * noise_amps[s] * temperature

        # ── Upsampled previous output ──
        if prev_output is not None:
            prev_up = F.interpolate(
                prev_output, size=(h, w), mode="bilinear", align_corners=False,
            )
        else:
            prev_up = None  # coarsest scale: pure noise

        # ── Generator forward pass ──
        # Softmax matches training-time inter-scale probability space
        prev_output = F.softmax(gen(noise, prev_up), dim=1)

    assert prev_output is not None, "No generators were loaded"
    return prev_output


# ── Output conversion ──────────────────────────────────────────────────────

def tensor_to_tile_ids(tensor: torch.Tensor) -> List[List[int]]:
    """Convert a ``(1, C, H, W)`` output tensor to a 2-D grid of integer tile
    IDs by taking the argmax over the channel dimension.

    Returns:
        A row-major list of lists: ``grid[row][col]`` is the tile ID.
    """
    indices = tensor.squeeze(0).argmax(dim=0)  # (H, W)
    return indices.cpu().tolist()


def tile_ids_to_text(
    tile_ids: List[List[int]],
    itos: Dict[int, str],
) -> str:
    """Convert a tile-ID grid back to human-readable level text.

    Compound tokens (e.g. ``pP``) are restored automatically.
    """
    from level_utils import _reverse_compound
    lines = []
    for row in tile_ids:
        lines.append("".join(itos.get(tid, "-") for tid in row))
    return _reverse_compound("\n".join(lines))


def build_unity_json(
    tile_ids: List[List[int]],
    itos: Dict[int, str],
    stoi: Dict[str, int],
) -> dict:
    """Build the Unity-ready JSON payload.

    Structure::

        {
            "height": int,
            "width": int,
            "tile_ids": [[int, ...], ...],      # row-major tile-ID grid
            "tile_map": {"0": "-", "1": "#", ...} # ID -> character lookup
        }
    """
    return {
        "height": len(tile_ids),
        "width": len(tile_ids[0]) if tile_ids else 0,
        "tile_ids": tile_ids,
        "tile_map": {str(k): v for k, v in itos.items()},
    }


# ── Main entry point ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Mario levels from a trained TOAD-GAN checkpoint")

    # Required
    parser.add_argument("--model_dir", required=True,
                        help="Directory containing toadgan_checkpoint.pt")
    parser.add_argument("--vocab", required=True,
                        help="Path to vocab.json")

    # Output paths
    parser.add_argument("--out", default="generated_level.txt",
                        help="Output text file path")
    parser.add_argument("--json_out", default="",
                        help="Output JSON file (default: <out>.json)")
    parser.add_argument("--png_out", default="",
                        help="Output PNG file (default: <out>.png)")

    # Generation controls
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Noise temperature (>1 more variation, <1 closer to training)")
    parser.add_argument("--scale_h", type=float, default=1.0,
                        help="Height multiplier relative to training level")
    parser.add_argument("--scale_w", type=float, default=1.0,
                        help="Width multiplier relative to training level")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of levels to generate")

    # Advanced
    parser.add_argument("--checkpoint", default="toadgan_checkpoint.pt",
                        help="Checkpoint filename inside model_dir")
    parser.add_argument("--no_png", action="store_true",
                        help="Skip PNG rendering")
    parser.add_argument("--no_json", action="store_true",
                        help="Skip JSON output")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    stoi, itos = load_vocab(args.vocab)

    generators, noise_amps, pyramid_shapes, num_tokens = load_checkpoint(
        args.model_dir, device, checkpoint_name=args.checkpoint)

    print(f"Loaded {len(generators)} scale(s) from {args.checkpoint}")
    print(f"Vocab: {num_tokens} tokens  |  Device: {device}")
    print(f"Generating {args.num_samples} sample(s)  "
          f"temp={args.temperature}  "
          f"scale_h={args.scale_h}  scale_w={args.scale_w}")
    print()

    for i in range(args.num_samples):
        # ── Run multi-scale generation pipeline ──
        output_tensor = generate_tensor(
            generators, noise_amps, pyramid_shapes, device,
            temperature=args.temperature,
            scale_h=args.scale_h,
            scale_w=args.scale_w,
        )

        # ── Convert to tile-ID grid ──
        tile_ids = tensor_to_tile_ids(output_tensor)
        level_text = tile_ids_to_text(tile_ids, itos)

        # ── Derive output paths (append index for multi-sample) ──
        def _indexed(path: str, idx: int, total: int) -> str:
            if total <= 1:
                return path
            base, ext = os.path.splitext(path)
            return f"{base}_{idx}{ext}"

        txt_path = _indexed(args.out, i, args.num_samples)
        json_path = _indexed(
            args.json_out or os.path.splitext(args.out)[0] + ".json",
            i, args.num_samples,
        )
        png_path = _indexed(
            args.png_out or os.path.splitext(args.out)[0] + ".png",
            i, args.num_samples,
        )

        # ── Save text ──
        os.makedirs(os.path.dirname(txt_path) or ".", exist_ok=True)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(level_text)

        h, w = len(tile_ids), len(tile_ids[0]) if tile_ids else 0
        print(f"[{i+1}/{args.num_samples}] {h}h x {w}w")
        print(f"  TXT  -> {txt_path}")

        # ── Save JSON (tile-ID grid for Unity) ──
        if not args.no_json:
            unity_payload = build_unity_json(tile_ids, itos, stoi)
            os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(unity_payload, f, ensure_ascii=False)
            print(f"  JSON -> {json_path}")

        # ── Save PNG ──
        if not args.no_png:
            try:
                os.makedirs(os.path.dirname(png_path) or ".", exist_ok=True)
                render_level_to_png(level_text, png_path)
                print(f"  PNG  -> {png_path}")
            except Exception as exc:
                print(f"  PNG render failed: {exc}")

        print()


if __name__ == "__main__":
    main()
