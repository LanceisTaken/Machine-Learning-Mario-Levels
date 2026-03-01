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


# ── Pipe post-processing ──────────────────────────────────────────────────

def fix_pipes(
    tile_ids: List[List[int]],
    stoi: Dict[str, int],
    min_height: int = 3,
) -> List[List[int]]:
    """Ensure pipe columns form unbroken vertical segments touching ground.

    Rules applied per column:
    1. Find the topmost ``§`` (pipe) token.
    2. Fill downward with ``§`` until hitting ``#`` (ground).
    3. Extend upward if needed so every pipe is at least *min_height* tiles tall.
    4. If there is no ground below, or if the pipe would overlap non-sky
       tiles when extending upward, remove the pipe entirely.
    """
    PIPE = stoi.get("§")
    GROUND = stoi.get("#")
    SKY = stoi.get("-")
    if PIPE is None or GROUND is None or SKY is None:
        return tile_ids  # vocab missing expected tokens – skip silently

    rows = len(tile_ids)
    cols = len(tile_ids[0]) if rows else 0

    for c in range(cols):
        # Find the bottommost ground row in this column
        ground_row = None
        for r in range(rows - 1, -1, -1):
            if tile_ids[r][c] == GROUND:
                ground_row = r
                break

        # Find the topmost pipe in this column
        top_pipe = None
        for r in range(rows):
            if tile_ids[r][c] == PIPE:
                top_pipe = r
                break

        if top_pipe is None:
            continue  # no pipe in this column

        if ground_row is None or top_pipe >= ground_row:
            # No ground below or pipe is at/below ground – erase it
            for r in range(rows):
                if tile_ids[r][c] == PIPE:
                    tile_ids[r][c] = SKY
            continue

        # Fill from topmost pipe down to (but not including) ground
        for r in range(top_pipe, ground_row):
            tile_ids[r][c] = PIPE

        # Enforce minimum height: extend upward if too short
        pipe_height = ground_row - top_pipe  # current height
        if pipe_height < min_height:
            new_top = ground_row - min_height
            if new_top < 0:
                # Not enough room – remove the pipe
                for r in range(rows):
                    if tile_ids[r][c] == PIPE:
                        tile_ids[r][c] = SKY
                continue
            # Check that extension cells are sky (don't overwrite blocks)
            can_extend = True
            for r in range(new_top, top_pipe):
                if tile_ids[r][c] != SKY:
                    can_extend = False
                    break
            if can_extend:
                for r in range(new_top, top_pipe):
                    tile_ids[r][c] = PIPE
            else:
                # Can't extend safely – remove the pipe
                for r in range(rows):
                    if tile_ids[r][c] == PIPE:
                        tile_ids[r][c] = SKY

    return tile_ids


# ── Lucky-block spacing ───────────────────────────────────────────────────

def fix_lucky_blocks(
    tile_ids: List[List[int]],
    stoi: Dict[str, int],
    min_gap: int = 3,
) -> List[List[int]]:
    """Ensure vertical spacing between ``?`` (lucky) blocks in each column.

    Big Mario is 2 blocks tall and needs room to jump, so there must be at
    least *min_gap* empty rows between any two ``?`` blocks in the same
    column.  When blocks are too close, the **upper** one is replaced with
    sky so the lower (more reachable) block stays.

    The scan proceeds bottom-to-top so the lowest block — the easiest to
    reach from the ground — is always kept.
    """
    LUCKY = stoi.get("?")
    SKY = stoi.get("-")
    if LUCKY is None or SKY is None:
        return tile_ids  # vocab missing expected tokens – skip silently

    rows = len(tile_ids)
    cols = len(tile_ids[0]) if rows else 0

    for c in range(cols):
        # Collect lucky-block row indices bottom-to-top
        lucky_rows = [r for r in range(rows - 1, -1, -1)
                      if tile_ids[r][c] == LUCKY]

        if len(lucky_rows) < 2:
            continue  # 0 or 1 block – nothing to fix

        # Walk bottom-to-top; keep the first (lowest), check gap for rest
        last_kept = lucky_rows[0]
        for r in lucky_rows[1:]:
            gap = last_kept - r - 1  # empty rows between r and last_kept
            if gap >= min_gap:
                last_kept = r  # sufficient room – keep this one too
            else:
                tile_ids[r][c] = SKY  # too close – remove upper block

    return tile_ids


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
        tile_ids = fix_pipes(tile_ids, stoi)
        tile_ids = fix_lucky_blocks(tile_ids, stoi)
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
