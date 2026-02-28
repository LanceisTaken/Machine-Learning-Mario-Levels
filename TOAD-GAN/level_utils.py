"""Level loading, one-hot encoding, Gaussian pyramid, and rendering utilities."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


# ── Compound-token handling (matches MarioTrainer.py) ────────────────────────

COMPOUND_TO_SINGLE = {"pP": "§"}
SINGLE_TO_COMPOUND = {v: k for k, v in COMPOUND_TO_SINGLE.items()}


def _apply_compound(text: str) -> str:
    for multi, single in COMPOUND_TO_SINGLE.items():
        text = text.replace(multi, single)
    return text


def _reverse_compound(text: str) -> str:
    for single, multi in SINGLE_TO_COMPOUND.items():
        text = text.replace(single, multi)
    return text


# ── Loading ──────────────────────────────────────────────────────────────────

def load_level(path: str) -> List[List[str]]:
    """Load a ``.txt`` level file and return a 2-D list of single characters.

    Compound tokens (e.g. ``pP``) are collapsed into a single symbol so every
    grid cell maps to exactly one vocab entry.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    raw = _apply_compound(raw)
    lines = raw.strip().splitlines()

    # Trim trailing blank line produced by some level files
    while lines and lines[-1].strip() == "":
        lines.pop()

    # Ensure rectangular grid (right-pad with sky '-')
    max_w = max(len(ln) for ln in lines)
    grid = [list(ln.ljust(max_w, "-")) for ln in lines]
    return grid


def load_vocab(path: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Return (stoi, itos) from ``vocab.json``."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    stoi: Dict[str, int] = data["stoi"]
    itos: Dict[int, str] = {int(k): v for k, v in data["itos"].items()}
    return stoi, itos


# ── Encoding / decoding ─────────────────────────────────────────────────────

def level_to_one_hot(
    grid: List[List[str]],
    stoi: Dict[str, int],
) -> torch.Tensor:
    """Convert a 2-D character grid to a ``(1, C, H, W)`` one-hot float tensor.

    ``C`` = vocabulary size, ``H`` = rows, ``W`` = columns.
    Unknown characters default to index 0 (newline / padding).
    """
    num_tokens = len(stoi)
    h = len(grid)
    w = len(grid[0]) if grid else 0

    indices = torch.zeros(h, w, dtype=torch.long)
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            indices[r, c] = stoi.get(ch, 0)

    # (H, W) → (1, C, H, W)
    one_hot = F.one_hot(indices, num_classes=num_tokens)  # (H, W, C)
    one_hot = one_hot.permute(2, 0, 1).unsqueeze(0).float()  # (1, C, H, W)
    return one_hot


def one_hot_to_level(
    tensor: torch.Tensor,
    itos: Dict[int, str],
) -> str:
    """Convert a ``(1, C, H, W)`` tensor back to level text.

    Takes the argmax along the channel dimension to pick the most likely token
    per cell.  Compound tokens are restored for the final output.
    """
    # (1, C, H, W) → (H, W)
    indices = tensor.squeeze(0).argmax(dim=0)  # (H, W)
    lines: List[str] = []
    for r in range(indices.shape[0]):
        row_chars = []
        for c in range(indices.shape[1]):
            idx = int(indices[r, c].item())
            row_chars.append(itos.get(idx, "-"))
        lines.append("".join(row_chars))

    text = "\n".join(lines)
    text = _reverse_compound(text)
    return text


# ── Gaussian pyramid ────────────────────────────────────────────────────────

def create_gaussian_pyramid(
    tensor: torch.Tensor,
    num_scales: int,
    scale_factor: float,
) -> List[torch.Tensor]:
    """Build a Gaussian pyramid from finest (index 0) to coarsest (index -1).

    Each successive level is downscaled by ``scale_factor`` using **nearest-
    neighbour** interpolation on the integer tile indices, then re-encoded to
    one-hot.  This preserves discrete token boundaries — bilinear interpolation
    would average the one-hot channels and erase rare tokens.

    The returned list is **reversed** so that index 0 is the *coarsest* scale
    (the training loop starts there).
    """
    num_tokens = tensor.shape[1]
    # Start from integer indices so downscaling stays discrete
    current_indices = tensor.squeeze(0).argmax(dim=0)  # (H, W)

    pyramid: List[torch.Tensor] = [tensor]
    for _ in range(1, num_scales):
        h = max(1, round(current_indices.shape[0] * scale_factor))
        w = max(1, round(current_indices.shape[1] * scale_factor))
        # Nearest-neighbour resize on indices
        idx_4d = current_indices.unsqueeze(0).unsqueeze(0).float()
        idx_down = F.interpolate(idx_4d, size=(h, w), mode="nearest")
        current_indices = idx_down.squeeze(0).squeeze(0).long()
        # Re-encode to one-hot  →  (1, C, H, W)
        oh = F.one_hot(current_indices, num_classes=num_tokens)
        oh = oh.permute(2, 0, 1).unsqueeze(0).float()
        pyramid.append(oh)

    # Reverse: coarsest first
    pyramid.reverse()
    return pyramid


def upsample_to(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Bilinear-upsample ``source`` to match the spatial size of ``target``."""
    return F.interpolate(
        source,
        size=(target.shape[2], target.shape[3]),
        mode="bilinear",
        align_corners=False,
    )


# ── Rendering ────────────────────────────────────────────────────────────────

_COLOR_MAP = {
    "-": (135, 206, 235),
    " ": (135, 206, 235),
    "#": (148, 82, 0),
    "B": (255, 156, 0),
    "?": (255, 216, 0),
    "Q": (255, 216, 0),
    "S": (200, 180, 100),
    "E": (180, 0, 0),
    "o": (255, 216, 0),
    "p": (0, 184, 0),
    "P": (0, 184, 0),
    "§": (0, 184, 0),
    "[": (0, 184, 0),
    "]": (0, 184, 0),
    "c": (120, 120, 120),
    "|": (160, 160, 160),
    "y": (160, 160, 160),
    "e": (220, 20, 60),
    "g": (220, 20, 60),
    "b": (139, 69, 19),
    "M": (220, 20, 60),
    "$": (255, 215, 0),
    "<": (0, 184, 0),
    ">": (0, 184, 0),
}


def render_level_to_png(
    level_text: str,
    out_path: str,
    tile_size: int = 16,
    bg_color: Tuple[int, int, int] = (135, 206, 235),
) -> None:
    """Render level text to a coloured-tile PNG image."""
    lines = level_text.splitlines()
    if not lines:
        raise ValueError("Empty level text")

    h_tiles = len(lines)
    w_tiles = max(len(ln) for ln in lines)

    img = Image.new("RGB", (w_tiles * tile_size, h_tiles * tile_size),
                     color=bg_color)
    draw = ImageDraw.Draw(img)

    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            colour = _COLOR_MAP.get(ch, (60, 60, 60))
            if ch in ("-", " "):
                continue
            x0, y0 = x * tile_size, y * tile_size
            draw.rectangle([x0, y0, x0 + tile_size, y0 + tile_size],
                           fill=colour)

    img.save(out_path)
