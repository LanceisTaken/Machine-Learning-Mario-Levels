import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image, ImageDraw

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_levels(levels_dir: str) -> str:
    paths = sorted(Path(levels_dir).glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No .txt files found in {levels_dir}")
    texts: List[str] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            texts.append(f.read().strip())
    # Join levels with a separator newline block to avoid bleeding
    return "\n\n".join(texts) + "\n"


# Compound token handling: treat 'pP' as a single token internally
COMPOUND_TO_SINGLE: Dict[str, str] = {
    "pP": "§",  # use a rare symbol as internal single-token placeholder
}
SINGLE_TO_COMPOUND: Dict[str, str] = {v: k for k, v in COMPOUND_TO_SINGLE.items()}


def apply_compound_tokens(text: str) -> str:
    for multi, single in COMPOUND_TO_SINGLE.items():
        text = text.replace(multi, single)
    return text


def reverse_compound_tokens(text: str) -> str:
    for single, multi in SINGLE_TO_COMPOUND.items():
        text = text.replace(single, multi)
    return text


def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    # Apply compound tokens before building vocab so they are single symbols
    text = apply_compound_tokens(text)
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> torch.Tensor:
    text = apply_compound_tokens(text)
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def decode(tokens: List[int], itos: Dict[int, str]) -> str:
    text = "".join(itos[i] for i in tokens)
    return reverse_compound_tokens(text)


class CharSequenceDataset(Dataset):
    def __init__(self, data: torch.Tensor, sequence_length: int):
        super().__init__()
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return max(0, self.data.size(0) - self.sequence_length)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.sequence_length + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class CharLSTM(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.proj(out)
        return logits, hidden


@dataclass
class TrainConfig:
    levels_dir: str
    output_dir: str
    seq_len: int = 128
    batch_size: int = 64
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    lr: float = 3e-3
    max_epochs: int = 20
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    text = read_levels(cfg.levels_dir)
    stoi, itos = build_vocab(text)
    encoded = encode(text, stoi)

    n_total = encoded.size(0)
    n_train = int(n_total * 0.95)
    train_data = encoded[:n_train]
    val_data = encoded[n_train:]

    train_ds = CharSequenceDataset(train_data, cfg.seq_len)
    val_ds = CharSequenceDataset(val_data, cfg.seq_len)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = CharLSTM(vocab_size=len(stoi), embedding_dim=cfg.embedding_dim, hidden_dim=cfg.hidden_dim, num_layers=cfg.num_layers, dropout=cfg.dropout).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val = math.inf
    best_path = os.path.join(cfg.output_dir, "mario_lstm.pt")

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / max(1, len(train_loader))

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for x, y in val_loader:
                x = x.to(cfg.device)
                y = y.to(cfg.device)
                logits, _ = model(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                val_loss += loss.item()
            avg_val = val_loss / max(1, len(val_loader))

        print(f"Epoch {epoch:02d}: train_loss={avg_train:.4f} val_loss={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            torch.save({
                "model_state": model.state_dict(),
                "config": {
                    "vocab_size": len(stoi),
                    "embedding_dim": cfg.embedding_dim,
                    "hidden_dim": cfg.hidden_dim,
                    "num_layers": cfg.num_layers,
                    "dropout": cfg.dropout,
                },
            }, best_path)

    with open(os.path.join(cfg.output_dir, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump({"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}, f, ensure_ascii=False, indent=2)


@torch.inference_mode()
def generate(
    model_path: str,
    vocab_path: str,
    length: int = 1000,
    temperature: float = 1.0,
    seed_text: str = "",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    top_k: int = 0,
    top_p: float = 0.0,
    hash_bias: float = 0.0,
    banned_tokens: List[str] = None,
) -> str:
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    stoi: Dict[str, int] = vocab["stoi"]
    itos: Dict[int, str] = {int(k): v for k, v in vocab["itos"].items()}

    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt["config"]
    model = CharLSTM(
        vocab_size=len(stoi),
        embedding_dim=cfg["embedding_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if seed_text:
        context_tokens = [stoi.get(c, 0) for c in seed_text]
    else:
        # start with a newline as neutral token if exists
        start_char = "\n" if "\n" in stoi else next(iter(stoi.keys()))
        context_tokens = [stoi[start_char]]

    input_ids = torch.tensor([context_tokens], dtype=torch.long, device=device)
    hidden = None

    # Feed the seed text to warm up hidden state
    logits, hidden = model(input_ids, hidden)
    next_id = input_ids[0, -1]
    generated: List[int] = list(context_tokens)

    # Prepare banned indices once
    banned_indices: List[int] = []
    if banned_tokens:
        for tok in banned_tokens:
            if tok in stoi:
                banned_indices.append(stoi[tok])

    for _ in range(max(0, length)):
        inp = next_id.view(1, 1)
        logits, hidden = model(inp, hidden)
        logits = logits[:, -1, :] / max(1e-6, temperature)

        # Optional: encourage ground token '#'
        if hash_bias != 0.0:
            if "#" in stoi:
                logits[:, stoi["#"]] = logits[:, stoi["#"]] + hash_bias

        # Sampling filters
        if top_k and top_k > 0:
            topk_vals, topk_idx = torch.topk(logits, k=min(top_k, logits.size(-1)), dim=-1)
            filt_logits = torch.full_like(logits, float("-inf"))
            filt_logits.scatter_(1, topk_idx, topk_vals)
            logits = filt_logits

        if top_p and top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            cutoff = (cumulative_probs > top_p).float()
            # keep first token above cutoff as well
            cutoff = torch.roll(cutoff, shifts=1, dims=-1)
            cutoff[..., 0] = 0
            sorted_logits[cutoff.bool()] = float("-inf")
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(1, sorted_indices, sorted_logits)

        # Apply hard bans by setting logits to -inf
        if banned_indices:
            logits[:, banned_indices] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).squeeze(0).squeeze(0)
        generated.append(int(next_id.item()))

    return decode(generated, itos)


def render_level_text_to_png(
    level_text: str,
    out_path: str,
    tile_size: int = 16,
    bg_color: Tuple[int, int, int] = (135, 206, 235),
) -> None:
    # Split into lines and remove any empty trailing lines
    lines = [line for line in level_text.splitlines()]
    if not lines:
        raise ValueError("Empty level text")

    height_tiles = len(lines)
    width_tiles = max(len(line) for line in lines)

    img = Image.new("RGB", (width_tiles * tile_size, height_tiles * tile_size), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Basic color mapping for common Mario symbols
    color_map: Dict[str, Tuple[int, int, int]] = {
        "-": (135, 206, 235),  # sky
        " ": (135, 206, 235),  # treat space as sky
        "#": (148, 82, 0),     # ground/blocks brown
        "B": (255, 156, 0),    # brick orange
        "?": (255, 216, 0),    # ? block yellow
        "*": (255, 216, 0),
        "+": (255, 216, 0),
        "O": (200, 140, 60),   # coin brick
        "o": (255, 216, 0),    # coin
        "P": (0, 184, 0),      # pipe
        "p": (0, 184, 0),
        "[": (0, 184, 0),
        "]": (0, 184, 0),
        "C": (120, 120, 120),  # cannon
        "c": (120, 120, 120),
        "|": (160, 160, 160),  # pole
        "y": (160, 160, 160),  # spring
        "Y": (200, 200, 200),
        "e": (220, 20, 60),    # enemies
        "g": (220, 20, 60),
        "k": (220, 20, 60),
        "K": (220, 20, 60),
        "t": (220, 20, 60),
        "l": (220, 20, 60),
        "V": (220, 20, 60),
        "h": (220, 20, 60),
    }

    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            color = color_map.get(ch, (60, 60, 60))
            if ch in ("-", " "):
                # Skip drawing sky; background already sky blue
                continue
            x0 = x * tile_size
            y0 = y * tile_size
            x1 = x0 + tile_size
            y1 = y0 + tile_size
            draw.rectangle([x0, y0, x1, y1], fill=color)

    img.save(out_path)


def postprocess_level_text(
    text: str,
    wrap_width: int = None,
    target_height: int = None,
    enforce_ground: bool = False,
    ground_char: str = "#",
    pad_char: str = "-",
) -> str:
    # Normalize newlines
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in text.split("\n") if ln is not None]
    if wrap_width is not None and wrap_width > 0:
        wrapped: List[str] = []
        for ln in lines if lines else [text]:
            if ln == "":
                wrapped.append("")
                continue
            start = 0
            while start < len(ln):
                wrapped.append(ln[start:start + wrap_width])
                start += wrap_width
        lines = wrapped

    if target_height is not None and target_height > 0:
        # Pad or trim to target height
        if len(lines) < target_height:
            width = max((len(l) for l in lines), default=wrap_width or 0)
            width = max(width, wrap_width or 0)
            while len(lines) < target_height:
                lines.append(pad_char * width)
        elif len(lines) > target_height:
            lines = lines[:target_height]

    # Normalize widths by right-padding with pad_char
    max_width = max((len(l) for l in lines), default=0)
    if max_width > 0:
        lines = [l + (pad_char * (max_width - len(l))) for l in lines]

    if enforce_ground and lines:
        # Set last row to ground_char
        lines[-1] = ground_char * len(lines[-1])

    return "\n".join(lines)


def enforce_generation_rules(
    text: str,
    ground_char: str = "#",
    sky_chars: List[str] = None,
    pipe_chars: List[str] = None,
    question_chars: List[str] = None,
    cannon_chars: List[str] = None,
) -> str:
    # Rule set focuses on pipes: ensure vertical stacking and ground support
    if sky_chars is None:
        sky_chars = ["-", " "]
    if pipe_chars is None:
        pipe_chars = ["p", "P"]
    if question_chars is None:
        question_chars = ["?"]
    if cannon_chars is None:
        cannon_chars = ["c"]

    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines or (len(lines) == 1 and lines[0] == ""):
        return text

    h = len(lines)
    w = max(len(l) for l in lines)
    # Pad to rectangle
    grid = [list(l + ("-" * (w - len(l)))) for l in lines]

    # Ensure bottom ground if any pipe reaches near bottom but lacks support
    # Pass 1: for any pipe tile with sky beneath, fill downwards with 'P' until ground or bottom-1; set bottom row ground
    for y in range(h - 1):
        for x in range(w):
            ch = grid[y][x]
            if ch in pipe_chars:
                yb = y + 1
                # fill sky beneath with vertical pipe body 'P'
                while yb < h and grid[yb][x] in sky_chars:
                    grid[yb][x] = "P"
                    yb += 1
                # if we reached bottom, put ground at bottom row
                if yb >= h:
                    grid[h - 1][x] = ground_char

    # Pass 2: if any pipe is floating (no ground directly below its bottom-most pipe), add ground under it (one tile thick)
    for x in range(w):
        # scan from bottom up to find bottom-most pipe in this column
        bottom_pipe_y = None
        for y in range(h - 1, -1, -1):
            if grid[y][x] in pipe_chars:
                bottom_pipe_y = y
                break
        if bottom_pipe_y is not None and bottom_pipe_y < h - 1:
            if grid[bottom_pipe_y + 1][x] not in pipe_chars and grid[bottom_pipe_y + 1][x] != ground_char:
                grid[bottom_pipe_y + 1][x] = ground_char

    # Pass 3: ensure vertical continuity: if a pipe cell has a non-pipe directly above within a pipe column, convert to pipe
    for x in range(w):
        # If column has any pipe, make contiguous from top-most pipe down to bottom-most pipe
        ys = [y for y in range(h) if grid[y][x] in pipe_chars]
        if ys:
            top_y = min(ys)
            bot_y = max(ys)
            for y in range(top_y, bot_y + 1):
                if grid[y][x] in sky_chars:
                    grid[y][x] = "P"

    # Pass 4: question blocks must have sky beneath (player can hit from below)
    for y in range(h - 1):
        for x in range(w):
            if grid[y][x] in question_chars:
                below = grid[y + 1][x]
                if below not in sky_chars:
                    # Prefer to move question up if there is sky above
                    if y > 0 and grid[y - 1][x] in sky_chars:
                        grid[y - 1][x] = grid[y][x]
                        grid[y][x] = "-"
                    else:
                        # As a fallback, clear the block below to sky unless it's ground under bottom pipes/cannons
                        if below != ground_char and below not in pipe_chars and below not in cannon_chars:
                            grid[y + 1][x] = "-"

    # Pass 5: cannons must be attached to ground (no floating cannons)
    for y in range(h):
        for x in range(w):
            if grid[y][x] in cannon_chars:
                if y == h - 1:
                    # Already on bottom row; ensure it's ground under if conceptually needed
                    continue
                if grid[y + 1][x] != ground_char:
                    grid[y + 1][x] = ground_char

    return "\n".join("".join(row) for row in grid)


def carve_ground_gaps(
    text: str,
    gap_rate: float = 0.0,
    min_width: int = 2,
    max_width: int = 6,
    ground_char: str = "#",
    sky_char: str = "-",
    protect_tokens: List[str] = None,
) -> str:
    if gap_rate <= 0.0 or max_width <= 0 or min_width <= 0:
        return text
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    if not lines:
        return text
    h = len(lines)
    w = max(len(l) for l in lines)
    grid = [list(l + (sky_char * (w - len(l)))) for l in lines]

    # Determine protected columns (those with pipes/cannons above)
    if protect_tokens is None:
        protect_tokens = ["p", "P", "c"]
    protected = set()
    for x in range(w):
        for y in range(h - 1):
            if grid[y][x] in protect_tokens:
                protected.add(x)
                break

    yb = h - 1
    x = 0
    rng = random.Random(12345)
    while x < w:
        if grid[yb][x] == ground_char and x not in protected and rng.random() < gap_rate:
            gap_w = rng.randint(min_width, max_width)
            end = min(w, x + gap_w)
            # Apply gap only where ground exists and not protected
            for gx in range(x, end):
                if gx in protected:
                    break
                grid[yb][gx] = sky_char
            # Leave at least one ground tile after a gap
            x = end + 1
        else:
            x += 1

    return "\n".join("".join(row) for row in grid)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and use an LSTM to generate Mario levels")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--levels_dir", default="Levels", type=str, help="Directory containing .txt levels")
    p_train.add_argument("--output_dir", default=".", type=str, help="Where to save model and vocab")
    p_train.add_argument("--seq_len", type=int, default=128)
    p_train.add_argument("--batch_size", type=int, default=64)
    p_train.add_argument("--embedding_dim", type=int, default=128)
    p_train.add_argument("--hidden_dim", type=int, default=256)
    p_train.add_argument("--num_layers", type=int, default=2)
    p_train.add_argument("--dropout", type=float, default=0.1)
    p_train.add_argument("--lr", type=float, default=3e-3)
    p_train.add_argument("--max_epochs", type=int, default=20)
    p_train.add_argument("--weight_decay", type=float, default=0.0)
    p_train.add_argument("--grad_clip", type=float, default=1.0)
    p_train.add_argument("--seed", type=int, default=42)

    p_gen = sub.add_parser("generate", help="Generate level text from a trained model")
    p_gen.add_argument("--model_path", default="mario_lstm.pt", type=str)
    p_gen.add_argument("--vocab_path", default="vocab.json", type=str)
    p_gen.add_argument("--length", type=int, default=2000)
    p_gen.add_argument("--temperature", type=float, default=1.0)
    p_gen.add_argument("--seed_text", type=str, default="")
    p_gen.add_argument("--out", type=str, default="generated_level.txt")
    p_gen.add_argument("--render_png", action="store_true", help="Also render a PNG of the generated level")
    p_gen.add_argument("--png_path", type=str, default="generated_level.png", help="Path to save rendered PNG")
    p_gen.add_argument("--wrap_width", type=int, default=0, help="Optional: wrap generated text to this many columns")
    p_gen.add_argument("--target_height", type=int, default=0, help="Optional: force level to this many rows")
    p_gen.add_argument("--enforce_ground", action="store_true", help="Force bottom row to ground '#'")
    p_gen.add_argument("--top_k", type=int, default=0, help="Top-k sampling (0 disables)")
    p_gen.add_argument("--top_p", type=float, default=0.0, help="Nucleus sampling p (0 disables)")
    p_gen.add_argument("--hash_bias", type=float, default=0.0, help="Additive logit bias for '#' token")
    p_gen.add_argument("--enforce_rules", action="store_true", help="Apply heuristic rules: pipes stack vertically and require ground support")
    p_gen.add_argument("--ban", type=str, nargs="*", default=["|"], help="Tokens to ban during sampling (default bans '|')")
    p_gen.add_argument("--gap_rate", type=float, default=0.0, help="Probability to start a ground gap on bottom row (0 disables)")
    p_gen.add_argument("--gap_min", type=int, default=2, help="Minimum gap width in tiles")
    p_gen.add_argument("--gap_max", type=int, default=6, help="Maximum gap width in tiles")

    args = parser.parse_args()

    if args.command == "train":
        cfg = TrainConfig(
            levels_dir=args.levels_dir,
            output_dir=args.output_dir,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            lr=args.lr,
            max_epochs=args.max_epochs,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            seed=args.seed,
        )
        train_model(cfg)
    elif args.command == "generate":
        raw_text = generate(
            model_path=args.model_path,
            vocab_path=args.vocab_path,
            length=args.length,
            temperature=args.temperature,
            seed_text=args.seed_text,
            top_k=args.top_k,
            top_p=args.top_p,
            hash_bias=args.hash_bias,
            banned_tokens=args.ban,
        )
        text = postprocess_level_text(
            raw_text,
            wrap_width=args.wrap_width if args.wrap_width > 0 else None,
            target_height=args.target_height if args.target_height > 0 else None,
            enforce_ground=bool(args.enforce_ground),
        )
        if getattr(args, "enforce_rules", False):
            text = enforce_generation_rules(text)
        if args.gap_rate and args.gap_rate > 0.0:
            text = carve_ground_gaps(
                text,
                gap_rate=args.gap_rate,
                min_width=max(1, args.gap_min),
                max_width=max(args.gap_min, args.gap_max),
            )
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved generated text to {args.out}")
        if getattr(args, "render_png", False):
            try:
                render_level_text_to_png(text, args.png_path)
                print(f"Saved rendered PNG to {args.png_path}")
            except Exception as e:
                print(f"Failed to render PNG: {e}")


if __name__ == "__main__":
    main()

