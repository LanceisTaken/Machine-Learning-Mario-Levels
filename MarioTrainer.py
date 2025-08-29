import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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


def build_vocab(text: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: Dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def decode(tokens: List[int], itos: Dict[int, str]) -> str:
    return "".join(itos[i] for i in tokens)


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

    for _ in range(max(0, length)):
        inp = next_id.view(1, 1)
        logits, hidden = model(inp, hidden)
        logits = logits[:, -1, :] / max(1e-6, temperature)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).squeeze(0).squeeze(0)
        generated.append(int(next_id.item()))

    return decode(generated, itos)


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
        text = generate(
            model_path=args.model_path,
            vocab_path=args.vocab_path,
            length=args.length,
            temperature=args.temperature,
            seed_text=args.seed_text,
        )
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Saved generated text to {args.out}")


if __name__ == "__main__":
    main()

