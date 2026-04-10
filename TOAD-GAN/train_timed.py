"""Train TOAD-GAN and record wall-clock time; default ``out_dir`` avoids overwriting ``output``."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from config import TOADGANConfig
from train import train

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent


def _resolve_existing_file(raw: str, label: str) -> str:
    """Resolve a user path; if missing, try the TOAD-GAN folder and repo root (common when cwd differs)."""
    p = Path(raw)
    if p.is_file():
        return str(p.resolve())
    if p.is_absolute():
        rp = p.resolve()
        if rp.is_file():
            return str(rp)
        raise FileNotFoundError(f"{label} not found: {raw!r} (resolved: {rp})")

    candidates: list[Path] = []
    for base in (Path.cwd(), _SCRIPT_DIR, _REPO_ROOT):
        c = (base / raw).resolve()
        if c not in candidates:
            candidates.append(c)
        if c.is_file():
            return str(c)

    tried = "\n  ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"{label} not found: {raw!r}\n"
        f"  Tried (cwd={Path.cwd()}):\n  {tried}"
    )


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.1f} min ({seconds:.0f} s)"
    hours = minutes / 60.0
    return f"{hours:.2f} h ({minutes:.1f} min)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train TOAD-GAN and record total wall-clock time (writes training_time_report.json).",
    )
    parser.add_argument("--level", required=True, help="Path to .txt level file")
    parser.add_argument("--vocab", required=True, help="Path to vocab.json")
    parser.add_argument(
        "--out_dir",
        default="output_timed",
        help="Output directory relative to cwd (default: output_timed; use from TOAD-GAN so it stays beside train.py)",
    )
    parser.add_argument("--num_scales", type=int, default=5)
    parser.add_argument("--scale_factor", type=float, default=0.67)
    parser.add_argument("--num_epochs", type=int, default=4000)
    parser.add_argument("--lr_g", type=float, default=5e-4)
    parser.add_argument("--lr_d", type=float, default=5e-4)
    parser.add_argument("--lambda_grad", type=float, default=0.1)
    parser.add_argument("--alpha_recon", type=float, default=10.0)
    parser.add_argument("--d_steps", type=int, default=3)
    parser.add_argument("--noise_amp", type=float, default=0.15)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--num_conv_layers", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    level_path = _resolve_existing_file(args.level, "Level file")
    vocab_path = _resolve_existing_file(args.vocab, "Vocab file")

    cfg = TOADGANConfig(
        level_path=level_path,
        vocab_path=vocab_path,
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

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[train_timed] Checkpoints and report will go to: {out.resolve()}\n")

    t0 = time.perf_counter()
    status = "failed"
    err_msg: str | None = None
    try:
        train(cfg)
        status = "completed"
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed = time.perf_counter() - t0
        report = {
            "status": status,
            "wall_clock_seconds": round(elapsed, 3),
            "wall_clock_human": _format_duration(elapsed),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "out_dir": str(out.resolve()),
            "level_path": str(Path(cfg.level_path).resolve()),
            "vocab_path": str(Path(cfg.vocab_path).resolve()),
            "num_scales": cfg.num_scales,
            "num_epochs": cfg.num_epochs,
            "scale_factor": cfg.scale_factor,
            "seed": cfg.seed,
        }
        if err_msg:
            report["error"] = err_msg

        report_path = out / "training_time_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"\n[train_timed] Wall time: {report['wall_clock_human']} "
              f"({report['wall_clock_seconds']} s)")
        print(f"[train_timed] Report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
