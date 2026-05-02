#!/usr/bin/env python3
"""
Tile Pattern KL-Divergence (TPKL-Div) between a training level and generated levels.

For n in {2, 3}, extracts overlapping horizontal and vertical n-tile patterns from
rectangular token grids, builds empirical frequency distributions, and computes
KL( P_gen || P_train ) with additive epsilon smoothing on a shared support.

Compound pipe: ``pP`` and ``§`` are collapsed to one tile symbol (``§``), matching
TOAD-GAN ``level_utils.load_level`` behavior. Rows are right-padded with ``-`` to
a common width.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.stats import entropy

# Match TOAD-GAN level_utils: compound pipe as one cell
COMPOUND_TO_SINGLE = {"pP": "§"}


def apply_compound(text: str) -> str:
    for multi, single in COMPOUND_TO_SINGLE.items():
        text = text.replace(multi, single)
    return text


def load_level_grid(path: str) -> List[List[str]]:
    """
    Load a VGLC-style .txt level as a 2D list of single-character cells.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    raw = apply_compound(raw)
    lines = raw.strip().splitlines()
    while lines and lines[-1].strip() == "":
        lines.pop()

    if not lines:
        return []

    max_w = max(len(ln) for ln in lines)
    return [list(ln.ljust(max_w, "-")) for ln in lines]


def iter_horizontal_ngrams(
    grid: Sequence[Sequence[str]], n: int
) -> Iterable[Tuple[str, ...]]:
    h, w = len(grid), len(grid[0]) if grid else 0
    if n > w or n < 1:
        return
    for r in range(h):
        row = grid[r]
        for c in range(w - n + 1):
            yield tuple(row[c : c + n])


def iter_vertical_ngrams(
    grid: Sequence[Sequence[str]], n: int
) -> Iterable[Tuple[str, ...]]:
    h, w = len(grid), len(grid[0]) if grid else 0
    if n > h or n < 1:
        return
    for c in range(w):
        for r in range(h - n + 1):
            yield tuple(grid[r + k][c] for k in range(n))


def ngram_counts(
    grid: List[List[str]], n: int, direction: str
) -> Counter:
    c: Counter = Counter()
    it = (
        iter_horizontal_ngrams(grid, n)
        if direction == "h"
        else iter_vertical_ngrams(grid, n)
    )
    for g in it:
        c[g] += 1
    return c


def counts_to_pmf(counts: Counter, keys: Sequence[Tuple[str, ...]], eps: float) -> np.ndarray:
    arr = np.array([float(counts.get(k, 0)) + eps for k in keys], dtype=np.float64)
    s = arr.sum()
    if s <= 0.0:
        raise ValueError("empty support after smoothing")
    return arr / s


def kl_gen_vs_train(
    train_counts: Counter,
    gen_counts: Counter,
    eps: float,
) -> float:
    """
    KL( P_gen || P_train ) with shared support = union of patterns seen in
    either distribution. Uses scipy.stats.entropy(p_gen, p_train) which equals
    sum_i p_gen(i) * log( p_gen(i) / p_train(i) ).
    """
    keys = sorted(set(train_counts) | set(gen_counts))
    if not keys:
        return 0.0
    p_gen = counts_to_pmf(gen_counts, keys, eps)
    p_tr = counts_to_pmf(train_counts, keys, eps)
    return float(entropy(p_gen, p_tr))


def default_generated_paths(folder: Path) -> List[Path]:
    """
    11 files: ``generated_level.txt`` and ``generated_level_0.txt`` …
    ``generated_level_9.txt`` (natural sort on numeric suffixes).
    """
    if not folder.is_dir():
        raise FileNotFoundError(f"Not a directory: {folder}")

    paths = sorted(folder.glob("generated_level*.txt"), key=_sort_key_level_txt)
    if len(paths) != 11:
        raise FileNotFoundError(
            f"Expected 11 files matching 'generated_level*.txt' in {folder}, found {len(paths)}"
        )
    return paths


def _sort_key_level_txt(p: Path) -> Tuple[int, str]:
    stem = p.stem  # generated_level, generated_level_0, ...
    if stem == "generated_level":
        return (0, "")
    # generated_level_N
    if "_" in stem:
        suffix = stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            return (1, f"{int(suffix):04d}")
    return (2, p.name)


def run_tpkl(
    training_path: Path,
    generated_paths: List[Path],
    eps: float,
) -> Tuple[List[Dict[str, object]], Dict[str, float], Dict[str, float]]:
    train_grid = load_level_grid(str(training_path))
    if not train_grid:
        raise ValueError(f"Empty or invalid training level: {training_path}")

    train_h2 = ngram_counts(train_grid, 2, "h")
    train_h3 = ngram_counts(train_grid, 3, "h")
    train_v2 = ngram_counts(train_grid, 2, "v")
    train_v3 = ngram_counts(train_grid, 3, "v")

    rows: List[Dict[str, object]] = []
    metrics = ("h2", "h3", "v2", "v3")
    for gp in generated_paths:
        ggrid = load_level_grid(str(gp))
        if not ggrid:
            raise ValueError(f"Empty or invalid level: {gp}")
        h2 = kl_gen_vs_train(train_h2, ngram_counts(ggrid, 2, "h"), eps)
        h3 = kl_gen_vs_train(train_h3, ngram_counts(ggrid, 3, "h"), eps)
        v2 = kl_gen_vs_train(train_v2, ngram_counts(ggrid, 2, "v"), eps)
        v3 = kl_gen_vs_train(train_v3, ngram_counts(ggrid, 3, "v"), eps)
        rows.append(
            {
                "file": gp.name,
                "path": str(gp),
                "h2": h2,
                "h3": h3,
                "v2": v2,
                "v3": v3,
            }
        )

    arrays = {m: np.array([r[m] for r in rows], dtype=np.float64) for m in metrics}
    means = {f"mean_{m}": float(arrays[m].mean()) for m in metrics}
    stds = {f"std_{m}": float(arrays[m].std(ddof=1)) for m in metrics}
    return rows, means, stds


def print_summary(
    training_path: Path,
    rows: List[Dict[str, object]],
    means: Dict[str, float],
    stds: Dict[str, float],
) -> None:
    w = max(len(str(r["file"])) for r in rows) if rows else 20
    w = max(w, 24)

    def fmt(x: object) -> str:
        if isinstance(x, (int, float)):
            return f"{float(x):.6f}"
        return str(x)

    header = (
        f"{'file':<{w}}  "
        f"{'h2 (n=2)':>12}  "
        f"{'h3 (n=3)':>12}  "
        f"{'v2 (n=2)':>12}  "
        f"{'v3 (n=3)':>12}"
    )
    line = "-" * len(header)

    print()
    print("TPKL-Div: KL( P_gen || P_train ) per n-gram type (lower = closer to training)")
    print(f"Training level: {training_path}")
    print()
    print(header)
    print(line)
    for r in rows:
        print(
            f"{r['file']!s:<{w}}  {fmt(r['h2']):>12}  {fmt(r['h3']):>12}  {fmt(r['v2']):>12}  {fmt(r['v3']):>12}"
        )
    print(line)
    print(
        f"{'MEAN (11 levels)':<{w}}  {fmt(means['mean_h2']):>12}  {fmt(means['mean_h3']):>12}  {fmt(means['mean_v2']):>12}  {fmt(means['mean_v3']):>12}"
    )
    print(
        f"{'STD  (11 levels)':<{w}}  {fmt(stds['std_h2']):>12}  {fmt(stds['std_h3']):>12}  {fmt(stds['std_v2']):>12}  {fmt(stds['std_v3']):>12}"
    )
    print()


def main() -> int:
    root = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description="Tile Pattern KL-Divergence (TPKL-Div) evaluation")
    ap.add_argument(
        "--train",
        type=Path,
        default=root / "Levels" / "SMB1_1.txt",
        help="Path to training level .txt (VGLC format)",
    )
    ap.add_argument(
        "--gen-dir",
        type=Path,
        default=root / "TOAD-GAN" / "Sample Level Output",
        help="Folder containing 11 generated_level*.txt files",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=root / "tpkl_results.csv",
        help="Output CSV path",
    )
    ap.add_argument(
        "--eps",
        type=float,
        default=1e-9,
        help="Smoothing added to every pattern count before normalizing to PMF",
    )
    ap.add_argument(
        "--gen",
        type=Path,
        nargs="*",
        help="Optional explicit list of generated .txt files (overrides --gen-dir default set)",
    )
    args = ap.parse_args()

    train = args.train
    if not train.is_file():
        print(f"Error: training file not found: {train}", file=sys.stderr)
        return 1

    if args.gen:
        gen_paths = [Path(p) for p in args.gen]
    else:
        try:
            gen_paths = default_generated_paths(args.gen_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

    for p in gen_paths:
        if not p.is_file():
            print(f"Error: missing generated file: {p}", file=sys.stderr)
            return 1

    rows, means, stds = run_tpkl(train, gen_paths, args.eps)
    print_summary(train, rows, means, stds)

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames: List[str] = ["file", "h2", "h3", "v2", "v3"]
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})
        w.writerow(
            {
                "file": "MEAN",
                "h2": means["mean_h2"],
                "h3": means["mean_h3"],
                "v2": means["mean_v2"],
                "v3": means["mean_v3"],
            }
        )
        w.writerow(
            {
                "file": "STD",
                "h2": stds["std_h2"],
                "h3": stds["std_h3"],
                "v2": stds["std_v2"],
                "v3": stds["std_v3"],
            }
        )

    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
