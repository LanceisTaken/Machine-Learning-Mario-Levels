"""Constraint validation and repair logging for TOAD-GAN post-processing.

Wraps the existing ``fix_*`` functions (without modifying them) with
instrumentation that:

* scans the tile grid for constraint violations **before** repair,
* diffs the grid to count tiles actually modified by repair,
* re-scans **after** repair to confirm all constraints now pass,
* prints a human-readable report to the console, and
* optionally appends a row to ``constraint_log.csv``.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence

# All recognised constraint rules, in fixed display / CSV order.
RULE_NAMES: list[str] = [
    "Pipe without ground support",
    "Pipe at or below ground level",
    "Pipe body gap (not contiguous to ground)",
    "Pipe too short (< min height)",
    "Block on pipe head",
    "Lucky block insufficient space below",
    "Lucky block insufficient space above",
    "Lucky blocks too close vertically",
]


# ── Report data class ──────────────────────────────────────────────────────

@dataclass
class ConstraintReport:
    height: int = 0
    width: int = 0
    total_tiles: int = 0
    violations_before: Dict[str, int] = field(default_factory=dict)
    total_invalid: int = 0
    tiles_repaired: int = 0
    repair_rate_pct: float = 0.0
    violations_after: Dict[str, int] = field(default_factory=dict)
    total_remaining: int = 0
    post_repair_valid: bool = True


# ── Grid helpers ────────────────────────────────────────────────────────────

def _deep_copy(grid: List[List[int]]) -> List[List[int]]:
    return [row[:] for row in grid]


def _count_diff(a: List[List[int]], b: List[List[int]]) -> int:
    n = 0
    for ra, rb in zip(a, b):
        for va, vb in zip(ra, rb):
            if va != vb:
                n += 1
    return n


# ── Read-only violation scanners ────────────────────────────────────────────
#
# Each scanner mirrors the logic of the corresponding ``fix_*`` function
# in generate.py but only *detects* violations — it never mutates the grid.

def _scan_pipe_violations(
    grid: List[List[int]],
    stoi: Dict[str, int],
    min_height: int = 3,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    PIPE = stoi.get("§")
    GROUND = stoi.get("#")
    SKY = stoi.get("-")
    if PIPE is None or GROUND is None or SKY is None:
        return counts

    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    for c in range(cols):
        ground_row = None
        for r in range(rows - 1, -1, -1):
            if grid[r][c] == GROUND:
                ground_row = r
                break

        top_pipe = None
        for r in range(rows):
            if grid[r][c] == PIPE:
                top_pipe = r
                break

        if top_pipe is None:
            continue

        pipe_tile_count = sum(1 for r in range(rows) if grid[r][c] == PIPE)

        if ground_row is None:
            counts["Pipe without ground support"] = (
                counts.get("Pipe without ground support", 0) + pipe_tile_count)
            continue

        if top_pipe >= ground_row:
            counts["Pipe at or below ground level"] = (
                counts.get("Pipe at or below ground level", 0) + pipe_tile_count)
            continue

        for r in range(top_pipe, ground_row):
            if grid[r][c] != PIPE:
                counts["Pipe body gap (not contiguous to ground)"] = (
                    counts.get("Pipe body gap (not contiguous to ground)", 0) + 1)

        pipe_height = ground_row - top_pipe
        if pipe_height < min_height:
            deficit = min_height - pipe_height
            counts["Pipe too short (< min height)"] = (
                counts.get("Pipe too short (< min height)", 0) + deficit)

    return counts


def _scan_blocks_on_pipes_violations(
    grid: List[List[int]],
    stoi: Dict[str, int],
    clearance: int = 2,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    PIPE = stoi.get("§")
    SKY = stoi.get("-")
    if PIPE is None or SKY is None:
        return counts

    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    for c in range(cols):
        top_pipe = None
        for r in range(rows):
            if grid[r][c] == PIPE:
                top_pipe = r
                break
        if top_pipe is None:
            continue

        for offset in range(1, clearance + 1):
            above = top_pipe - offset
            if above >= 0 and grid[above][c] != SKY:
                counts["Block on pipe head"] = (
                    counts.get("Block on pipe head", 0) + 1)

    return counts


def _scan_lucky_block_violations(
    grid: List[List[int]],
    stoi: Dict[str, int],
    min_gap: int = 3,
    min_gap_above: int = 1,
) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    LUCKY = stoi.get("?")
    SKY = stoi.get("-")
    if LUCKY is None or SKY is None:
        return counts

    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    for c in range(cols):
        lucky_rows = [r for r in range(rows - 1, -1, -1)
                      if grid[r][c] == LUCKY]
        if not lucky_rows:
            continue

        # Pass 1a — insufficient sky below
        surviving: list[int] = []
        for r in lucky_rows:
            gap = 0
            for below in range(r + 1, rows):
                if grid[below][c] == SKY:
                    gap += 1
                else:
                    break
            if gap < min_gap:
                counts["Lucky block insufficient space below"] = (
                    counts.get("Lucky block insufficient space below", 0) + 1)
            else:
                surviving.append(r)

        # Pass 1b — insufficient sky above (only among 1a survivors)
        still_surviving: list[int] = []
        for r in surviving:
            gap_above = 0
            hit_solid = False
            for above in range(r - 1, -1, -1):
                if grid[above][c] == SKY:
                    gap_above += 1
                else:
                    hit_solid = True
                    break
            if hit_solid and gap_above < min_gap_above:
                counts["Lucky block insufficient space above"] = (
                    counts.get("Lucky block insufficient space above", 0) + 1)
            else:
                still_surviving.append(r)

        # Pass 2 — two ? blocks too close in the same column
        if len(still_surviving) >= 2:
            last_kept = still_surviving[0]
            for r in still_surviving[1:]:
                between = last_kept - r - 1
                if between >= min_gap:
                    last_kept = r
                else:
                    counts["Lucky blocks too close vertically"] = (
                        counts.get("Lucky blocks too close vertically", 0) + 1)

    return counts


def scan_all_violations(
    grid: List[List[int]],
    stoi: Dict[str, int],
) -> Dict[str, int]:
    """Run every constraint scanner and return merged violation counts."""
    merged: Dict[str, int] = {}
    for scan_fn in (_scan_pipe_violations,
                    _scan_blocks_on_pipes_violations,
                    _scan_lucky_block_violations):
        for rule, cnt in scan_fn(grid, stoi).items():
            merged[rule] = merged.get(rule, 0) + cnt
    return merged


# ── Console report ──────────────────────────────────────────────────────────

def format_report(r: ConstraintReport) -> str:
    lines = [
        "=== Constraint Report ===",
        f"Level size: {r.height} x {r.width} ({r.total_tiles} total tiles)",
        f"Invalid tiles detected: {r.total_invalid} ({r.repair_rate_pct:.2f}%)",
    ]
    if r.violations_before:
        lines.append("Breakdown:")
        for rule in RULE_NAMES:
            cnt = r.violations_before.get(rule, 0)
            if cnt:
                lines.append(f"  - {rule}: {cnt}")
    lines.append(f"Tiles repaired: {r.tiles_repaired}")
    lines.append(f"Post-repair valid: {'TRUE' if r.post_repair_valid else 'FALSE'}")
    if not r.post_repair_valid and r.violations_after:
        lines.append("Remaining violations after repair:")
        for rule in RULE_NAMES:
            cnt = r.violations_after.get(rule, 0)
            if cnt:
                lines.append(f"  - {rule}: {cnt}")
        lines.append(f"Total remaining invalid: {r.total_remaining}")
    lines.append("=" * 25)
    return "\n".join(lines)


def print_report(r: ConstraintReport) -> None:
    print(format_report(r))


# ── CSV logging ─────────────────────────────────────────────────────────────

_CSV_FIXED_COLS = [
    "timestamp", "level_width", "level_height", "total_tiles",
    "invalid_detected", "repair_rate_pct", "tiles_repaired",
    "post_repair_valid",
]


def append_csv(r: ConstraintReport, csv_path: str) -> None:
    """Append one row for this generation to the CSV log file."""
    header = _CSV_FIXED_COLS + RULE_NAMES
    write_header = not os.path.exists(csv_path)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level_width": r.width,
        "level_height": r.height,
        "total_tiles": r.total_tiles,
        "invalid_detected": r.total_invalid,
        "repair_rate_pct": round(r.repair_rate_pct, 4),
        "tiles_repaired": r.tiles_repaired,
        "post_repair_valid": r.post_repair_valid,
    }
    for rule in RULE_NAMES:
        row[rule] = r.violations_before.get(rule, 0)

    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── Instrumented wrapper ───────────────────────────────────────────────────

FixFn = Callable[[List[List[int]], Dict[str, int]], List[List[int]]]


def apply_fixes_with_report(
    tile_ids: List[List[int]],
    stoi: Dict[str, int],
    fix_fns: Sequence[FixFn],
    *,
    csv_path: Optional[str] = None,
    quiet: bool = False,
) -> tuple[List[List[int]], ConstraintReport]:
    """Apply *fix_fns* in order with full constraint reporting.

    The fix functions themselves are called **unmodified** — this wrapper
    only adds scanning and bookkeeping around them.

    Parameters
    ----------
    tile_ids : mutable grid (modified in-place by the fix functions)
    stoi     : vocabulary mapping (character → tile ID)
    fix_fns  : sequence of callables with signature
               ``fn(tile_ids, stoi) -> tile_ids``
    csv_path : if set, append one row to this CSV file
    quiet    : suppress console output when True

    Returns
    -------
    (tile_ids, ConstraintReport)
    """
    rows = len(tile_ids)
    cols = len(tile_ids[0]) if rows else 0
    total_tiles = rows * cols

    # 1. Pre-repair scan
    violations_before = scan_all_violations(tile_ids, stoi)
    total_invalid = sum(violations_before.values())

    # 2. Snapshot, then run existing fix functions
    snapshot = _deep_copy(tile_ids)
    for fn in fix_fns:
        tile_ids = fn(tile_ids, stoi)

    # 3. Count tiles actually modified
    tiles_repaired = _count_diff(snapshot, tile_ids)

    # 4. Post-repair scan
    violations_after = scan_all_violations(tile_ids, stoi)
    total_remaining = sum(violations_after.values())

    report = ConstraintReport(
        height=rows,
        width=cols,
        total_tiles=total_tiles,
        violations_before=violations_before,
        total_invalid=total_invalid,
        tiles_repaired=tiles_repaired,
        repair_rate_pct=(total_invalid / total_tiles * 100) if total_tiles else 0.0,
        violations_after=violations_after,
        total_remaining=total_remaining,
        post_repair_valid=(total_remaining == 0),
    )

    # 5. Console output
    if not quiet:
        print_report(report)

    # 6. CSV (optional)
    if csv_path:
        append_csv(report, csv_path)

    return tile_ids, report
