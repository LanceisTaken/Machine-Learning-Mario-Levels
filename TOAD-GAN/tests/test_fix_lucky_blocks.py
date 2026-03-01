"""Tests for the fix_lucky_blocks post-processing step in generate.py."""

import sys
import os

# Allow imports from the parent TOAD-GAN directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from generate import fix_lucky_blocks  # noqa: E402

# Minimal vocab mapping (only the tokens fix_lucky_blocks cares about)
STOI = {"?": 6, "-": 3, "#": 1, "B": 7}
LUCKY = STOI["?"]
SKY = STOI["-"]
GROUND = STOI["#"]
BRICK = STOI["B"]


def _col(grid, c):
    """Extract a single column from a 2-D grid for easy assertion."""
    return [grid[r][c] for r in range(len(grid))]


# ── Test cases ──────────────────────────────────────────────────────────────


def test_stacked_directly_removes_upper():
    """Two ? blocks 1 row apart → upper removed."""
    S, L, G = SKY, LUCKY, GROUND
    grid = [
        [S],  # row 0
        [L],  # row 1  ← upper ? (should be removed)
        [S],  # row 2  ← only 1 empty row of gap
        [L],  # row 3  ← lower ? (kept)
        [G],  # row 4
    ]
    result = fix_lucky_blocks(grid, STOI)
    assert _col(result, 0) == [S, S, S, L, G], (
        "Upper ? should be replaced with sky when gap < 3"
    )


def test_sufficient_gap_keeps_both():
    """Two ? blocks with 4 rows between them → both kept."""
    S, L, G = SKY, LUCKY, GROUND
    grid = [
        [L],  # row 0  ← upper ?
        [S],  # row 1
        [S],  # row 2
        [S],  # row 3  ← 3 empty rows between row 0 and row 4
        [L],  # row 4  ← lower ?
        [G],  # row 5
    ]
    result = fix_lucky_blocks(grid, STOI)
    assert _col(result, 0) == [L, S, S, S, L, G], (
        "Both ? blocks should be kept when gap >= 3"
    )


def test_three_stacked_removes_middle():
    """Three ? blocks each 1 row apart → middle removed, top kept (gap=3 to bottom)."""
    S, L, G = SKY, LUCKY, GROUND
    grid = [
        [L],  # row 0  ← topmost ? (gap from row 4 = 3, kept)
        [S],  # row 1
        [L],  # row 2  ← middle ? (gap to row 4 = 1, removed)
        [S],  # row 3
        [L],  # row 4  ← bottom ? (kept)
        [G],  # row 5
    ]
    result = fix_lucky_blocks(grid, STOI)
    assert _col(result, 0) == [L, S, S, S, L, G], (
        "Bottom ? kept, middle removed (gap=1), top kept (gap=3 from bottom)"
    )


def test_no_lucky_blocks_unchanged():
    """Column with no ? blocks → level unchanged."""
    S, G = SKY, GROUND
    grid = [
        [S],
        [S],
        [G],
    ]
    result = fix_lucky_blocks(grid, STOI)
    assert _col(result, 0) == [S, S, G]


def test_single_lucky_block_unchanged():
    """One ? block → level unchanged."""
    S, L, G = SKY, LUCKY, GROUND
    grid = [
        [S],
        [L],
        [G],
    ]
    result = fix_lucky_blocks(grid, STOI)
    assert _col(result, 0) == [S, L, G]


def test_mixed_tokens_only_lucky_affected():
    """B (brick) blocks interleaved with ? → only ? spacing enforced."""
    S, L, G, B = SKY, LUCKY, GROUND, BRICK
    grid = [
        [L],  # row 0  ← upper ? (removed: gap to row 3 is only 2)
        [B],  # row 1  ← brick (left alone)
        [S],  # row 2
        [L],  # row 3  ← lower ? (kept)
        [G],  # row 4
    ]
    result = fix_lucky_blocks(grid, STOI)
    assert result[1][0] == B, "Brick block must not be modified"
    assert result[0][0] == S, "Upper ? should be removed (gap < 3)"
    assert result[3][0] == L, "Lower ? should be kept"


def test_multiple_columns_independent():
    """Each column is processed independently."""
    S, L, G = SKY, LUCKY, GROUND
    grid = [
        [L, S],  # row 0
        [S, L],  # row 1
        [L, S],  # row 2
        [G, L],  # row 3
        [G, G],  # row 4
    ]
    result = fix_lucky_blocks(grid, STOI)
    # Column 0: rows 0 and 2 have ?, gap is 1 → upper (row 0) removed
    assert _col(result, 0) == [S, S, L, G, G]
    # Column 1: rows 1 and 3 have ?, gap is 1 → upper (row 1) removed
    assert _col(result, 1) == [S, S, S, L, G]


def test_missing_vocab_returns_unchanged():
    """If vocab is missing '?' or '-', skip silently."""
    grid = [[3, 6], [6, 3]]
    result = fix_lucky_blocks(grid, {})  # empty vocab
    assert result == grid, "Grid must be returned unchanged with missing vocab"
