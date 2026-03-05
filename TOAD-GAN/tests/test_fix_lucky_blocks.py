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
    """Two ? blocks 1 row apart → upper removed (inter-block gap < 3)."""
    S, L, G = SKY, LUCKY, GROUND
    grid = [
        [S],  # row 0
        [L],  # row 1  ← upper ? (should be removed: only 1 row to lower)
        [S],  # row 2
        [L],  # row 3  ← lower ? (kept: 3 sky rows to ground)
        [S],  # row 4
        [S],  # row 5
        [S],  # row 6
        [G],  # row 7
    ]
    result = fix_lucky_blocks(grid, STOI)
    assert _col(result, 0) == [S, S, S, L, S, S, S, G], (
        "Upper ? should be replaced with sky when inter-block gap < 3"
    )


def test_sufficient_gap_keeps_both():
    """Two ? blocks with 3 rows between them and enough room below → both kept."""
    S, L, G = SKY, LUCKY, GROUND
    grid = [
        [L],  # row 0  ← upper ?
        [S],  # row 1
        [S],  # row 2
        [S],  # row 3
        [L],  # row 4  ← lower ? (3 sky rows to ground at row 8)
        [S],  # row 5
        [S],  # row 6
        [S],  # row 7
        [G],  # row 8
    ]
    result = fix_lucky_blocks(grid, STOI)
    assert _col(result, 0) == [L, S, S, S, L, S, S, S, G], (
        "Both ? blocks should be kept when gaps >= 3"
    )


def test_three_stacked_removes_middle():
    """Three ? blocks each 1 row apart → middle removed, top kept (gap=3 to bottom)."""
    S, L, G = SKY, LUCKY, GROUND
    grid = [
        [L],  # row 0  ← topmost ? (gap to row 4 = 3, kept by Pass 2)
        [S],  # row 1
        [L],  # row 2  ← middle ? (gap to row 4 = 1, removed by Pass 2)
        [S],  # row 3
        [L],  # row 4  ← bottom ? (kept: 3 sky rows to ground)
        [S],  # row 5
        [S],  # row 6
        [S],  # row 7
        [G],  # row 8
    ]
    result = fix_lucky_blocks(grid, STOI)
    assert _col(result, 0) == [L, S, S, S, L, S, S, S, G], (
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


def test_single_lucky_block_on_ground_removed():
    """One ? block with only 0 sky rows to ground → removed."""
    S, L, G = SKY, LUCKY, GROUND
    grid = [
        [S],
        [L],  # row 1 – ground is row 2, gap = 0
        [G],
    ]
    result = fix_lucky_blocks(grid, STOI)
    assert _col(result, 0) == [S, S, G], (
        "? should be removed when gap to ground < 3"
    )


def test_single_lucky_block_sufficient_gap_kept():
    """One ? block with 3 sky rows to ground → kept."""
    S, L, G = SKY, LUCKY, GROUND
    grid = [
        [S],
        [L],  # row 1 – ground is row 5, gap = 3
        [S],
        [S],
        [S],
        [G],
    ]
    result = fix_lucky_blocks(grid, STOI)
    assert _col(result, 0) == [S, L, S, S, S, G], (
        "? should be kept when gap to ground >= 3"
    )


def test_lucky_directly_on_ground_removed():
    """? sitting directly on # with 0 sky rows → removed."""
    S, L, G = SKY, LUCKY, GROUND
    grid = [
        [S],
        [S],
        [L],  # row 2 – ground is row 3, gap = 0
        [G],
    ]
    result = fix_lucky_blocks(grid, STOI)
    assert _col(result, 0) == [S, S, S, G], (
        "? directly on ground must be removed"
    )


def test_lucky_on_stairs_removed():
    """? sitting on a stair structure with only 1 sky row gap → removed."""
    S, L, G = SKY, LUCKY, GROUND
    grid = [
        [S],
        [S],
        [L],  # row 2 – ground is row 4; gap = 1
        [S],
        [G],
        [G],
    ]
    result = fix_lucky_blocks(grid, STOI)
    assert _col(result, 0) == [S, S, S, S, G, G], (
        "? above stairs with gap < 3 must be removed"
    )


def test_mixed_tokens_only_lucky_affected():
    """B (brick) blocks interleaved with ? → only ? spacing enforced."""
    S, L, G, B = SKY, LUCKY, GROUND, BRICK
    grid = [
        [L],  # row 0  ← upper ? (removed: gap to row 1 brick is only 0)
        [B],  # row 1  ← brick (left alone)
        [S],  # row 2
        [L],  # row 3  ← lower ? (gap to ground row 4 is 0, removed)
        [G],  # row 4
    ]
    result = fix_lucky_blocks(grid, STOI)
    assert result[1][0] == B, "Brick block must not be modified"
    assert result[0][0] == S, "Upper ? should be removed (gap < 3)"
    assert result[3][0] == S, "Lower ? should be removed (gap < 3)"


def test_multiple_columns_independent():
    """Each column is processed independently."""
    S, L, G = SKY, LUCKY, GROUND
    grid = [
        [L, S],   # row 0
        [S, L],   # row 1
        [S, S],   # row 2
        [S, S],   # row 3
        [S, S],   # row 4
        [L, S],   # row 5
        [S, L],   # row 6
        [S, S],   # row 7
        [S, S],   # row 8
        [S, S],   # row 9
        [G, G],   # row 10
    ]
    result = fix_lucky_blocks(grid, STOI)
    # Column 0: row 0 ? has gap=4 to row 5 ?, row 5 ? has gap=4 to ground → both kept
    assert result[0][0] == L
    assert result[5][0] == L
    # Column 1: row 1 ? has gap=4 to row 6 ?, row 6 ? has gap=3 to ground → both kept
    assert result[1][1] == L
    assert result[6][1] == L


def test_missing_vocab_returns_unchanged():
    """If vocab is missing '?' or '-', skip silently."""
    grid = [[3, 6], [6, 3]]
    result = fix_lucky_blocks(grid, {})  # empty vocab
    assert result == grid, "Grid must be returned unchanged with missing vocab"

