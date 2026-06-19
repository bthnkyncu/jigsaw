"""Phase 3 — grid detector tests against init-view fixtures."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from puzzle_assistant.calibration.board_detector import detect_board
from puzzle_assistant.calibration.grid_detector import (
    detect_grid_from_init_view,
    estimate_grid_from_aspect,
)
from puzzle_assistant.config import load_settings

WIN_X, WIN_Y, WIN_W, WIN_H = 0, 64, 1920, 1016


def _board_crop(fixtures_dir: Path, name: str) -> "cv2.typing.MatLike":
    full = cv2.imread(str(fixtures_dir / f"{name}.png"))
    img = full[WIN_Y:WIN_Y + WIN_H, WIN_X:WIN_X + WIN_W]
    settings = load_settings(None)
    bbox = detect_board(img, settings)
    assert bbox is not None
    return img[bbox.y:bbox.y + bbox.h, bbox.x:bbox.x + bbox.w]


@pytest.mark.parametrize(
    "name,expected_rows,expected_cols",
    [
        ("init_view_1", 13, 19),
        ("init_view_3", 14, 17),
        ("init_view_4", 14, 17),
        ("init_view_5", 14, 17),
    ],
)
def test_grid_from_init_view(
    fixtures_dir: Path, name: str, expected_rows: int, expected_cols: int
) -> None:
    settings = load_settings(None)
    crop = _board_crop(fixtures_dir, name)
    grid = detect_grid_from_init_view(crop, settings)
    assert grid is not None, f"grid not detected on {name}"
    assert grid.rows == expected_rows, f"{name}: rows={grid.rows}"
    assert grid.cols == expected_cols, f"{name}: cols={grid.cols}"


@pytest.mark.parametrize(
    "name,nominal_count,expected_rows,expected_cols",
    [
        ("init_150", 150, 10, 15),
        ("init_100", 100, 8, 12),
        ("init_70", 70, 7, 10),
        ("init_50", 50, 6, 8),
        ("init_30", 30, 5, 6),
    ],
)
def test_count_anchored_grid(
    fixtures_dir: Path,
    name: str,
    nominal_count: int,
    expected_rows: int,
    expected_cols: int,
) -> None:
    """With the user-entered piece count as an anchor, the exact rows×cols is
    recovered for any count (octave errors eliminated)."""
    settings = load_settings(None)
    settings.target_piece_count = nominal_count
    crop = _board_crop(fixtures_dir, name)
    grid = detect_grid_from_init_view(crop, settings)
    assert grid is not None, f"grid not detected on {name}"
    assert (grid.rows, grid.cols) == (expected_rows, expected_cols), (
        f"{name}: got {grid.rows}x{grid.cols}"
    )


@pytest.mark.parametrize(
    "name,nominal_count",
    [
        ("init_horizontal_1", 100),  # landscape board
        ("init_vertical_1", 100),    # portrait board — orientation regression guard
    ],
)
def test_grid_orientation(fixtures_dir: Path, name: str, nominal_count: int) -> None:
    """A portrait board must yield a grid (more rows than cols) just as a
    landscape one does. These are full-window captures, so they bypass the
    legacy WIN_Y crop helper."""
    settings = load_settings(None)
    settings.target_piece_count = nominal_count
    full = cv2.imread(str(fixtures_dir / f"{name}.png"))
    assert full is not None, f"missing fixture: {name}.png"
    bbox = detect_board(full, settings)
    assert bbox is not None, f"board not detected in {name}"
    crop = full[bbox.y:bbox.y + bbox.h, bbox.x:bbox.x + bbox.w]
    grid = detect_grid_from_init_view(crop, settings)
    assert grid is not None, f"grid not detected on {name}"
    total = grid.rows * grid.cols
    assert abs(total - nominal_count) <= nominal_count * 0.2, (
        f"{name}: {grid.rows}x{grid.cols}={total} far from {nominal_count}"
    )
    # Board orientation must carry into the grid: portrait → rows>cols.
    if bbox.h > bbox.w:
        assert grid.rows > grid.cols, f"{name}: portrait board gave {grid.rows}x{grid.cols}"
    else:
        assert grid.cols > grid.rows, f"{name}: landscape board gave {grid.rows}x{grid.cols}"


def test_aspect_fallback_disabled() -> None:
    # Aspect alone can't determine the piece count, so the parametric design
    # no longer guesses a grid from it — the periodicity path is authoritative.
    settings = load_settings(None)
    assert estimate_grid_from_aspect(744, 558, settings) is None
