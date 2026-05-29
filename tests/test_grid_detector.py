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


def test_aspect_fallback_returns_plausible_grid() -> None:
    settings = load_settings(None)
    grid = estimate_grid_from_aspect(744, 558, settings)
    assert grid is not None
    assert settings.expected_piece_count_min <= grid.total_pieces <= settings.expected_piece_count_max
