"""Phase 5 — segmentation + pickup + group classification tests."""

from __future__ import annotations

from pathlib import Path

import cv2

from puzzle_assistant.calibration.grid_detector import estimate_grid_from_aspect
from puzzle_assistant.config import load_settings
from puzzle_assistant.piece.group_detection import classify
from puzzle_assistant.piece.pickup import pickup_from_window
from puzzle_assistant.utils.coords import GridSpec

WIN_X, WIN_Y, WIN_W, WIN_H = 0, 64, 1920, 1016


def _window_crop(fixtures_dir: Path, name: str) -> "cv2.typing.MatLike":
    full = cv2.imread(str(fixtures_dir / f"{name}.png"))
    return full[WIN_Y:WIN_Y + WIN_H, WIN_X:WIN_X + WIN_W]


def test_extract_single_piece_from_board_center(fixtures_dir: Path) -> None:
    settings = load_settings(None)
    img = _window_crop(fixtures_dir, "dragging_single_1")
    grid = estimate_grid_from_aspect(744, 558, settings)
    assert grid is not None

    # Cursor right on top of the piece sitting in board center.
    result = pickup_from_window(img, (905, 390), grid, settings)
    assert result is not None, "piece should be segmented"
    bb = result.piece.bbox
    # bbox is relative to the pickup region; just sanity-check size.
    assert 20 < bb.w < 120, f"unexpected piece width {bb.w}"
    assert 20 < bb.h < 120, f"unexpected piece height {bb.h}"
    assert result.piece.area_px > 800


def test_group_classification_thresholds(fixtures_dir: Path) -> None:
    settings = load_settings(None)
    grid = GridSpec(cols=17, rows=14, cell_w=744 / 17, cell_h=558 / 14)

    # Synthesise a single-piece PickedPiece (small bbox + area).
    import numpy as np

    from puzzle_assistant.piece.segmentation import PickedPiece
    from puzzle_assistant.utils.coords import Bbox
    small = PickedPiece(
        piece_full=np.zeros((40, 40, 3), dtype=np.uint8),
        piece_core=np.zeros((40, 40, 3), dtype=np.uint8),
        bbox=Bbox(0, 0, 40, 40),
        area_px=1500,
    )
    assert classify(small, grid, settings) == "single"

    big = PickedPiece(
        piece_full=np.zeros((120, 120, 3), dtype=np.uint8),
        piece_core=np.zeros((120, 120, 3), dtype=np.uint8),
        bbox=Bbox(0, 0, 120, 120),
        area_px=10000,
    )
    assert classify(big, grid, settings) == "group"


def test_extract_returns_none_on_empty_desk(fixtures_dir: Path) -> None:
    """A click on the bare desk should not segment anything piece-like."""

    settings = load_settings(None)
    img = _window_crop(fixtures_dir, "mid_game_1")
    grid = estimate_grid_from_aspect(744, 558, settings)
    assert grid is not None
    # Click smack in the middle of the empty board interior.
    result = pickup_from_window(img, (900, 400), grid, settings)
    # Either no piece, or a tiny noise blob.
    if result is not None:
        assert result.piece.area_px < 800
