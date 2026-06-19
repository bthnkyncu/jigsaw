"""Phase 2 — board detector tests against captured fixtures."""

from __future__ import annotations

from pathlib import Path

import cv2
import pytest

from puzzle_assistant.calibration.board_detector import detect_board
from puzzle_assistant.config import load_settings

# Window bounds for the captured screenshots (the live capture path will hand
# us a window-cropped frame; fixtures still include the desktop, so we crop
# the same way the runtime does).
WIN_X, WIN_Y, WIN_W, WIN_H = 0, 64, 1920, 1016


def _load_window_crop(fixtures_dir: Path, name: str) -> "cv2.typing.MatLike":
    full = cv2.imread(str(fixtures_dir / f"{name}.png"))
    assert full is not None, f"missing fixture: {name}.png"
    return full[WIN_Y:WIN_Y + WIN_H, WIN_X:WIN_X + WIN_W]


@pytest.mark.parametrize(
    "name,expected_x,expected_y,expected_w,expected_h",
    [
        # Tolerances are wide intentionally — these fixtures encode the Gamyun
        # window furniture not the customer's 2560×1440 spec; coarse bounding
        # is enough for downstream grid detection to succeed.
        ("init_view_4", 525, 138, 744, 558),
        ("init_view_5", 525, 138, 744, 558),
        ("mid_game_1",  525, 138, 743, 557),
        ("mid_game_2",  525, 138, 743, 558),
        ("dragging_single_1", 525, 138, 743, 557),
    ],
)
def test_board_detect_known_fixtures(
    fixtures_dir: Path,
    name: str,
    expected_x: int,
    expected_y: int,
    expected_w: int,
    expected_h: int,
) -> None:
    settings = load_settings(None)
    img = _load_window_crop(fixtures_dir, name)
    bbox = detect_board(img, settings)
    assert bbox is not None, f"board not detected in {name}"
    assert abs(bbox.x - expected_x) <= 10
    assert abs(bbox.y - expected_y) <= 10
    assert abs(bbox.w - expected_w) <= 20
    assert abs(bbox.h - expected_h) <= 20


def test_board_detect_returns_none_on_blank() -> None:
    import numpy as np
    settings = load_settings(None)
    blank = np.zeros((600, 800, 3), dtype=np.uint8)
    assert detect_board(blank, settings) is None


@pytest.mark.parametrize(
    "name,min_aspect,max_aspect",
    [
        # Landscape board: wider than tall (aspect > 1). Always worked.
        ("init_horizontal_1", 1.5, 2.5),
        # Portrait board: taller than wide (aspect ~0.67). This is the
        # regression guard — the old aspect floor of 0.7 silently rejected
        # every vertical puzzle, so detection only worked on landscape images.
        ("init_vertical_1", 0.55, 0.80),
    ],
)
def test_board_detect_orientation(
    fixtures_dir: Path, name: str, min_aspect: float, max_aspect: float
) -> None:
    settings = load_settings(None)
    full = cv2.imread(str(fixtures_dir / f"{name}.png"))
    assert full is not None, f"missing fixture: {name}.png"
    # These fixtures are full-window captures (title bar included), exactly what
    # the live capture path hands detect_board — no manual crop.
    bbox = detect_board(full, settings)
    assert bbox is not None, f"board not detected in {name} (orientation gate?)"
    aspect = bbox.w / bbox.h
    assert min_aspect <= aspect <= max_aspect, f"{name}: aspect {aspect:.2f} out of range"
