"""Phase 4 — target map construction tests."""

from __future__ import annotations

from pathlib import Path

import cv2

from puzzle_assistant.calibration.board_detector import detect_board
from puzzle_assistant.calibration.grid_detector import detect_grid_from_init_view
from puzzle_assistant.calibration.reference_panel import detect_reference_panel
from puzzle_assistant.config import load_settings
from puzzle_assistant.reference.target_map import (
    build_from_init_view,
    build_from_reference_panel,
)

WIN_X, WIN_Y, WIN_W, WIN_H = 0, 64, 1920, 1016


def test_primary_target_map_from_init_view(fixtures_dir: Path) -> None:
    settings = load_settings(None)
    full = cv2.imread(str(fixtures_dir / "init_view_4.png"))
    img = full[WIN_Y:WIN_Y + WIN_H, WIN_X:WIN_X + WIN_W]
    board_bbox = detect_board(img, settings)
    assert board_bbox is not None
    board_crop = img[board_bbox.y:board_bbox.y + board_bbox.h, board_bbox.x:board_bbox.x + board_bbox.w]
    grid = detect_grid_from_init_view(board_crop, settings)
    assert grid is not None

    tmap = build_from_init_view(board_crop, grid, settings)
    assert tmap.quality == "primary"
    assert len(tmap.cells) == grid.rows
    assert all(len(row) == grid.cols for row in tmap.cells)
    # First cell should have a real image, not the 1×1 placeholder.
    h, w = tmap.cells[0][0].image.shape[:2]
    assert h > 5 and w > 5
    assert tmap.cells[0][0].lab_mean.shape == (3,)


def test_fallback_target_map_from_panel(fixtures_dir: Path) -> None:
    settings = load_settings(None)
    full = cv2.imread(str(fixtures_dir / "mid_game_1.png"))
    img = full[WIN_Y:WIN_Y + WIN_H, WIN_X:WIN_X + WIN_W]
    panel_bbox = detect_reference_panel(img, settings)
    assert panel_bbox is not None
    panel = img[panel_bbox.y:panel_bbox.y + panel_bbox.h, panel_bbox.x:panel_bbox.x + panel_bbox.w]

    # Use the grid we know for this puzzle (mid_game_1 corresponds to plage/cliff
    # init_view_5 → 14×17).
    from puzzle_assistant.utils.coords import GridSpec
    grid = GridSpec(cols=17, rows=14, cell_w=744 / 17, cell_h=558 / 14)
    tmap = build_from_reference_panel(panel, grid, 744, 558, settings)
    assert tmap.quality == "fallback"
    assert len(tmap.cells) == 14
    assert all(len(row) == 17 for row in tmap.cells)
