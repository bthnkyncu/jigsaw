"""Crop the area around the cursor and run segmentation.

Brief §7.8: on each ``WM_LBUTTONDOWN`` we capture a square region of side
``cell × CURSOR_CAPTURE_RADIUS_CELL_MULTIPLIER * 2`` centered on the cursor,
then hand it to ``segmentation.extract_piece``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from puzzle_assistant.config import Settings
from puzzle_assistant.piece.segmentation import PickedPiece, extract_piece
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.coords import Bbox, GridSpec


@dataclass
class PickupResult:
    piece: PickedPiece
    region_bbox: Bbox            # in window-local coords
    cursor_in_region: tuple[int, int]


def pickup_from_window(
    window_bgr: np.ndarray,
    cursor_window: tuple[int, int],
    grid: GridSpec,
    settings: Settings,
) -> PickupResult | None:
    """Crop the pickup region and segment the piece, or return ``None``."""

    h, w = window_bgr.shape[:2]
    cx, cy = cursor_window
    radius = int(max(grid.cell_w, grid.cell_h) * settings.cursor_capture_radius_cell_multiplier)

    x0 = max(0, cx - radius)
    y0 = max(0, cy - radius)
    x1 = min(w, cx + radius)
    y1 = min(h, cy + radius)
    if x1 <= x0 or y1 <= y0:
        plog.event("pickup_region_empty", level=logging.WARNING)
        return None

    region = window_bgr[y0:y1, x0:x1]
    cursor_local = (cx - x0, cy - y0)
    piece = extract_piece(region, cursor_local, settings)
    if piece is None:
        return None

    return PickupResult(
        piece=piece,
        region_bbox=Bbox(x=x0, y=y0, w=x1 - x0, h=y1 - y0),
        cursor_in_region=cursor_local,
    )
