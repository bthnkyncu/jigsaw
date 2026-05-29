"""Classify a segmented piece as a *single piece* or a *merged group*.

Brief §7.10:
    expected_piece_area = cell_w * cell_h * 1.15
    if area > expected * GROUP_AREA_RATIO  → group

Bbox dimension checks are intentionally absent: puzzle tabs protrude ~30-50%
beyond the cell boundary in every direction, so any bbox multiplier tight
enough to catch real groups would misclassify normal single pieces.
Area is the reliable discriminator — a two-piece group is ~2× expected area.
"""

from __future__ import annotations

import logging
from typing import Literal

from puzzle_assistant.config import Settings
from puzzle_assistant.piece.segmentation import PickedPiece
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.coords import GridSpec

PieceClass = Literal["single", "group"]


def classify(piece: PickedPiece, grid: GridSpec, settings: Settings) -> PieceClass:
    expected = grid.cell_w * grid.cell_h * 1.15
    threshold = expected * settings.group_area_ratio
    if piece.area_px > threshold:
        plog.event(
            "group_classify_detail",
            level=logging.DEBUG,
            area=piece.area_px,
            expected=int(expected),
            threshold=int(threshold),
            bbox_w=piece.bbox.w,
            bbox_h=piece.bbox.h,
        )
        return "group"
    return "single"
