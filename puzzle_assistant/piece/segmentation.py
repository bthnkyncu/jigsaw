"""Segment the piece a player is currently holding.

Brief §7.9: starting from a window-local region centered on the cursor,

    1. Convert to HSV.
    2. Mask off the dark-blue desk background AND the light board surface.
    3. Connected components → pick the component nearest the cursor.
    4. Tight bbox around that component.
    5. **Core** = bbox content minus eroded edges (kills the puzzle tabs).

Returns the full piece, the core, the bbox, and area in pixels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from puzzle_assistant.config import Settings
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.coords import Bbox


@dataclass
class PickedPiece:
    """The result of segmentation around a cursor click."""

    piece_full: np.ndarray   # BGR crop covering the tab silhouette
    piece_core: np.ndarray   # BGR crop with edge tabs eroded away
    bbox: Bbox               # in region-local coords (relative to the input region)
    area_px: int             # number of pixels in the segmentation mask


def extract_piece(
    region_bgr: np.ndarray,
    cursor_local: tuple[int, int],
    settings: Settings,
) -> PickedPiece | None:
    """Segment the piece nearest ``cursor_local`` inside ``region_bgr``.

    Returns ``None`` if no piece-like component is present (e.g. the click
    landed on empty desk or already-placed board area).
    """

    if region_bgr.size == 0:
        return None

    hsv = cv2.cvtColor(region_bgr, cv2.COLOR_BGR2HSV)

    bg_low = np.array(settings.background_blue_hsv_low, dtype=np.uint8)
    bg_high = np.array(settings.background_blue_hsv_high, dtype=np.uint8)
    board_low = np.array(settings.board_light_hsv_low, dtype=np.uint8)
    board_high = np.array(settings.board_light_hsv_high, dtype=np.uint8)

    bg_mask = cv2.inRange(hsv, bg_low, bg_high)
    board_mask = cv2.inRange(hsv, board_low, board_high)

    piece_mask = cv2.bitwise_not(cv2.bitwise_or(bg_mask, board_mask))
    # Light cleanup.
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    piece_mask = cv2.morphologyEx(piece_mask, cv2.MORPH_OPEN, open_k)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(piece_mask, connectivity=8)
    if n_labels <= 1:
        plog.event("seg_no_components", level=logging.DEBUG)
        return None

    cx, cy = cursor_local
    best_label = -1
    best_dist = float("inf")
    for label in range(1, n_labels):
        x, y, w, h, area = stats[label]
        if area < 50:
            continue
        comp_cx = x + w / 2
        comp_cy = y + h / 2
        dist = (comp_cx - cx) ** 2 + (comp_cy - cy) ** 2
        if dist < best_dist:
            best_dist = dist
            best_label = label

    if best_label < 0:
        plog.event("seg_no_valid_component", level=logging.DEBUG)
        return None

    x, y, w, h, area = stats[best_label]
    piece_full = region_bgr[y : y + h, x : x + w].copy()

    component_mask = (labels[y : y + h, x : x + w] == best_label).astype(np.uint8) * 255
    # Erode the component mask by ``PIECE_CORE_ERODE_RATIO`` of its shorter side
    # to wipe out the puzzle tabs.
    erode_px = max(1, int(min(w, h) * settings.piece_core_erode_ratio))
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
    eroded_mask = cv2.erode(component_mask, erode_k)
    piece_core = cv2.bitwise_and(piece_full, piece_full, mask=eroded_mask)

    return PickedPiece(
        piece_full=piece_full,
        piece_core=piece_core,
        bbox=Bbox(x=int(x), y=int(y), w=int(w), h=int(h)),
        area_px=int(area),
    )
