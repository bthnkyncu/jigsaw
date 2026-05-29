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
    expected_cell: tuple[float, float] | None = None,
) -> PickedPiece | None:
    """Segment the piece under ``cursor_local`` inside ``region_bgr``.

    ``expected_cell`` is the ``(cell_w, cell_h)`` of one puzzle cell. When
    given, the segmenter rejects components far larger than a single piece
    (i.e. several pieces stuck together near a pile) so that the matcher is
    never fed a multi-piece blob — a wrong overlay is worse than no overlay.

    Returns ``None`` if no clean single-piece component sits under the cursor.
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

    # A single piece, tabs included, can span ~2× a cell per side (tabs
    # protrude ~40 % on each edge) and ~3× a cell in area. Anything larger is
    # a clump of touching pieces and must be rejected.
    max_dim = None
    max_area = None
    if expected_cell is not None:
        ecw, ech = expected_cell
        max_dim = max(ecw, ech) * 2.0
        max_area = ecw * ech * 3.0

    # Prefer the component that actually sits under the cursor; fall back to
    # the nearest one. Reject components that are too big to be one piece.
    cursor_label = -1
    if 0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1]:
        lbl_here = int(labels[cy, cx])
        if lbl_here != 0:
            cursor_label = lbl_here

    candidates: list[int] = []
    if cursor_label > 0:
        candidates.append(cursor_label)
    # Also gather nearby components in case the cursor sits on a tab gap.
    for label in range(1, n_labels):
        if label == cursor_label:
            continue
        x, y, w, h, area = stats[label]
        if area < 50:
            continue
        comp_cx = x + w / 2
        comp_cy = y + h / 2
        if (comp_cx - cx) ** 2 + (comp_cy - cy) ** 2 <= (max(region_bgr.shape[:2]) * 0.4) ** 2:
            candidates.append(label)

    best_label = -1
    best_dist = float("inf")
    for label in candidates:
        x, y, w, h, area = stats[label]
        if area < 50:
            continue
        if max_dim is not None and (w > max_dim or h > max_dim):
            continue
        if max_area is not None and area > max_area:
            continue
        comp_cx = x + w / 2
        comp_cy = y + h / 2
        dist = (comp_cx - cx) ** 2 + (comp_cy - cy) ** 2
        # The cursor's own component wins ties strongly.
        if label == cursor_label:
            dist *= 0.25
        if dist < best_dist:
            best_dist = dist
            best_label = label

    if best_label < 0:
        plog.event(
            "seg_no_valid_component",
            level=logging.DEBUG,
            cursor_label=cursor_label,
            n_components=n_labels - 1,
        )
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
