"""Detect the puzzle board bbox inside the Gamyun masa window.

The Gamyun desk surrounding the board is a calibrated mid-saturation blue
(roughly HSV ``(108, 73, 210)`` on the captured fixtures). Hue alone is too
permissive — sky/water inside the puzzle image overlaps the desk hue — but
the desk's *saturation band* (~60–120) is distinct from sky (S < 50) and from
neon highlights (S > 150). We mask on the joint H/S/V band, then take the
largest non-desk contour after excluding the right-side scoreboard column.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from puzzle_assistant.config import Settings
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.coords import Bbox

# Regions of the window we never look at (game UI furniture).
RIGHT_PANEL_WIDTH_RATIO = 0.10
BOTTOM_CHAT_HEIGHT_RATIO = 0.15
TOP_BAR_HEIGHT_RATIO = 0.04

MIN_BOARD_AREA_RATIO = 0.05
MAX_BOARD_AREA_RATIO = 0.45
# A puzzle board must be at least this tall in pixels to exclude UI strips.
MIN_BOARD_HEIGHT_PX = 250


def detect_board(frame_bgr: np.ndarray, settings: Settings) -> Bbox | None:
    """Return the puzzle board bbox in *frame-local* coordinates, or ``None``."""

    if frame_bgr.size == 0 or frame_bgr.ndim != 3:
        return None
    full_h, full_w = frame_bgr.shape[:2]

    work_x0 = 0
    work_y0 = int(full_h * TOP_BAR_HEIGHT_RATIO)
    work_x1 = int(full_w * (1.0 - RIGHT_PANEL_WIDTH_RATIO))
    work_y1 = int(full_h * (1.0 - BOTTOM_CHAT_HEIGHT_RATIO))
    work = frame_bgr[work_y0:work_y1, work_x0:work_x1]
    work_area = work.shape[0] * work.shape[1]

    desk_mask = _desk_background_mask(work, settings)
    fg_mask = cv2.bitwise_not(desk_mask)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        plog.event("board_detect_no_contour", level=logging.WARNING)
        return None

    best: tuple[int, int, int, int] | None = None
    best_score = 0.0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < work_area * MIN_BOARD_AREA_RATIO:
            continue
        if area > work_area * MAX_BOARD_AREA_RATIO:
            continue
        if ch < MIN_BOARD_HEIGHT_PX:
            continue
        aspect = cw / max(ch, 1)
        if aspect < 0.7 or aspect > 3.0:
            continue
        # Favor large, centrally-placed rectangles.
        cx = x + cw / 2
        center_offset = abs(cx - work.shape[1] / 2) / work.shape[1]
        score = area * (1.0 - 0.5 * center_offset)
        if score > best_score:
            best = (x, y, cw, ch)
            best_score = score

    if best is None:
        plog.event("board_detect_no_valid_contour", level=logging.WARNING)
        return None

    x, y, cw, ch = best
    bbox = Bbox(x=x + work_x0, y=y + work_y0, w=cw, h=ch)
    plog.event("board_detect_ok", bbox=[bbox.x, bbox.y, bbox.w, bbox.h])
    return bbox


def _desk_background_mask(work_bgr: np.ndarray, settings: Settings) -> np.ndarray:
    """Mask of the desk-blue pixels in ``work_bgr``."""

    hsv = cv2.cvtColor(work_bgr, cv2.COLOR_BGR2HSV)
    low = np.array(settings.background_blue_hsv_low, dtype=np.uint8)
    high = np.array(settings.background_blue_hsv_high, dtype=np.uint8)
    mask = cv2.inRange(hsv, low, high)

    # Close tiny holes (piece shadows on the desk).
    close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close)
    return mask
