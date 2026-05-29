"""Locate the small reference-image panel in the top-right of the window
and produce a perceptual signature for "new puzzle started" detection.

Brief §7.6. Strategy:

    The Gamyun scoreboard column always has the same vertical layout:
        - top: username badge (~20 % of the column height)
        - photo panel (the reference!) immediately under the badge
        - empty blue gap (the "biggest" feature)
        - bottom: rumuz / player list

    So we find the panel by scanning the column's per-row gray-variance
    profile: it is the *upper* band of high-variance rows that comes right
    after a short low-variance gap. Lateral bounds are then derived from
    the per-column variance profile within that band.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from puzzle_assistant.config import Settings
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.coords import Bbox

# Where to look — the right-side scoreboard column.
SEARCH_RIGHT_RATIO = 0.12
SEARCH_TOP_RATIO = 0.50

VARIANCE_WINDOW = 9
ROW_VAR_THRESHOLD = 8.0  # mean per-row std above this counts as "content"
MIN_PANEL_PX = 60


@dataclass(frozen=True)
class PanelSignature:
    """Combined Lab-mean + tiny dHash, for change-detection."""

    lab_mean: tuple[float, float, float]
    dhash: int

    def distance(self, other: "PanelSignature") -> float:
        lab_dist = float(np.linalg.norm(
            np.array(self.lab_mean) - np.array(other.lab_mean)
        )) / 100.0
        hash_dist = bin(self.dhash ^ other.dhash).count("1") / 64.0
        return 0.5 * lab_dist + 0.5 * hash_dist


def detect_reference_panel(frame_bgr: np.ndarray, settings: Settings) -> Bbox | None:
    """Return the panel bbox in *frame-local* coordinates, or ``None``."""

    _ = settings
    h, w = frame_bgr.shape[:2]
    sx = int(w * (1.0 - SEARCH_RIGHT_RATIO))
    sw = w - sx
    sh = int(h * SEARCH_TOP_RATIO)
    region = frame_bgr[0:sh, sx : sx + sw]
    if region.size == 0:
        return None

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.boxFilter(gray, ddepth=-1, ksize=(VARIANCE_WINDOW, VARIANCE_WINDOW))
    sqmean = cv2.boxFilter(gray * gray, ddepth=-1, ksize=(VARIANCE_WINDOW, VARIANCE_WINDOW))
    std = np.sqrt(np.maximum(sqmean - mean * mean, 0.0))

    rows_high = std.mean(axis=1) > ROW_VAR_THRESHOLD
    bands = _contiguous_bands(rows_high, gap_tolerance=8)
    bands = [(y0, y1) for y0, y1 in bands if (y1 - y0) >= MIN_PANEL_PX]
    if not bands:
        plog.event("ref_panel_no_row_band", level=logging.WARNING)
        return None

    # The photo panel is the **first sufficiently tall band** after the
    # username badge: pick the first band whose height >= MIN_PANEL_PX and
    # is at least 1.5× as tall as everything above it.
    candidate: tuple[int, int] | None = None
    for y0, y1 in bands:
        height = y1 - y0
        if height < MIN_PANEL_PX:
            continue
        candidate = (y0, y1)
        break

    if candidate is None:
        plog.event("ref_panel_no_valid_band", level=logging.WARNING)
        return None

    y0, y1 = candidate
    band_std = std[y0:y1]
    cols_high = band_std.mean(axis=0) > ROW_VAR_THRESHOLD
    col_bands = _contiguous_bands(cols_high, gap_tolerance=8)
    col_bands = [(x0, x1) for x0, x1 in col_bands if (x1 - x0) >= MIN_PANEL_PX]
    if not col_bands:
        plog.event("ref_panel_no_col_band", level=logging.WARNING)
        return None
    # Widest column band = the photo.
    x0, x1 = max(col_bands, key=lambda b: b[1] - b[0])

    bbox = Bbox(x=x0 + sx, y=y0, w=x1 - x0, h=y1 - y0)
    plog.event("ref_panel_ok", bbox=[bbox.x, bbox.y, bbox.w, bbox.h])
    return bbox


def _contiguous_bands(mask: np.ndarray, *, gap_tolerance: int = 0) -> list[tuple[int, int]]:
    """Return ``[(start, end), ...]`` for runs of ``True`` in ``mask``.

    ``gap_tolerance`` allows short runs of ``False`` inside a band without
    breaking it.
    """

    bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    gap = 0
    for i, m in enumerate(mask.tolist()):
        if m:
            if not in_band:
                start = i
                in_band = True
            gap = 0
        else:
            if in_band:
                gap += 1
                if gap > gap_tolerance:
                    bands.append((start, i - gap + 1))
                    in_band = False
                    gap = 0
    if in_band:
        bands.append((start, len(mask)))
    return bands


def compute_signature(panel_bgr: np.ndarray) -> PanelSignature:
    """Compute the Lab-mean + 8×8 dHash signature of a panel crop."""

    lab = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2LAB)
    lab_mean = tuple(float(v) for v in lab.reshape(-1, 3).mean(axis=0))
    small = cv2.resize(
        cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2GRAY),
        (9, 8),
        interpolation=cv2.INTER_AREA,
    )
    diff = small[:, 1:] > small[:, :-1]
    bits = diff.flatten()
    h = 0
    for bit in bits:
        h = (h << 1) | int(bit)
    return PanelSignature(
        lab_mean=(lab_mean[0], lab_mean[1], lab_mean[2]),
        dhash=h,
    )


def upscale_panel_to_board(panel_bgr: np.ndarray, board_w: int, board_h: int) -> np.ndarray:
    """Bicubic upscale used by the *fallback* target-map path."""

    return cv2.resize(panel_bgr, (board_w, board_h), interpolation=cv2.INTER_CUBIC)
