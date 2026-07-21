"""Endgame rescue: match the piece's silhouette to the holes left on the board.

Appearance matching stalls on the last handful of pieces — they sit in regions
that look alike, so the score or margin gate rejects them and the player is left
to find those pieces alone. But the *live* board carries a second, completely
independent signal: an unfilled cell shows as a hole in the bare-board colour,
and that hole's outline is the exact complement of the piece belonging in it.

So when appearance gives up we compare the dragged piece's own silhouette
against each remaining hole. Measured on 43 recorded pickups the matcher failed
to predict: at the settings below this rescues 13 of them with zero errors.

Two properties keep it safe:

* It only ever runs when the matcher produced **no** prediction, so it cannot
  overturn a correct answer — it can only add one where there was none. (Used as
  a re-ranker instead it *would* be harmful: on pickups the matcher already had
  right, shape disagreed 4 times out of 9.)
* It fires only when one hole fits clearly better than the runner-up
  (``hole_shape_min_gap``). Early game the holes merge into big multi-cell
  regions and every fit is equally bad, so the gap collapses and nothing fires —
  measured 0 % accuracy above ~30 empty cells, which this correctly declines.
"""

from __future__ import annotations

import cv2
import numpy as np

from puzzle_assistant.config import Settings
from puzzle_assistant.utils.coords import CellAddress, GridSpec


def rescue_by_hole_shape(
    piece_fg: np.ndarray,
    live_board_bgr: np.ndarray,
    grid: GridSpec,
    settings: Settings,
) -> CellAddress | None:
    """Return the cell whose hole the piece's silhouette fits, or ``None``.

    ``piece_fg`` is the piece's foreground mask (non-zero = piece), in the same
    pixel scale as ``live_board_bgr`` — both come from the same screen frame.
    """
    if piece_fg.size == 0 or live_board_bgr.size == 0:
        return None

    silhouette = _tight(piece_fg)
    if silhouette is None:
        return None

    holes = _hole_mask(live_board_bgr, settings)
    cell_h = live_board_bgr.shape[0] / grid.rows
    cell_w = live_board_bgr.shape[1] / grid.cols

    candidates = _empty_cells(holes, grid, cell_w, cell_h, settings)
    if not candidates or len(candidates) > settings.hole_shape_max_empty_cells:
        return None

    scores = sorted(
        (_best_fit(holes, silhouette, r, c, cell_w, cell_h, settings), (r, c))
        for r, c in candidates
    )
    scores.reverse()
    best_score, best_cell = scores[0]
    runner_up = scores[1][0] if len(scores) > 1 else 0.0
    if best_score - runner_up < settings.hole_shape_min_gap:
        return None
    return CellAddress(row=best_cell[0], col=best_cell[1])


def _tight(mask: np.ndarray) -> np.ndarray | None:
    """Crop a binary mask to its content."""
    rows = np.where(mask.max(axis=1) > 0)[0]
    cols = np.where(mask.max(axis=0) > 0)[0]
    if len(rows) < 4 or len(cols) < 4:
        return None
    return (mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1] > 0).astype(np.uint8)


def _hole_mask(live_board_bgr: np.ndarray, settings: Settings) -> np.ndarray:
    """Pixels showing bare board — i.e. not covered by a placed piece."""
    hsv = cv2.cvtColor(live_board_bgr, cv2.COLOR_BGR2HSV)
    low = np.array(settings.board_light_hsv_low, dtype=np.uint8)
    high = np.array(settings.board_light_hsv_high, dtype=np.uint8)
    return cv2.inRange(hsv, low, high)


def _empty_cells(
    holes: np.ndarray, grid: GridSpec, cell_w: float, cell_h: float, settings: Settings
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for row in range(grid.rows):
        for col in range(grid.cols):
            patch = holes[
                int(row * cell_h):int((row + 1) * cell_h),
                int(col * cell_w):int((col + 1) * cell_w),
            ]
            if patch.size and float((patch > 0).mean()) >= settings.hole_shape_empty_min:
                out.append((row, col))
    return out


def _best_fit(
    holes: np.ndarray,
    silhouette: np.ndarray,
    row: int,
    col: int,
    cell_w: float,
    cell_h: float,
    settings: Settings,
) -> float:
    """Best silhouette/hole IoU over a small alignment search.

    The search matters far more than it looks: the piece's centre is only known
    to within a few pixels, and without it the same rule misfires — measured 2
    errors at gap 0.05 with no search versus none at a ±8 px search.
    """
    sh, sw = silhouette.shape
    top = round((row + 0.5) * cell_h - sh / 2)
    left = round((col + 0.5) * cell_w - sw / 2)
    radius = settings.hole_shape_align_radius
    best = 0.0
    for dy in range(-radius, radius + 1, 2):
        for dx in range(-radius, radius + 1, 2):
            best = max(best, _iou(holes, silhouette, top + dy, left + dx))
    return best


def _iou(holes: np.ndarray, silhouette: np.ndarray, top: int, left: int) -> float:
    h, w = holes.shape
    sh, sw = silhouette.shape
    y0, x0 = max(0, top), max(0, left)
    y1, x1 = min(h, top + sh), min(w, left + sw)
    if y1 <= y0 or x1 <= x0:
        return 0.0
    hole_part = (holes[y0:y1, x0:x1] > 0).astype(np.uint8)
    sil_part = silhouette[y0 - top:y1 - top, x0 - left:x1 - left]
    union = int((hole_part | sil_part).sum())
    if union == 0:
        return 0.0
    return int((hole_part & sil_part).sum()) / union
