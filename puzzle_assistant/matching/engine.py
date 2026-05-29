"""Top-level matching pipeline — sliding-window template match.

The earlier cell-by-cell ensemble (resize piece to one cell, score it against
every cell's cached features) was unreliable in practice: a dragged piece
carries cursor pixels, motion blur and a black-masked background, so every
cell scored ~0.28 with near-zero margin and nothing was ever selected.

The Puzzle_Game prototype instead localizes the piece directly on the full
reference board with a masked normalized cross-correlation, which is what we
do here:

1. Compute a foreground mask of the piece (non-background pixels) and tight-
   crop the piece to that mask.
2. ``cv2.matchTemplate(board, piece, TM_CCORR_NORMED, mask=fg)`` finds where
   the piece sits on the board. CCORR alone is permissive on flat regions, so
   we also run ``TM_CCOEFF_NORMED`` (mean-subtracted) on the tight crop and
   blend the two scores at each candidate.
3. Take the top-N non-max-suppressed peaks, pick the best blended score, and
   convert its board pixel position into a grid ``CellAddress``.

The returned ``MatchResult`` keeps the same shape the rest of the system
expects (``cell`` / ``combined`` / ``margin`` / ``rejected_reason``).
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from puzzle_assistant.config import Settings
from puzzle_assistant.matching.ensemble import MatchResult
from puzzle_assistant.reference.target_map import TargetMap
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.coords import CellAddress

_FG_THRESHOLD = 35.0


def match_piece(
    piece_bgr: np.ndarray,
    target_map: TargetMap,
    settings: Settings,
) -> MatchResult:
    """Localize ``piece_bgr`` on the reference board and return its cell."""

    started = time.monotonic()
    result = _match(piece_bgr, target_map, settings)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    plog.event(
        "match",
        level=logging.INFO,
        cell=[result.cell.row, result.cell.col] if result.cell else None,
        combined=round(result.combined, 3),
        margin=round(result.margin, 3),
        quality=target_map.quality,
        elapsed_ms=round(elapsed_ms, 1),
        rejected_reason=result.rejected_reason,
    )
    return result


def _match(
    piece_bgr: np.ndarray,
    target_map: TargetMap,
    settings: Settings,
) -> MatchResult:
    if piece_bgr.size == 0:
        return MatchResult(cell=None, combined=0.0, margin=0.0, rejected_reason="empty_piece")

    board = target_map.board_image
    bh, bw = board.shape[:2]

    # Foreground mask + tight crop to the piece's real silhouette. If the mask
    # comes back nearly empty (e.g. a blue sky piece that the background
    # heuristic mistakes for desk), fall back to treating the whole crop as
    # foreground rather than rejecting — the CCOEFF signal still localizes it.
    fg = _foreground_mask(piece_bgr)
    if float(fg.mean()) < 12.0:
        fg = np.full(piece_bgr.shape[:2], 255, dtype=np.uint8)
    cols_any = np.where(fg.max(axis=0) > 0)[0]
    rows_any = np.where(fg.max(axis=1) > 0)[0]
    if len(cols_any) < 4 or len(rows_any) < 4:
        return MatchResult(cell=None, combined=0.0, margin=0.0, rejected_reason="empty_fg")
    cx0, cx1 = int(cols_any[0]), int(cols_any[-1]) + 1
    cy0, cy1 = int(rows_any[0]), int(rows_any[-1]) + 1
    piece = piece_bgr[cy0:cy1, cx0:cx1]
    fg = fg[cy0:cy1, cx0:cx1]
    ph, pw = piece.shape[:2]

    if pw > bw or ph > bh or pw < 8 or ph < 8:
        return MatchResult(cell=None, combined=0.0, margin=0.0, rejected_reason="bad_piece_size")

    # Masked CCORR — primary localizer.
    try:
        ccorr = cv2.matchTemplate(board, piece, cv2.TM_CCORR_NORMED, mask=fg)
        ccorr = np.nan_to_num(ccorr, nan=0.0)
        ccorr = np.clip(ccorr, 0.0, 1.0)
    except cv2.error:
        return MatchResult(cell=None, combined=0.0, margin=0.0, rejected_reason="ccorr_failed")

    # CCOEFF on the same crop — mean-subtracted, far better at discriminating
    # the true location on a textured board.
    try:
        ccoeff = cv2.matchTemplate(board, piece, cv2.TM_CCOEFF_NORMED)
        ccoeff = np.clip(ccoeff, 0.0, 1.0)
    except cv2.error:
        ccoeff = None

    candidates = _top_n_candidates(ccorr, pw, ph, n=12, min_score=0.0)
    if not candidates:
        return MatchResult(cell=None, combined=0.0, margin=0.0, rejected_reason="no_candidate")

    scored: list[tuple[int, int, float]] = []
    for x, y, ccorr_score in candidates:
        cc = 0.0
        if ccoeff is not None and 0 <= y < ccoeff.shape[0] and 0 <= x < ccoeff.shape[1]:
            cc = float(ccoeff[y, x])
        combined = 0.55 * cc + 0.45 * ccorr_score
        scored.append((x, y, combined))

    scored.sort(key=lambda s: s[2], reverse=True)
    best_x, best_y, best_combined = scored[0]
    second = scored[1][2] if len(scored) > 1 else 0.0
    margin = best_combined - second

    min_combined = (
        settings.min_combined_score
        if target_map.quality == "primary"
        else settings.fallback_min_combined_score
    )
    min_margin = (
        settings.min_margin
        if target_map.quality == "primary"
        else settings.fallback_min_margin
    )

    if best_combined < min_combined:
        return MatchResult(
            cell=None, combined=best_combined, margin=margin, rejected_reason="low_score"
        )
    if margin < min_margin:
        return MatchResult(
            cell=None, combined=best_combined, margin=margin, rejected_reason="low_margin"
        )

    # Piece center on the board → grid cell.
    center_x = best_x + pw / 2
    center_y = best_y + ph / 2
    col = int(min(target_map.grid.cols - 1, max(0, center_x // target_map.grid.cell_w)))
    row = int(min(target_map.grid.rows - 1, max(0, center_y // target_map.grid.cell_h)))
    return MatchResult(
        cell=CellAddress(row=row, col=col),
        combined=best_combined,
        margin=margin,
        rejected_reason=None,
    )


def _foreground_mask(piece_bgr: np.ndarray) -> np.ndarray:
    """Non-background mask via corner-sampled L2 distance, with HSV fallback.

    Adapted from Puzzle_Game's ``_compute_fg_mask``: sample the four corners
    (which a tight cursor crop almost always fills with desk/board), estimate
    the background colour, and threshold pixels by distance from it.
    """

    h, w = piece_bgr.shape[:2]
    if h < 10 or w < 10:
        return np.ones((h, w), dtype=np.uint8) * 255

    cs = max(4, min(h, w) // 10)
    corners = np.vstack([
        piece_bgr[:cs, :cs].reshape(-1, 3),
        piece_bgr[:cs, w - cs:].reshape(-1, 3),
        piece_bgr[h - cs:, :cs].reshape(-1, 3),
        piece_bgr[h - cs:, w - cs:].reshape(-1, 3),
    ]).astype(np.float32)
    corner_std = float(corners.std(axis=0).mean())
    if corner_std < 30.0:
        bg = np.median(corners, axis=0)
        diff = piece_bgr.astype(np.float32) - bg
        dist = np.sqrt((diff ** 2).sum(axis=2))
        fg = (dist >= _FG_THRESHOLD).astype(np.uint8) * 255
        if 3 < float(fg.mean()) < 252:
            return _clean(fg)

    # Fallback: blue-dominant pixels are background.
    b = piece_bgr[:, :, 0].astype(np.int16)
    r = piece_bgr[:, :, 2].astype(np.int16)
    is_bg = ((b - r) > 20) & (b > 90)
    fg = (~is_bg).astype(np.uint8) * 255
    return _clean(fg)


def _clean(mask: np.ndarray) -> np.ndarray:
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k3)
    return out


def _top_n_candidates(
    result_map: np.ndarray, piece_w: int, piece_h: int, n: int, min_score: float
) -> list[tuple[int, int, float]]:
    """Non-max-suppressed top-N peaks of a score map: ``[(x, y, score), ...]``."""

    min_dist = max(piece_w, piece_h) // 2
    out: list[tuple[int, int, float]] = []
    work = result_map.copy()
    for _ in range(n):
        _, max_val, _, max_loc = cv2.minMaxLoc(work)
        if float(max_val) < min_score:
            break
        x, y = int(max_loc[0]), int(max_loc[1])
        out.append((x, y, float(max_val)))
        x1 = max(0, x - min_dist)
        y1 = max(0, y - min_dist)
        x2 = min(work.shape[1], x + min_dist)
        y2 = min(work.shape[0], y + min_dist)
        work[y1:y2, x1:x2] = 0.0
    return out
