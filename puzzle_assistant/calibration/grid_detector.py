"""Estimate the puzzle grid (cols × rows) from the init-view board crop.

Parametric — there is NO hardcoded piece count and no fixed count band. The
grid is read purely from the cut-line periodicity, so any piece count works
(30, 50, 100, 150, 250 …) and any aspect (landscape or portrait), regardless
of how far the player zoomed before starting.

Algorithm:

1. Canny the board crop so the cut-lines become a clean high-contrast signal.
2. Project edges onto each axis and take the FFT autocorrelation. A grid with
   cell period ``p`` peaks at lag ``p`` (and weaker peaks at its multiples).
   Collect the strongest candidate periods per axis within an *absolute*
   cell-size range (in board pixels) — not a piece-count-derived range.
3. Choose the (row_period, col_period) pair that (a) keeps every cell within
   the pixel range, (b) keeps cells ~square (``cell_w/cell_h`` within
   ``grid_cell_aspect_max``), and (c) has the strongest combined
   autocorrelation. The squareness constraint is the key robustness lever: a
   half/double-period harmonic makes one axis ~2× the other, so it is rejected
   — this is what previously doubled the row count on low-piece boards and
   produced a squashed overlay box.
4. ``rows = round(h / row_period)``, ``cols = round(w / col_period)``.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from puzzle_assistant.config import Settings
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.coords import GridSpec

# Autocorrelation peak must reach this normalised strength to count.
_ACF_MIN_STRENGTH = 0.08
# Candidate periods kept per axis (fundamental + a few harmonics/alternatives).
_TOP_K = 6
# Count-anchored tie-break weight: penalty per unit relative distance of a
# candidate's total from the user-entered count. Small, so real edge contrast
# decides and this only separates near-equal near-count factorisations.
_COUNT_CLOSENESS_W = 0.4


def detect_grid_from_init_view(board_bgr: np.ndarray, settings: Settings) -> GridSpec | None:
    """Return the discovered grid, or ``None`` on failure."""

    if board_bgr.size == 0 or board_bgr.ndim != 3:
        return None
    h, w = board_bgr.shape[:2]

    gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 25, 75)

    horiz_profile = edges.sum(axis=1).astype(np.float32)  # per-row → rows axis
    vert_profile = edges.sum(axis=0).astype(np.float32)   # per-col → cols axis

    # Count-anchored path: if the user told us the piece count, only consider
    # rows×cols near it and let the pixels pick the exact grid. This is the
    # robust path (no octave errors); the unanchored periodicity search below is
    # the fallback when no count is given.
    if settings.target_piece_count and settings.target_piece_count >= 9:
        return _grid_from_count(h, w, horiz_profile, vert_profile, settings)

    min_p = settings.grid_min_cell_px
    max_p = settings.grid_max_cell_px
    row_cands = _autocorrelation_candidates(horiz_profile, min_p, max_p)
    col_cands = _autocorrelation_candidates(vert_profile, min_p, max_p)
    if not row_cands or not col_cands:
        plog.event("grid_detect_no_period", level=logging.WARNING)
        return None

    aspect_max = settings.grid_cell_aspect_max
    best: tuple[float, int, int, float, float, float] | None = None
    for row_p, row_str in row_cands:
        rows = max(1, round(h / row_p))
        cell_h = h / rows
        for col_p, col_str in col_cands:
            cols = max(1, round(w / col_p))
            cell_w = w / cols
            if not (min_p <= cell_w <= max_p and min_p <= cell_h <= max_p):
                continue
            ratio = max(cell_w / cell_h, cell_h / cell_w)
            if ratio > aspect_max:  # not square enough → a harmonic
                continue
            total = rows * cols
            if total < 9 or total > settings.grid_max_total_pieces:
                continue
            strength = row_str + col_str
            if best is None or strength > best[0]:
                best = (strength, rows, cols, row_p, col_p, cell_w / cell_h)

    if best is None:
        plog.event(
            "grid_detect_no_square_pair",
            level=logging.WARNING,
            row_cands=[round(p, 1) for p, _ in row_cands],
            col_cands=[round(p, 1) for p, _ in col_cands],
        )
        return None

    _, rows, cols, row_p, col_p, aspect = best
    spec = GridSpec(cols=cols, rows=rows, cell_w=w / cols, cell_h=h / rows)
    plog.event(
        "grid_detect_ok",
        rows=rows,
        cols=cols,
        total=rows * cols,
        col_period=round(col_p, 2),
        row_period=round(row_p, 2),
        cell_aspect=round(aspect, 2),
    )
    return spec


def _grid_from_count(
    h: int, w: int, horiz: np.ndarray, vert: np.ndarray, settings: Settings
) -> GridSpec | None:
    """Pick the rows×cols whose total is near ``target_piece_count`` and whose
    implied cell periods best match the board's cut-line edges.

    The count window rules out octave errors (a half/double/third period changes
    the total far more than the tolerance), so among the few near-count
    factorisations the line/midpoint edge contrast cleanly selects the real grid.
    """
    target = settings.target_piece_count
    assert target is not None
    tol = settings.target_piece_count_tolerance
    lo_total = max(4, round(target * (1.0 - tol)))
    hi_total = round(target * (1.0 + tol))
    min_p = settings.grid_min_cell_px
    max_p = settings.grid_max_cell_px
    aspect_max = settings.grid_cell_aspect_max

    best: tuple[float, int, int] | None = None
    max_rows = max(2, int(h / min_p))
    max_cols = max(2, int(w / min_p))
    for rows in range(2, max_rows + 1):
        cell_h = h / rows
        if not (min_p * 0.8 <= cell_h <= max_p):
            continue
        for cols in range(2, max_cols + 1):
            total = rows * cols
            if not (lo_total <= total <= hi_total):
                continue
            cell_w = w / cols
            if not (min_p * 0.8 <= cell_w <= max_p):
                continue
            if max(cell_w / cell_h, cell_h / cell_w) > aspect_max:
                continue
            contrast = _axis_contrast(horiz, round(cell_h)) + _axis_contrast(vert, round(cell_w))
            # Tie-break toward the count the user actually entered: real cut-line
            # contrast dominates, but when two near-count factorisations score
            # alike this nudges toward the one matching the stated total.
            closeness = abs(total - target) / target
            # Squareness. A jigsaw cell is near-square, so an oblong candidate is
            # almost certainly a mis-read of the periodicity. On a pale, low
            # contrast image (a sea-and-sky photo) the cut lines barely register,
            # the contrast term goes noisy, and an oblong factorisation can win:
            # a 360x540 board came out 15x7 (cell 51x36, aspect 1.43) where the
            # true grid was 8x12 (45x45, aspect 1.00). The aspect gate alone
            # cannot fix that — it must stay loose enough for genuinely oblong
            # boards — so pay a price proportional to how far from square it is.
            aspect = max(cell_w / cell_h, cell_h / cell_w)
            score = (
                contrast
                - _COUNT_CLOSENESS_W * closeness
                - settings.grid_squareness_weight * (aspect - 1.0)
            )
            if best is None or score > best[0]:
                best = (score, rows, cols)

    if best is None or best[0] <= 0.0:
        plog.event(
            "grid_count_no_fit", level=logging.WARNING,
            target=target, lo=lo_total, hi=hi_total,
        )
        return None
    score, rows, cols = best
    spec = GridSpec(cols=cols, rows=rows, cell_w=w / cols, cell_h=h / rows)
    plog.event(
        "grid_detect_ok",
        rows=rows, cols=cols, total=rows * cols, target=target,
        cell_aspect=round((w / cols) / (h / rows), 2),
        score=round(score, 3),
    )
    return spec


def _axis_contrast(profile: np.ndarray, step: int) -> float:
    """Best-phase edge contrast between grid lines and their midpoints for cell
    period ``step``. High when ``step`` aligns with real cut-lines.
    """
    size = profile.size
    if step < 2 or size < step * 2:
        return 0.0
    best_sum = -1.0
    best_off = 0
    for off in range(step):
        s = float(profile[off::step].sum())
        if s > best_sum:
            best_sum = s
            best_off = off
    lines = profile[best_off::step]
    mids = profile[(best_off + step // 2) % step::step]
    if lines.size < 2 or mids.size == 0:
        return 0.0
    line_mean = float(lines.mean())
    mid_mean = float(mids.mean())
    return (line_mean - mid_mean) / (line_mean + mid_mean + 1e-6)


def estimate_grid_from_aspect(
    board_w: int,
    board_h: int,
    settings: Settings,
) -> GridSpec | None:
    """Deprecated fallback — intentionally returns ``None``.

    Aspect ratio alone cannot determine the piece count (6×8, 9×12 and 12×16
    are all equally square), so any answer here would be a hardcoded-count
    guess — exactly what this parametric design avoids. The periodicity path
    (:func:`detect_grid_from_init_view`) is authoritative; if it can't read the
    grid we wait for a cleaner init view rather than place a wrong grid.
    """
    plog.event("grid_aspect_fallback_disabled", level=logging.INFO)
    return None


def _autocorrelation_candidates(
    profile: np.ndarray, min_period: float, max_period: float
) -> list[tuple[float, float]]:
    """Strongest autocorrelation peaks of ``profile`` within the period range.

    Returns ``[(period_px, strength), ...]`` sorted by strength (descending),
    using FFT-based autocorrelation. Multiple candidates are returned so the
    caller can pick the fundamental (vs a harmonic) via the squareness rule.
    """

    n = profile.size
    lo = round(min_period)
    hi = round(max_period)
    if lo < 2 or n < lo * 2:
        return []

    centered = profile - profile.mean()
    fft_vals = np.fft.rfft(centered, n=2 * n)
    acf = np.fft.irfft(fft_vals * np.conj(fft_vals))[:n].real
    if acf[0] < 1e-9:
        return []
    acf = acf / acf[0]

    hi = min(hi, n // 2 - 1)
    if hi <= lo:
        return []

    cands: list[tuple[float, float]] = []
    for lag in range(lo, hi + 1):
        v = float(acf[lag])
        if v < _ACF_MIN_STRENGTH:
            continue
        if acf[lag] >= acf[lag - 1] and acf[lag] >= acf[lag + 1]:  # local max
            cands.append((float(lag), v))

    cands.sort(key=lambda c: c[1], reverse=True)
    return cands[:_TOP_K]
