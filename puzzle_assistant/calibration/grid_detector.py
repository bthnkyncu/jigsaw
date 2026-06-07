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


def detect_grid_from_init_view(board_bgr: np.ndarray, settings: Settings) -> GridSpec | None:
    """Return the discovered grid, or ``None`` on failure."""

    if board_bgr.size == 0 or board_bgr.ndim != 3:
        return None
    h, w = board_bgr.shape[:2]

    gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 25, 75)

    horiz_profile = edges.sum(axis=1).astype(np.float32)  # per-row → rows axis
    vert_profile = edges.sum(axis=0).astype(np.float32)   # per-col → cols axis

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
