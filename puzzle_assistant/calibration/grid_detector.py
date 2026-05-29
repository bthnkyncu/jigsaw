"""Estimate the puzzle grid (cols × rows) from the init-view board crop.

The implementation here is adapted from the Puzzle_Game prototype because
its autocorrelation-based period search held up well on real Gamyun init
views, while a peak-count or frequency-bucket approach is sensitive to
content-driven noise.

Algorithm:

1. Take the board crop, run Canny so the puzzle cut-lines become a clean
   high-contrast signal.
2. Project edges onto each axis. A grid with cell period ``p`` produces an
   axis profile whose autocorrelation peaks at lag ``p``.
3. Refine each grid line by snapping to the local maximum within ``±p/6``.
4. ``cols = len(col_lines)``, ``rows = len(row_lines)``.

Fallback (init view missed): aspect-ratio search using the reference panel
shape — implemented as ``estimate_grid_from_aspect``.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from puzzle_assistant.config import Settings
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.coords import GridSpec

# Minimum cell size in pixels. A 250-piece board on a ~700px tall window
# yields ~40px cells; using 25 keeps a margin without inviting sub-cell
# harmonics that produce 40x grids.
_MIN_PERIOD_PX = 25
_MAX_PERIOD_PX = 120


def detect_grid_from_init_view(board_bgr: np.ndarray, settings: Settings) -> GridSpec | None:
    """Return the discovered grid, or ``None`` on failure."""

    if board_bgr.size == 0 or board_bgr.ndim != 3:
        return None
    h, w = board_bgr.shape[:2]

    gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 25, 75)

    horiz_profile = edges.sum(axis=1).astype(np.float32)
    vert_profile = edges.sum(axis=0).astype(np.float32)

    # Bound the period search by the configured piece-count band. A square
    # board with N pieces has roughly ``sqrt(N)`` cells per axis, so cell size
    # falls in ``[axis_size / sqrt(max_pieces), axis_size / sqrt(min_pieces)]``.
    # This rules out content-harmonics — e.g. a puzzle photo with horizontal
    # bands that make every ~25px look periodic when the real grid sits at 47px.
    min_pieces = settings.expected_piece_count_min
    max_pieces = settings.expected_piece_count_max
    # Per-axis upper bound on cell count: sqrt(max_pieces * aspect_factor).
    # Allow a generous 1.6x factor so a thin board (cols >> rows) still fits.
    aspect_slack = 1.6
    max_cells_per_axis = math.sqrt(max_pieces * aspect_slack)
    min_cells_per_axis = math.sqrt(min_pieces / aspect_slack)
    row_min_p = max(20, int(h / max_cells_per_axis))
    row_max_p = min(200, int(h / max(1, min_cells_per_axis)))
    col_min_p = max(20, int(w / max_cells_per_axis))
    col_max_p = min(200, int(w / max(1, min_cells_per_axis)))

    row_period = _autocorrelation_period(horiz_profile, row_min_p, row_max_p)
    col_period = _autocorrelation_period(vert_profile, col_min_p, col_max_p)

    if row_period is None or col_period is None:
        plog.event("grid_detect_no_period", level=logging.WARNING)
        return None

    # Cell count is derived from the period directly. ``_grid_lines`` is
    # used for diagnostics / future cell-snapping work but can drop the
    # board-edge line depending on phase, so counting it gives unstable
    # results (rows might come out one short).
    rows = max(1, round(h / row_period))
    cols = max(1, round(w / col_period))
    total = rows * cols
    if not (settings.expected_piece_count_min <= total <= settings.expected_piece_count_max):
        plog.event(
            "grid_count_out_of_range",
            level=logging.WARNING,
            rows=rows,
            cols=cols,
            total=total,
            min=settings.expected_piece_count_min,
            max=settings.expected_piece_count_max,
            col_period=round(col_period, 2),
            row_period=round(row_period, 2),
        )
        return None

    spec = GridSpec(cols=cols, rows=rows, cell_w=w / cols, cell_h=h / rows)
    plog.event(
        "grid_detect_ok",
        rows=rows,
        cols=cols,
        total=total,
        col_period=round(col_period, 2),
        row_period=round(row_period, 2),
    )
    return spec


def estimate_grid_from_aspect(
    board_w: int,
    board_h: int,
    settings: Settings,
) -> GridSpec | None:
    """Pick the (cols, rows) pair closest to the board's aspect ratio
    inside ``[expected_piece_count_min, expected_piece_count_max]``.

    Used when the init-view path failed and we only have the reference panel.
    """

    if board_w <= 0 or board_h <= 0:
        return None
    target_aspect = board_w / board_h

    best: tuple[int, int] | None = None
    best_score = float("inf")
    for total in range(settings.expected_piece_count_min, settings.expected_piece_count_max + 1):
        for cols in range(8, total // 4 + 1):
            if total % cols != 0:
                continue
            rows = total // cols
            cell_aspect = (board_w / cols) / (board_h / rows)
            score = abs(cell_aspect - 1.0) + 0.1 * abs((cols / rows) - target_aspect)
            if score < best_score:
                best_score = score
                best = (cols, rows)

    if best is None:
        return None
    cols, rows = best
    spec = GridSpec(cols=cols, rows=rows, cell_w=board_w / cols, cell_h=board_h / rows)
    plog.event("grid_estimate_ok", rows=rows, cols=cols, total=cols * rows)
    return spec


def _autocorrelation_period(
    profile: np.ndarray,
    min_period: int = _MIN_PERIOD_PX,
    max_period: int = _MAX_PERIOD_PX,
) -> float | None:
    """Find the dominant lag of ``profile`` using FFT-based autocorrelation.

    Returns the lag (in pixels) where the autocorrelation peaks within
    ``[min_period, max_period]``, or ``None`` if the peak is too weak.
    """

    n = profile.size
    if n < min_period * 2:
        return None

    centered = profile - profile.mean()
    n_pad = 2 * n
    fft_vals = np.fft.rfft(centered, n=n_pad)
    acf = np.fft.irfft(fft_vals * np.conj(fft_vals))[:n].real
    if acf[0] < 1e-9:
        return None
    acf = acf / acf[0]

    search_end = min(max_period + 1, n // 2)
    if search_end <= min_period:
        return None

    best_lag = int(np.argmax(acf[min_period:search_end])) + min_period
    if acf[best_lag] < 0.08:
        return None
    return float(best_lag)


def _grid_lines(profile: np.ndarray, period: float, size: int) -> list[int]:
    """Place grid lines at the optimal phase offset for ``period``, then
    snap each one to the nearest local maximum within ``±period/6``.
    """

    step = max(1, round(period))
    half = max(3, step // 6)

    best_offset = 0
    best_score = -1.0
    for offset in range(step):
        score = 0.0
        pos = offset
        while pos < size:
            score += float(profile[pos])
            pos += step
        if score > best_score:
            best_score = score
            best_offset = offset

    lines: list[int] = []
    pos = best_offset
    while pos < size:
        lo = max(0, pos - half)
        hi = min(size, pos + half + 1)
        if hi - lo >= 2:
            peak = lo + int(np.argmax(profile[lo:hi]))
            lines.append(peak)
        pos += step
    return lines
