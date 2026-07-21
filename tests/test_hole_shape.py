"""Endgame hole-shape rescue.

The rescue only speaks when appearance matching stayed silent, so these tests
pin the two behaviours that make that safe: it identifies the hole a piece fits
when one clearly stands out, and it declines whenever the evidence is thin.
"""

from __future__ import annotations

import cv2
import numpy as np

from puzzle_assistant.config import load_settings
from puzzle_assistant.matching.hole_shape import rescue_by_hole_shape
from puzzle_assistant.utils.coords import GridSpec

_ROWS, _COLS, _CELL = 6, 8, 50
_BOARD_LIGHT = (245, 235, 225)  # bare board shows through unfilled cells
_CONTENT = (60, 90, 40)         # a placed piece (dark, saturated)


def _grid() -> GridSpec:
    return GridSpec(cols=_COLS, rows=_ROWS, cell_w=_CELL, cell_h=_CELL)


def _piece_silhouette(tab: bool = True) -> np.ndarray:
    """A cell-sized blob whose right edge carries either a tab or a blank.

    Tab versus blank is the distinction that actually decides where a piece can
    go: a piece with a tab cannot enter a hole that also needs one there. Two
    shapes differing only in *which* edge holds the tab overlap ~92 %, which the
    rescue rightly treats as inconclusive.
    """
    mask = np.zeros((_CELL, _CELL), np.uint8)
    cv2.rectangle(mask, (6, 6), (_CELL - 7, _CELL - 7), 255, -1)
    bump = (_CELL - 6, _CELL // 2) if tab else (_CELL - 7, _CELL // 2)
    cv2.circle(mask, bump, 9, 255 if tab else 0, -1)
    return mask


def _board_with_holes(holes: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    """Filled board, with the given cells punched out to bare-board colour."""
    board = np.full((_ROWS * _CELL, _COLS * _CELL, 3), _CONTENT, np.uint8)
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 12, board.shape)
    board = np.clip(board.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    for (row, col), shape in holes.items():
        y0, x0 = row * _CELL, col * _CELL
        region = board[y0:y0 + _CELL, x0:x0 + _CELL]
        region[shape > 0] = _BOARD_LIGHT
    return board


def test_finds_the_hole_the_piece_fits() -> None:
    """One hole needs a tab, the other a blank — pick the one that fits."""
    settings = load_settings(None)
    piece = _piece_silhouette(tab=True)
    decoy = _piece_silhouette(tab=False)
    board = _board_with_holes({(2, 3): piece, (4, 6): decoy})

    cell = rescue_by_hole_shape(piece, board, _grid(), settings)
    assert cell is not None, "a clearly matching hole should be found"
    assert (cell.row, cell.col) == (2, 3)


def test_declines_when_too_much_of_the_board_is_empty() -> None:
    """Early game the holes merge and shape means nothing, so stay silent.

    Measured on real captures: above roughly 30 empty cells the shape signal is
    right 0 % of the time, so firing there would only produce wrong overlays.
    """
    settings = load_settings(None)
    piece = _piece_silhouette()
    holes = {
        (r, c): piece
        for r in range(_ROWS)
        for c in range(_COLS)
        if (r * _COLS + c) % 2 == 0
    }
    board = _board_with_holes(holes)

    assert rescue_by_hole_shape(piece, board, _grid(), settings) is None


def test_declines_when_two_holes_fit_equally_well() -> None:
    """No lead over the runner-up means no answer — a missing overlay beats a
    wrong one."""
    settings = load_settings(None)
    piece = _piece_silhouette()
    board = _board_with_holes({(1, 1): piece, (4, 5): piece})

    assert rescue_by_hole_shape(piece, board, _grid(), settings) is None


def test_declines_on_a_full_board() -> None:
    settings = load_settings(None)
    piece = _piece_silhouette()
    board = _board_with_holes({})

    assert rescue_by_hole_shape(piece, board, _grid(), settings) is None
