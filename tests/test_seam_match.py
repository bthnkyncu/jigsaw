"""Seam tie-breaker — live-neighbour continuity.

Synthetic scenarios prove the precision-safe contract: a repeated-texture tie
is resolved towards the candidate whose placed neighbour's edge colour-continues
into the piece, and is left unresolved (``None``) when the only neighbour
mismatches or there is no neighbour evidence at all.
"""

from __future__ import annotations

import numpy as np

from puzzle_assistant.config import load_settings
from puzzle_assistant.matching.seam_match import seam_break_tie
from puzzle_assistant.piece.board_state import BoardState
from puzzle_assistant.utils.coords import GridSpec

_A = (50, 100, 150)  # piece / matching-neighbour colour (BGR)
_B = (200, 40, 40)   # a clearly different region colour (BGR)
_LIGHT = (255, 255, 255)  # bare board light → empty cell


def _grid() -> GridSpec:
    return GridSpec(cols=4, rows=4, cell_w=50.0, cell_h=50.0)


def _board_with_neighbour(colour: tuple[int, int, int]) -> np.ndarray:
    """200×200 board crop: all cells empty (light) except (1,0) holding ``colour``."""
    crop = np.full((200, 200, 3), _LIGHT, dtype=np.uint8)
    crop[50:100, 0:50] = colour  # cell (row=1, col=0) is "placed"
    return crop


def _solid_piece(colour: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    piece = np.full((60, 60, 3), colour, dtype=np.uint8)
    fg = np.full((60, 60), 255, dtype=np.uint8)
    return piece, fg


def test_seam_resolves_towards_continuous_neighbour() -> None:
    settings = load_settings(None)
    state = BoardState(_grid())
    state.update(_board_with_neighbour(_A), settings)
    assert state.is_filled(1, 0)  # the neighbour is recognised as placed

    piece, fg = _solid_piece(_A)
    # Two tied candidates: cell (1,1) has the continuous left neighbour (1,0);
    # cell (3,3) sits in untouched territory (no placed neighbours).
    tie = [(60, 60, 0.62, 1, 1), (180, 180, 0.60, 3, 3)]
    winner = seam_break_tie(piece, fg, tie, state, settings)
    assert winner is not None
    wx, wy, _wcombined, seam_score = winner
    assert (wx, wy) == (60, 60)  # the (1,1) candidate won
    assert seam_score >= settings.seam_min_score


def test_seam_abstains_when_neighbour_mismatches() -> None:
    settings = load_settings(None)
    state = BoardState(_grid())
    # The only placed neighbour is a different region's colour.
    state.update(_board_with_neighbour(_B), settings)

    piece, fg = _solid_piece(_A)
    tie = [(60, 60, 0.62, 1, 1), (180, 180, 0.60, 3, 3)]
    # No candidate has continuous evidence → keep the (precision-safe) rejection.
    assert seam_break_tie(piece, fg, tie, state, settings) is None


def test_seam_abstains_without_any_neighbour() -> None:
    settings = load_settings(None)
    state = BoardState(_grid())
    state.update(np.full((200, 200, 3), _LIGHT, dtype=np.uint8), settings)

    piece, fg = _solid_piece(_A)
    tie = [(100, 100, 0.62, 2, 2), (180, 180, 0.60, 3, 3)]
    assert seam_break_tie(piece, fg, tie, state, settings) is None


def test_live_cell_returns_neighbour_pixels() -> None:
    state = BoardState(_grid())
    state.update(_board_with_neighbour(_A), load_settings(None))
    cell = state.live_cell(1, 0)
    assert cell is not None
    # The cell crop is dominated by colour _A.
    assert tuple(int(v) for v in cell.reshape(-1, 3).mean(axis=0).round()) == _A
    assert state.live_cell(99, 99) is None  # out of range
