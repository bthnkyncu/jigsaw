"""The candidate filters must not decide the answer on their own.

The empty-cell and flat-edge filters only *remove* candidates, which looks
harmless — but removing the runner-up also destroys the quantity the margin gate
measures. A lone survivor ends up with ``margin == combined``, which passes every
gate, so whichever cell survives is accepted no matter how weakly it matches.

Live this produced a sink: board-state mis-read one occupied cell as empty, that
cell became the only survivor for piece after piece, and four different pieces
were all predicted onto it (combined 0.42-0.48, margin == combined each time).
"""

from __future__ import annotations

import cv2
import numpy as np

from puzzle_assistant.config import load_settings
from puzzle_assistant.matching.engine import match_piece
from puzzle_assistant.piece.board_state import BoardState
from puzzle_assistant.reference.target_map import build_from_init_view
from puzzle_assistant.utils.coords import GridSpec

_ROWS, _COLS, _CELL = 4, 4, 64


def _board_with_weak_twin() -> np.ndarray:
    """A board where cell (3,3) is a degraded copy of cell (1,1).

    (3,3) therefore shows up as a real but clearly weaker candidate for a piece
    cut from (1,1) — the shape the sink needs.
    """
    rng = np.random.default_rng(7)
    board = (rng.random((_ROWS * _CELL, _COLS * _CELL, 3)) * 255).astype(np.uint8)
    board = cv2.GaussianBlur(board, (9, 9), 0)
    src = board[_CELL:2 * _CELL, _CELL:2 * _CELL].astype(np.float32)
    noisy = np.clip(src + rng.normal(0, 45, src.shape), 0, 255)
    board[3 * _CELL:4 * _CELL, 3 * _CELL:4 * _CELL] = noisy.astype(np.uint8)
    return board


def _grid() -> GridSpec:
    return GridSpec(cols=_COLS, rows=_ROWS, cell_w=_CELL, cell_h=_CELL)


def test_filter_left_alone_survivor_is_not_trusted() -> None:
    """If the filters discard a much stronger candidate, reject.

    Here board-state claims every cell except (3,3) is occupied, so the correct
    candidate (1,1) is filtered out and only the weak (3,3) remains. Accepting it
    would be the sink; the piece must be rejected instead.
    """
    settings = load_settings(None)
    board = _board_with_weak_twin()
    grid = _grid()
    target_map = build_from_init_view(board, grid, settings)
    piece = board[_CELL:2 * _CELL, _CELL:2 * _CELL].copy()

    board_state = BoardState(grid)
    for row in range(_ROWS):
        for col in range(_COLS):
            board_state.set_filled(row, col, (row, col) != (3, 3))

    result = match_piece(piece, target_map, settings, board_state)
    assert result.cell is None, (
        f"piece funnelled into the only cell left by filtering: {result.cell}"
    )


def test_unfiltered_match_still_finds_the_right_cell() -> None:
    """Control: with no board-state the same piece localises correctly.

    Proves the rejection above comes from the filter guard, not from the piece
    being unmatchable.
    """
    settings = load_settings(None)
    board = _board_with_weak_twin()
    grid = _grid()
    target_map = build_from_init_view(board, grid, settings)
    piece = board[_CELL:2 * _CELL, _CELL:2 * _CELL].copy()

    result = match_piece(piece, target_map, settings, None)
    assert result.cell is not None
    assert (result.cell.row, result.cell.col) == (1, 1)


def test_occupied_twin_removal_still_works() -> None:
    """The filter's legitimate rescue must survive the guard.

    When the *true* cell is empty and its lookalike twin is genuinely occupied,
    removing the twin is exactly what the empty-cell filter is for. The two score
    alike, so the discarded lead stays small and the match is still accepted.
    """
    settings = load_settings(None)
    board = _board_with_weak_twin()
    grid = _grid()
    target_map = build_from_init_view(board, grid, settings)
    piece = board[_CELL:2 * _CELL, _CELL:2 * _CELL].copy()

    board_state = BoardState(grid)
    board_state.set_filled(3, 3, True)  # the weak twin is occupied; (1,1) is free

    result = match_piece(piece, target_map, settings, board_state)
    assert result.cell is not None, "legitimate twin removal must not be blocked"
    assert (result.cell.row, result.cell.col) == (1, 1)
