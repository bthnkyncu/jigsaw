"""Border pieces — the reference must be padded, and filled cells never shown.

Both guards come out of the same 200-piece game. Border cells were 86 % of
everything the assistant failed to predict, and the one prediction it did make
late in that game pointed at a cell the player could see was already occupied.
"""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np

from puzzle_assistant.config import load_settings
from puzzle_assistant.matching.engine import match_piece
from puzzle_assistant.piece.board_state import BoardState
from puzzle_assistant.reference.target_map import build_from_init_view
from puzzle_assistant.utils.coords import GridSpec

CELL = 44
COLS, ROWS = 7, 6
OVERHANG = 7  # px of the piece crop that sit outside the board (shadow / outline)


def _board() -> np.ndarray:
    """A board whose every cell carries its own distinctive texture."""
    rng = np.random.default_rng(7)
    board = np.zeros((ROWS * CELL, COLS * CELL, 3), np.uint8)
    for r in range(ROWS):
        for c in range(COLS):
            patch = rng.integers(0, 255, (CELL, CELL, 3), dtype=np.uint8)
            patch = cv2.GaussianBlur(patch, (5, 5), 0)
            board[r * CELL:(r + 1) * CELL, c * CELL:(c + 1) * CELL] = patch
    return board


def _grid() -> GridSpec:
    return GridSpec(cols=COLS, rows=ROWS, cell_w=float(CELL), cell_h=float(CELL))


def _top_row_piece(board: np.ndarray, col: int) -> np.ndarray:
    """The piece for cell ``(0, col)``, carrying overhang above the board edge.

    A dragged piece is segmented off the desk together with its drop shadow and
    outline, so its crop is a few pixels larger than the footprint it occupies
    once placed. On an interior cell that is harmless — the match just slides.
    On the top row those pixels have to sit *above* the board, which the
    template match cannot do unless the reference is padded.
    """
    x0 = int(col * CELL - CELL * 0.25)
    piece = board[0:CELL + CELL // 4, x0:x0 + int(CELL * 1.5)].copy()
    return np.vstack([np.repeat(piece[:1], OVERHANG, axis=0), piece])


def _predict(piece: np.ndarray, board: np.ndarray, settings) -> tuple[int, int] | None:  # type: ignore[no-untyped-def]
    tmap = build_from_init_view(board, _grid(), settings)
    result = match_piece(piece, tmap, settings)
    return (result.cell.row, result.cell.col) if result.cell else None


def test_padding_recovers_a_top_row_piece() -> None:
    settings = load_settings(None)
    board = _board()
    col = 3
    piece = _top_row_piece(board, col)

    unpadded = dataclasses.replace(settings, board_match_pad_cells=0.0)
    assert _predict(piece, board, unpadded) != (0, col), (
        "fixture is not exercising the overhang — it already matched unpadded"
    )
    assert _predict(piece, board, settings) == (0, col), (
        "a top-row piece must be localisable once the reference is padded"
    )


def test_padding_leaves_interior_pieces_alone() -> None:
    """Padding must be free for interior pieces — they were never the problem."""
    settings = load_settings(None)
    board = _board()
    unpadded = dataclasses.replace(settings, board_match_pad_cells=0.0)
    for r, c in ((2, 3), (3, 1), (4, 4)):
        piece = board[
            int(r * CELL - CELL * 0.2):int((r + 1) * CELL + CELL * 0.2),
            int(c * CELL - CELL * 0.2):int((c + 1) * CELL + CELL * 0.2),
        ].copy()
        assert _predict(piece, board, unpadded) == (r, c)
        assert _predict(piece, board, settings) == (r, c)


def test_never_points_at_a_cell_known_to_be_filled() -> None:
    """When every candidate is occupied, say nothing.

    A piece can only land in an empty cell, so a prediction on a filled one is
    wrong by construction. The engine used to fall through unfiltered here on
    the theory that board-state might be stale — measured across 791
    ground-truth pickups it never once called the true cell filled, and the
    fall-through instead put a green box on an occupied cell again and again
    late in a 200-piece game.
    """
    settings = load_settings(None)
    board = _board()
    r, c = 2, 3
    piece = board[
        int(r * CELL - CELL * 0.2):int((r + 1) * CELL + CELL * 0.2),
        int(c * CELL - CELL * 0.2):int((c + 1) * CELL + CELL * 0.2),
    ].copy()

    state = BoardState(_grid())
    assert _predict(piece, board, settings) == (r, c)

    for rr in range(ROWS):
        for cc in range(COLS):
            state.set_filled(rr, cc, True)
    tmap = build_from_init_view(board, _grid(), settings)
    result = match_piece(piece, tmap, settings, state)
    assert result.cell is None, "must not point at a cell known to be occupied"
    assert result.rejected_reason == "all_filled"
