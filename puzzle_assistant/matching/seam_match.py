"""Neighbour-seam continuity score (spec §2.2).

When a candidate empty cell has already-placed neighbours, the dragged piece —
if it truly belongs there — must continue smoothly into them: the colours and
gradients across the shared edge line up. This is decisive for flat / single-
colour pieces that template and colour can't localize (dozens of board cells
look alike, but only the true cell's *neighbours* continue the piece's edges).

The signal is sampled from the **live board** (real placed-piece pixels held by
``BoardState``), not the reference board. No rotation in this game, so the
piece's top edge maps to the cell's top edge directly.

Returns ``None`` when the cell has no filled neighbour — there is no seam to
judge, so the caller must not penalize the candidate; it falls back to the
other signals.
"""

from __future__ import annotations

import cv2
import numpy as np

from puzzle_assistant.config import Settings
from puzzle_assistant.piece.board_state import BoardState

# (dr, dc, piece_edge, neighbour_edge) for the 4 sides. ``piece_edge`` is the
# strip of the piece touching the neighbour; ``neighbour_edge`` is the strip of
# the neighbour touching the piece.
_SIDES = [
    (-1, 0, "top", "bottom"),
    (1, 0, "bottom", "top"),
    (0, -1, "left", "right"),
    (0, 1, "right", "left"),
]


def seam_score(
    piece_cell_bgr: np.ndarray,
    row: int,
    col: int,
    board_state: BoardState,
    settings: Settings,
) -> float | None:
    """Mean edge-continuity score over the candidate cell's filled neighbours.

    ``piece_cell_bgr`` must already be resized to one cell. Score is in [0, 1]
    (1 = perfect continuity). ``None`` when the cell has fewer than
    ``seam_min_filled_neighbours`` placed neighbours — too few to trust, and a
    lone accidental neighbour on the wrong side of the board would otherwise
    inflate a wrong position and collapse the margin.
    """

    if board_state.filled_neighbour_count(row, col) < settings.seam_min_filled_neighbours:
        return None

    scores: list[float] = []
    for dr, dc, p_edge, n_edge in _SIDES:
        nr, nc = row + dr, col + dc
        if not board_state.is_filled(nr, nc):
            continue
        neighbour = board_state.cell_crop(nr, nc)
        if neighbour is None or neighbour.size == 0:
            continue
        nb = cv2.resize(
            neighbour, (piece_cell_bgr.shape[1], piece_cell_bgr.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        s = _edge_continuity(piece_cell_bgr, nb, p_edge, n_edge, settings)
        if s is not None:
            scores.append(s)
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def _strip(img: np.ndarray, edge: str, px: int) -> np.ndarray:
    if edge == "top":
        return img[:px, :]
    if edge == "bottom":
        return img[-px:, :]
    if edge == "left":
        return img[:, :px]
    return img[:, -px:]  # right


def _edge_continuity(
    piece: np.ndarray,
    neighbour: np.ndarray,
    p_edge: str,
    n_edge: str,
    settings: Settings,
) -> float | None:
    px = max(2, settings.seam_strip_px)
    ps = _strip(piece, p_edge, px)
    ns = _strip(neighbour, n_edge, px)
    if ps.size == 0 or ns.size == 0 or ps.shape != ns.shape:
        return None
    # Compare the two strips in Lab; the seam is continuous when the mean
    # colour across it barely changes. ~30 Lab units of difference halves the
    # score (same decay the colour signal uses elsewhere).
    p_lab = cv2.cvtColor(ps, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)
    n_lab = cv2.cvtColor(ns, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)
    dist = float(np.linalg.norm(p_lab - n_lab))
    return max(0.0, 1.0 - dist / 60.0)
