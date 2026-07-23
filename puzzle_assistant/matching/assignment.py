"""Endgame: decide the last pieces jointly instead of one at a time.

Scoring each piece on its own throws away the one constraint that is free at
the end of a game — every open cell takes exactly one piece. Once only a
handful are left, that turns the problem into a bipartite assignment, and
solving it as a whole can place a piece whose own best guess is wrong, because
every *other* piece fits that cell better.

Measured on the tails of eight recorded games (greedy per-piece versus the
joint solution): 81 % -> 100 % on v9, 85 % -> 100 % on v17, 95 % -> 100 % on
v15, 94 % -> 100 % on v6. This is the "last few pieces the assistant can never
show me" complaint, and on ordinary puzzles it closes it.

**The same measurement is why the confidence gate below is mandatory.** Joint
assignment amplifies whatever is in the cost matrix. Given signal it reaches
100 %; given noise it produces a globally consistent but wrong permutation and
destroys even the cells the per-piece answer had right — on a near-uniform leaf
image it went 21 % -> 7 %. So a solution is only trusted where the numbers say
the pairing is forced, never merely because the solver returned one.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from puzzle_assistant.config import Settings
from puzzle_assistant.matching.engine import Prepared
from puzzle_assistant.reference.target_map import TargetMap
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.coords import CellAddress

_BLOCKED = -1e3


class UnplacedPieces:
    """The pieces the player has picked up and the matcher could not place.

    Assignment needs several pieces at once, and there is only ever one in hand,
    so they have to be remembered across pickups. No extra segmentation is
    needed: the endgame is spent cycling through the same few pieces and the
    normal pickup path already sees each of them.

    Two things must not happen. The same physical piece must not be held twice,
    or the solver is asked to put it in two cells at once — hence the appearance
    check in :meth:`_index_of`. And a piece that has since been placed must not
    linger, or it competes for a cell it no longer needs — so an entry is
    dropped as soon as the matcher succeeds on it, and in any case after
    ``assignment_forget_after`` pickups.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._items: list[dict[str, object]] = []
        self._clock = 0

    def __len__(self) -> int:
        return len(self._items)

    def images(self) -> list[np.ndarray]:
        return [it["image"] for it in self._items]  # type: ignore[misc]

    def prepared(self) -> list[Prepared]:
        return [it["prepared"] for it in self._items]  # type: ignore[misc]

    def tick(self) -> None:
        """Advance the pickup clock and forget stale entries."""
        self._clock += 1
        keep = self._settings.assignment_forget_after
        self._items = [
            it for it in self._items if self._clock - int(it["seen"]) <= keep  # type: ignore[call-overload]
        ]

    def _index_of(self, piece_bgr: np.ndarray) -> int | None:
        for i, it in enumerate(self._items):
            if _looks_like(piece_bgr, it["image"], self._settings):  # type: ignore[arg-type]
                return i
        return None

    def remember(self, piece_bgr: np.ndarray, prepared: Prepared) -> int:
        """Add (or refresh) a piece; returns its index.

        ``prepared`` is kept because it is the expensive half of scoring and it
        does not change while the piece waits — only the piece in hand has to be
        prepared on each pickup, not the whole buffer. It is tied to the current
        reference, so :meth:`clear` must be called if calibration is redone.
        """
        entry = {"image": piece_bgr, "prepared": prepared, "seen": self._clock}
        found = self._index_of(piece_bgr)
        if found is not None:
            self._items[found] = entry
            return found
        self._items.append(entry)
        if len(self._items) > self._settings.assignment_buffer_max:
            self._items.pop(0)
        return len(self._items) - 1

    def forget(self, piece_bgr: np.ndarray) -> None:
        """Drop a piece — the matcher placed it, so it is no longer waiting."""
        found = self._index_of(piece_bgr)
        if found is not None:
            self._items.pop(found)

    def clear(self) -> None:
        self._items.clear()


def _looks_like(a: np.ndarray, b: np.ndarray, settings: Settings) -> bool:
    """Are these two crops the same physical piece, picked up twice?"""
    if a.size == 0 or b.size == 0:
        return False
    if abs(a.shape[0] - b.shape[0]) > 7 or abs(a.shape[1] - b.shape[1]) > 7:
        return False
    x = cv2.resize(a, (32, 32)).astype(np.float32)
    y = cv2.resize(b, (32, 32)).astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    y = (y - y.mean()) / (y.std() + 1e-6)
    return float((x * y).mean()) > settings.assignment_same_piece_min_corr


def score_matrix(
    prepared: list[Prepared], cells: list[tuple[int, int]], target_map: TargetMap
) -> np.ndarray:
    """``S[i][j]`` — how well piece *i* fits cell *j*."""
    grid = target_map.grid
    out = np.zeros((len(prepared), len(cells)), dtype=np.float64)
    for i, prep in enumerate(prepared):
        for j, (row, col) in enumerate(cells):
            hit = prep.best_in_cell(row, col, grid)
            if hit is not None:
                out[i, j] = hit[0]
    return out


def solve_for(
    scores: np.ndarray, index: int, settings: Settings
) -> tuple[int, float] | None:
    """Cell for piece ``index`` under the best joint assignment, plus its regret.

    ``regret`` is how much total score the whole solution loses if this piece is
    forbidden from the cell it was given. It is the honest confidence measure
    here: a large value means no other arrangement comes close, so the pairing
    is forced by the ensemble rather than by one shaky score. Returns ``None``
    when the problem is too small or too big to be worth solving.
    """
    n_pieces, n_cells = scores.shape
    # With one piece or one cell there is no competition to exploit — the
    # "assignment" is just the per-piece answer wearing a hat, and a single
    # mis-read cell would swallow every piece. Refuse.
    if n_pieces < 2 or n_cells < 2:
        return None
    if n_cells > settings.assignment_max_cells:
        return None
    # Too few pieces for the number of holes and the constraint is vacuous: each
    # piece can go almost anywhere without crowding the others, so the solver
    # returns per-piece answers dressed as a joint solution.
    if n_pieces < settings.assignment_min_coverage * n_cells:
        return None

    rows, cols = linear_sum_assignment(-scores)
    order = {int(r): int(c) for r, c in zip(rows, cols, strict=True)}
    if index not in order:
        return None
    j = order[index]

    blocked = scores.copy()
    blocked[index, j] = _BLOCKED
    b_rows, b_cols = linear_sum_assignment(-blocked)
    regret = float(scores[rows, cols].sum() - blocked[b_rows, b_cols].sum())
    return j, regret


def assign_cell(
    prepared: list[Prepared],
    index: int,
    cells: list[tuple[int, int]],
    target_map: TargetMap,
    settings: Settings,
) -> CellAddress | None:
    """Cell for ``prepared[index]``, or ``None`` if the solution is not forced.

    Logs the regret even when it refuses. The gate was set from reconstructed
    endgames, and it fires rarely by design — without a record of the near
    misses a live session that shows no overlay is indistinguishable from one
    where the feature never ran at all, and there would be nothing to retune on.
    """
    if not cells or index >= len(prepared):
        return None
    scores = score_matrix(prepared, cells, target_map)
    solved = solve_for(scores, index, settings)
    if solved is None:
        return None
    j, regret = solved
    row, col = cells[j]
    accepted = regret >= settings.assignment_min_regret
    plog.event(
        "assignment_considered",
        level=logging.INFO,
        held=len(prepared),
        open_cells=len(cells),
        coverage=round(len(prepared) / len(cells), 2),
        regret=round(regret, 4),
        floor=settings.assignment_min_regret,
        cell=[row, col],
        accepted=accepted,
    )
    if not accepted:
        return None
    return CellAddress(row=row, col=col)
