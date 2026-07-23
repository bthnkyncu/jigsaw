"""Endgame joint assignment — the solver, and the gate that keeps it honest.

The gate is the point of these tests. Joint assignment amplifies whatever is in
the cost matrix: measured on real games it turns 81 % into 100 % where there is
signal, and 21 % into 7 % where there is not. So "the solver returned a
permutation" must never be sufficient to draw an overlay.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from puzzle_assistant.config import load_settings
from puzzle_assistant.matching.assignment import solve_for

CELLS = [(0, 0), (0, 1), (1, 0)]


def _settings(**kw):  # type: ignore[no-untyped-def]
    return dataclasses.replace(load_settings(None), **kw)


def test_swaps_a_piece_off_its_own_best_cell() -> None:
    """The whole point: one piece yields its favourite to a piece that needs it.

    Piece 0 scores best on cell 0, but piece 1 fits cell 0 far better and has
    nowhere else to go. The only consistent arrangement moves piece 0 to cell 1
    — an answer per-piece scoring cannot reach.
    """
    scores = np.array([
        [0.70, 0.65, 0.10],
        [0.95, 0.20, 0.10],
        [0.10, 0.10, 0.90],
    ])
    st = _settings(assignment_min_regret=0.0)
    assert solve_for(scores, 0, st)[0] == 1
    assert solve_for(scores, 1, st)[0] == 0
    assert solve_for(scores, 2, st)[0] == 2


def test_regret_is_large_when_the_pairing_is_forced() -> None:
    scores = np.array([
        [0.95, 0.10, 0.10],
        [0.10, 0.95, 0.10],
        [0.10, 0.10, 0.95],
    ])
    st = _settings(assignment_min_regret=0.0)
    _cell, regret = solve_for(scores, 0, st)
    assert regret > 0.5, f"a forced pairing must show a large regret, got {regret}"


def test_regret_collapses_when_every_piece_fits_everywhere() -> None:
    """The near-uniform-image case, which the gate exists to refuse.

    All pieces score alike on all cells, so the solver still returns a
    permutation — a confident-looking answer built on nothing. Regret is what
    exposes it: swapping costs almost no total score.
    """
    rng = np.random.default_rng(0)
    scores = 0.60 + rng.normal(0, 0.004, (4, 4))
    st = _settings(assignment_min_regret=0.0)
    _cell, regret = solve_for(scores, 0, st)
    assert regret < 0.05, f"an arbitrary permutation must show near-zero regret, got {regret}"


def test_refuses_a_single_piece_or_a_single_cell() -> None:
    """One piece, or one cell, means no competition — the sink from
    ``tests/test_filter_sink.py`` reached by yet another route."""
    st = _settings(assignment_min_regret=0.0)
    assert solve_for(np.array([[0.9, 0.1, 0.1]]), 0, st) is None
    assert solve_for(np.array([[0.9], [0.2], [0.1]]), 0, st) is None


def test_refuses_when_too_many_cells_are_open() -> None:
    """Early game the constraint is vacuous — N pieces for a hundred holes says
    nothing, and solving it just invents pairings."""
    rng = np.random.default_rng(1)
    scores = rng.random((3, 40))
    st = _settings(assignment_max_cells=14, assignment_min_regret=0.0)
    assert solve_for(scores, 0, st) is None


def _piece(seed: int, h: int = 60, w: int = 58) -> np.ndarray:
    import cv2
    rng = np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0, 255, (h, w, 3), dtype=np.uint8), (5, 5), 0)


# The buffer stores whatever the caller hands it alongside the image; these
# tests exercise identity and expiry, which do not touch the scoring payload.
_ANY_PREPARED = object()


def test_buffer_holds_each_physical_piece_once() -> None:
    """A piece picked up twice must not be held twice.

    The player cycles through the same unplaced pieces, so without this the
    buffer fills with copies of one piece and the solver is asked to put it in
    several cells at once — a contradiction it cannot express, and it satisfies
    it by displacing the pieces that were right.
    """
    from puzzle_assistant.matching.assignment import UnplacedPieces

    buf = UnplacedPieces(load_settings(None))
    a, b = _piece(1), _piece(2)
    buf.remember(a, _ANY_PREPARED)
    buf.remember(b, _ANY_PREPARED)
    assert len(buf) == 2

    # same piece again, with the slight jitter a re-segmentation gives it
    again = a.copy()
    again[0, 0] = (0, 0, 0)
    buf.remember(again, _ANY_PREPARED)
    assert len(buf) == 2, "the same piece was remembered twice"


def test_buffer_drops_a_piece_once_it_is_placed() -> None:
    from puzzle_assistant.matching.assignment import UnplacedPieces

    buf = UnplacedPieces(load_settings(None))
    a, b = _piece(3), _piece(4)
    buf.remember(a, _ANY_PREPARED)
    buf.remember(b, _ANY_PREPARED)
    buf.forget(a)
    assert len(buf) == 1
    assert buf.images()[0] is b


def test_buffer_forgets_stale_pieces() -> None:
    """Insurance for a piece placed without the matcher noticing."""
    from puzzle_assistant.matching.assignment import UnplacedPieces

    st = _settings(assignment_forget_after=3)
    buf = UnplacedPieces(st)
    buf.remember(_piece(5), _ANY_PREPARED)
    for _ in range(4):
        buf.tick()
    assert len(buf) == 0


def test_refuses_when_the_buffer_covers_too_few_of_the_open_cells() -> None:
    """The constraint has to actually bite.

    Measured in a full live simulation the buffer normally holds ~3 pieces
    against ~22 open cells, and there the joint solution is no better than
    per-piece scoring (18 of 67 versus 16) while looking just as confident. The
    win only exists once the player is stuck and the remaining pieces account
    for most of the remaining holes.
    """
    rng = np.random.default_rng(2)
    scores = rng.random((3, 12))          # coverage 0.25
    st = _settings(assignment_min_regret=0.0, assignment_min_coverage=0.6)
    assert solve_for(scores, 0, st) is None

    scores = rng.random((8, 12))          # coverage 0.67
    assert solve_for(scores, 0, st) is not None


def test_the_shipped_regret_floor_rejects_a_shaky_solution() -> None:
    """Pins the gate to what was measured, not to what looks reasonable.

    Over 68 piece-decisions from nine stuck endgames, correct decisions had
    median regret 0.116 and wrong ones 0.023. The default floor must sit above
    the wrong ones' p90 (0.066), and a matrix whose pieces barely disagree must
    be refused outright.
    """
    st = load_settings(None)
    assert st.assignment_min_regret >= 0.10, "gate weakened below the measured error band"

    rng = np.random.default_rng(3)
    mushy = 0.60 + rng.normal(0, 0.01, (6, 8))
    from puzzle_assistant.matching.assignment import assign_cell
    solved = solve_for(mushy, 0, st)
    if solved is not None:
        _cell, regret = solved
        assert regret < st.assignment_min_regret, (
            "a near-uniform cost matrix must not clear the regret floor"
        )
    assert assign_cell is not None  # imported for the wiring test below
