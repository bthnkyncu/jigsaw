"""Flat-edge detection + border constraint.

Synthetic silhouettes prove the precision-safe contract: straight edges are
recognised as flat (→ board border), tabbed/blank edges are not, a clipped side
is never trusted, and the border filter only ever *removes* candidates.
"""

from __future__ import annotations

import numpy as np

from puzzle_assistant.config import load_settings
from puzzle_assistant.matching.engine import _detect_flat_edges, _edge_is_flat, _on_border
from puzzle_assistant.utils.coords import GridSpec

_NO_CLIP = (False, False, False, False)


def _piece(top: str, bottom: str, left: str, right: str) -> np.ndarray:
    """Build a 100×100 silhouette with each edge 'flat' | 'tab' | 'blank'.

    The body is a centred 60×60 square (rows/cols 20:80); a tab adds a bump
    outside it, a blank carves a notch inside it, flat leaves the body edge.
    """
    fg = np.zeros((100, 100), dtype=np.uint8)
    fg[20:80, 20:80] = 255
    mid = slice(40, 60)  # central third of an edge
    if top == "tab":
        fg[5:20, mid] = 255
    elif top == "blank":
        fg[20:35, mid] = 0
    if bottom == "tab":
        fg[80:95, mid] = 255
    elif bottom == "blank":
        fg[65:80, mid] = 0
    if left == "tab":
        fg[mid, 5:20] = 255
    elif left == "blank":
        fg[mid, 20:35] = 0
    if right == "tab":
        fg[mid, 80:95] = 255
    elif right == "blank":
        fg[mid, 65:80] = 0
    return fg


def test_corner_piece_has_two_flat_edges() -> None:
    settings = load_settings(None)
    fg = _piece(top="flat", left="flat", bottom="tab", right="tab")
    assert _detect_flat_edges(fg, _NO_CLIP, settings) == (True, False, True, False)


def test_edge_piece_has_one_flat_edge() -> None:
    settings = load_settings(None)
    fg = _piece(top="flat", bottom="tab", left="tab", right="blank")
    top, bottom, left, right = _detect_flat_edges(fg, _NO_CLIP, settings)
    assert top is True
    assert bottom is False and left is False and right is False


def test_interior_piece_has_no_flat_edge() -> None:
    settings = load_settings(None)
    fg = _piece(top="tab", bottom="blank", left="tab", right="blank")
    assert _detect_flat_edges(fg, _NO_CLIP, settings) == (False, False, False, False)


def test_blank_edge_is_not_flat() -> None:
    # A blank (inward notch) must not be mistaken for a straight border.
    assert _edge_is_flat(_piece("blank", "tab", "tab", "tab"), "top", 0.12) is False


def test_clipped_side_is_never_flat() -> None:
    settings = load_settings(None)
    fg = _piece(top="flat", bottom="tab", left="tab", right="tab")
    # Top genuinely flat, but if the crop clipped the top we must not trust it.
    clipped_top = (True, False, False, False)
    assert _detect_flat_edges(fg, clipped_top, settings) == (False, False, False, False)


def test_on_border_filters_to_implied_border() -> None:
    grid = GridSpec(cols=18, rows=15, cell_w=40.0, cell_h=40.0)

    class _TM:
        pass

    tm = _TM()
    tm.grid = grid  # type: ignore[attr-defined]

    flat_top = (True, False, False, False)
    assert _on_border((0, 7), flat_top, tm) is True       # row 0 → allowed
    assert _on_border((5, 7), flat_top, tm) is False      # interior → dropped

    corner = (True, False, True, False)  # flat top + flat left
    assert _on_border((0, 0), corner, tm) is True
    assert _on_border((0, 5), corner, tm) is False        # right row but wrong col
    assert _on_border((14, 0), corner, tm) is False
