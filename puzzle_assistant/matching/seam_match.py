"""Break repeated-texture ties using live-board neighbour continuity.

The bouquet image repeats, so a dragged piece often scores almost identically
at two or more distant board positions: high ``combined`` but near-zero
``margin`` → ``low_margin`` rejection. The board-state filter resolves this once
a twin is *filled*, but when both twins are still empty appearance alone cannot
decide.

The discriminator is the *live* board. The player builds outward from placed
pieces, so the true cell usually has already-placed neighbours whose edge
content the dragged piece continues; the wrong twin either sits in untouched
territory (no neighbours) or against a different region (colour mismatch across
the seam). We therefore compare the piece's facing edge band to each filled
neighbour cell's facing edge band (segmented Lab means, foreground-masked) and
pick the candidate whose seam genuinely continues.

This is deliberately conservative: a candidate needs real neighbour evidence
*and* enough continuity (``seam_min_score``), and when several tied candidates
have evidence the winner must beat the rest by ``seam_min_margin``. Otherwise we
return ``None`` and the caller keeps the precision-preserving rejection.
"""

from __future__ import annotations

import cv2
import numpy as np

from puzzle_assistant.config import Settings
from puzzle_assistant.piece.board_state import BoardState

# (row delta, col delta) of the neighbour on each side, and the neighbour's
# edge that faces the piece's edge.
_NEIGHBOURS: dict[str, tuple[int, int, str]] = {
    "left": (0, -1, "right"),
    "right": (0, 1, "left"),
    "top": (-1, 0, "bottom"),
    "bottom": (1, 0, "top"),
}

# A Lab gap of this many units across the seam halves the continuity score.
_LAB_HALF = 40.0


def seam_break_tie(
    piece_bgr: np.ndarray,
    fg: np.ndarray,
    tie: list[tuple[int, int, float, int, int]],
    board_state: BoardState,
    settings: Settings,
) -> tuple[int, int, float, float] | None:
    """Pick the tied candidate whose seam continues into its placed neighbours.

    ``tie`` is ``[(x, y, combined, row, col), ...]`` (distinct cells, already
    within the tie band). Returns ``(x, y, combined, seam_score)`` of the
    winner, or ``None`` when there is no clear neighbour-backed evidence.
    """

    frac = settings.seam_edge_frac
    segments = settings.seam_segments

    scored: list[tuple[int, int, float, float]] = []
    for x, y, combined, row, col in tie:
        continuities: list[float] = []
        for side, (dr, dc, opp) in _NEIGHBOURS.items():
            nr, nc = row + dr, col + dc
            if not board_state.is_filled(nr, nc):
                continue
            neighbour = board_state.live_cell(nr, nc)
            if neighbour is None or neighbour.size == 0:
                continue
            piece_desc = _edge_descriptor(piece_bgr, fg, side, frac, segments)
            nb_desc = _edge_descriptor(neighbour, None, opp, frac, segments)
            cont = _continuity(piece_desc, nb_desc)
            if cont is not None:
                continuities.append(cont)
        if continuities:
            scored.append((x, y, combined, sum(continuities) / len(continuities)))

    if not scored:
        return None
    scored.sort(key=lambda s: s[3], reverse=True)
    best = scored[0]
    if best[3] < settings.seam_min_score:
        return None
    if len(scored) > 1 and (best[3] - scored[1][3]) < settings.seam_min_margin:
        return None
    return best


def _edge_band(img: np.ndarray, side: str, frac: float) -> np.ndarray:
    h, w = img.shape[:2]
    bw = max(1, round(w * frac))
    bh = max(1, round(h * frac))
    if side == "left":
        return img[:, :bw]
    if side == "right":
        return img[:, w - bw:]
    if side == "top":
        return img[:bh, :]
    return img[h - bh:, :]  # bottom


def _edge_descriptor(
    img: np.ndarray, fg: np.ndarray | None, side: str, frac: float, segments: int
) -> list[np.ndarray | None] | None:
    """Per-segment mean Lab colour along the facing edge band.

    The band is split into ``segments`` cells *along the seam* (rows for
    left/right, cols for top/bottom) so colour variation is captured, not just
    a single mean. Foreground-masked when ``fg`` is given; segments with too few
    foreground pixels become ``None`` so a tab/blank notch doesn't fabricate a
    colour.
    """
    band = _edge_band(img, side, frac)
    if band.size == 0:
        return None
    lab = cv2.cvtColor(band, cv2.COLOR_BGR2LAB).astype(np.float32)
    band_fg = _edge_band(fg, side, frac) if fg is not None else None

    along_rows = side in ("left", "right")
    length = lab.shape[0] if along_rows else lab.shape[1]
    n = max(1, min(segments, length))

    out: list[np.ndarray | None] = []
    for i in range(n):
        lo = int(i * length / n)
        hi = int((i + 1) * length / n)
        if along_rows:
            seg = lab[lo:hi, :, :]
            seg_fg = band_fg[lo:hi, :] if band_fg is not None else None
        else:
            seg = lab[:, lo:hi, :]
            seg_fg = band_fg[:, lo:hi] if band_fg is not None else None
        if seg_fg is not None:
            mask = seg_fg > 0
            if int(mask.sum()) < 3:
                out.append(None)
                continue
            out.append(seg[mask].mean(axis=0))
        else:
            out.append(seg.reshape(-1, 3).mean(axis=0))
    return out


def _continuity(
    piece_desc: list[np.ndarray | None] | None,
    nb_desc: list[np.ndarray | None] | None,
) -> float | None:
    """1.0 when the two edge descriptors match, decaying with mean Lab gap."""
    if piece_desc is None or nb_desc is None:
        return None
    pairs = [
        (a, b)
        for a, b in zip(piece_desc, nb_desc, strict=False)
        if a is not None and b is not None
    ]
    if not pairs:
        return None
    mean_dist = sum(float(np.linalg.norm(a - b)) for a, b in pairs) / len(pairs)
    return max(0.0, 1.0 - mean_dist / _LAB_HALF)
