#!/usr/bin/env python3
"""De-risk probe: can we read piece tab/blank SHAPE from the init-view at the
known (count-anchored) grid boundaries — especially in LOW-TEXTURE cells?

For each init-view fixture it builds the exact grid, then for every internal
cell boundary measures:
  (1) cut-line detectability — edge density in the boundary band vs the cell
      interior (ratio > 1 ⇒ the cut line stands out as a real feature);
  (2) bulge sign + confidence — the tab bumps ~±20% of cell size across the
      boundary; we find the cut-line's signed offset at the edge midpoint and
      how cleanly it reads.
Results are split by cell texture (low vs high) so we see whether shape survives
exactly where appearance fails. Verdict guides: build the shape filter (lines
readable) vs fall back to live-neighbour interlocking (lines vanish low-texture).

    python scripts/probe_shape.py [fixtures_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puzzle_assistant.calibration.board_detector import detect_board
from puzzle_assistant.calibration.grid_detector import detect_grid_from_init_view
from puzzle_assistant.config import load_settings

CASES = [("init_150.png", 150), ("init_100.png", 100), ("init_70.png", 70),
         ("init_50.png", 50), ("init_30.png", 30)]


def _profile_prominence(
    mag: np.ndarray, center: float, a_lo: int, a_hi: int, span: int, vertical: bool
) -> tuple[float, float]:
    """Average the gradient magnitude over the apex band into a 1-D profile
    across the boundary, then return (signed offset of its peak from
    ``center``, peak prominence = (peak-median)/(peak+median)).

    Averaging over the apex band keeps the vertically-coherent cut line and
    washes out incoherent content edges, so prominence is high only when a real
    boundary line sits there."""
    a_lo, a_hi = max(0, a_lo), a_hi
    lo, hi = max(0, int(center - span)), int(center + span)
    if hi - lo < 5 or a_hi - a_lo < 2:
        return 0.0, 0.0
    block = mag[a_lo:a_hi, lo:hi] if vertical else mag[lo:hi, a_lo:a_hi]
    if block.size == 0:
        return 0.0, 0.0
    prof = block.mean(axis=0) if vertical else block.mean(axis=1)
    if prof.size < 3:
        return 0.0, 0.0
    peak_i = int(np.argmax(prof))
    peak, med = float(prof[peak_i]), float(np.median(prof))
    prom = (peak - med) / (peak + med + 1e-6)
    return (lo + peak_i) - center, prom


def main(fixtures_dir: str) -> int:
    settings = load_settings(None)
    base = Path(fixtures_dir)
    for name, nominal in CASES:
        path = base / name
        if not path.exists():
            print(f"{name}: yok"); continue
        img = cv2.imread(str(path))
        bb = detect_board(img, settings)
        if bb is None:
            print(f"{name}: board yok"); continue
        crop = img[bb.y:bb.y + bb.h, bb.x:bb.x + bb.w]
        settings.target_piece_count = nominal
        grid = detect_grid_from_init_view(crop, settings)
        if grid is None:
            print(f"{name}: grid yok"); continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        cw, ch = grid.cell_w, grid.cell_h

        # Per-cell interior texture → low/high bucket (median split).
        tex = np.zeros((grid.rows, grid.cols))
        for r in range(grid.rows):
            for c in range(grid.cols):
                y0, y1 = int((r + 0.2) * ch), int((r + 0.8) * ch)
                x0, x1 = int((c + 0.2) * cw), int((c + 0.8) * cw)
                tex[r, c] = gray[y0:y1, x0:x1].std()
        tex_med = float(np.median(tex))

        # For each internal boundary: gradient-profile prominence AT the
        # boundary vs at a mid-cell control (no boundary). If the cut line is a
        # real feature, boundary prominence >> control. Win = boundary clearly
        # beats its own control. Split by cell texture.
        span = int(0.32 * min(cw, ch))
        b_lo: list[float] = []   # boundary prominence, low-texture cells
        b_hi: list[float] = []
        win_lo = win_hi = 0
        for r in range(grid.rows):
            for c in range(grid.cols):
                low = tex[r, c] < tex_med
                if c < grid.cols - 1:  # vertical boundary right of (r,c)
                    ay0, ay1 = int((r + 0.42) * ch), int((r + 0.58) * ch)
                    _, bp = _profile_prominence(gx, (c + 1) * cw, ay0, ay1, span, True)
                    _, cp = _profile_prominence(gx, (c + 0.5) * cw, ay0, ay1, span, True)
                    (b_lo if low else b_hi).append(bp)
                    if bp > cp + 0.05:
                        win_lo += low; win_hi += not low
                if r < grid.rows - 1:  # horizontal boundary below (r,c)
                    ax0, ax1 = int((c + 0.42) * cw), int((c + 0.58) * cw)
                    _, bp = _profile_prominence(gy, (r + 1) * ch, ax0, ax1, span, False)
                    _, cp = _profile_prominence(gy, (r + 0.5) * ch, ax0, ax1, span, False)
                    (b_lo if low else b_hi).append(bp)
                    if bp > cp + 0.05:
                        win_lo += low; win_hi += not low

        n_lo, n_hi = max(1, len(b_lo)), max(1, len(b_hi))
        print(
            f"{name:14s} {grid.rows}x{grid.cols}  "
            f"LOW-tex: prom_med={np.median(b_lo):.2f} beats_control={100*win_lo/n_lo:.0f}%  | "
            f"HIGH-tex: prom_med={np.median(b_hi):.2f} beats_control={100*win_hi/n_hi:.0f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures"))
