#!/usr/bin/env python3
"""Verify parametric grid detection across piece counts.

Drop the init-view screenshots (the assembled image shown at game start) into
a folder under the expected names below, then run::

    python scripts/verify_grid.py [fixtures_dir]   # default: tests/fixtures

For each image it runs the real calibration path (detect_board → crop →
detect_grid_from_init_view) and prints the measured rows × cols, cell size and
cell aspect against the customer-provided ground truth. The grid must be
orientation-correct (cols > rows on a landscape board) and cells ~square.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puzzle_assistant.calibration.board_detector import detect_board
from puzzle_assistant.calibration.grid_detector import detect_grid_from_init_view
from puzzle_assistant.config import load_settings

# (filename, nominal_count_the_user_types, expected_rows, expected_cols).
# The nominal count is what the game shows / the user enters before "Başlat";
# the actual total may drift a few pieces (e.g. "100" → 8×12 = 96).
CASES: list[tuple[str, int, int, int]] = [
    ("init_150.png", 150, 10, 15),
    ("init_100.png", 100, 8, 12),
    ("init_70.png", 70, 7, 10),
    ("init_50.png", 50, 6, 8),
    ("init_30.png", 30, 5, 6),
]


def main(fixtures_dir: str) -> int:
    base = Path(fixtures_dir)
    any_run = False
    for name, nominal, exp_rows, exp_cols in CASES:
        settings = load_settings(None)
        settings.target_piece_count = nominal  # simulate the user's entry
        path = base / name
        if not path.exists():
            print(f"{name:14s} : (yok — atlandı)")
            continue
        any_run = True
        img = cv2.imread(str(path))
        if img is None:
            print(f"{name:14s} : okunamadı")
            continue
        bbox = detect_board(img, settings)
        if bbox is None:
            print(f"{name:14s} : board tespit edilemedi")
            continue
        crop = img[bbox.y:bbox.y + bbox.h, bbox.x:bbox.x + bbox.w]
        grid = detect_grid_from_init_view(crop, settings)
        if grid is None:
            print(f"{name:14s} : GRID YOK  (beklenen {exp_rows}x{exp_cols})")
            continue
        aspect = grid.cell_w / grid.cell_h
        match = grid.rows == exp_rows and grid.cols == exp_cols
        tag = "OK" if match else f"≠ (±{abs(grid.rows-exp_rows)}r/{abs(grid.cols-exp_cols)}c)"
        print(
            f"{name:14s} : ölçülen {grid.rows:2d}x{grid.cols:2d}  "
            f"(beklenen {exp_rows}x{exp_cols})  "
            f"hücre {grid.cell_w:.0f}x{grid.cell_h:.0f}px  aspect {aspect:.2f}  {tag}"
        )
    if not any_run:
        print(f"\nHiç görsel bulunamadı. PNG'leri {base}/ altına şu adlarla koy:")
        for name, r, c in CASES:
            print(f"  {name}  ({r}x{c})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures"))
