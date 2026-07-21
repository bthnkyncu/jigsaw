#!/usr/bin/env python3
"""Render a montage of recorded eval samples for spot-checking the auto-labels.

Consumes the samples written by ``PUZZLE_RECORD_DIR`` capture (see
``utils/recorder.record_eval_sample``). Each row shows, side by side:

    PIECE | actual-cell patch | predicted-cell patch

The *actual* cell is the physical landing (board-state diff / drop point) — the
ground-truth label we want to trust. Eyeball that the PIECE matches the
actual-cell patch; if it doesn't, that sample's label is bad and should be
dropped before measuring/tuning. Also flags predicted!=actual (matcher misses).

Usage:
    python scripts/verify_captures.py captures_mario/ \
        --out /tmp/montage.png [--only-wrong]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _cell_patch(board: np.ndarray, grid: dict, rc: list[int] | None) -> np.ndarray:
    if rc is None:
        return np.full((80, 80, 3), 60, np.uint8)
    r, c = rc
    cw, ch = grid["cell_w"], grid["cell_h"]
    y0, y1 = int(r * ch), int((r + 1) * ch)
    x0, x1 = int(c * cw), int((c + 1) * cw)
    patch = board[y0:y1, x0:x1]
    if patch.size == 0:
        return np.full((80, 80, 3), 60, np.uint8)
    return cv2.resize(patch, (80, 80))


def _norm(im: np.ndarray) -> np.ndarray:
    if im is None or im.size == 0:
        return np.zeros((80, 80, 3), np.uint8)
    return cv2.resize(im, (80, 80))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("captures_dir")
    ap.add_argument("--out", default="/tmp/montage.png")
    ap.add_argument("--only-wrong", action="store_true",
                    help="only show samples where predicted != actual")
    args = ap.parse_args()

    root = Path(args.captures_dir)
    samples = sorted(root.glob("pickup_*"))
    rows = []
    n_total = n_wrong = n_drop = 0
    for s in samples:
        meta_f = s / "meta.json"
        if not meta_f.exists():
            continue
        meta = json.loads(meta_f.read_text())
        piece = cv2.imread(str(s / "piece.png"))
        board = cv2.imread(str(s / "board.png"))
        if piece is None or board is None:
            continue
        grid = meta["grid"]
        actual = meta.get("actual_cell")
        pred = meta.get("predicted_cell")
        src = meta.get("actual_source", "?")
        n_total += 1
        wrong = pred != actual
        if wrong:
            n_wrong += 1
        if src == "drop":
            n_drop += 1
        if args.only_wrong and not wrong:
            continue
        strip = np.hstack([
            _norm(piece),
            _cell_patch(board, grid, actual),
            _cell_patch(board, grid, pred),
        ])
        bar = np.full((22, strip.shape[1], 3), 40, np.uint8)
        tag = "WRONG" if wrong else "ok"
        txt = f"{s.name} PIECE|act{actual}({src})|pred{pred} {tag}"
        color = (0, 0, 255) if wrong else (200, 200, 200)
        cv2.putText(bar, txt, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        rows.append(np.vstack([bar, strip]))

    if not rows:
        print("no samples to render")
        return
    # Chunk into columns so a long run stays viewable.
    per_col = 20
    cols = [np.vstack(rows[i:i + per_col]) for i in range(0, len(rows), per_col)]
    max_h = max(c.shape[0] for c in cols)
    cols = [np.vstack([c, np.full((max_h - c.shape[0], c.shape[1], 3), 20, np.uint8)])
            for c in cols]
    montage = np.hstack(cols)
    cv2.imwrite(args.out, montage)
    print(f"samples={n_total}  matcher-wrong={n_wrong} "
          f"(precision {100*(n_total-n_wrong)/max(n_total,1):.1f}%)  "
          f"drop-labeled={n_drop}")
    print(f"montage -> {args.out}  (cols: PIECE | actual-cell | predicted-cell)")


if __name__ == "__main__":
    main()
