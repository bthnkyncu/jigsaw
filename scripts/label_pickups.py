#!/usr/bin/env python3
"""Label recorded pickups with their true target cell.

Reads the samples written by ``--record`` (see ``utils/recorder.py``) and, for
each one, shows the reference board with the grid drawn on it next to the
picked piece. You click the cell the piece truly belongs to (or press ``s`` to
skip an unclear one). The result is written to a JSON file the eval harness
consumes.

Usage:
    python scripts/label_pickups.py captures/ \
        --out tests/fixtures/labeled_pickups.json

Keys while labeling:
    left-click  — mark the clicked cell as the correct one and advance
    s           — skip this pickup (not written)
    u           — undo: re-label the previous pickup
    q / Esc     — quit and save what's labeled so far
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

WINDOW = "label pickups  (click correct cell | s=skip u=undo q=quit)"


def _draw_grid(board: np.ndarray, cols: int, rows: int) -> np.ndarray:
    canvas = board.copy()
    h, w = canvas.shape[:2]
    cw, ch = w / cols, h / rows
    for c in range(1, cols):
        x = round(c * cw)
        cv2.line(canvas, (x, 0), (x, h), (0, 255, 0), 1)
    for r in range(1, rows):
        y = round(r * ch)
        cv2.line(canvas, (0, y), (w, y), (0, 255, 0), 1)
    return canvas


def _compose(board_grid: np.ndarray, piece: np.ndarray) -> np.ndarray:
    """Board on the left, the piece (scaled up) on the right."""
    bh = board_grid.shape[0]
    scale = min(4.0, bh / max(piece.shape[0], 1))
    pv = cv2.resize(
        piece, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST
    )
    panel = np.full((bh, max(pv.shape[1] + 20, 160), 3), 40, dtype=np.uint8)
    panel[10 : 10 + pv.shape[0], 10 : 10 + pv.shape[1]] = pv[
        : bh - 10, : panel.shape[1] - 10
    ]
    return np.hstack([board_grid, panel])


def _load_existing(out_path: Path) -> list[dict[str, Any]]:
    if out_path.exists():
        data = json.loads(out_path.read_text(encoding="utf-8"))
        return list(data) if isinstance(data, list) else []
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("captures_dir", type=Path)
    ap.add_argument(
        "--out", type=Path, default=Path("tests/fixtures/labeled_pickups.json")
    )
    args = ap.parse_args()

    samples = sorted(args.captures_dir.glob("pickup_*"))
    if not samples:
        print(f"No pickup_* dirs under {args.captures_dir}")
        return 1

    out_path: Path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled = _load_existing(out_path)
    already = {entry["sample"] for entry in labeled}

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    state: dict[str, Any] = {"click": None}

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    cv2.setMouseCallback(WINDOW, on_mouse)

    i = 0
    while i < len(samples):
        sample = samples[i]
        name = sample.name
        if name in already:
            i += 1
            continue
        meta = json.loads((sample / "meta.json").read_text(encoding="utf-8"))
        board = cv2.imread(str(sample / "board.png"))
        piece = cv2.imread(str(sample / "piece.png"))
        if board is None or piece is None:
            i += 1
            continue
        cols, rows = meta["grid"]["cols"], meta["grid"]["rows"]
        bh, bw = board.shape[:2]
        cw, ch = bw / cols, bh / rows
        board_grid = _draw_grid(board, cols, rows)
        view = _compose(board_grid, piece)

        state["click"] = None
        while True:
            shown = view.copy()
            cv2.putText(
                shown, f"{name}  ({i + 1}/{len(samples)})", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
            )
            cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(20) & 0xFF
            if state["click"] is not None:
                cx, cy = state["click"]
                if cx < bw:  # click landed on the board
                    col = int(min(cols - 1, max(0, cx // cw)))
                    row = int(min(rows - 1, max(0, cy // ch)))
                    labeled = [e for e in labeled if e["sample"] != name]
                    labeled.append({
                        "sample": name,
                        "dir": str(sample),
                        "grid": meta["grid"],
                        "correct_cell": [row, col],
                    })
                    out_path.write_text(
                        json.dumps(labeled, indent=2), encoding="utf-8"
                    )
                    print(f"{name}: ({row},{col})  [{len(labeled)} labeled]")
                    i += 1
                    break
                state["click"] = None
            if key in (ord("q"), 27):
                cv2.destroyAllWindows()
                print(f"Saved {len(labeled)} labels to {out_path}")
                return 0
            if key == ord("s"):
                print(f"{name}: skipped")
                i += 1
                break
            if key == ord("u") and i > 0:
                i = max(0, i - 1)
                prev = samples[i].name
                labeled = [e for e in labeled if e["sample"] != prev]
                already.discard(prev)
                break

    cv2.destroyAllWindows()
    out_path.write_text(json.dumps(labeled, indent=2), encoding="utf-8")
    print(f"Done. {len(labeled)} labels in {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
