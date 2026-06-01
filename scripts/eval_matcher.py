#!/usr/bin/env python3
"""Measure matcher precision / recall on a labeled pickup set.

Consumes ``labeled_pickups.json`` (produced by ``scripts/label_pickups.py``)
and replays each pickup through ``match_piece`` against the reference board it
was recorded with. Reports two configurations side by side:

* **baseline**     — the matcher as-is.
* **board-state**  — same matcher, but with the empty-cell filter active. The
                     filled[][] map is derived from the recorded ``live.png``
                     (the real game board at pickup time), so this is an honest
                     A/B, not a simulation.

Metrics (spec §4.2):
    precision = correct / predicted        (target: >= 0.97)
    recall    = predicted / total
plus a breakdown of rejections by reason.

Usage:
    python scripts/eval_matcher.py --labels tests/fixtures/labeled_pickups.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from puzzle_assistant.config import Settings, load_settings
from puzzle_assistant.matching.engine import match_piece
from puzzle_assistant.piece.board_state import BoardState
from puzzle_assistant.reference.target_map import build_from_init_view
from puzzle_assistant.utils.coords import GridSpec


@dataclass
class Tally:
    total: int = 0
    predicted: int = 0
    correct: int = 0
    rejections: dict[str, int] = field(default_factory=dict)

    def add(self, accepted: bool, is_correct: bool, reason: str | None) -> None:
        self.total += 1
        if accepted:
            self.predicted += 1
            if is_correct:
                self.correct += 1
        else:
            key = reason or "unknown"
            self.rejections[key] = self.rejections.get(key, 0) + 1

    def report(self, name: str) -> str:
        prec = self.correct / self.predicted if self.predicted else 0.0
        recall = self.predicted / self.total if self.total else 0.0
        wrong = self.predicted - self.correct
        lines = [
            f"--- {name} ---",
            f"  total pickups : {self.total}",
            f"  predicted     : {self.predicted}  (recall {recall:.1%})",
            f"  correct       : {self.correct}",
            f"  wrong         : {wrong}",
            f"  precision     : {prec:.1%}   {'OK' if prec >= 0.97 else 'BELOW 97%'}",
        ]
        if self.rejections:
            lines.append("  rejections:")
            for reason, n in sorted(self.rejections.items()):
                lines.append(f"    {reason}: {n}")
        return "\n".join(lines)


def _grid_from_meta(meta_grid: dict[str, float]) -> GridSpec:
    return GridSpec(
        cols=int(meta_grid["cols"]),
        rows=int(meta_grid["rows"]),
        cell_w=float(meta_grid["cell_w"]),
        cell_h=float(meta_grid["cell_h"]),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--labels", type=Path, default=Path("tests/fixtures/labeled_pickups.json")
    )
    ap.add_argument("--config", type=Path, default=None)
    args = ap.parse_args()

    settings: Settings = load_settings(args.config)
    entries = json.loads(args.labels.read_text(encoding="utf-8"))
    if not entries:
        print("No labeled entries.")
        return 1

    baseline = Tally()
    with_state = Tally()

    for e in entries:
        sample = Path(e["dir"])
        piece = cv2.imread(str(sample / "piece.png"))
        board = cv2.imread(str(sample / "board.png"))
        if piece is None or board is None:
            continue
        grid = _grid_from_meta(e["grid"])
        truth = tuple(e["correct_cell"])  # [row, col]
        tmap = build_from_init_view(board, grid, settings)

        # Baseline.
        r0 = match_piece(piece, tmap, settings)
        ok0 = r0.cell is not None and (r0.cell.row, r0.cell.col) == truth
        baseline.add(r0.cell is not None, ok0, r0.rejected_reason)

        # Board-state: derive filled[][] from the live board crop if present.
        live = cv2.imread(str(sample / "live.png"))
        bs: BoardState | None = None
        if live is not None and live.size:
            bs = BoardState(grid)
            bs.update(live, settings)
        r1 = match_piece(piece, tmap, settings, bs)
        ok1 = r1.cell is not None and (r1.cell.row, r1.cell.col) == truth
        with_state.add(r1.cell is not None, ok1, r1.rejected_reason)

    print(baseline.report("baseline"))
    print()
    print(with_state.report("board-state"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
