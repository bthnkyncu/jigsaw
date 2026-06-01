"""Capture pickups to disk for offline labeling (``--record`` mode).

Each valid single-piece pickup writes a self-contained sample under
``captures/pickup_NNNN/``:

* ``piece.png``    — the segmented piece crop (``piece_full``)
* ``board.png``    — the reference board image the matcher localizes against
* ``live.png``     — the live game board crop at pickup time (with already
                     placed pieces), so the eval harness can derive which
                     cells were filled and A/B-test the board-state filter
* ``meta.json``    — grid spec, cursor, predicted cell, so the labeler can
                     render the board grid and the operator can mark the truth

The labeling tool (``scripts/label_pickups.py``) consumes these and emits
``tests/fixtures/labeled_pickups.json`` with the human-marked correct cell.
This is deliberately decoupled from the matcher: recording observes, it never
influences a prediction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.coords import CellAddress, GridSpec


class PickupRecorder:
    def __init__(self, out_dir: Path) -> None:
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._count = self._next_index()

    def _next_index(self) -> int:
        existing = sorted(self._out_dir.glob("pickup_*"))
        if not existing:
            return 0
        last = existing[-1].name.removeprefix("pickup_")
        try:
            return int(last) + 1
        except ValueError:
            return len(existing)

    def record(
        self,
        piece_bgr: np.ndarray,
        board_bgr: np.ndarray,
        grid: GridSpec,
        cursor_window: tuple[int, int],
        predicted_cell: CellAddress | None,
        live_board_bgr: np.ndarray | None = None,
    ) -> None:
        sample_dir = self._out_dir / f"pickup_{self._count:04d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(sample_dir / "piece.png"), piece_bgr)
        cv2.imwrite(str(sample_dir / "board.png"), board_bgr)
        if live_board_bgr is not None and live_board_bgr.size:
            cv2.imwrite(str(sample_dir / "live.png"), live_board_bgr)
        meta: dict[str, Any] = {
            "grid": {
                "cols": grid.cols,
                "rows": grid.rows,
                "cell_w": grid.cell_w,
                "cell_h": grid.cell_h,
            },
            "cursor_window": list(cursor_window),
            "predicted_cell": (
                [predicted_cell.row, predicted_cell.col] if predicted_cell else None
            ),
        }
        (sample_dir / "meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        plog.event("pickup_recorded", index=self._count, dir=str(sample_dir))
        self._count += 1
