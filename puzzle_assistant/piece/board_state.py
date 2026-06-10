"""Track which grid cells are already filled with placed pieces.

A dragged piece can only go to an *empty* cell. On this puzzle the image
repeats (a bouquet of similar flowers), so a piece's appearance often matches
two or more distant board positions equally well — the appearance score ties
and the match is rejected on margin. But by mid/late game one of those twin
positions is usually already filled; dropping filled candidates leaves the
true empty cell alone and the margin recovers. This is the precision-safe way
to rescue those repeated-texture pieces: it only removes occupied positions,
never invents a match.

Empty-cell test: the light board surface shows through an empty cell, so a
cell dominated by the board-light HSV range (the same range board detection
uses) is empty; once a piece sits there it is full of image content and the
light fraction collapses. Live boards show a clean bimodal split (empty cells
~1.0 light fraction, filled ~0.0), so a mid threshold classifies robustly.
"""

from __future__ import annotations

import cv2
import numpy as np

from puzzle_assistant.config import Settings
from puzzle_assistant.utils.coords import GridSpec


class BoardState:
    """Maintains a ``filled[r][c]`` matrix from successive board crops."""

    def __init__(self, grid: GridSpec) -> None:
        self._grid = grid
        self._filled: list[list[bool]] = [
            [False] * grid.cols for _ in range(grid.rows)
        ]

    def is_filled(self, row: int, col: int) -> bool:
        if 0 <= row < self._grid.rows and 0 <= col < self._grid.cols:
            return self._filled[row][col]
        return False

    def filled_count(self) -> int:
        return sum(v for row in self._filled for v in row)

    def filled_cells(self) -> frozenset[tuple[int, int]]:
        """Set of ``(row, col)`` currently marked filled — for before/after
        diffing to find where a dropped piece landed (self-supervised eval)."""
        return frozenset(
            (r, c)
            for r in range(self._grid.rows)
            for c in range(self._grid.cols)
            if self._filled[r][c]
        )

    def total_cells(self) -> int:
        return self._grid.rows * self._grid.cols

    def update(self, board_bgr: np.ndarray, settings: Settings) -> None:
        """Recompute the filled matrix from a fresh full-board crop."""

        if board_bgr.size == 0:
            return
        hsv = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)
        low = np.array(settings.board_light_hsv_low, dtype=np.uint8)
        high = np.array(settings.board_light_hsv_high, dtype=np.uint8)
        light = cv2.inRange(hsv, low, high)
        h, w = light.shape[:2]
        cell_w = w / self._grid.cols
        cell_h = h / self._grid.rows
        empty_threshold = 1.0 - settings.empty_cell_min_content_ratio
        for r in range(self._grid.rows):
            for c in range(self._grid.cols):
                x0, y0 = round(c * cell_w), round(r * cell_h)
                x1, y1 = round((c + 1) * cell_w), round((r + 1) * cell_h)
                patch = light[y0:y1, x0:x1]
                if patch.size == 0:
                    continue
                light_frac = float((patch > 0).mean())
                cell_std = float(gray[y0:y1, x0:x1].std())
                # Empty = bare board: mostly light AND smooth (uniform surface).
                # A placed piece is either non-light OR textured photo content —
                # the texture clause rescues bright/glossy pieces (white balls,
                # numbers, highlights) that look "light" yet are clearly filled,
                # which a light-only test wrongly called empty.
                is_empty = (
                    light_frac >= empty_threshold
                    and cell_std < settings.empty_cell_max_std
                )
                self._filled[r][c] = not is_empty
