"""Track which grid cells are already filled with placed pieces.

Spec §2.1. A dragged piece can only go to an *empty* cell, so knowing which
cells are filled lets the matcher drop occupied positions from the candidate
set. As the puzzle fills in, the empty set shrinks and the margin between
candidates rises naturally — which is exactly where the single-/flat-colour
pieces were failing (dozens of look-alike positions tying at margin≈0).

Empty-cell test: the light board surface shows through an empty cell, so a cell
whose pixels are dominated by the board-light HSV range (the same range board
detection and segmentation already use) is empty; once a piece sits there the
cell is full of image content and the light fraction collapses.
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
        # Latest live board crop (real pixels of already-placed pieces). Seam
        # matching samples neighbour edge strips from this, NOT from the
        # reference board, because the seam signal is about how the dragged
        # piece continues into its *actually placed* neighbours.
        self._live_board: np.ndarray | None = None

    @property
    def filled(self) -> list[list[bool]]:
        return self._filled

    def is_filled(self, row: int, col: int) -> bool:
        if 0 <= row < self._grid.rows and 0 <= col < self._grid.cols:
            return self._filled[row][col]
        return False

    def filled_count(self) -> int:
        return sum(v for row in self._filled for v in row)

    def total_cells(self) -> int:
        return self._grid.rows * self._grid.cols

    def filled_neighbour_count(self, row: int, col: int) -> int:
        return sum(
            self.is_filled(row + dr, col + dc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )

    def cell_crop(self, row: int, col: int) -> np.ndarray | None:
        """The live BGR pixels of cell ``(row, col)`` from the latest board."""
        if self._live_board is None:
            return None
        h, w = self._live_board.shape[:2]
        cw, ch = w / self._grid.cols, h / self._grid.rows
        x0, y0 = round(col * cw), round(row * ch)
        x1, y1 = round((col + 1) * cw), round((row + 1) * ch)
        crop = self._live_board[y0:y1, x0:x1]
        return crop if crop.size else None

    def update(self, board_bgr: np.ndarray, settings: Settings) -> None:
        """Recompute the filled matrix from a fresh full-board crop."""

        if board_bgr.size == 0:
            return
        self._live_board = board_bgr.copy()
        light = _board_light_mask(board_bgr, settings)
        h, w = light.shape[:2]
        cell_w = w / self._grid.cols
        cell_h = h / self._grid.rows
        # A cell counts as empty when most of it is bare board light.
        empty_threshold = 1.0 - settings.empty_cell_min_content_ratio
        for r in range(self._grid.rows):
            for c in range(self._grid.cols):
                x0, y0 = round(c * cell_w), round(r * cell_h)
                x1, y1 = round((c + 1) * cell_w), round((r + 1) * cell_h)
                patch = light[y0:y1, x0:x1]
                if patch.size == 0:
                    continue
                light_frac = float((patch > 0).mean())
                self._filled[r][c] = light_frac < empty_threshold


def _board_light_mask(board_bgr: np.ndarray, settings: Settings) -> np.ndarray:
    hsv = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2HSV)
    low = np.array(settings.board_light_hsv_low, dtype=np.uint8)
    high = np.array(settings.board_light_hsv_high, dtype=np.uint8)
    return cv2.inRange(hsv, low, high)
