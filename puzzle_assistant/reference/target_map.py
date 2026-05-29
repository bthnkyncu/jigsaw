"""Per-cell reference image with pre-computed match features.

Brief §7.7. The target map is the ground-truth lookup the matching engine
queries when a piece is picked up. It can be built from one of two sources:

* **Primary** — the init view (full, sharp, cut-line accurate).
* **Fallback** — an upscaled crop of the right-side reference panel.

Each grid cell is sliced out once and its ORB descriptors + Lab mean are
cached for the entire puzzle session.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np

from puzzle_assistant.config import Settings
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.coords import GridSpec

Quality = Literal["primary", "fallback"]


@dataclass
class CellFeatures:
    """Lazily-computable per-cell features."""

    image: np.ndarray             # BGR slice for template matching
    lab_mean: np.ndarray          # shape (3,) float32
    orb_descriptors: np.ndarray | None  # ORB descriptors (or None if too featureless)


@dataclass
class TargetMap:
    grid: GridSpec
    quality: Quality
    cells: list[list[CellFeatures]]
    board_image: np.ndarray  # full reference board BGR (for sliding-window match)

    @property
    def cell_size(self) -> tuple[int, int]:
        any_cell = self.cells[0][0].image
        return any_cell.shape[1], any_cell.shape[0]  # (w, h)


def build_from_init_view(
    board_bgr: np.ndarray,
    grid: GridSpec,
    settings: Settings,
) -> TargetMap:
    """Primary path — slice the rendered board into grid cells."""

    return _build(board_bgr, grid, settings, quality="primary")


def build_from_reference_panel(
    panel_bgr: np.ndarray,
    grid: GridSpec,
    board_w: int,
    board_h: int,
    settings: Settings,
) -> TargetMap:
    """Fallback path — bicubic-upscale the tiny panel to board dimensions
    then slice into grid cells.
    """

    upscaled = cv2.resize(panel_bgr, (board_w, board_h), interpolation=cv2.INTER_CUBIC)
    return _build(upscaled, grid, settings, quality="fallback")


def _build(
    board_bgr: np.ndarray,
    grid: GridSpec,
    settings: Settings,
    *,
    quality: Quality,
) -> TargetMap:
    h, w = board_bgr.shape[:2]
    cell_w = w / grid.cols
    cell_h = h / grid.rows

    orb: Any = cv2.ORB_create(nfeatures=settings.orb_n_features)  # type: ignore[attr-defined]
    cells: list[list[CellFeatures]] = []
    for r in range(grid.rows):
        row: list[CellFeatures] = []
        for c in range(grid.cols):
            x0 = round(c * cell_w)
            y0 = round(r * cell_h)
            x1 = round((c + 1) * cell_w)
            y1 = round((r + 1) * cell_h)
            crop = board_bgr[y0:y1, x0:x1]
            if crop.size == 0:
                # Edge case — synthesise a 1×1 placeholder so the grid index
                # never gets a ``None`` here. Matching will score it close to 0.
                crop = np.zeros((1, 1, 3), dtype=np.uint8)
            row.append(_cell_features(crop, orb))
        cells.append(row)

    plog.event(
        "target_map_built",
        quality=quality,
        rows=grid.rows,
        cols=grid.cols,
        total_cells=grid.total_pieces,
    )
    return TargetMap(
        grid=grid, quality=quality, cells=cells, board_image=board_bgr.copy()
    )


def _cell_features(cell_bgr: np.ndarray, orb: Any) -> CellFeatures:
    lab = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2LAB)
    lab_mean = lab.reshape(-1, 3).mean(axis=0).astype(np.float32)
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    _kp, des = orb.detectAndCompute(gray, None)
    descriptors: np.ndarray | None
    if des is None or des.shape[0] == 0:
        descriptors = None
        plog.event(
            "target_cell_no_descriptors",
            level=logging.DEBUG,
            shape=[cell_bgr.shape[0], cell_bgr.shape[1]],
        )
    else:
        descriptors = des
    return CellFeatures(image=cell_bgr.copy(), lab_mean=lab_mean, orb_descriptors=descriptors)
