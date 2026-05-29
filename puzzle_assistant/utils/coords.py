"""Coordinate helpers shared between capture, calibration, and overlay."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Bbox:
    """Axis-aligned bounding box in screen pixels (x, y, w, h)."""

    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def area(self) -> int:
        return self.w * self.h

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x2 and self.y <= py < self.y2

    def relative_change_pct(self, other: "Bbox") -> float:
        """Mean percentage change across (x, y, w, h). Used by the bbox watchdog."""

        if other.w == 0 or other.h == 0:
            return 100.0
        dx = abs(self.x - other.x) / max(other.w, 1)
        dy = abs(self.y - other.y) / max(other.h, 1)
        dw = abs(self.w - other.w) / max(other.w, 1)
        dh = abs(self.h - other.h) / max(other.h, 1)
        return ((dx + dy + dw + dh) / 4.0) * 100.0


@dataclass(frozen=True)
class GridSpec:
    """Discovered grid geometry over a board bbox."""

    cols: int
    rows: int
    cell_w: float
    cell_h: float

    @property
    def total_pieces(self) -> int:
        return self.cols * self.rows


@dataclass(frozen=True)
class CellAddress:
    row: int
    col: int


def screen_to_window(sx: int, sy: int, window: Bbox) -> tuple[int, int]:
    return sx - window.x, sy - window.y


def cell_bbox(board: Bbox, grid: GridSpec, addr: CellAddress) -> Bbox:
    x = board.x + round(addr.col * grid.cell_w)
    y = board.y + round(addr.row * grid.cell_h)
    w = round(grid.cell_w)
    h = round(grid.cell_h)
    return Bbox(x, y, w, h)
