"""In-memory window capture for headless tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from puzzle_assistant.capture.interfaces import WindowCandidate
from puzzle_assistant.utils.coords import Bbox


class MockWindowCapture:
    """Returns a fixed BGR frame regardless of bbox arguments.

    Used by tests and ``--test-mode`` so the matching pipeline can run on a
    Linux developer machine without a real game window.
    """

    def __init__(
        self,
        frame: np.ndarray | None = None,
        title: str = "Masa #2 @ YapBoz Salonu 2 [Yönetici: tester]",
        bbox: Bbox | None = None,
    ) -> None:
        if frame is None:
            frame = np.zeros((1440, 2560, 3), dtype=np.uint8)
        self._frame = frame
        self._title = title
        h, w = frame.shape[:2]
        self._bbox = bbox or Bbox(0, 0, w, h)

    @classmethod
    def from_png(cls, path: Path, title: str | None = None) -> "MockWindowCapture":
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read fixture: {path}")
        if title is None:
            return cls(frame=img)
        return cls(frame=img, title=title)

    def find_candidates(self, title_substring: str) -> list[WindowCandidate]:
        if title_substring in self._title:
            return [WindowCandidate(handle=1, title=self._title, bbox=self._bbox)]
        return []

    def get_bbox(self, handle: int) -> Bbox | None:
        if handle != 1:
            return None
        return self._bbox

    def capture(self, bbox: Bbox) -> np.ndarray:
        # Return the slice that intersects the requested bbox.
        x1 = max(bbox.x, 0)
        y1 = max(bbox.y, 0)
        x2 = min(bbox.x + bbox.w, self._frame.shape[1])
        y2 = min(bbox.y + bbox.h, self._frame.shape[0])
        if x2 <= x1 or y2 <= y1:
            return np.zeros((max(bbox.h, 1), max(bbox.w, 1), 3), dtype=np.uint8)
        return self._frame[y1:y2, x1:x2].copy()

    def raise_window(self, handle: int) -> None:
        pass
