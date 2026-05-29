"""Windows window capture (pywin32 + mss).

Implemented for the customer's Windows 11 deployment. The Linux X11 backend
is the primary development target; this file is exercised by the customer
acceptance run (Phase 9 in the plan).

The Win32 imports live inside the method bodies so this module imports
cleanly on Linux too (the platform factory never invokes the class on Linux).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import mss
import numpy as np

from puzzle_assistant.capture.interfaces import WindowCandidate
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.coords import Bbox


class WindowsWindowCapture:
    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("WindowsWindowCapture requires sys.platform == 'win32'.")
        self._mss: Any = None

    def find_candidates(self, title_substring: str) -> list[WindowCandidate]:  # pragma: no cover
        import win32gui

        results: list[WindowCandidate] = []

        def _enum(hwnd: int, _lparam: int) -> bool:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            title = win32gui.GetWindowText(hwnd)
            if not title or title_substring not in title:
                return True
            rect = win32gui.GetWindowRect(hwnd)
            x, y, x2, y2 = rect
            results.append(WindowCandidate(
                handle=hwnd, title=title, bbox=Bbox(x=x, y=y, w=x2 - x, h=y2 - y),
            ))
            return True

        win32gui.EnumWindows(_enum, 0)
        return results

    def get_bbox(self, handle: int) -> Bbox | None:  # pragma: no cover
        import win32con
        import win32gui

        try:
            rect = win32gui.GetWindowRect(handle)
        except OSError as exc:
            plog.event("win_bbox_failed", level=logging.WARNING, handle=handle, error=str(exc))
            return None
        x, y, x2, y2 = rect
        if x2 - x <= 0 or y2 - y <= 0:
            return None
        placement = win32gui.GetWindowPlacement(handle)
        if placement[1] == win32con.SW_SHOWMINIMIZED:
            return None
        return Bbox(x=x, y=y, w=x2 - x, h=y2 - y)

    def capture(self, bbox: Bbox) -> np.ndarray:  # pragma: no cover
        if self._mss is None:
            self._mss = mss.mss()
        region = {"left": bbox.x, "top": bbox.y, "width": bbox.w, "height": bbox.h}
        shot = self._mss.grab(region)
        arr = np.frombuffer(shot.rgb, dtype=np.uint8).reshape(shot.height, shot.width, 3)
        result: np.ndarray = arr[:, :, ::-1].copy()
        return result

    def raise_window(self, handle: int) -> None:  # pragma: no cover
        import win32con
        import win32gui
        try:
            win32gui.ShowWindow(handle, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(handle)
        except OSError:
            pass
