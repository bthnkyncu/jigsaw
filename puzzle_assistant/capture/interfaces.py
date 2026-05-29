"""Protocol definitions for platform-specific capture + input modules."""

from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from puzzle_assistant.utils.coords import Bbox


@dataclass(frozen=True)
class WindowCandidate:
    handle: int
    title: str
    bbox: Bbox


MouseEventType = Literal["down", "up", "move"]


@dataclass(frozen=True)
class MouseEvent:
    type: MouseEventType
    x: int
    y: int
    ts: float


@runtime_checkable
class WindowCaptureInterface(Protocol):
    def find_candidates(self, title_substring: str) -> list[WindowCandidate]:
        """Return all top-level windows whose title contains ``title_substring``."""

    def get_bbox(self, handle: int) -> Bbox | None:
        """Refresh and return the current bbox for a window handle, or ``None``
        if the window is gone / minimized."""

    def capture(self, bbox: Bbox) -> np.ndarray:
        """Return a BGR ``np.ndarray`` of the screen region."""

    def raise_window(self, handle: int) -> None:
        """Bring the window to the front so screen-capture sees its content.
        No-op if the platform backend does not support it."""


@runtime_checkable
class MouseHookInterface(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    @property
    def queue(self) -> "queue.Queue[MouseEvent]": ...
