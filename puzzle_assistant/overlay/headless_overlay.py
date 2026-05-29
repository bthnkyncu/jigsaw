"""Headless overlay used by ``--test-mode`` and unit tests.

Records the latest ``show`` / ``hide`` call without opening any GUI window.
"""

from __future__ import annotations

from puzzle_assistant.utils.coords import Bbox


class HeadlessOverlay:
    def __init__(self) -> None:
        self.last_shown: Bbox | None = None
        self.hide_count = 0

    def show(self, target: Bbox) -> None:
        self.last_shown = target

    def hide(self) -> None:
        self.last_shown = None
        self.hide_count += 1

    def shutdown(self) -> None:
        self.last_shown = None
