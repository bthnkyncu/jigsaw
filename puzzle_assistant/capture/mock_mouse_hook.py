"""In-memory mouse hook for tests + ``--test-mode``."""

from __future__ import annotations

import queue

from puzzle_assistant.capture.interfaces import MouseEvent


class MockMouseHook:
    """A no-op listener whose queue is fed manually by test fixtures."""

    def __init__(self) -> None:
        self._q: queue.Queue[MouseEvent] = queue.Queue()
        self._running = False

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    @property
    def queue(self) -> "queue.Queue[MouseEvent]":
        return self._q

    # --- test helper ---
    def emit(self, event: MouseEvent) -> None:
        self._q.put(event)
