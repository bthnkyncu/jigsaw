"""Linux global mouse hook via pynput (XRecord on X11).

``pynput`` opens its own thread; we drop every click event into a thread-safe
``queue.Queue`` that the main loop drains.
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Literal

from pynput import mouse

from puzzle_assistant.capture.interfaces import MouseEvent
from puzzle_assistant.utils import logger as plog


class LinuxMouseHook:
    def __init__(self) -> None:
        self._q: queue.Queue[MouseEvent] = queue.Queue()
        self._listener: mouse.Listener | None = None

    def start(self) -> None:
        if self._listener is not None:
            return
        self._listener = mouse.Listener(on_click=self._on_click)
        self._listener.start()
        plog.event("mouse_hook_started", backend="pynput")

    def stop(self) -> None:
        if self._listener is None:
            return
        self._listener.stop()
        self._listener = None
        plog.event("mouse_hook_stopped")

    @property
    def queue(self) -> "queue.Queue[MouseEvent]":
        return self._q

    # --- pynput callback ---
    def _on_click(self, x: int, y: int, button: object, pressed: bool) -> None:
        # Only listen to the left mouse button — middle/right are not used
        # by the game's drag interaction.
        if str(button) not in ("Button.left", "1"):
            return
        evt_type: Literal["down", "up"] = "down" if pressed else "up"
        try:
            self._q.put_nowait(MouseEvent(type=evt_type, x=int(x), y=int(y), ts=time.monotonic()))
        except queue.Full:  # pragma: no cover — Queue has no bound by default
            plog.event("mouse_queue_full", level=logging.WARNING)
