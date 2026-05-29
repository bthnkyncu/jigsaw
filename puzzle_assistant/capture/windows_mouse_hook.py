"""Windows global low-level mouse hook (WH_MOUSE_LL).

Use ``pynput`` rather than raw ctypes to stay consistent with the Linux side;
``pynput.mouse.Listener`` already installs WH_MOUSE_LL underneath on Windows.
"""

from __future__ import annotations

import logging
import queue
import sys
import time
from typing import Literal

from pynput import mouse

from puzzle_assistant.capture.interfaces import MouseEvent
from puzzle_assistant.utils import logger as plog


class WindowsMouseHook:
    def __init__(self) -> None:
        if sys.platform != "win32":
            raise RuntimeError("WindowsMouseHook requires sys.platform == 'win32'.")
        self._q: queue.Queue[MouseEvent] = queue.Queue()
        self._listener: mouse.Listener | None = None

    def start(self) -> None:  # pragma: no cover
        if self._listener is not None:
            return
        self._listener = mouse.Listener(on_click=self._on_click)
        self._listener.start()
        plog.event("mouse_hook_started", backend="pynput-win")

    def stop(self) -> None:  # pragma: no cover
        if self._listener is None:
            return
        self._listener.stop()
        self._listener = None

    @property
    def queue(self) -> "queue.Queue[MouseEvent]":
        return self._q

    def _on_click(self, x: int, y: int, button: object, pressed: bool) -> None:  # pragma: no cover
        if str(button) not in ("Button.left", "1"):
            return
        evt_type: Literal["down", "up"] = "down" if pressed else "up"
        try:
            self._q.put_nowait(MouseEvent(type=evt_type, x=int(x), y=int(y), ts=time.monotonic()))
        except queue.Full:
            plog.event("mouse_queue_full", level=logging.WARNING)
