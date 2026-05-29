"""Phase 5 — mouse hook smoke tests (mock + factory)."""

from __future__ import annotations

import time

from puzzle_assistant.capture.interfaces import MouseEvent
from puzzle_assistant.capture.mock_mouse_hook import MockMouseHook
from puzzle_assistant.utils.platform import make_mouse_hook


def test_mock_emits_and_drains() -> None:
    hook = MockMouseHook()
    hook.start()
    hook.emit(MouseEvent(type="down", x=100, y=200, ts=time.monotonic()))
    hook.emit(MouseEvent(type="up", x=110, y=210, ts=time.monotonic()))
    q = hook.queue
    assert q.qsize() == 2
    e1 = q.get_nowait()
    e2 = q.get_nowait()
    assert e1.type == "down" and e1.x == 100
    assert e2.type == "up" and e2.x == 110
    hook.stop()


def test_factory_returns_hook_object() -> None:
    hook = make_mouse_hook("mock")
    assert hasattr(hook, "start")
    assert hasattr(hook, "stop")
    assert hook.queue is not None
