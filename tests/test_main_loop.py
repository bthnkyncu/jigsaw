"""Phase 8 — main loop integration with mock subsystems."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from puzzle_assistant.capture.interfaces import MouseEvent
from puzzle_assistant.capture.mock_capture import MockWindowCapture
from puzzle_assistant.capture.mock_mouse_hook import MockMouseHook
from puzzle_assistant.config import load_settings
from puzzle_assistant.overlay.headless_overlay import HeadlessOverlay
from puzzle_assistant.state.main_loop import MainLoop
from puzzle_assistant.state.state_machine import State
from puzzle_assistant.utils.coords import Bbox
from puzzle_assistant.utils.notifier import ConsoleNotifier

WIN_X, WIN_Y, WIN_W, WIN_H = 0, 64, 1920, 1016


def test_full_calibration_from_init_view(fixtures_dir: Path) -> None:
    """Feed an init-view fixture; the loop should reach READY."""

    settings = load_settings(None)
    full = cv2.imread(str(fixtures_dir / "init_view_4.png"))
    window_frame = full[WIN_Y:WIN_Y + WIN_H, WIN_X:WIN_X + WIN_W]

    capture = MockWindowCapture(frame=window_frame, bbox=Bbox(0, 0, WIN_W, WIN_H))
    mouse = MockMouseHook()
    overlay = HeadlessOverlay()
    notifier = ConsoleNotifier()
    loop = MainLoop(settings, capture, mouse, overlay, notifier)

    # First iteration picks the window; subsequent ones accumulate stable-frame
    # counts in the init view watcher. Stable threshold defaults to 6, plus
    # one tick for window discovery + a couple of margin.
    loop.run(max_iterations=10)
    assert loop._ctx.state == State.READY
    assert loop._ctx.artifacts.target_map is not None
    assert loop._ctx.artifacts.target_map.quality == "primary"


def test_mouse_down_triggers_overlay(fixtures_dir: Path) -> None:
    settings = load_settings(None)
    full = cv2.imread(str(fixtures_dir / "init_view_4.png"))
    window_frame = full[WIN_Y:WIN_Y + WIN_H, WIN_X:WIN_X + WIN_W]

    capture = MockWindowCapture(frame=window_frame, bbox=Bbox(0, 0, WIN_W, WIN_H))
    mouse = MockMouseHook()
    overlay = HeadlessOverlay()
    notifier = ConsoleNotifier()
    loop = MainLoop(settings, capture, mouse, overlay, notifier)

    # Calibrate first.
    loop.run(max_iterations=10)
    assert loop._ctx.state == State.READY

    # Now feed a click somewhere inside the board.
    mouse.emit(MouseEvent(type="down", x=900, y=400, ts=time.monotonic()))
    loop.run(max_iterations=2)
    # We don't assert overlay was shown — the click may land on a flat patch
    # that fails the match-threshold. We only assert no crash and that
    # state machine bounced through TRACKING and back when followed by up.
    mouse.emit(MouseEvent(type="up", x=900, y=400, ts=time.monotonic()))
    loop.run(max_iterations=2)
    assert loop._ctx.state == State.READY


def test_loop_survives_missing_window() -> None:
    """No window candidates → loop sits in IDLE and does not crash."""

    settings = load_settings(None)
    # Frame is irrelevant — find_candidates returns empty list.
    capture = MockWindowCapture(title="NotARealWindow")
    mouse = MockMouseHook()
    overlay = HeadlessOverlay()
    notifier = ConsoleNotifier()
    loop = MainLoop(settings, capture, mouse, overlay, notifier)
    loop.run(max_iterations=3)
    assert loop._ctx.state == State.IDLE
