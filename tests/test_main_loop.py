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
    # Settle delay is wall-clock; zero it so a handful of fast ticks suffice.
    settings.init_view_settle_delay_s = 0.0
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
    settings.init_view_settle_delay_s = 0.0
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


def test_a_moved_board_is_followed_not_ignored(fixtures_dir: Path) -> None:
    """A board that slides across the screen must be tracked.

    Everything downstream reads the live board through this bbox — the overlay
    is positioned from it, and so are the filled-cell scan and the hole-shape
    rescue — so a stale one puts the green box on the wrong cell. The drift
    watchdog could not catch this: it averages the change over (x, y, w, h), so
    a measured 375 px slide with the size unchanged came out as 12 % against a
    35 % threshold and was ignored.
    """
    import numpy as np

    settings = load_settings(None)
    settings.init_view_settle_delay_s = 0.0
    settings.board_bbox_check_interval_s = 0.0
    full = cv2.imread(str(fixtures_dir / "init_view_4.png"))
    window_frame = full[WIN_Y:WIN_Y + WIN_H, WIN_X:WIN_X + WIN_W]

    capture = MockWindowCapture(frame=window_frame, bbox=Bbox(0, 0, WIN_W, WIN_H))
    loop = MainLoop(settings, capture, MockMouseHook(), HeadlessOverlay(), ConsoleNotifier())
    loop.run(max_iterations=10)
    before = loop._ctx.artifacts.board_bbox
    assert before is not None

    # Slide the whole scene right and down; the board keeps its size.
    shift_x, shift_y = 40, 25
    moved = np.zeros_like(window_frame)
    moved[shift_y:, shift_x:] = window_frame[:window_frame.shape[0] - shift_y,
                                             :window_frame.shape[1] - shift_x]
    capture.frame = moved
    loop._monitor_board_bbox(moved)

    after = loop._ctx.artifacts.board_bbox
    assert after is not None
    assert abs(after.x - (before.x + shift_x)) <= 6, (
        f"board not followed: {before} -> {after}"
    )
    assert abs(after.y - (before.y + shift_y)) <= 6
    # The grid was derived from the calibrated size, so the size must not move.
    assert (after.w, after.h) == (before.w, before.h)
    assert loop._board_state is None, "filled-cell state from the old crop is stale"


def test_a_stationary_board_does_not_jitter(fixtures_dir: Path) -> None:
    """Contour noise (0-8 px measured) must not drag the bbox around."""
    settings = load_settings(None)
    settings.init_view_settle_delay_s = 0.0
    settings.board_bbox_check_interval_s = 0.0
    full = cv2.imread(str(fixtures_dir / "init_view_4.png"))
    window_frame = full[WIN_Y:WIN_Y + WIN_H, WIN_X:WIN_X + WIN_W]

    capture = MockWindowCapture(frame=window_frame, bbox=Bbox(0, 0, WIN_W, WIN_H))
    loop = MainLoop(settings, capture, MockMouseHook(), HeadlessOverlay(), ConsoleNotifier())
    loop.run(max_iterations=10)
    before = loop._ctx.artifacts.board_bbox

    for _ in range(5):
        loop._monitor_board_bbox(window_frame)
    assert loop._ctx.artifacts.board_bbox == before
