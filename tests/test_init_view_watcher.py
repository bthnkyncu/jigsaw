"""Phase 2 — init view watcher tests."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

from puzzle_assistant.calibration.init_view_watcher import InitViewWatcher
from puzzle_assistant.config import load_settings
from puzzle_assistant.utils.coords import Bbox

WIN_X, WIN_Y, WIN_W, WIN_H = 0, 64, 1920, 1016


def _load_window_crop(fixtures_dir: Path, name: str) -> "cv2.typing.MatLike":
    full = cv2.imread(str(fixtures_dir / f"{name}.png"))
    return full[WIN_Y:WIN_Y + WIN_H, WIN_X:WIN_X + WIN_W]


def test_init_view_detected_on_full_image(fixtures_dir: Path) -> None:
    settings = load_settings(None)
    # Settle delay uses wall-clock time; zero it so the test isn't time-bound.
    settings.init_view_settle_delay_s = 0.0
    watcher = InitViewWatcher(settings)
    img = _load_window_crop(fixtures_dir, "init_view_4")
    # Use the known board bbox from board_detector tests.
    board = Bbox(525, 138, 744, 558)
    # The watcher requires several consecutive stable frames to fire — feed
    # the same frame repeatedly to simulate the ~2 second motionless init view.
    decision = None
    for _ in range(settings.init_view_stable_frame_count + 1):
        decision = watcher.assess(img, board, panel_bbox=None)
    assert decision is not None
    assert decision.captured, f"init view not detected, variance={decision.variance}"


def test_blank_board_does_not_trigger(fixtures_dir: Path) -> None:
    """An empty board (mid-game with cleared center) must NOT be flagged."""

    settings = load_settings(None)
    watcher = InitViewWatcher(settings)
    img = _load_window_crop(fixtures_dir, "mid_game_1")
    board = Bbox(525, 138, 743, 557)
    decision = watcher.assess(img, board, panel_bbox=None)
    assert not decision.captured, f"falsely flagged init view, variance={decision.variance}"


def test_timeout_after_window(fixtures_dir: Path) -> None:
    settings = load_settings(None)
    # Cut the timeout to something we can hit fast.
    settings.init_view_wait_timeout_s = 0.05
    watcher = InitViewWatcher(settings)
    img = _load_window_crop(fixtures_dir, "mid_game_1")
    board = Bbox(525, 138, 743, 557)
    # First assess starts the clock.
    watcher.assess(img, board, panel_bbox=None)
    time.sleep(0.1)
    decision = watcher.assess(img, board, panel_bbox=None)
    assert decision.timed_out
