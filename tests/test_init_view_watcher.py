"""Phase 2 — init view watcher tests."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

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
    # With no reference panel the decisive "board matches the thumbnail" check
    # is unavailable, so the watcher demands proportionally more motionless
    # frames before it will trust the board; feed that many.
    needed = (
        settings.init_view_stable_frame_count
        * settings.init_view_no_panel_stable_multiplier
    )
    decision = None
    for _ in range(needed + 1):
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


def test_bbox_change_resets_stability(fixtures_dir: Path) -> None:
    """A moving board bbox must count as motion, not as a still frame.

    While the pieces fly apart, ``detect_board`` reports a board that grows row
    by row. The pixel test cannot see that — it resizes every crop to 128×96, so
    a half board and a full board of the same picture both read as unchanged
    (observed live: frame_diff 0.0 while the height went 540 → 271). Without
    this reset the watcher captured its reference off a half-built board and the
    whole game ran against a wrong target map.
    """
    settings = load_settings(None)
    settings.init_view_settle_delay_s = 0.0
    watcher = InitViewWatcher(settings)
    img = _load_window_crop(fixtures_dir, "init_view_4")
    full = Bbox(525, 138, 744, 558)
    half = Bbox(525, 138, 744, 279)

    needed = (
        settings.init_view_stable_frame_count
        * settings.init_view_no_panel_stable_multiplier
    )
    for _ in range(needed):
        watcher.assess(img, full, panel_bbox=None)
    # Board bbox suddenly halves — the run of stable frames must start over.
    decision = watcher.assess(img, half, panel_bbox=None)
    assert decision.stable_count == 0, "bbox change should reset the stable run"
    assert not decision.captured, "must not capture on the frame the board jumped"


def test_countdown_screen_is_not_captured(fixtures_dir: Path) -> None:
    """The pre-game countdown must not be taken for the assembled puzzle.

    Before a game starts the board shows a banner over an otherwise bare
    surface. It is motionless and has enough variance from the banner, so every
    other criterion passes and it was captured as the reference — the whole game
    then matched against a picture of a countdown. The assembled puzzle covers
    the board; this screen does not.
    """
    settings = load_settings(None)
    settings.init_view_settle_delay_s = 0.0
    watcher = InitViewWatcher(settings)

    # Bare board with a bright banner across the middle, as the game draws it.
    board = np.full((540, 360, 3), (235, 225, 215), np.uint8)
    board[230:300, 30:330] = (40, 200, 245)
    frame = np.zeros((700, 600, 3), np.uint8)
    frame[60:600, 100:460] = board
    bbox = Bbox(100, 60, 360, 540)

    decision = None
    for _ in range(settings.init_view_stable_frame_count
                   * settings.init_view_no_panel_stable_multiplier + 5):
        decision = watcher.assess(frame, bbox, panel_bbox=None)
    assert decision is not None
    assert not decision.captured, (
        f"countdown screen captured as init view (bare={decision.bare_board:.2f})"
    )
