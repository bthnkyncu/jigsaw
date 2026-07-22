"""Watch the board region until the ``init view`` appears.

When the player starts a new puzzle, the game shows the *fully assembled*
puzzle on the board for ~2 seconds before pieces scatter. We need to catch
that window because it is the only high-resolution reference of the finished
image (the right-side scoreboard panel is tiny).

Strategy (adapted from the Puzzle_Game prototype, which was reliable in
practice): combine **stable-frame counting** with a board-content sanity
check.

1. Compute mean absolute pixel diff between consecutive board crops. A diff
   below ``stable_frame_diff_max`` (~2.5) counts as one stable frame.
2. Require ``stable_frame_count_min`` (~6) consecutive stable frames. While
   pieces are scattering the diff is large; once they settle / once the
   init view is on screen the board is dead still.
3. Sanity check: the board crop must have at least
   ``init_view_variance_min`` variance, ruling out empty / monochrome boards.

After ``init_view_wait_timeout_s`` without satisfying all three, the caller
should fall back to the reference-panel path.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np

from puzzle_assistant.config import Settings
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.coords import Bbox


@dataclass
class WatcherDecision:
    captured: bool
    timed_out: bool
    variance: float
    panel_correlation: float
    elapsed_s: float
    stable_count: int = 0
    frame_diff: float = 0.0


class InitViewWatcher:
    """Tracks the board over time and signals when an init view appears."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._started_at: float | None = None
        self._prev_board: np.ndarray | None = None
        self._stable_count = 0
        # Monotonic time at which all init-view criteria first held continuously.
        self._criteria_since: float | None = None
        self._prev_bbox: Bbox | None = None

    def reset(self) -> None:
        self._started_at = None
        self._prev_board = None
        self._stable_count = 0
        self._criteria_since = None
        self._prev_bbox = None

    def assess(
        self,
        frame_bgr: np.ndarray,
        board_bbox: Bbox,
        panel_bbox: Bbox | None,
    ) -> WatcherDecision:
        """Evaluate the latest frame; returns whether init view was captured."""

        now = time.monotonic()
        if self._started_at is None:
            self._started_at = now
        elapsed = now - self._started_at

        # While the pieces fly apart, detect_board reports a board that grows
        # row by row, so the bbox jumps between frames. The pixel-stability test
        # can't see that — it resizes every crop to 128x96, so a half board and
        # a full board of the same picture both read as "unchanged" (measured:
        # frame_diff 0.0 while the height went 540 -> 271). Treat a bbox change
        # as motion in its own right, otherwise a mid-animation frame can be
        # captured as the reference.
        if self._prev_bbox is not None and not _bbox_close(self._prev_bbox, board_bbox):
            self._stable_count = 0
            self._criteria_since = None
        self._prev_bbox = board_bbox

        board_crop = _safe_crop(frame_bgr, board_bbox)
        variance = float(np.var(board_crop)) if board_crop.size else 0.0

        # Frame-to-frame diff on the board region. Resize both to a small
        # common shape so geometric drift in detect_board doesn't dominate.
        frame_diff = float("inf")
        if board_crop.size and self._prev_board is not None:
            try:
                a = cv2.resize(board_crop, (128, 96), interpolation=cv2.INTER_AREA)
                b = cv2.resize(self._prev_board, (128, 96), interpolation=cv2.INTER_AREA)
                frame_diff = float(cv2.absdiff(a, b).mean())
            except cv2.error:
                frame_diff = float("inf")
        if board_crop.size:
            self._prev_board = board_crop.copy()

        if frame_diff <= self._settings.init_view_stable_diff_max:
            self._stable_count += 1
        else:
            self._stable_count = max(0, self._stable_count - 1)

        panel_corr = 0.0
        if panel_bbox is not None:
            panel_crop = _safe_crop(frame_bgr, panel_bbox)
            if panel_crop.size and board_crop.size:
                panel_corr = _hist_correlation(board_crop, panel_crop)

        # Init view = the *assembled* puzzle sits on the board. Three signals
        # must hold together:
        #   - stable: the board has been motionless for several frames (pieces
        #     haven't scattered yet),
        #   - rich: variance rules out an empty/monochrome board,
        #   - panel match: the board content correlates with the right-side
        #     reference thumbnail (same image).
        # The panel-match check is the decisive one: a board full of *scattered*
        # pieces is also stable and high-variance, but its histogram does NOT
        # match the reference panel (live runs showed corr~0.04 when scattered
        # vs corr~0.5 on the true init view).
        # The panel check is the decisive one, so losing it is dangerous: on a
        # pale blue/white puzzle the thumbnail blends into the panel background,
        # detection fails, and "no panel" used to mean "panel check passed" —
        # leaving only stability, which a mid-scatter frame satisfies. We still
        # can't demand a panel (some boards genuinely lack one), but without it
        # we demand far more motionless frames before believing the board.
        panel_missing = panel_bbox is None
        need_stable = self._settings.init_view_stable_frame_count
        if panel_missing:
            need_stable *= self._settings.init_view_no_panel_stable_multiplier
        stable_ok = self._stable_count >= need_stable
        variance_ok = variance >= self._settings.init_view_variance_min
        panel_ok = panel_missing or panel_corr >= self._settings.init_view_panel_corr_min
        criteria_now = stable_ok and variance_ok and panel_ok

        # The puzzle fills in top-to-bottom over ~1s. Don't capture the instant
        # the criteria first hold (bottom rows may still be settling) — require
        # them to hold continuously for ``init_view_settle_delay_s`` first.
        if criteria_now:
            if self._criteria_since is None:
                self._criteria_since = now
            settled = (now - self._criteria_since) >= self._settings.init_view_settle_delay_s
        else:
            self._criteria_since = None
            settled = False
        criteria_ok = criteria_now and settled

        timed_out = (
            elapsed >= self._settings.init_view_wait_timeout_s and not criteria_ok
        )
        decision = WatcherDecision(
            captured=criteria_ok,
            timed_out=timed_out,
            variance=variance,
            panel_correlation=panel_corr,
            elapsed_s=elapsed,
            stable_count=self._stable_count,
            frame_diff=frame_diff if frame_diff != float("inf") else -1.0,
        )
        plog.event(
            "init_view_assess",
            level=logging.INFO,
            variance=round(variance, 1),
            panel_corr=round(panel_corr, 3),
            elapsed_s=round(elapsed, 2),
            frame_diff=round(decision.frame_diff, 2),
            stable_count=self._stable_count,
            captured=criteria_ok,
            timed_out=timed_out,
            panel_missing=panel_missing,
        )
        if criteria_ok or timed_out:
            self._started_at = None
            self._stable_count = 0
            self._prev_board = None
            self._criteria_since = None
        return decision


def _bbox_close(a: Bbox, b: Bbox, tol: int = 6) -> bool:
    """Same board region, allowing for a few pixels of detection jitter."""
    return (
        abs(a.x - b.x) <= tol and abs(a.y - b.y) <= tol
        and abs(a.w - b.w) <= tol and abs(a.h - b.h) <= tol
    )


def _safe_crop(frame: np.ndarray, bbox: Bbox) -> np.ndarray:
    h, w = frame.shape[:2]
    x1 = max(0, bbox.x)
    y1 = max(0, bbox.y)
    x2 = min(w, bbox.x + bbox.w)
    y2 = min(h, bbox.y + bbox.h)
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=frame.dtype)
    return frame[y1:y2, x1:x2]


def _hist_correlation(a_bgr: np.ndarray, b_bgr: np.ndarray) -> float:
    """Channel-averaged histogram correlation in [0, 1].

    Both inputs are resized to a small common shape so a 800×500 board can be
    compared against a 140×80 reference panel without bias.
    """

    target_size = (64, 48)
    a = cv2.resize(a_bgr, target_size, interpolation=cv2.INTER_AREA)
    b = cv2.resize(b_bgr, target_size, interpolation=cv2.INTER_AREA)

    correlations: list[float] = []
    for ch in range(3):
        hist_a = cv2.calcHist([a], [ch], None, [32], [0, 256])
        hist_b = cv2.calcHist([b], [ch], None, [32], [0, 256])
        cv2.normalize(hist_a, hist_a, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
        cv2.normalize(hist_b, hist_b, alpha=1.0, beta=0.0, norm_type=cv2.NORM_L1)
        c = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)
        correlations.append(max(0.0, c))
    return float(sum(correlations) / len(correlations))
