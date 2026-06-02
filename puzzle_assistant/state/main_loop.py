"""End-to-end main loop tying every subsystem to the state machine.

The loop owns one window-capture object, one mouse hook, and one overlay.
It pulls frames at ``target_fps`` and dispatches based on the current state.
Every iteration runs under the watchdog so a slow frame can't stall the loop.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from puzzle_assistant.calibration.board_detector import detect_board
from puzzle_assistant.calibration.grid_detector import (
    detect_grid_from_init_view,
    estimate_grid_from_aspect,
)
from puzzle_assistant.calibration.init_view_watcher import InitViewWatcher
from puzzle_assistant.calibration.reference_panel import (
    PanelSignature,
    compute_signature,
    detect_reference_panel,
)
from puzzle_assistant.config import Settings
from puzzle_assistant.matching.engine import match_piece
from puzzle_assistant.piece.board_state import BoardState
from puzzle_assistant.piece.group_detection import classify as classify_piece
from puzzle_assistant.piece.pickup import pickup_from_window
from puzzle_assistant.reference.target_map import build_from_init_view
from puzzle_assistant.state import state_machine as sm
from puzzle_assistant.state.watchdog import watch
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.coords import Bbox, cell_bbox

if TYPE_CHECKING:
    from puzzle_assistant.capture.interfaces import MouseHookInterface, WindowCaptureInterface
    from puzzle_assistant.overlay.interfaces import NotifierInterface, OverlayInterface


class MainLoop:
    def __init__(
        self,
        settings: Settings,
        capture: "WindowCaptureInterface",
        mouse_hook: "MouseHookInterface",
        overlay: "OverlayInterface",
        notifier: "NotifierInterface",
    ) -> None:
        self._settings = settings
        self._capture = capture
        self._mouse = mouse_hook
        self._overlay = overlay
        self._notifier = notifier
        self._ctx = sm.Context()
        self._frame_count = 0
        self._handle: int | None = None
        self._last_panel_check_ts = 0.0
        self._init_view_watcher = InitViewWatcher(settings)
        self._prev_state = sm.State.IDLE
        self._running = False
        self._board_state: BoardState | None = None
        self._last_board_state_ts = 0.0

    def run(self, max_iterations: int | None = None) -> None:
        """Drive the loop. ``max_iterations`` is used by tests; ``None`` = forever."""

        self._running = True
        self._mouse.start()
        plog.event("main_loop_start")
        iteration_period = 1.0 / max(self._settings.target_fps, 1)
        try:
            i = 0
            while self._running:
                started = time.monotonic()
                with watch("iteration", self._settings):
                    self._tick()
                elapsed = time.monotonic() - started
                if max_iterations is not None:
                    i += 1
                    if i >= max_iterations:
                        break
                sleep_for = iteration_period - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            self._mouse.stop()
            self._overlay.shutdown()
            plog.event("main_loop_stop")

    def stop(self) -> None:
        self._running = False

    def status_snapshot(self) -> dict[str, object]:
        """Read-only snapshot for the GUI status panel (safe to call from
        another thread — it only reads simple fields)."""
        tmap = self._ctx.artifacts.target_map
        return {
            "running": self._running,
            "state": self._ctx.state.value,
            "window_found": self._handle is not None,
            "calibrated": tmap is not None,
            "quality": tmap.quality if tmap is not None else None,
        }

    # ------------------------------ private -------------------------------

    def _tick(self) -> None:
        try:
            self._refresh_window()
            if self._ctx.state == sm.State.IDLE:
                return
            frame = self._grab_window_frame()
            if frame is None:
                return
            self._frame_count += 1

            # Fresh entry into WAIT_FOR_NEW_PUZZLE (new game / drift / panel
            # change) must start the init-view watcher from scratch — otherwise
            # the previous game's stale stable-frame count and start time leak
            # in and the watcher neither captures nor times out.
            if (
                self._ctx.state == sm.State.WAIT_FOR_NEW_PUZZLE
                and self._prev_state != sm.State.WAIT_FOR_NEW_PUZZLE
            ):
                self._init_view_watcher.reset()
                self._ctx.fallback_warning_issued = False

            if self._ctx.state == sm.State.WAIT_FOR_NEW_PUZZLE:
                self._board_state = None
                self._try_calibrate(frame)
            elif self._ctx.state in (sm.State.READY, sm.State.TRACKING):
                self._monitor_panel(frame)
                self._monitor_board_bbox(frame)
                # Refresh the filled-cell map only while READY: during a drag
                # detect_board is unreliable (the moving piece swells the bbox),
                # so a TRACKING-time scan would be noisy.
                if self._ctx.state == sm.State.READY:
                    self._refresh_board_state(frame)
                self._handle_mouse_events(frame)
            self._prev_state = self._ctx.state
        except Exception as exc:
            plog.event(
                "tick_exception",
                level=logging.ERROR,
                state=self._ctx.state.value,
                error=str(exc),
            )

    def _refresh_window(self) -> None:
        if self._frame_count % self._settings.window_bbox_refresh_every_n_frames == 0 \
                or self._handle is None:
            candidates = self._capture.find_candidates(
                self._settings.game_window_title_substring
            )
            # The 'Masa #' title is the actual play window; prefer it when present.
            masa = [c for c in candidates if "masa" in c.title.lower()]
            chosen = masa[0] if masa else (candidates[0] if candidates else None)
            if chosen is None:
                if self._handle is not None:
                    sm.on_window_lost(self._ctx)
                    self._notifier.notify(
                        "Yapboz Asistanı", "YapBoz penceresi bulunamadı.", urgency="normal"
                    )
                self._handle = None
                return

            bbox = self._capture.get_bbox(chosen.handle)
            if bbox is None:
                return
            if self._handle != chosen.handle:
                plog.event("window_picked", handle=chosen.handle, title=chosen.title)
                self._handle = chosen.handle
                self._notifier.notify(
                    "Yapboz Asistanı",
                    "Pencere bulundu — yeni oyunu başlatın.",
                    urgency="normal",
                )
            sm.on_window_found(self._ctx, bbox)
            # Keep game window in front so mss captures its actual content.
            if self._ctx.state == sm.State.WAIT_FOR_NEW_PUZZLE:
                self._capture.raise_window(chosen.handle)

    def _grab_window_frame(self) -> Any:
        bbox = self._ctx.artifacts.window_bbox
        if bbox is None or self._handle is None:
            return None
        # Live tests showed that XComposite returns a stale, cached pixmap when
        # the Java game window doesn't damage its surface every frame — the
        # frame_diff between consecutive ticks dropped to exactly 0.0, breaking
        # the init-view watcher. mss reads what's on the screen, so as long as
        # we keep the game window in front (raise_window during calibration)
        # this is the reliable path.
        return self._capture.capture(bbox)

    def _try_calibrate(self, frame: Any) -> None:
        board_bbox = detect_board(frame, self._settings)
        if board_bbox is None:
            plog.event(
                "calibrate_no_board",
                level=logging.DEBUG,
                frame_shape=list(frame.shape),
            )
            return
        panel_bbox = detect_reference_panel(frame, self._settings)
        panel_signature: PanelSignature | None = None
        if panel_bbox is not None:
            panel_crop = frame[panel_bbox.y : panel_bbox.y + panel_bbox.h,
                               panel_bbox.x : panel_bbox.x + panel_bbox.w]
            panel_signature = compute_signature(panel_crop)

        board_crop = frame[board_bbox.y : board_bbox.y + board_bbox.h,
                           board_bbox.x : board_bbox.x + board_bbox.w]
        decision = self._init_view_watcher.assess(frame, board_bbox, panel_bbox)

        # Primary path: init view captured.
        if decision.captured:
            grid = detect_grid_from_init_view(board_crop, self._settings)
            if grid is None:
                # Periodicity search failed — fall back to aspect-ratio search,
                # which always returns a plausible grid for the configured
                # piece-count band. We still use the high-resolution init view
                # crop for the target map, so this is much better than waiting
                # for the panel fallback.
                grid = estimate_grid_from_aspect(
                    board_bbox.w, board_bbox.h, self._settings
                )
            if grid is not None:
                tmap = build_from_init_view(board_crop, grid, self._settings)
                sm.on_calibrated_primary(
                    self._ctx, board_bbox, grid, tmap, panel_bbox, panel_signature
                )
                self._notifier.notify(
                    "Yapboz Asistanı", "Hazır — bol şans!", urgency="normal"
                )
                return

        # Timeout: the assembled init view never appeared (board stayed empty
        # or we missed the ~2s window). The right-side panel is far too small
        # (~5-8 px per piece at 250 pieces) to produce usable matches, so we do
        # NOT calibrate from it — that path produced the bad predictions. Keep
        # waiting and prompt the user to (re)start a game so we can grab a
        # high-quality reference from the assembled board.
        if decision.timed_out and not self._ctx.fallback_warning_issued:
            self._notifier.notify(
                "Yapboz Asistanı",
                "Yapbozun yapılı halini göremedim — lütfen yeni oyun başlatın.",
                urgency="critical",
            )
            self._ctx.fallback_warning_issued = True
            # The watcher already reset its own clock on timeout, so the next
            # assembled view within the next window will be captured.

    def _monitor_panel(self, frame: Any) -> None:
        now = time.monotonic()
        if now - self._last_panel_check_ts < self._settings.ref_panel_check_interval_s:
            return
        self._last_panel_check_ts = now

        pb = self._ctx.artifacts.panel_bbox
        if pb is None:
            return
        crop = frame[pb.y : pb.y + pb.h, pb.x : pb.x + pb.w]
        new_sig = compute_signature(crop)
        old_sig = self._ctx.artifacts.panel_signature
        if old_sig is None:
            self._ctx.artifacts.panel_signature = new_sig
            return
        dist = new_sig.distance(old_sig)
        if dist > self._settings.ref_panel_change_threshold:
            sm.on_panel_signature_changed(self._ctx)

    def _monitor_board_bbox(self, frame: Any) -> None:
        bb_now = detect_board(frame, self._settings)
        old = self._ctx.artifacts.board_bbox
        if bb_now is None or old is None:
            return
        drift = old.relative_change_pct(bb_now)
        if drift > self._settings.board_bbox_change_threshold_pct:
            sm.on_board_bbox_drift(self._ctx)
            self._notifier.notify(
                "Yapboz Asistanı",
                "Tahta hareketi algılandı — yeniden kalibre ediyorum.",
                urgency="low",
            )

    def _refresh_board_state(self, frame: Any) -> None:
        grid = self._ctx.artifacts.grid
        board_bbox = self._ctx.artifacts.board_bbox
        if grid is None or board_bbox is None:
            return
        now = time.monotonic()
        if (
            self._board_state is not None
            and now - self._last_board_state_ts < self._settings.board_state_refresh_s
        ):
            return
        self._last_board_state_ts = now
        if self._board_state is None:
            self._board_state = BoardState(grid)
        crop = frame[
            board_bbox.y : board_bbox.y + board_bbox.h,
            board_bbox.x : board_bbox.x + board_bbox.w,
        ]
        self._board_state.update(crop, self._settings)

    def _handle_mouse_events(self, frame: Any) -> None:
        window_bbox = self._ctx.artifacts.window_bbox
        if window_bbox is None or self._ctx.artifacts.target_map is None \
                or self._ctx.artifacts.grid is None:
            return

        # Drain all pending events without blocking.
        while not self._mouse.queue.empty():
            evt = self._mouse.queue.get_nowait()
            if not window_bbox.contains(evt.x, evt.y):
                continue
            if evt.type == "down":
                sm.on_mouse_down(self._ctx)
                self._try_pickup_and_match(frame, evt.x, evt.y)
            elif evt.type == "up":
                sm.on_mouse_up(self._ctx)
                self._overlay.hide()

    def _try_pickup_and_match(self, frame: Any, sx: int, sy: int) -> None:
        window_bbox = self._ctx.artifacts.window_bbox
        grid = self._ctx.artifacts.grid
        tmap = self._ctx.artifacts.target_map
        board_bbox = self._ctx.artifacts.board_bbox
        assert window_bbox is not None and grid is not None and tmap is not None
        assert board_bbox is not None
        cursor_window = (sx - window_bbox.x, sy - window_bbox.y)
        result = pickup_from_window(frame, cursor_window, grid, self._settings)
        if result is None:
            return
        kind = classify_piece(result.piece, grid, self._settings)
        if kind == "group":
            plog.event("group_detected")
            return

        # Sliding-window matcher localizes the piece on the board itself, so it
        # wants the full piece silhouette (with background) — it computes its
        # own foreground mask. The eroded core would discard the tab shape that
        # helps disambiguate position.
        match = match_piece(
            result.piece.piece_full, tmap, self._settings, self._board_state
        )
        if match.cell is None:
            return
        target_local = cell_bbox(board_bbox, grid, match.cell)
        target_screen = Bbox(
            x=target_local.x + window_bbox.x,
            y=target_local.y + window_bbox.y,
            w=target_local.w,
            h=target_local.h,
        )
        self._ctx.last_tracking_target = target_screen
        self._overlay.show(target_screen)
