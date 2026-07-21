"""End-to-end main loop tying every subsystem to the state machine.

The loop owns one window-capture object, one mouse hook, and one overlay.
It pulls frames at ``target_fps`` and dispatches based on the current state.
Every iteration runs under the watchdog so a slow frame can't stall the loop.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import replace
from pathlib import Path
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
from puzzle_assistant.matching.engine import _foreground_mask, match_piece
from puzzle_assistant.matching.hole_shape import rescue_by_hole_shape
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
        self._last_board_bbox_check_ts = 0.0
        # Self-supervised eval: prediction captured at pickup, resolved against
        # the actual landing cell after the drop settles.
        self._eval_pending: dict[str, Any] | None = None
        self._eval_up_ts = 0.0
        # Optional ground-truth dataset capture. Set PUZZLE_RECORD_DIR to a path
        # to save (piece, board, actual-cell) per drop for offline evaluation.
        # Off by default; observation only, never touches matching.
        self._recorder = None
        record_dir = os.environ.get("PUZZLE_RECORD_DIR")
        if record_dir:
            from puzzle_assistant.utils.recorder import PickupRecorder
            self._recorder = PickupRecorder(Path(record_dir))
            plog.event("record_enabled", dir=record_dir)

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
                # detect_board is expensive and unreliable during a drag (the
                # moving piece swells the contour), so run the drift check and
                # the filled-cell scan only while READY — never mid-TRACKING.
                # This also keeps the per-drag tick light (less mouse lag).
                if self._ctx.state == sm.State.READY:
                    self._monitor_board_bbox(frame)
                    self._refresh_board_state(frame)
                    self._eval_placement(frame)
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
        now = time.monotonic()
        if now - self._last_board_bbox_check_ts < self._settings.board_bbox_check_interval_s:
            return
        self._last_board_bbox_check_ts = now
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

    def _eval_placement(self, frame: Any, force: bool = False) -> None:
        """After a drop settles, find the cell the piece actually landed in
        (board-state diff) and log predicted-vs-actual. Self-supervised ground
        truth: no manual labelling, zero precision risk (logging only).

        ``force`` skips the settle wait. A player often grabs the next piece
        well inside ``eval_settle_s`` (measured: 0.42 s between drop and the
        next pickup), which used to overwrite the pending evaluation and lose
        the sample — 100 placements produced only 2 records. The game snaps a
        piece home immediately, so resolving at the next pickup is accurate.
        """
        if self._eval_pending is None or self._eval_up_ts == 0.0:
            return
        if not force and time.monotonic() - self._eval_up_ts < self._settings.eval_settle_s:
            return
        pend = self._eval_pending
        self._eval_pending = None
        self._eval_up_ts = 0.0

        grid = self._ctx.artifacts.grid
        board_bbox = self._ctx.artifacts.board_bbox
        if self._board_state is None or grid is None or board_bbox is None:
            return
        crop = frame[
            board_bbox.y : board_bbox.y + board_bbox.h,
            board_bbox.x : board_bbox.x + board_bbox.w,
        ]
        self._board_state.update(crop, self._settings)  # fresh post-drop state
        newly = self._board_state.filled_cells() - pend["filled_before"]
        if len(newly) == 1:
            actual = list(next(iter(newly)))
            source = "diff"
        elif pend["drop"] is not None:
            # board-state didn't cleanly diff (0 / >1 new cells); fall back to
            # where the user released the piece (they place it where it belongs).
            actual = pend["drop"]
            source = "drop"
        else:
            plog.event("eval_skip", reason="newly_filled", n=len(newly), pred=pend["pred"])
            return
        top = pend["top"]
        plog.event(
            "eval",
            predicted=pend["pred"],
            top=top,
            actual=actual,
            correct=pend["pred"] == actual,
            recoverable=pend["pred"] is None and top == actual,  # reject that WOULD be right
            source=source,
            combined=pend["combined"],
            margin=pend["margin"],
            texture=pend["texture"],
            rejected=pend["rejected"],
        )
        # Ground-truth dataset capture: save the piece + reference board tagged
        # with the ACTUAL landing cell (physical, appearance-independent label).
        if self._recorder is not None and pend.get("piece_img") is not None:
            self._recorder.record_eval_sample(
                pend["piece_img"],
                pend["board_img"],
                {
                    "_region_img": pend.get("region_img"),
                    "_live_img": pend.get("live_img"),
                    "grid": {"cols": grid.cols, "rows": grid.rows,
                             "cell_w": grid.cell_w, "cell_h": grid.cell_h},
                    "actual_cell": actual,
                    "predicted_cell": pend["pred"],
                    "top_cell": top,
                    "actual_source": source,  # "diff" (reliable) or "drop" (fallback)
                    "combined": pend["combined"],
                    "margin": pend["margin"],
                    "texture": pend["texture"],
                    "rejected": pend["rejected"],
                },
            )

    def _drop_cell(self, sx: int, sy: int) -> list[int] | None:
        """The grid cell under the screen point ``(sx, sy)`` where a piece was
        released, or ``None`` if outside the board. Fallback ground truth."""
        wb = self._ctx.artifacts.window_bbox
        bb = self._ctx.artifacts.board_bbox
        grid = self._ctx.artifacts.grid
        if wb is None or bb is None or grid is None:
            return None
        bx = (sx - wb.x) - bb.x
        by = (sy - wb.y) - bb.y
        if not (0 <= bx < bb.w and 0 <= by < bb.h):
            return None
        col = int(min(grid.cols - 1, max(0, bx // grid.cell_w)))
        row = int(min(grid.rows - 1, max(0, by // grid.cell_h)))
        return [row, col]

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
                if self._eval_pending is not None:
                    self._eval_up_ts = time.monotonic()
                    self._eval_pending["drop"] = self._drop_cell(evt.x, evt.y)

    def _try_pickup_and_match(self, frame: Any, sx: int, sy: int) -> None:
        window_bbox = self._ctx.artifacts.window_bbox
        grid = self._ctx.artifacts.grid
        tmap = self._ctx.artifacts.target_map
        board_bbox = self._ctx.artifacts.board_bbox
        assert window_bbox is not None and grid is not None and tmap is not None
        assert board_bbox is not None
        # Resolve any evaluation still waiting on the settle timer before it is
        # overwritten below — otherwise fast play discards nearly every sample.
        if self._eval_pending is not None and self._eval_up_ts != 0.0:
            self._eval_placement(frame, force=True)

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
            result.piece.piece_full, tmap, self._settings, self._board_state,
            clipped_sides=result.piece.clipped_sides,
        )
        # Endgame rescue. Appearance stalls on the last few pieces, but the holes
        # left on the live board are shaped like the pieces that fill them. Only
        # consulted when appearance said nothing, so a correct overlay can never
        # be overturned — it can only supply one that was missing.
        shape_rescued = False
        if match.cell is None:
            live_board = frame[
                board_bbox.y:board_bbox.y + board_bbox.h,
                board_bbox.x:board_bbox.x + board_bbox.w,
            ]
            cell = rescue_by_hole_shape(
                _foreground_mask(result.piece.piece_full), live_board, grid, self._settings
            )
            if cell is not None:
                plog.event(
                    "hole_shape_rescue", cell=[cell.row, cell.col],
                    after=match.rejected_reason,
                )
                match = replace(match, cell=cell, rejected_reason=None)
                shape_rescued = True

        # Capture the prediction for self-supervised eval (resolved on drop).
        self._eval_pending = {
            "pred": [match.cell.row, match.cell.col] if match.cell else None,
            "top": list(match.top_cell) if match.top_cell else None,
            "combined": round(match.combined, 3),
            "margin": round(match.margin, 3),
            "texture": round(match.texture, 1),
            "rejected": match.rejected_reason,
            "shape_rescued": shape_rescued,
            "filled_before": (
                self._board_state.filled_cells() if self._board_state else frozenset()
            ),
            "drop": None,
            # Images for offline dataset capture (only kept when recording).
            "piece_img": result.piece.piece_full if self._recorder else None,
            "board_img": tmap.board_image if self._recorder else None,
            # Live board at pickup time — shows which neighbours are already
            # placed, which is what an interlocking (tab/blank) signal would
            # need. board_img is the *reference*, so it cannot answer that.
            "live_img": (
                frame[
                    board_bbox.y:board_bbox.y + board_bbox.h,
                    board_bbox.x:board_bbox.x + board_bbox.w,
                ].copy()
                if self._recorder else None
            ),
            # Raw pre-segmentation crop, so segmentation itself can be re-run
            # and diagnosed offline (this is where the sky-piece bug lived).
            "region_img": (
                # ``frame`` is already window-local (so is region_bbox) — adding
                # the window origin again pointed this at the wrong place and
                # made the recorded crops useless for offline diagnosis.
                frame[
                    result.region_bbox.y:result.region_bbox.y + result.region_bbox.h,
                    result.region_bbox.x:result.region_bbox.x + result.region_bbox.w,
                ].copy()
                if self._recorder else None
            ),
        }
        self._eval_up_ts = 0.0
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
