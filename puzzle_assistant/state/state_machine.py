"""Top-level state machine.

Brief §5. States:

    IDLE → WAIT_FOR_NEW_PUZZLE → CALIBRATING_PRIMARY|FALLBACK → READY → TRACKING

The state object also keeps the calibrated artefacts (board bbox, grid spec,
target map, panel signature) — once a transition is taken those values are
available to the rest of the system without re-running detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum

from puzzle_assistant.calibration.reference_panel import PanelSignature
from puzzle_assistant.reference.target_map import TargetMap
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.coords import Bbox, GridSpec


class State(StrEnum):
    IDLE = "IDLE"
    WAIT_FOR_NEW_PUZZLE = "WAIT_FOR_NEW_PUZZLE"
    CALIBRATING_PRIMARY = "CALIBRATING_PRIMARY"
    CALIBRATING_FALLBACK = "CALIBRATING_FALLBACK"
    READY = "READY"
    TRACKING = "TRACKING"


@dataclass
class CalibrationArtifacts:
    window_bbox: Bbox | None = None
    board_bbox: Bbox | None = None
    grid: GridSpec | None = None
    panel_bbox: Bbox | None = None
    panel_signature: PanelSignature | None = None
    target_map: TargetMap | None = None


@dataclass
class Context:
    """Shared mutable state for the main loop."""

    state: State = State.IDLE
    artifacts: CalibrationArtifacts = field(default_factory=CalibrationArtifacts)
    fallback_warning_issued: bool = False
    last_tracking_target: Bbox | None = None


def transition(ctx: Context, to: State, *, reason: str = "") -> None:
    """Centralised state mutation with structured logging."""

    if ctx.state == to:
        return
    plog.event(
        "state_change",
        level=logging.INFO,
        **{"from": ctx.state.value, "to": to.value, "reason": reason},
    )
    ctx.state = to


def on_window_found(ctx: Context, bbox: Bbox) -> None:
    """The game window appeared (or moved)."""

    if ctx.artifacts.window_bbox is None or _bbox_drift_significant(
        ctx.artifacts.window_bbox, bbox, pct=5.0
    ):
        ctx.artifacts.window_bbox = bbox
        if ctx.state == State.IDLE:
            transition(ctx, State.WAIT_FOR_NEW_PUZZLE, reason="window_found")
        else:
            transition(ctx, State.WAIT_FOR_NEW_PUZZLE, reason="window_moved_or_resized")
            # Clear calibration; we'll rebuild on the new bbox.
            ctx.artifacts.board_bbox = None
            ctx.artifacts.grid = None
            ctx.artifacts.target_map = None


def on_window_lost(ctx: Context) -> None:
    """The game window disappeared."""

    ctx.artifacts.window_bbox = None
    ctx.artifacts.board_bbox = None
    ctx.artifacts.grid = None
    ctx.artifacts.target_map = None
    transition(ctx, State.IDLE, reason="window_lost")


def on_calibrated_primary(
    ctx: Context,
    board: Bbox,
    grid: GridSpec,
    target_map: TargetMap,
    panel_bbox: Bbox | None,
    panel_signature: PanelSignature | None,
) -> None:
    ctx.artifacts.board_bbox = board
    ctx.artifacts.grid = grid
    ctx.artifacts.target_map = target_map
    ctx.artifacts.panel_bbox = panel_bbox
    ctx.artifacts.panel_signature = panel_signature
    ctx.fallback_warning_issued = False
    transition(ctx, State.READY, reason="calibration_primary_complete")


def on_calibrated_fallback(
    ctx: Context,
    board: Bbox,
    grid: GridSpec,
    target_map: TargetMap,
    panel_bbox: Bbox | None,
    panel_signature: PanelSignature | None,
) -> None:
    ctx.artifacts.board_bbox = board
    ctx.artifacts.grid = grid
    ctx.artifacts.target_map = target_map
    ctx.artifacts.panel_bbox = panel_bbox
    ctx.artifacts.panel_signature = panel_signature
    transition(ctx, State.READY, reason="calibration_fallback_complete")


def on_panel_signature_changed(ctx: Context) -> None:
    """The reference panel changed → a new puzzle started."""

    plog.event("new_puzzle_detected")
    ctx.artifacts.board_bbox = None
    ctx.artifacts.grid = None
    ctx.artifacts.target_map = None
    ctx.artifacts.panel_signature = None
    ctx.last_tracking_target = None
    transition(ctx, State.WAIT_FOR_NEW_PUZZLE, reason="panel_signature_changed")


def on_board_bbox_drift(ctx: Context) -> None:
    """Board bbox moved noticeably — likely zoom or resize."""

    plog.event("board_bbox_drift", level=logging.WARNING)
    ctx.artifacts.board_bbox = None
    ctx.artifacts.grid = None
    ctx.artifacts.target_map = None
    transition(ctx, State.WAIT_FOR_NEW_PUZZLE, reason="board_bbox_drift")


def on_mouse_down(ctx: Context) -> None:
    if ctx.state == State.READY:
        transition(ctx, State.TRACKING, reason="mouse_down")


def on_mouse_up(ctx: Context) -> None:
    if ctx.state == State.TRACKING:
        ctx.last_tracking_target = None
        transition(ctx, State.READY, reason="mouse_up")


def _bbox_drift_significant(a: Bbox, b: Bbox, *, pct: float) -> bool:
    return a.relative_change_pct(b) > pct
