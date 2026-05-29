"""Phase 7 — state machine transitions."""

from __future__ import annotations

import numpy as np

from puzzle_assistant.calibration.reference_panel import PanelSignature
from puzzle_assistant.reference.target_map import CellFeatures, TargetMap
from puzzle_assistant.state.state_machine import (
    Context,
    State,
    on_board_bbox_drift,
    on_calibrated_primary,
    on_mouse_down,
    on_mouse_up,
    on_panel_signature_changed,
    on_window_found,
    on_window_lost,
)
from puzzle_assistant.utils.coords import Bbox, GridSpec


def _make_target_map() -> TargetMap:
    cell = CellFeatures(
        image=np.zeros((20, 20, 3), dtype=np.uint8),
        lab_mean=np.zeros(3, dtype=np.float32),
        orb_descriptors=None,
    )
    return TargetMap(
        grid=GridSpec(cols=2, rows=2, cell_w=20.0, cell_h=20.0),
        quality="primary",
        cells=[[cell, cell], [cell, cell]],
        board_image=np.zeros((40, 40, 3), dtype=np.uint8),
    )


def test_idle_to_wait_on_window_found() -> None:
    ctx = Context()
    assert ctx.state == State.IDLE
    on_window_found(ctx, Bbox(0, 0, 1920, 1080))
    assert ctx.state == State.WAIT_FOR_NEW_PUZZLE


def test_calibration_to_ready() -> None:
    ctx = Context()
    on_window_found(ctx, Bbox(0, 0, 1920, 1080))
    on_calibrated_primary(
        ctx,
        board=Bbox(500, 100, 700, 500),
        grid=GridSpec(cols=17, rows=14, cell_w=41.0, cell_h=35.7),
        target_map=_make_target_map(),
        panel_bbox=Bbox(1800, 100, 100, 80),
        panel_signature=PanelSignature(lab_mean=(0, 0, 0), dhash=0),
    )
    assert ctx.state == State.READY
    assert ctx.artifacts.target_map is not None


def test_mouse_down_up_round_trip() -> None:
    ctx = Context()
    ctx.state = State.READY
    on_mouse_down(ctx)
    assert ctx.state == State.TRACKING
    on_mouse_up(ctx)
    assert ctx.state == State.READY


def test_panel_change_triggers_recalibrate() -> None:
    ctx = Context()
    ctx.state = State.READY
    ctx.artifacts.target_map = _make_target_map()
    on_panel_signature_changed(ctx)
    assert ctx.state == State.WAIT_FOR_NEW_PUZZLE
    assert ctx.artifacts.target_map is None


def test_window_lost_clears_artifacts() -> None:
    ctx = Context()
    ctx.state = State.READY
    ctx.artifacts.window_bbox = Bbox(0, 0, 100, 100)
    ctx.artifacts.target_map = _make_target_map()
    on_window_lost(ctx)
    assert ctx.state == State.IDLE
    assert ctx.artifacts.window_bbox is None
    assert ctx.artifacts.target_map is None


def test_board_bbox_drift_triggers_recalibrate() -> None:
    ctx = Context()
    ctx.state = State.READY
    ctx.artifacts.board_bbox = Bbox(500, 100, 700, 500)
    on_board_bbox_drift(ctx)
    assert ctx.state == State.WAIT_FOR_NEW_PUZZLE
    assert ctx.artifacts.board_bbox is None
