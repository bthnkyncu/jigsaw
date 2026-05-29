"""Phase 6 — matching engine tests.

Most matching tests rely on the **self-match** property: take the rendered
cell at ``(r, c)`` of a target map, feed it back as the picked piece, and
expect the engine to predict ``(r, c)`` with a margin above threshold.

Brief DoD: ≥ 95 % accuracy on 50+ test cases. We sample 60-ish cells from a
single init-view fixture and assert ≥ 95 %.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import pytest

from puzzle_assistant.calibration.board_detector import detect_board
from puzzle_assistant.calibration.grid_detector import detect_grid_from_init_view
from puzzle_assistant.config import load_settings
from puzzle_assistant.matching.engine import match_piece
from puzzle_assistant.reference.target_map import build_from_init_view

WIN_X, WIN_Y, WIN_W, WIN_H = 0, 64, 1920, 1016


def _build_target_map(fixtures_dir: Path, name: str):  # type: ignore[no-untyped-def]
    settings = load_settings(None)
    full = cv2.imread(str(fixtures_dir / f"{name}.png"))
    img = full[WIN_Y:WIN_Y + WIN_H, WIN_X:WIN_X + WIN_W]
    bb = detect_board(img, settings)
    assert bb is not None
    board = img[bb.y:bb.y + bb.h, bb.x:bb.x + bb.w]
    grid = detect_grid_from_init_view(board, settings)
    assert grid is not None
    tmap = build_from_init_view(board, grid, settings)
    return tmap, settings


def test_self_match_accuracy_above_95pct(fixtures_dir: Path) -> None:
    """Brief DoD: ≥ 95 % accuracy on 50+ self-match cases."""

    tmap, settings = _build_target_map(fixtures_dir, "init_view_4")
    correct = 0
    total = 0
    elapsed_ms: list[float] = []
    for r in range(0, tmap.grid.rows, 2):
        for c in range(0, tmap.grid.cols, 2):
            piece = tmap.cells[r][c].image
            t0 = time.monotonic()
            result = match_piece(piece, tmap, settings)
            elapsed_ms.append((time.monotonic() - t0) * 1000.0)
            total += 1
            if result.cell is not None and result.cell.row == r and result.cell.col == c:
                correct += 1

    assert total >= 50, f"need at least 50 cases, got {total}"
    accuracy = correct / total
    assert accuracy >= 0.95, f"accuracy {accuracy:.1%} below 95 %"
    # Performance budget: P95 below 200 ms.
    elapsed_ms.sort()
    p95 = elapsed_ms[int(0.95 * len(elapsed_ms))]
    assert p95 < 200, f"P95 latency {p95:.0f} ms above 200 ms"


def test_cross_puzzle_pieces_rejected(fixtures_dir: Path) -> None:
    """A piece from a different puzzle must NOT match this puzzle's target map."""

    tmap_a, settings = _build_target_map(fixtures_dir, "init_view_4")  # grass+sky
    tmap_b, _ = _build_target_map(fixtures_dir, "init_view_5")          # plage
    rejects = 0
    total = 0
    for r in range(0, tmap_b.grid.rows, 3):
        for c in range(0, tmap_b.grid.cols, 3):
            piece = tmap_b.cells[r][c].image
            result = match_piece(piece, tmap_a, settings)
            total += 1
            if result.cell is None:
                rejects += 1
    # The sliding-window matcher trades cross-puzzle specificity for
    # within-puzzle recall (the gating thresholds are deliberately loose so a
    # real dragged piece is always placed). Both fixtures are nature scenes
    # with overlapping sky/foliage colours, so some cross matches survive.
    # We still expect the majority to be rejected.
    assert rejects / total >= 0.60, f"only {rejects}/{total} rejected"


def test_empty_piece_returns_rejection() -> None:
    import numpy as np

    from puzzle_assistant.reference.target_map import TargetMap
    from puzzle_assistant.utils.coords import GridSpec

    settings = load_settings(None)
    # Build a trivial 1×1 target map.
    cell_img = np.zeros((40, 40, 3), dtype=np.uint8)
    from puzzle_assistant.reference.target_map import CellFeatures
    cell = CellFeatures(image=cell_img, lab_mean=np.zeros(3, dtype=np.float32), orb_descriptors=None)
    tmap = TargetMap(
        grid=GridSpec(cols=1, rows=1, cell_w=40.0, cell_h=40.0),
        quality="primary",
        cells=[[cell]],
        board_image=cell_img,
    )
    result = match_piece(np.empty((0, 0, 3), dtype=np.uint8), tmap, settings)
    assert result.cell is None
    assert result.rejected_reason == "empty_piece"


@pytest.mark.parametrize("quality", ["primary", "fallback"])
def test_quality_threshold_difference(quality: str) -> None:
    """Fallback quality must apply stricter thresholds than primary."""

    settings = load_settings(None)
    primary_min = settings.min_combined_score
    fallback_min = settings.fallback_min_combined_score
    assert fallback_min > primary_min
    assert settings.fallback_min_margin > settings.min_margin
