"""Phase 3 — reference panel detection + signature change tests."""

from __future__ import annotations

from pathlib import Path

import cv2

from puzzle_assistant.calibration.reference_panel import (
    compute_signature,
    detect_reference_panel,
    upscale_panel_to_board,
)
from puzzle_assistant.config import load_settings

WIN_X, WIN_Y, WIN_W, WIN_H = 0, 64, 1920, 1016


def _window_crop(fixtures_dir: Path, name: str) -> "cv2.typing.MatLike":
    full = cv2.imread(str(fixtures_dir / f"{name}.png"))
    return full[WIN_Y:WIN_Y + WIN_H, WIN_X:WIN_X + WIN_W]


def test_panel_detected_on_known_fixtures(fixtures_dir: Path) -> None:
    settings = load_settings(None)
    for name in ["init_view_1", "init_view_4", "mid_game_1", "dragging_single_1"]:
        img = _window_crop(fixtures_dir, name)
        bbox = detect_reference_panel(img, settings)
        assert bbox is not None, f"panel not detected in {name}"
        # Panel sits in the upper-right column. Sanity bounds.
        assert bbox.x > WIN_W * 0.85
        assert bbox.y < WIN_H * 0.30
        assert bbox.w >= 60 and bbox.h >= 60


def test_signature_distance_zero_for_same_image(fixtures_dir: Path) -> None:
    settings = load_settings(None)
    img = _window_crop(fixtures_dir, "mid_game_1")
    bbox = detect_reference_panel(img, settings)
    assert bbox is not None
    crop = img[bbox.y:bbox.y + bbox.h, bbox.x:bbox.x + bbox.w]
    sig_a = compute_signature(crop)
    sig_b = compute_signature(crop)
    assert sig_a.distance(sig_b) == 0.0


def test_signature_distance_nonzero_across_puzzles(fixtures_dir: Path) -> None:
    """Two visually different puzzles must give a signature distance above zero."""

    settings = load_settings(None)
    img_a = _window_crop(fixtures_dir, "init_view_1")  # car gauge
    img_b = _window_crop(fixtures_dir, "init_view_4")  # green grass + sky
    bbox_a = detect_reference_panel(img_a, settings)
    bbox_b = detect_reference_panel(img_b, settings)
    assert bbox_a is not None and bbox_b is not None
    crop_a = img_a[bbox_a.y:bbox_a.y + bbox_a.h, bbox_a.x:bbox_a.x + bbox_a.w]
    crop_b = img_b[bbox_b.y:bbox_b.y + bbox_b.h, bbox_b.x:bbox_b.x + bbox_b.w]
    sig_a = compute_signature(crop_a)
    sig_b = compute_signature(crop_b)
    assert sig_a.distance(sig_b) > 0.10


def test_upscale_panel_to_board(fixtures_dir: Path) -> None:
    settings = load_settings(None)
    img = _window_crop(fixtures_dir, "mid_game_1")
    bbox = detect_reference_panel(img, settings)
    assert bbox is not None
    crop = img[bbox.y:bbox.y + bbox.h, bbox.x:bbox.x + bbox.w]
    up = upscale_panel_to_board(crop, 744, 558)
    assert up.shape == (558, 744, 3)
