"""Phase 5 — segmentation + pickup + group classification tests."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from puzzle_assistant.config import load_settings
from puzzle_assistant.piece.group_detection import classify
from puzzle_assistant.piece.pickup import pickup_from_window
from puzzle_assistant.piece.segmentation import extract_piece
from puzzle_assistant.utils.coords import GridSpec

WIN_X, WIN_Y, WIN_W, WIN_H = 0, 64, 1920, 1016


def _window_crop(fixtures_dir: Path, name: str) -> "cv2.typing.MatLike":
    full = cv2.imread(str(fixtures_dir / f"{name}.png"))
    return full[WIN_Y:WIN_Y + WIN_H, WIN_X:WIN_X + WIN_W]


def test_extract_single_piece_from_board_center(fixtures_dir: Path) -> None:
    settings = load_settings(None)
    img = _window_crop(fixtures_dir, "dragging_single_1")
    grid = GridSpec(cols=17, rows=14, cell_w=744 / 17, cell_h=558 / 14)

    # Cursor right on top of the piece sitting in board center.
    result = pickup_from_window(img, (905, 390), grid, settings)
    assert result is not None, "piece should be segmented"
    bb = result.piece.bbox
    # bbox is relative to the pickup region; just sanity-check size.
    assert 20 < bb.w < 120, f"unexpected piece width {bb.w}"
    assert 20 < bb.h < 120, f"unexpected piece height {bb.h}"
    assert result.piece.area_px > 800


def test_group_classification_thresholds(fixtures_dir: Path) -> None:
    settings = load_settings(None)
    grid = GridSpec(cols=17, rows=14, cell_w=744 / 17, cell_h=558 / 14)

    # Synthesise a single-piece PickedPiece (small bbox + area).
    import numpy as np

    from puzzle_assistant.piece.segmentation import PickedPiece
    from puzzle_assistant.utils.coords import Bbox
    small = PickedPiece(
        piece_full=np.zeros((40, 40, 3), dtype=np.uint8),
        piece_core=np.zeros((40, 40, 3), dtype=np.uint8),
        bbox=Bbox(0, 0, 40, 40),
        area_px=1500,
    )
    assert classify(small, grid, settings) == "single"

    big = PickedPiece(
        piece_full=np.zeros((120, 120, 3), dtype=np.uint8),
        piece_core=np.zeros((120, 120, 3), dtype=np.uint8),
        bbox=Bbox(0, 0, 120, 120),
        area_px=10000,
    )
    assert classify(big, grid, settings) == "group"


def test_extract_returns_none_on_empty_desk(fixtures_dir: Path) -> None:
    """A click on the bare desk should not segment anything piece-like."""

    settings = load_settings(None)
    img = _window_crop(fixtures_dir, "mid_game_1")
    grid = GridSpec(cols=17, rows=14, cell_w=744 / 17, cell_h=558 / 14)
    # Click smack in the middle of the empty board interior.
    result = pickup_from_window(img, (900, 400), grid, settings)
    # Either no piece, or a tiny noise blob.
    if result is not None:
        assert result.piece.area_px < 800


def test_desk_coloured_piece_is_not_erased() -> None:
    """A piece whose image content shares the desk's hue must survive segmentation.

    Regression guard for the biggest accuracy bug found in the field: the desk
    is matched with a *fixed* blue hue band, and a puzzle containing sky/water
    falls in that same band — so ~40 % of a sky piece was erased as background
    and the matcher got a sliver (a real capture came out 17 px tall instead of
    ~68), which localises nowhere and yields low-margin rejects or wrong cells.
    Segmentation now measures the desk colour off the crop border and requires
    flatness, so same-hue content is kept.
    """
    settings = load_settings(None)
    desk = (210, 174, 150)  # the real Gamyun desk colour (measured std = 0)
    cell_w, cell_h = 53, 48

    # Sky-like content: the desk hue with real texture on top.
    rng = np.random.default_rng(0)
    cell = np.full((cell_h, cell_w, 3), desk, np.uint8)
    noise = rng.normal(0, 14, (cell_h, cell_w, 3))
    grad = np.linspace(-18, 18, cell_h)[:, None, None]
    cell = np.clip(cell.astype(np.float32) + noise + grad, 0, 255).astype(np.uint8)

    scene = np.full((int(cell_h * 2.4), int(cell_w * 2.4), 3), desk, np.uint8)
    oy = (scene.shape[0] - cell_h) // 2
    ox = (scene.shape[1] - cell_w) // 2
    scene[oy:oy + cell_h, ox:ox + cell_w] = cell

    picked = extract_piece(
        scene, (scene.shape[1] // 2, scene.shape[0] // 2), settings,
        expected_cell=(cell_w, cell_h),
    )
    assert picked is not None, "sky-coloured piece was erased entirely"
    h, w = picked.piece_full.shape[:2]
    # Must recover essentially the whole piece, not a sliver.
    assert h >= cell_h * 0.8, f"piece height {h} collapsed (expected >= {cell_h * 0.8:.0f})"
    assert w >= cell_w * 0.8, f"piece width {w} collapsed (expected >= {cell_w * 0.8:.0f})"


def test_clipped_sides_reports_the_capture_window_edge() -> None:
    """Segmentation must say which sides the capture window cut off.

    The matcher suppresses flat-edge detection on a cut-off side, because such a
    side looks straight by artefact. It used to derive that from the piece crop
    itself — but that crop is tight to the piece, so every side touches its own
    border and the answer was always "all four clipped", which silently disabled
    the border constraint for every piece. Only segmentation, which sees the
    piece inside the whole region, can answer it.
    """
    settings = load_settings(None)
    desk = (210, 174, 150)
    cell_w, cell_h = 53, 48
    rng = np.random.default_rng(1)

    def scene(offset_x: int) -> np.ndarray:
        canvas = np.full((int(cell_h * 2.4), int(cell_w * 2.4), 3), desk, np.uint8)
        content = np.clip(
            np.full((cell_h, cell_w, 3), (90, 140, 70), np.float32)
            + rng.normal(0, 15, (cell_h, cell_w, 3)),
            0, 255,
        ).astype(np.uint8)
        top = (canvas.shape[0] - cell_h) // 2
        canvas[top:top + cell_h, offset_x:offset_x + cell_w] = content
        return canvas

    centred = scene((int(cell_w * 2.4) - cell_w) // 2)
    picked = extract_piece(
        centred, (centred.shape[1] // 2, centred.shape[0] // 2), settings,
        expected_cell=(cell_w, cell_h),
    )
    assert picked is not None
    assert not any(picked.clipped_sides), "a fully visible piece is not clipped"

    against_left = scene(0)
    picked = extract_piece(
        against_left, (cell_w // 2, against_left.shape[0] // 2), settings,
        expected_cell=(cell_w, cell_h),
    )
    assert picked is not None
    assert picked.clipped_sides[2], "piece touching the left window edge is clipped there"
