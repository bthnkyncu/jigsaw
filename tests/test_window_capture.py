"""Phase 1 — window capture tests."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from puzzle_assistant.capture.mock_capture import MockWindowCapture
from puzzle_assistant.utils.coords import Bbox
from puzzle_assistant.utils.platform import detect, make_window_capture


def test_mock_find_and_capture(tmp_path: Path) -> None:
    frame = np.full((720, 1280, 3), 200, dtype=np.uint8)
    frame[100:200, 100:200] = (0, 0, 255)  # red square
    fixture = tmp_path / "fake.png"
    cv2.imwrite(str(fixture), frame)

    cap = MockWindowCapture.from_png(fixture)
    candidates = cap.find_candidates("YapBoz")
    assert len(candidates) == 1
    assert candidates[0].handle == 1
    assert candidates[0].bbox.w == 1280
    assert candidates[0].bbox.h == 720

    bbox = cap.get_bbox(1)
    assert bbox is not None
    img = cap.capture(bbox)
    assert img.shape == (720, 1280, 3)
    # red pixel preserved
    assert tuple(int(c) for c in img[150, 150]) == (0, 0, 255)


def test_mock_capture_clips_to_frame() -> None:
    cap = MockWindowCapture()
    img = cap.capture(Bbox(0, 0, 10, 10))
    assert img.shape == (10, 10, 3)


def test_factory_returns_platform_appropriate() -> None:
    cap = make_window_capture("mock")
    assert cap.find_candidates("anything") != [] or cap.find_candidates("YapBoz") != []


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="X11 DISPLAY not set")
def test_linux_x11_capture_lists_windows() -> None:
    cap = make_window_capture("linux")
    # Even if no YapBoz window is open, the API should not raise.
    candidates = cap.find_candidates("DoesNotExistZZZ")
    assert isinstance(candidates, list)


def test_detect_returns_known_platform() -> None:
    assert detect() in ("linux", "windows", "mock")
