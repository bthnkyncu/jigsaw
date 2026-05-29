"""Phase 7 — overlay tests (headless variant)."""

from __future__ import annotations

from puzzle_assistant.overlay.headless_overlay import HeadlessOverlay
from puzzle_assistant.utils.coords import Bbox
from puzzle_assistant.utils.platform import make_overlay


def test_headless_overlay_records_show_and_hide() -> None:
    o = HeadlessOverlay()
    o.show(Bbox(100, 200, 50, 60))
    assert o.last_shown == Bbox(100, 200, 50, 60)
    o.hide()
    assert o.last_shown is None
    assert o.hide_count == 1
    o.shutdown()


def test_overlay_factory_returns_object() -> None:
    o = make_overlay("mock")
    assert hasattr(o, "show")
    assert hasattr(o, "hide")
