"""Phase 0 smoke tests: package imports + settings + logger boot."""

from __future__ import annotations

from pathlib import Path

from puzzle_assistant import __version__
from puzzle_assistant.config import Settings, load_settings
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.coords import Bbox, CellAddress, GridSpec, cell_bbox


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_settings_defaults_load() -> None:
    s = load_settings(None)
    assert isinstance(s, Settings)
    assert s.target_fps == 20
    assert s.game_window_title_substring == "YapBoz"
    assert s.expected_piece_count_min < s.expected_piece_count_max


def test_settings_override(tmp_path: Path) -> None:
    cfg = tmp_path / "settings.json"
    cfg.write_text('{"target_fps": 30, "unknown_key": 42}', encoding="utf-8")
    s = load_settings(cfg)
    assert s.target_fps == 30
    assert s.extra.get("unknown_key") == 42


def test_logger_install_idempotent(tmp_path: Path) -> None:
    plog.install(tmp_path, "smoke.log", rotate_bytes=1024, rotate_backups=1)
    plog.install(tmp_path, "smoke.log", rotate_bytes=1024, rotate_backups=1)
    plog.event("smoke_event", foo="bar")
    assert (tmp_path / "smoke.log").exists()


def test_bbox_contains_and_change() -> None:
    a = Bbox(10, 20, 100, 50)
    assert a.contains(50, 30)
    assert not a.contains(0, 0)
    b = Bbox(15, 22, 100, 50)
    assert a.relative_change_pct(b) < 10.0


def test_cell_bbox() -> None:
    board = Bbox(100, 100, 500, 200)
    grid = GridSpec(cols=25, rows=10, cell_w=20.0, cell_h=20.0)
    bb = cell_bbox(board, grid, CellAddress(row=5, col=12))
    assert bb.x == 100 + 12 * 20
    assert bb.y == 100 + 5 * 20
    assert bb.w == 20
    assert bb.h == 20
