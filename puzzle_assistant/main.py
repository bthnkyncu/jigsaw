"""Entry point.

Builds the platform-specific subsystems (capture, mouse hook, overlay,
notifier) via the platform factory and hands them to ``MainLoop``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from puzzle_assistant.config import Settings, load_settings
from puzzle_assistant.state.main_loop import MainLoop
from puzzle_assistant.utils import dpi
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.platform import (
    Platform,
    make_mouse_hook,
    make_notifier,
    make_overlay,
    make_window_capture,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yapboz-asistani",
        description="Gamyun YapBoz Salonu visual matching assistant.",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional JSON file overriding values from config/defaults.py.",
    )
    parser.add_argument(
        "--test-mode", action="store_true",
        help="Run a one-shot pipeline against fixture PNGs and exit.",
    )
    parser.add_argument("--init-view", type=Path, default=None)
    parser.add_argument("--fixture", type=Path, default=None)
    parser.add_argument("--cursor", type=str, default=None)
    parser.add_argument(
        "--platform-override",
        choices=["linux", "windows", "mock"],
        default=None,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    settings = load_settings(args.config)

    plog.install(
        log_dir=Path(settings.log_dir),
        log_filename=settings.log_filename,
        rotate_bytes=settings.log_rotate_bytes,
        rotate_backups=settings.log_rotate_backups,
    )
    plog.event("boot", platform_override=args.platform_override, test_mode=args.test_mode)
    dpi.ensure_compatible_scale(settings.expected_dpi_scale)

    platform: Platform | None = args.platform_override
    if args.test_mode:
        return _run_test_mode(args, settings, platform)

    return _run_live(settings, platform)


def _run_live(settings: Settings, platform: Platform | None) -> int:
    capture = make_window_capture(platform)
    hook = make_mouse_hook(platform)
    overlay = make_overlay(platform)
    notifier = make_notifier(platform)
    loop = MainLoop(settings, capture, hook, overlay, notifier)
    try:
        loop.run()
    except KeyboardInterrupt:
        plog.event("user_interrupt")
        loop.stop()
    return 0


def _run_test_mode(
    args: argparse.Namespace, settings: Settings, platform: Platform | None
) -> int:
    plog.event(
        "test_mode_stub",
        init_view=str(args.init_view),
        fixture=str(args.fixture),
        cursor=args.cursor,
    )
    # The real test-mode pipeline is exercised through pytest fixtures; this
    # CLI flag stays for ad-hoc debugging.
    return 0


if __name__ == "__main__":
    sys.exit(main())
