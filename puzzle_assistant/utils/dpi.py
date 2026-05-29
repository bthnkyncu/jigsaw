"""DPI / scale awareness — Linux X11 today, Windows in Phase 9."""

from __future__ import annotations

import logging
import os
import subprocess
import sys

from puzzle_assistant.utils import logger as plog


def ensure_compatible_scale(expected_scale: float) -> None:
    """Verify the desktop scale matches expectation; log a warning if not."""

    if sys.platform == "win32":
        _windows_dpi_awareness()
        return

    actual = _detect_linux_scale()
    if actual is None:
        plog.event("dpi_check_skipped", reason="no_method")
        return
    if abs(actual - expected_scale) > 0.05:
        plog.event(
            "dpi_mismatch",
            level=logging.WARNING,
            expected=expected_scale,
            actual=actual,
        )
    else:
        plog.event("dpi_ok", scale=actual)


def _detect_linux_scale() -> float | None:
    """Best-effort X11 scale detection via ``xrandr``.

    Returns 1.0 for unscaled displays. ``None`` if detection failed entirely.
    """

    if not os.environ.get("DISPLAY"):
        return None
    try:
        result = subprocess.run(
            ["xrandr", "--current"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    # xrandr does not report a "scale" directly; assume 1.0 unless transform is set.
    if "Transform" in result.stdout or "scale" in result.stdout.lower():
        return None
    return 1.0


def _windows_dpi_awareness() -> None:
    """Phase 9 placeholder. Real implementation lives in ``capture/windows_capture.py``."""

    plog.event("dpi_windows_pending", note="implemented_in_phase_9")
