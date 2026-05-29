"""Iteration watchdog.

Brief §9: an iteration above ``iteration_warn_ms`` is logged at WARNING;
above ``iteration_abort_ms`` the caller should drop that frame's results.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager

from puzzle_assistant.config import Settings
from puzzle_assistant.utils import logger as plog


@contextmanager
def watch(name: str, settings: Settings) -> Iterator[None]:
    start = time.monotonic()
    try:
        yield
    finally:
        elapsed_ms = (time.monotonic() - start) * 1000.0
        if elapsed_ms >= settings.iteration_abort_ms:
            plog.event(
                "perf_abort",
                level=logging.ERROR,
                name=name,
                elapsed_ms=round(elapsed_ms, 1),
            )
        elif elapsed_ms >= settings.iteration_warn_ms:
            plog.event(
                "perf_warning",
                level=logging.WARNING,
                name=name,
                elapsed_ms=round(elapsed_ms, 1),
            )
