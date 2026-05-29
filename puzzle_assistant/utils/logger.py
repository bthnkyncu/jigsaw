"""JSON-line logger. Brief §9.

Every log line is a single JSON object so external tooling can stream-parse it.
Writes to stdout AND a rotating file under ``logs/``.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import time
from pathlib import Path
from typing import Any


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int((record.created % 1) * 1000):03d}Z",
            "lvl": record.levelname,
            "evt": getattr(record, "evt", record.getMessage()),
        }
        # Stash any structured fields the caller attached via `extra={...}`.
        for key, value in record.__dict__.items():
            if key in payload or key in _RESERVED_RECORD_FIELDS:
                continue
            payload[key] = _safe(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


_RESERVED_RECORD_FIELDS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "evt",
}


def _safe(value: Any) -> Any:
    """Make values JSON-serializable; fall back to ``str`` for exotic types."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    return str(value)


_INSTALLED = False


def install(log_dir: Path, log_filename: str, rotate_bytes: int, rotate_backups: int) -> None:
    """Configure the root logger. Idempotent across calls."""

    global _INSTALLED
    if _INSTALLED:
        return

    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = _JsonFormatter()

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Clear inherited handlers from libraries.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stdout_h = logging.StreamHandler(sys.stdout)
    stdout_h.setFormatter(formatter)
    root.addHandler(stdout_h)

    file_h = logging.handlers.RotatingFileHandler(
        log_dir / log_filename,
        maxBytes=rotate_bytes,
        backupCount=rotate_backups,
        encoding="utf-8",
    )
    file_h.setFormatter(formatter)
    root.addHandler(file_h)

    _INSTALLED = True


def event(evt: str, level: int = logging.INFO, **fields: Any) -> None:
    """Emit a single structured event."""

    # ``logging`` raises KeyError if an ``extra`` key shadows a built-in
    # LogRecord attribute (e.g. ``name``, ``module``). A crash from a *log*
    # call is unacceptable here, so rename any colliding keys instead.
    safe: dict[str, Any] = {"evt": evt}
    for key, value in fields.items():
        safe[f"{key}_" if key in _RESERVED_RECORD_FIELDS else key] = value
    logging.getLogger("puzzle_assistant").log(level, evt, extra=safe)
