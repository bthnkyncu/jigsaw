"""User-facing notifications.

Linux backend uses ``notify-send`` (libnotify) if available; otherwise it falls
back to stderr. Windows backend will be wired up in Phase 9.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys

from puzzle_assistant.utils import logger as plog


class ConsoleNotifier:
    """No-GUI notifier used in tests and headless mode."""

    def notify(self, title: str, message: str, urgency: str = "normal") -> None:
        plog.event("notify", title=title, notify_message=message, urgency=urgency)
        print(f"[{urgency.upper()}] {title}: {message}", file=sys.stderr)


class LinuxNotifier:
    """Linux libnotify backend with stderr fallback."""

    def __init__(self) -> None:
        self._notify_send = shutil.which("notify-send")

    def notify(self, title: str, message: str, urgency: str = "normal") -> None:
        plog.event("notify", title=title, notify_message=message, urgency=urgency)
        if self._notify_send is None:
            print(f"[{urgency.upper()}] {title}: {message}", file=sys.stderr)
            return
        try:
            subprocess.run(
                [
                    self._notify_send,
                    "--app-name=YapbozAsistani",
                    f"--urgency={urgency}",
                    title,
                    message,
                ],
                timeout=2.0,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            plog.event("notify_failed", level=logging.WARNING, error=str(exc))


class WindowsNotifier:
    """Phase 9 placeholder — replaced by ``win10toast`` integration."""

    def notify(self, title: str, message: str, urgency: str = "normal") -> None:
        plog.event("notify_windows_pending", title=title, notify_message=message)
        print(f"[{urgency.upper()}] {title}: {message}", file=sys.stderr)
