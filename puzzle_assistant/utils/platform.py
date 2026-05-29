"""Factory functions selecting the right platform backend at runtime.

Every platform-specific module is imported lazily so a Linux-only checkout still
type-checks and runs without ``pywin32`` installed.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from puzzle_assistant.capture.interfaces import (
        MouseHookInterface,
        WindowCaptureInterface,
    )
    from puzzle_assistant.overlay.interfaces import NotifierInterface, OverlayInterface

Platform = Literal["linux", "windows", "mock"]


def detect() -> Platform:
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "win32":
        return "windows"
    return "mock"


def make_window_capture(platform: Platform | None = None) -> WindowCaptureInterface:
    plat = platform or detect()
    if plat == "linux":
        from puzzle_assistant.capture.linux_x11_capture import LinuxX11WindowCapture
        return cast("WindowCaptureInterface", LinuxX11WindowCapture())
    if plat == "windows":
        from puzzle_assistant.capture.windows_capture import WindowsWindowCapture
        return cast("WindowCaptureInterface", WindowsWindowCapture())
    from puzzle_assistant.capture.mock_capture import MockWindowCapture
    return cast("WindowCaptureInterface", MockWindowCapture())


def make_mouse_hook(platform: Platform | None = None) -> MouseHookInterface:
    plat = platform or detect()
    if plat == "linux":
        from puzzle_assistant.capture.linux_mouse_hook import LinuxMouseHook
        return cast("MouseHookInterface", LinuxMouseHook())
    if plat == "windows":
        from puzzle_assistant.capture.windows_mouse_hook import WindowsMouseHook
        return cast("MouseHookInterface", WindowsMouseHook())
    from puzzle_assistant.capture.mock_mouse_hook import MockMouseHook
    return cast("MouseHookInterface", MockMouseHook())


def make_overlay(platform: Platform | None = None) -> OverlayInterface:
    plat = platform or detect()
    if plat == "linux":
        from puzzle_assistant.overlay.linux_overlay import LinuxOverlay
        return cast("OverlayInterface", LinuxOverlay())
    if plat == "windows":
        from puzzle_assistant.overlay.windows_overlay import WindowsOverlay
        return cast("OverlayInterface", WindowsOverlay())
    from puzzle_assistant.overlay.headless_overlay import HeadlessOverlay
    return cast("OverlayInterface", HeadlessOverlay())


def make_notifier(platform: Platform | None = None) -> NotifierInterface:
    plat = platform or detect()
    if plat == "linux":
        from puzzle_assistant.utils.notifier import LinuxNotifier
        return cast("NotifierInterface", LinuxNotifier())
    if plat == "windows":
        from puzzle_assistant.utils.notifier import WindowsNotifier
        return cast("NotifierInterface", WindowsNotifier())
    from puzzle_assistant.utils.notifier import ConsoleNotifier
    return cast("NotifierInterface", ConsoleNotifier())
