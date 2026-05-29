"""Overlay + notifier protocol definitions."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from puzzle_assistant.utils.coords import Bbox


@runtime_checkable
class OverlayInterface(Protocol):
    def show(self, target: Bbox) -> None:
        """Display the click-through guidance rectangle over ``target``."""

    def hide(self) -> None: ...
    def shutdown(self) -> None: ...


@runtime_checkable
class NotifierInterface(Protocol):
    def notify(self, title: str, message: str, urgency: str = "normal") -> None: ...
