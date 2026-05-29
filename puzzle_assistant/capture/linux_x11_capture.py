"""Linux X11 implementation of WindowCaptureInterface.

Walks the X11 window tree, matches ``_NET_WM_NAME``/``WM_NAME`` against the
user-provided substring, and grabs screen pixels with ``mss``. Works only under
X11 / XWayland — pure Wayland sessions need a different backend (Phase 9+).

Capture uses XComposite (window backing-store pixmap) when available so that
the game window's actual content is captured even when another window sits on
top of it. Falls back to mss (screen-region grab) if XComposite is unavailable
or the window doesn't support redirection.
"""

from __future__ import annotations

import logging
from typing import Any, cast

import mss
import numpy as np
from Xlib import X, display
from Xlib.error import BadDrawable, BadMatch, BadWindow, XError
from Xlib.ext import composite
from Xlib.xobject.drawable import Window

from puzzle_assistant.capture.interfaces import WindowCandidate
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.coords import Bbox


class LinuxX11WindowCapture:
    """Find + screenshot a top-level X11 window by title substring."""

    def __init__(self) -> None:
        self._display = display.Display()
        self._root = self._display.screen().root
        self._net_wm_name = self._display.intern_atom("_NET_WM_NAME")
        self._utf8_string = self._display.intern_atom("UTF8_STRING")
        self._mss: mss.base.MSSBase | None = None
        # Track which handles have been successfully redirected via XComposite.
        self._composite_handles: set[int] = set()
        self._composite_available: bool | None = None  # None = not yet tested

    # ----------------------------- public API -----------------------------

    def find_candidates(self, title_substring: str) -> list[WindowCandidate]:
        try:
            candidates: list[WindowCandidate] = []
            self._walk(self._root, title_substring, candidates)
            return candidates
        except (BadWindow, BadDrawable) as exc:
            plog.event("x11_walk_failed", level=logging.WARNING, error=str(exc))
            return []

    def get_bbox(self, handle: int) -> Bbox | None:
        win = self._window_by_id(handle)
        if win is None:
            return None
        bbox = self._absolute_bbox(win)
        if bbox is None or bbox.w <= 0 or bbox.h <= 0:
            return None
        return bbox

    def capture(self, bbox: Bbox) -> np.ndarray:
        """Capture the screen region via mss (always reads visible pixels)."""
        if self._mss is None:
            self._mss = mss.mss()
        region = {"left": bbox.x, "top": bbox.y, "width": bbox.w, "height": bbox.h}
        shot = self._mss.grab(region)
        # ``shot.raw`` is BGRA bytes; convert to BGR ``np.ndarray``.
        arr = np.frombuffer(shot.rgb, dtype=np.uint8).reshape(shot.height, shot.width, 3)
        # mss returns RGB order; OpenCV pipeline expects BGR.
        return arr[:, :, ::-1].copy()

    def capture_window(self, handle: int, w: int, h: int) -> np.ndarray | None:
        """Capture a window's backing-store content via XComposite.

        Returns the window's actual pixel content regardless of occlusion, or
        ``None`` if XComposite is unavailable / fails for this window.
        """
        if self._composite_available is False:
            return None
        win = self._window_by_id(handle)
        if win is None:
            return None
        try:
            if handle not in self._composite_handles:
                composite.redirect_window(win, composite.RedirectAutomatic)
                self._display.sync()
                self._composite_handles.add(handle)
                self._composite_available = True

            pixmap = composite.name_window_pixmap(win)
            self._display.sync()
            image: Any = pixmap.get_image(0, 0, w, h, X.ZPixmap, 0xFFFFFFFF)
            arr = np.frombuffer(image.data, dtype=np.uint8).reshape(h, w, 4)
            # XImage is BGRA; drop alpha and copy.
            bgr: np.ndarray = arr[:, :, :3].copy()
            pixmap.free()
            return bgr
        except (BadWindow, BadDrawable, BadMatch, XError) as exc:
            plog.event(
                "composite_capture_failed", level=logging.DEBUG, handle=handle, error=str(exc)
            )
            if handle in self._composite_handles:
                self._composite_handles.discard(handle)
            if self._composite_available is None:
                self._composite_available = False
            return None

    # --------------------------- internals --------------------------------

    def _walk(
        self,
        window: Window,
        title_substring: str,
        out: list[WindowCandidate],
    ) -> None:
        title = self._window_title(window)
        if title and title_substring in title:
            bbox = self._absolute_bbox(window)
            if bbox is not None and bbox.area > 0:
                out.append(
                    WindowCandidate(handle=int(window.id), title=title, bbox=bbox)
                )
        try:
            children = window.query_tree().children
        except (BadWindow, BadDrawable):
            return
        for child in children:
            self._walk(child, title_substring, out)

    def _window_title(self, window: Window) -> str | None:
        try:
            prop = window.get_full_property(self._net_wm_name, self._utf8_string)
            if prop and prop.value:
                value = prop.value
                if isinstance(value, bytes):
                    return value.decode("utf-8", errors="replace")
                return str(value)
            # Fallback to legacy WM_NAME.
            name = window.get_wm_name()
            if isinstance(name, bytes):
                return name.decode("utf-8", errors="replace")
            return cast("str | None", name)
        except (BadWindow, BadDrawable, UnicodeDecodeError):
            return None

    def _absolute_bbox(self, window: Window) -> Bbox | None:
        try:
            geom = window.get_geometry()
        except (BadWindow, BadDrawable):
            return None
        # Translate (0,0) of this window into root coordinates.
        try:
            translated = window.translate_coords(self._root, 0, 0)
        except (BadWindow, BadDrawable):
            return None
        # ``translate_coords`` returns ``-(abs_x, abs_y)`` when going window→root
        # via Xlib semantics, so we negate to get screen-absolute origin.
        abs_x = -translated.x
        abs_y = -translated.y
        return Bbox(x=abs_x, y=abs_y, w=int(geom.width), h=int(geom.height))

    def _window_by_id(self, wid: int) -> Window | None:
        try:
            return self._display.create_resource_object("window", wid)
        except (BadWindow, BadDrawable, ValueError) as exc:
            plog.event("x11_window_by_id_failed", level=logging.WARNING, wid=wid, error=str(exc))
            return None

    def raise_window(self, handle: int) -> None:
        """Bring the window to the front so mss captures its actual content."""
        win = self._window_by_id(handle)
        if win is None:
            return
        try:
            win.raise_window()
            self._display.flush()
        except (BadWindow, BadDrawable):
            pass

    def close(self) -> None:
        if self._mss is not None:
            self._mss.close()
            self._mss = None
        self._display.close()


def main_cli() -> int:
    """CLI entry: list candidate windows matching a substring."""

    import argparse

    parser = argparse.ArgumentParser(description="List X11 windows matching a title substring.")
    parser.add_argument("--substring", default="YapBoz")
    args = parser.parse_args()

    cap = LinuxX11WindowCapture()
    try:
        candidates = cap.find_candidates(args.substring)
        if not candidates:
            print(f"No window with title containing {args.substring!r}.")
            return 1
        for i, c in enumerate(candidates):
            print(
                f"[{i}] handle=0x{c.handle:x} bbox={c.bbox.x},{c.bbox.y},"
                f"{c.bbox.w}x{c.bbox.h} title={c.title!r}"
            )
        return 0
    finally:
        cap.close()


if __name__ == "__main__":
    raise SystemExit(main_cli())
