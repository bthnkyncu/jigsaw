"""Overlay rendered as a Toplevel of the GUI's single Tk root (main thread).

Tk is not thread-safe and a process must have only ONE ``Tk()`` (on the main
thread). The standalone Linux/Windows overlays each create their own ``tk.Tk()``
on a background thread — fine when they are the only Tk in the process (the
headless CLI), but once the CustomTkinter GUI owns a main-thread root, a second
root on another thread is unsupported and left the green box invisible
(especially on Windows: mouse-hook + matching worked, yet nothing was drawn).

This overlay fixes that: it is a ``Toplevel`` of the GUI's existing root and
does ALL Tk work on the main thread. ``show()/hide()/shutdown()`` are
thread-safe (they only enqueue a command); the GUI root's ``after`` loop
(:meth:`_pump`) drains the queue and draws. Click-through is applied on the main
thread (Win32 extended styles on Windows, X11 XShape on Linux).
"""

from __future__ import annotations

import contextlib
import logging
import queue
import sys
import tkinter as tk
from dataclasses import dataclass
from typing import Any, Literal

from puzzle_assistant.config import Settings
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.coords import Bbox


@dataclass
class _Command:
    kind: Literal["show", "hide", "shutdown"]
    bbox: Bbox | None = None


class GuiOverlay:
    """Single-root, main-thread overlay driven by a thread-safe command queue."""

    def __init__(self, root: Any, settings: Settings) -> None:
        self._root = root
        self._settings = settings
        self._cmd_q: queue.Queue[_Command] = queue.Queue()
        self._top: tk.Toplevel | None = None
        self._canvas: tk.Canvas | None = None
        self._visible = False
        self._bright = True
        # Drive both loops on the GUI's main-thread event loop.
        root.after(30, self._pump)
        root.after(settings.overlay_blink_interval_ms, self._blink)

    # --- thread-safe API (called from the MainLoop worker thread) ---------

    def show(self, target: Bbox) -> None:
        self._cmd_q.put(_Command("show", target))

    def hide(self) -> None:
        self._cmd_q.put(_Command("hide"))

    def shutdown(self) -> None:
        # The GUI owns the root for the whole app lifetime; just hide so a later
        # BAŞLAT reuses the same Toplevel. Do not stop the pump/blink loops.
        self._cmd_q.put(_Command("hide"))

    # --- main-thread internals --------------------------------------------

    def _ensure_window(self) -> None:
        if self._top is not None:
            return
        top = tk.Toplevel(self._root)
        top.overrideredirect(True)
        top.attributes("-topmost", True)
        with contextlib.suppress(tk.TclError):
            # Windows-only: makes the black background fully transparent.
            top.attributes("-transparentcolor", "black")
        top.attributes("-alpha", self._settings.overlay_alpha_high)
        top.config(bg="black")
        canvas = tk.Canvas(top, bg="black", highlightthickness=0, borderwidth=0)
        canvas.pack(fill="both", expand=True)
        top.update_idletasks()
        self._apply_click_through(top)
        top.withdraw()
        self._top = top
        self._canvas = canvas

    def _apply_click_through(self, top: tk.Toplevel) -> None:
        try:
            if sys.platform == "win32":
                import win32con
                import win32gui

                hwnd = top.winfo_id()
                ex = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
                win32gui.SetWindowLong(
                    hwnd, win32con.GWL_EXSTYLE,
                    ex | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT,
                )
            else:
                from Xlib import display as xd
                from Xlib.ext import shape

                disp = xd.Display()
                win = disp.create_resource_object("window", top.winfo_id())
                win.shape_rectangles(shape.SO.Set, shape.SK.Input, 0, 0, 0, [])
                disp.sync()
                disp.close()
        except Exception as exc:  # pragma: no cover — platform-specific
            plog.event("overlay_clickthrough_failed", level=logging.WARNING, error=str(exc))

    def _draw(self, bbox: Bbox) -> None:
        self._ensure_window()
        top, canvas = self._top, self._canvas
        if top is None or canvas is None:
            return
        top.geometry(f"{bbox.w}x{bbox.h}+{bbox.x}+{bbox.y}")
        canvas.delete("all")
        border = self._settings.overlay_border_px
        canvas.create_rectangle(
            border // 2, border // 2,
            bbox.w - border // 2, bbox.h - border // 2,
            outline=self._settings.overlay_color, width=border,
        )
        top.deiconify()
        top.lift()

    def _blink(self) -> None:
        if self._visible and self._top is not None:
            alpha = (
                self._settings.overlay_alpha_high
                if self._bright
                else self._settings.overlay_alpha_low
            )
            with contextlib.suppress(tk.TclError):  # pragma: no cover
                self._top.attributes("-alpha", alpha)
            self._bright = not self._bright
        self._root.after(self._settings.overlay_blink_interval_ms, self._blink)

    def _pump(self) -> None:
        try:
            while True:
                cmd = self._cmd_q.get_nowait()
                if cmd.kind == "show" and cmd.bbox is not None:
                    self._visible = True
                    self._draw(cmd.bbox)
                else:  # hide / shutdown
                    self._visible = False
                    if self._top is not None:
                        self._top.withdraw()
        except queue.Empty:
            pass
        self._root.after(30, self._pump)
