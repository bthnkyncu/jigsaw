"""Windows click-through overlay (Tkinter + WS_EX_LAYERED|WS_EX_TRANSPARENT).

Mirrors the Linux overlay (Phase 7) but uses the Windows-native extended
style bits for click-through instead of the X11 XShape extension.

Win32 imports live inside the method bodies so this module imports cleanly
on Linux too (the platform factory never instantiates the class on Linux).
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import tkinter as tk
from dataclasses import dataclass
from typing import Literal

from puzzle_assistant.config import Settings, load_settings
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.coords import Bbox


@dataclass
class _Command:
    kind: Literal["show", "hide", "shutdown"]
    bbox: Bbox | None = None


class WindowsOverlay:
    def __init__(self, settings: Settings | None = None) -> None:
        if sys.platform != "win32":
            raise RuntimeError("WindowsOverlay requires sys.platform == 'win32'.")
        self._settings: Settings = settings if settings is not None else load_settings(None)
        self._cmd_q: queue.Queue[_Command] = queue.Queue()
        self._ready: threading.Event = threading.Event()
        self._thread: threading.Thread = threading.Thread(
            target=self._run, name="overlay-win", daemon=True
        )
        self._thread.start()
        self._ready.wait(timeout=2.0)

    def show(self, target: Bbox) -> None:
        self._cmd_q.put(_Command("show", target))

    def hide(self) -> None:
        self._cmd_q.put(_Command("hide"))

    def shutdown(self) -> None:
        self._cmd_q.put(_Command("shutdown"))
        self._thread.join(timeout=2.0)

    def _run(self) -> None:  # pragma: no cover — only exercised on Windows
        import win32con
        import win32gui

        try:
            root = tk.Tk()
            root.withdraw()
            top = tk.Toplevel(root)
            top.overrideredirect(True)
            top.attributes("-topmost", True)
            top.attributes("-transparentcolor", "black")
            top.attributes("-alpha", self._settings.overlay_alpha_high)
            top.config(bg="black")

            canvas = tk.Canvas(top, bg="black", highlightthickness=0, borderwidth=0)
            canvas.pack(fill="both", expand=True)

            hwnd = top.winfo_id()
            ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            win32gui.SetWindowLong(
                hwnd, win32con.GWL_EXSTYLE,
                ex_style | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT,
            )

            state = {"visible": False, "bright": True}

            def _draw(bbox: Bbox) -> None:
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

            def _blink() -> None:
                if state["visible"]:
                    alpha = (
                        self._settings.overlay_alpha_high
                        if state["bright"]
                        else self._settings.overlay_alpha_low
                    )
                    top.attributes("-alpha", alpha)
                    state["bright"] = not state["bright"]
                root.after(self._settings.overlay_blink_interval_ms, _blink)

            def _poll() -> None:
                try:
                    while True:
                        cmd = self._cmd_q.get_nowait()
                        if cmd.kind == "show" and cmd.bbox is not None:
                            state["visible"] = True
                            _draw(cmd.bbox)
                        elif cmd.kind == "hide":
                            state["visible"] = False
                            top.withdraw()
                        elif cmd.kind == "shutdown":
                            root.quit()
                            return
                except queue.Empty:
                    pass
                root.after(30, _poll)

            top.withdraw()
            self._ready.set()
            root.after(30, _poll)
            root.after(self._settings.overlay_blink_interval_ms, _blink)
            root.mainloop()
        except (OSError, RuntimeError) as exc:
            plog.event("overlay_loop_failed", level=logging.ERROR, error=str(exc))
            self._ready.set()
