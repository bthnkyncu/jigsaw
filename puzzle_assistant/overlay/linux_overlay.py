"""Linux click-through overlay (Tkinter + X11 XShape input region).

Brief §7.12. We open a borderless ``Toplevel`` window, paint a hollow green
rectangle on a black background that the window manager renders transparent
via ``-transparentcolor``, and then use ``python-xlib``'s XShape extension
to set the *input* region of the window to an empty rectangle list — every
mouse event flies straight through to the Gamyun window underneath.

The blink animation is driven by ``after`` on the Tk event loop. Tkinter's
mainloop is hostile in a multi-threaded daemon, so we run the overlay on a
dedicated thread and communicate via a ``queue.Queue`` of show/hide commands.
"""

from __future__ import annotations

import logging
import queue
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


class LinuxOverlay:
    """Thread-safe wrapper around a Tk overlay."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or load_settings(None)
        self._cmd_q: queue.Queue[_Command] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._start_thread()

    def _start_thread(self) -> None:
        self._thread = threading.Thread(target=self._run, name="overlay", daemon=True)
        self._thread.start()
        # Wait briefly for the Tk loop to come up before the first call.
        self._ready.wait(timeout=2.0)

    def show(self, target: Bbox) -> None:
        self._cmd_q.put(_Command("show", target))

    def hide(self) -> None:
        self._cmd_q.put(_Command("hide"))

    def shutdown(self) -> None:
        self._cmd_q.put(_Command("shutdown"))
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        try:
            root = tk.Tk()
            root.withdraw()
            top = tk.Toplevel(root)
            top.overrideredirect(True)
            top.attributes("-topmost", True)
            # ``-transparentcolor`` is windows-only on Tk; on X11 we fall back to
            # an alpha attribute + XShape for input transparency.
            top.attributes("-alpha", self._settings.overlay_alpha_high)
            top.config(bg="black")

            canvas = tk.Canvas(
                top, bg="black", highlightthickness=0, borderwidth=0
            )
            canvas.pack(fill="both", expand=True)

            try:
                _apply_xshape_pass_through(top)
            except Exception as exc:  # pragma: no cover — only fails on Wayland/nokey
                plog.event("overlay_xshape_failed", level=logging.WARNING, error=str(exc))

            state = {"visible": False, "bright": True, "bbox": None}

            def _draw(bbox: Bbox) -> None:
                top.geometry(f"{bbox.w}x{bbox.h}+{bbox.x}+{bbox.y}")
                canvas.delete("all")
                color = self._settings.overlay_color
                border = self._settings.overlay_border_px
                canvas.create_rectangle(
                    border // 2, border // 2,
                    bbox.w - border // 2, bbox.h - border // 2,
                    outline=color, width=border,
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
                            state["bbox"] = cmd.bbox  # type: ignore[assignment]
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
        except Exception as exc:  # pragma: no cover
            plog.event("overlay_loop_failed", level=logging.ERROR, error=str(exc))
            self._ready.set()


def _apply_xshape_pass_through(top: tk.Toplevel) -> None:
    """Make ``top`` ignore mouse events via the X11 XShape extension."""

    from Xlib import display as xd
    from Xlib.ext import shape

    wid = top.winfo_id()
    disp = xd.Display()
    win = disp.create_resource_object("window", wid)
    # ShapeInput region set to an empty rectangle list = "no input region".
    win.shape_rectangles(
        shape.SO.Set,
        shape.SK.Input,
        0,
        0,
        0,
        [],
    )
    disp.sync()
    disp.close()
