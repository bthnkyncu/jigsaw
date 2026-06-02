"""Simple control-panel GUI for the puzzle assistant.

A minimal Tkinter window the end user (a non-developer customer) can run:
a Start/Stop button, a live status line, and built-in usage instructions.
The heavy work (``MainLoop``) runs on a daemon thread; the GUI only polls a
read-only status snapshot, so it never blocks. The click-through overlay keeps
running on its own thread exactly as before — this panel does not touch it.

Packaged as the ``.exe`` entry point (see ``scripts/build_exe.ps1`` and the
GitHub Actions workflow). Runs the same on Linux for local testing.
"""

from __future__ import annotations

import argparse
import threading
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext, ttk

from puzzle_assistant.config import Settings, load_settings
from puzzle_assistant.state.main_loop import MainLoop
from puzzle_assistant.utils import dpi
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.platform import (
    Platform,
    make_mouse_hook,
    make_notifier,
    make_overlay,
    make_window_capture,
)

_STATE_TR: dict[str, str] = {
    "IDLE": "Pencere aranıyor…",
    "WAIT_FOR_NEW_PUZZLE": "Yeni oyun bekleniyor — oyunda YENİ OYUN'a basın",
    "CALIBRATING_PRIMARY": "Açılış görüntüsü yakalanıyor…",
    "CALIBRATING_FALLBACK": "Yedek kaynaktan kalibre ediliyor…",
    "READY": "Hazır — parça sürükleyin, doğru hücre yeşil yanar",
    "TRACKING": "Parça takip ediliyor…",
}

INSTRUCTIONS = """\
NASIL KULLANILIR

1) Oyunu açın ve yapboz masasına girin ("Yapboz oyun salonu").
2) Bu pencerede BAŞLAT'a basın. Asistan oyun penceresini bulur.
   Durum satırı "Pencere aranıyor…"dan "Yeni oyun bekleniyor"a döner.
3) Oyunda YENİ OYUN başlatın. Oyun ~2 saniye tamamlanmış resmi gösterir;
   asistan bu görüntüyü referans olarak yakalar. Durum "Hazır" olur.
4) Parçaları sürükleyin. Doğru hücre, ekranda YEŞİL bir çerçeveyle yanıp söner.
   Tıklamalar bu yeşil katmanın içinden oyuna geçer (oynamayı engellemez).
5) DURDUR ile asistanı durdurabilir, BAŞLAT ile tekrar başlatabilirsiniz.

ÖNEMLİ NOTLAR
• Ekran ölçeklendirmesi %100 olmalı (Ayarlar > Ekran > Ölçek). Aksi halde
  yeşil çerçeve yanlış yere düşer.
• Oyun penceresini taşır veya zoom yaparsanız asistan kendini yeniden
  kalibre eder; birkaç saniye "Hazır"ı bekleyin.
• Tahmin çıkmıyorsa: YENİ OYUN'u tekrar başlatın; açılıştaki tam resmin
  net görünmesi gerekir (asistan referansı oradan alır).
• Bu bir geliştirme sürümüdür: bazı parçalar (özellikle tek renkli/benzer
  bölgeler) tahmin edilmeyebilir. Yanlış göstermektense hiç göstermemeyi
  tercih eder. Geliştirme sürüyor.
"""


class AssistantGUI:
    """Tkinter control panel that drives a background ``MainLoop``."""

    def __init__(self, settings: Settings, platform: Platform | None) -> None:
        self._settings = settings
        self._platform = platform
        self._loop: MainLoop | None = None
        self._thread: threading.Thread | None = None

        self.root = tk.Tk()
        self.root.title("Yapboz Asistanı")
        self.root.geometry("560x520")
        self.root.minsize(480, 440)

        self._status_var = tk.StringVar(value="Durduruldu")
        self._build_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(500, self._poll_status)

    def _build_widgets(self) -> None:
        header = ttk.Label(
            self.root, text="Yapboz Asistanı", font=("Segoe UI", 16, "bold")
        )
        header.pack(pady=(12, 4))

        status_frame = ttk.Frame(self.root)
        status_frame.pack(fill="x", padx=16, pady=4)
        ttk.Label(status_frame, text="Durum:", font=("Segoe UI", 10, "bold")).pack(
            side="left"
        )
        ttk.Label(status_frame, textvariable=self._status_var).pack(
            side="left", padx=(6, 0)
        )

        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=8)
        self._start_btn = ttk.Button(button_frame, text="BAŞLAT", command=self._start)
        self._start_btn.pack(side="left", padx=6)
        self._stop_btn = ttk.Button(
            button_frame, text="DURDUR", command=self._stop, state="disabled"
        )
        self._stop_btn.pack(side="left", padx=6)

        box = scrolledtext.ScrolledText(
            self.root, wrap="word", font=("Segoe UI", 10), height=18
        )
        box.insert("1.0", INSTRUCTIONS)
        box.configure(state="disabled")
        box.pack(fill="both", expand=True, padx=16, pady=(8, 16))

    # ------------------------------ actions -------------------------------

    def _start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        capture = make_window_capture(self._platform)
        hook = make_mouse_hook(self._platform)
        overlay = make_overlay(self._platform)
        notifier = make_notifier(self._platform)
        self._loop = MainLoop(self._settings, capture, hook, overlay, notifier)
        self._thread = threading.Thread(
            target=self._loop.run, name="main-loop", daemon=True
        )
        self._thread.start()
        plog.event("gui_start")
        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")

    def _stop(self) -> None:
        if self._loop is not None:
            self._loop.stop()
            plog.event("gui_stop")
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")

    def _poll_status(self) -> None:
        running = self._thread is not None and self._thread.is_alive()
        if running and self._loop is not None:
            snap = self._loop.status_snapshot()
            if not snap["window_found"]:
                text = "Oyun penceresi aranıyor… (oyun açık mı?)"
            else:
                state = str(snap["state"])
                text = _STATE_TR.get(state, state)
                if state == "READY" and snap.get("quality") == "fallback":
                    text += "  (düşük kaliteli kaynak)"
            self._status_var.set(text)
        else:
            self._status_var.set("Durduruldu")
        self.root.after(500, self._poll_status)

    def _on_close(self) -> None:
        self._stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def run_gui(settings: Settings, platform: Platform | None) -> int:
    dpi.ensure_compatible_scale(settings.expected_dpi_scale)
    AssistantGUI(settings, platform).run()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="yapboz-asistani-gui")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--platform-override", choices=["linux", "windows", "mock"], default=None
    )
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    plog.install(
        log_dir=Path(settings.log_dir),
        log_filename=settings.log_filename,
        rotate_bytes=settings.log_rotate_bytes,
        rotate_backups=settings.log_rotate_backups,
    )
    plog.event("gui_boot", platform_override=args.platform_override)
    return run_gui(settings, args.platform_override)


if __name__ == "__main__":
    raise SystemExit(main())
