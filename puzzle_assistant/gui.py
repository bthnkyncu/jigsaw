"""Modern control-panel GUI for the puzzle assistant.

A polished CustomTkinter window the end user (a non-developer customer) runs:
a header, a live status card with a coloured indicator, Start/Stop buttons,
and built-in usage instructions. The heavy work (``MainLoop``) runs on a daemon
thread; the GUI only polls a read-only status snapshot, so it never blocks. The
click-through overlay keeps running on its own thread exactly as before — this
panel does not touch it.

Packaged as the ``.exe`` entry point (see ``scripts/build_exe.ps1`` and the
GitHub Actions workflow). Runs the same on Linux for local testing.

CustomTkinter is pure-Python on top of stdlib tkinter (no native deps), so it
bundles cleanly with PyInstaller via ``--collect-all customtkinter``.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import customtkinter as ctk

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

# state -> (Turkish status text, indicator colour)
_GREY = "#6b7280"
_AMBER = "#f59e0b"
_GREEN = "#22c55e"
_STATE_TR: dict[str, tuple[str, str]] = {
    "IDLE": ("Oyun penceresi aranıyor…", _AMBER),
    "WAIT_FOR_NEW_PUZZLE": ("Yeni oyun bekleniyor — oyunda YENİ OYUN'a basın", _AMBER),
    "CALIBRATING_PRIMARY": ("Açılış görüntüsü yakalanıyor…", _AMBER),
    "CALIBRATING_FALLBACK": ("Yedek kaynaktan kalibre ediliyor…", _AMBER),
    "READY": ("Hazır — parça sürükleyin, doğru hücre yeşil yanar", _GREEN),
    "TRACKING": ("Parça takip ediliyor…", _GREEN),
}

INSTRUCTIONS = """\
1)  Oyunu açın ve yapboz masasına girin ("Yapboz oyun salonu").

2)  BAŞLAT'a basın. Asistan oyun penceresini bulur; durum
    "Yeni oyun bekleniyor"a döner.

3)  Oyunda YENİ OYUN başlatın. Oyun ~2 saniye tamamlanmış resmi
    gösterir; asistan bunu referans alır. Durum "Hazır" olur.

4)  Parçaları sürükleyin. Doğru hücre ekranda YEŞİL bir çerçeveyle
    yanıp söner. Tıklamalar bu katmanın içinden oyuna geçer.

5)  DURDUR ile durdurabilir, BAŞLAT ile yeniden başlatabilirsiniz.

ÖNEMLİ
•  Ekran ölçeklendirmesi %100 olmalı; aksi halde yeşil çerçeve
   yanlış yere düşer.
•  Pencereyi taşır/zoom yaparsanız asistan yeniden kalibre eder;
   birkaç saniye "Hazır"ı bekleyin.
•  Tahmin çıkmıyorsa YENİ OYUN'u tekrar başlatın (açılış resmi net
   görünmeli).
•  Geliştirme sürümü: tek renkli/benzer bölgelerdeki bazı parçalar
   tahmin edilmeyebilir — yanlış göstermektense hiç göstermez.
"""


class AssistantGUI:
    """Modern CustomTkinter control panel driving a background ``MainLoop``."""

    def __init__(self, settings: Settings, platform: Platform | None) -> None:
        self._settings = settings
        self._platform = platform
        self._loop: MainLoop | None = None
        self._thread: threading.Thread | None = None

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        self.root = ctk.CTk()
        self.root.title("Yapboz Asistanı")
        self.root.geometry("640x720")
        self.root.minsize(560, 620)

        self._build_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(500, self._poll_status)

    def _build_widgets(self) -> None:
        root = self.root
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(3, weight=1)  # instructions row stretches

        # --- Header ---
        header = ctk.CTkFrame(root, corner_radius=0, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=24, pady=(22, 8))
        ctk.CTkLabel(
            header, text="Yapboz Asistanı",
            font=ctk.CTkFont(size=26, weight="bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            header, text="Gamyun Yapboz — görsel yerleştirme asistanı",
            font=ctk.CTkFont(size=13), text_color="#9ca3af",
        ).pack(anchor="w", pady=(2, 0))

        # --- Status card ---
        card = ctk.CTkFrame(root, corner_radius=14)
        card.grid(row=1, column=0, sticky="ew", padx=24, pady=8)
        card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            card, text="DURUM", font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#9ca3af",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(12, 0))
        self._dot = ctk.CTkLabel(
            card, text="●", font=ctk.CTkFont(size=20), text_color=_GREY,
        )
        self._dot.grid(row=1, column=0, sticky="w", padx=(16, 6), pady=(0, 14))
        self._status_label = ctk.CTkLabel(
            card, text="Durduruldu", font=ctk.CTkFont(size=14),
            anchor="w", justify="left", wraplength=480,
        )
        self._status_label.grid(row=1, column=1, sticky="w", pady=(0, 14))

        # --- Buttons ---
        buttons = ctk.CTkFrame(root, fg_color="transparent")
        buttons.grid(row=2, column=0, sticky="ew", padx=24, pady=8)
        buttons.grid_columnconfigure((0, 1), weight=1)
        self._start_btn = ctk.CTkButton(
            buttons, text="BAŞLAT", height=44, command=self._start,
            font=ctk.CTkFont(size=15, weight="bold"),
        )
        self._start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self._stop_btn = ctk.CTkButton(
            buttons, text="DURDUR", height=44, command=self._stop, state="disabled",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#374151", hover_color="#b91c1c",
        )
        self._stop_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # --- Instructions ---
        instr = ctk.CTkFrame(root, corner_radius=14)
        instr.grid(row=3, column=0, sticky="nsew", padx=24, pady=8)
        instr.grid_columnconfigure(0, weight=1)
        instr.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            instr, text="NASIL KULLANILIR",
            font=ctk.CTkFont(size=11, weight="bold"), text_color="#9ca3af",
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(12, 4))
        box = ctk.CTkTextbox(
            instr, font=ctk.CTkFont(size=13), wrap="word", fg_color="transparent",
        )
        box.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 12))
        box.insert("1.0", INSTRUCTIONS)
        box.configure(state="disabled")

        # --- Footer ---
        ctk.CTkLabel(
            root, text="Fareyi kullanmaz · ekranı okur ve yeşil çerçeve çizer · %100 ölçek gerekir",
            font=ctk.CTkFont(size=11), text_color="#6b7280",
        ).grid(row=4, column=0, pady=(0, 14))

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

    def _set_status(self, text: str, colour: str) -> None:
        self._status_label.configure(text=text)
        self._dot.configure(text_color=colour)

    def _poll_status(self) -> None:
        running = self._thread is not None and self._thread.is_alive()
        if running and self._loop is not None:
            snap = self._loop.status_snapshot()
            if not snap["window_found"]:
                self._set_status("Oyun penceresi aranıyor… (oyun açık mı?)", _AMBER)
            else:
                state = str(snap["state"])
                text, colour = _STATE_TR.get(state, (state, _AMBER))
                if state == "READY" and snap.get("quality") == "fallback":
                    text += "  (düşük kaliteli kaynak)"
                self._set_status(text, colour)
        else:
            self._set_status("Durduruldu", _GREY)
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
