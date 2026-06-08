"""Modern control-panel GUI for the puzzle assistant.

A polished CustomTkinter window the end user (a non-developer customer) runs:
a coloured header, a live status card with a colour-coded indicator, Start/Stop
buttons, and a dedicated "How to play" window opened from a button. The heavy
work (``MainLoop``) runs on a daemon thread; the GUI only polls a read-only
status snapshot, so it never blocks. The click-through overlay keeps running on
its own thread exactly as before — this panel does not touch it.

Packaged as the ``.exe`` entry point (see ``scripts/build_exe.ps1`` and the
GitHub Actions workflow). Runs the same on Linux for local testing.

CustomTkinter is pure-Python on top of stdlib tkinter (no native deps), so it
bundles cleanly with PyInstaller via ``--collect-all customtkinter``.
"""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

import customtkinter as ctk

from puzzle_assistant.config import Settings, load_settings
from puzzle_assistant.overlay.gui_overlay import GuiOverlay
from puzzle_assistant.state.main_loop import MainLoop
from puzzle_assistant.utils import dpi
from puzzle_assistant.utils import logger as plog
from puzzle_assistant.utils.platform import (
    Platform,
    make_mouse_hook,
    make_notifier,
    make_window_capture,
)

# Palette
_GREY = "#6b7280"
_AMBER = "#f59e0b"
_GREEN = "#22c55e"
_ACCENT = "#16a34a"
_ACCENT_HOVER = "#15803d"
_INFO = "#0ea5e9"
_INFO_HOVER = "#0284c7"
_BADGE_COLORS = ["#16a34a", "#0ea5e9", "#f59e0b", "#8b5cf6", "#ef4444"]

# state -> (Turkish status text, indicator colour)
_STATE_TR: dict[str, tuple[str, str]] = {
    "IDLE": ("Oyun penceresi aranıyor…", _AMBER),
    "WAIT_FOR_NEW_PUZZLE": ("Yeni oyun bekleniyor — oyunda YENİ OYUN'a basın", _AMBER),
    "CALIBRATING_PRIMARY": ("Açılış görüntüsü yakalanıyor…", _AMBER),
    "CALIBRATING_FALLBACK": ("Yedek kaynaktan kalibre ediliyor…", _AMBER),
    "READY": ("Hazır — parça sürükleyin, doğru hücre yeşil yanar", _GREEN),
    "TRACKING": ("Parça takip ediliyor…", _GREEN),
}

# How-to-play steps: (number, title, body)
_STEPS: list[tuple[str, str, str]] = [
    ("1", "Oyunu açın",
     "Gamyun'da yapboz masasına girin (\"Yapboz oyun salonu\")."),
    ("2", "BAŞLAT'a basın",
     "Asistan oyun penceresini bulur; durum \"Yeni oyun bekleniyor\"a döner."),
    ("3", "YENİ OYUN başlatın",
     "Oyun ~2 saniye tamamlanmış resmi gösterir; asistan bunu referans alır. "
     "Durum \"Hazır\" olur."),
    ("4", "Parçaları sürükleyin",
     "Doğru hücre ekranda YEŞİL bir çerçeveyle yanıp söner. Tıklamalar bu "
     "katmanın içinden oyuna geçer, oynamanızı engellemez."),
    ("5", "Durdurun / sürdürün",
     "İstediğiniz an DURDUR ile durdurup BAŞLAT ile tekrar başlatabilirsiniz."),
]

_NOTES: list[str] = [
    "Ekran ölçeklendirmesi %100 olmalı; aksi halde yeşil çerçeve yanlış yere düşer.",
    "Pencereyi taşır veya zoom yaparsanız asistan yeniden kalibre eder; birkaç "
    "saniye \"Hazır\"ı bekleyin.",
    "Tahmin çıkmıyorsa YENİ OYUN'u tekrar başlatın — açılıştaki tam resmin net "
    "görünmesi gerekir.",
    "Bu bir geliştirme sürümüdür: tek renkli / birbirine çok benzeyen bölgelerdeki "
    "bazı parçalar tahmin edilmeyebilir. Yanlış göstermektense hiç göstermez.",
]

# High-frequency events hidden from the in-app log feed so the meaningful ones
# (state changes, matches, errors) stay readable.
_LOG_HIDE = {"board_detect_ok", "init_view_assess", "ref_panel_ok"}


def _read_log_feed(max_events: int = 250) -> str:
    """Tail the active log file into a compact, readable event feed.

    Drops the high-frequency calibration spam (``_LOG_HIDE``) and renders each
    remaining JSON event as ``HH:MM:SS  evt  k=v …`` so a non-developer can see
    whether dragging a piece produces TRACKING / match events.
    """
    path = plog.current_log_path()
    if path is None or not path.exists():
        return "Log dosyası henüz yok. BAŞLAT'a basıp birkaç saniye bekleyin."
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"Log okunamadı: {exc}"

    out: list[str] = []
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except ValueError:
            out.append(raw)
            continue
        evt = rec.get("evt", "?")
        if evt in _LOG_HIDE:
            continue
        ts = str(rec.get("ts", ""))[11:19]
        extras = "  ".join(
            f"{k}={v}"
            for k, v in rec.items()
            if k not in ("ts", "lvl", "evt") and v is not None
        )
        mark = " !" if rec.get("lvl") in ("WARNING", "ERROR") else "  "
        out.append(f"{ts}{mark} {evt}  {extras}".rstrip())

    if not out:
        return "Henüz önemli olay yok. BAŞLAT → yeni oyun → parça sürükleyin."
    return "\n".join(out[-max_events:])


class AssistantGUI:
    """Modern CustomTkinter control panel driving a background ``MainLoop``."""

    def __init__(self, settings: Settings, platform: Platform | None) -> None:
        self._settings = settings
        self._platform = platform
        self._loop: MainLoop | None = None
        self._thread: threading.Thread | None = None
        self._help_win: ctk.CTkToplevel | None = None
        self._logs_win: ctk.CTkToplevel | None = None
        self._logs_box: ctk.CTkTextbox | None = None

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        self.root = ctk.CTk()
        self.root.title("Yapboz Asistanı")
        self.root.geometry("500x600")
        self.root.minsize(460, 560)

        # One Tk root for the whole process: the overlay is a Toplevel of this
        # root driven on the main thread (see GuiOverlay) — never a second
        # tk.Tk() on a worker thread.
        self._overlay = GuiOverlay(self.root, settings)

        self._build_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(500, self._poll_status)

    def _build_widgets(self) -> None:
        root = self.root
        root.grid_columnconfigure(0, weight=1)

        # --- Coloured header banner ---
        banner = ctk.CTkFrame(root, corner_radius=0, fg_color=_ACCENT, height=92)
        banner.grid(row=0, column=0, sticky="ew")
        banner.grid_propagate(False)
        ctk.CTkLabel(
            banner, text="🧩  Yapboz Asistanı",
            font=ctk.CTkFont(size=24, weight="bold"), text_color="white",
        ).pack(anchor="w", padx=24, pady=(20, 0))
        ctk.CTkLabel(
            banner, text="Gamyun Yapboz — görsel yerleştirme asistanı",
            font=ctk.CTkFont(size=13), text_color="#dcfce7",
        ).pack(anchor="w", padx=24)

        # --- Status card ---
        card = ctk.CTkFrame(root, corner_radius=14)
        card.grid(row=1, column=0, sticky="ew", padx=20, pady=(18, 8))
        card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            card, text="DURUM", font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#9ca3af",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(12, 0))
        self._dot = ctk.CTkLabel(
            card, text="●", font=ctk.CTkFont(size=22), text_color=_GREY,
        )
        self._dot.grid(row=1, column=0, sticky="w", padx=(16, 6), pady=(0, 14))
        self._status_label = ctk.CTkLabel(
            card, text="Durduruldu", font=ctk.CTkFont(size=14),
            anchor="w", justify="left", wraplength=360,
        )
        self._status_label.grid(row=1, column=1, sticky="w", pady=(0, 14))

        # --- Piece-count input (enter BEFORE Başlat) ---
        count_card = ctk.CTkFrame(root, corner_radius=14)
        count_card.grid(row=2, column=0, sticky="ew", padx=20, pady=8)
        count_card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            count_card, text="Parça sayısı",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=0, column=0, sticky="w", padx=(16, 8), pady=14)
        self._count_entry = ctk.CTkEntry(
            count_card, placeholder_text="örn. 150  (oyunda seçtiğiniz sayı)",
            font=ctk.CTkFont(size=14), height=36,
        )
        self._count_entry.grid(row=0, column=1, sticky="ew", padx=(0, 16), pady=14)

        # --- Start / Stop ---
        buttons = ctk.CTkFrame(root, fg_color="transparent")
        buttons.grid(row=3, column=0, sticky="ew", padx=20, pady=6)
        buttons.grid_columnconfigure((0, 1), weight=1)
        self._start_btn = ctk.CTkButton(
            buttons, text="BAŞLAT", height=46, command=self._start,
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color=_ACCENT, hover_color=_ACCENT_HOVER,
        )
        self._start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self._stop_btn = ctk.CTkButton(
            buttons, text="DURDUR", height=46, command=self._stop, state="disabled",
            font=ctk.CTkFont(size=15, weight="bold"),
            fg_color="#374151", hover_color="#b91c1c",
        )
        self._stop_btn.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # --- Secondary buttons: help + logs ---
        secondary = ctk.CTkFrame(root, fg_color="transparent")
        secondary.grid(row=4, column=0, sticky="ew", padx=20, pady=(6, 8))
        secondary.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(
            secondary, text="📖  Nasıl Oynanır?", height=44, command=self._open_help,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=_INFO, hover_color=_INFO_HOVER,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkButton(
            secondary, text="📋  Loglar", height=44, command=self._open_logs,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#475569", hover_color="#334155",
        ).grid(row=0, column=1, sticky="ew", padx=(6, 0))

        # --- Footer ---
        ctk.CTkLabel(
            root,
            text="Fareyi kullanmaz · ekranı okur ve yeşil çerçeve çizer · %100 ölçek gerekir",
            font=ctk.CTkFont(size=11), text_color="#6b7280", wraplength=440,
        ).grid(row=5, column=0, pady=(4, 14))

    # ------------------------------ help window ---------------------------

    def _open_help(self) -> None:
        if self._help_win is not None and self._help_win.winfo_exists():
            self._help_win.focus()
            self._help_win.lift()
            return
        win = ctk.CTkToplevel(self.root)
        self._help_win = win
        win.title("Nasıl Oynanır?")
        win.geometry("560x640")
        win.minsize(480, 520)
        win.transient(self.root)
        win.grid_columnconfigure(0, weight=1)
        win.grid_rowconfigure(1, weight=1)

        banner = ctk.CTkFrame(win, corner_radius=0, fg_color=_INFO, height=70)
        banner.grid(row=0, column=0, sticky="ew")
        banner.grid_propagate(False)
        ctk.CTkLabel(
            banner, text="📖  Nasıl Oynanır?",
            font=ctk.CTkFont(size=20, weight="bold"), text_color="white",
        ).pack(anchor="w", padx=24, pady=20)

        body = ctk.CTkScrollableFrame(win, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        body.grid_columnconfigure(0, weight=1)

        for i, (num, title, desc) in enumerate(_STEPS):
            self._step_card(body, i, num, title, desc, _BADGE_COLORS[i % len(_BADGE_COLORS)])

        notes = ctk.CTkFrame(body, corner_radius=12, fg_color="#1f2937")
        notes.grid(row=len(_STEPS), column=0, sticky="ew", padx=8, pady=(12, 6))
        notes.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            notes, text="⚠  ÖNEMLİ NOTLAR",
            font=ctk.CTkFont(size=13, weight="bold"), text_color=_AMBER,
        ).grid(row=0, column=0, sticky="w", padx=16, pady=(12, 4))
        for j, note in enumerate(_NOTES):
            ctk.CTkLabel(
                notes, text=f"•  {note}", font=ctk.CTkFont(size=12),
                anchor="w", justify="left", wraplength=470, text_color="#d1d5db",
            ).grid(row=j + 1, column=0, sticky="w", padx=16, pady=2)
        ctk.CTkLabel(notes, text="").grid(row=len(_NOTES) + 1, column=0, pady=4)

        ctk.CTkButton(
            win, text="Kapat", height=40, command=win.destroy,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=_ACCENT, hover_color=_ACCENT_HOVER,
        ).grid(row=2, column=0, sticky="ew", padx=20, pady=(4, 14))

        win.after(120, win.lift)  # ensure it surfaces above the main window

    def _step_card(
        self, parent: ctk.CTkScrollableFrame, row: int,
        num: str, title: str, desc: str, colour: str,
    ) -> None:
        card = ctk.CTkFrame(parent, corner_radius=12)
        card.grid(row=row, column=0, sticky="ew", padx=8, pady=6)
        card.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(
            card, text=num, width=36, height=36, corner_radius=18,
            fg_color=colour, text_color="white",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).grid(row=0, column=0, rowspan=2, padx=(14, 12), pady=14)
        ctk.CTkLabel(
            card, text=title, font=ctk.CTkFont(size=15, weight="bold"), anchor="w",
        ).grid(row=0, column=1, sticky="w", pady=(12, 0), padx=(0, 14))
        ctk.CTkLabel(
            card, text=desc, font=ctk.CTkFont(size=12), text_color="#9ca3af",
            anchor="w", justify="left", wraplength=420,
        ).grid(row=1, column=1, sticky="w", pady=(0, 12), padx=(0, 14))

    # ------------------------------ log viewer ----------------------------

    def _open_logs(self) -> None:
        if self._logs_win is not None and self._logs_win.winfo_exists():
            self._logs_win.focus()
            self._logs_win.lift()
            return
        win = ctk.CTkToplevel(self.root)
        self._logs_win = win
        win.title("Loglar / Olaylar")
        win.geometry("700x520")
        win.minsize(540, 360)
        win.transient(self.root)
        win.grid_columnconfigure(0, weight=1)
        win.grid_rowconfigure(1, weight=1)

        banner = ctk.CTkFrame(win, corner_radius=0, fg_color="#334155", height=64)
        banner.grid(row=0, column=0, sticky="ew")
        banner.grid_propagate(False)
        ctk.CTkLabel(
            banner, text="📋  Loglar / Olaylar",
            font=ctk.CTkFont(size=18, weight="bold"), text_color="white",
        ).pack(side="left", padx=24, pady=16)
        ctk.CTkLabel(
            banner, text="canlı — sürüklerken TRACKING / match satırlarına bakın",
            font=ctk.CTkFont(size=12), text_color="#cbd5e1",
        ).pack(side="left", pady=16)

        box = ctk.CTkTextbox(
            win, font=ctk.CTkFont(family="Consolas", size=12), wrap="none",
        )
        box.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)
        box.configure(state="disabled")
        self._logs_box = box

        ctk.CTkButton(
            win, text="Kapat", height=40, command=win.destroy,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=_ACCENT, hover_color=_ACCENT_HOVER,
        ).grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 14))

        self._refresh_logs()
        win.after(150, win.lift)

    def _refresh_logs(self) -> None:
        win = self._logs_win
        box = self._logs_box
        if win is None or not win.winfo_exists() or box is None:
            return
        feed = _read_log_feed()
        at_bottom = box.yview()[1] > 0.999
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.insert("1.0", feed)
        box.configure(state="disabled")
        if at_bottom:
            box.see("end")
        win.after(800, self._refresh_logs)

    # ------------------------------ actions -------------------------------

    def _start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        # Read the piece count entered before Başlat; anchors grid detection.
        raw = self._count_entry.get().strip()
        try:
            self._settings.target_piece_count = int(raw) if raw else None
        except ValueError:
            self._settings.target_piece_count = None
        self._count_entry.configure(state="disabled")

        capture = make_window_capture(self._platform)
        hook = make_mouse_hook(self._platform)
        notifier = make_notifier(self._platform)
        self._loop = MainLoop(self._settings, capture, hook, self._overlay, notifier)
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
        self._count_entry.configure(state="normal")

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
