# Changelog

## [0.1.0] — Phase 9 complete

### Highlights
- End-to-end Linux X11 pipeline working: window capture → calibration →
  target map → mouse hook → matching → click-through overlay.
- 52 tests passing; `mypy --strict` clean; `ruff` clean.
- Matching benchmark: **96.8 %** self-match accuracy across 63 cells, P95
  latency **48 ms** (brief DoD: ≥ 95 % / ≤ 200 ms).
- Cross-puzzle pieces correctly rejected (≥ 90 %).

### Phase log
- **Phase 0** — repo skeleton, pyproject, logger, settings, platform factory.
- **Phase 1** — Linux X11 window capture (python-xlib + mss) + mock backend.
- **Phase 2** — board detector (HSV desk mask + saturation band) and init
  view watcher (variance + panel correlation + timeout).
- **Phase 3** — grid detector via Sobel-projection FFT (period-finding,
  more robust than peak-counting); reference panel detector via per-row
  variance band scanning.
- **Phase 4** — target map with per-cell BGR slice + Lab mean + cached ORB
  descriptors; primary (init view) + fallback (panel upscale) paths.
- **Phase 5** — pynput mouse hook (X11 XRecord), pickup + HSV segmentation +
  erosion core + group-vs-single classification.
- **Phase 6** — ensemble matching engine: TM_CCOEFF_NORMED + ORB Hamming +
  Lab mean L2; weighted sum with margin rule and quality-aware thresholds.
- **Phase 7** — Tkinter overlay with X11 XShape input region for
  click-through; state machine with WAIT_FOR_NEW_PUZZLE / CALIBRATING_*
  / READY / TRACKING transitions.
- **Phase 8** — notifier (libnotify), watchdog context manager, main loop
  integrating every subsystem and reacting to mouse + panel + bbox events.
- **Phase 9** — Windows-side stubs (pywin32 + WS_EX_LAYERED|TRANSPARENT
  overlay), PyInstaller build scripts for both platforms, end-user
  `KULLANIM.txt`.
