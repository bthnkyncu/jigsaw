# Yapboz Asistanı

Visual matching assistant for Gamyun.com's _YapBoz Salonu_ (Java desktop puzzle
game). The assistant **never moves the mouse**; it draws a translucent
click-through overlay over the correct target cell while the user drags.

## Status

In active development — Phase 0 (skeleton) complete.

## Quick start (Ubuntu 22.04, X11)

```bash
conda activate puzzle
pip install -e ".[dev]"
python -m puzzle_assistant.main --help
pytest
```

## Phase map

| Phase | Scope |
|-------|-------|
| 0 | Repo skeleton, config, logger, platform factory |
| 1 | Window capture (Linux X11 + mock) |
| 2 | Init-view watcher + board detector |
| 3 | Grid detector (cut lines) + reference panel |
| 4 | Target map (primary + fallback) |
| 5 | Mouse hook + pickup + segmentation + group detection |
| 6 | Matching engine (template + ORB + Lab ensemble) |
| 7 | Overlay (click-through) + state machine |
| 8 | Notifier + fault tolerance + watchdog |
| 9 | Windows port + packaging + tuning |
