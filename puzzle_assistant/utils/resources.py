"""Locate bundled asset files in both a dev checkout and a PyInstaller build.

PyInstaller unpacks data files into a temporary directory it points to with
``sys._MEIPASS``; in a normal checkout the same files sit under the project
root. ``asset_path`` hides that difference so the GUI can just ask for
``images/logo_icon.png`` and get a working path either way.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _base_dir() -> Path:
    bundled = getattr(sys, "_MEIPASS", None)
    if bundled:
        return Path(bundled)
    # utils/ -> puzzle_assistant/ -> project root
    return Path(__file__).resolve().parent.parent.parent


def asset_path(relative: str) -> Path:
    """Absolute path to a bundled asset, e.g. ``asset_path("images/logo_icon.png")``."""
    return _base_dir() / relative


LOGO_PNG = "images/logo_icon.png"
LOGO_ICO = "images/logo_icon.ico"
