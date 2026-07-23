#!/usr/bin/env bash
# Build a single-file Linux binary with PyInstaller.
# Run from project root: `bash scripts/build_linux.sh`

set -euo pipefail

if [[ -z "${CONDA_PREFIX:-}" ]] || [[ "$(basename "$CONDA_PREFIX")" != "puzzle" ]]; then
  echo "Activate the 'puzzle' conda env first: conda activate puzzle"
  exit 1
fi

pip install --quiet pyinstaller

rm -rf build/ dist/

pyinstaller --onefile --name "JigsawSolver" \
  --add-data "puzzle_assistant/config/defaults.py:puzzle_assistant/config" \
  --hidden-import "Xlib.ext.shape" \
  --hidden-import "pynput.mouse._xorg" \
  --noconfirm \
  -p . \
  puzzle_assistant/main.py

echo
echo "Built: dist/JigsawSolver"
