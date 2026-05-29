# Build a single-file Windows binary with PyInstaller.
# Run from project root in a PowerShell window with the 'puzzle' conda env active:
#   conda activate puzzle
#   .\scripts\build_exe.ps1

$ErrorActionPreference = "Stop"

if ($env:CONDA_DEFAULT_ENV -ne "puzzle") {
    Write-Host "Activate the 'puzzle' conda env first: conda activate puzzle"
    exit 1
}

pip install --quiet pyinstaller pywin32 win10toast

if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path dist)  { Remove-Item -Recurse -Force dist }

pyinstaller --onefile --noconsole --name "YapbozAsistani" `
    --add-data "puzzle_assistant\config\defaults.py;puzzle_assistant\config" `
    --hidden-import "pynput.mouse._win32" `
    --hidden-import "win32gui" `
    --hidden-import "win32con" `
    --noconfirm `
    -p . `
    puzzle_assistant\main.py

Write-Host "`nBuilt: dist\YapbozAsistani.exe"
