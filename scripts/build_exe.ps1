# Build a single-file Windows binary with PyInstaller.
# Run from project root in a PowerShell window with the 'puzzle' conda env active:
#   conda activate puzzle
#   .\scripts\build_exe.ps1

$ErrorActionPreference = "Stop"

if ($env:CONDA_DEFAULT_ENV -ne "puzzle") {
    Write-Host "Activate the 'puzzle' conda env first: conda activate puzzle"
    exit 1
}

pip install --quiet pyinstaller customtkinter pywin32 win10toast

if (Test-Path build) { Remove-Item -Recurse -Force build }
if (Test-Path dist)  { Remove-Item -Recurse -Force dist }

# Entry is the GUI panel (Baslat/Durdur + status + instructions). --windowed
# keeps it a GUI app with no console window. --collect-all customtkinter
# bundles its theme JSON data files.
pyinstaller --onefile --windowed --name "YapbozAsistani" `
    --paths . `
    --collect-submodules pynput `
    --collect-all customtkinter `
    --collect-all win10toast `
    --hidden-import "win32gui" `
    --hidden-import "win32con" `
    --noconfirm `
    puzzle_assistant\gui.py

Write-Host "`nBuilt: dist\YapbozAsistani.exe"
