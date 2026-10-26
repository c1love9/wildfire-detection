@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
chcp 65001 >nul
if not exist ".venv\Scripts\python.exe" (
    echo Project Python environment missing. See docs\quick_start.md.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" scripts\export_dual_annotation.py
set "task_export_exit=%errorlevel%"
if errorlevel 1 (
    echo Export failed: check A/B completion, image integrity and the error above.
) else (
    echo Submission package is saved under data\fire_dataset\quality_deliveries.
)
pause
exit /b %task_export_exit%
