@echo off
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "FIRE_DATASET_ROOT=%~dp0data\fire_dataset"
if not exist ".venv\Scripts\python.exe" (
    echo Project Python environment missing. See docs\quick_start.md.
    pause
    exit /b 1
)
echo Independent annotation workspace: http://127.0.0.1:8506/Quality
echo Keep this window open while annotating. A and B must use separate browser sessions.
for %%F in ("1_*_Home.py") do (
    ".venv\Scripts\python.exe" -m streamlit run "%%~fF" --server.address 127.0.0.1 --server.port 8506 --server.headless true --browser.gatherUsageStats false
)
if errorlevel 1 pause
endlocal
