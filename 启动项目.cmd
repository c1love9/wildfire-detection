@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project environment missing. See docs\quick_start.md.
    pause
    exit /b 1
)
for %%F in ("1_*_Home.py") do (
    ".venv\Scripts\python.exe" -m streamlit run "%%~fF" --server.address 127.0.0.1 --server.port 8501 --server.headless true
)
if errorlevel 1 pause
endlocal
