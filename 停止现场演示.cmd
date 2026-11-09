@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
echo Stopping this copy: "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\demo_runtime.ps1" -Action stop
set "demoStopResult=%ERRORLEVEL%"
if not "%demoStopResult%"=="0" echo [ERROR] Stop failed. Read the message above before closing this window.
pause
endlocal & exit /b %demoStopResult%
