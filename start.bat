@echo off
title SatyaDrishti AI Full-Stack Launcher
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" run_app.py %*
) else (
    python run_app.py %*
)
pause
