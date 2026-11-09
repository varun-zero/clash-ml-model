@echo off
title Clash Royale Simulator & AlphaStar Agent UI
cd /d "%~dp0"
echo ===========================================================================
echo       Clash Royale Headless Simulator & AlphaStar Agent Web UI
echo ===========================================================================
echo Starting web server on http://127.0.0.1:8080 ...
start "ClashBot UI server" cmd /k python tools\ui.py --port 8080
timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:8080
echo Close the ClashBot UI server window to stop the simulator.
pause
