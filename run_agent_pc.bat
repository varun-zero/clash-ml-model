@echo off
title ClashBot - Google Play Games PC Live Agent
echo ===========================================================================
echo       ClashBot AlphaStar Agent - Google Play Games PC Bridge
echo ===========================================================================
echo.
echo Make sure Clash Royale is running in Google Play Games on your screen.
echo.
echo [1] Live Auto-Play (AlphaStar plays autonomously with live HUD window)
echo [2] Live Auto-Play Headless (Plays autonomously in background, no HUD window)
echo [3] Dry-Run Mode (Live HUD tactical overlay, no clicks sent)
echo [4] Exit
echo.
set /p mode="Select Mode [1-4] (Default: 1): "
if "%mode%"=="" set mode=1

if "%mode%"=="1" (
    echo Starting Live Auto-Play with HUD Window...
    python deploy_bluestacks.py --checkpoint checkpoints/latest.pt --visualize
) else if "%mode%"=="2" (
    echo Starting Live Auto-Play Headless...
    python deploy_bluestacks.py --checkpoint checkpoints/latest.pt --no-gui
) else if "%mode%"=="3" (
    echo Starting Dry-Run Mode with HUD Overlay...
    python deploy_bluestacks.py --checkpoint checkpoints/latest.pt --dry-run --visualize
) else (
    exit /b 0
)
pause

