@echo off
title ClashBot - Google Play Games PC Live Agent
echo ===========================================================================
echo       ClashBot AlphaStar Agent - Google Play Games PC Bridge
echo ===========================================================================
echo.
echo Make sure Clash Royale is running in Google Play Games on your screen.
echo.
echo [1] Live Auto-Play (Google Play Games PC - Hardware SendInput)
echo [2] Live Auto-Play Drag-and-Drop Mode (--drag)
echo [3] Test Placement & Calibration Grid Overlay (--test-placement)
echo [4] Legacy BlueStacks / OpenCV HUD Bridge (deploy_bluestacks.py)
echo [5] Exit
echo.
set /p mode="Select Mode [1-5] (Default: 1): "
if "%mode%"=="" set mode=1

if "%mode%"=="1" (
    echo Starting Live Auto-Play on Google Play Games PC...
    python deploy_google_play.py --live --checkpoint checkpoints/latest.pt
) else if "%mode%"=="2" (
    echo Starting Live Auto-Play Drag Mode...
    python deploy_google_play.py --live --drag --checkpoint checkpoints/latest.pt
) else if "%mode%"=="3" (
    echo Running Test-Placement & Visual Calibration Grid...
    python deploy_google_play.py --test-placement
) else if "%mode%"=="4" (
    echo Starting Legacy BlueStacks Bridge...
    python deploy_bluestacks.py --checkpoint checkpoints/latest.pt --visualize
) else (
    exit /b 0
)
pause

