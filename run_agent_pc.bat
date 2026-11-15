@echo off
title ClashBot - Live MuMu Player Bridge
echo ===========================================================================
echo       ClashBot AlphaStar Agent - Live MuMu Player Deployment
echo ===========================================================================
echo.
echo Supported Environments:
echo   - MuMu Player 12 / Android Emulator (Background ADB)
echo.
echo [1] Live Auto-Play: MuMu Player (Background ADB)
echo [2] Live Auto-Play with OpenCV HUD Overlay (deploy_mumu.py --visualize)
echo [3] Dry-Run Simulation: Test Perception & Model (No clicks sent)
echo [4] Exit
echo.
set /p mode="Select Mode [1-4] (Default: 1): "
if "%mode%"=="" set mode=1

if "%mode%"=="1" (
    echo Starting Background Auto-Play on MuMu Player...
    python deploy_mumu.py --checkpoint checkpoints/latest.pt --matches 5
) else if "%mode%"=="2" (
    echo Starting Background Auto-Play with HUD Overlay...
    python deploy_mumu.py --checkpoint checkpoints/latest.pt --matches 5 --visualize
) else if "%mode%"=="3" (
    echo Running Dry-Run Simulation...
    python deploy_mumu.py --checkpoint checkpoints/latest.pt --dry-run --visualize
) else (
    exit /b 0
)
pause

