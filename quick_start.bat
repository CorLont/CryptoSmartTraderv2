@echo off
title CryptoSmartTrader V2 - Quick Start
color 0F
echo ==========================================
echo  CryptoSmartTrader V2 - Quick Start
echo ==========================================

REM Check if this is first run
if not exist "logs" (
    echo [FIRST RUN] Detected first-time setup
    echo Running initial environment setup...
    echo.
    call setup_windows_environment.bat
    if errorlevel 1 (
        echo [ERROR] Environment setup failed
        pause
        exit /b 1
    )
    echo.
    echo [SUCCESS] Environment setup completed
    echo.
)

REM Quick dependency check
echo [CHECK] Verifying system readiness...
python -c "import streamlit, pandas, numpy, plotly, ccxt" >nul 2>&1
if errorlevel 1 (
    echo [REPAIR] Installing missing dependencies...
    call install_dependencies.bat
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
)

echo [OK] System ready
echo.

REM Launch the main application
echo [LAUNCH] Starting CryptoSmartTrader V2...
call start_cryptotrader.bat