@echo off
title CryptoSmartTrader V2 - One-Click Runner
echo ==========================================
echo CryptoSmartTrader V2 - One-Click Runner
echo ==========================================
echo.
echo 🔄 Complete pipeline: scrape → features → predict → strict gate → export → eval → logs/daily
echo.

REM Set environment
set PYTHONPATH=%cd%
set CRYPTOSMARTTRADER_ENV=production

REM Start timestamp
echo 🕐 Started: %date% %time%
echo.

REM Run complete pipeline
python scripts/oneclick_pipeline.py
if %errorlevel% neq 0 (
    echo.
    echo ❌ Pipeline failed with error code: %errorlevel%
    echo 📋 Check logs/daily/ for detailed error information
    echo.
    pause
    exit /b 1
)

echo.
echo ✅ Complete pipeline finished successfully!
echo 🕐 Completed: %date% %time%
echo.
echo 📋 Results available in:
echo    - exports/daily/
echo    - logs/daily/
echo.
echo 🚀 To view results, run: start_dashboard.bat
echo.
pause
