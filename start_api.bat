@echo off
:: CryptoSmartTrader V2 - API Server
:: Start FastAPI backend met metrics endpoint

title CryptoSmartTrader V2 - API Server

echo ================================================
echo CryptoSmartTrader V2 - Starting API Server
echo ================================================

:: Set environment variables
set PYTHONPATH=%CD%
set UVICORN_WORKERS=1

:: Activate virtual environment
call .venv\Scripts\activate
if errorlevel 1 (
    echo ERROR: Virtual environment niet gevonden. Run setup_env.bat eerst.
    pause
    exit /b 1
)

echo ✓ Virtual environment geactiveerd
echo Starting API server op port 8001...

:: Start API server
uvicorn src.cryptosmarttrader.api.main:app --host 0.0.0.0 --port 8001 --proxy-headers --timeout-keep-alive 120 --reload

pause