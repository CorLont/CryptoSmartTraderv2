@echo off
:: CryptoSmartTrader V2 - Background Workers
:: Start data ingestion, agents en execution workers

title CryptoSmartTrader V2 - Workers

echo ================================================
echo CryptoSmartTrader V2 - Starting Background Workers
echo ================================================

:: Set environment variables
set PYTHONPATH=%CD%

:: Activate virtual environment
call .venv\Scripts\activate
if errorlevel 1 (
    echo ERROR: Virtual environment niet gevonden. Run setup_env.bat eerst.
    pause
    exit /b 1
)

echo ✓ Virtual environment geactiveerd
echo Starting orchestrator met alle agents...

:: Start centralized observability API
echo Starting Centralized Observability API...
start /B python -m src.cryptosmarttrader.observability.centralized_observability_api

:: Wait for API to start
timeout /t 3 /nobreak >nul

:: Start main orchestrator
echo Starting main orchestrator...
python -m src.cryptosmarttrader.orchestrator.run_all

pause