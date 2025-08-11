@echo off
REM CryptoSmartTrader V2 - Production Run Script
setlocal enabledelayedexpansion

REM Generate run ID
set RUN_ID=%DATE:~6,4%%DATE:~3,2%%DATE:~0,2%_%TIME:~0,2%%TIME:~3,2%
set RUN_ID=%RUN_ID: =0%

echo ===================================
echo CryptoSmartTrader V2 Production Run
echo Run ID: %RUN_ID%
echo ===================================

REM Activate virtual environment
call .venv\Scripts\activate
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    echo Run install.bat first
    pause
    exit /b 1
)

REM Create run log directory
mkdir logs\daily\%RUN_ID% 2>nul
set LOG_DIR=logs\daily\%RUN_ID%

REM Set environment
set PYTHONPATH=%CD%
set RUN_ID=%RUN_ID%

echo Starting production pipeline at %DATE% %TIME%
echo Log directory: %LOG_DIR%
echo.

REM Run orchestrator (handles all steps)
echo Running production orchestrator...
python scripts\orchestrator.py > %LOG_DIR%\orchestrator.log 2>&1
set ORCHESTRATOR_EXIT=%ERRORLEVEL%

if %ORCHESTRATOR_EXIT% equ 0 (
    echo.
    echo ===================================
    echo PRODUCTION PIPELINE SUCCESS
    echo ===================================
    echo Run ID: %RUN_ID%
    echo Logs: %LOG_DIR%
    echo.
    
    REM Check if predictions were generated
    if exist exports\production\predictions.parquet (
        echo ✓ Predictions generated: exports\production\predictions.parquet
    )
    
    if exist logs\daily\latest.json (
        echo ✓ Evaluation report: logs\daily\latest.json
    )
    
    echo.
    echo Starting dashboard...
    echo Dashboard will open at http://localhost:5000
    echo Press Ctrl+C to stop
    echo.
    
    REM Start dashboard
    streamlit run app_working.py --server.port 5000 > %LOG_DIR%\dashboard.log 2>&1
    
) else (
    echo.
    echo ===================================
    echo PRODUCTION PIPELINE FAILED
    echo ===================================
    echo Exit code: %ORCHESTRATOR_EXIT%
    echo.
    echo Check logs in: %LOG_DIR%
    echo.
    echo Common issues:
    echo - Missing API keys in .env file
    echo - Network connectivity issues
    echo - Insufficient disk space
    echo.
    
    REM Show last few lines of orchestrator log
    echo Last orchestrator log lines:
    echo --------------------------------
    if exist %LOG_DIR%\orchestrator.log (
        powershell "Get-Content '%LOG_DIR%\orchestrator.log' -Tail 10"
    )
    
    pause
    exit /b %ORCHESTRATOR_EXIT%
)

pause