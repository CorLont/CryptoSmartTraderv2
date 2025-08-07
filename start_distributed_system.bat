@echo off
echo ========================================
echo CryptoSmartTrader V2 - Distributed System
echo ========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please run install_dependencies.bat first
    pause
    exit /b 1
)

REM Set environment variables
set PYTHONPATH=%CD%
set ENVIRONMENT=production

echo Starting distributed orchestrator with monitoring...
echo.
echo Available endpoints:
echo - Main Dashboard: http://localhost:5000
echo - Monitoring API: http://localhost:8001/dashboard
echo - Health Check: http://localhost:8001/health
echo - Metrics: http://localhost:8001/metrics
echo.

REM Start the distributed system
python run_distributed_system.py

REM If the script exits, show status
if errorlevel 1 (
    echo.
    echo ERROR: Distributed system failed to start
    echo Check the logs in the logs/ directory for details
) else (
    echo.
    echo Distributed system stopped gracefully
)

echo.
pause