@echo off
echo ========================================
echo CryptoSmartTrader V2 - Monitoring Only
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
set ENVIRONMENT=monitoring

echo Starting monitoring dashboard...
echo.
echo Monitoring endpoints:
echo - Dashboard: http://localhost:8001/dashboard
echo - Health API: http://localhost:8001/health
echo - Metrics API: http://localhost:8001/metrics
echo - Alerts API: http://localhost:8001/alerts
echo.
echo This mode only runs the monitoring server without agents.
echo Use this for system observation and debugging.
echo.

REM Start monitoring only
python run_distributed_system.py --monitoring-only

REM If the script exits, show status
if errorlevel 1 (
    echo.
    echo ERROR: Monitoring system failed to start
    echo Check the logs for details
) else (
    echo.
    echo Monitoring system stopped gracefully
)

echo.
pause