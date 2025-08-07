@echo off
echo ========================================
echo CryptoSmartTrader V2 - Main Dashboard
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

echo Starting CryptoSmartTrader main dashboard...
echo.
echo Dashboard will be available at: http://localhost:5000
echo.
echo Available modes:
echo - This script: Main Streamlit dashboard (recommended for users)
echo - start_distributed_system.bat: Advanced distributed mode with monitoring
echo - start_monitoring.bat: Monitoring dashboard only
echo.

REM Start the main Streamlit application
streamlit run app.py --server.port 5000

REM If Streamlit exits, show status
if errorlevel 1 (
    echo.
    echo ERROR: CryptoSmartTrader failed to start
    echo Common issues:
    echo 1. Missing dependencies - run install_dependencies.bat
    echo 2. Port 5000 already in use - close other applications
    echo 3. Missing API keys - check .env file
) else (
    echo.
    echo CryptoSmartTrader stopped gracefully
)

echo.
pause