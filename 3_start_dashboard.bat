@echo off
setlocal EnableDelayedExpansion

echo =====================================================
echo    CryptoSmartTrader V2 - Dashboard Launcher
echo =====================================================
echo.

:: Set UTF-8 encoding
chcp 65001 >nul

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please run 1_install_all_dependencies.bat first
    pause
    exit /b 1
)

:: Check if Streamlit is installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Streamlit not installed. Please run 1_install_all_dependencies.bat first
    pause
    exit /b 1
)

echo Preparing CryptoSmartTrader V2 Dashboard...
echo.

:: Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

:: Check if configuration files exist
if not exist ".streamlit\config.toml" (
    echo Creating Streamlit configuration...
    if not exist ".streamlit" mkdir .streamlit
    echo [server] > .streamlit\config.toml
    echo headless = true >> .streamlit\config.toml
    echo address = "0.0.0.0" >> .streamlit\config.toml
    echo port = 5000 >> .streamlit\config.toml
    echo enableXsrfProtection = false >> .streamlit\config.toml
    echo.
    echo [theme] >> .streamlit\config.toml
    echo primaryColor = "#FF6B35" >> .streamlit\config.toml
    echo backgroundColor = "#FFFFFF" >> .streamlit\config.toml
    echo secondaryBackgroundColor = "#F0F2F6" >> .streamlit\config.toml
    echo textColor = "#262730" >> .streamlit\config.toml
)

:: Check if environment file exists
if not exist ".env" (
    echo Creating environment template...
    echo # CryptoSmartTrader V2 Environment Variables > .env
    echo OPENAI_API_KEY=your_openai_api_key_here >> .env
    echo KRAKEN_API_KEY=your_kraken_api_key >> .env
    echo KRAKEN_SECRET=your_kraken_secret >> .env
    echo ENVIRONMENT=production >> .env
    echo LOG_LEVEL=INFO >> .env
)

:: System Health Check
echo [1/4] Running quick system health check...
python -c "
import sys
import os
from pathlib import Path

print('System Health Check:')
print('=' * 30)

# Check Python version
import sys
print(f'✓ Python {sys.version.split()[0]}')

# Check key dependencies
dependencies = [
    'streamlit', 'pandas', 'numpy', 'plotly', 
    'scikit-learn', 'ccxt', 'textblob'
]

for dep in dependencies:
    try:
        __import__(dep)
        print(f'✓ {dep}')
    except ImportError:
        print(f'✗ {dep} - Missing')

# Check directory structure
directories = ['data', 'logs', 'models', 'core', 'dashboards']
for directory in directories:
    if Path(directory).exists():
        print(f'✓ {directory}/')
    else:
        print(f'⚠ {directory}/ - Missing')

# Check configuration
if Path('.streamlit/config.toml').exists():
    print('✓ Streamlit config')
else:
    print('⚠ Streamlit config - Missing')

if Path('.env').exists():
    print('✓ Environment file')
else:
    print('⚠ Environment file - Missing')

print('\\nHealth check completed!')
"

echo.
echo [2/4] Verifying background services...

:: Check if background services are running
tasklist /FI "WINDOWTITLE eq CryptoTrader*" 2>nul | find "python.exe" >nul
if errorlevel 1 (
    echo ⚠ Background services not detected
    echo   Run 2_start_background_services.bat to start them
) else (
    echo ✓ Background services are running
)

echo.
echo [3/4] Starting dashboard server...

:: Kill any existing Streamlit processes on port 5000
for /f "tokens=5" %%a in ('netstat -ano ^| find ":5000"') do taskkill /PID %%a /F >nul 2>&1

:: Set environment variables
set PYTHONPATH=%CD%
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

echo.
echo =====================================================
echo    CryptoSmartTrader V2 Dashboard Starting...
echo =====================================================
echo.
echo Dashboard URL: http://localhost:5000
echo.
echo Available Dashboards:
echo ✓ Main Dashboard - System overview and quick actions
echo ✓ Comprehensive Market - Real-time market analysis
echo ✓ Causal Inference - Market relationship discovery
echo ✓ RL Portfolio Allocation - AI-powered optimization  
echo ✓ Self-Healing System - Autonomous protection
echo ✓ Synthetic Data Augmentation - Stress testing
echo ✓ Human-in-the-Loop - Expert feedback integration
echo ✓ Shadow Trading - Paper trading validation
echo.
echo [4/4] Launching Streamlit application...
echo.

:: Start the main dashboard
streamlit run app_minimal.py --server.headless=true --server.address=0.0.0.0 --server.port=5000

:: If Streamlit exits, show error message
echo.
echo =====================================================
echo Dashboard has stopped
echo =====================================================
echo.
echo If you encountered an error:
echo 1. Check that all dependencies are installed
echo 2. Verify background services are running
echo 3. Check logs/ directory for error details
echo 4. Ensure no other application is using port 5000
echo.
echo To restart: Run this script again
echo To install dependencies: Run 1_install_all_dependencies.bat
echo To start services: Run 2_start_background_services.bat
echo.
pause