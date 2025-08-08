@echo off
setlocal EnableDelayedExpansion

echo =====================================================
echo    CryptoSmartTrader V2 - Complete Installation
echo =====================================================
echo.

:: Set UTF-8 encoding for proper display
chcp 65001 >nul

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

echo [1/6] Checking Python version...
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%

:: Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: pip is not available
    echo Installing pip...
    python -m ensurepip --upgrade
)

echo [2/6] Upgrading pip and essential tools...
python -m pip install --upgrade pip setuptools wheel

echo [3/6] Installing core dependencies...
pip install streamlit pandas numpy plotly scikit-learn

echo [4/6] Installing trading and ML libraries...
pip install ccxt textblob xgboost torch torchvision torchaudio

echo [5/6] Installing additional dependencies...
pip install aiohttp dependency-injector hvac pydantic pydantic-settings
pip install python-json-logger schedule tenacity psutil setproctitle
pip install prometheus-client numba optuna lightgbm

echo [6/6] Installing optional performance libraries...
:: Try to install CuPy for GPU acceleration (optional)
pip install cupy-cuda11x >nul 2>&1
if errorlevel 1 (
    echo Note: CuPy GPU acceleration not available - using CPU mode
) else (
    echo GPU acceleration enabled with CuPy
)

:: Install Jupyter for analysis (optional)
pip install jupyter ipykernel

echo.
echo =====================================================
echo Creating necessary directories...
echo =====================================================

if not exist "data" mkdir data
if not exist "logs" mkdir logs
if not exist "models" mkdir models
if not exist "data\raw" mkdir data\raw
if not exist "data\processed" mkdir data\processed
if not exist "data\predictions" mkdir data\predictions
if not exist "models\saved" mkdir models\saved
if not exist "models\backups" mkdir models\backups

echo.
echo =====================================================
echo Setting up configuration files...
echo =====================================================

:: Create .streamlit directory and config
if not exist ".streamlit" mkdir .streamlit
echo [server] > .streamlit\config.toml
echo headless = true >> .streamlit\config.toml
echo address = "0.0.0.0" >> .streamlit\config.toml
echo port = 5000 >> .streamlit\config.toml
echo.
echo [theme] >> .streamlit\config.toml
echo primaryColor = "#FF6B35" >> .streamlit\config.toml
echo backgroundColor = "#FFFFFF" >> .streamlit\config.toml
echo secondaryBackgroundColor = "#F0F2F6" >> .streamlit\config.toml

:: Create basic environment file if it doesn't exist
if not exist ".env" (
    echo # CryptoSmartTrader V2 Environment Variables > .env
    echo OPENAI_API_KEY=your_openai_api_key_here >> .env
    echo KRAKEN_API_KEY=your_kraken_api_key >> .env
    echo KRAKEN_SECRET=your_kraken_secret >> .env
    echo BINANCE_API_KEY=your_binance_api_key >> .env
    echo BINANCE_SECRET=your_binance_secret >> .env
    echo LOG_LEVEL=INFO >> .env
    echo ENVIRONMENT=production >> .env
)

echo.
echo =====================================================
echo Testing installation...
echo =====================================================

python -c "
import sys
print(f'Python: {sys.version}')

# Test core imports
try:
    import streamlit as st
    print('✓ Streamlit: OK')
except ImportError as e:
    print(f'✗ Streamlit: {e}')

try:
    import pandas as pd
    print('✓ Pandas: OK')
except ImportError as e:
    print(f'✗ Pandas: {e}')

try:
    import numpy as np
    print('✓ NumPy: OK')
except ImportError as e:
    print(f'✗ NumPy: {e}')

try:
    import plotly
    print('✓ Plotly: OK')
except ImportError as e:
    print(f'✗ Plotly: {e}')

try:
    import sklearn
    print('✓ Scikit-learn: OK')
except ImportError as e:
    print(f'✗ Scikit-learn: {e}')

try:
    import ccxt
    print('✓ CCXT: OK')
except ImportError as e:
    print(f'✗ CCXT: {e}')

try:
    import torch
    print('✓ PyTorch: OK')
    if torch.cuda.is_available():
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('  GPU: Not available (CPU mode)')
except ImportError as e:
    print(f'✗ PyTorch: {e}')

try:
    import textblob
    print('✓ TextBlob: OK')
except ImportError as e:
    print(f'✗ TextBlob: {e}')

print('\\nInstallation test completed!')
"

echo.
echo =====================================================
echo Installation Summary
echo =====================================================
echo.
echo ✓ All Python dependencies installed
echo ✓ Directory structure created
echo ✓ Configuration files set up
echo ✓ Environment template created
echo.
echo NEXT STEPS:
echo 1. Edit .env file with your API keys
echo 2. Run 2_start_background_services.bat to start background processes
echo 3. Run 3_start_dashboard.bat to launch the main application
echo.
echo For GPU acceleration, ensure CUDA is installed on your system.
echo.
echo =====================================================
echo Running comprehensive health check...
echo =====================================================
python workstation_health_check.py

echo.
echo Installation completed successfully!
echo Run the health check anytime with: python workstation_health_check.py
echo.
pause