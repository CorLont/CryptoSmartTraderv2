@echo off
echo ========================================
echo CryptoSmartTrader V2 Dependency Installer
echo Distributed System with Enhanced Monitoring
echo ========================================

REM Check Python version
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

REM Update pip and core tools
echo.
echo [1/8] Updating pip and core tools...
python -m pip install --upgrade pip setuptools wheel

REM Install TA-Lib binary (adjust for Python version)
echo.
echo [2/8] Installing TA-Lib binary...
REM For Python 3.10 64-bit
pip install https://github.com/mrjbq7/ta-lib/releases/download/0.4.0/TA_Lib-0.4.0-cp310-cp310-win_amd64.whl
REM Uncomment below for Python 3.11 64-bit
REM pip install https://github.com/mrjbq7/ta-lib/releases/download/0.4.0/TA_Lib-0.4.0-cp311-cp311-win_amd64.whl

REM Install core dependencies
echo.
echo [3/8] Installing core Python dependencies...
pip install streamlit pandas numpy plotly scikit-learn xgboost
pip install ccxt textblob requests python-dateutil
pip install psutil setproctitle

REM Install distributed system dependencies
echo.
echo [4/8] Installing distributed orchestrator dependencies...
pip install aiohttp asyncio multiprocessing
pip install prometheus-client
pip install dependency-injector pydantic pydantic-settings
pip install tenacity schedule

REM Install monitoring and async dependencies  
echo.
echo [5/8] Installing monitoring and async dependencies...
pip install aiofiles aiohttp-cors
pip install uvloop
pip install python-json-logger

REM Install security and vault dependencies
echo.
echo [6/8] Installing security dependencies...
pip install hvac cryptography
pip install python-dotenv

REM Install optional GPU/ML acceleration
echo.
echo [7/8] Installing optional GPU/ML dependencies...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets
REM Uncomment if you have CUDA-capable GPU
REM pip install cupy-cuda11x

REM Install development and testing tools
echo.
echo [8/8] Installing development tools...
pip install pytest pytest-asyncio pytest-cov
pip install black flake8 mypy
pip install pre-commit

REM Install additional requirements if file exists
if exist requirements.txt (
    echo.
    echo Installing additional requirements from requirements.txt...
    pip install -r requirements.txt
)

REM CUDA Toolkit notice
echo.
echo ========================================
echo MANUAL INSTALLATION REQUIRED:
echo ========================================
echo.
echo 1. NVIDIA CUDA Toolkit (for GPU acceleration):
echo    Download from: https://developer.nvidia.com/cuda-downloads
echo    Required for GPU-accelerated ML training
echo.
echo 2. Redis Server (optional, for advanced message queuing):
echo    Download from: https://github.com/microsoftarchive/redis/releases
echo    Or use Windows Subsystem for Linux (WSL)
echo.

REM Setup pre-commit hooks
echo Setting up pre-commit hooks...
if exist .pre-commit-config.yaml (
    pre-commit install
    echo Pre-commit hooks installed successfully
) else (
    echo No .pre-commit-config.yaml found, skipping pre-commit setup
)

REM Create necessary directories
echo.
echo Creating necessary directories...
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "exports" mkdir exports
if not exist ".streamlit" mkdir .streamlit

REM Create Streamlit config
echo.
echo Creating Streamlit configuration...
echo [server] > .streamlit\config.toml
echo headless = true >> .streamlit\config.toml
echo address = "0.0.0.0" >> .streamlit\config.toml
echo port = 5000 >> .streamlit\config.toml
echo.
echo [browser] >> .streamlit\config.toml
echo gatherUsageStats = false >> .streamlit\config.toml

REM Create environment template
echo.
echo Creating environment template...
if not exist ".env" (
    echo # CryptoSmartTrader Environment Configuration > .env
    echo # API Keys (add your actual keys) >> .env
    echo OPENAI_API_KEY=your_openai_api_key_here >> .env
    echo BINANCE_API_KEY=your_binance_api_key_here >> .env
    echo BINANCE_SECRET_KEY=your_binance_secret_here >> .env
    echo. >> .env
    echo # System Configuration >> .env
    echo ENVIRONMENT=development >> .env
    echo LOG_LEVEL=INFO >> .env
    echo MAX_WORKERS=8 >> .env
    echo. >> .env
    echo # Monitoring >> .env
    echo MONITORING_PORT=8001 >> .env
    echo METRICS_RETENTION_HOURS=24 >> .env
)

REM Test imports
echo.
echo ========================================
echo Testing critical imports...
echo ========================================
python -c "
try:
    import streamlit as st
    print('✓ Streamlit: OK')
except ImportError as e:
    print('✗ Streamlit: FAILED -', e)

try:
    import pandas as pd
    print('✓ Pandas: OK')
except ImportError as e:
    print('✗ Pandas: FAILED -', e)

try:
    import numpy as np
    print('✓ NumPy: OK')
except ImportError as e:
    print('✗ NumPy: FAILED -', e)

try:
    import ccxt
    print('✓ CCXT: OK')
except ImportError as e:
    print('✗ CCXT: FAILED -', e)

try:
    import talib
    print('✓ TA-Lib: OK')
except ImportError as e:
    print('✗ TA-Lib: FAILED -', e)

try:
    import aiohttp
    print('✓ AioHTTP: OK')
except ImportError as e:
    print('✗ AioHTTP: FAILED -', e)

try:
    import psutil
    print('✓ PSUtil: OK')
except ImportError as e:
    print('✗ PSUtil: FAILED -', e)

try:
    import setproctitle
    print('✓ SetProcTitle: OK')
except ImportError as e:
    print('✗ SetProcTitle: FAILED -', e)

try:
    from core.distributed_orchestrator import distributed_orchestrator
    print('✓ Distributed Orchestrator: OK')
except ImportError as e:
    print('✗ Distributed Orchestrator: FAILED -', e)

try:
    from core.centralized_monitoring import centralized_monitoring
    print('✓ Centralized Monitoring: OK')
except ImportError as e:
    print('✗ Centralized Monitoring: FAILED -', e)
"

echo.
echo ========================================
echo INSTALLATION COMPLETE!
echo ========================================
echo.
echo Next steps:
echo 1. Edit .env file with your API keys
echo 2. Install CUDA Toolkit for GPU acceleration (optional)
echo 3. Start the system:
echo    - Streamlit Dashboard: python -m streamlit run app.py --server.port 5000
echo    - Distributed System: python run_distributed_system.py
echo    - Monitoring Only: python run_distributed_system.py --monitoring-only
echo.
echo Available start scripts:
echo - start_cryptotrader.bat (Main Streamlit interface)
echo - start_distributed_system.bat (Advanced distributed mode)
echo - start_monitoring.bat (Monitoring dashboard only)
echo.
pause