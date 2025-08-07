@echo off
echo ========================================
echo CryptoSmartTrader V2 - Windows Setup
echo Complete Environment Configuration
echo ========================================

REM Run pre-installation health check
echo [0/10] Running pre-installation health check...
python system_health_check.py
if errorlevel 1 (
    echo.
    echo WARNING: Health check found critical issues
    echo Review the results above before continuing
    echo.
    set /p continue="Continue anyway? (y/N): "
    if /i not "%continue%"=="y" (
        echo Installation cancelled
        pause
        exit /b 1
    )
)

REM Check admin privileges
net session >nul 2>&1
if errorlevel 1 (
    echo WARNING: Not running as administrator
    echo Some features may require admin privileges
    echo.
)

REM Check Python installation
echo [1/10] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.10+ from:
    echo https://www.python.org/downloads/
    echo.
    echo During installation, make sure to:
    echo 1. Check "Add Python to PATH"
    echo 2. Choose "Customize installation"
    echo 3. Check "pip" and "py launcher"
    echo.
    pause
    exit /b 1
) else (
    python --version
    echo ✓ Python is available
)

REM Check Git installation
echo.
echo [2/10] Checking Git installation...
git --version >nul 2>&1
if errorlevel 1 (
    echo WARNING: Git is not installed
    echo Git is recommended for version control and updates
    echo Download from: https://git-scm.com/downloads
) else (
    git --version
    echo ✓ Git is available
)

REM Update pip and essential tools
echo.
echo [3/10] Updating Python package manager...
python -m pip install --upgrade pip setuptools wheel

REM Install TA-Lib (Windows specific)
echo.
echo [4/10] Installing TA-Lib binary for Windows...
python -c "import sys; print('Python version:', sys.version_info[:2])"
python -c "import platform; print('Architecture:', platform.architecture()[0])"

REM Detect Python version and install appropriate TA-Lib
for /f "tokens=2 delims=." %%i in ('python -c "import sys; print(sys.version_info[0]); print(sys.version_info[1])"') do set PYTHON_MINOR=%%i
for /f "tokens=1 delims=." %%i in ('python -c "import sys; print(sys.version_info[0]); print(sys.version_info[1])"') do set PYTHON_MAJOR=%%i

echo Detected Python %PYTHON_MAJOR%.%PYTHON_MINOR%

if "%PYTHON_MAJOR%"=="3" if "%PYTHON_MINOR%"=="10" (
    echo Installing TA-Lib for Python 3.10...
    pip install https://github.com/mrjbq7/ta-lib/releases/download/0.4.0/TA_Lib-0.4.0-cp310-cp310-win_amd64.whl
) else if "%PYTHON_MAJOR%"=="3" if "%PYTHON_MINOR%"=="11" (
    echo Installing TA-Lib for Python 3.11...
    pip install https://github.com/mrjbq7/ta-lib/releases/download/0.4.0/TA_Lib-0.4.0-cp311-cp311-win_amd64.whl
) else if "%PYTHON_MAJOR%"=="3" if "%PYTHON_MINOR%"=="12" (
    echo Installing TA-Lib for Python 3.12...
    pip install https://github.com/mrjbq7/ta-lib/releases/download/0.4.0/TA_Lib-0.4.0-cp312-cp312-win_amd64.whl
) else (
    echo WARNING: Unsupported Python version for pre-built TA-Lib
    echo Attempting generic installation...
    pip install TA-Lib
)

REM Install core dependencies
echo.
echo [5/10] Installing core dependencies...
pip install streamlit pandas numpy plotly scikit-learn xgboost
pip install ccxt textblob requests python-dateutil

REM Install distributed system dependencies
echo.
echo [6/10] Installing distributed orchestrator dependencies...
pip install psutil setproctitle aiohttp asyncio
pip install multiprocessing prometheus-client
pip install dependency-injector pydantic pydantic-settings
pip install tenacity schedule python-json-logger

REM Install optional GPU dependencies
echo.
echo [7/10] Installing GPU/ML acceleration (optional)...
echo Checking for NVIDIA GPU...
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo ✓ NVIDIA GPU detected, installing CUDA dependencies...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    pip install transformers datasets
    echo ✓ GPU acceleration enabled
) else (
    echo No NVIDIA GPU detected, skipping CUDA dependencies
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
)

REM Install development tools
echo.
echo [8/10] Installing development and testing tools...
pip install pytest pytest-asyncio pytest-cov
pip install black flake8 mypy pre-commit

REM Setup project structure
echo.
echo [9/10] Setting up project structure...

REM Create directories
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "exports" mkdir exports
if not exist "cache" mkdir cache
if not exist ".streamlit" mkdir .streamlit

REM Create Streamlit config
echo [server] > .streamlit\config.toml
echo headless = true >> .streamlit\config.toml
echo address = "0.0.0.0" >> .streamlit\config.toml
echo port = 5000 >> .streamlit\config.toml
echo.
echo [browser] >> .streamlit\config.toml
echo gatherUsageStats = false >> .streamlit\config.toml

REM Create environment file
if not exist ".env" (
    echo # CryptoSmartTrader V2 Environment Configuration > .env
    echo # API Keys (optional but recommended) >> .env
    echo OPENAI_API_KEY=your_openai_api_key_here >> .env
    echo. >> .env
    echo # Exchange API Keys (optional, for paper trading) >> .env
    echo BINANCE_API_KEY=your_binance_api_key_here >> .env
    echo BINANCE_SECRET_KEY=your_binance_secret_here >> .env
    echo KRAKEN_API_KEY=your_kraken_api_key_here >> .env
    echo KRAKEN_SECRET_KEY=your_kraken_secret_here >> .env
    echo. >> .env
    echo # System Configuration >> .env
    echo ENVIRONMENT=production >> .env
    echo LOG_LEVEL=INFO >> .env
    echo MAX_WORKERS=8 >> .env
    echo. >> .env
    echo # Monitoring Configuration >> .env
    echo MONITORING_PORT=8001 >> .env
    echo METRICS_RETENTION_HOURS=24 >> .env
    echo ENABLE_ALERTS=true >> .env
    echo. >> .env
    echo # Performance Settings >> .env
    echo ENABLE_GPU=auto >> .env
    echo CACHE_ENABLED=true >> .env
    echo ASYNC_WORKERS=4 >> .env
    echo ✓ Created .env configuration file
) else (
    echo ✓ Configuration file already exists
)

REM Setup pre-commit hooks
if exist .pre-commit-config.yaml (
    echo Setting up pre-commit hooks...
    pre-commit install
    echo ✓ Pre-commit hooks installed
)

REM Windows-specific optimizations
echo.
echo [10/10] Applying Windows optimizations...

REM Create Windows shortcuts
echo Creating desktop shortcuts...
echo @echo off > "%USERPROFILE%\Desktop\CryptoSmartTrader.bat"
echo cd /d "%CD%" >> "%USERPROFILE%\Desktop\CryptoSmartTrader.bat"
echo start_cryptotrader.bat >> "%USERPROFILE%\Desktop\CryptoSmartTrader.bat"

echo @echo off > "%USERPROFILE%\Desktop\CryptoSmartTrader-Monitoring.bat"
echo cd /d "%CD%" >> "%USERPROFILE%\Desktop\CryptoSmartTrader-Monitoring.bat"
echo start_monitoring.bat >> "%USERPROFILE%\Desktop\CryptoSmartTrader-Monitoring.bat"

REM Windows firewall notification
echo.
echo ========================================
echo WINDOWS SECURITY NOTICE
echo ========================================
echo When you first run CryptoSmartTrader, Windows may show:
echo "Windows Security Alert - Python wants to access the network"
echo.
echo ✓ Click "Allow access" to enable:
echo   - Market data downloading
echo   - API connections to exchanges
echo   - Monitoring dashboard access
echo.
echo This is normal and required for the system to function.
echo ========================================

REM Final verification
echo.
echo Testing critical imports...
python -c "
import sys
print('Python version:', sys.version)
print()

tests = [
    ('Streamlit', 'streamlit'),
    ('Pandas', 'pandas'), 
    ('NumPy', 'numpy'),
    ('Plotly', 'plotly'),
    ('CCXT', 'ccxt'),
    ('TA-Lib', 'talib'),
    ('AioHTTP', 'aiohttp'),
    ('PSUtil', 'psutil'),
    ('SetProcTitle', 'setproctitle'),
]

passed = 0
for name, module in tests:
    try:
        __import__(module)
        print(f'✓ {name}: OK')
        passed += 1
    except ImportError as e:
        print(f'✗ {name}: FAILED - {e}')

print(f'\nTest Results: {passed}/{len(tests)} passed')

if passed == len(tests):
    print('\n🎉 ALL DEPENDENCIES INSTALLED SUCCESSFULLY!')
else:
    print(f'\n⚠️ {len(tests)-passed} dependencies failed to install')
"

echo.
echo ========================================
echo SETUP COMPLETE!
echo ========================================
echo.
echo Your CryptoSmartTrader V2 environment is ready!
echo.
echo Start options:
echo 1. start_cryptotrader.bat - Main Streamlit dashboard
echo 2. start_distributed_system.bat - Advanced distributed mode
echo 3. start_monitoring.bat - Monitoring dashboard only
echo 4. Desktop shortcuts created for quick access
echo.
echo Next steps:
echo 1. Edit .env file with your API keys (optional but recommended)
echo 2. Run your preferred start script
echo 3. Open http://localhost:5000 in your browser
echo.
echo Documentation:
echo - README.md - Quick start guide
echo - SETUP_GUIDE.md - Detailed setup instructions
echo - TECHNICAL_REVIEW.md - Technical documentation
echo.
pause