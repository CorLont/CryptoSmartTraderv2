@echo off
title CryptoSmartTrader V2 - Dependency Installation
color 0E
echo =============================================
echo  CryptoSmartTrader V2 - Dependency Installation
echo =============================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please run setup_windows_environment.bat first
    pause
    exit /b 1
)

echo [INFO] Installing/updating all dependencies...
echo [INFO] This may take several minutes depending on your internet connection
echo.

REM Upgrade pip first
echo [PIP] Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip, continuing...
)

REM Core web framework
echo [CORE] Installing Streamlit...
python -m pip install --upgrade streamlit
if errorlevel 1 (
    echo [ERROR] Failed to install Streamlit
    pause
    exit /b 1
)

REM Data processing
echo [DATA] Installing data processing libraries...
python -m pip install --upgrade pandas numpy
if errorlevel 1 (
    echo [ERROR] Failed to install data processing libraries
    pause
    exit /b 1
)

REM Visualization
echo [VIZ] Installing visualization libraries...
python -m pip install --upgrade plotly
if errorlevel 1 (
    echo [ERROR] Failed to install visualization libraries
    pause
    exit /b 1
)

REM Trading and exchange connectivity
echo [TRADING] Installing trading libraries...
python -m pip install --upgrade ccxt
if errorlevel 1 (
    echo [ERROR] Failed to install trading libraries
    pause
    exit /b 1
)

REM Machine learning
echo [ML] Installing machine learning libraries...
python -m pip install --upgrade scikit-learn xgboost
if errorlevel 1 (
    echo [WARNING] Some ML libraries failed to install
    echo Core functionality will still work
)

REM Natural language processing
echo [NLP] Installing NLP libraries...
python -m pip install --upgrade textblob
if errorlevel 1 (
    echo [WARNING] NLP libraries failed to install
    echo Sentiment analysis may have reduced functionality
)

REM Async and HTTP
echo [ASYNC] Installing async and HTTP libraries...
python -m pip install --upgrade aiohttp
if errorlevel 1 (
    echo [WARNING] Async libraries failed to install
)

REM Configuration and validation
echo [CONFIG] Installing configuration libraries...
python -m pip install --upgrade pydantic pydantic-settings
if errorlevel 1 (
    echo [WARNING] Configuration libraries failed to install
)

REM Scheduling and utilities
echo [UTILS] Installing utility libraries...
python -m pip install --upgrade schedule tenacity
if errorlevel 1 (
    echo [WARNING] Utility libraries failed to install
)

REM System monitoring
echo [MONITOR] Installing monitoring libraries...
python -m pip install --upgrade psutil setproctitle
if errorlevel 1 (
    echo [WARNING] Monitoring libraries failed to install
)

REM Optional performance libraries
echo [PERF] Installing performance libraries...
python -m pip install --upgrade numba
if errorlevel 1 (
    echo [INFO] Numba failed to install (optional performance library)
)

REM Prometheus for metrics (optional)
echo [METRICS] Installing metrics libraries...
python -m pip install --upgrade prometheus-client
if errorlevel 1 (
    echo [INFO] Prometheus client failed to install (optional metrics library)
)

REM JSON logging
echo [LOG] Installing logging libraries...
python -m pip install --upgrade python-json-logger
if errorlevel 1 (
    echo [WARNING] JSON logging failed to install
)

REM Web scraping for sentiment analysis
echo [SCRAPE] Installing web scraping libraries...
python -m pip install --upgrade trafilatura
if errorlevel 1 (
    echo [WARNING] Web scraping libraries failed to install
    echo News sentiment analysis may be limited
)

REM OpenAI integration (optional)
echo [AI] Installing AI libraries...
python -m pip install --upgrade openai
if errorlevel 1 (
    echo [INFO] OpenAI library failed to install (optional AI enhancement)
)

REM Dependency injection framework
echo [DI] Installing dependency injection...
python -m pip install --upgrade dependency-injector
if errorlevel 1 (
    echo [WARNING] Dependency injection framework failed to install
)

REM Security libraries
echo [SEC] Installing security libraries...
python -m pip install --upgrade hvac
if errorlevel 1 (
    echo [INFO] Security libraries failed to install (optional vault integration)
)

echo.
echo [TEST] Testing critical dependencies...
python -c "
import sys
failed_imports = []
critical_packages = [
    'streamlit', 'pandas', 'numpy', 'plotly', 'ccxt'
]

print('[TEST] Testing critical package imports...')
for package in critical_packages:
    try:
        __import__(package)
        print(f'[OK] {package}')
    except ImportError:
        print(f'[FAIL] {package}')
        failed_imports.append(package)

if failed_imports:
    print(f'[ERROR] Failed to import: {failed_imports}')
    print('[FIX] Please check your internet connection and try again')
    sys.exit(1)
else:
    print('[SUCCESS] All critical dependencies installed and working!')
"

if errorlevel 1 (
    echo [ERROR] Dependency test failed
    echo Please check the error messages above and try again
    pause
    exit /b 1
)

echo.
echo =============================================
echo   Dependency Installation Complete!
echo =============================================
echo.
echo [SUCCESS] All dependencies installed successfully
echo [INFO] You can now run start_cryptotrader.bat
echo.
pause