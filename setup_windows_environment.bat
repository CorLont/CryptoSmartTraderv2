@echo off
title CryptoSmartTrader V2 - Windows Environment Setup
color 0B
echo ================================================
echo  CryptoSmartTrader V2 - Windows Environment Setup
echo ================================================

REM Check if running as administrator
net session >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Not running as administrator
    echo Some installations may fail without admin privileges
    echo Consider running as administrator for full functionality
    echo.
    timeout /t 3 >nul
)

REM Check if Python is installed
echo [CHECK] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python 3.10 or later from:
    echo https://www.python.org/downloads/
    echo.
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
) else (
    for /f "tokens=2" %%i in ('python --version') do set PYTHON_VERSION=%%i
    echo [OK] Python %PYTHON_VERSION% found
)

REM Check Python version (must be 3.10+)
python -c "
import sys
version = sys.version_info
if version.major < 3 or (version.major == 3 and version.minor < 10):
    print('[ERROR] Python 3.10 or later required')
    print(f'[INFO] Current version: {version.major}.{version.minor}.{version.micro}')
    sys.exit(1)
else:
    print(f'[OK] Python version {version.major}.{version.minor}.{version.micro} is compatible')
"
if errorlevel 1 (
    echo.
    echo Please upgrade to Python 3.10 or later
    pause
    exit /b 1
)

REM Check if pip is available
echo [CHECK] Checking pip installation...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip is not available
    echo.
    echo Please reinstall Python with pip included
    pause
    exit /b 1
) else (
    echo [OK] pip is available
)

REM Upgrade pip to latest version
echo [UPDATE] Upgrading pip to latest version...
python -m pip install --upgrade pip >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip (continuing anyway)
) else (
    echo [OK] pip upgraded successfully
)

REM Create virtual environment (optional but recommended)
echo [SETUP] Setting up project environment...
if not exist "venv" (
    echo [CREATE] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [WARNING] Failed to create virtual environment
        echo Continuing with global Python installation
    ) else (
        echo [OK] Virtual environment created
    )
)

REM Install required system packages
echo [INSTALL] Installing system requirements...

REM Core dependencies
echo [INSTALL] Installing core dependencies...
python -m pip install --upgrade streamlit pandas numpy plotly
if errorlevel 1 (
    echo [ERROR] Failed to install core dependencies
    pause
    exit /b 1
)

REM Trading and ML dependencies  
echo [INSTALL] Installing trading and ML dependencies...
python -m pip install --upgrade ccxt scikit-learn xgboost textblob
if errorlevel 1 (
    echo [WARNING] Some trading/ML dependencies may have failed
    echo System will still function with reduced capabilities
)

REM Additional utility dependencies
echo [INSTALL] Installing utility dependencies...
python -m pip install --upgrade aiohttp tenacity pydantic pydantic-settings schedule
if errorlevel 1 (
    echo [WARNING] Some utility dependencies may have failed
)

REM Optional performance dependencies
echo [INSTALL] Installing optional performance dependencies...
python -m pip install --upgrade numba psutil setproctitle prometheus-client
if errorlevel 1 (
    echo [WARNING] Some performance dependencies may have failed
    echo Core functionality will still work
)

REM Create necessary directories
echo [SETUP] Creating necessary directories...
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "config" mkdir config
echo [OK] Directories created

REM Set up configuration files
echo [CONFIG] Setting up configuration files...
if not exist "config.json" (
    echo [CREATE] Creating default config.json...
    echo { > config.json
    echo   "exchange": "kraken", >> config.json
    echo   "risk_management": { >> config.json
    echo     "max_position_size": 0.1, >> config.json
    echo     "stop_loss": 0.05, >> config.json
    echo     "take_profit": 0.15 >> config.json
    echo   }, >> config.json
    echo   "ml_settings": { >> config.json
    echo     "model_retrain_hours": 24, >> config.json
    echo     "confidence_threshold": 0.7 >> config.json
    echo   } >> config.json
    echo } >> config.json
    echo [OK] Default configuration created
)

REM Test installation
echo [TEST] Testing installation...
python -c "
try:
    import streamlit
    import pandas as pd
    import numpy as np
    import plotly
    import ccxt
    print('[OK] All critical dependencies imported successfully')
    
    # Test basic functionality
    df = pd.DataFrame({'test': [1, 2, 3]})
    arr = np.array([1, 2, 3])
    exchanges = ccxt.exchanges
    
    print(f'[OK] Pandas working - test dataframe shape: {df.shape}')
    print(f'[OK] NumPy working - test array: {arr}')
    print(f'[OK] CCXT working - {len(exchanges)} exchanges available')
    print('[OK] Installation test passed!')
    
except ImportError as e:
    print(f'[ERROR] Import failed: {e}')
    print('[FIX] Please run install_dependencies.bat')
    exit(1)
except Exception as e:
    print(f'[WARNING] Test error: {e}')
    print('Installation may be incomplete but basic functionality should work')
"

if errorlevel 1 (
    echo [ERROR] Installation test failed
    echo Please check the error messages above
    pause
    exit /b 1
)

REM Create quick health check script
echo [CREATE] Creating system health check script...
echo @echo off > quick_health_check.bat
echo title CryptoSmartTrader V2 - Health Check >> quick_health_check.bat
echo echo [HEALTH] Running system health check... >> quick_health_check.bat
echo python -c "from core.health_monitor import HealthMonitor; hm = HealthMonitor(); print(hm.get_system_status())" >> quick_health_check.bat
echo pause >> quick_health_check.bat

echo.
echo ================================================
echo   Windows Environment Setup Complete!
echo ================================================
echo.
echo [SUCCESS] Environment setup completed successfully
echo [INFO] You can now run start_cryptotrader.bat to launch the system
echo [INFO] Use quick_health_check.bat to verify system health anytime
echo.
echo [NEXT STEPS]
echo 1. Run: start_cryptotrader.bat
echo 2. Open browser to: http://localhost:5000
echo 3. Configure your trading settings
echo.
pause