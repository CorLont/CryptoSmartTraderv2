@echo off
echo ===================================================
echo CryptoSmartTrader V2 - Complete Dependency Installation
echo ===================================================

REM Check Python installation
echo [1/8] Checking Python installation...
python --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)
echo ✓ Python version:
python --version

REM Create virtual environment if needed
echo.
echo [2/8] Setting up virtual environment...
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo.
echo [3/8] Upgrading pip...
python -m pip install --upgrade pip

REM Install core dependencies from pyproject.toml
echo.
echo [4/8] Installing core project dependencies...
if exist pyproject.toml (
    pip install -e .
) else (
    echo ⚠️ pyproject.toml not found, installing direct dependencies
)

REM Install critical integrations
echo.
echo [5/8] Installing critical integrations...
pip install openai>=1.0.0
pip install streamlit>=1.28.0
pip install ccxt>=4.0.0
pip install plotly>=5.15.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install scikit-learn>=1.3.0
pip install xgboost>=1.7.0
pip install pydantic>=2.0.0
pip install tenacity>=8.2.0
pip install aiohttp>=3.8.0
pip install textblob>=0.17.0
pip install trafilatura>=2.0.0
pip install transformers>=4.30.0
pip install torch>=2.0.0
pip install prometheus-client>=0.17.0
pip install psutil>=5.9.0
pip install dependency-injector>=4.41.0
pip install python-json-logger>=2.0.0

REM Install ML/AI performance packages
echo.
echo [6/8] Installing ML/AI performance packages...
pip install lightgbm>=4.0.0
pip install numba>=0.57.0
pip install cupy-cuda12x || pip install cupy-cuda11x || echo "⚠️ CuPy installation failed (requires CUDA 11 or 12)"

REM Install development and testing tools
echo.
echo [7/8] Installing development tools...
pip install pytest>=8.4.1
pip install pytest-asyncio>=1.1.0
pip install pytest-cov>=6.2.1
pip install pytest-mock>=3.14.1
pip install pytest-xdist>=3.8.0
pip install black>=23.7.0
pip install isort>=5.12.0
pip install flake8>=6.0.0

REM Create project structure and configure system
echo.
echo [8/8] Setting up project structure...
mkdir logs 2>nul
mkdir logs\daily 2>nul
mkdir data 2>nul
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir data\predictions 2>nul
mkdir exports 2>nul
mkdir exports\production 2>nul
mkdir models 2>nul
mkdir models\backup 2>nul
mkdir cache 2>nul
mkdir cache\temp 2>nul

REM Configure Windows optimizations
echo.
echo Configuring Windows optimizations...
powershell -Command "Add-MpPreference -ExclusionPath '%CD%'" 2>nul || echo "⚠️ Windows Defender exclusion failed (run as admin)"
powershell -Command "Add-MpPreference -ExclusionProcess 'python.exe'" 2>nul
powershell -Command "powercfg -setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c" 2>nul || echo "⚠️ High performance power plan failed"

REM GPU check
echo.
echo Checking GPU support...
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Devices:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>nul || echo "⚠️ PyTorch GPU check failed"

REM Environment setup check
echo.
echo Checking environment configuration...
if not exist .env (
    echo ⚠️ WARNING: .env file not found
    echo Create .env with your API keys:
    echo   KRAKEN_API_KEY=your_kraken_api_key
    echo   KRAKEN_SECRET=your_kraken_secret
    echo   OPENAI_API_KEY=your_openai_api_key
) else (
    echo ✓ .env file found
)

echo.
echo ===================================================
echo ✅ INSTALLATION COMPLETE!
echo ===================================================
echo.
echo Next steps:
echo   1. Configure .env with your API keys (if not done)
echo   2. Run validation: workstation_validation.bat
echo   3. Start services: 2_start_background_services.bat  
echo   4. Start dashboard: 3_start_dashboard.bat
echo.
echo For validation before startup:
echo   workstation_validation.bat
echo.
echo MANDATORY EXECUTION GATEWAY: All trading operations secured ✓
echo.
pause
