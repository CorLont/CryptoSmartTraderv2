@echo off
REM CryptoSmartTrader V2 - Windows Installation Script
echo ===================================
echo CryptoSmartTrader V2 Installation
echo ===================================

REM Check Python installation
python --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found
    echo Install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python version:
python --version

REM Create virtual environment
echo.
echo Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
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
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo.
echo Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

REM GPU check
echo.
echo Checking GPU support...
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Devices:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"

REM Create directories
echo.
echo Creating directories...
mkdir data\raw 2>nul
mkdir data\processed 2>nul
mkdir exports\production 2>nul
mkdir logs\daily 2>nul
mkdir models\baseline 2>nul

REM Environment setup check
echo.
echo Checking environment setup...
if not exist .env (
    echo WARNING: .env file not found
    echo Create .env with your API keys:
    echo KRAKEN_API_KEY=your_kraken_api_key
    echo KRAKEN_SECRET=your_kraken_secret
    echo OPENAI_API_KEY=your_openai_api_key
)

echo.
echo ===================================
echo Installation Complete!
echo ===================================
echo.
echo Next steps:
echo 1. Configure .env with your API keys
echo 2. Run: run.bat
echo.
pause