@echo off
echo ==========================================
echo CryptoSmartTrader V2 - Installation
echo ==========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found! Please install Python 3.11 or later
    echo Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Install dependencies
echo 📦 Installing dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

echo ✅ Dependencies installed
echo.

REM Configure antivirus exceptions
echo 🛡️ Configuring antivirus exceptions...
python scripts/windows_deployment.py --antivirus
echo.

REM Configure firewall
echo 🔥 Configuring firewall rules...
python scripts/windows_deployment.py --firewall
echo.

REM Create directories
echo 📁 Creating directories...
mkdir logs 2>nul
mkdir cache 2>nul
mkdir exports 2>nul
mkdir model_backup 2>nul
mkdir backups 2>nul

echo ✅ Installation completed!
echo.
echo 🚀 To start the system, run: start_dashboard.bat
pause
