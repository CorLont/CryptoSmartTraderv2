@echo off
:: CryptoSmartTrader V2 - Environment Setup
:: Eenmalige setup voor Windows workstation

echo ================================================
echo CryptoSmartTrader V2 - Environment Setup
echo ================================================

:: Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python niet gevonden. Installeer Python 3.11 of hoger.
    pause
    exit /b 1
)

echo ✓ Python detectie succesvol

:: Create virtual environment
echo Creating virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ERROR: Kan virtual environment niet maken
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate
if errorlevel 1 (
    echo ERROR: Kan virtual environment niet activeren
    pause
    exit /b 1
)

echo ✓ Virtual environment geactiveerd

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install dependencies
echo Installing dependencies...
pip install -e .
if errorlevel 1 (
    echo ERROR: Dependency installatie gefaald
    pause
    exit /b 1
)

echo ✓ Dependencies geïnstalleerd

:: Create .env from template if not exists
if not exist .env (
    if exist .env.example (
        echo Creating .env from template...
        copy .env.example .env
        echo ⚠️  BELANGRIJK: Configureer je API keys in .env bestand
    )
)

echo ================================================
echo Setup voltooid!
echo ================================================
echo Volgende stappen:
echo 1. Configureer API keys in .env bestand
echo 2. Start services met start_all.bat
echo ================================================
pause