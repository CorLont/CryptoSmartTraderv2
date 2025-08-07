@echo off
REM CryptoSmartTrader V2 - Quick Start (All-in-One)
REM Complete setup: dependencies + background services + dashboard

title CryptoSmartTrader V2 - Quick Start

echo.
echo  ╔══════════════════════════════════════════════════════════╗
echo  ║               CryptoSmartTrader V2                       ║
echo  ║            Quick Start - All-in-One                     ║
echo  ║                                                          ║
echo  ║  • Installs all dependencies                             ║
echo  ║  • Starts ML/AI background services                      ║  
echo  ║  • Opens dashboard for full analysis                     ║
echo  ╚══════════════════════════════════════════════════════════╝
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found. Please install Python 3.11+ first.
    echo    Download from: https://python.org
    pause
    exit /b 1
)

echo ✓ Python found: 
python --version
echo.

echo 📦 Installing dependencies...
pip install --quiet --upgrade streamlit pandas numpy plotly scikit-learn ccxt xgboost textblob trafilatura aiohttp pydantic dependency-injector prometheus-client psutil schedule tenacity python-json-logger pydantic-settings hvac

echo.
echo 🔧 Setting up environment...
if not exist .env (
    echo ENVIRONMENT=production > .env
    echo LOG_LEVEL=INFO >> .env
    echo ENABLE_REAL_TIME=true >> .env
)

if not exist logs mkdir logs
if not exist cache mkdir cache
if not exist models mkdir models

echo.
echo 🚀 Starting background services...

REM Start background services in separate window
start "CryptoTrader Background Services" cmd /c start_background_services.bat

REM Wait a moment for services to initialize
timeout /t 3 /nobreak >nul

echo.
echo 🌐 Starting dashboard...
echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║                    SYSTEM READY!                        ║
echo ╠══════════════════════════════════════════════════════════╣
echo ║                                                          ║
echo ║  Dashboard URL: http://localhost:5000                    ║
echo ║                                                          ║
echo ║  Available Features:                                     ║
echo ║  • 1457+ trading pairs analysis                         ║
echo ║  • Real-time opportunity detection                       ║
echo ║  • Multi-horizon ML predictions                         ║
echo ║  • Sentiment analysis & whale detection                 ║
echo ║  • Advanced AI/ML differentiators                       ║
echo ║                                                          ║
echo ║  For Alpha Seeking (500%+ returns):                     ║
echo ║  1. Use Comprehensive Market Dashboard                   ║
echo ║  2. Filter confidence ≥80%                              ║
echo ║  3. Sort by 30-day expected returns                     ║
echo ║  4. Check ML/AI Differentiators for insights           ║
echo ║                                                          ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

REM Launch dashboard (this will open browser automatically)
streamlit run app.py --server.port 5000

REM If dashboard stops, provide restart instructions
echo.
echo Dashboard stopped. To restart:
echo   - Run: quick_start.bat
echo   - Or: streamlit run app.py --server.port 5000
echo.
pause