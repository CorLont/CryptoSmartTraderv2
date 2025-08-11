@echo off
echo ===================================================
echo CryptoSmartTrader V2 - Starting Main Dashboard
echo ===================================================

REM Activate virtual environment
echo [1/5] Activating virtual environment...
call .venv\Scripts\activate 2>nul
if errorlevel 1 (
    echo ⚠️ Virtual environment not found. Run 1_install_all_dependencies.bat first
    pause
    exit /b 1
)

REM Check if main app exists
echo.
echo [2/5] Checking application files...
if not exist app_fixed_all_issues.py (
    echo ❌ ERROR: app_fixed_all_issues.py not found
    echo Make sure you're in the correct directory
    pause
    exit /b 1
)
echo ✓ Main application found

REM Configure system for optimal performance
echo.
echo [3/5] Configuring system optimizations...
powershell -Command "powercfg -setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c" 2>nul || echo "⚠️ High performance power plan failed (run as admin for full optimization)"

REM Set environment variables
echo.
echo [4/5] Setting environment variables...
set PYTHONPATH=%CD%
set CRYPTOSMARTTRADER_ENV=production
set STREAMLIT_SERVER_PORT=5000
set STREAMLIT_SERVER_ADDRESS=0.0.0.0

REM Pre-flight checks
echo.
echo [5/5] Running pre-flight checks...
python -c "import streamlit, pandas, plotly, ccxt, openai" 2>nul && echo "✓ Core dependencies available" || echo "⚠️ Some dependencies missing"

REM Check API keys
if exist .env (
    findstr /C:"KRAKEN_API_KEY" .env >nul && echo "✓ Kraken API key configured" || echo "⚠️ Kraken API key not found in .env"
    findstr /C:"OPENAI_API_KEY" .env >nul && echo "✓ OpenAI API key configured" || echo "⚠️ OpenAI API key not found in .env"
) else (
    echo "⚠️ .env file not found - using environment variables"
)

REM Check background services
echo.
echo Checking background services...
netstat -an | findstr :8090 >nul && echo "✓ Metrics server running" || echo "⚠️ Metrics server not running (optional)"

echo.
echo ===================================================
echo 🚀 STARTING CRYPTOSMARTTRADER V2 DASHBOARD
echo ===================================================
echo.
echo Dashboard Features:
echo   • Real-time analysis of 471+ cryptocurrencies
echo   • ML-powered predictions with 80%% confidence gate
echo   • Sentiment analysis and whale detection
echo   • Technical analysis and risk assessment
echo   • Dutch language support
echo.
echo Starting dashboard on http://localhost:5000
echo Press Ctrl+C to stop the dashboard
echo.

REM Create run log
set RUN_ID=%DATE:~6,4%%DATE:~3,2%%DATE:~0,2%_%TIME:~0,2%%TIME:~3,2%
set RUN_ID=%RUN_ID: =0%
mkdir logs\dashboard 2>nul
set LOG_FILE=logs\dashboard\dashboard_%RUN_ID%.log

echo Dashboard starting at %DATE% %TIME% > %LOG_FILE%

REM Start the main dashboard
streamlit run app_fixed_all_issues.py --server.port 5000 --server.address 0.0.0.0 --server.headless true

echo.
echo ===================================================
echo Dashboard stopped at %DATE% %TIME%
echo ===================================================
echo.
echo Log file: %LOG_FILE%
echo.
pause
