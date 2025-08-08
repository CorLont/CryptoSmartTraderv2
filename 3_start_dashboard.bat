@echo off
echo ==========================================
echo CryptoSmartTrader V2 - Starting Dashboard
echo ==========================================
echo.

REM Check if system is already running
netstat -an | find ":5000" >nul 2>&1
if %errorlevel% equ 0 (
    echo ⚠️ Dashboard already running on port 5000
    echo Opening browser...
    start http://localhost:5000
    pause
    exit /b 0
)

echo 🚀 Starting CryptoSmartTrader Dashboard...
echo.
echo 📊 Dashboard will be available at: http://localhost:5000
echo 📈 Test app will be available at: http://localhost:5001
echo.
echo Press Ctrl+C to stop the system
echo.

REM Start the dashboard
python -m streamlit run app_minimal.py --server.port 5000 --server.headless true

pause
