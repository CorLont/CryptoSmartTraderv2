@echo off
:: CryptoSmartTrader V2 - UI Dashboard
:: Start Streamlit dashboard interface

title CryptoSmartTrader V2 - UI Dashboard

echo ================================================
echo CryptoSmartTrader V2 - Starting UI Dashboard
echo ================================================

:: Set environment variables
set PYTHONPATH=%CD%

:: Activate virtual environment
call .venv\Scripts\activate
if errorlevel 1 (
    echo ERROR: Virtual environment niet gevonden. Run setup_env.bat eerst.
    pause
    exit /b 1
)

echo ✓ Virtual environment geactiveerd
echo Starting UI dashboard op port 5000...

:: Start Streamlit dashboard
streamlit run enhanced_alpha_motor_dashboard.py --server.headless true --server.port 5000 --server.address 0.0.0.0

pause