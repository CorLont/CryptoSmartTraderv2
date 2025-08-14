@echo off
:: CryptoSmartTrader V2 - Complete System Startup
:: Start alle services in afzonderlijke vensters

echo ================================================
echo CryptoSmartTrader V2 - Complete System Startup
echo ================================================

:: Check if virtual environment exists
if not exist .venv (
    echo ERROR: Virtual environment niet gevonden.
    echo Run setup_env.bat eerst om de environment op te zetten.
    pause
    exit /b 1
)

echo Starting alle CryptoSmartTrader V2 services...
echo.

:: Start API Server
echo Starting API Server (port 8001)...
start "CryptoSmartTrader API" cmd /k start_api.bat

:: Wait a bit for API to start
timeout /t 3 /nobreak >nul

:: Start UI Dashboard  
echo Starting UI Dashboard (port 5000)...
start "CryptoSmartTrader UI" cmd /k start_ui.bat

:: Wait a bit for UI to start
timeout /t 3 /nobreak >nul

:: Start Background Workers
echo Starting Background Workers...
start "CryptoSmartTrader Workers" cmd /k start_workers.bat

echo.
echo ================================================
echo Alle services worden gestart!
echo ================================================
echo.
echo Services:
echo - API Server:     http://localhost:8001
echo - UI Dashboard:   http://localhost:5000  
echo - Workers:        Background processes
echo.
echo Wacht even tot alle services volledig geladen zijn...
echo Check de individuele vensters voor status updates.
echo.

pause