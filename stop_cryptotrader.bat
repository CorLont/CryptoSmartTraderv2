@echo off
title CryptoSmartTrader V2 - Shutdown
color 0C
echo ==========================================
echo  CryptoSmartTrader V2 - System Shutdown
echo ==========================================

echo [STOP] Stopping CryptoSmartTrader services...

REM Kill any running Streamlit processes
echo [KILL] Stopping Streamlit processes...
taskkill /f /im python.exe /fi "WINDOWTITLE eq*streamlit*" >nul 2>&1
taskkill /f /im python.exe /fi "COMMANDLINE eq*streamlit*" >nul 2>&1
taskkill /f /im streamlit.exe >nul 2>&1

REM More aggressive cleanup - kill python processes running our app
for /f "tokens=2" %%i in ('tasklist /fi "imagename eq python.exe" /fo csv ^| find "app.py"') do (
    taskkill /f /pid %%i >nul 2>&1
)

REM Kill processes using port 5000
echo [KILL] Freeing port 5000...
for /f "tokens=5" %%a in ('netstat -aon ^| find ":5000" ^| find "LISTENING"') do (
    taskkill /f /pid %%a >nul 2>&1
)

echo [OK] System shutdown complete
echo [INFO] All CryptoSmartTrader processes have been stopped
echo [INFO] Port 5000 has been freed
echo.
pause