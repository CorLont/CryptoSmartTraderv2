@echo off
:: CryptoSmartTrader V2 - Stop All Services
:: Gracefully stop alle running services

title CryptoSmartTrader V2 - Stopping Services

echo ================================================
echo CryptoSmartTrader V2 - Stopping All Services
echo ================================================

echo Stopping Python processes...
taskkill /f /im python.exe 2>nul
if %errorlevel% equ 0 (
    echo ✓ Python processes stopped
) else (
    echo ⚠️ No Python processes found
)

echo Stopping Streamlit processes...
taskkill /f /im streamlit.exe 2>nul
if %errorlevel% equ 0 (
    echo ✓ Streamlit processes stopped
) else (
    echo ⚠️ No Streamlit processes found
)

echo Stopping Uvicorn processes...
taskkill /f /im uvicorn.exe 2>nul
if %errorlevel% equ 0 (
    echo ✓ Uvicorn processes stopped
) else (
    echo ⚠️ No Uvicorn processes found
)

echo.
echo ================================================
echo All CryptoSmartTrader V2 services stopped
echo ================================================

pause