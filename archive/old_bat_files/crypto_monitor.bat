@echo off
REM CryptoSmartTrader V2 - Complete Background Monitoring Suite
REM Starts all background services for comprehensive crypto analysis

title CryptoSmartTrader V2 - Complete Monitoring Suite

echo ========================================================
echo CryptoSmartTrader V2 - Complete Background Monitor
echo ========================================================
echo.

REM Get script directory
set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ and add it to PATH
    pause
    exit /b 1
)

echo Starting comprehensive crypto monitoring system...
echo.

REM Create logs directory
if not exist "%PROJECT_DIR%\logs" mkdir "%PROJECT_DIR%\logs"

REM Start main dashboard in background
echo [1/4] Starting main dashboard...
start "CryptoSmartTrader Dashboard" /min cmd /c "cd /d "%PROJECT_DIR%" && python -m streamlit run app.py --server.port 5000 > logs\dashboard.log 2>&1"

REM Wait a moment for dashboard to initialize
timeout /t 3 /nobreak >nul

REM Start ML analysis service
echo [2/4] Starting ML analysis service...
start "ML Analysis Service" /min cmd /c "cd /d "%SCRIPT_DIR%" && start_ml_analysis.bat"

REM Wait a moment
timeout /t 2 /nobreak >nul

REM Start social media scraper
echo [3/4] Starting social media scraper...
start "Social Scraper Service" /min cmd /c "cd /d "%SCRIPT_DIR%" && start_social_scraper.bat"

REM Wait a moment
timeout /t 2 /nobreak >nul

REM Start system monitoring
echo [4/4] Starting system monitoring...
start "System Monitor" /min cmd /c "cd /d "%PROJECT_DIR%" && python scripts\system_monitor.py > logs\system_monitor.log 2>&1"

echo.
echo ========================================================
echo  All Services Started Successfully!
echo ========================================================
echo.
echo Running Services:
echo  ✓ Main Dashboard         - http://localhost:5000
echo  ✓ ML Analysis Service    - Background processing
echo  ✓ Social Media Scraper   - Reddit & Twitter monitoring  
echo  ✓ System Monitor         - Health & performance tracking
echo.
echo Log Files:
echo  • Dashboard: %PROJECT_DIR%\logs\dashboard.log
echo  • ML Service: %PROJECT_DIR%\logs\ml_service.log
echo  • Social Scraper: %PROJECT_DIR%\logs\social_scraper.log
echo  • System Monitor: %PROJECT_DIR%\logs\system_monitor.log
echo.
echo Data Directories:
echo  • Analysis Results: %PROJECT_DIR%\data\analysis\
echo  • Social Data: %PROJECT_DIR%\data\social\
echo  • ML Models: %PROJECT_DIR%\models\
echo.
echo ========================================================
echo  Management Commands:
echo ========================================================
echo.
echo  [1] View Running Services
echo  [2] Stop All Services  
echo  [3] Restart ML Analysis
echo  [4] Restart Social Scraper
echo  [5] Open Dashboard
echo  [6] View System Status
echo  [0] Exit
echo.

:menu
set /p choice="Select option (0-6): "

if "%choice%"=="1" goto view_services
if "%choice%"=="2" goto stop_services
if "%choice%"=="3" goto restart_ml
if "%choice%"=="4" goto restart_social
if "%choice%"=="5" goto open_dashboard
if "%choice%"=="6" goto system_status
if "%choice%"=="0" goto exit_script

echo Invalid choice. Please select 0-6.
goto menu

:view_services
echo.
echo Current Running Services:
tasklist /FI "WINDOWTITLE eq CryptoSmartTrader*" /FO TABLE
echo.
goto menu

:stop_services
echo.
echo Stopping all CryptoSmartTrader services...
taskkill /FI "WINDOWTITLE eq CryptoSmartTrader*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq ML Analysis*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Social Scraper*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq System Monitor*" /F >nul 2>&1
echo All services stopped.
echo.
goto menu

:restart_ml
echo.
echo Restarting ML Analysis Service...
taskkill /FI "WINDOWTITLE eq ML Analysis*" /F >nul 2>&1
timeout /t 2 /nobreak >nul
start "ML Analysis Service" /min cmd /c "cd /d "%SCRIPT_DIR%" && start_ml_analysis.bat"
echo ML Analysis Service restarted.
echo.
goto menu

:restart_social
echo.
echo Restarting Social Scraper Service...
taskkill /FI "WINDOWTITLE eq Social Scraper*" /F >nul 2>&1
timeout /t 2 /nobreak >nul
start "Social Scraper Service" /min cmd /c "cd /d "%SCRIPT_DIR%" && start_social_scraper.bat"
echo Social Scraper Service restarted.
echo.
goto menu

:open_dashboard
echo.
echo Opening CryptoSmartTrader Dashboard...
start http://localhost:5000
echo.
goto menu

:system_status
echo.
echo System Status Check:
echo ==================
echo.

REM Check if main dashboard is running
netstat -an | find "5000" >nul
if %errorlevel% equ 0 (
    echo ✓ Main Dashboard: RUNNING (Port 5000)
) else (
    echo ✗ Main Dashboard: NOT RUNNING
)

REM Check log files
if exist "%PROJECT_DIR%\logs\ml_service.log" (
    echo ✓ ML Service: LOG ACTIVE
) else (
    echo ✗ ML Service: NO LOG FOUND
)

if exist "%PROJECT_DIR%\logs\social_scraper.log" (
    echo ✓ Social Scraper: LOG ACTIVE  
) else (
    echo ✗ Social Scraper: NO LOG FOUND
)

REM Check data directories
if exist "%PROJECT_DIR%\data\analysis\" (
    for /f %%i in ('dir /b "%PROJECT_DIR%\data\analysis\" ^| find /c /v ""') do set analysis_files=%%i
    echo ✓ Analysis Data: !analysis_files! files
) else (
    echo ✗ Analysis Data: NO DATA DIRECTORY
)

if exist "%PROJECT_DIR%\data\social\" (
    for /f %%i in ('dir /b "%PROJECT_DIR%\data\social\" ^| find /c /v ""') do set social_files=%%i
    echo ✓ Social Data: !social_files! files
) else (
    echo ✗ Social Data: NO DATA DIRECTORY
)

echo.
goto menu

:exit_script
echo.
echo Do you want to stop all services before exiting? (y/n)
set /p stop_choice=""
if /i "%stop_choice%"=="y" (
    echo Stopping all services...
    taskkill /FI "WINDOWTITLE eq CryptoSmartTrader*" /F >nul 2>&1
    taskkill /FI "WINDOWTITLE eq ML Analysis*" /F >nul 2>&1
    taskkill /FI "WINDOWTITLE eq Social Scraper*" /F >nul 2>&1
    taskkill /FI "WINDOWTITLE eq System Monitor*" /F >nul 2>&1
    echo All services stopped.
)

echo.
echo CryptoSmartTrader V2 Background Monitor - Session ended
pause