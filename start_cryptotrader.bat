@echo off
title CryptoSmartTrader V2 - Starting System
color 0A
echo ==========================================
echo  CryptoSmartTrader V2 - System Startup
echo ==========================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please run setup_windows_environment.bat first
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
)

REM Check critical dependencies
echo [INFO] Checking critical dependencies...
python -c "import streamlit, pandas, numpy, plotly, ccxt" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Critical dependencies missing
    echo Running automatic dependency installation...
    call install_dependencies.bat
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed
        echo Please check your internet connection and try again
        pause
        exit /b 1
    )
)

REM Set environment variables
set PYTHONPATH=%CD%
set ENVIRONMENT=production
set STREAMLIT_SERVER_HEADLESS=true
set STREAMLIT_SERVER_ADDRESS=0.0.0.0
set STREAMLIT_SERVER_PORT=5000

echo [OK] Environment configured
echo.

REM Create logs directory with today's date
for /f "tokens=2-4 delims=/ " %%a in ('date /t') do (set mydate=%%c-%%a-%%b)
if not exist "logs" mkdir logs
if not exist "logs\%mydate%" mkdir "logs\%mydate%"

REM Quick system health check
echo [INFO] Running quick system health check...
python -c "
try:
    import sys, os
    sys.path.append('.')
    from config.daily_logging_config import setup_daily_logging
    from core.config_manager import ConfigManager
    print('[OK] Core systems operational')
    setup_daily_logging()
    print('[OK] Daily logging initialized')
    config = ConfigManager()
    print('[OK] Configuration loaded')
    print('[OK] System health check passed')
except Exception as e:
    print(f'[WARNING] System health check warning: {e}')
    print('System will continue but may have reduced functionality')
" 2>nul

echo.
echo [START] Starting CryptoSmartTrader system...
echo [INFO] Web interface will be available at: http://localhost:5000
echo [INFO] Logs location: logs\%mydate%\
echo [INFO] Press Ctrl+C to stop the system
echo.

REM Start the main Streamlit application with enhanced error handling
python -c "
import sys
import os
import traceback
import subprocess

try:
    print('[INIT] Initializing Streamlit application...')
    
    # Add current directory to Python path
    sys.path.insert(0, os.getcwd())
    
    # Start Streamlit with proper configuration
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'app.py',
        '--server.port', '5000',
        '--server.headless', 'true',
        '--server.address', '0.0.0.0',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false'
    ]
    
    print('[START] Launching application...')
    subprocess.run(cmd)
    
except KeyboardInterrupt:
    print('\n[STOP] Application stopped by user')
except FileNotFoundError as e:
    print(f'\n[ERROR] File not found: {e}')
    print('[FIX] Please ensure app.py exists in the current directory')
except ImportError as e:
    print(f'\n[ERROR] Import error: {e}')
    print('[FIX] Please run install_dependencies.bat to install missing packages')
except Exception as e:
    print(f'\n[ERROR] Application error: {e}')
    print('[DEBUG] Error details:')
    traceback.print_exc()
    print('\n[HELP] Troubleshooting:')
    print('1. Check if port 5000 is available')
    print('2. Verify all dependencies are installed')
    print('3. Check logs directory permissions')
    print('4. Try running as administrator')
"

echo.
echo [EXIT] Application has stopped.
echo Press any key to exit...
pause >nul