@echo off
REM CryptoSmartTrader V2 - ML Analysis Background Service
REM Starts machine learning analysis and social media scraping

title CryptoSmartTrader V2 - ML Analysis Service

echo ============================================
echo CryptoSmartTrader V2 - ML Analysis Service
echo ============================================
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

REM Check if virtual environment exists
if not exist "%PROJECT_DIR%\venv\Scripts\python.exe" (
    echo Creating Python virtual environment...
    python -m venv "%PROJECT_DIR%\venv"
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call "%PROJECT_DIR%\venv\Scripts\activate.bat"

REM Set Python path
set PYTHONPATH=%PROJECT_DIR%;%PYTHONPATH%

REM Load environment variables if .env exists
if exist "%PROJECT_DIR%\.env" (
    echo Loading environment variables...
    for /f "tokens=1,2 delims==" %%a in ('type "%PROJECT_DIR%\.env" ^| findstr /v "^#"') do (
        set %%a=%%b
    )
)

REM Check for required API keys
echo Checking API configurations...
if "%OPENAI_API_KEY%"=="" (
    echo WARNING: OPENAI_API_KEY not found in environment
    echo Advanced sentiment analysis will use TextBlob fallback
)

if "%REDDIT_CLIENT_ID%"=="" (
    echo WARNING: Reddit API credentials not configured
    echo Reddit scraping will use synthetic data fallback
)

if "%TWITTER_BEARER_TOKEN%"=="" (
    echo WARNING: Twitter API credentials not configured  
    echo Twitter scraping will use synthetic data fallback
)

REM Create logs directory
if not exist "%PROJECT_DIR%\logs" mkdir "%PROJECT_DIR%\logs"

REM Start ML analysis service
echo.
echo Starting ML Analysis Service...
echo Service will run in background - check logs for status
echo Log files: %PROJECT_DIR%\logs\
echo.
echo Press Ctrl+C to stop the service
echo.

REM Start the ML analysis service with proper error handling
python "%PROJECT_DIR%\scripts\ml_background_service.py" 2>&1 | tee "%PROJECT_DIR%\logs\ml_service.log"

if %errorlevel% neq 0 (
    echo.
    echo ERROR: ML Analysis Service failed to start
    echo Check the log file for details: %PROJECT_DIR%\logs\ml_service.log
    pause
    exit /b 1
)

echo.
echo ML Analysis Service stopped
pause