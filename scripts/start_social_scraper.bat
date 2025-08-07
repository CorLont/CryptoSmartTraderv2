@echo off
REM CryptoSmartTrader V2 - Social Media Scraping Service
REM Handles Reddit, Twitter, and other social sentiment data collection

title CryptoSmartTrader V2 - Social Scraper Service

echo ===============================================
echo CryptoSmartTrader V2 - Social Scraper Service
echo ===============================================
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

REM Activate virtual environment
echo Activating virtual environment...
if exist "%PROJECT_DIR%\venv\Scripts\python.exe" (
    call "%PROJECT_DIR%\venv\Scripts\activate.bat"
    set PYTHON_CMD=%PROJECT_DIR%\venv\Scripts\python.exe
) else (
    echo WARNING: Virtual environment not found, using system Python
    set PYTHON_CMD=python
)

REM Set Python path
set PYTHONPATH=%PROJECT_DIR%;%PYTHONPATH%

REM Load environment variables
if exist "%PROJECT_DIR%\.env" (
    echo Loading environment variables...
    for /f "tokens=1,2 delims==" %%a in ('type "%PROJECT_DIR%\.env" ^| findstr /v "^#"') do (
        set %%a=%%b
    )
)

REM API Key validation with user prompts
echo Checking social media API credentials...

if "%REDDIT_CLIENT_ID%"=="" (
    echo.
    echo Reddit API Setup Required:
    echo 1. Go to https://www.reddit.com/prefs/apps
    echo 2. Create a new application
    echo 3. Copy Client ID and Secret
    echo.
    set /p REDDIT_CLIENT_ID=Enter Reddit Client ID: 
    set /p REDDIT_CLIENT_SECRET=Enter Reddit Client Secret: 
    
    REM Save to .env file
    echo REDDIT_CLIENT_ID=%REDDIT_CLIENT_ID% >> "%PROJECT_DIR%\.env"
    echo REDDIT_CLIENT_SECRET=%REDDIT_CLIENT_SECRET% >> "%PROJECT_DIR%\.env"
)

if "%TWITTER_BEARER_TOKEN%"=="" (
    echo.
    echo Twitter API Setup Required:
    echo 1. Go to https://developer.twitter.com/
    echo 2. Create a new project and app
    echo 3. Copy Bearer Token from Keys and Tokens
    echo.
    set /p TWITTER_BEARER_TOKEN=Enter Twitter Bearer Token: 
    
    REM Save to .env file
    echo TWITTER_BEARER_TOKEN=%TWITTER_BEARER_TOKEN% >> "%PROJECT_DIR%\.env"
)

REM Create logs directory
if not exist "%PROJECT_DIR%\logs" mkdir "%PROJECT_DIR%\logs"
if not exist "%PROJECT_DIR%\data\social" mkdir "%PROJECT_DIR%\data\social"

REM Display service configuration
echo.
echo Service Configuration:
echo - Reddit API: %REDDIT_CLIENT_ID:~0,8%...
echo - Twitter API: %TWITTER_BEARER_TOKEN:~0,8%...
echo - Data directory: %PROJECT_DIR%\data\social
echo - Log directory: %PROJECT_DIR%\logs
echo.

REM Start social scraping service
echo Starting Social Media Scraper Service...
echo Service will run continuously in background
echo.
echo Monitoring:
echo - Reddit: r/CryptoCurrency, r/Bitcoin, r/ethereum
echo - Twitter: #Bitcoin, #Ethereum, #Crypto hashtags
echo - Update interval: 5 minutes
echo.
echo Press Ctrl+C to stop the service
echo.

REM Start the service with error handling and logging
%PYTHON_CMD% "%PROJECT_DIR%\scripts\social_scraper_service.py" 2>&1 | tee "%PROJECT_DIR%\logs\social_scraper.log"

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Social Scraper Service failed to start
    echo Check the log file: %PROJECT_DIR%\logs\social_scraper.log
    echo.
    echo Common issues:
    echo - Invalid API credentials
    echo - Network connectivity problems
    echo - Rate limiting from social platforms
    pause
    exit /b 1
)

echo.
echo Social Scraper Service stopped
pause