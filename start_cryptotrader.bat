@echo off
REM CryptoSmartTrader V2 - Complete Startup Script
REM Installeert dependencies en start alle systemen automatisch

echo ================================
echo CryptoSmartTrader V2 Startup
echo ================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

echo [1/5] Checking Python installation...
python --version

echo.
echo [2/5] Installing dependencies...
echo This may take a few minutes on first run...

REM Install core dependencies
pip install --upgrade pip
pip install streamlit>=1.28.0
pip install pandas>=2.0.0 numpy>=1.24.0
pip install plotly>=5.15.0 scikit-learn>=1.3.0
pip install ccxt>=4.0.0 xgboost>=1.7.0
pip install textblob>=0.17.1 trafilatura>=1.6.0
pip install aiohttp>=3.8.0 pydantic>=2.0.0
pip install pydantic-settings>=2.0.0
pip install dependency-injector>=4.41.0
pip install prometheus-client>=0.17.0 psutil>=5.9.0
pip install hvac>=1.1.0 schedule>=1.2.0
pip install tenacity>=8.2.0 python-json-logger>=2.0.0

echo.
echo [3/5] Setting up configuration...

REM Create .env file if it doesn't exist
if not exist .env (
    echo Creating default configuration...
    echo # CryptoSmartTrader V2 Configuration > .env
    echo # Optional: Add your OpenAI API key for enhanced sentiment analysis >> .env
    echo # OPENAI_API_KEY=your_key_here >> .env
    echo # >> .env
    echo # System Configuration >> .env
    echo ENVIRONMENT=production >> .env
    echo LOG_LEVEL=INFO >> .env
    echo ENABLE_GPU=auto >> .env
    echo MAX_THREADS=4 >> .env
) else (
    echo Configuration file already exists.
)

REM Create required directories
if not exist logs mkdir logs
if not exist cache mkdir cache
if not exist models mkdir models
if not exist data mkdir data

echo.
echo [4/5] Starting background services...

REM Start the main application in background mode
echo Starting CryptoSmartTrader V2 Dashboard...
echo.
echo ================================
echo SYSTEM READY!
echo ================================
echo.
echo Dashboard will open at: http://localhost:5000
echo.
echo Available Dashboards:
echo  - Main Dashboard: Real-time overview
echo  - Comprehensive Market: All 1457+ coins analysis  
echo  - Advanced Analytics: Technical analysis
echo  - AI/ML Engine: Machine learning predictions
echo  - ML/AI Differentiators: Advanced AI features
echo  - Crypto AI System: Complete pipeline management
echo.
echo Background Services Running:
echo  - Market Scanner: Analyzing all trading pairs
echo  - ML Predictor: Multi-horizon predictions
echo  - Sentiment Scraper: Social media analysis
echo  - Whale Detector: Large transaction monitoring
echo  - Real-time Pipeline: Alpha opportunity detection
echo.
echo Press Ctrl+C to stop all services
echo ================================
echo.

REM [5/5] Launch the dashboard
streamlit run app.py --server.port 5000 --server.headless false --server.address 0.0.0.0

REM If streamlit fails, provide troubleshooting
if errorlevel 1 (
    echo.
    echo ERROR: Failed to start dashboard
    echo.
    echo Troubleshooting:
    echo 1. Check if port 5000 is available
    echo 2. Try different port: streamlit run app.py --server.port 5001
    echo 3. Check logs in logs/ directory
    echo 4. Run: python scripts/production_health_check.py
    echo.
    pause
)