@echo off
echo Starting CryptoSmartTrader Distributed Multi-Process System...
echo.

REM Kill any existing services first
echo Stopping existing services...
taskkill /F /IM python.exe 2>nul
timeout /t 2 /nobreak >nul

REM Create necessary directories
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "data\market_data" mkdir data\market_data

REM Start distributed orchestrator
echo [MAIN] Starting Distributed Orchestrator...
start "Distributed Orchestrator" python -m core.distributed_orchestrator

echo.
echo Waiting for orchestrator to initialize agents...
timeout /t 5 /nobreak >nul

echo.
echo ✅ Distributed multi-process system started!
echo.
echo Architecture Features:
echo   • Process isolation per agent (8 independent agents)
echo   • Automatic restart with exponential backoff
echo   • Circuit breakers for failing agents  
echo   • Resource monitoring and memory limits
echo   • Health checks every 30 seconds
echo   • Agent-specific logs in logs/ directory
echo.
echo Agents running in isolated processes:
echo   1. Data Collector - Live market data from Kraken
echo   2. Sentiment Analyzer - Social/news sentiment analysis
echo   3. Technical Analyzer - RSI, MACD, Bollinger Bands
echo   4. ML Predictor - LSTM/Transformer predictions
echo   5. Whale Detector - Large transaction monitoring
echo   6. Risk Manager - Portfolio risk assessment
echo   7. Portfolio Optimizer - Asset allocation optimization
echo   8. Health Monitor - System health and alerts
echo.
echo Check orchestrator window for detailed agent status.
echo Individual agent logs available in logs/ directory.
echo.
pause