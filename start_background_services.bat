@echo off
REM CryptoSmartTrader V2 - Background Services Starter
REM Start alle ML, scraping en whale detection services op de achtergrond

echo ================================
echo Starting Background Services
echo ================================

REM Start real-time pipeline in background
echo [1/4] Starting Real-time Alpha Pipeline...
start /B python -c "
import sys
sys.path.append('.')
from containers import ApplicationContainer
container = ApplicationContainer()
pipeline = container.real_time_pipeline()
pipeline.start_continuous_monitoring()
print('Real-time pipeline started')
while True:
    import time
    time.sleep(60)
    pipeline.run_pipeline()
"

REM Start ML batch inference in background  
echo [2/4] Starting ML Batch Inference...
start /B python -c "
import sys
sys.path.append('.')
from containers import ApplicationContainer
import time
container = ApplicationContainer()
ml_system = container.multi_horizon_ml()
print('ML system started')
while True:
    try:
        opportunities = ml_system.get_alpha_opportunities(min_confidence=0.80)
        print(f'Found {len(opportunities)} alpha opportunities')
    except Exception as e:
        print(f'ML inference error: {e}')
    time.sleep(300)  # Run every 5 minutes
"

REM Start sentiment scraping in background
echo [3/4] Starting Sentiment Scraping...
start /B python -c "
import sys
sys.path.append('.')
from agents.sentiment_agent import SentimentAgent
from containers import ApplicationContainer
import time
container = ApplicationContainer()
sentiment_agent = SentimentAgent(container)
print('Sentiment scraping started')
while True:
    try:
        sentiment_agent.run_analysis()
        print('Sentiment analysis completed')
    except Exception as e:
        print(f'Sentiment error: {e}')
    time.sleep(600)  # Run every 10 minutes
"

REM Start whale detection in background
echo [4/4] Starting Whale Detection...
start /B python -c "
import sys
sys.path.append('.')
from agents.whale_detector_agent import WhaleDetectorAgent
from containers import ApplicationContainer
import time
container = ApplicationContainer()
whale_agent = WhaleDetectorAgent(container)
print('Whale detection started')
while True:
    try:
        whale_agent.run_analysis()
        print('Whale detection completed')
    except Exception as e:
        print(f'Whale detection error: {e}')
    time.sleep(180)  # Run every 3 minutes
"

echo.
echo ================================
echo All Background Services Started!
echo ================================
echo.
echo Services running:
echo  ✓ Real-time Alpha Pipeline (continuous)
echo  ✓ ML Batch Inference (every 5 min)
echo  ✓ Sentiment Scraping (every 10 min)  
echo  ✓ Whale Detection (every 3 min)
echo.
echo These services will continue running in background.
echo To stop: Close this window or press Ctrl+C
echo.

REM Keep window open to show service status
:loop
timeout /t 30 /nobreak >nul
echo [%time%] Background services running...
goto loop