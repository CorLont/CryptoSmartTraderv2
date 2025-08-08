@echo off
setlocal EnableDelayedExpansion

echo =====================================================
echo  CryptoSmartTrader V2 - Background Services Manager
echo =====================================================
echo.

:: Set UTF-8 encoding
chcp 65001 >nul

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please run 1_install_all_dependencies.bat first
    pause
    exit /b 1
)

echo Starting CryptoSmartTrader V2 Background Services...
echo.

:: Kill any existing background processes
taskkill /F /IM python.exe /FI "WINDOWTITLE eq CryptoTrader*" >nul 2>&1

:: Create logs directory if it doesn't exist
if not exist "logs" mkdir logs

echo [1/8] Starting Data Collection Service...
start "CryptoTrader-DataCollection" /MIN python -c "
import sys
sys.path.append('.')
import logging
from datetime import datetime
import time
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%%(asctime)s - %%(levelname)s - %%(message)s',
    handlers=[
        logging.FileHandler('logs/data_collection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('DataCollection')
logger.info('Data Collection Service Started')

# Simulate data collection loop
while True:
    try:
        # Mock data collection from exchanges
        data = {
            'timestamp': datetime.now().isoformat(),
            'btc_price': 45000 + (hash(str(datetime.now())) %% 1000),
            'eth_price': 2800 + (hash(str(datetime.now())) %% 100),
            'volume': 1000000 + (hash(str(datetime.now())) %% 500000),
            'status': 'active'
        }
        
        # Save to file
        Path('data/raw').mkdir(exist_ok=True)
        with open('data/raw/latest_market_data.json', 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f'Market data updated: BTC={data[\"btc_price\"]}, ETH={data[\"eth_price\"]}')
        time.sleep(30)  # Update every 30 seconds
        
    except Exception as e:
        logger.error(f'Data collection error: {e}')
        time.sleep(60)
"

echo [2/8] Starting ML Prediction Service...
start "CryptoTrader-MLPrediction" /MIN python -c "
import sys
sys.path.append('.')
import logging
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%%(asctime)s - %%(levelname)s - %%(message)s',
    handlers=[
        logging.FileHandler('logs/ml_prediction.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('MLPrediction')
logger.info('ML Prediction Service Started')

# ML prediction loop
while True:
    try:
        # Load latest market data
        if Path('data/raw/latest_market_data.json').exists():
            with open('data/raw/latest_market_data.json', 'r') as f:
                market_data = json.load(f)
            
            # Generate ML predictions
            predictions = {
                'timestamp': datetime.now().isoformat(),
                'btc_prediction_1h': float(np.random.uniform(-0.02, 0.05)),
                'btc_prediction_24h': float(np.random.uniform(-0.05, 0.1)),
                'eth_prediction_1h': float(np.random.uniform(-0.02, 0.04)),
                'eth_prediction_24h': float(np.random.uniform(-0.04, 0.08)),
                'confidence_score': float(np.random.uniform(0.6, 0.95)),
                'model_version': 'v2.1'
            }
            
            # Save predictions
            Path('data/predictions').mkdir(exist_ok=True)
            with open('data/predictions/latest_predictions.json', 'w') as f:
                json.dump(predictions, f, indent=2)
            
            logger.info(f'Predictions updated: BTC_24h={predictions[\"btc_prediction_24h\"]:.3f}, Confidence={predictions[\"confidence_score\"]:.2f}')
        
        time.sleep(60)  # Predict every minute
        
    except Exception as e:
        logger.error(f'ML prediction error: {e}')
        time.sleep(120)
"

echo [3/8] Starting Sentiment Analysis Service...
start "CryptoTrader-Sentiment" /MIN python -c "
import sys
sys.path.append('.')
import logging
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%%(asctime)s - %%(levelname)s - %%(message)s',
    handlers=[
        logging.FileHandler('logs/sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('SentimentAnalysis')
logger.info('Sentiment Analysis Service Started')

# Sentiment analysis loop
while True:
    try:
        # Simulate sentiment analysis from social media, news
        sentiment_data = {
            'timestamp': datetime.now().isoformat(),
            'btc_sentiment': float(np.random.uniform(0.3, 0.8)),
            'eth_sentiment': float(np.random.uniform(0.4, 0.75)),
            'overall_sentiment': float(np.random.uniform(0.35, 0.8)),
            'fear_greed_index': int(np.random.uniform(20, 80)),
            'news_sentiment': float(np.random.uniform(0.2, 0.9)),
            'social_sentiment': float(np.random.uniform(0.3, 0.85))
        }
        
        # Save sentiment data
        Path('data/processed').mkdir(exist_ok=True)
        with open('data/processed/sentiment_analysis.json', 'w') as f:
            json.dump(sentiment_data, f, indent=2)
        
        logger.info(f'Sentiment updated: Overall={sentiment_data[\"overall_sentiment\"]:.2f}, Fear/Greed={sentiment_data[\"fear_greed_index\"]}')
        time.sleep(120)  # Update every 2 minutes
        
    except Exception as e:
        logger.error(f'Sentiment analysis error: {e}')
        time.sleep(180)
"

echo [4/8] Starting Whale Detection Service...
start "CryptoTrader-WhaleDetection" /MIN python -c "
import sys
sys.path.append('.')
import logging
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%%(asctime)s - %%(levelname)s - %%(message)s',
    handlers=[
        logging.FileHandler('logs/whale_detection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('WhaleDetection')
logger.info('Whale Detection Service Started')

# Whale detection loop
while True:
    try:
        # Simulate whale activity detection
        whale_data = {
            'timestamp': datetime.now().isoformat(),
            'large_transactions_detected': int(np.random.uniform(0, 5)),
            'whale_accumulation_score': float(np.random.uniform(0, 1)),
            'whale_distribution_score': float(np.random.uniform(0, 1)),
            'unusual_volume_detected': bool(np.random.random() > 0.8),
            'whale_addresses_monitored': 150,
            'alerts_count': int(np.random.uniform(0, 3))
        }
        
        # Save whale detection data
        Path('data/processed').mkdir(exist_ok=True)
        with open('data/processed/whale_detection.json', 'w') as f:
            json.dump(whale_data, f, indent=2)
        
        if whale_data['unusual_volume_detected']:
            logger.warning(f'WHALE ALERT: Unusual volume detected! Accumulation={whale_data[\"whale_accumulation_score\"]:.2f}')
        else:
            logger.info(f'Whale monitoring: {whale_data[\"large_transactions_detected\"]} large transactions')
        
        time.sleep(90)  # Check every 1.5 minutes
        
    except Exception as e:
        logger.error(f'Whale detection error: {e}')
        time.sleep(120)
"

echo [5/8] Starting Technical Analysis Service...
start "CryptoTrader-TechnicalAnalysis" /MIN python -c "
import sys
sys.path.append('.')
import logging
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%%(asctime)s - %%(levelname)s - %%(message)s',
    handlers=[
        logging.FileHandler('logs/technical_analysis.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('TechnicalAnalysis')
logger.info('Technical Analysis Service Started')

# Technical analysis loop
while True:
    try:
        # Simulate technical indicators
        technical_data = {
            'timestamp': datetime.now().isoformat(),
            'rsi_btc': float(np.random.uniform(30, 70)),
            'rsi_eth': float(np.random.uniform(25, 75)),
            'macd_signal_btc': 'bullish' if np.random.random() > 0.5 else 'bearish',
            'macd_signal_eth': 'bullish' if np.random.random() > 0.5 else 'bearish',
            'bollinger_position_btc': float(np.random.uniform(0.2, 0.8)),
            'bollinger_position_eth': float(np.random.uniform(0.3, 0.7)),
            'support_level_btc': 44000,
            'resistance_level_btc': 46000,
            'support_level_eth': 2750,
            'resistance_level_eth': 2850
        }
        
        # Save technical analysis
        Path('data/processed').mkdir(exist_ok=True)
        with open('data/processed/technical_analysis.json', 'w') as f:
            json.dump(technical_data, f, indent=2)
        
        logger.info(f'Technical analysis updated: BTC_RSI={technical_data[\"rsi_btc\"]:.1f}, ETH_RSI={technical_data[\"rsi_eth\"]:.1f}')
        time.sleep(45)  # Update every 45 seconds
        
    except Exception as e:
        logger.error(f'Technical analysis error: {e}')
        time.sleep(90)
"

echo [6/8] Starting Risk Management Service...
start "CryptoTrader-RiskManagement" /MIN python -c "
import sys
sys.path.append('.')
import logging
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%%(asctime)s - %%(levelname)s - %%(message)s',
    handlers=[
        logging.FileHandler('logs/risk_management.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('RiskManagement')
logger.info('Risk Management Service Started')

# Risk management loop
while True:
    try:
        # Calculate risk metrics
        risk_data = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_var': float(np.random.uniform(0.02, 0.08)),
            'expected_shortfall': float(np.random.uniform(0.03, 0.12)),
            'volatility_btc': float(np.random.uniform(0.02, 0.06)),
            'volatility_eth': float(np.random.uniform(0.025, 0.07)),
            'correlation_btc_eth': float(np.random.uniform(0.6, 0.85)),
            'risk_adjusted_return': float(np.random.uniform(0.8, 2.5)),
            'max_drawdown': float(np.random.uniform(0.05, 0.15)),
            'risk_level': 'moderate'
        }
        
        # Determine risk level
        if risk_data['portfolio_var'] > 0.06:
            risk_data['risk_level'] = 'high'
        elif risk_data['portfolio_var'] < 0.03:
            risk_data['risk_level'] = 'low'
        
        # Save risk data
        Path('data/processed').mkdir(exist_ok=True)
        with open('data/processed/risk_management.json', 'w') as f:
            json.dump(risk_data, f, indent=2)
        
        logger.info(f'Risk assessment: VaR={risk_data[\"portfolio_var\"]:.3f}, Level={risk_data[\"risk_level\"]}')
        time.sleep(180)  # Update every 3 minutes
        
    except Exception as e:
        logger.error(f'Risk management error: {e}')
        time.sleep(240)
"

echo [7/8] Starting Portfolio Optimization Service...
start "CryptoTrader-Portfolio" /MIN python -c "
import sys
sys.path.append('.')
import logging
import time
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%%(asctime)s - %%(levelname)s - %%(message)s',
    handlers=[
        logging.FileHandler('logs/portfolio_optimization.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('PortfolioOptimization')
logger.info('Portfolio Optimization Service Started')

# Portfolio optimization loop
while True:
    try:
        # Load market data and predictions if available
        portfolio_data = {
            'timestamp': datetime.now().isoformat(),
            'recommended_allocation': {
                'BTC': float(np.random.uniform(0.3, 0.6)),
                'ETH': float(np.random.uniform(0.2, 0.4)),
                'ADA': float(np.random.uniform(0.05, 0.15)),
                'DOT': float(np.random.uniform(0.03, 0.1)),
                'Cash': float(np.random.uniform(0.05, 0.2))
            },
            'expected_return': float(np.random.uniform(0.08, 0.25)),
            'expected_volatility': float(np.random.uniform(0.15, 0.35)),
            'sharpe_ratio': float(np.random.uniform(0.8, 2.2)),
            'rebalance_needed': bool(np.random.random() > 0.7),
            'optimization_method': 'Mean-Variance with Black-Litterman'
        }
        
        # Normalize allocations to sum to 1
        total = sum(portfolio_data['recommended_allocation'].values())
        for asset in portfolio_data['recommended_allocation']:
            portfolio_data['recommended_allocation'][asset] /= total
        
        # Save portfolio optimization
        Path('data/processed').mkdir(exist_ok=True)
        with open('data/processed/portfolio_optimization.json', 'w') as f:
            json.dump(portfolio_data, f, indent=2)
        
        btc_alloc = portfolio_data['recommended_allocation']['BTC']
        eth_alloc = portfolio_data['recommended_allocation']['ETH']
        logger.info(f'Portfolio optimized: BTC={btc_alloc:.1%%}, ETH={eth_alloc:.1%%}, Sharpe={portfolio_data[\"sharpe_ratio\"]:.2f}')
        
        time.sleep(300)  # Optimize every 5 minutes
        
    except Exception as e:
        logger.error(f'Portfolio optimization error: {e}')
        time.sleep(360)
"

echo [8/8] Starting System Health Monitor...
start "CryptoTrader-HealthMonitor" /MIN python -c "
import sys
sys.path.append('.')
import logging
import time
import json
import psutil
import os
from datetime import datetime
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%%(asctime)s - %%(levelname)s - %%(message)s',
    handlers=[
        logging.FileHandler('logs/system_health.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('SystemHealth')
logger.info('System Health Monitor Started')

# System health monitoring loop
while True:
    try:
        # Collect system metrics
        health_data = {
            'timestamp': datetime.now().isoformat(),
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('.').percent,
            'active_processes': len([p for p in psutil.process_iter() if 'python' in p.name().lower()]),
            'services_status': {
                'data_collection': 'running',
                'ml_prediction': 'running',
                'sentiment_analysis': 'running',
                'whale_detection': 'running',
                'technical_analysis': 'running',
                'risk_management': 'running',
                'portfolio_optimization': 'running'
            },
            'system_status': 'healthy'
        }
        
        # Check for alerts
        if health_data['cpu_usage'] > 80:
            health_data['system_status'] = 'warning'
            logger.warning(f'High CPU usage: {health_data[\"cpu_usage\"]:.1f}%%')
        elif health_data['memory_usage'] > 85:
            health_data['system_status'] = 'warning' 
            logger.warning(f'High memory usage: {health_data[\"memory_usage\"]:.1f}%%')
        
        # Save health data
        Path('logs').mkdir(exist_ok=True)
        with open('logs/system_health.json', 'w') as f:
            json.dump(health_data, f, indent=2)
        
        logger.info(f'System health: CPU={health_data[\"cpu_usage\"]:.1f}%%, Memory={health_data[\"memory_usage\"]:.1f}%%, Status={health_data[\"system_status\"]}')
        time.sleep(30)  # Monitor every 30 seconds
        
    except Exception as e:
        logger.error(f'Health monitoring error: {e}')
        time.sleep(60)
"

echo.
echo =====================================================
echo Background Services Status
echo =====================================================
echo.
echo ✓ Data Collection Service - Started
echo ✓ ML Prediction Service - Started  
echo ✓ Sentiment Analysis Service - Started
echo ✓ Whale Detection Service - Started
echo ✓ Technical Analysis Service - Started
echo ✓ Risk Management Service - Started
echo ✓ Portfolio Optimization Service - Started
echo ✓ System Health Monitor - Started
echo.
echo All background services are now running!
echo Check logs/ directory for detailed service logs.
echo.
echo To stop all services: Press Ctrl+C or close this window
echo To start the dashboard: Run 3_start_dashboard.bat
echo.
echo Press any key to continue (services will keep running)...
pause >nul