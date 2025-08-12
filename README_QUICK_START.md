# CryptoSmartTrader V2 - Quick Start

> 🚀 **Enterprise Cryptocurrency Trading Intelligence Platform**  
> Sophisticated multi-agent system designed for institutional-grade analysis and automated trading strategies targeting 500%+ returns.

## ⚡ Quick Setup

### Prerequisites
- Python 3.11+ 
- UV package manager
- API Keys: Kraken, OpenAI

### 1. Clone & Install
```bash
git clone https://github.com/cryptosmarttrader/cryptosmarttrader-v2.git
cd cryptosmarttrader-v2
uv sync
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys:
# KRAKEN_API_KEY=your_key
# KRAKEN_SECRET=your_secret  
# OPENAI_API_KEY=your_key
```

### 3. Start Services
```bash
# Single command - starts all services
uv sync && (uv run python api/health_endpoint.py & uv run python metrics/metrics_server.py & uv run streamlit run app_fixed_all_issues.py --server.port 5000 --server.headless true --server.address 0.0.0.0 & wait)
```

### 4. Access Dashboard
- **Main Dashboard**: http://localhost:5000
- **API Health**: http://localhost:8001/health
- **Metrics**: http://localhost:8000/metrics

## 🎯 Core Features

### Multi-Agent Intelligence
- **8 Specialized Agents**: Data Collector, Sentiment Analyzer, Technical Analyzer, ML Predictor, Whale Detector, Risk Manager, Portfolio Optimizer, Health Monitor
- **Real-time Analysis**: 1457+ cryptocurrencies via Kraken API
- **80% Confidence Gate**: Strict quality threshold for trade signals

### Advanced ML/AI
- **Deep Learning**: PyTorch LSTM/GRU/Transformer models  
- **Ensemble Predictions**: Multi-horizon forecasting (1H, 24H, 7D, 30D)
- **Uncertainty Quantification**: Bayesian inference and Monte Carlo methods
- **Regime Detection**: Hidden Markov Models for market state recognition

### Risk Management
- **Shadow Trading**: Risk-free signal validation
- **Dynamic Position Sizing**: Kelly-lite optimization with correlation caps
- **Kill-Switch**: Emergency stop mechanism
- **Portfolio Limits**: Hard caps and correlation controls

## 📊 System Architecture

### Multi-Service Design
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Dashboard  │    │  API Server │    │   Metrics   │
│   Port 5000 │    │  Port 8001  │    │  Port 8000  │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       └───────────────────┼───────────────────┘
                           │
                ┌─────────────────┐
                │  Core Agents    │
                │  • Data         │
                │  • Sentiment    │
                │  • Technical    │
                │  • ML Predictor │
                │  • Risk Manager │
                └─────────────────┘
```

### Technology Stack
- **Backend**: Python 3.11, FastAPI, asyncio
- **Frontend**: Streamlit with real-time updates
- **ML**: PyTorch, scikit-learn, XGBoost, LSTM/GRU
- **Data**: CCXT (Kraken/Binance), pandas, numpy
- **Monitoring**: Prometheus, structured JSON logging

## 🔧 Development

### Test Suite
```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=src/cryptosmarttrader --cov-report=html

# Specific test markers
uv run pytest -m "not slow"  # Skip slow tests
uv run pytest -m "unit"      # Unit tests only
```

### Code Quality
```bash
# Linting & formatting
uvx ruff check .
uvx black --check .
uvx mypy .

# Security audit
uvx pip-audit
```

## 📈 Usage Examples

### Basic Trading Analysis
```python
from cryptosmarttrader.agents import TechnicalAgent, SentimentAgent
from cryptosmarttrader.core import DataManager

# Initialize components
data_manager = DataManager()
ta_agent = TechnicalAgent()
sentiment_agent = SentimentAgent()

# Get market data
btc_data = data_manager.get_ohlcv("BTC/USD", timeframe="1h")

# Technical analysis
ta_signals = ta_agent.analyze(btc_data)
print(f"RSI: {ta_signals['rsi']:.2f}")
print(f"Signal: {ta_signals['signal']}")

# Sentiment analysis  
sentiment = sentiment_agent.analyze_market("BTC")
print(f"Sentiment Score: {sentiment['score']:.2f}")
```

### ML Predictions
```python
from cryptosmarttrader.ml import PredictionEngine

# Multi-horizon predictions
predictor = PredictionEngine()
predictions = predictor.predict(
    symbol="BTC/USD",
    horizons=["1h", "24h", "7d"]
)

for horizon, pred in predictions.items():
    print(f"{horizon}: {pred['price']:.2f} (confidence: {pred['confidence']:.1%})")
```

## 🎯 Target Performance

### Return Objectives
- **Primary Goal**: 500%+ returns within 6 months
- **Risk-Adjusted**: Sharpe ratio > 2.0
- **Drawdown Limit**: < 15% maximum
- **Win Rate**: > 60% profitable positions

### Quality Metrics
- **Prediction Accuracy**: > 70% directional accuracy
- **Signal Quality**: 80%+ confidence threshold
- **System Uptime**: 99.9% availability target
- **Data Freshness**: < 60 second latency

## 🆘 Quick Troubleshooting

### Common Issues

**Services Not Starting**
```bash
# Check ports availability
ss -tulpn | grep -E "(5000|8001|8000)"

# Restart services
pkill -f "streamlit\|uvicorn\|python.*health\|python.*metrics"
# Then restart with main command
```

**API Key Issues**
```bash
# Verify environment
uv run python -c "import os; print('KRAKEN_API_KEY' in os.environ)"
# Should print: True
```

**Import Errors After Update**
```bash
# Reinstall in development mode
uv pip install -e .
```

### Performance Issues
- **Memory Usage**: Monitor with `htop` - expect 2-4GB normal usage
- **CPU Load**: High during ML training (normal), idle < 10%  
- **Disk Space**: Models and logs grow over time, monitor `/logs/` and `/models/`

## 📚 Next Steps

1. **[Operations Guide](README_OPERATIONS.md)** - Production deployment and monitoring
2. **[API Documentation](docs/api/)** - Full API reference
3. **[Architecture Deep Dive](docs/architecture/)** - System design details
4. **[Contributing Guide](CONTRIBUTING.md)** - Development workflow

## 🔗 Links

- **Dashboard**: http://localhost:5000
- **Health API**: http://localhost:8001/health  
- **Metrics**: http://localhost:8000/metrics
- **Documentation**: https://docs.cryptosmarttrader.com
- **Issues**: https://github.com/cryptosmarttrader/cryptosmarttrader-v2/issues