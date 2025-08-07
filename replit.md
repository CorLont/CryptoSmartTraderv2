# Replit.md

## Overview
CryptoSmartTrader V2 is a sophisticated multi-agent cryptocurrency trading intelligence system designed for professional institutional-grade analysis. It analyzes over 1457+ cryptocurrencies using a coordinated ensemble of specialized AI agents, providing real-time market analysis, deep learning-powered price predictions, sentiment analysis, technical analysis, backtesting, and automated trade execution with comprehensive risk management. The system aims for high returns by combining technical indicators, sentiment analysis, whale detection, and ML predictions with OpenAI intelligence, focusing on comprehensive analysis for detecting fast-growing cryptocurrencies.

The system emphasizes:
- Zero synthetic data tolerance and mandatory deep learning (LSTM/Transformer/N-BEATS models).
- Cross-coin feature fusion and true async processing for high concurrency.
- Enterprise security, Bayesian uncertainty quantification, and dynamic coin discovery.
- Enhanced capabilities across sentiment, technical, whale, ML, and backtest agents, as well as the orchestrator.
- Advanced AI/ML features like automated feature engineering, meta-learning, causal inference, and shadow trading.
- Multi-agent cooperation, model monitoring, and black swan simulation.
- Deep learning time series, multi-modal feature fusion, confidence filtering, and self-learning loops.
- SHAP explainability, anomaly detection, AI news mining, and AI portfolio optimization.
- Real-time order book data integration, CI/CD pipeline, and comprehensive testing.
- Robustness features including an Enterprise Security Manager, Async Coordination Manager, and comprehensive audit trails.
- Advanced ML/AI features like Feature Fusion Engine, Market Regime Detector, and Shadow Testing Engine.
- Core deep learning with LSTM, GRU, Transformer, and N-BEATS models, alongside multimodal data processing.
- Uncertainty modeling with Bayesian neural networks and quantile regression for probabilistic outputs.
- A distributed orchestrator for high availability and resource optimization.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Core Architecture Pattern
The system employs a multi-agent coordination architecture with six specialized AI agents operating in parallel: Sentiment Analysis, Technical Analysis, ML Predictor, Backtest, Trade Executor, and Whale Detector.

### Technology Stack
- **Frontend:** Streamlit.
- **Backend:** Python with multi-threaded agent coordination.
- **Machine Learning:** PyTorch for deep learning (LSTM, GRU, Transformer, N-BEATS), scikit-learn, XGBoost, Optuna for AutoML.
- **Data Processing:** pandas, numpy, CuPy, Numba.
- **Exchange Integration:** CCXT.
- **Visualization:** Plotly.
- **Storage:** Local file system with JSON/CSV and caching.

### Key Architectural Decisions
- **Universal Configuration Management:** Centralized with validation and backup.
- **Intelligent Cache Manager:** Memory-aware caching with TTL and cleanup.
- **Health Monitoring System:** Continuous checks with alerts.
- **Multi-Exchange Architecture:** CCXT-based abstraction with rate limiting and failover.
- **Real-time Data Pipeline:** Parallel processing with rate limiting, timeouts, and strict validation.
- **GPU Optimization:** Automatic GPU/CPU detection and adaptive tuning.
- **Enterprise Logging:** Structured logging with file rotation.
- **Thread-Safe Design:** All components use threading locks.
- **Daily Analysis Coordination:** Automated scheduling for ML analysis and social scraping.
- **Dependency Injection:** Using `dependency-injector` for explicit and testable design.
- **Enhanced Configuration Management:** Pydantic settings with environment variable support.
- **Async I/O & Resilience:** Asynchronous HTTP with exponential backoff and error handling.
- **Input Validation & Security:** Pydantic models for strict validation.
- **Monitoring & Observability:** Prometheus metrics, structured JSON logging, OpenAPI documentation.
- **Multi-Horizon ML System:** Training and inference across 1H, 24H, 7D, and 30D horizons with feature engineering, ensemble modeling, and self-learning.

### Dashboard Architecture
Five specialized Streamlit dashboards: Main, Comprehensive Market, Analysis Control, Agent, Portfolio, Production Monitoring, and Crypto AI System.

### Data Management
- **Centralized Data Manager:** Coordinates data collection from exchanges.
- **Coin Registry:** Dynamic cryptocurrency discovery.
- **ML Model Manager:** Handles model training and ensemble management.

## External Dependencies

### Core Python Libraries
- **streamlit**
- **pandas**, **numpy**
- **plotly**
- **threading**, **pathlib**, **psutil**

### Machine Learning Stack
- **scikit-learn**
- **xgboost**, **lightgbm**
- **numba**

### Trading and Market Data
- **ccxt**
- **textblob**

### Optional Performance Libraries
- **cupy**
- **openai**

### System Utilities
- **logging**, **json**, **pickle**
- **smtplib**
- **Pydantic**
- **dependency-injector**
- **FastAPI**

### External Services
- **OpenAI API**
- **Cryptocurrency Exchanges:** Kraken, Binance, KuCoin, Huobi (via CCXT)
- **Email SMTP**
- **HashiCorp Vault**
- **Prometheus**