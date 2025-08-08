# Replit.md

## Overview
CryptoSmartTrader V2 is a sophisticated multi-agent cryptocurrency trading intelligence system designed for professional institutional-grade analysis. It analyzes over 1457+ cryptocurrencies using a coordinated ensemble of specialized AI agents, providing real-time market analysis, deep learning-powered price predictions, sentiment analysis, technical analysis, backtesting, and automated trade execution with comprehensive risk management. The system aims for high returns by combining technical indicators, sentiment analysis, whale detection, and ML predictions with OpenAI intelligence, focusing on comprehensive analysis for detecting fast-growing cryptocurrencies. Key features include zero synthetic data tolerance, mandatory deep learning, cross-coin feature fusion, true async processing, enterprise security, Bayesian uncertainty quantification, dynamic coin discovery, advanced AI/ML capabilities (automated feature engineering, meta-learning, causal inference, shadow trading), multi-agent cooperation, model monitoring, black swan simulation, and real-time order book data integration.

## User Preferences
Preferred communication style: Simple, everyday language.
Data integrity policy: No demo/synthetic data - only authentic data from real sources. When data is unavailable or unreliable, show clear error states with specific reasons and required actions.

## System Architecture

### Core Architecture Pattern
**Distributed Multi-Process Architecture (August 2025)** - The system employs a distributed orchestrator managing 8 isolated agent processes with circuit breakers, exponential backoff, health monitoring, and automatic restart capabilities. Each agent runs in complete process isolation to eliminate single points of failure: Data Collector, Sentiment Analyzer, Technical Analyzer, ML Predictor, Whale Detector, Risk Manager, Portfolio Optimizer, and Health Monitor.

### Perfect Workstation Deployment (August 2025)
**Geoptimaliseerd voor perfect draaien op workstation met 3 hoofd .bat bestanden:**
- **1_install_all_dependencies.bat:** Complete installatie van alle Python dependencies, directory setup, GPU configuratie
- **2_start_background_services.bat:** 8 background services (data collection, ML prediction, sentiment analysis, whale detection, technical analysis, risk management, portfolio optimization, health monitoring)
- **3_start_dashboard.bat:** Dashboard launcher met health check en alle 8 gespecialiseerde dashboards

### Technology Stack
- **Frontend:** Streamlit.
- **Backend:** Python with multi-threaded agent coordination.
- **Machine Learning:** PyTorch for deep learning (LSTM, GRU, Transformer, N-BEATS), scikit-learn, XGBoost, Optuna for AutoML.
- **Data Processing:** pandas, numpy, CuPy, Numba.
- **Exchange Integration:** CCXT.
- **Visualization:** Plotly.
- **Storage:** Local file system with JSON/CSV and caching.

### Key Architectural Decisions
- **Production-Grade Logging & Monitoring (August 2025):** Enterprise structured JSON logging with correlation IDs, Prometheus metrics (latency/completeness/error-ratio), intelligent alerting with configurable thresholds, full observability stack.
- **Dependency Injection Container (August 2025):** Full dependency-injector implementation with constructor injection, eliminates global singletons, enables comprehensive testing and mocking.
- **Pydantic Configuration Management (August 2025):** Type-safe settings with BaseSettings, environment variable support, automatic validation, and centralized .env configuration.
- **Async I/O Architecture (August 2025):** Complete async/await implementation with aiohttp, global rate limiting (15 req/sec), tenacity retries with exponential backoff + jitter, idempotent atomic file writes.
- **Distributed Multi-Process Architecture (August 2025):** 8 isolated agent processes with circuit breakers, health monitoring, automatic restart capabilities, eliminates single points of failure.
- **Universal Configuration Management:** Centralized with validation and backup.
- **Intelligent Cache Manager:** Memory-aware caching with TTL and cleanup.
- **Health Monitoring System:** Continuous checks with alerts.
- **Multi-Exchange Architecture:** CCXT-based abstraction with rate limiting and failover.
- **Real-time Data Pipeline:** Parallel processing with rate limiting, timeouts, and strict validation.
- **GPU Optimization:** Automatic GPU/CPU detection and adaptive tuning.
- **Enterprise Logging:** Structured logging with file rotation.
- **Thread-Safe Design:** All components use threading locks.
- **Daily Analysis Coordination:** Automated scheduling for ML analysis and social scraping.
- **Input Validation & Security:** Pydantic models for strict validation.
- **Monitoring & Observability:** Prometheus metrics, structured JSON logging, OpenAPI documentation.
- **Multi-Horizon ML System:** Training and inference across 1H, 24H, 7D, and 30D horizons with feature engineering, ensemble modeling, and self-learning.
- **Dashboard Architecture:** Five specialized Streamlit dashboards: Main, Comprehensive Market, Analysis Control, Agent, Portfolio, Production Monitoring, and Crypto AI System.
- **Data Management:** Centralized Data Manager, Dynamic Coin Registry, ML Model Manager.
- **Deep Learning Engine:** Standardized use of LSTM, GRU, Transformer, N-BEATS for complex pattern detection.
- **Uncertainty Modeling:** Bayesian neural networks, quantile regression, and ensemble methods for probabilistic outputs and risk-aware trading.
- **Continual Learning & Meta-Learning:** Automated retraining with drift detection, online learning, and few-shot adaptation for new markets.
- **Automated Feature Engineering:** Auto-featuretools, deep feature synthesis, and genetic algorithms for continuous feature discovery.
- **Market Regime Detection:** Unsupervised learning (autoencoders, clustering) for adaptive model switching.
- **Causal Inference:** Double Machine Learning, Granger Causality, and counterfactual predictions for understanding market movements.
- **Reinforcement Learning:** PPO for dynamic, risk-aware portfolio allocation and real-time rebalancing.
- **Self-Healing System:** Autonomous monitoring, black swan detection, and auto-disabling/recovery features.
- **Synthetic Data Augmentation:** Generation of black swan, regime shift, and flash crash scenarios for robust stress testing.
- **Human-in-the-Loop & Explainability:** Active learning, feedback-driven learning, and SHAP for model transparency and human validation.
- **Shadow Trading:** Full paper trading engine with realistic market simulation for pre-production model validation.

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