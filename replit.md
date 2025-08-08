# Replit.md

## Overview
CryptoSmartTrader V2 is a sophisticated multi-agent cryptocurrency trading intelligence system designed for professional institutional-grade analysis. It analyzes over 1457+ cryptocurrencies using a coordinated ensemble of specialized AI agents, providing real-time market analysis, deep learning-powered price predictions, sentiment analysis, technical analysis, backtesting, and automated trade execution with comprehensive risk management. The system aims for high returns by combining technical indicators, sentiment analysis, whale detection, and ML predictions with OpenAI intelligence, focusing on comprehensive analysis for detecting fast-growing cryptocurrencies. Key features include zero synthetic data tolerance, mandatory deep learning, cross-coin feature fusion, true async processing, enterprise security, Bayesian uncertainty quantification, dynamic coin discovery, advanced AI/ML capabilities (automated feature engineering, meta-learning, causal inference, shadow trading), multi-agent cooperation, model monitoring, black swan simulation, and real-time order book data integration.

## User Preferences
Preferred communication style: Simple, everyday language.
Data integrity policy: ZERO-TOLERANCE for incomplete data - coins with missing sentiment/on-chain/technical data are HARD BLOCKED from training and display. Only authentic data from real sources with 80%+ completeness allowed. No fallback, no synthetic data, no incomplete coins passed through.

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
- **Comprehensive Testing Framework (August 2025):** Enterprise pytest suite with 80%+ coverage gates, asyncio testing, exchange/API mocks, contract tests for CCXT/sentiment parsers, smoke tests for dashboard, CI/CD integration.
- **Enterprise Security Framework (August 2025):** Secrets management with Vault integration, log redaction, and principled error handling to prevent API key leakage. All credentials secured via .env/Vault with automatic redaction in logs and tracebacks.
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
- **Enterprise ML Uncertainty Quantification (August 2025):** Bayesian LSTM with Monte Carlo sampling, quantile regression ensembles, confidence intervals (68%, 95%, 99%), epistemic vs aleatoric uncertainty decomposition, production-ready uncertainty thresholds.
- **Market Regime Detection & Routing (August 2025):** Unsupervised regime detector (bull/bear/sideways/volatile), regime-specific model routing with confidence scoring, adaptive feature weighting per regime, regime transition smoothing.
- **ML Feature Engineering Monitor (August 2025):** Comprehensive leakage detection (look-ahead, target, temporal), SHAP-based feature importance analysis, statistical drift monitoring, automated feature pruning recommendations.
- **SLO-Driven ML Operations (August 2025):** Formal SLO monitoring with MAE/MAPE/precision@K thresholds, automatic retraining triggers, model rollback capabilities, coverage validation for prediction intervals.
- **Continual Learning & Meta-Learning:** Automated retraining with drift detection, online learning, and few-shot adaptation for new markets.
- **Automated Feature Engineering:** Auto-featuretools, deep feature synthesis, and genetic algorithms for continuous feature discovery.
- **Causal Inference:** Double Machine Learning, Granger Causality, and counterfactual predictions for understanding market movements.
- **Reinforcement Learning:** PPO for dynamic, risk-aware portfolio allocation and real-time rebalancing.
- **Self-Healing System:** Autonomous monitoring, black swan detection, and auto-disabling/recovery features.
- **Synthetic Data Augmentation:** Generation of black swan, regime shift, and flash crash scenarios for robust stress testing.
- **Human-in-the-Loop & Explainability:** Active learning, feedback-driven learning, and SHAP for model transparency and human validation.
- **Shadow Trading:** Full paper trading engine with realistic market simulation for pre-production model validation.
- **Advanced Execution Simulation (August 2025):** Level-2 order book simulation with market impact, partial fills, exchange-specific latency/fees, maintenance windows, and realistic slippage modeling for accurate backtesting.
- **Enterprise Portfolio Risk Management (August 2025):** Per-coin hard caps (value & ADV %), correlation limits, automated position kill-switch based on health scores, GO/NOGO thresholds, and comprehensive risk monitoring with emergency flattening capabilities.
- **Coverage Audit System (August 2025):** Daily automated audits ensuring 99%+ exchange coverage with missing coin alerts, new listing detection, impact scoring, and comprehensive gap analysis to guarantee no trading opportunities are missed. Hard gates block trading when coverage <95% or data completeness <98% per coin.
- **Hard Data Integrity Filter (August 2025):** Zero-tolerance policy for incomplete data - coins with missing sentiment/on-chain/technical data are HARD BLOCKED from training and display, with 80% minimum completeness threshold and comprehensive component validation.
- **Multi-Horizon Batch Inference Engine (August 2025):** Unified batch processing system for all coins across all horizons (1H/4H/24H/7D/30D) with atomic operations, uniform feature engineering, parallel processing, and comprehensive SLO monitoring for systematic ML predictions.
- **Strict Confidence Gate Manager (August 2025):** Hard confidence threshold enforcement in orchestration and dashboard - shows NOTHING when no candidates meet 80% confidence threshold, with comprehensive empty state handling and gate status monitoring.
- **Shadow Trading Engine with Mandatory Soak Period (August 2025):** Comprehensive paper trading system with 4-8 week mandatory soak period, P&L verification, false positive ratio monitoring, and automated live trading authorization only after passing all validation criteria.

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