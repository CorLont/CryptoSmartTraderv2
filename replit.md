# replit.md

## Overview
CryptoSmartTrader V2 is a sophisticated multi-agent cryptocurrency trading intelligence system designed for professional institutional-grade analysis. It analyzes over 1457+ cryptocurrencies using a coordinated ensemble of specialized AI agents, providing real-time market analysis, deep learning-powered price predictions, sentiment analysis, technical analysis, backtesting, and automated trade execution with comprehensive risk management. The system aims for high returns by combining technical indicators, sentiment analysis, whale detection, and ML predictions with OpenAI intelligence, focusing on comprehensive analysis for detecting fast-growing cryptocurrencies.

The enterprise architecture goals include higher prediction accuracy, zero dummy-data tolerance, strict confidence gating, robust async scraping, modern ML/AI techniques (uncertainty, regime, ensembles), and daily evaluation with GO/NOGO decisions. Key capabilities include zero synthetic data tolerance, mandatory deep learning, cross-coin feature fusion, true async processing, enterprise security, Bayesian uncertainty quantification, dynamic coin discovery, advanced AI/ML, multi-agent cooperation, model monitoring, black swan simulation, and real-time order book data integration.

## User Preferences
Preferred communication style: Simple, everyday language.
Data integrity policy: ZERO-TOLERANCE for incomplete data - coins with missing sentiment/on-chain/technical data are HARD BLOCKED from training and display. Only authentic data from real sources with 80%+ completeness allowed. No fallback, no synthetic data, no incomplete coins passed through.

## System Architecture

### Core Architecture Pattern
The system employs a Distributed Multi-Process Architecture, managing 8 isolated agent processes (Data Collector, Sentiment Analyzer, Technical Analyzer, ML Predictor, Whale Detector, Risk Manager, Portfolio Optimizer, and Health Monitor) with circuit breakers, exponential backoff, health monitoring, and automatic restart capabilities. Each agent runs in complete process isolation to eliminate single points of failure.

### Technology Stack
- **Frontend:** Streamlit.
- **Backend:** Python with multi-threaded agent coordination.
- **Machine Learning:** PyTorch (LSTM, GRU, Transformer, N-BEATS), scikit-learn, XGBoost, Optuna.
- **Data Processing:** pandas, numpy, CuPy, Numba.
- **Exchange Integration:** CCXT.
- **Visualization:** Plotly.
- **Storage:** Local file system with JSON/CSV and caching.

### Key Architectural Decisions
- **Comprehensive Testing Framework:** Enterprise pytest suite with 80%+ coverage gates, asyncio testing, API mocks, and CI/CD integration.
- **Enterprise Security Framework:** Secrets management with Vault integration, log redaction, and principled error handling. All credentials secured via .env/Vault.
- **Production-Grade Logging & Monitoring:** Structured JSON logging with correlation IDs, Prometheus metrics, intelligent alerting, and full observability.
- **Dependency Injection Container:** Full dependency-injector implementation for comprehensive testing and mocking.
- **Pydantic Configuration Management:** Type-safe settings with environment variable support and automatic validation.
- **Async I/O Architecture:** Complete async/await implementation with aiohttp, global rate limiting, tenacity retries, and idempotent atomic file writes.
- **Universal Configuration Management:** Centralized with validation and backup.
- **Intelligent Cache Manager:** Memory-aware caching with TTL and cleanup.
- **Health Monitoring System:** Continuous checks with alerts and GO/NO-GO gates based on weighted system health scores.
- **Multi-Exchange Architecture:** CCXT-based abstraction with rate limiting and failover.
- **Real-time Data Pipeline:** Parallel processing with rate limiting, timeouts, and strict validation.
- **GPU Optimization:** Automatic GPU/CPU detection and adaptive tuning.
- **Enterprise Logging:** Structured logging with file rotation.
- **Regime Detection & Market Regime Routing:** HMM-based and rule-based classification for dynamic model routing and A/B testing.
- **Thread-Safe Design:** All components use threading locks.
- **Daily Analysis Coordination:** Automated scheduling for ML analysis and social scraping.
- **Input Validation & Security:** Pydantic models for strict validation.
- **Monitoring & Observability:** Prometheus metrics, structured JSON logging, OpenAPI documentation.
- **Multi-Horizon ML System:** Training and inference across 1H, 24H, 7D, and 30D horizons with feature engineering, ensemble modeling, and self-learning.
- **Dashboard Architecture:** Five specialized Streamlit dashboards: Main, Comprehensive Market, Analysis Control, Agent, Portfolio, Production Monitoring, and Crypto AI System.
- **Data Management:** Centralized Data Manager, Dynamic Coin Registry, ML Model Manager.
- **Deep Learning Engine:** Standardized use of LSTM, GRU, Transformer, N-BEATS.
- **Enterprise ML Uncertainty Quantification:** Bayesian LSTM with Monte Carlo sampling, quantile regression ensembles, and confidence intervals.
- **Strict 80% Confidence Gate System:** Enterprise-grade filtering in orchestration and dashboard, with empty state handling and SHAP explainability.
- **ML Feature Engineering Monitor:** Leakage detection, SHAP-based feature importance, drift monitoring, and automated feature pruning.
- **SLO-Driven ML Operations:** Formal SLO monitoring with thresholds, automatic retraining triggers, and model rollback capabilities.
- **Continual Learning & Meta-Learning:** Automated retraining with drift detection, online learning, and few-shot adaptation.
- **Automated Feature Engineering:** Auto-featuretools, deep feature synthesis, and genetic algorithms.
- **Causal Inference:** Double Machine Learning, Granger Causality, and counterfactual predictions.
- **Reinforcement Learning:** PPO for dynamic, risk-aware portfolio allocation.
- **Self-Healing System:** Autonomous monitoring, black swan detection, and auto-disabling/recovery.
- **Synthetic Data Augmentation:** Generation of black swan, regime shift, and flash crash scenarios for stress testing.
- **Human-in-the-Loop & Explainability:** Active learning, feedback-driven learning, and SHAP for model transparency.
- **Shadow Trading:** Full paper trading engine with realistic market simulation for pre-production model validation, including a mandatory soak period.
- **Advanced Execution Simulation:** Level-2 order book simulation with market impact, partial fills, exchange-specific latency/fees, and realistic slippage modeling for accurate backtesting.
- **Enterprise Portfolio Risk Management:** Per-coin hard caps, correlation limits, automated position kill-switch, and comprehensive risk monitoring.
- **Coverage Audit System:** Daily automated audits ensuring 99%+ exchange coverage with alerts for missing coins and new listings. Hard gates block trading when coverage or data completeness is insufficient.
- **Hard Data Integrity Filter:** Zero-tolerance policy for incomplete data; coins with missing data are hard blocked.
- **Multi-Horizon Batch Inference Engine:** Unified batch processing for all coins across all horizons with atomic operations, parallel processing, and SLO monitoring.
- **Multi-Horizon Signal Quality Validator:** Enterprise signal validation system with defined metrics (Precision@K, hit-rates, MAE, Sharpe, max drawdown, calibration testing).
- **System Health Monitor with GO/NO-GO Gates:** Enterprise health scoring system with weighted components (validation accuracy, Sharpe ratio, feedback hit-rate, error ratio, data completeness, tuning freshness) determining live trading authorization.
- **Enterprise Meetscripts Suite:** Complete operational measurement toolkit including `strict_filter.py`, `coverage_audit.py`, `evaluator.py`, `calibration.py`, and `health_score.py`.
- **Complete Operational Playbook:** Full enterprise workflow automation with `nightly_batch.py`, `post_batch_validation.py`, `dashboard_gate_checker.py`, and `shadow_to_live_validator.py`.
- **Daily Metrics Logging System:** Comprehensive daily performance tracking with structured JSON logging and historical trend analysis.
- **Enterprise Evaluation System:** Complete evaluation framework including performance metrics, coverage audit, and GO/NO-GO decisions.
- **Distributed Process Isolation System:** Multi-process agent architecture with complete isolation, automatic restart, health monitoring, and circuit breakers.
- **Async Queue System with Rate Limiting:** Enterprise async message queue system using `asyncio.Queue` (Redis compatible) with centralized rate limiter, message prioritization, and retry logic.
- **Prometheus Metrics & Monitoring:** Comprehensive metrics collection (latency, error-ratio, data completeness, CPU/memory/GPU usage) via Prometheus HTTP endpoint (port 8090) for Grafana integration.
- **Drift Detection System (August 2025):** Complete drift detection with error trending analysis, KS-test for feature distribution monitoring, and performance degradation detection. Triggers alerts for relative error increases >15%, distribution shifts (p<0.01), and accuracy drops >10% with severity classification (low/medium/high/critical).
- **Fine-Tune Scheduler with EWC (August 2025):** Automated fine-tuning system with replay buffer (10k samples), Elastic Weight Consolidation (EWC) for continual learning, priority-based job scheduling, and small learning-rate updates (1e-4 base, 5e-5 for drift). Supports drift-triggered, scheduled, and manual fine-tuning jobs.
- **Auto-Disable System (August 2025):** Trading safety system that automatically disables live trading when health <60 (paper trading), <30 (all trading disabled), with grace period for auto-enable, manual override capabilities, and comprehensive status change logging with reason tracking.
- **Drift-Fine-Tune-Auto-Disable Integration (August 2025):** Complete integration system that monitors drift detection, triggers fine-tuning responses with delays, calculates system health scores, and automatically manages trading modes. Includes pending drift response queuing, critical drift immediate disable, and health-based fine-tune job creation.
- **L2 Orderbook Simulator (August 2025):** Complete Level-2 order book simulation with realistic market microstructure, partial fills, exchange fees, Time-in-Force handling (GTC/IOC/FOK/DAY), latency simulation, and market impact modeling. Supports all order types with realistic execution scenarios.
- **Slippage Estimator (August 2025):** Real-time slippage estimation system with p50/p90/p95 percentile calculations across order size buckets. Features outlier removal, confidence intervals, market impact estimation, and predictive slippage modeling for trading evaluation.
- **Paper Trading Engine (August 2025):** Mandatory 4-week paper trading validation system with complete execution logging, realistic market simulation, comprehensive performance metrics calculation, and strict validation criteria. Includes Sharpe ratio calculation, drawdown analysis, and complete audit trail for regulatory compliance.
- **MLflow Manager (August 2025):** Complete MLflow integration for model tracking, versioning, and registry with horizon/regime tags. Features local file-based fallback when MLflow unavailable, run management, metrics/parameters logging, and model artifact storage with metadata.
- **Backup/Restore System (August 2025):** Complete backup and restore scripts for models, configs, logs, and data. Features incremental backups, pre-restore validation, categorized restoration, and system snapshot creation with verification.
- **Windows Deployment Automation (August 2025):** Full Windows deployment with antivirus/firewall exception configuration, port setup, and comprehensive .bat runners for installation, services, evaluation, and one-click pipeline execution.
- **One-Click Pipeline (August 2025):** Complete automated pipeline runner executing scrape → features → predict → strict gate → export → eval → logs/daily workflow with fallback mechanisms and comprehensive logging.
- **Strict 80% Confidence Gate (August 2025):** Ultra-strict orchestration filtering with `strict_toplist()` function ensuring only >= 80% confidence predictions pass through, with horizon-specific filtering and comprehensive gate status reporting.
- **Monte Carlo Dropout Inference (August 2025):** Bayesian uncertainty quantification for neural networks using MC Dropout with 30+ forward passes, confidence interval calculation, and uncertainty-aware prediction filtering.
- **Regime Detection HMM (August 2025):** Hidden Markov Model-based market regime classification (bear/neutral/bull) with fallback volatility/trend analysis, regime statistics calculation, and adaptive trading strategy support.
- **Daily Logging Bundler (August 2025):** Ultra-compact daily metrics logging system with automatic bundling, compression, cleanup, and timestamped file management including convenience functions for different log types.

## External Dependencies

### Core Python Libraries
- `streamlit`
- `pandas`, `numpy`
- `plotly`
- `threading`, `pathlib`, `psutil`

### Machine Learning Stack
- `scikit-learn`
- `xgboost`, `lightgbm`
- `numba`

### Trading and Market Data
- `ccxt`
- `textblob`

### Optional Performance Libraries
- `cupy`

### System Utilities
- `logging`, `json`, `pickle`
- `smtplib`
- `Pydantic`
- `dependency-injector`
- `FastAPI`

### External Services
- OpenAI API
- Cryptocurrency Exchanges: Kraken, Binance, KuCoin, Huobi (via CCXT)
- Email SMTP
- HashiCorp Vault
- Prometheus