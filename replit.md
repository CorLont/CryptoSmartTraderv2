# replit.md

## Overview
CryptoSmartTrader V2 is a sophisticated multi-agent cryptocurrency trading intelligence system designed for professional institutional-grade analysis. It analyzes over 1457+ cryptocurrencies using a coordinated ensemble of specialized AI agents, providing real-time market analysis, deep learning-powered price predictions, sentiment analysis, technical analysis, backtesting, and automated trade execution with comprehensive risk management. The system aims for high returns by combining technical indicators, sentiment analysis, whale detection, and ML predictions, focusing on comprehensive analysis for detecting fast-growing cryptocurrencies.

The enterprise architecture goals include higher prediction accuracy, zero dummy-data tolerance, strict confidence gating, robust async scraping, modern ML/AI techniques (uncertainty, regime, ensembles), and daily evaluation with GO/NOGO decisions. Key capabilities include zero synthetic data tolerance, mandatory deep learning, cross-coin feature fusion, true async processing, enterprise security, Bayesian uncertainty quantification, dynamic coin discovery, advanced AI/ML, multi-agent cooperation, model monitoring, black swan simulation, and real-time order book data integration.

## User Preferences
Preferred communication style: Simple, everyday language.
Data integrity policy: ZERO-TOLERANCE for incomplete data - coins with missing sentiment/on-chain/technical data are HARD BLOCKED from training and display. Only authentic data from real sources with 80%+ completeness allowed. No fallback, no synthetic data, no incomplete coins passed through.

## System Architecture

### Core Architecture Pattern
The system employs a Distributed Multi-Process Architecture, managing 8 isolated agent processes (Data Collector, Sentiment Analyzer, Technical Analyzer, ML Predictor, Whale Detector, Risk Manager, Portfolio Optimizer, and Health Monitor) with circuit breakers, exponential backoff, health monitoring, and automatic restart capabilities. Each agent runs in complete process isolation to eliminate single points of failure.

### Technology Stack
- **Frontend:** Streamlit.
- **Backend:** Python.
- **Machine Learning:** PyTorch (LSTM, GRU, Transformer, N-BEATS), scikit-learn, XGBoost.
- **Data Processing:** pandas, numpy, CuPy, Numba.
- **Exchange Integration:** CCXT.
- **Visualization:** Plotly.
- **Storage:** Local file system with JSON/CSV and caching.

### Key Architectural Decisions
- **Comprehensive Testing Framework:** Enterprise pytest suite with 80%+ coverage.
- **Enterprise Security Framework:** Secrets management with Vault integration, log redaction.
- **Production-Grade Logging & Monitoring:** Structured JSON logging with correlation IDs, Prometheus metrics.
- **Dependency Injection Container:** Full dependency-injector implementation.
- **Pydantic Configuration Management:** Type-safe settings with environment variable support.
- **Async I/O Architecture:** Complete async/await implementation with aiohttp, global rate limiting.
- **Universal Configuration Management:** Centralized with validation and backup.
- **Intelligent Cache Manager:** Memory-aware caching with TTL and cleanup.
- **Health Monitoring System:** Continuous checks with alerts and GO/NO-GO gates.
- **Multi-Exchange Architecture:** CCXT-based abstraction with rate limiting and failover.
- **Real-time Data Pipeline:** Parallel processing with rate limiting, timeouts, and validation.
- **GPU Optimization:** Automatic GPU/CPU detection and adaptive tuning.
- **Regime Detection & Market Regime Routing:** HMM-based and rule-based classification for dynamic model routing.
- **Multi-Horizon ML System:** Training and inference across 1H, 24H, 7D, and 30D horizons with ensemble modeling.
- **Dashboard Architecture:** Five specialized Streamlit dashboards.
- **Data Management:** Centralized Data Manager, Dynamic Coin Registry, ML Model Manager.
- **Enterprise ML Uncertainty Quantification:** Bayesian LSTM with Monte Carlo sampling and confidence intervals.
- **Strict 80% Confidence Gate System:** Enterprise-grade filtering in orchestration and dashboard.
- **SLO-Driven ML Operations:** Formal SLO monitoring with thresholds, automatic retraining triggers, and model rollback.
- **Continual Learning & Meta-Learning:** Automated retraining with drift detection, online learning.
- **Automated Feature Engineering:** Auto-featuretools, deep feature synthesis.
- **Causal Inference:** Double Machine Learning, Granger Causality.
- **Reinforcement Learning:** PPO for dynamic, risk-aware portfolio allocation.
- **Self-Healing System:** Autonomous monitoring, black swan detection, and auto-disabling/recovery.
- **Synthetic Data Augmentation:** Generation of black swan, regime shift scenarios for stress testing.
- **Human-in-the-Loop & Explainability:** Active learning, feedback-driven learning, and SHAP for model transparency.
- **Shadow Trading:** Full paper trading engine with realistic market simulation for pre-production model validation.
- **Advanced Execution Simulation:** Level-2 order book simulation with market impact, partial fills, exchange-specific latency/fees.
- **Enterprise Portfolio Risk Management:** Per-coin hard caps, correlation limits, automated position kill-switch.
- **Coverage Audit System:** Daily automated audits ensuring 99%+ exchange coverage with alerts.
- **Hard Data Integrity Filter:** Zero-tolerance policy for incomplete data; coins with missing data are hard blocked.
- **Multi-Horizon Batch Inference Engine:** Unified batch processing for all coins across all horizons with atomic operations.
- **Multi-Horizon Signal Quality Validator:** Enterprise signal validation system with defined metrics.
- **System Health Monitor with GO/NO-GO Gates:** Enterprise health scoring system determining live trading authorization.
- **Enterprise Meetscripts Suite:** Operational measurement toolkit (`strict_filter.py`, `coverage_audit.py`, `evaluator.py`, `calibration.py`, `health_score.py`).
- **Complete Operational Playbook:** Full enterprise workflow automation (`nightly_batch.py`, `post_batch_validation.py`, `dashboard_gate_checker.py`, `shadow_to_live_validator.py`).
- **Drift Detection System:** Complete drift detection with error trending analysis and distribution monitoring.
- **Fine-Tune Scheduler with EWC:** Automated fine-tuning system with replay buffer and Elastic Weight Consolidation.
- **Auto-Disable System:** Trading safety system that automatically disables live trading based on health scores.
- **Drift-Fine-Tune-Auto-Disable Integration:** Complete integration system for drift monitoring, fine-tuning, and trading mode management.
- **L2 Orderbook Simulator:** Complete Level-2 order book simulation with realistic market microstructure.
- **Slippage Estimator:** Real-time slippage estimation system with percentile calculations.
- **Paper Trading Engine:** Mandatory 4-week paper trading validation system with complete execution logging.
- **MLflow Manager:** Complete MLflow integration for model tracking, versioning, and registry.
- **Backup/Restore System:** Complete backup and restore scripts for models, configs, logs, and data.
- **Windows Deployment Automation:** Full Windows deployment with antivirus/firewall exception configuration.
- **One-Click Pipeline:** Complete automated pipeline runner executing scrape → features → predict → strict gate → export → eval → logs/daily workflow.
- **Monte Carlo Dropout Inference:** Bayesian uncertainty quantification for neural networks using MC Dropout.
- **Regime Detection HMM:** Hidden Markov Model-based market regime classification.
- **Daily Logging Bundler:** Ultra-compact daily metrics logging system with automatic bundling and compression.
- **Enterprise Risk Mitigation:** Comprehensive risk mitigation system covering data gaps, overfitting, GPU bottlenecks, and complexity management.
- **Data Completeness Gate:** Zero-tolerance data quality gate that hard-blocks coins with insufficient data completeness.

## External Dependencies

- **AI/ML Services:** OpenAI API
- **Cryptocurrency Exchanges:** Kraken, Binance, KuCoin, Huobi (via CCXT)
- **Monitoring:** Prometheus
- **Secrets Management:** HashiCorp Vault
- **Email:** SMTP
- **Core Python Libraries:** `streamlit`, `pandas`, `numpy`, `plotly`, `threading`, `pathlib`, `psutil`, `scikit-learn`, `xgboost`, `lightgbm`, `numba`, `ccxt`, `textblob`, `cupy`, `logging`, `json`, `pickle`, `smtplib`, `Pydantic`, `dependency-injector`, `FastAPI`