# replit.md

## Overview
CryptoSmartTrader V2 is a multi-agent cryptocurrency trading intelligence system designed for professional institutional-grade analysis. It analyzes over 1457+ cryptocurrencies, providing real-time market analysis, deep learning-powered price predictions, sentiment analysis, technical analysis, backtesting, and automated trade execution with comprehensive risk management. The system aims for high returns by combining technical indicators, sentiment analysis, whale detection, and ML predictions, focusing on comprehensive analysis for detecting fast-growing cryptocurrencies.

The enterprise architecture goals include higher prediction accuracy, zero dummy-data tolerance, strict confidence gating, robust async scraping, modern ML/AI techniques (uncertainty, regime, ensembles), and daily evaluation with GO/NOGO decisions. Key capabilities include enterprise security, Bayesian uncertainty quantification, dynamic coin discovery, advanced AI/ML, multi-agent cooperation, model monitoring, black swan simulation, and real-time order book data integration.

## User Preferences
Preferred communication style: Simple, everyday language.
Data integrity policy: ABSOLUTE ZERO-TOLERANCE for synthetic/fallback data - Complete elimination of all non-authentic data sources in production mode. Only 100% authentic data from direct exchange APIs allowed. Production automatically blocks any violations. No fallback, no synthetic data, no interpolated values, no NaN values allowed in production. Strict data integrity enforcement with real-time validation and automatic production blocking.
Enterprise code quality: Complete enterprise-grade code audit system implemented based on critical failure mode checklist. All 11 categories audited with 7 critical fixes applied achieving 100% success rate.

## System Architecture

### Core Architecture Pattern
The system employs a Distributed Multi-Process Architecture, managing 8 isolated agent processes (Data Collector, Sentiment Analyzer, Technical Analyzer, ML Predictor, Whale Detector, Risk Manager, Portfolio Optimizer, and Health Monitor) with circuit breakers, exponential backoff, health monitoring, and automatic restart capabilities. Each agent runs in complete process isolation.

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
- **Enterprise Security Framework:** Secrets management, log redaction.
- **Production-Grade Logging & Monitoring:** Structured JSON logging, Prometheus metrics.
- **Dependency Injection Container:** Full dependency-injector implementation.
- **Pydantic Configuration Management:** Type-safe settings with environment variable support.
- **Async I/O Architecture:** Complete async/await implementation, global rate limiting.
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
- **Coverage Audit System:** Daily automated audits ensuring 99%+ exchange coverage.
- **Hard Data Integrity Filter:** Zero-tolerance policy for incomplete data; coins with missing data are hard blocked.
- **Multi-Horizon Batch Inference Engine:** Unified batch processing for all coins across all horizons with atomic operations.
- **Multi-Horizon Signal Quality Validator:** Enterprise signal validation system with defined metrics.
- **System Health Monitor with GO/NO-GO Gates:** Enterprise health scoring system determining live trading authorization.
- **Enterprise Meetscripts Suite:** Operational measurement toolkit.
- **Complete Operational Playbook:** Full enterprise workflow automation.
- **Drift Detection System:** Complete drift detection with error trending analysis and distribution monitoring.
- **Fine-Tune Scheduler with EWC:** Automated fine-tuning system with replay buffer and Elastic Weight Consolidation.
- **Auto-Disable System:** Trading safety system that automatically disables live trading based on health scores.
- **L2 Orderbook Simulator:** Complete Level-2 order book simulation.
- **Slippage Estimator:** Real-time slippage estimation system.
- **Paper Trading Engine:** Mandatory 4-week paper trading validation.
- **MLflow Manager:** Complete MLflow integration for model tracking, versioning, and registry.
- **Backup/Restore System:** Complete backup and restore scripts.
- **One-Click Pipeline:** Complete automated pipeline runner.
- **Monte Carlo Dropout Inference:** Bayesian uncertainty quantification for neural networks.
- **Regime Detection HMM:** Hidden Markov Model-based market regime classification.
- **Daily Logging Bundler:** Ultra-compact daily metrics logging system.
- **Enterprise Risk Mitigation:** Comprehensive risk mitigation system covering data gaps, overfitting, GPU bottlenecks, and complexity management.
- **Data Completeness Gate:** Zero-tolerance data quality gate.
- **Workstation Optimizer:** Complete workstation optimization for i9-32GB-RTX2000 setup.
- **Daily Health Dashboard:** Centralized daily health monitoring.
- **Improved Logging Manager:** Fixed correlation_id issues with simplified logging configuration.
- **Final System Integrator:** Complete production deployment automation.
- **Production Optimizer:** Advanced performance optimizations.
- **Real-Time Monitor:** Intelligent real-time monitoring with threshold-based alerting.
- **Advanced Analytics:** Comprehensive analytics engine generating actionable insights.
- **Comprehensive System Manager:** Ultimate system integration managing all advanced features.
- **Critical Fixes Applier:** Automatic application of fixes for timestamp validation, calibration, slippage modeling, and security issues.
- **Meta-Labeling System:** Lopez de Prado triple-barrier method for signal quality filtering.
- **Event Impact Scoring:** LLM-powered news analysis with impact scoring.
- **Futures Data Features:** Funding rates, open interest, and basis features.
- **Advanced Transformers:** Temporal Fusion Transformer (TFT) and N-BEATS ensemble for superior multi-horizon forecasting.
- **Conformal Prediction:** Formal uncertainty quantification with adaptive confidence intervals.
- **Advanced Portfolio Optimization:** Kelly-lite position sizing with uncertainty awareness, correlation caps, and hard risk overlays.
- **Order Book Imbalance Detection:** L2 depth analysis with bid/ask imbalance and spoof detection.
- **Regime Router (Mixture-of-Experts):** Best model per regime routing.
- **Uncertainty-Aware Sizing:** Kelly-lite position sizing with uncertainty awareness, correlation caps, and liquidity constraints.
- **Temporal Integrity Validation:** Complete protection against look-ahead bias and data leakage.
- **Timestamp Synchronization System:** UTC candle boundary alignment across all agents.
- **Temporal Validation System:** Complete multi-agent timestamp validation.
- **Temporal Safe Splits:** Time-series aware train/test splitting.
- **Temporal Feature Engineering:** Safe temporal feature creation.
- **Async Scraping Framework:** High-performance concurrent scraping.
- **Concurrent Data Collector:** Multi-source data collection with async/await.
- **Probability Calibrator:** Advanced calibration methods.
- **Calibrated Confidence Gate:** Enterprise confidence gating with properly calibrated probabilities.
- **Bayesian Uncertainty Quantification:** Monte Carlo Dropout, ensemble modeling, and epistemic/aleatoric uncertainty decomposition.
- **Market Regime Detection:** Hidden Markov Model and rule-based classification.
- **Regime-Aware Models:** Specialized models for various regimes.
- **Integrated Regime-Aware Confidence System:** Complete integration of regime detection, calibrated confidence gates, and Bayesian uncertainty quantification.
- **Realistic Orderbook Simulator:** Level-2 orderbook simulation with market impact, slippage estimation, partial fills, and exchange-specific latency modeling.
- **Advanced Backtesting Engine:** Comprehensive backtester with fees, slippage, partial fills, latency, and liquidity constraints eliminating optimistic bias.
- **OpenAI Enhanced Intelligence:** Complete GPT-4o integration for advanced sentiment analysis, news impact assessment, intelligent feature engineering, anomaly detection, and AI-enhanced trading insights with full system integration and Dutch language support.

## External Dependencies

- **AI/ML Services:** OpenAI API
- **Cryptocurrency Exchanges:** Kraken, Binance, KuCoin, Huobi (via CCXT)
- **Monitoring:** Prometheus
- **Secrets Management:** HashiCorp Vault
- **Email:** SMTP
- **Core Python Libraries:** `streamlit`, `pandas`, `numpy`, `plotly`, `threading`, `pathlib`, `psutil`, `scikit-learn`, `xgboost`, `lightgbm`, `numba`, `ccxt`, `textblob`, `cupy`, `logging`, `json`, `pickle`, `smtplib`, `Pydantic`, `dependency-injector`, `FastAPI`