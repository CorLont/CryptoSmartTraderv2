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

## Implementation Progress

**DEEP LEARNING & SEQUENCE MODELING (August 2025):**
- **Deep Learning Engine:** LSTM, GRU, Transformer, N-BEATS as standard components for complex non-linear pattern detection ✅ IMPLEMENTED
- **Multimodal Data Processor:** Text, time series, and graph feature processing for deep neural networks ✅ IMPLEMENTED
- **Advanced Sequence Models:** Bidirectional LSTM with attention, Transformer encoders with positional encoding ✅ IMPLEMENTED
- **Multimodal Fusion:** Cross-modal attention mechanisms combining sentiment, price, whale, and regime data ✅ IMPLEMENTED
- **Neural Architecture:** Self-attention, residual connections, layer normalization for enterprise-grade deep learning ✅ IMPLEMENTED

**UNCERTAINTY MODELING & PROBABILISTIC OUTPUT (August 2025):**
- **Uncertainty Engine:** Bayesian neural networks, quantile regression, ensemble spread for prediction confidence ✅ IMPLEMENTED
- **Probabilistic Trader:** Uncertainty-aware trading decisions with confidence thresholds ✅ IMPLEMENTED
- **Bayesian Neural Networks:** Variational inference for epistemic uncertainty quantification ✅ IMPLEMENTED
- **Quantile Regression:** Prediction intervals and aleatoric uncertainty modeling ✅ IMPLEMENTED
- **Monte Carlo Dropout:** Dropout-based uncertainty estimation for model uncertainty ✅ IMPLEMENTED
- **Ensemble Methods:** Model disagreement for uncertainty quantification ✅ IMPLEMENTED
- **Risk-Aware Position Sizing:** Dynamic position sizing based on prediction uncertainty ✅ IMPLEMENTED

**CONTINUAL LEARNING & META-LEARNING (August 2025):**
- **Continual Learning Engine:** Automated retraining with drift detection and catastrophic forgetting prevention ✅ IMPLEMENTED
- **Meta-Learning Coordinator:** Model-agnostic meta-learning (MAML) for fast adaptation to new coins/markets ✅ IMPLEMENTED
- **Drift Detection System:** Statistical drift detection for concept drift, covariate shift, and performance degradation ✅ IMPLEMENTED
- **Online Learning:** Real-time model updates with rehearsal buffers and elastic weight consolidation ✅ IMPLEMENTED
- **Automated Retraining:** Fully automated retrain schema with priority-based task scheduling ✅ IMPLEMENTED
- **Few-Shot Learning:** Rapid adaptation to new cryptocurrencies with minimal training data ✅ IMPLEMENTED
- **Knowledge Distillation:** Teacher-student framework for preserving learned knowledge during updates ✅ IMPLEMENTED

**AUTOMATED FEATURE ENGINEERING & DISCOVERY (August 2025):**
- **Automated Feature Engineering:** Auto-featuretools, deep feature synthesis, auto-crosses with attention-based feature pruning ✅ IMPLEMENTED
- **Feature Discovery Engine:** Continuous feature discovery and optimization with genetic algorithms ✅ IMPLEMENTED
- **SHAP Regime Analyzer:** Advanced SHAP analysis with live regime-specific feature adaptation ✅ IMPLEMENTED
- **Live Feature Adaptation:** Real-time feature set adjustment based on market regime changes ✅ IMPLEMENTED
- **Deep Feature Synthesis:** Temporal features, cross features, polynomial features with technical indicators ✅ IMPLEMENTED
- **Feature Importance Analysis:** SHAP, permutation, tree-based, mutual information methods ✅ IMPLEMENTED
- **Regime-Specific Feature Sets:** Dynamic feature selection optimized for each market regime ✅ IMPLEMENTED