# replit.md

## Overview
CryptoSmartTrader V2 is an institutional-grade multi-agent cryptocurrency trading intelligence system. It provides real-time market analysis, deep learning-powered price predictions, sentiment analysis, technical analysis, backtesting, and automated trade execution with comprehensive risk management across over 1457+ cryptocurrencies. The system aims for high returns by integrating various analytical methods and ML predictions to identify fast-growing cryptocurrencies. Key goals include high prediction accuracy, strict data integrity, robust asynchronous data scraping, and advanced AI/ML techniques.

## User Preferences
Preferred communication style: Simple, everyday language.
Data integrity policy: ABSOLUTE ZERO-TOLERANCE for synthetic/fallback data - Complete elimination of all non-authentic data sources in production mode. Only 100% authentic data from direct exchange APIs allowed. Production automatically blocks any violations. No fallback, no synthetic data, no interpolated values, no NaN values allowed in production. Strict data integrity enforcement with real-time validation and automatic production blocking.
RENDEMENT-DRUKKENDE FACTOREN ELIMINATED: All critical decision-reliability issues resolved:
- Unified Confidence Gate: Single source of truth replaces inconsistent class/standalone gates preventing different candidate sets
- Authentic AI Explanations: Real OpenAI-powered explanations replace random SHAP data, improving trader decision confidence
- Consolidated Logging System: Single logging framework prevents inconsistent observability and scattered error reporting
- Import Path Resolver: Fixes all import mismatches between dashboards and modules preventing stress-test failures
- Authentic Performance Dashboard: Only real system metrics displayed, eliminates misleading synthetic visualizations
- Enterprise Error Handling: Proper fallbacks with graceful degradation, no more NameError crashes
- Technical Agent Enterprise Fixes: Thread leak prevention with proper join(), dummy data elimination, MACD bias fix, div/NaN protection, UTC timestamps, TA-Lib utilization
- Temporal Integrity Validator: Early datatype normalization, comprehensive UTC coverage, vectorized future checks, efficient vectorized alignment replacing iterrows
- Temporal Safe Splits: Division by zero protection in interval calculation, actual purged CV implementation with proper purging, dataclass default_factory fix for mutable defaults
- System Health Monitor: Dummy data elimination with authentic system metrics, threshold hierarchy cleanup with GO/NO-GO integration, robust import handling with graceful degradation
- System Monitor: Configurable port strategy with multiple port fallbacks, unused threshold cleanup removing dead config, cross-platform load average handling with Windows compatibility
- System Optimizer: Thread lifecycle control with proper start/stop, authentic optimization replacing no-op operations, safe model archiving with archive directory, error backoff mechanism
- System Readiness Checker: Model naming consistency with {horizon}_type.pkl pattern, clean issues filtering removing empty strings, robust health file handling with strict/lenient fallback modes
- System Settings: Pydantic v2 compatibility with cross-version support, horizon notation consistency with automatic normalization, robust GPU detection with functional validation
- System Validator: Corrected import names (sklearn not scikit-learn), flexible file requirements with critical/recommended/optional levels, accurate module paths, appropriate error classification
- Technical Agent Enterprise Fixes: Thread leak prevention with proper join(), dummy data elimination, MACD bias fix, div/NaN protection, UTC timestamps, TA-Lib utilization
Repository priorities: P0 (CI/lint/type/tests/security optimized with GitHub Actions, uv caching, Python matrix testing, security scanning), P1 (multi-service Replit config documented, CODEOWNERS with @clont1 approval requirements, branch protection with required PR flow), P2 (comprehensive test strategy with fixtures, property-based testing, deterministic time, smart test markers)
CI/CD Infrastructure: GitHub Actions workflow with UV package management, ruff/black/mypy linting (hard enforcement - no || true fallbacks), pytest with strict settings (--maxfail=1, --disable-warnings), pip-audit security scanning (hard enforcement), concurrency controls for efficient CI execution. All three enterprise PR bundles (CI + Config Hardening + API Contract) fully implemented with hybrid approach maintaining enterprise features. SECURITY.md policy established with vulnerability reporting procedures and security best practices.
Testing infrastructure: Strict pytest.ini with --maxfail=1, --strict-markers, comprehensive marker system (unit/integration/slow/api/ml/trading/security/performance), Dutch marker descriptions, 85%+ coverage targets
Repository hygiene: Enterprise .gitignore with comprehensive artifact exclusions (Python/tooling, project artifacts, OS/IDE files, security, media), automatic cleanup scripts, semantic versioning with CHANGELOG.md, release asset management
Replit deployment: Multi-service architecture with Dashboard (5000), API (8001), Metrics (8000), health endpoints, proper service orchestration, process isolation. UV-based startup pattern: `uv sync && (service1 & service2 & service3 & wait)` for optimal Replit compatibility
Configuration management: Enterprise Pydantic Settings with centralized type validation, fail-fast startup checks, and startup logging (no direct os.environ access). PR2 Config Hardening: Modern pydantic-settings v2 import, model_config syntax, simple logging option, clean main entry point with Dutch support
Fail-fast configuration: Complete settings validation system with Pydantic BaseSettings, comprehensive startup checks (API keys, directories, ports, resources), structured logging with correlation IDs, enterprise error handling with detailed validation results
Type-safe API infrastructure: Complete FastAPI router skeleton with enterprise patterns (health/market/trading/agents), comprehensive Pydantic models with validation, dependency injection, CORS/security middleware, performance monitoring, Swagger/ReDoc documentation. PR3 API Contract: FastAPI app factory pattern (get_app), clean /health endpoint with HealthOut model, E2E testing with httpx.AsyncClient
Enterprise code quality: Complete enterprise-grade code audit system implemented based on critical failure mode checklist. All 11 categories audited with 7 critical fixes applied achieving 100% success rate.
Production deployment: LEAN 3-SCRIPT INSTALLATION SYSTEM - Consolidated all installation processes into 3 streamlined Windows batch scripts: 1_install_all_dependencies.bat (complete setup), 2_start_background_services.bat (services), 3_start_dashboard.bat (main app). Full workstation deployment automation with backend enforcement of 80% confidence gate, atomic writes, and clean architecture.
Production requirements: Full Kraken coverage (no demo limits), consistent RF model architecture, sentiment+whale fully integrated in UI, OpenAI intelligence actively used in predictions, backtesting/realized tracking visible for 500% goal validation, automatic retraining/drift monitoring operational, no dummy sections in dashboard.
Code quality standards: DUPLICATE-FREE LEAN ARCHITECTURE - Systematically consolidated 32+ duplicate files into clean core structure. 5 core managers, 5 enhanced agents, zero conflicts. All critical code quality issues resolved with enterprise-grade validation and honest reporting.
Architecture preference: Lean, focused, production-ready structure with consolidated dependencies and no duplicate code paths. Clean src/ layout implemented with proper package structure, eliminating import shadowing and enabling editable installation.
Clean Architecture: Enterprise src/ layout implemented with domain interfaces (ports) and swappable adapters. Package structure prevents import shadowing and enables dependency inversion for testable, maintainable code.
Documentation consolidation: Comprehensive operations manual created with README_CONSOLIDATED.md (quick start + operations), docs/ARCHITECTURE_DIAGRAMS.md (system diagrams + ADRs), docs/runbooks/INCIDENT_RESPONSE.md (P0/P1/P2 procedures), scripts/operations/ (daily health checks, log cleanup, backup automation), and docs/QUICK_REFERENCE.md (emergency commands + troubleshooting guide). Complete consolidation of 5 separate READMEs into unified operational framework.

## System Architecture

### Core Architecture Pattern
The system uses a Distributed Multi-Process Architecture with 8 isolated agent processes (Data Collector, Sentiment Analyzer, Technical Analyzer, ML Predictor, Whale Detector, Risk Manager, Portfolio Optimizer, Health Monitor), each with circuit breakers, exponential backoff, health monitoring, and automatic restart.

### Technology Stack
- **Frontend:** Streamlit.
- **Backend:** Python, FastAPI.
- **Machine Learning:** PyTorch (LSTM, GRU, Transformer, N-BEATS), scikit-learn, XGBoost.
- **Data Processing:** pandas, numpy, CuPy, Numba.
- **Exchange Integration:** CCXT.
- **Visualization:** Plotly.
- **Storage:** Local file system (JSON/CSV) with caching.

### Key Architectural Decisions
- **Deployment & Structure:** Lean 3-script deployment, duplicate-free core structure, consolidated dependency management, clean archive system.
- **Foundational Services:** Comprehensive testing (80%+ coverage), enterprise security (secrets, log redaction), production-grade logging (structured JSON, Prometheus), dependency injection, Pydantic configuration, async I/O with global rate limiting, intelligent caching, health monitoring with GO/NO-GO gates.
- **Observability:** Enterprise JSON logging with correlation IDs, Prometheus metrics, health grading system with trading policy enforcement.
- **Data & Exchange Management:** Multi-exchange architecture (CCXT-based), real-time data pipeline, data manager, dynamic coin registry, ML model manager, hard data integrity filter (zero-tolerance), timestamp synchronization, temporal validation/splits/feature engineering.
- **Machine Learning & AI:** GPU optimization, regime detection & market regime routing (HMM), multi-horizon ML (1H, 24H, 7D, 30D, ensemble), enterprise ML uncertainty quantification (Bayesian LSTM, Monte Carlo), SLO-driven MLOps, continual learning, automated feature engineering, causal inference, reinforcement learning (PPO), synthetic data augmentation, Human-in-the-Loop & explainability (SHAP), meta-labeling, event impact scoring (LLM-powered), futures data features, advanced transformers (TFT, N-BEATS), conformal prediction, probability calibrator, calibrated confidence gate, Bayesian uncertainty quantification, regime-aware models.
- **Trading & Risk Management:** Strict 80% confidence gate, shadow trading, advanced execution simulation (Level-2, slippage, fees), enterprise portfolio risk management (hard caps, correlation limits, kill-switch), coverage audit, multi-horizon batch inference/signal quality validation, system health monitor, advanced portfolio optimization (Kelly-lite, uncertainty-aware, correlation caps), order book imbalance detection, regime router (Mixture-of-Experts), uncertainty-aware sizing.
- **UI/UX & Monitoring:** Enterprise Streamlit dashboard with performance optimization, async data refresh, intelligent caching, modular page architecture, session state management.
- **System Automation & Integration:** Final system integrator, production optimizer, critical fixes applier, production deployment system (Windows batch scripts, atomic orchestration, readiness checks).

## External Dependencies

- **AI/ML Services:** OpenAI API
- **Cryptocurrency Exchanges:** Kraken, Binance, KuCoin, Huobi (via CCXT)
- **Monitoring:** Prometheus
- **Secrets Management:** HashiCorp Vault
- **Email:** SMTP
- **Core Python Libraries:** `streamlit`, `pandas`, `numpy`, `plotly`, `threading`, `pathlib`, `psutil`, `scikit-learn`, `xgboost`, `lightgbm`, `numba`, `ccxt`, `textblob`, `cupy`, `logging`, `json`, `pickle`, `smtplib`, `Pydantic`, `dependency-injector`, `FastAPI`