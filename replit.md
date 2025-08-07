# Replit.md

## Overview
CryptoSmartTrader V2 is a sophisticated multi-agent cryptocurrency trading intelligence system designed for professional institutional-grade analysis. The system analyzes over 453 cryptocurrencies using a coordinated ensemble of specialized AI agents that operate in parallel. The platform provides real-time market analysis, machine learning-powered price predictions, sentiment analysis, technical analysis, backtesting capabilities, and automated trade execution with comprehensive risk management.

**Recent Update (August 2025):** Fully implemented Dutch architectural requirements with enterprise-grade production enhancements including dependency injection with system orchestrator, Pydantic-based configuration validation, structured JSON logging with audit trails, comprehensive security management with input validation and rate limiting, CI/CD pipeline with automated quality gates, platform-independent deployment scripts, async HTTP clients with retry logic, comprehensive testing framework, Prometheus metrics, intelligent error handling with recovery strategies, adaptive rate limiting with priority support, real-time performance monitoring with automatic optimization, and smart configuration tuning based on system performance patterns.

**Latest Production Enhancement (August 2025):** Successfully addressed all critical architectural analysis points with enterprise-grade implementations: system orchestrator with workflow management, comprehensive security framework, structured logging with multiple channels, type-safe configuration validation, automated CI/CD pipeline, and platform-independent deployment capabilities.

**Daily Analysis Integration (August 2025):** Implemented comprehensive Daily Analysis Scheduler that coordinates continuous ML analysis and social media scraping for automated daily reporting. Features scheduled analysis at 6:00, 12:00, 18:00, and 23:30 with continuous 15-minute ML updates, 5-minute social sentiment monitoring, and 30-minute technical analysis cycles. Includes integrated dashboard controls, real-time progress tracking, daily summary generation with market trends and sentiment analysis, and automated report export functionality.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Core Architecture Pattern
The system employs a multi-agent coordination architecture with six specialized AI agents operating in parallel:
- **Sentiment Analysis Agent:** Processes social media and news sentiment using TextBlob and optional OpenAI integration
- **Technical Analysis Agent:** Calculates 50+ technical indicators using the `ta` library
- **ML Predictor Agent:** Utilizes ensemble machine learning models (XGBoost, LightGBM, scikit-learn) for price forecasting across multiple horizons (1h to 30d)
- **Backtest Agent:** Validates trading strategies against historical data with comprehensive performance metrics
- **Trade Executor Agent:** Generates trading signals and manages risk with configurable stop-loss and take-profit levels
- **Whale Detector Agent:** Monitors large transactions and unusual trading activities

### Technology Stack
- **Frontend:** Streamlit for web-based dashboards with interactive visualizations
- **Backend:** Python with multi-threaded agent coordination
- **Machine Learning:** scikit-learn, XGBoost, LightGBM ensemble models with automated hyperparameter tuning
- **Data Processing:** pandas, numpy with vectorized calculations, CuPy for optional GPU acceleration
- **Exchange Integration:** CCXT library for multi-exchange connectivity (Kraken, Binance, KuCoin, Huobi)
- **Visualization:** Plotly for interactive charts and real-time data visualization
- **Storage:** Local file system with JSON/CSV exports and intelligent caching

### Key Architectural Decisions
- **Universal Configuration Management:** Centralized configuration system with automatic validation, backup/rollback capabilities, and default value management
- **Intelligent Cache Manager:** Memory-aware caching with TTL management, automatic cleanup, and type-specific optimization for pandas/numpy objects
- **Health Monitoring System:** Continuous health checks with A-F grading, real-time alerts, and comprehensive system metrics monitoring
- **Multi-Exchange Architecture:** CCXT-based exchange abstraction layer with rate limiting, health monitoring, and failover capabilities
- **Real-time Data Pipeline:** Parallel processing with global API rate limiting, timeout protection, and intelligent retry logic
- **GPU Optimization:** Automatic GPU/CPU detection with dynamic load monitoring and adaptive parameter tuning using CuPy and Numba
- **Enterprise Logging:** Structured logging with file rotation, multiple output channels, and configurable log levels
- **Thread-Safe Design:** All components use threading locks for concurrent access and data consistency
- **Daily Analysis Coordination:** Automated scheduler integrating continuous ML analysis and social scraping with scheduled daily reporting, progress tracking, and export capabilities

### Dashboard Architecture
Three specialized Streamlit dashboards provide different views:
- **Main Dashboard:** System overview, market analysis, and agent performance monitoring
- **Agent Dashboard:** Individual agent monitoring, configuration, and performance analytics
- **Portfolio Dashboard:** Portfolio management, risk metrics, position tracking, and trade history

### Data Management
- **Centralized Data Manager:** Coordinates data collection from multiple exchanges with automatic freshness monitoring
- **Coin Registry:** Dynamic cryptocurrency discovery and management system with metadata caching
- **ML Model Manager:** Handles model training, ensemble management, and performance tracking across multiple prediction horizons

## External Dependencies

### Core Python Libraries
- **streamlit:** Web application framework for dashboards
- **pandas:** Data manipulation and analysis
- **numpy:** Numerical computing
- **plotly:** Interactive visualization
- **threading:** Multi-threaded agent coordination
- **pathlib:** File system operations
- **psutil:** System resource monitoring

### Machine Learning Stack
- **scikit-learn:** Machine learning algorithms and preprocessing
- **xgboost:** Gradient boosting framework
- **lightgbm:** Gradient boosting framework
- **numba:** JIT compilation for performance optimization

### Trading and Market Data
- **ccxt:** Multi-exchange trading library for API connectivity
- **textblob:** Natural language processing for sentiment analysis

### Optional Performance Libraries
- **cupy:** GPU-accelerated computing (optional)
- **openai:** Advanced sentiment analysis (optional, requires API key)

### System Utilities
- **logging:** Comprehensive logging system
- **json:** Configuration and data serialization
- **pickle:** Model serialization and caching
- **smtplib:** Email alerts and notifications

### External Services
- **OpenAI API:** Advanced sentiment analysis (optional)
- **Cryptocurrency Exchanges:** Live market data via CCXT (Kraken, Binance, KuCoin, Huobi)
- **Email SMTP:** Alert notifications system
- **HashiCorp Vault:** Secure secret management (optional)
- **Prometheus:** Metrics collection and monitoring

## Production Enhancements (August 2025)

### Dependency Injection Architecture
- **ApplicationContainer:** Centralized dependency management using dependency-injector library
- **Explicit Dependencies:** All components receive dependencies through constructor injection
- **Testable Design:** Easy mocking and testing with injected dependencies

### Enhanced Configuration Management
- **Pydantic Settings:** Type-safe configuration with environment variable support
- **Secret Management:** HashiCorp Vault integration with fallback to environment variables
- **Validation:** Automatic configuration validation and type checking

### Async I/O & Resilience
- **AsyncHTTPClient:** Asynchronous HTTP client with exponential backoff retry logic
- **Concurrent Processing:** Multi-threaded agent coordination with rate limiting
- **Error Handling:** Comprehensive error handling with structured logging

### Testing & Quality Assurance
- **Pytest Framework:** Comprehensive unit and integration tests
- **Pre-commit Hooks:** Automated code quality checks (Black, isort, flake8)
- **Type Hints:** Complete type annotation throughout codebase
- **Coverage Reports:** Test coverage monitoring and reporting

### Input Validation & Security
- **Pydantic Models:** Strict input validation for all API endpoints
- **Sanitization:** Input sanitization to prevent injection attacks
- **Error Responses:** Structured error handling with proper HTTP status codes

### Monitoring & Observability
- **Prometheus Metrics:** System performance and health metrics
- **Structured Logging:** JSON-formatted logs with rotation and categorization
- **Health Checks:** Comprehensive system health monitoring with A-F grading
- **OpenAPI Documentation:** Automatic API documentation generation

### REST API Layer
- **FastAPI Integration:** High-performance async API server
- **Type Safety:** Full request/response validation with Pydantic models
- **Background Tasks:** Long-running operations handled asynchronously
- **CORS Support:** Cross-origin resource sharing for web integration