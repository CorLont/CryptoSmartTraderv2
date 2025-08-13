# CryptoSmartTrader V2 - Quick Start Guide

🚀 **Enterprise-grade cryptocurrency trading intelligence system** designed to spot fast-growing cryptocurrencies early and achieve minimum 500% returns within 6 months.

## 🎯 Key Features

- **Multi-Agent Intelligence**: 8+ specialized agents for market analysis, sentiment tracking, and arbitrage detection
- **Enterprise Security**: Complete secrets management, log sanitization, and compliance monitoring
- **ML Model Registry**: Walk-forward training with canary deployment and drift detection
- **Risk Management**: Progressive escalation system with kill-switch capabilities
- **Real-time Analytics**: Prometheus monitoring with 24/7 observability
- **Exchange Integration**: Kraken, Binance, KuCoin support with API rate limiting
- **Backtest-Live Parity**: <20 bps/day tracking error validation

## ⚡ Quick Installation

### Prerequisites

- Python 3.11+ 
- UV package manager
- Git
- 8GB+ RAM
- Stable internet connection

### 1-Minute Setup

```bash
# Clone repository
git clone <repository-url>
cd CryptoSmartTrader

# Install with UV (recommended)
uv sync

# Set environment variables
cp .env.example .env
# Edit .env with your API keys

# Start all services
uv run python start_replit_services.py
```

### Windows Batch Installation

```cmd
# Run the 3-script installation system
1_install_all_dependencies.bat
2_start_background_services.bat  
3_start_dashboard.bat
```

## 🔑 Required API Keys

Add these to your `.env` file:

```env
# Exchange APIs (required)
KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_SECRET=your_kraken_secret

# AI/ML APIs (optional but recommended)  
OPENAI_API_KEY=sk-your_openai_key

# Monitoring (optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

## 🌐 Access Points

After starting the system:

- **Main Dashboard**: http://localhost:5000
- **API Endpoint**: http://localhost:8001
- **Metrics**: http://localhost:8000/metrics
- **Health Check**: http://localhost:8001/health

## 🎮 Basic Usage

### 1. Dashboard Overview

The main dashboard provides:
- Real-time market analysis
- ML model predictions  
- Portfolio performance
- Risk metrics
- Agent status monitoring

### 2. API Integration

```python
import httpx

# Health check
response = httpx.get("http://localhost:8001/health")
print(response.json())

# Get market predictions
predictions = httpx.get("http://localhost:8001/api/v1/predictions")
print(predictions.json())

# Get portfolio status
portfolio = httpx.get("http://localhost:8001/api/v1/portfolio")
print(portfolio.json())
```

### 3. Agent Management

```python
# Check agent status
agents = httpx.get("http://localhost:8001/api/v1/agents/status")

# Start specific agent
httpx.post("http://localhost:8001/api/v1/agents/start", 
          json={"agent_name": "technical_agent"})

# Emergency stop all agents
httpx.post("http://localhost:8001/api/v1/emergency/stop_all")
```

## 📊 Performance Targets

- **Returns**: Minimum 500% within 6 months
- **Uptime**: 99.5% availability SLO
- **Tracking Error**: <20 bps/day vs backtest
- **Latency**: <1s API response time
- **Coverage**: 470+ cryptocurrency pairs

## 🛡️ Security Features

- **Secrets Management**: Encrypted storage with auto-rotation
- **Log Sanitization**: 17 protection rules for PII/credentials
- **Access Control**: Role-based permissions with audit trail
- **Compliance**: Exchange ToS monitoring and enforcement
- **Emergency Procedures**: Kill-switch and revocation capabilities

## 🔧 Configuration

### Production Mode

```bash
# Enable production mode
export ENVIRONMENT=production
export DATA_INTEGRITY_MODE=strict

# Start with production configuration
uv run python start_replit_services.py --environment=production
```

### Development Mode

```bash
# Development with demo data
export ENVIRONMENT=development
export DEMO_MODE=true

# Start development services
uv run python start_replit_services.py --demo
```

## 📈 Monitoring

### Prometheus Metrics

Key metrics to monitor:

- `trading_signals_received_total`: Signal generation rate
- `trading_orders_sent_total`: Order execution rate  
- `trading_equity_usd`: Portfolio value
- `trading_drawdown_pct`: Risk level
- `api_request_duration_seconds`: System performance

### Health Checks

```bash
# System health
curl http://localhost:8001/health

# Agent health
curl http://localhost:8001/api/v1/agents/health

# Database health  
curl http://localhost:8001/api/v1/database/health

# Exchange connectivity
curl http://localhost:8001/api/v1/exchanges/health
```

## 🚨 Emergency Procedures

### Kill Switch Activation

```python
# Emergency stop all trading
httpx.post("http://localhost:8001/api/v1/emergency/kill_switch")

# Emergency portfolio liquidation
httpx.post("http://localhost:8001/api/v1/emergency/liquidate_all")
```

### Manual Agent Restart

```bash
# Restart all agents
uv run python start_agents.py --restart-all

# Restart specific agent
uv run python start_agents.py --restart technical_agent
```

## 🧪 Testing

### Run Test Suite

```bash
# Full test suite
uv run pytest

# Quick smoke tests
uv run pytest -m "not slow"

# Security tests
uv run pytest -m security

# Performance tests
uv run pytest -m performance
```

### Validate Installation

```bash
# System validation
uv run python -m cryptosmarttrader.validation.system_check

# Exchange connectivity test
uv run python -m cryptosmarttrader.validation.exchange_test

# ML model validation
uv run python -m cryptosmarttrader.validation.model_test
```

## 📚 Documentation

- **Operations Manual**: `README_OPERATIONS.md`
- **API Documentation**: `http://localhost:8001/docs`
- **Security Policy**: `SECURITY.md`
- **Change Log**: `CHANGELOG.md`
- **Architecture Guide**: `docs/ARCHITECTURE.md`

## 🆘 Support

### Common Issues

1. **Service won't start**: Check API keys in `.env`
2. **High latency**: Verify internet connection and exchange APIs
3. **Memory issues**: Ensure 8GB+ RAM available
4. **Permission errors**: Run with appropriate user permissions

### Getting Help

- **Documentation**: Check `README_OPERATIONS.md` for detailed procedures
- **Logs**: Check `logs/` directory for error details
- **Health Status**: Monitor `http://localhost:8001/health`
- **Metrics**: Review Prometheus metrics at `:8000/metrics`

### Emergency Contact

For critical production issues:
- **Kill Switch**: Use emergency procedures above
- **System Recovery**: Follow `README_OPERATIONS.md` recovery procedures
- **Data Issues**: Validate with zero-tolerance data integrity policy

## 🔄 Updates & Releases

### Update System

```bash
# Update to latest version
git pull origin main
uv sync
uv run python scripts/migrate_database.py
```

### Version Information

```bash
# Check current version
uv run python -c "from cryptosmarttrader import __version__; print(__version__)"

# Check component versions  
curl http://localhost:8001/api/v1/version
```

---

**🎯 Goal**: Achieve 500% returns within 6 months through enterprise-grade cryptocurrency intelligence.

**⚠️ Risk Notice**: Cryptocurrency trading involves significant risk. Past performance does not guarantee future results.

**📋 Compliance**: This system enforces strict data integrity and exchange ToS compliance.