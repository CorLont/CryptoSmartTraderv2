# CryptoSmartTrader V2

Een geavanceerd multi-agent cryptocurrency trading intelligence systeem met institutionele analysecapaciteiten.

## 🚀 Quick Start (60 seconden)

### Replit Deployment (Aanbevolen)
```bash
# Automatische multi-service startup met UV
uv sync && (uv run python api/health_endpoint.py & uv run python metrics/metrics_server.py & uv run streamlit run app_fixed_all_issues.py --server.port 5000 --server.headless true --server.address 0.0.0.0 & wait)
```

### Lokale Development
```bash
# 1. Environment configuratie
cp .env.example .env
# Voeg je API keys toe: KRAKEN_API_KEY, OPENAI_API_KEY

# 2. Start alle services
python start_multi_service.py

# 3. Toegang tot het systeem
# Dashboard: http://localhost:5000
# API: http://localhost:8001/api/docs  
# Metrics: http://localhost:8000/metrics
```

### Service Status Controle
```bash
# Snelle health check
python test_replit_services.py

# Uitgebreide dagelijkse controle
python scripts/operations/daily_health_check.py
```

## 🎯 Enterprise Features

### Multi-Agent Intelligence Platform (8 Agents)
- **📊 Data Collector:** Real-time marktdata van 471+ cryptocurrencies
- **💭 Sentiment Agent:** AI-powered news en social media analyse  
- **📈 Technical Agent:** 50+ technische indicatoren met TA-Lib
- **🤖 ML Predictor:** Deep learning voorspellingen (5 tijdhorizonten)
- **🐋 Whale Detector:** Detectie van grote transacties en market movers
- **⚖️ Risk Manager:** Enterprise risicobeheer met 80% confidence gates
- **💼 Portfolio Optimizer:** Kelly criterion + uncertainty-aware allocatie
- **🏥 Health Monitor:** Real-time systeemmonitoring met GO/NO-GO gates

### Enterprise Architecture & Operations
- **Zero-Tolerance Data Integrity:** Alleen authentieke data, geen fallbacks
- **Process Isolation:** Complete agent scheiding voor fault tolerance  
- **UV-based Deployment:** 10x snellere dependency management
- **Prometheus Metrics:** Enterprise monitoring met SLOs en alerting
- **Comprehensive Runbooks:** Incident response, backup/recovery procedures
- **Architecture Decision Records:** Gedocumenteerde technische beslissingen

### Production Ready Features
- **🔄 Multi-Service Architecture:** Dashboard (5000), API (8001), Metrics (8000)
- **📋 Operational Excellence:** Dagelijkse health checks, log cleanup, backup systeem
- **🚨 Incident Response:** P0/P1/P2 procedures met escalatie protocols
- **📊 SLO Monitoring:** 99.5% beschikbaarheid, <2s response times
- **🛡️ Security:** API key management, secure logging, audit trails

## 📋 Operations Manual

### Daily Operations
```bash
# Dagelijkse health check (automatisch rapport)
python scripts/operations/daily_health_check.py

# Log cleanup (behoudt 30 dagen)
python scripts/operations/cleanup_logs.py --days 30

# System backup
python scripts/operations/backup_system.py

# Service restart (indien nodig)
python start_multi_service.py
```

### Incident Response (Zie: docs/runbooks/INCIDENT_RESPONSE.md)
```bash
# P0 Critical: System down
pkill -f streamlit && python start_multi_service.py

# P1 High: Service degraded  
python scripts/diagnose_errors.py --last-hour

# P2 Medium: Performance issues
python scripts/performance_profile.py --optimize
```

### Monitoring & Metrics
- **Health Status:** http://localhost:8001/health/detailed
- **Prometheus Metrics:** http://localhost:8000/metrics  
- **Service Logs:** logs/app.log (geroteerd dagelijks)
- **Performance Dashboard:** Streamlit interface met real-time metrics

### Architecture & Documentation
- **System Architecture:** docs/ARCHITECTURE_DIAGRAMS.md
- **Decision Records:** docs/ADR_RECORDS.md  
- **Complete Operations Guide:** README_CONSOLIDATED.md
- **Multi-Service Config:** REPLIT_MULTI_SERVICE_CONFIG.md
- `SENTIMENT_ANALYSIS_ENABLED` - Enable sentiment analysis
- `ML_PREDICTION_ENABLED` - Enable ML predictions

## 🧪 Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=./
```

## 🔧 Development

### Code Quality
Pre-commit hooks are configured for:
- Black code formatting
- isort import sorting  
- flake8 linting
- Type checking with mypy

Install pre-commit:
```bash
pre-commit install
```

### Adding New Features
1. Follow the dependency injection pattern
2. Add proper type hints
3. Write tests in the `tests/` directory
4. Update documentation

## 📈 Monitoring

### Health Checks
The system provides A-F health grading:
- **A (90-100%):** Excellent - All systems optimal
- **B (80-89%):** Good - Minor issues
- **C (70-79%):** Fair - Some performance degradation  
- **D (60-69%):** Poor - Significant issues
- **F (<60%):** Critical - Major problems

### Metrics
When Prometheus is enabled, metrics are available at `http://localhost:8000`:
- Request counts and latency
- Agent performance scores
- Data freshness indicators
- Error rates and health scores

### Logging  
Structured JSON logs are written to the `logs/` directory:
- `cryptotrader.json` - General application logs
- `errors.json` - Error-specific logs

## 🔒 Security

### Secret Management  
- Environment variables for configuration
- Optional HashiCorp Vault integration
- Input validation on all endpoints
- Sanitization to prevent injection attacks

### API Security
- CORS configuration for web integration
- Request validation with Pydantic models
- Structured error responses
- Rate limiting and timeout protection

## 🎛️ Advanced Configuration

### Vault Integration
For production secret management:
```bash
VAULT_ENABLED=true
VAULT_ADDR=https://vault.example.com
VAULT_TOKEN=your_vault_token
```

### GPU Acceleration
Enable GPU processing for ML models:
```bash
ENABLE_GPU=true
```

### Custom Agents
Extend the system by adding custom agents to the `agents/` directory following the existing patterns.

## 📋 System Requirements

- Python 3.11+
- Memory: 4GB+ recommended
- Storage: 2GB+ for data and models
- Network: Internet connection for market data

## 🆘 Troubleshooting

### Common Issues

**Import Errors:**
- Check that all dependencies are installed
- Verify Python version compatibility

**API Connection Issues:**  
- Verify your API keys are correct
- Check network connectivity
- Review logs for detailed error messages

**Performance Issues:**
- Monitor memory usage in health dashboard
- Check cache hit ratios in metrics
- Consider enabling GPU acceleration

### Support
Check the logs in the `logs/` directory for detailed error information. The health dashboard provides real-time system status and performance metrics.

---

**Version:** 2.0.0  
**Last Updated:** August 2025  
**Architecture:** Production-ready with Dutch architectural requirements implemented