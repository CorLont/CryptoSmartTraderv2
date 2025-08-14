# CryptoSmartTrader V2 - Complete Operations Guide

## 🚀 Quick Start

### Prerequisites
- **Python 3.10+**
- **8GB+ RAM** (16GB recommended)  
- **Windows 10/11** or **Linux/macOS**
- **Internet connection**
- **NVIDIA GPU** (optional, for CUDA acceleration)

### 1-Minute Setup
```bash
# Clone and enter directory
git clone <repository_url>
cd CryptoSmartTrader-V2

# Copy environment template  
cp .env.example .env
# Add your API keys to .env (Kraken, OpenAI)

# Start with UV (Recommended for Replit)
uv sync && (uv run python api/health_endpoint.py & uv run python metrics/metrics_server.py & uv run streamlit run app_fixed_all_issues.py --server.port 5000 --server.headless true --server.address 0.0.0.0 & wait)

# Alternative: Python script
python start_multi_service.py
```

### Access Points
- **Main Dashboard:** http://localhost:5000
- **API Documentation:** http://localhost:8001/api/docs  
- **System Metrics:** http://localhost:8000/metrics
- **Health Status:** http://localhost:8001/health/detailed

---

## 🏗️ System Architecture

### Multi-Agent Intelligence Platform
```
┌─────────────────────────────────────────────────────────────┐
│                    CryptoSmartTrader V2                    │
├─────────────────────────────────────────────────────────────┤
│  Frontend Layer: Streamlit Dashboard (Port 5000)           │
├─────────────────────────────────────────────────────────────┤
│  API Layer: FastAPI Health & Status (Port 8001)            │
├─────────────────────────────────────────────────────────────┤
│  Monitoring: Prometheus Metrics (Port 8000)                │
├─────────────────────────────────────────────────────────────┤
│                   Agent Architecture                       │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │  Sentiment  │ │ Technical   │ │ ML Predictor│          │
│  │   Agent     │ │   Agent     │ │   Agent     │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │    Whale    │ │    Risk     │ │ Portfolio   │          │
│  │  Detector   │ │  Manager    │ │ Optimizer   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                     Data Layer                             │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐          │
│  │   Kraken    │ │   Market    │ │   ML Model  │          │
│  │     API     │ │    Data     │ │   Storage   │          │
│  └─────────────┘ └─────────────┘ └─────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

### Core Components
- **Distributed Multi-Process Architecture:** 8 isolated agent processes
- **Circuit Breakers:** Automatic failure handling with exponential backoff
- **Health Monitoring:** Comprehensive system monitoring with GO/NO-GO gates
- **Async I/O:** High-performance concurrent operations
- **Type Safety:** Full Pydantic validation and type hints

---

## 📊 Operations Manual

### Key Performance Metrics & SLOs

#### Service Level Objectives (SLOs)
| Service | Availability | Response Time | Error Rate |
|---------|-------------|---------------|------------|
| Dashboard | 99.5% | < 2s page load | < 1% |
| API Health | 99.9% | < 100ms | < 0.1% |
| Metrics | 99.5% | < 500ms | < 1% |
| Data Pipeline | 98% | < 5s refresh | < 5% |

#### Critical Metrics
```bash
# System Health Score (Target: > 80%)
curl http://localhost:8001/health/detailed | jq '.application_health.health_score'

# Trading Status
curl http://localhost:8001/health/detailed | jq '.application_health.trading_status'

# Active Trades Count
curl http://localhost:8000/metrics | grep cryptotrader_active_trades

# Portfolio Value
curl http://localhost:8000/metrics | grep cryptotrader_portfolio_value_usd
```

### Log Management & Rotation

#### Log Locations
```
logs/
├── app.log              # Main application logs
├── agents/              # Individual agent logs
├── api/                 # API service logs  
├── metrics/             # Metrics server logs
├── system/              # System health logs
└── archived/            # Rotated logs
```

#### Automatic Log Rotation
```bash
# Daily rotation (configured in logging setup)
python -c "from core.logging_manager import LoggingManager; LoggingManager().rotate_logs()"

# Manual cleanup (keeps last 30 days)
python scripts/cleanup_logs.py --days 30

# Archive old logs
python scripts/archive_logs.py --compress --days 7
```

---

## 🚨 Incident Response Runbook

### Priority Levels
- **P0 (Critical):** System down, data loss risk
- **P1 (High):** Service degraded, user impact
- **P2 (Medium):** Performance issues
- **P3 (Low):** Minor issues, maintenance

### P0: System Down

#### Symptom: Dashboard Not Accessible
```bash
# 1. Check process status
ps aux | grep -E "(streamlit|uvicorn|python)"

# 2. Check port availability  
netstat -tulpn | grep -E ":5000|:8001|:8000"

# 3. Kill conflicting processes
pkill -f streamlit && pkill -f uvicorn

# 4. Restart services
python start_multi_service.py

# 5. Verify health
curl http://localhost:5000/_stcore/health
curl http://localhost:8001/health
curl http://localhost:8000/health
```

#### Symptom: Database/Storage Corruption
```bash
# 1. Check disk space
df -h

# 2. Verify data integrity
python -c "from core.data_manager import DataManager; dm = DataManager(); print(dm.verify_integrity())"

# 3. Restore from backup
python scripts/backup_restore.py restore backups/latest_backup.zip

# 4. Restart with clean state
python start_multi_service.py --reset
```

### P1: Service Degraded

#### Symptom: High Error Rate (>5%)
```bash
# 1. Check system health score
curl http://localhost:8001/health/detailed | jq '.application_health.health_score'

# 2. Review recent errors
tail -100 logs/app.log | grep ERROR

# 3. Check agent status
python -c "from orchestration.distributed_orchestrator import DistributedOrchestrator; o = DistributedOrchestrator(); print(o.get_agent_status())"

# 4. Restart failing agents
python scripts/restart_agents.py --unhealthy-only
```

#### Symptom: ML Model Performance Degradation
```bash
# 1. Check drift detection
python -c "from core.drift_detection import DriftDetectionSystem; d = DriftDetectionSystem(); print(d.get_alerts())"

# 2. Review prediction accuracy
python -c "from ml.model_manager import ModelManager; mm = ModelManager(); print(mm.get_performance_metrics())"

# 3. Force model retraining
python scripts/retrain_models.py --force --horizon all

# 4. Fallback to paper trading if needed
python -c "from core.auto_disable_system import AutoDisableSystem; a = AutoDisableSystem(); a.force_disable('model_degradation')"
```

### P2: Performance Issues

#### Symptom: Slow Response Times (>5s)
```bash
# 1. Check system resources
python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%')"

# 2. Analyze slow queries/operations
grep -i "slow\|timeout\|delay" logs/app.log | tail -20

# 3. Enable performance optimization
python -c "from core.performance_optimizer import PerformanceOptimizer; po = PerformanceOptimizer(); po.optimize_system()"

# 4. Restart with optimized configuration
python start_multi_service.py --optimize
```

### Recovery Procedures

#### Complete System Recovery
```bash
# 1. Stop all services
pkill -f "streamlit\|uvicorn\|python.*health\|python.*metrics"

# 2. Clean temporary files
rm -rf logs/temp/ cache/temp/ .streamlit/

# 3. Restore from backup
python scripts/backup_restore.py restore backups/latest_stable.zip

# 4. Verify environment
python scripts/environment_check.py

# 5. Start services in order
python api/health_endpoint.py &
sleep 2
python metrics/metrics_server.py &
sleep 2  
streamlit run app_fixed_all_issues.py --server.port 5000 &

# 6. Verify health
python test_replit_services.py
```

#### Rollback Procedure
```bash
# 1. Identify last known good state
python scripts/backup_manager.py list --stable-only

# 2. Stop services gracefully
python scripts/graceful_shutdown.py

# 3. Rollback to previous version
python scripts/rollback.py --version <stable_version>

# 4. Verify configuration
python scripts/config_validator.py

# 5. Restart and test
python start_multi_service.py
python test_replit_services.py
```

---

## 🏛️ Architecture Decision Records (ADRs)

### ADR-001: Multi-Agent Architecture
**Date:** 2025-01-11  
**Status:** Accepted  

**Context:** Need for scalable, maintainable cryptocurrency analysis system.

**Decision:** Implement distributed multi-agent architecture with process isolation.

**Rationale:**
- **Scalability:** Independent agent scaling
- **Fault Tolerance:** Process isolation prevents cascade failures  
- **Maintainability:** Clear separation of concerns
- **Performance:** Parallel processing capabilities

**Consequences:**
- **Positive:** High availability, easy debugging, modular development
- **Negative:** Increased complexity, inter-process communication overhead

---

### ADR-002: UV-based Dependency Management
**Date:** 2025-01-11  
**Status:** Accepted

**Context:** Need for fast, reliable dependency management in Replit environment.

**Decision:** Use UV package manager with background process coordination.

**Rationale:**
- **Speed:** UV is significantly faster than pip
- **Reliability:** Better dependency resolution
- **Replit Compatibility:** Optimal for Replit's environment
- **Development Experience:** Faster iteration cycles

**Consequences:**
- **Positive:** Faster builds, better dependency management, Replit optimization
- **Negative:** Learning curve, newer tool with evolving ecosystem

---

### ADR-003: Multi-Service Port Strategy
**Date:** 2025-01-11  
**Status:** Accepted

**Context:** Need for service isolation and monitoring in constrained environment.

**Decision:** Use ports 5000 (Dashboard), 8001 (API), 8000 (Metrics).

**Rationale:**
- **Port 5000:** Replit's default webview port for main application
- **Port 8001:** Non-conflicting port for API services
- **Port 8000:** Standard Prometheus metrics port
- **Isolation:** Each service has dedicated port for health monitoring

**Consequences:**
- **Positive:** Clear service separation, standard compliance, monitoring capability
- **Negative:** Port management complexity, potential conflicts with other services

---

### ADR-004: Confidence Gating System
**Date:** 2025-01-11  
**Status:** Accepted

**Context:** Need for risk management in trading decisions.

**Decision:** Implement 80% confidence threshold with GO/NO-GO trading gates.

**Rationale:**
- **Risk Management:** Prevents trading on uncertain predictions
- **Quality Assurance:** Maintains high prediction standards
- **Regulatory Compliance:** Auditable decision making process
- **Performance:** Improves overall system accuracy

**Consequences:**
- **Positive:** Reduced risk, better performance metrics, compliance
- **Negative:** Fewer trading opportunities, complex threshold management

---

### ADR-005: Streamlit as Primary UI Framework
**Date:** 2025-01-11  
**Status:** Accepted

**Context:** Need for rapid development of data-rich trading interface.

**Decision:** Use Streamlit for main dashboard with custom optimization.

**Rationale:**
- **Development Speed:** Rapid prototyping and iteration
- **Data Visualization:** Excellent integration with Plotly/pandas
- **Python Integration:** Native Python ecosystem compatibility
- **Deployment:** Simple deployment with headless mode

**Consequences:**
- **Positive:** Fast development, great data viz, Python-native
- **Negative:** Limited UI customization, performance constraints for complex UIs

---

## 🔧 Maintenance Procedures

### Daily Operations
```bash
# Morning health check
python scripts/daily_health_check.py

# Log cleanup
python scripts/cleanup_logs.py --days 7

# Performance metrics review
curl http://localhost:8000/metrics | grep -E "error_rate|response_time|health_score"
```

### Weekly Operations
```bash
# Full system backup
python scripts/backup_system.py --full

# Model performance review
python scripts/model_performance_report.py --week

# Security audit
python scripts/security_audit.py

# Dependency updates check
uv sync --dry-run
```

### Monthly Operations
```bash
# Full system optimization
python scripts/system_optimizer.py --full

# Archive old data
python scripts/archive_data.py --month

# Performance trend analysis
python scripts/performance_trends.py --month

# Documentation update
python scripts/update_documentation.py
```

---

## 📈 Monitoring & Alerting

### Key Dashboards
1. **System Health:** Health score, uptime, error rates
2. **Trading Performance:** Portfolio value, active trades, P&L
3. **Technical Metrics:** API response times, memory usage, CPU
4. **Business Metrics:** Prediction accuracy, confidence scores

### Alert Thresholds
```yaml
critical:
  health_score: < 60
  error_rate: > 10%
  response_time: > 10s
  disk_space: < 1GB

warning:
  health_score: < 80
  error_rate: > 5%
  response_time: > 5s
  disk_space: < 5GB
```

---

## 🚀 Deployment Scenarios

### Development Environment
```bash
# Local development with hot reload
streamlit run app_fixed_all_issues.py --server.port 5000
```

### Staging Environment  
```bash
# Staging with full services
python start_multi_service.py --environment staging
```

### Production Environment
```bash
# Production with monitoring
uv sync && (uv run python api/health_endpoint.py & uv run python metrics/metrics_server.py & uv run streamlit run app_fixed_all_issues.py --server.port 5000 --server.headless true --server.address 0.0.0.0 & wait)
```

### Replit Deployment
```bash
# Optimized for Replit
export REPLIT_DEPLOYMENT=true
python start_replit_services.py
```

---

*For additional support, consult the individual component documentation in the `/docs` directory or check the GitHub issues for known problems and solutions.*