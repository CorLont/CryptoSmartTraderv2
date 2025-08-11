# CryptoSmartTrader V2 - Quick Reference

## 🚀 Essential Commands

### Start/Stop System
```bash
# Start all services (Replit optimized)
uv sync && (uv run python api/health_endpoint.py & uv run python metrics/metrics_server.py & uv run streamlit run app_fixed_all_issues.py --server.port 5000 --server.headless true --server.address 0.0.0.0 & wait)

# Alternative startup
python start_multi_service.py

# Emergency stop
pkill -f "streamlit" && pkill -f "uvicorn"

# Graceful restart
python scripts/graceful_restart.py
```

### Health Checks
```bash
# Quick health check
python test_replit_services.py

# Comprehensive daily check
python scripts/operations/daily_health_check.py

# Service-specific checks
curl http://localhost:5000/_stcore/health  # Dashboard
curl http://localhost:8001/health          # API
curl http://localhost:8000/health          # Metrics
```

### System Maintenance
```bash
# Clean logs (keep 30 days)
python scripts/operations/cleanup_logs.py --days 30

# System backup
python scripts/operations/backup_system.py

# Performance optimization
python scripts/performance_optimizer.py --full

# Update system settings
python scripts/update_settings.py
```

## 🔧 Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| **Dashboard** | http://localhost:5000 | Main trading interface |
| **API Health** | http://localhost:8001/health | Service health status |
| **API Docs** | http://localhost:8001/api/docs | Interactive API documentation |
| **Metrics** | http://localhost:8000/metrics | Prometheus metrics |
| **Detailed Health** | http://localhost:8001/health/detailed | Comprehensive system status |

## 📊 Key Metrics & Thresholds

### Service Level Objectives (SLOs)
- **Dashboard Availability:** >99.5%
- **API Response Time:** <100ms
- **Error Rate:** <1%
- **Health Score:** >80%

### Critical Thresholds
```bash
# Check current values
curl -s http://localhost:8001/health/detailed | jq '.application_health.health_score'
curl -s http://localhost:8000/metrics | grep cryptotrader_portfolio_value
```

### Alert Levels
- **P0 Critical:** System down, health score <60
- **P1 High:** Service degraded, error rate >5%
- **P2 Medium:** Performance issues, response time >5s
- **P3 Low:** Minor issues, non-critical warnings

## 🚨 Emergency Procedures

### System Down (P0)
```bash
# 1. Check processes
ps aux | grep -E "(streamlit|uvicorn|python)"

# 2. Check resources
df -h && free -h

# 3. Emergency restart
pkill -f "streamlit" && pkill -f "uvicorn"
python start_multi_service.py

# 4. Verify recovery
python test_replit_services.py
```

### High Error Rate (P1)
```bash
# 1. Check recent errors
tail -100 logs/app.log | grep ERROR

# 2. Check service health
curl http://localhost:8001/health/detailed

# 3. Restart failing services
python scripts/restart_unhealthy_services.py
```

### Performance Issues (P2)
```bash
# 1. Check resources
python -c "import psutil; print(f'CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%')"

# 2. Enable performance mode
python scripts/enable_performance_mode.py

# 3. Clear caches
python scripts/clear_caches.py
```

## 📂 Important File Locations

### Configuration
- `.env` - Environment variables and API keys
- `config.json` - System configuration
- `replit.md` - Project documentation and preferences
- `pyproject.toml` - Python dependencies

### Logs
- `logs/app.log` - Main application log
- `logs/agents/` - Individual agent logs
- `logs/system/` - System health logs
- `logs/api/` - API service logs

### Data
- `cache/` - Cached market data
- `models/` - ML model files
- `data/` - Processed data files
- `backups/` - System backups

### Scripts
- `scripts/operations/` - Operational scripts
- `scripts/maintenance/` - Maintenance utilities
- `test_*.py` - Testing and validation scripts

## 🔍 Troubleshooting

### Common Issues

#### Port Already in Use
```bash
# Find process using port
lsof -i :5000
# Kill specific process
kill -9 <PID>
```

#### Service Won't Start
```bash
# Check dependencies
uv sync
# Check configuration
python scripts/validate_config.py
```

#### High Memory Usage
```bash
# Check memory by process
ps aux --sort=-%mem | head -10
# Clear caches
python scripts/clear_caches.py --all
```

#### Data Issues
```bash
# Validate data integrity
python scripts/validate_data.py
# Refresh market data
python scripts/refresh_market_data.py
```

## 📈 Performance Optimization

### Quick Optimizations
```bash
# Enable performance mode
export PERFORMANCE_MODE=true

# Increase cache limits
export CACHE_SIZE_MB=1000

# Reduce update frequency
export UPDATE_INTERVAL_SECONDS=5
```

### Advanced Optimizations
```bash
# Full system optimization
python scripts/system_optimizer.py --full

# Model optimization
python scripts/optimize_models.py --prune

# Database optimization
python scripts/optimize_database.py
```

## 🔐 Security

### API Key Management
```bash
# Check API key status
python scripts/check_api_keys.py

# Rotate API keys
python scripts/rotate_api_keys.py

# Validate permissions
python scripts/validate_permissions.py
```

### Security Audit
```bash
# Run security scan
python scripts/security_audit.py

# Check for vulnerabilities
python scripts/vulnerability_scan.py

# Validate access controls
python scripts/access_control_check.py
```

## 📞 Support & Documentation

### Documentation Hierarchy
1. **This Quick Reference** - Immediate actions and commands
2. **README_CONSOLIDATED.md** - Complete operations guide
3. **docs/runbooks/INCIDENT_RESPONSE.md** - Detailed incident procedures
4. **docs/ARCHITECTURE_DIAGRAMS.md** - System architecture
5. **docs/ADR_RECORDS.md** - Architecture decisions

### Getting Help
1. Check system health: `python test_replit_services.py`
2. Review recent logs: `tail -50 logs/app.log`
3. Run diagnostics: `python scripts/diagnose_system.py`
4. Consult runbooks: `docs/runbooks/`
5. Check GitHub issues for known problems

---

**Last Updated:** 2025-01-11  
**Version:** 2.0.0  
**Environment:** Multi-Service Architecture