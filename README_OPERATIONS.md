# CryptoSmartTrader V2 - Operations & Runbook

> 📋 **Production Operations Manual**  
> Comprehensive guide for incident response, system monitoring, and operational procedures.

## 🏥 System Health & Monitoring

### Health Grading System

The system uses a 5-tier health grading system with automatic policy enforcement:

#### **A-Grade (Excellent): 90-100%**
- ✅ All agents operational
- ✅ Data latency < 30 seconds
- ✅ Prediction accuracy > 75%
- ✅ No critical errors in 24h
- **Trading Policy**: Full automated trading enabled

#### **B-Grade (Good): 80-89%**  
- ✅ Core agents operational
- ⚠️ Minor performance degradation
- ✅ Prediction accuracy > 70%
- ✅ < 5 non-critical errors/hour
- **Trading Policy**: Automated trading with reduced position sizes

#### **C-Grade (Fair): 70-79%**
- ⚠️ Some agents degraded
- ⚠️ Data latency 30-60 seconds
- ⚠️ Prediction accuracy 65-70%
- ⚠️ 5-15 errors/hour
- **Trading Policy**: Shadow trading only - no real positions

#### **D-Grade (Poor): 60-69%**
- 🚨 Multiple agent failures
- 🚨 Data latency > 60 seconds  
- 🚨 Prediction accuracy < 65%
- 🚨 > 15 errors/hour
- **Trading Policy**: TRADING SUSPENDED - monitoring only

#### **F-Grade (Failing): < 60%**
- 🚨 System critical failure
- 🚨 Multiple service outages
- 🚨 Data pipeline failures
- **Trading Policy**: KILL-SWITCH ACTIVATED - all trading stopped

### GO/NO-GO Decision Matrix

```
┌──────────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│ Health Grade │    A    │    B    │    C    │    D    │    F    │
├──────────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ Auto Trading │   ✅    │   ✅    │   ❌    │   ❌    │   ❌    │
│ Position Size│  100%   │  75%    │   0%    │   0%    │   0%    │
│ New Entries  │   ✅    │   ✅    │   ❌    │   ❌    │   ❌    │
│ Exits Only   │   ❌    │   ❌    │   ✅    │   ✅    │   ✅    │
│ Kill Switch  │   ❌    │   ❌    │   ❌    │   ❌    │   ✅    │
└──────────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
```

## 🚨 Incident Response Procedures

### P0 - Critical (Response: < 15 minutes)
**System down, trading halted, data loss risk**

#### Immediate Actions:
```bash
# 1. Check system status
curl http://localhost:8001/health
curl http://localhost:8000/metrics

# 2. Check service status
ps aux | grep -E "(streamlit|uvicorn|python.*health|python.*metrics)"

# 3. Check logs for critical errors
tail -f logs/system_health.log | grep -i "critical\|error\|failed"

# 4. Activate kill-switch if needed
curl -X POST http://localhost:8001/kill-switch/activate
```

#### Escalation:
1. **0-15 min**: On-call engineer response
2. **15-30 min**: Senior engineer + system architect
3. **30-60 min**: Management notification
4. **60+ min**: Client notification required

### P1 - High (Response: < 1 hour)
**Degraded performance, some features unavailable**

#### Actions:
```bash
# 1. Identify affected components
uv run python -c "
from cryptosmarttrader.core import SystemHealth
health = SystemHealth()
status = health.get_detailed_status()
print(f'Overall Grade: {status[\"grade\"]}')
for agent, health in status['agents'].items():
    if health['status'] != 'healthy':
        print(f'⚠️ {agent}: {health[\"status\"]} - {health[\"error\"]}')
"

# 2. Restart affected services
# Individual service restart
pkill -f "python.*health_endpoint" && uv run python api/health_endpoint.py &
pkill -f "python.*metrics_server" && uv run python metrics/metrics_server.py &

# 3. Monitor recovery
watch -n 5 "curl -s http://localhost:8001/health | jq '.health_grade'"
```

### P2 - Medium (Response: < 4 hours)
**Minor issues, system functional but suboptimal**

#### Actions:
- Monitor and document
- Schedule maintenance window if needed
- Update runbooks based on findings

## 🔧 Operational Procedures

### Daily Health Checks

**Morning Checklist (09:00 UTC)**
```bash
#!/bin/bash
# Daily health check script

echo "=== CryptoSmartTrader V2 Daily Health Check ==="
echo "Date: $(date)"
echo ""

# 1. Service Status
echo "1. Service Health:"
curl -s http://localhost:8001/health | jq '.'
echo ""

# 2. System Resources
echo "2. System Resources:"
echo "Memory: $(free -h | grep Mem | awk '{print $3"/"$2}')"
echo "Disk: $(df -h | grep -v tmpfs | awk 'NR==2{print $3"/"$2" ("$5")"}')"
echo "Load: $(uptime | awk -F'load average:' '{print $2}')"
echo ""

# 3. Trading Status
echo "3. Trading Status:"
curl -s http://localhost:8001/trading/status | jq '.status'
echo ""

# 4. Recent Errors
echo "4. Recent Critical Errors (last 24h):"
grep -c "CRITICAL\|ERROR" logs/system_health.log | tail -1
echo ""

# 5. Data Pipeline Health
echo "5. Data Pipeline:"
echo "Last data update: $(curl -s http://localhost:8001/data/status | jq -r '.last_update')"
echo "Active pairs: $(curl -s http://localhost:8001/data/status | jq '.active_pairs')"
```

### Weekly Maintenance

**Every Sunday 02:00 UTC**
```bash
#!/bin/bash
# Weekly maintenance script

# 1. Log rotation
find logs/ -name "*.log" -mtime +7 -exec gzip {} \;
find logs/ -name "*.log.gz" -mtime +30 -delete

# 2. Model backup
cp -r models/ backup/models_$(date +%Y%m%d)/

# 3. Database cleanup (if applicable)
# Clean old prediction records
# Vacuum/optimize storage

# 4. Performance metrics collection
uv run python scripts/weekly_performance_report.py

# 5. Dependency updates check
uv sync --upgrade-package
```

### Kill-Switch Procedures

The kill-switch is the ultimate safety mechanism that immediately halts all trading activities.

#### Automatic Activation Triggers:
- System health grade drops to F (< 60%)
- Data pipeline failure > 5 minutes
- Critical error rate > 50/minute
- Memory usage > 90%
- Disk space < 1GB

#### Manual Activation:
```bash
# Via API
curl -X POST http://localhost:8001/kill-switch/activate \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"reason": "Manual intervention", "operator": "admin"}'

# Via emergency script
uv run python scripts/emergency_stop.py --reason "market_volatility"
```

#### Kill-Switch Status Check:
```bash
curl http://localhost:8001/kill-switch/status
# Response: {"active": true/false, "activated_at": "timestamp", "reason": "..."}
```

#### Deactivation (Requires Manual Approval):
```bash
# 1. Verify system health
curl http://localhost:8001/health

# 2. Run system diagnostics
uv run python scripts/pre_activation_check.py

# 3. Deactivate if all clear
curl -X POST http://localhost:8001/kill-switch/deactivate \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"operator": "admin", "verified_checks": true}'
```

## 📊 Monitoring & Alerting

### Key Metrics to Monitor

**System Level:**
- CPU usage (alert if > 80% for 5 min)
- Memory usage (alert if > 85%)
- Disk space (alert if < 5GB free)
- Network latency to exchanges (alert if > 500ms)

**Application Level:**
- Health grade (alert if drops below B)
- Data lag (alert if > 60 seconds)
- Prediction accuracy (alert if < 65% daily average)
- Error rate (alert if > 10/minute)

**Trading Level:**
- Position sizes vs limits
- Daily P&L vs expectations
- Confidence gate violations
- Risk metric breaches

### Prometheus Metrics Endpoints

```bash
# System metrics
curl http://localhost:8000/metrics | grep cryptotrader_

# Key metrics:
# - cryptotrader_health_grade
# - cryptotrader_prediction_accuracy  
# - cryptotrader_data_lag_seconds
# - cryptotrader_active_positions
# - cryptotrader_daily_pnl
```

### Log Analysis

**Critical Log Locations:**
- `/logs/system_health.log` - System health and grades
- `/logs/trading.log` - Trading decisions and executions
- `/logs/data_pipeline.log` - Data collection and processing
- `/logs/ml_predictions.log` - ML model outputs and accuracy
- `/logs/errors.log` - Application errors and exceptions

**Log Analysis Commands:**
```bash
# Real-time critical errors
tail -f logs/errors.log | grep -i "critical\|fatal"

# Trading decision analysis
grep "TRADE_DECISION" logs/trading.log | tail -20

# Health grade changes
grep "HEALTH_GRADE_CHANGE" logs/system_health.log | tail -10

# Data pipeline issues
grep -E "(timeout|failed|error)" logs/data_pipeline.log | tail -20
```

## 🔄 Backup & Recovery

### Automated Backups

**Daily Backups (03:00 UTC):**
- Model files and parameters
- Configuration files
- Trading history and positions
- System logs (last 7 days)

**Weekly Backups (Sunday 04:00 UTC):**
- Complete database snapshot
- Full log archive
- Performance metrics history

### Recovery Procedures

**Service Recovery:**
```bash
# 1. Stop all services
pkill -f "streamlit\|uvicorn\|python.*health\|python.*metrics"

# 2. Restore from backup (if needed)
cp -r backup/models_latest/* models/
cp backup/config_latest.json config/

# 3. Restart services
uv sync && (uv run python api/health_endpoint.py & uv run python metrics/metrics_server.py & uv run streamlit run app_fixed_all_issues.py --server.port 5000 --server.headless true --server.address 0.0.0.0 & wait)

# 4. Verify recovery
curl http://localhost:8001/health
```

**Database Recovery:**
```bash
# Restore from backup
cp backup/trading_db_latest.json data/trading_database.json

# Verify data integrity
uv run python scripts/verify_data_integrity.py
```

## 📞 Emergency Contacts

**On-Call Rotation:**
- **Primary**: System Administrator (24/7)
- **Secondary**: Senior Engineer (business hours)
- **Escalation**: System Architect (critical issues)

**External Dependencies:**
- **Kraken API**: https://status.kraken.com/
- **OpenAI API**: https://status.openai.com/
- **Infrastructure**: Replit Status

## 🔍 Troubleshooting Guide

### Common Issues

**Issue: Services Won't Start**
```bash
# Check port conflicts
netstat -tlnp | grep -E "(5000|8001|8000)"

# Check permissions
ls -la logs/ models/ config/

# Check disk space
df -h

# Fix: Kill conflicting processes and restart
```

**Issue: High Memory Usage**
```bash
# Identify memory hogs
ps aux --sort=-%mem | head -10

# Check for memory leaks in agents
pgrep -f python | xargs -I {} cat /proc/{}/status | grep -E "(Name|VmRSS)"

# Fix: Restart memory-intensive agents
```

**Issue: Data Pipeline Lag**
```bash
# Check network connectivity
ping api.kraken.com

# Check API rate limits
curl -H "API-Key: $KRAKEN_API_KEY" https://api.kraken.com/0/private/TradeBalance

# Fix: Implement exponential backoff
```

For additional support, consult the [Quick Start Guide](README_QUICK_START.md) or create an issue in the repository.