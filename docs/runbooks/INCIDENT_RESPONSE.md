# CryptoSmartTrader V2 - Incident Response Runbook

## 🚨 Emergency Contacts & Escalation

### Severity Levels
- **P0 (Critical):** System down, data loss, security breach
- **P1 (High):** Service degraded, user impact, revenue impact  
- **P2 (Medium):** Performance issues, non-critical failures
- **P3 (Low):** Minor issues, maintenance items

### Response Times
- **P0:** Immediate response (0-15 minutes)
- **P1:** 1 hour response
- **P2:** 4 hour response  
- **P3:** Next business day

---

## 🔥 P0 Critical Incidents

### System Completely Down

**Symptoms:**
- Dashboard (port 5000) not accessible
- All services returning connection errors
- Health endpoints timing out

**Immediate Actions:**
```bash
# 1. Check if processes are running
ps aux | grep -E "(streamlit|uvicorn|python)"

# 2. Check port availability
netstat -tulpn | grep -E ":5000|:8001|:8000"

# 3. Check system resources
df -h  # Disk space
free -h  # Memory
top  # CPU usage

# 4. Emergency restart
pkill -f "streamlit" && pkill -f "uvicorn"
sleep 5
python start_multi_service.py

# 5. Verify recovery
curl http://localhost:5000/_stcore/health
curl http://localhost:8001/health  
curl http://localhost:8000/health
```

**If restart fails:**
```bash
# Check logs for errors
tail -100 logs/app.log | grep ERROR
tail -50 logs/system.log

# Check disk space (critical threshold: < 1GB)
if [ $(df --output=avail / | tail -1) -lt 1048576 ]; then
    echo "CRITICAL: Disk space low"
    # Emergency cleanup
    python scripts/emergency_cleanup.py
fi

# Restore from backup if needed
python scripts/backup_restore.py restore backups/latest_stable.zip --force
```

### Data Corruption Detected

**Symptoms:**
- Data integrity warnings in logs
- Validation failures
- Inconsistent market data

**Immediate Actions:**
```bash
# 1. Stop all trading immediately
python -c "from core.trading_engine import TradingEngine; te = TradingEngine(); te.emergency_stop()"

# 2. Assess corruption scope
python scripts/data_integrity_check.py --full-scan

# 3. Switch to backup data source
python scripts/switch_data_source.py --source backup

# 4. Restore from known good backup
python scripts/restore_market_data.py --date $(date -d "1 day ago" +%Y-%m-%d)
```

### Security Breach

**Symptoms:**
- Unauthorized access alerts
- Unusual API usage patterns
- Configuration changes

**Immediate Actions:**
```bash
# 1. Isolate system immediately
iptables -I INPUT -j DROP  # Block all incoming traffic

# 2. Revoke all API keys
python scripts/revoke_api_keys.py --all

# 3. Secure logs
tar -czf incident_logs_$(date +%Y%m%d_%H%M%S).tar.gz logs/
chmod 600 incident_logs_*.tar.gz

# 4. Generate incident report
python scripts/security_incident_report.py --timestamp $(date +%Y%m%d_%H%M%S)
```

---

## ⚠️ P1 High Priority Incidents

### Service Degraded (High Error Rate)

**Symptoms:**
- Error rate > 5%
- Response times > 5 seconds
- Intermittent failures

**Investigation Steps:**
```bash
# 1. Check service health
curl http://localhost:8001/health/detailed | jq .

# 2. Monitor error rates
curl http://localhost:8000/metrics | grep error_rate

# 3. Check recent logs
tail -200 logs/app.log | grep ERROR | tail -20

# 4. Identify failing component
python scripts/diagnose_errors.py --last-hour
```

**Resolution Actions:**
```bash
# If API service is failing
systemctl restart api_service
# or
pkill -f "health_endpoint" && python api/health_endpoint.py &

# If specific agent is failing
python scripts/restart_agent.py --agent sentiment_agent

# If database issues
python scripts/rebuild_cache.py
python scripts/repair_data_files.py
```

### ML Model Performance Degradation

**Symptoms:**
- Prediction accuracy < 60%
- High model uncertainty
- Drift detection alerts

**Investigation:**
```bash
# 1. Check model performance
python -c "from ml.model_manager import ModelManager; mm = ModelManager(); print(mm.get_performance_summary())"

# 2. Check for data drift
python scripts/drift_analysis.py --models all --timeframe 24h

# 3. Validate training data quality
python scripts/validate_training_data.py --recent
```

**Resolution:**
```bash
# 1. Switch to backup model
python scripts/switch_model.py --model backup --horizon all

# 2. Trigger emergency retraining
python scripts/emergency_retrain.py --priority high

# 3. Reduce confidence threshold temporarily
python scripts/adjust_confidence.py --threshold 0.7 --duration 2h

# 4. Enable paper trading mode
python scripts/enable_paper_trading.py --reason "model_degradation"
```

### API Rate Limiting Issues

**Symptoms:**
- 429 errors from Kraken API
- Data update delays
- Missing market data

**Resolution:**
```bash
# 1. Check current rate limits
python scripts/check_rate_limits.py --exchange kraken

# 2. Enable rate limit backoff
python scripts/enable_backoff.py --aggressive

# 3. Switch to cached data temporarily
python scripts/enable_cache_mode.py --duration 30m

# 4. Distribute requests across time
python scripts/spread_requests.py --interval 2s
```

---

## 📊 P2 Medium Priority Incidents

### Performance Issues

**Symptoms:**
- Slow page loads (2-5 seconds)
- High memory usage
- CPU usage > 80%

**Investigation:**
```bash
# 1. Performance profiling
python scripts/performance_profile.py --duration 5m

# 2. Memory analysis
python scripts/memory_analysis.py --dump-large-objects

# 3. Database query analysis
python scripts/analyze_slow_queries.py --threshold 1s
```

**Optimization:**
```bash
# 1. Clear caches
python scripts/clear_caches.py --selective

# 2. Optimize database
python scripts/optimize_database.py

# 3. Enable performance mode
python scripts/enable_performance_mode.py

# 4. Reduce update frequency
python scripts/adjust_update_frequency.py --reduce 50%
```

---

## 🛠️ Diagnostic Tools

### Quick Health Check
```bash
#!/bin/bash
# scripts/quick_health_check.sh

echo "=== CryptoSmartTrader V2 Health Check ==="
echo "Time: $(date)"
echo

# Service status
echo "Service Status:"
curl -s http://localhost:5000/_stcore/health && echo " ✓ Dashboard OK" || echo " ✗ Dashboard FAIL"
curl -s http://localhost:8001/health && echo " ✓ API OK" || echo " ✗ API FAIL"
curl -s http://localhost:8000/health && echo " ✓ Metrics OK" || echo " ✗ Metrics FAIL"

# System resources
echo -e "\nSystem Resources:"
echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)%"
echo "Memory: $(free | grep Mem | awk '{printf "%.1f%%", $3/$2 * 100.0}')"
echo "Disk: $(df -h / | awk 'NR==2{print $5}')"

# Recent errors
echo -e "\nRecent Errors:"
tail -10 logs/app.log | grep ERROR | wc -l | xargs echo "Error count (last 10 lines):"

# Trading status
echo -e "\nTrading Status:"
curl -s http://localhost:8001/health/detailed | jq -r '.application_health.trading_status' 2>/dev/null || echo "Unable to determine"
```

### Deep System Analysis
```bash
#!/bin/bash
# scripts/deep_analysis.sh

echo "=== Deep System Analysis ==="

# Process tree
echo "Process Tree:"
pstree -p $(pgrep -f streamlit) 2>/dev/null

# Port analysis
echo -e "\nPort Analysis:"
netstat -tulpn | grep -E ":5000|:8001|:8000"

# Log analysis
echo -e "\nError Pattern Analysis:"
grep -c ERROR logs/app.log | xargs echo "Total errors in app.log:"
grep ERROR logs/app.log | tail -5

# Memory analysis
echo -e "\nMemory Usage by Process:"
ps aux --sort=-%mem | grep python | head -5

# Disk usage
echo -e "\nDisk Usage:"
du -sh logs/ cache/ models/ 2>/dev/null

# Network connectivity
echo -e "\nNetwork Connectivity:"
curl -s -o /dev/null -w "Kraken API: %{http_code} (%{time_total}s)\n" https://api.kraken.com/0/public/SystemStatus
```

---

## 📋 Recovery Checklists

### Post-Incident Recovery Checklist

- [ ] **Immediate Response**
  - [ ] Incident acknowledged within SLA
  - [ ] Initial assessment completed
  - [ ] Stakeholders notified

- [ ] **Investigation**
  - [ ] Root cause identified
  - [ ] Impact scope determined
  - [ ] Timeline established

- [ ] **Resolution**
  - [ ] Fix implemented
  - [ ] System functionality verified
  - [ ] Performance metrics normal

- [ ] **Post-Incident**
  - [ ] Incident report completed
  - [ ] Lessons learned documented
  - [ ] Prevention measures implemented
  - [ ] Team debriefing conducted

### System Recovery Verification

```bash
#!/bin/bash
# scripts/verify_recovery.sh

echo "=== Recovery Verification ==="

# 1. All services healthy
python test_replit_services.py | grep "Overall Status" | grep "ALL HEALTHY" || exit 1

# 2. Data integrity check
python scripts/data_integrity_check.py --quick || exit 1

# 3. Performance within thresholds
RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}' http://localhost:5000)
if (( $(echo "$RESPONSE_TIME > 5" | bc -l) )); then
    echo "ERROR: Response time too high: ${RESPONSE_TIME}s"
    exit 1
fi

# 4. Trading system operational
TRADING_STATUS=$(curl -s http://localhost:8001/health/detailed | jq -r '.application_health.trading_status')
if [ "$TRADING_STATUS" = "NO-GO" ]; then
    echo "WARNING: Trading system in NO-GO state"
fi

echo "✓ Recovery verification complete"
```

---

## 📞 Communication Templates

### P0 Incident Notification
```
PRIORITY: CRITICAL (P0)
System: CryptoSmartTrader V2
Status: INCIDENT ACTIVE
Time: [TIMESTAMP]

IMPACT: [Description of user/business impact]
CURRENT STATUS: [What's happening now]
ACTIONS TAKEN: [What we've done so far]
NEXT STEPS: [What we're doing next]
ETA: [Estimated resolution time]

Updates will be provided every 15 minutes until resolved.
```

### Resolution Notification
```
RESOLVED: CryptoSmartTrader V2 Incident
Duration: [Start time] to [End time]
Impact: [Final impact assessment]

ROOT CAUSE: [Brief technical explanation]
RESOLUTION: [What fixed it]
PREVENTION: [What we're doing to prevent recurrence]

Full incident report will be available within 24 hours.
```

---

## 🔄 Runbook Maintenance

This runbook should be:
- **Reviewed:** Monthly for accuracy
- **Updated:** After each incident
- **Tested:** Quarterly during disaster recovery drills
- **Improved:** Based on incident learnings

**Last Updated:** 2025-01-11  
**Next Review:** 2025-02-11