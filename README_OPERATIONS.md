# CryptoSmartTrader V2 - Operations Runbook

📋 **Enterprise operations manual** for incident response, maintenance procedures, and system administration.

## 🚨 Emergency Procedures

### 1. Kill Switch Activation

**When to Use**: Immediately stop all trading activity due to:
- Unexpected market conditions
- System malfunction
- Security breach
- Regulatory requirements

**Procedures**:

```bash
# Method 1: API Kill Switch
curl -X POST http://localhost:8001/api/v1/emergency/kill_switch \
  -H "Content-Type: application/json" \
  -d '{"reason": "emergency_stop", "user_id": "operator"}'

# Method 2: Manual Process Kill
pkill -f "python.*agent"
pkill -f "streamlit"

# Method 3: Emergency Script
uv run python scripts/emergency_stop.py --all

# Method 4: Database Kill Switch
uv run python -c "
from cryptosmarttrader.core.risk_guard import RiskGuard
guard = RiskGuard()
guard.emergency_stop('manual_intervention')
"
```

**Verification**:
```bash
# Confirm all trading stopped
curl http://localhost:8001/api/v1/trading/status
# Should return: {"trading_active": false, "mode": "emergency_stopped"}

# Check agent status
curl http://localhost:8001/api/v1/agents/status
# All agents should show "stopped" or "emergency"
```

### 2. Portfolio Liquidation

**When to Use**: Emergency exit from all positions

```python
# Emergency liquidation
import httpx

response = httpx.post("http://localhost:8001/api/v1/emergency/liquidate_all", 
                     json={
                         "reason": "emergency_liquidation",
                         "max_slippage": 0.05,  # 5% max slippage
                         "time_limit": 300       # 5 minutes
                     })

print(f"Liquidation initiated: {response.json()}")
```

**Monitor Progress**:
```bash
# Check liquidation status
curl http://localhost:8001/api/v1/portfolio/liquidation_status

# Monitor positions
curl http://localhost:8001/api/v1/portfolio/positions
```

### 3. System Recovery

**Recovery Steps**:

1. **Assess Situation**
   ```bash
   # Check system health
   curl http://localhost:8001/health
   
   # Review recent logs
   tail -f logs/system.log
   tail -f logs/trading.log
   tail -f logs/error.log
   ```

2. **Restart Services**
   ```bash
   # Restart background services
   ./2_start_background_services.bat
   
   # Restart main dashboard
   ./3_start_dashboard.bat
   
   # Or restart all
   uv run python start_replit_services.py --restart
   ```

3. **Validate Recovery**
   ```bash
   # Run system checks
   uv run python scripts/system_health_check.py
   
   # Test exchange connectivity
   uv run python scripts/test_exchange_connectivity.py
   
   # Validate ML models
   uv run python scripts/validate_models.py
   ```

## 🔄 Rollback Procedures

### 1. Code Rollback

**Git-based Rollback**:
```bash
# Check recent commits
git log --oneline -10

# Rollback to specific commit
git reset --hard <commit_hash>

# Restart services
uv sync
uv run python start_replit_services.py --restart
```

**Checkpoint Rollback** (if using Replit):
```bash
# Use Replit rollback interface
# 1. Click "View Checkpoints" button
# 2. Select desired checkpoint
# 3. Confirm rollback
```

### 2. Configuration Rollback

```bash
# Backup current config
cp config.json config.json.backup

# Restore from backup
cp config/production_backup.json config.json

# Restart with new config
uv run python start_replit_services.py --restart
```

### 3. Database Rollback

```bash
# Create emergency backup
uv run python scripts/backup_database.py --emergency

# Restore from backup
uv run python scripts/restore_database.py --backup-id=<backup_id>

# Validate data integrity
uv run python scripts/validate_database.py
```

### 4. Model Rollback

```python
# Rollback to previous model version
from cryptosmarttrader.ml.model_registry import ModelRegistry

registry = ModelRegistry()

# List available models
models = registry.list_models()
print("Available models:", models)

# Rollback to previous version
success = registry.rollback_model("rf_24h", "v1.2.3")
print(f"Rollback success: {success}")

# Verify rollback
current_model = registry.get_current_model("rf_24h")
print(f"Current model: {current_model['version']}")
```

## 🔐 Secret Rotation Procedures

### 1. Manual Secret Rotation

**Kraken API Keys**:
```python
from cryptosmarttrader.core.secrets_manager import create_secrets_manager

# Initialize secrets manager
secrets_manager = create_secrets_manager()

# Rotate Kraken API key
new_api_key = "new_kraken_api_key_here"
success = secrets_manager.rotate_secret(
    secret_id="kraken_api_key",
    new_secret_value=new_api_key,
    user_id="admin"
)

print(f"Rotation success: {success}")
```

**OpenAI API Keys**:
```python
# Rotate OpenAI key
new_openai_key = "sk-new_openai_key_here"
success = secrets_manager.rotate_secret(
    secret_id="openai_api_key", 
    new_secret_value=new_openai_key,
    user_id="admin"
)

# Restart AI agents to use new key
import httpx
httpx.post("http://localhost:8001/api/v1/agents/restart", 
          json={"agents": ["sentiment_predictor", "technical_analyzer"]})
```

### 2. Automated Secret Rotation

```bash
# Check which secrets need rotation
uv run python scripts/check_secret_rotation.py

# Run automated rotation
uv run python scripts/auto_rotate_secrets.py --dry-run

# Execute rotation (remove --dry-run when ready)
uv run python scripts/auto_rotate_secrets.py
```

### 3. Emergency Secret Revocation

```python
# Emergency revoke compromised secret
success = secrets_manager.revoke_secret(
    secret_id="compromised_secret_id",
    user_id="security_admin"
)

# Verify revocation
secret = secrets_manager.get_secret("compromised_secret_id")
assert secret is None, "Secret should be revoked"

print("Secret successfully revoked")
```

## 📊 Incident Response

### 1. Incident Classification

**Severity Levels**:

- **P0 - Critical**: Trading stopped, data loss, security breach
- **P1 - High**: Performance degradation, partial outage
- **P2 - Medium**: Non-critical feature issues  
- **P3 - Low**: Minor bugs, documentation issues

### 2. Response Procedures

**P0 Critical Incidents**:

1. **Immediate Actions** (0-5 minutes):
   ```bash
   # Activate kill switch if trading-related
   curl -X POST http://localhost:8001/api/v1/emergency/kill_switch
   
   # Notify stakeholders
   uv run python scripts/incident_notification.py --severity=P0
   
   # Create incident log
   echo "$(date): P0 incident detected" >> logs/incidents.log
   ```

2. **Assessment** (5-15 minutes):
   ```bash
   # Gather system status
   uv run python scripts/incident_assessment.py --generate-report
   
   # Check recent changes
   git log --since="1 hour ago" --oneline
   
   # Review error logs
   tail -100 logs/error.log
   ```

3. **Resolution** (15+ minutes):
   ```bash
   # Apply fix or rollback
   git reset --hard <known_good_commit>
   
   # Restart services
   uv run python start_replit_services.py --restart
   
   # Validate fix
   uv run python scripts/post_incident_validation.py
   ```

**P1 High Incidents**:

1. **Response** (0-30 minutes):
   ```bash
   # Log incident
   uv run python scripts/log_incident.py --severity=P1 --description="<issue>"
   
   # Investigate root cause
   uv run python scripts/investigate_issue.py --timeframe="last_hour"
   ```

2. **Mitigation**:
   ```bash
   # Apply temporary fix
   uv run python scripts/apply_hotfix.py --issue-id=<incident_id>
   
   # Monitor for improvement
   curl http://localhost:8001/api/v1/health/detailed
   ```

### 3. Post-Incident Procedures

```bash
# Generate incident report
uv run python scripts/generate_incident_report.py --incident-id=<id>

# Update runbook based on lessons learned
# (Manual process - document in incidents/ directory)

# Schedule follow-up review
echo "Follow-up review scheduled for $(date -d '+1 week')" >> logs/incidents.log
```

## 💾 Backup & Recovery

### 1. Automated Backups

**Database Backup**:
```bash
# Daily backup (automated via cron)
0 2 * * * /usr/bin/uv run python scripts/backup_database.py --type=daily

# Weekly backup
0 2 * * 0 /usr/bin/uv run python scripts/backup_database.py --type=weekly

# Before major changes
uv run python scripts/backup_database.py --type=pre_deployment
```

**Configuration Backup**:
```bash
# Backup all configs
uv run python scripts/backup_configs.py

# Backup secrets (encrypted)
uv run python scripts/backup_secrets.py --encrypted
```

### 2. Recovery Procedures

**Database Recovery**:
```bash
# List available backups
uv run python scripts/list_backups.py

# Restore from specific backup
uv run python scripts/restore_database.py --backup-date=2025-01-13

# Validate restoration
uv run python scripts/validate_restored_data.py
```

**Configuration Recovery**:
```bash
# Restore configurations
uv run python scripts/restore_configs.py --backup-date=2025-01-13

# Restart services with restored config
uv run python start_replit_services.py --restart
```

## 📈 Performance Monitoring

### 1. Key Metrics

**System Health**:
```bash
# CPU and Memory usage
curl http://localhost:8000/metrics | grep "process_"

# API response times
curl http://localhost:8000/metrics | grep "http_request_duration"

# Trading metrics
curl http://localhost:8000/metrics | grep "trading_"
```

**Alert Thresholds**:
- CPU usage > 80% for 5 minutes
- Memory usage > 85% for 3 minutes  
- API latency > 2 seconds (95th percentile)
- Trading signals < 10/hour for 30 minutes
- Drawdown > 5% for 1 hour

### 2. Log Analysis

**Error Pattern Detection**:
```bash
# Check for error spikes
grep -c "ERROR" logs/system.log | tail -10

# Look for specific error patterns
grep "ConnectionError\|TimeoutError\|APIError" logs/error.log

# Analyze trading errors
grep "ORDER_FAILED\|EXECUTION_ERROR" logs/trading.log
```

**Performance Analysis**:
```bash
# Slow query detection
grep "slow_query" logs/database.log

# API latency analysis
grep "request_duration" logs/api.log | awk '{print $NF}' | sort -n
```

## 🛠️ Maintenance Procedures

### 1. Scheduled Maintenance

**Weekly Maintenance** (Sundays 02:00 UTC):
```bash
# System health check
uv run python scripts/weekly_health_check.py

# Log rotation
uv run python scripts/rotate_logs.py

# Database optimization
uv run python scripts/optimize_database.py

# Model retraining check
uv run python scripts/check_model_drift.py
```

**Monthly Maintenance**:
```bash
# Dependency updates
uv lock --upgrade
uv sync

# Security audit
uv run python scripts/security_audit.py

# Performance benchmarking
uv run python scripts/performance_benchmark.py

# Backup cleanup
uv run python scripts/cleanup_old_backups.py --keep-days=30
```

### 2. Agent Maintenance

**Agent Health Monitoring**:
```python
# Check all agent status
import httpx

agents_status = httpx.get("http://localhost:8001/api/v1/agents/status")
print("Agent Status:", agents_status.json())

# Restart unhealthy agents
for agent in agents_status.json():
    if agent['status'] != 'healthy':
        restart_response = httpx.post(
            f"http://localhost:8001/api/v1/agents/{agent['name']}/restart"
        )
        print(f"Restarted {agent['name']}: {restart_response.status_code}")
```

**Model Refresh**:
```bash
# Check model performance
uv run python scripts/evaluate_model_performance.py

# Retrain underperforming models
uv run python scripts/retrain_models.py --performance-threshold=0.6

# Update model registry
uv run python scripts/update_model_registry.py
```

## 📞 Contact Information

### Emergency Contacts

- **System Administrator**: admin@cryptosmarttrader.com
- **Security Team**: security@cryptosmarttrader.com  
- **On-Call Engineer**: +1-XXX-XXX-XXXX

### Escalation Procedures

1. **Level 1**: Automated alerts and self-healing
2. **Level 2**: On-call engineer notification
3. **Level 3**: Security team and management escalation
4. **Level 4**: External vendor and regulatory notification

### Communication Channels

- **Incident Updates**: #incidents Slack channel
- **System Status**: status.cryptosmarttrader.com
- **Documentation Updates**: #docs Slack channel

---

**📋 Remember**: Always document incidents, test recovery procedures regularly, and keep this runbook updated with operational learnings.