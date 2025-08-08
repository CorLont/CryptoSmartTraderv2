# CryptoSmartTrader V2 - Deployment Guide

## Incident Playbook

### 🚨 Critical System Failures

#### Dashboard Won't Start
```bash
# Check if ports are in use
netstat -an | findstr ":5000"
netstat -an | findstr ":5001"

# Kill existing processes
taskkill /f /im python.exe
taskkill /f /im streamlit.exe

# Restart services
3_start_dashboard.bat
```

#### Data Pipeline Failures
```bash
# Check system health
python system_health_check.py

# Run diagnostic
python -c "from core.distributed_orchestrator import DistributedOrchestrator; o = DistributedOrchestrator(); print(o.get_system_status())"

# Restart with clean state
python orchestration/distributed_orchestrator.py --reset
```

#### Model Performance Degradation
```bash
# Check drift detection
python -c "from core.drift_detection import DriftDetectionSystem; d = DriftDetectionSystem(); print(d.get_alerts())"

# Trigger manual fine-tuning
python -c "from core.fine_tune_scheduler import FineTuneScheduler; f = FineTuneScheduler(); f.create_job('manual_intervention', 'critical')"

# Fallback to paper trading
python -c "from core.auto_disable_system import AutoDisableSystem; a = AutoDisableSystem(); a.force_disable('manual_intervention')"
```

#### Database/Storage Issues
```bash
# Check disk space
python -c "import shutil; print(f'Disk space: {shutil.disk_usage(\".\")[2] / 1e9:.1f} GB free')"

# Clean old logs
python scripts/cleanup_logs.py --days 30

# Restore from backup
python scripts/backup_restore.py restore backups/latest_backup.zip --dry-run
```

### 📊 Performance Monitoring

#### Key Metrics to Monitor
- **System Health Score**: Should be >60 for live trading
- **Confidence Gate Pass Rate**: Should be >10% daily
- **Data Completeness**: Should be >80% for all coins
- **Memory Usage**: Should be <8GB
- **CPU Usage**: Should be <70% sustained

#### Monitoring Commands
```bash
# Real-time system status
python system_health_check.py --continuous

# Generate health report
python eval/evaluator.py --health-report

# Check confidence gate status
python -c "from core.confidence_gate_manager import get_confidence_gate_manager; print(get_confidence_gate_manager().get_daily_report())"
```

## Rollout/Rollback Procedures

### 🚀 Production Rollout

#### Pre-Rollout Checklist
- [ ] All tests passing (`python -m pytest tests/`)
- [ ] System health check passed (`python system_health_check.py`)
- [ ] Backup created (`create_backup.bat`)
- [ ] Paper trading validation completed (4+ weeks)
- [ ] Confidence gate functioning (>80% pass rate in testing)
- [ ] All dependencies installed (`1_install_all_dependencies.bat`)

#### Rollout Steps
1. **Create Pre-Rollout Backup**
   ```bash
   python scripts/backup_restore.py backup --name "pre_rollout_$(date +%Y%m%d)" --include-logs
   ```

2. **Deploy New Version**
   ```bash
   # Update code
   git pull origin main
   
   # Install new dependencies
   1_install_all_dependencies.bat
   
   # Run validation
   python system_health_check.py
   ```

3. **Staged Deployment**
   ```bash
   # Start in paper trading mode
   python -c "from core.auto_disable_system import AutoDisableSystem; AutoDisableSystem().set_mode('paper')"
   
   # Monitor for 24 hours
   python scripts/monitor_deployment.py --hours 24
   
   # Enable live trading if stable
   python -c "from core.auto_disable_system import AutoDisableSystem; AutoDisableSystem().enable_live_trading()"
   ```

### ⏪ Emergency Rollback

#### Immediate Rollback (< 5 minutes)
```bash
# Stop all services
taskkill /f /im python.exe

# Restore from backup
python scripts/backup_restore.py restore backups/pre_rollout_backup.zip

# Restart services
3_start_dashboard.bat

# Verify rollback
python system_health_check.py
```

#### Planned Rollback
```bash
# Create rollback backup
python scripts/backup_restore.py backup --name "rollback_point"

# Graceful shutdown
python orchestration/distributed_orchestrator.py --shutdown

# Restore previous version
python scripts/backup_restore.py restore backups/stable_version.zip

# Restart and validate
3_start_dashboard.bat
python system_health_check.py
```

## Release Checklist

### 🔍 Pre-Release Testing

#### Automated Tests
- [ ] Unit tests: `python -m pytest tests/unit/`
- [ ] Integration tests: `python -m pytest tests/integration/`
- [ ] System tests: `python test_complete_system.py`
- [ ] Performance tests: `python test_performance_benchmarks.py`

#### Manual Testing
- [ ] Dashboard loads correctly
- [ ] All agents start successfully
- [ ] Data pipeline processes real data
- [ ] Confidence gate filters correctly
- [ ] Export functionality works
- [ ] Backup/restore works
- [ ] One-click pipeline completes

#### Security Testing
- [ ] No credentials in logs
- [ ] API keys properly secured
- [ ] File permissions correct
- [ ] Network ports configured

### 📋 Release Process

#### Version Management
```bash
# Tag release
git tag -a v2.0.0 -m "CryptoSmartTrader V2.0.0 - Full system release"

# Update version in config
python -c "import json; config = json.load(open('config.json')); config['version'] = '2.0.0'; json.dump(config, open('config.json', 'w'), indent=2)"
```

#### Documentation Updates
- [ ] Update README.md with new features
- [ ] Update API documentation
- [ ] Update user guide
- [ ] Update deployment instructions

#### Release Package Creation
```bash
# Create release backup
python scripts/backup_restore.py backup --name "release_v2.0.0" --include-logs --output-dir releases

# Create deployment package
python scripts/create_deployment_package.py --version 2.0.0

# Generate release notes
python scripts/generate_release_notes.py --version 2.0.0
```

### 🎯 Post-Release Validation

#### Immediate Checks (0-2 hours)
- [ ] All services started successfully
- [ ] No critical errors in logs
- [ ] Dashboard accessible
- [ ] Data pipeline processing
- [ ] System health >80%

#### Short-term Monitoring (2-24 hours)
- [ ] Memory usage stable
- [ ] No memory leaks
- [ ] Performance within expected range
- [ ] Error rates <1%
- [ ] Confidence gate functioning

#### Long-term Validation (1-7 days)
- [ ] System health maintained >60%
- [ ] No data quality degradation
- [ ] Model performance stable
- [ ] No unusual trading patterns
- [ ] User reports positive

### 🚨 Emergency Contacts

#### Technical Issues
- System Admin: Check system logs
- Database Issues: Check storage and connections
- Network Issues: Check firewall and port configuration

#### Emergency Procedures
1. **System Down**: Run `3_start_dashboard.bat`
2. **Data Loss**: Restore from latest backup
3. **Performance Issues**: Check `system_health_check.py`
4. **Security Breach**: Disable system, rotate API keys

### 📞 Escalation Matrix

| Severity | Response Time | Actions |
|----------|---------------|---------|
| P1 - System Down | 5 minutes | Stop trading, rollback, investigate |
| P2 - Performance Degraded | 30 minutes | Monitor, prepare rollback |
| P3 - Minor Issues | 2 hours | Log, schedule fix |
| P4 - Enhancement | Next release | Plan, develop, test |

### 📈 Success Metrics

#### Technical Metrics
- **Uptime**: >99.5%
- **Response Time**: <2 seconds for dashboard
- **Data Freshness**: <5 minutes lag
- **Error Rate**: <0.1% per day

#### Business Metrics
- **Confidence Gate Pass Rate**: >15% daily
- **Model Accuracy**: >70% for next day predictions
- **System Health**: >70% average
- **Trading Opportunities**: >5 per day with high confidence

---

## Quick Reference Commands

```bash
# System Status
python system_health_check.py

# Start System
3_start_dashboard.bat

# Run Pipeline
oneclick_runner.bat

# Create Backup
create_backup.bat

# Emergency Stop
taskkill /f /im python.exe

# Check Logs
type logs\daily\%date:~-4,4%%date:~-10,2%%date:~-7,2%\pipeline.log
```