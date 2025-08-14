# CryptoSmartTrader V2 - Documentation Index

## 📚 Documentation Hierarchy

### 🚀 Quick Start & Operations
| Document | Purpose | Audience |
|----------|---------|----------|
| **README.md** | Main entry point, 60-second setup | All users |
| **docs/QUICK_REFERENCE.md** | Emergency commands, troubleshooting | Operations team |
| **README_CONSOLIDATED.md** | Complete operations manual | System administrators |

### 🏗️ Architecture & Design
| Document | Purpose | Audience |
|----------|---------|----------|
| **docs/ARCHITECTURE_DIAGRAMS.md** | System diagrams, component flow | Developers, architects |
| **docs/ADR_RECORDS.md** | Architecture Decision Records | Technical leads |
| **replit.md** | Project preferences, user guidelines | Development team |

### 🚨 Operations & Maintenance  
| Document | Purpose | Audience |
|----------|---------|----------|
| **docs/runbooks/INCIDENT_RESPONSE.md** | P0/P1/P2 incident procedures | Operations team |
| **REPLIT_MULTI_SERVICE_CONFIG.md** | Multi-service configuration | DevOps, deployment |
| **REPLIT_RUN_CONFIGURATIONS.md** | UV-based startup patterns | Replit deployment |

### 📜 Legacy Documentation (Archived)
| Document | Status | Notes |
|----------|--------|-------|
| **README_DEPLOYMENT.md** | Archived | Consolidated into README_CONSOLIDATED.md |
| **README_INSTALLATION.md** | Archived | Merged into main README.md |
| **README_WINDOWS_DEPLOYMENT.md** | Archived | Specific deployment scenarios moved |
| **README_WORKSTATION_DEPLOYMENT.md** | Archived | Enterprise deployment consolidated |

---

## 🎯 Documentation by Use Case

### Getting Started
1. **First Time Setup:** README.md → Quick Start section
2. **Replit Deployment:** REPLIT_MULTI_SERVICE_CONFIG.md
3. **Local Development:** README_CONSOLIDATED.md → Development section

### Daily Operations
1. **System Health:** docs/QUICK_REFERENCE.md → Health Checks
2. **Maintenance:** scripts/operations/ directory
3. **Monitoring:** README_CONSOLIDATED.md → Monitoring section

### Troubleshooting
1. **Immediate Issues:** docs/QUICK_REFERENCE.md → Emergency Procedures
2. **Incident Response:** docs/runbooks/INCIDENT_RESPONSE.md
3. **Deep Diagnostics:** README_CONSOLIDATED.md → Recovery Procedures

### Development
1. **Architecture Understanding:** docs/ARCHITECTURE_DIAGRAMS.md
2. **Technical Decisions:** docs/ADR_RECORDS.md
3. **Configuration:** replit.md → System Architecture

---

## 🛠️ Operational Scripts

### Daily Operations (`scripts/operations/`)
```bash
# Health monitoring
python scripts/operations/daily_health_check.py

# Log management
python scripts/operations/cleanup_logs.py --days 30

# System backup
python scripts/operations/backup_system.py

# Service testing
python test_replit_services.py
```

### Quick Commands
```bash
# System status
curl http://localhost:8001/health/detailed | jq .

# Service restart
python start_multi_service.py

# Emergency stop
pkill -f "streamlit" && pkill -f "uvicorn"
```

---

## 📊 Key Metrics & Endpoints

### Health Monitoring
- **Dashboard:** http://localhost:5000
- **API Health:** http://localhost:8001/health
- **Detailed Status:** http://localhost:8001/health/detailed
- **Metrics:** http://localhost:8000/metrics
- **API Docs:** http://localhost:8001/api/docs

### Performance Thresholds
- **Availability SLO:** >99.5%
- **Response Time:** <2s (Dashboard), <100ms (API)
- **Error Rate:** <1%
- **Health Score:** >80%

---

## 🔄 Documentation Maintenance

### Update Schedule
- **Daily:** Incident logs, health reports
- **Weekly:** Operational metrics, performance trends
- **Monthly:** Architecture updates, ADR reviews
- **Quarterly:** Full documentation review

### Update Triggers
- **System Changes:** Update architecture diagrams
- **New Features:** Update ADRs and user guides
- **Incidents:** Update runbooks and procedures
- **Performance Issues:** Update optimization guides

### Version Control
All documentation is version controlled with the codebase:
- **Major Changes:** Update version numbers
- **ADR Changes:** Mark status and review dates
- **Runbook Updates:** Date stamp and incident references

---

## 📞 Support & Escalation

### Documentation Issues
1. Check docs/QUICK_REFERENCE.md for immediate help
2. Review relevant runbook in docs/runbooks/
3. Consult architecture docs for system understanding
4. Check GitHub issues for known problems

### Emergency Contact Chain
1. **P0 Critical:** Immediate response required
2. **P1 High:** 1-hour response SLA
3. **P2 Medium:** 4-hour response SLA  
4. **P3 Low:** Next business day

---

**Documentation Version:** 2.0.0  
**Last Updated:** 2025-01-11  
**Maintained By:** Development Team  
**Review Schedule:** Monthly