# CryptoSmartTrader V2 - Windows Workstation Deployment

## Snelle Start voor Windows

### 1. Eenmalige Setup

```batch
# Download/clone het project
git clone <repository-url>
cd cryptosmarttrader-v2

# Run eenmalige setup
setup_env.bat
```

### 2. Configuratie

Bewerk `.env` bestand met je API keys:

```env
# Exchange APIs
KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_SECRET=your_kraken_secret

# AI Services  
OPENAI_API_KEY=your_openai_api_key

# Social Media (optioneel)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# Database (gebruik defaults voor development)
DATABASE_URL=postgresql://user:pass@localhost:5432/cryptotrader
REDIS_URL=redis://localhost:6379
```

### 3. Start het Complete Systeem

```batch
# Start alle services in losse vensters
start_all.bat
```

**Of start services individueel:**

```batch
# API Backend (port 8001)
start_api.bat

# UI Dashboard (port 5000)  
start_ui.bat

# Background Workers
start_workers.bat
```

## Service Overzicht

| Service | Port | Beschrijving |
|---------|------|--------------|
| **API Server** | 8001 | FastAPI backend met metrics |
| **UI Dashboard** | 5000 | Streamlit Alpha Motor dashboard |
| **Workers** | - | Data ingestion, agents, execution |
| **Observability** | 8002 | Centralized metrics & alerts |

## URLs na Opstarten

- **Main Dashboard**: http://localhost:5000
- **API Docs**: http://localhost:8001/docs
- **Health Check**: http://localhost:8001/health
- **Metrics**: http://localhost:8002/metrics
- **Alerts**: http://localhost:8002/alerts

## Troubleshooting

### Python Niet Gevonden
```batch
# Installeer Python 3.11+ van python.org
# Zorg dat Python in PATH staat
python --version
```

### Virtual Environment Errors
```batch
# Remove oude venv en herstart
rmdir /s .venv
setup_env.bat
```

### Port Already in Use
```batch
# Stop alle Python processen
taskkill /f /im python.exe
taskkill /f /im streamlit.exe

# Of wijzig poorten in de .bat bestanden
```

### API Keys Errors
- Check `.env` bestand voor correcte formatting
- Geen spaties rond = tekens
- Gebruik quotes voor keys met speciale characters

### Memory/Performance Issues
```batch
# Verhoog virtual memory
# Task Manager -> Performance -> Memory
# Recommended: 8GB+ RAM voor optimal performance
```

## Monitoring

### System Health
- **API Health**: GET http://localhost:8001/health
- **Workers Status**: Check "CryptoSmartTrader Workers" window
- **UI Status**: Check "CryptoSmartTrader UI" window

### Logs Locaties
- **API Logs**: API Server window console
- **UI Logs**: UI Dashboard window console  
- **Worker Logs**: Workers window console
- **System Logs**: `logs/` directory

### Performance Metrics
- **CPU Usage**: Task Manager -> Performance
- **Memory Usage**: Task Manager -> Performance  
- **Network**: Task Manager -> Performance -> Ethernet
- **Trading Metrics**: http://localhost:8002/metrics

## Production Deployment

Voor production deployment op Windows Server:

### 1. Service Installation
```batch
# Install als Windows Service met NSSM
nssm install CryptoSmartTraderAPI "C:\path\to\python.exe" "-m uvicorn src.cryptosmarttrader.api.main:app --host 0.0.0.0 --port 8001"
nssm install CryptoSmartTraderUI "C:\path\to\streamlit.exe" "run enhanced_alpha_motor_dashboard.py --server.port 5000"
```

### 2. Auto-start Configuration
```batch
# Set services to auto-start
nssm set CryptoSmartTraderAPI Start SERVICE_AUTO_START
nssm set CryptoSmartTraderUI Start SERVICE_AUTO_START
```

### 3. Monitoring Setup
- Configure Windows Event Log forwarding
- Setup Prometheus/Grafana voor metrics
- Configure alert notifications (email/SMS)

## Security Considerations

### API Keys Management
- Store production keys in Azure Key Vault
- Use environment-specific .env files
- Rotate keys monthly
- Monitor API usage limits

### Network Security
- Configure Windows Firewall rules
- Use HTTPS in production (reverse proxy)
- VPN access voor remote management
- Regular security updates

### Data Protection
- Encrypt sensitive data at rest
- Secure database connections (SSL)
- Regular backups (automated)
- GDPR compliance voor EU users

## Backup & Recovery

### Automated Backups
```batch
# Database backup script
backup_database.bat

# Configuration backup
backup_config.bat

# Complete system backup
backup_all.bat
```

### Recovery Procedures
1. Stop alle services
2. Restore database from backup
3. Restore configuration files
4. Restart services
5. Verify system health

## Support & Maintenance

### Regular Maintenance
- **Daily**: Check system health, review alerts
- **Weekly**: Review performance metrics, update dependencies
- **Monthly**: Rotate API keys, backup configurations
- **Quarterly**: Security audit, performance optimization

### Emergency Procedures
- **System Down**: Check service status, restart if needed
- **Data Issues**: Activate backup systems, investigate root cause
- **Security Breach**: Isolate system, rotate all keys, audit logs

Voor technische support: Raadpleeg project documentatie of contact development team.