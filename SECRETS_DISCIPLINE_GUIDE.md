# Secrets Management Discipline - CryptoSmartTrader V2

## Overzicht

CryptoSmartTrader V2 gebruikt een strikte secrets-discipline met centrale configuratie, security validatie en environment-specifieke vereisten.

## Secrets Architectuur

```
config/
├── secrets_manager.py    # Centraal secrets beheer
├── security.py          # Security validatie en audit
└── validation.py        # Configuratie validatie

.env.example              # Template voor environment variabelen
```

## Environment Specificaties

### Development Mode
**Vereiste secrets:**
- `KRAKEN_API_KEY` - Kraken exchange API key
- `KRAKEN_SECRET` - Kraken exchange secret  
- `OPENAI_API_KEY` - OpenAI API voor AI features

**Optionele secrets:**
- `ANTHROPIC_API_KEY` - Claude models
- `GEMINI_API_KEY` - Google Gemini
- `SLACK_BOT_TOKEN` + `SLACK_CHANNEL_ID` - Alerts

### Production Mode
**Vereiste secrets (alle van development +):**
- `JWT_SECRET_KEY` - JWT token security
- `API_SECRET_KEY` - Internal API authentication
- Alle exchange credentials voor live trading

### Testing Mode
**Vereiste secrets:** Geen
- Gebruikt mock data en simulaties
- Geen echte API calls

## Setup Instructies

### 1. Voor Replit Development

```bash
# Gebruik Replit Secrets tab (aanbevolen)
# Ga naar Secrets tab in je Replit
# Voeg secrets toe:
KRAKEN_API_KEY=your_actual_key
KRAKEN_SECRET=your_actual_secret  
OPENAI_API_KEY=your_actual_key
```

### 2. Voor Lokale Development

```bash
# Kopieer template
cp .env.example .env

# Edit .env met je werkelijke API keys
nano .env

# Vul minimaal in:
KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_SECRET=your_kraken_secret
OPENAI_API_KEY=your_openai_api_key
```

### 3. Voor Production Deployment

```bash
# Alle vereiste secrets via environment variabelen
export ENVIRONMENT=production
export TRADING_MODE=live
export KRAKEN_API_KEY=your_production_key
export KRAKEN_SECRET=your_production_secret
export OPENAI_API_KEY=your_production_openai_key
export JWT_SECRET_KEY=$(openssl rand -hex 32)
export API_SECRET_KEY=$(openssl rand -hex 32)
```

## Secrets Validatie

Het systeem voert automatische validatie uit:

### 1. Format Validatie
- API keys: 10-200 karakters, alphanumeriek
- Secrets: Aanwezigheid en lengte check
- JWT keys: Minimaal 32 karakters

### 2. Environment Validatie
- Development: Waarschuwing bij missende secrets
- Production: Error en system stop bij missende secrets
- Live trading: Extra validatie voor exchange credentials

### 3. Security Audit
- Alle secrets access wordt gelogd (gehashed)
- Rate limiting op API access
- Lockout na teveel mislukte pogingen

## API Usage

### Secrets Manager Gebruik

```python
from config.secrets_manager import get_secrets_manager

# Haal secrets manager op
secrets = get_secrets_manager()

# Exchange credentials
kraken_creds = secrets.get_exchange_credentials("kraken")
api_key = kraken_creds["api_key"]
secret = kraken_creds["secret"]

# AI API keys
openai_key = secrets.get_ai_api_key("openai")

# System configuratie
system_config = secrets.get_system_config()
trading_mode = system_config["trading_mode"]

# Health check
health = secrets.get_health_status()
print(f"Secrets health: {health['status']}")
```

### Security Validatie

```python
from config.security import security_manager

# Valideer input
if security_manager.validate_input(api_key, "api_key"):
    # Gebruik API key
    pass

# Hash sensitive data voor logging
hashed = security_manager.hash_sensitive_data(secret_value)
logger.info(f"Processing key: {hashed}")
```

## Security Features

### 1. Input Sanitization
- Alle externe input wordt gevalideerd
- Dangerous characters worden weggehaald
- Length limits worden afgedwongen

### 2. Audit Trail
- Alle secrets access wordt gelogd
- Security events in `logs/security_audit.log`
- Gehashed identifiers voor privacy

### 3. Rate Limiting
- Maximum API calls per identifier
- Automatic lockout na mislukte pogingen
- Configurable windows en limits

### 4. Environment Enforcement
- Live trading alleen in production
- Automatic mode switching validatie
- Fail-fast bij incorrecte configuratie

## Troubleshooting

### "Missing required secrets" Error

```bash
# Check welke secrets ontbreken
python -c "
from config.secrets_manager import get_secrets_manager
health = get_secrets_manager().get_health_status()
print(health)
"

# Voor development mode
export KRAKEN_API_KEY=your_key
export KRAKEN_SECRET=your_secret  
export OPENAI_API_KEY=your_key
```

### "Invalid API key format" Warning

```bash
# Check API key format
python -c "
from config.security import security_manager
valid = security_manager.validate_input('your_key', 'api_key')
print(f'Valid: {valid}')
"
```

### Secrets Health Check

```bash
# Run health check
python -c "
from config.secrets_manager import get_secrets_manager
health = get_secrets_manager().get_health_status()
print(f'Status: {health[\"status\"]}')
for secret, status in health['details'].items():
    print(f'{secret}: present={status[\"present\"]}, valid={status[\"valid_format\"]}')
"
```

## Best Practices

### 1. Development
- Gebruik altijd development environment
- Test met paper trading mode
- Commit nooit .env bestanden

### 2. Production  
- Gebruik environment variabelen
- Enable alle security features
- Monitor security audit logs
- Roteer API keys regelmatig

### 3. Security
- Gebruik sterke JWT secrets (32+ chars)
- Enable rate limiting
- Monitor failed attempts
- Audit secrets access patterns

## Integration met Replit

Voor Replit deployment gebruikt het systeem:

1. **Replit Secrets** (aanbevolen)
   - Gebruik Secrets tab in Replit interface
   - Automatisch beschikbaar als environment variabelen

2. **Environment variabelen**
   - Fallback naar .env file
   - Geschikt voor lokale development

3. **Health monitoring**
   - Dashboard toont secrets status
   - Automatic health checks bij startup

Het systeem detecteert automatisch Replit environment en past configuratie aan.