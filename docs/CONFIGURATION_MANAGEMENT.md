# Configuration Management - CryptoSmartTrader V2

## 🚀 Fail-Fast Configuration System

### Enterprise Pydantic Settings

The system uses centralized configuration management with fail-fast validation to ensure proper startup.

#### Core Configuration Classes

```python
from src.cryptosmarttrader.config import Settings, get_settings

# Load validated settings (singleton pattern)
settings = get_settings()
```

### 📋 Configuration Categories

#### System Settings
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `DEBUG_MODE` - Enable debug mode (default: False)
- `PERFORMANCE_MODE` - Enable performance optimizations (default: True)

#### Service Configuration
- `DASHBOARD_PORT` - Streamlit dashboard port (default: 5000)
- `API_PORT` - FastAPI service port (default: 8001)
- `METRICS_PORT` - Prometheus metrics port (default: 8000)

#### API Keys (Production Required)
- `KRAKEN_API_KEY` - Kraken exchange API key
- `KRAKEN_SECRET` - Kraken exchange secret
- `OPENAI_API_KEY` - OpenAI API key for sentiment analysis

#### Trading Configuration
- `CONFIDENCE_THRESHOLD` - Minimum confidence threshold (default: 0.8)
- `MAX_POSITIONS` - Maximum concurrent positions (default: 10)
- `RISK_LIMIT_PERCENT` - Maximum risk per trade % (default: 2.0)

#### Feature Toggles
- `ENABLE_PROMETHEUS` - Enable Prometheus metrics (default: True)
- `ENABLE_SENTIMENT` - Enable sentiment analysis (default: True)
- `ENABLE_WHALE_DETECTION` - Enable whale detection (default: True)
- `ENABLE_ML_PREDICTIONS` - Enable ML predictions (default: True)
- `ENABLE_PAPER_TRADING` - Enable paper trading mode (default: True)

### 🔍 Validation Framework

#### Startup Validation
```python
# Automatic validation on settings load
settings = get_settings()  # Validates or exits with error

# Manual validation
validation_result = settings.validate_startup_requirements()
if not validation_result["startup_ready"]:
    print("System not ready for startup")
```

#### Validation Categories

**Critical Issues (System Exit)**
- Missing required API keys for live trading
- Invalid port numbers
- Insufficient directory permissions
- Invalid configuration values

**Warnings (Logged)**
- Missing optional API keys
- Low system resources
- Port conflicts
- Performance recommendations

### 🚨 Fail-Fast Behavior

#### Configuration Loading
```python
# src/cryptosmarttrader/__main__.py
def main():
    # 1. Environment validation
    validate_environment()
    
    # 2. Load and validate settings (fail-fast)
    settings = load_and_validate_settings()
    
    # 3. Setup logging with validated settings
    setup_logging(level=settings.LOG_LEVEL)
    
    # 4. Start services
    start_services(settings)
```

#### Validation Results
- **SUCCESS**: Configuration valid, system starts
- **WARNING**: System starts with reduced functionality
- **CRITICAL**: System exits immediately with clear error messages

### 📊 Environment Variables

#### Required for Production
```bash
# Kraken exchange (required for live trading)
KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_SECRET=your_kraken_secret

# OpenAI (optional, for enhanced sentiment)
OPENAI_API_KEY=your_openai_api_key
```

#### Optional Configuration
```bash
# Logging
LOG_LEVEL=INFO
DEBUG_MODE=false

# Performance
PERFORMANCE_MODE=true
CACHE_SIZE_MB=500
MAX_WORKERS=4

# Trading thresholds
CONFIDENCE_THRESHOLD=0.8
RISK_LIMIT_PERCENT=2.0
MAX_POSITIONS=10

# Service ports
DASHBOARD_PORT=5000
API_PORT=8001
METRICS_PORT=8000

# Feature toggles
ENABLE_PROMETHEUS=true
ENABLE_SENTIMENT=true
ENABLE_WHALE_DETECTION=true
ENABLE_ML_PREDICTIONS=true
ENABLE_PAPER_TRADING=true
```

### 🛠️ Configuration Utilities

#### Settings Validation Script
```bash
# Validate configuration without starting services
python scripts/validate_settings.py

# Output example:
✅ Configuration Valid
{
  "version": "2.0.0",
  "environment": "production",
  "services": {
    "dashboard": 5000,
    "api": 8001,
    "metrics": 8000
  }
}
```

#### Configuration Summary
```python
# Get configuration summary for monitoring
settings = get_settings()
summary = settings.get_summary()

# Returns structured configuration information
{
    "version": "2.0.0",
    "environment": "production|development",
    "services": {"dashboard": 5000, "api": 8001, "metrics": 8000},
    "features": {"prometheus": true, "sentiment": true, ...},
    "thresholds": {"confidence": 0.8, "risk_limit": 2.0, ...}
}
```

### 🔧 Integration Examples

#### Service Startup
```python
from src.cryptosmarttrader.config import get_settings

def start_dashboard():
    settings = get_settings()
    streamlit.run(
        "app.py",
        port=settings.DASHBOARD_PORT,
        server_address="0.0.0.0"
    )
```

#### Agent Configuration
```python
class TradingAgent:
    def __init__(self):
        self.settings = get_settings()
        self.confidence_threshold = self.settings.CONFIDENCE_THRESHOLD
        self.max_positions = self.settings.MAX_POSITIONS
    
    def should_trade(self, confidence: float) -> bool:
        return confidence >= self.confidence_threshold
```

### 📈 Monitoring Integration

#### Configuration Health Check
```python
@app.get("/health/config")
def config_health():
    settings = get_settings()
    validation = settings.validate_startup_requirements()
    
    return {
        "config_valid": validation["startup_ready"],
        "warnings": validation["warnings"],
        "critical_issues": validation["critical_issues"]
    }
```

#### Prometheus Metrics
```python
# Expose configuration as metrics
config_info = Info("cryptotrader_config", "Configuration information")
config_info.info({
    "version": "2.0.0",
    "environment": settings.DEBUG_MODE and "dev" or "prod",
    "confidence_threshold": str(settings.CONFIDENCE_THRESHOLD)
})
```

### 🔐 Security Considerations

#### Secret Management
- Environment variables for sensitive data
- No secrets in configuration files
- Pydantic field descriptions for documentation
- Validation prevents exposure in logs

#### Configuration Validation
- Type safety with Pydantic
- Range validation for numeric values
- Port availability checking
- Directory permission validation

### 🚀 Quick Start Commands

```bash
# Validate configuration
python scripts/validate_settings.py

# Start with configuration validation
python -m src.cryptosmarttrader

# Check specific setting
python -c "from src.cryptosmarttrader.config import get_settings; print(get_settings().CONFIDENCE_THRESHOLD)"

# Environment setup
cp .env.example .env
# Edit .env with your API keys
python scripts/validate_settings.py
```

---

**Last Updated:** 2025-01-11  
**Configuration Version:** Enterprise 2.0  
**Validation Strategy:** Fail-Fast with Comprehensive Checks