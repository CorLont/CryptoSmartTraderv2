# Replit Multi-Service Configuration Guide
## CryptoSmartTrader V2 - Enterprise Setup

### Recommended .replit Configuration
**Note:** Direct editing of .replit file is restricted. This is the recommended configuration for multi-service deployment:

```ini
# Multi-service run command voor CryptoSmartTrader V2
run = "uv sync && (uv run streamlit run app_fixed_all_issues.py --server.port 5000 --server.headless true & wait)"

# Alternative full multi-service (indien API en metrics modules beschikbaar):
# run = "uv sync && (uv run python api/main.py --port 8001 & uv run streamlit run app_fixed_all_issues.py --server.port 5000 --server.headless true & uv run python -m orchestration.metrics & wait)"
```

### Port Configuration
Current port mapping in .replit:
```toml
[[ports]]
localPort = 5000    # Streamlit Dashboard
externalPort = 80   # Public access

[[ports]]
localPort = 5001    # Available for API
externalPort = 3000

[[ports]]
localPort = 5002    # Available for Metrics
externalPort = 3001

[[ports]]
localPort = 5003    # Available for additional services
externalPort = 3002
```

### Service Overview
- **Dashboard (Port 5000):** Main Streamlit interface - `app_fixed_all_issues.py`
- **API (Port 8001):** FastAPI service (when available) - `api/main.py`
- **Metrics (Port 8000):** Prometheus metrics endpoint
- **Public Access:** Primary service accessible via external port 80

### Multi-Service Benefits
- **Concurrent execution:** Multiple services running simultaneously
- **Service isolation:** Each service on dedicated port
- **Public endpoint:** Single main access point for users
- **Additional ports:** Managed via Replit Ports panel
- **Automatic sync:** Dependencies installed before startup

### Manual Configuration Steps
1. Navigate to Replit project settings
2. Update run command in .replit file (if editing allowed)
3. Configure additional ports via Ports panel
4. Verify service startup via console logs

### Current Limitations
- .replit file editing restricted in current environment
- Multi-service configuration requires manual setup
- Port management through Replit Ports panel interface