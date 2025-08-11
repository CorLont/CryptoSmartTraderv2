# Replit Multi-Service Configuration

CryptoSmartTrader V2 runs multiple services simultaneously on different ports for optimal performance and separation of concerns.

## Service Architecture

| Service | Port | Purpose | Health Endpoint |
|---------|------|---------|----------------|
| **Dashboard** | 5000 | Main Streamlit trading interface | `/_stcore/health` |
| **API** | 8001 | Health checks and status API | `/health` |
| **Metrics** | 8000 | Prometheus metrics collection | `/health` |

## Current .replit Configuration

```toml
# Recommended .replit run configuration
run = "uv sync && (uv run python api/health_endpoint.py & uv run python metrics/metrics_server.py & uv run streamlit run app_fixed_all_issues.py --server.port 5000 --server.headless true --server.address 0.0.0.0 & wait)"

# Alternative workflow configuration
[[workflows.workflow]]
name = "MultiService"
author = "agent"

[[workflows.workflow.tasks]]
task = "shell.exec"
args = "python start_multi_service.py"
waitForPort = 5000

# Port mappings for Replit
[[ports]]
localPort = 5000    # Dashboard (Streamlit)
externalPort = 80   # Public access

[[ports]]
localPort = 8001    # API service
externalPort = 3000 # Internal/monitoring access

[[ports]]
localPort = 8000    # Metrics service  
externalPort = 3001 # Prometheus/monitoring access
```

## Service URLs

### External Access (via Replit)
- **Main Dashboard**: `https://{repl-name}.{username}.repl.co/` (port 80 → 5000)
- **API Service**: `https://{repl-name}.{username}.repl.co:3000/` (port 3000 → 8001)
- **Metrics Service**: `https://{repl-name}.{username}.repl.co:3001/` (port 3001 → 8000)

### Internal Access (within Replit)
- **Dashboard**: `http://localhost:5000`
- **API**: `http://localhost:8001`
- **Metrics**: `http://localhost:8000`

## Service Details

### 🎯 Dashboard Service (Port 5000)
- **File**: `app_fixed_all_issues.py`
- **Framework**: Streamlit
- **Purpose**: Main trading interface with real-time data
- **Health Check**: Streamlit built-in `/_stcore/health`
- **Features**: Portfolio management, trading signals, market analysis

### 🏥 API Service (Port 8001)
- **File**: `api/health_endpoint.py`
- **Framework**: FastAPI
- **Purpose**: Health monitoring and system status
- **Health Check**: `/health` (returns 200 OK)
- **Features**: 
  - Basic health check for Replit
  - Detailed system metrics
  - Service status monitoring
  - API documentation at `/api/docs`

### 📊 Metrics Service (Port 8000)
- **File**: `metrics/metrics_server.py`
- **Framework**: FastAPI + Prometheus
- **Purpose**: Metrics collection and monitoring
- **Health Check**: `/health`
- **Features**:
  - Prometheus metrics at `/metrics`
  - Trading performance metrics
  - System resource monitoring
  - Custom business metrics

## Multi-Service Startup

### Automatic Startup Options

#### Option 1: UV-based Startup (Recommended for .replit)
```bash
uv sync && (uv run python api/health_endpoint.py & uv run python metrics/metrics_server.py & uv run streamlit run app_fixed_all_issues.py --server.port 5000 --server.headless true --server.address 0.0.0.0 & wait)
```

#### Option 2: Coordinated Startup Script
```bash
python start_multi_service.py
```

#### Option 3: Shell Script
```bash
./start_uv_services.sh
```

All methods:
1. Start API service (port 8001)
2. Start Metrics service (port 8000) 
3. Start Dashboard service (port 5000)
4. Handle graceful shutdown
5. Provide service coordination

### Manual Service Control

Start individual services for development:

```bash
# Start API service
python api/health_endpoint.py

# Start Metrics service  
python metrics/metrics_server.py

# Start Dashboard
streamlit run app_fixed_all_issues.py --server.port 5000
```

## Health Monitoring

### Service Health Checks
All services expose health endpoints compatible with Replit's monitoring:

- **API**: `GET /health` → `{"status": "healthy", "timestamp": "..."}`
- **Metrics**: `GET /health` → `{"status": "healthy", "service": "metrics", ...}`
- **Dashboard**: `GET /_stcore/health` → Streamlit built-in health check

### Detailed Health Information
Extended health information available at:
- **API**: `GET /health/detailed` → Complete system metrics
- **Dashboard**: Real-time health widget in sidebar

## Development & Debugging

### Logs
Multi-service startup provides labeled logs:
```
[API] Starting Health & Status API on port 8001...
[METRICS] Starting Metrics Server on port 8000...
[DASHBOARD] Starting Main Trading Dashboard on port 5000...
```

### Port Management
Replit automatically manages port forwarding. The `start_multi_service.py` script:
- Binds all services to `0.0.0.0` for external access
- Implements proper startup sequencing
- Provides health check coordination
- Monitors service status

### Error Handling
- Services restart automatically on failure
- Health checks prevent cascading failures
- Graceful shutdown on interruption
- Process isolation prevents cross-service issues

## Production Considerations

### Performance
- Each service runs in separate process
- No shared state between services
- Independent scaling and monitoring
- Resource isolation

### Security
- Health endpoints provide minimal information
- API documentation restricted to development
- Metrics access can be limited to monitoring tools
- Services communicate via localhost only

### Monitoring
- Prometheus metrics for system monitoring
- Health checks for uptime monitoring
- Application logs for debugging
- Performance metrics for optimization

## Troubleshooting

### Port Conflicts
If ports are in use:
1. Check Replit's Ports panel
2. Kill existing processes: `pkill -f "streamlit\|uvicorn"`
3. Restart services: `python start_multi_service.py`

### Service Failures
1. Check service logs in console
2. Verify health endpoints
3. Review Replit Ports panel
4. Check system resources

### Connectivity Issues
1. Ensure services bind to `0.0.0.0` not `localhost`
2. Verify port mappings in Replit
3. Check firewall/security settings
4. Test internal connectivity first

---

**Note**: This configuration provides enterprise-grade service orchestration within Replit's infrastructure while maintaining development flexibility.