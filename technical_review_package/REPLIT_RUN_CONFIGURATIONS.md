# Replit Run Configurations for CryptoSmartTrader V2

## Recommended .replit Run Configuration

```toml
[deployment]
run = "uv sync && (uv run python api/health_endpoint.py & uv run python metrics/metrics_server.py & uv run streamlit run app_fixed_all_issues.py --server.port 5000 --server.headless true --server.address 0.0.0.0 & wait)"
```

## Configuration Breakdown

### 1. Dependency Management
```bash
uv sync
```
- Ensures all dependencies are installed and synchronized
- Fast dependency resolution and installation
- Consistent environment across runs

### 2. Multi-Service Background Execution
```bash
(service1 & service2 & service3 & wait)
```
- **`&`**: Runs each service in background
- **`wait`**: Waits for all background processes to complete
- **`()`**: Groups commands for proper process management

### 3. Service Definitions

#### API Service (Port 8001)
```bash
uv run python api/health_endpoint.py
```
- Health monitoring and status endpoints
- FastAPI-based RESTful API
- Automatic health checks for Replit

#### Metrics Service (Port 8000)
```bash
uv run python metrics/metrics_server.py
```
- Prometheus metrics collection
- System performance monitoring
- Trading analytics metrics

#### Dashboard Service (Port 5000) 
```bash
uv run streamlit run app_fixed_all_issues.py --server.port 5000 --server.headless true --server.address 0.0.0.0
```
- Main trading interface
- **`--server.headless true`**: Optimized for Replit deployment
- **`--server.address 0.0.0.0`**: Accessible from external connections
- **`--server.port 5000`**: Primary port for web access

## Alternative Configurations

### Option 1: Simple Sequential Start
```toml
run = "uv sync && python start_multi_service.py"
```

### Option 2: Shell Script Approach
```toml
run = "uv sync && ./start_uv_services.sh"
```

### Option 3: Individual Service Development
```toml
# For API development only
run = "uv sync && uv run python api/health_endpoint.py"

# For Dashboard development only  
run = "uv sync && uv run streamlit run app_fixed_all_issues.py --server.port 5000"

# For Metrics development only
run = "uv sync && uv run python metrics/metrics_server.py"
```

## Port Configuration

```toml
[[ports]]
localPort = 5000    # Dashboard (Primary)
externalPort = 80   # Public web access

[[ports]]
localPort = 8001    # API Service
externalPort = 3000 # API access

[[ports]]
localPort = 8000    # Metrics Service  
externalPort = 3001 # Metrics access
```

## Benefits of UV-based Configuration

### 1. **Fast Startup**
- UV's rapid dependency resolution
- Parallel service initialization
- Optimized for Replit's environment

### 2. **Process Management**
- Proper background process handling
- Graceful shutdown with `wait`
- Signal propagation to all services

### 3. **Development Flexibility**
- Easy to modify individual services
- Simple debugging and testing
- Clear service separation

### 4. **Production Ready**
- Headless Streamlit for server deployment
- Health endpoints for monitoring
- Metrics collection for observability

## Troubleshooting

### Common Issues

1. **Port Conflicts**
```bash
# Check what's using ports
lsof -i :5000 -i :8001 -i :8000

# Kill conflicting processes
pkill -f streamlit
pkill -f uvicorn
```

2. **UV Sync Failures**
```bash
# Clear UV cache
uv cache clean

# Reinstall dependencies
rm uv.lock
uv sync
```

3. **Service Health Check**
```bash
# Test all services
curl http://localhost:5000/_stcore/health
curl http://localhost:8001/health  
curl http://localhost:8000/health
```

### Debug Mode Configuration

For development and debugging:

```toml
run = "uv sync && (uv run python api/health_endpoint.py --debug & uv run python metrics/metrics_server.py --debug & uv run streamlit run app_fixed_all_issues.py --server.port 5000 --server.headless false & wait)"
```

Changes:
- `--debug` flags for verbose logging
- `--server.headless false` for Streamlit UI in development

## Performance Optimization

### Resource Management
- Services run in separate processes
- Memory isolation between components
- Independent scaling capabilities

### Startup Sequence
1. **UV Sync** (2-5 seconds)
2. **API Service** starts first (fastest startup)
3. **Metrics Service** starts second
4. **Dashboard Service** starts last (longest startup)
5. **Wait** keeps all processes alive

### Health Monitoring
- All services provide `/health` endpoints
- Replit automatically monitors port 5000
- Internal health checks between services
- Graceful degradation on service failures

---

**Note**: This configuration provides optimal Replit compatibility while maintaining enterprise-grade service orchestration and monitoring capabilities.