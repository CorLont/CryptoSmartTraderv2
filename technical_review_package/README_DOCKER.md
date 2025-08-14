# Docker Deployment Guide
## CryptoSmartTrader V2 Container & Runtime Configuration

### 🐳 Container Architecture

#### Multi-Stage Production Dockerfile
- **Base Image**: `python:3.11.10-slim-bookworm` (pinned for security)
- **Non-Root User**: Security hardened with trader user (UID 1000)
- **Health Checks**: Built-in health monitoring every 30 seconds
- **Multi-Service**: Dashboard (5000) + API (8001) + Metrics (8000)

#### Security Features
- Non-root container execution
- Read-only root filesystem support
- Capability dropping (ALL capabilities removed)
- Resource limits and memory constraints
- Security scanning with pinned dependencies

### 🚀 Quick Start

#### Prerequisites
```bash
# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

#### Single Container Deployment
```bash
# Build and deploy
./scripts/deploy.sh container

# Check status
./scripts/deploy.sh status
```

#### Docker Compose Deployment (Recommended)
```bash
# Deploy with monitoring stack
./scripts/deploy.sh compose

# Verify all services
docker-compose ps
```

### 📋 Configuration Management

#### Environment Variables (Pydantic Settings)
```bash
# Required API keys (fail-fast validation)
KRAKEN_API_KEY=your_kraken_api_key
KRAKEN_SECRET=your_kraken_secret
OPENAI_API_KEY=your_openai_key

# Optional services
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_gemini_key

# Runtime configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
PYTHONUNBUFFERED=1
```

#### Fail-Fast Startup Validation
- All required environment variables validated at startup
- Missing API keys cause immediate container exit
- Comprehensive logging of validation results
- Health checks confirm service readiness

### 🏥 Health Checks & Probes

#### Docker Health Check
```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1
```

#### Kubernetes Probes
```yaml
# Liveness probe
livenessProbe:
  httpGet:
    path: /health
    port: 8001
  initialDelaySeconds: 30
  periodSeconds: 30

# Readiness probe  
readinessProbe:
  httpGet:
    path: /health
    port: 8001
  initialDelaySeconds: 15
  periodSeconds: 10

# Startup probe (slow initialization)
startupProbe:
  httpGet:
    path: /health
    port: 8001
  initialDelaySeconds: 10
  periodSeconds: 10
  failureThreshold: 12  # 2 minutes
```

### ☸️ Kubernetes Deployment

#### Quick K8s Deployment
```bash
# Create namespace and secrets
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secret.yaml  # Configure secrets first

# Deploy application
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check status
kubectl get pods -n cryptosmarttrader
kubectl logs -f deployment/cryptosmarttrader -n cryptosmarttrader
```

#### Production K8s Configuration
- **Replicas**: 1 (stateful trading system)
- **Rolling Updates**: Zero-downtime deployments
- **Resource Limits**: 4GB memory, 2 CPU cores
- **Persistent Storage**: 10GB data, 20GB models, 5GB logs
- **Security Context**: Non-root, capability dropping

### 📊 Service Endpoints

#### Port Configuration
```yaml
Dashboard:  5000  # Streamlit web interface
Health API: 8001  # Health checks and status
Metrics:    8000  # Prometheus metrics
Grafana:    3000  # Monitoring dashboard (compose only)
Prometheus: 9090  # Metrics storage (compose only)
```

#### Health Endpoint Response
```json
{
  "status": "healthy",
  "timestamp": "2025-01-13T10:30:45.123456",
  "service": "cryptosmarttrader-api",
  "version": "2.0.0"
}
```

### 🔧 Operational Commands

#### Container Management
```bash
# View logs
docker logs cryptosmarttrader-main

# Follow logs in real-time
docker logs -f cryptosmarttrader-main

# Execute shell in container
docker exec -it cryptosmarttrader-main bash

# Check resource usage
docker stats cryptosmarttrader-main
```

#### Docker Compose Operations
```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# Rebuild and restart
docker-compose up -d --build

# View service logs
docker-compose logs -f cryptosmarttrader
```

#### Kubernetes Operations
```bash
# Check pod status
kubectl get pods -n cryptosmarttrader

# View logs
kubectl logs -f deployment/cryptosmarttrader -n cryptosmarttrader

# Port forwarding for local access
kubectl port-forward svc/cryptosmarttrader-dashboard 5000:5000 -n cryptosmarttrader

# Scale deployment (not recommended for trading system)
kubectl scale deployment cryptosmarttrader --replicas=1 -n cryptosmarttrader
```

### 🔍 Monitoring & Observability

#### Prometheus Metrics
- **Trading Metrics**: Orders, fills, PnL, positions
- **System Metrics**: Memory, CPU, network, errors
- **Application Metrics**: Request duration, success rates
- **Custom Metrics**: Model predictions, market data health

#### Grafana Dashboards
- System performance monitoring
- Trading activity visualization
- Error rate and latency tracking
- Resource utilization trends

#### Alert Configuration
- Service down alerts (critical)
- High memory/CPU usage (warning)
- Trading system errors (critical)
- API failure rate monitoring (warning)

### 🛡️ Security Considerations

#### Container Security
- Non-root user execution (UID 1000)
- Minimal base image (slim-bookworm)
- No unnecessary packages or tools
- Security scanning with Bandit integration
- Secrets management via environment variables

#### Network Security
- Internal service communication only
- Exposed ports documented and minimal
- Health checks on internal endpoints
- No privileged container access

#### Data Security
- Persistent volume encryption recommended
- Log rotation and retention policies
- Secret rotation capabilities
- Audit logging for all operations

### 🚨 Troubleshooting

#### Common Issues
```bash
# Container won't start
docker logs cryptosmarttrader-main
# Check for missing environment variables

# Health check failing
curl http://localhost:8001/health
# Verify API service is running

# High memory usage
docker stats
# Check for memory leaks in logs

# Kubernetes pod crashes
kubectl describe pod <pod-name> -n cryptosmarttrader
# Check resource limits and probes
```

#### Recovery Procedures
1. **Service Restart**: `docker-compose restart cryptosmarttrader`
2. **Clean Restart**: `docker-compose down && docker-compose up -d`
3. **Volume Reset**: Remove persistent volumes if corrupted
4. **Image Rebuild**: `./scripts/deploy.sh build` for code changes

### 📈 Performance Optimization

#### Resource Tuning
- Memory: 2GB minimum, 4GB recommended
- CPU: 1 core minimum, 2 cores for ML workloads
- Storage: SSD recommended for model loading
- Network: Low latency connection for market data

#### Caching Strategy
- Model caching in memory
- Market data caching with TTL
- Configuration caching at startup
- Volume mounting for persistent cache

This containerized deployment provides enterprise-grade reliability, security, and observability for the CryptoSmartTrader V2 system.