#!/bin/bash
# Docker entrypoint script for CryptoSmartTrader V2
# FASE E - Production deployment orchestration

set -e

# Configuration
export PYTHONPATH="/app"
export CRYPTOSMARTTRADER_ENV="${CRYPTOSMARTTRADER_ENV:-production}"

# Logging
echo "🚀 Starting CryptoSmartTrader V2 (${CRYPTOSMARTTRADER_ENV})"
echo "📂 Working directory: $(pwd)"
echo "🐍 Python path: ${PYTHONPATH}"
echo "👤 Running as: $(whoami)"

# Health check function
health_check() {
    echo "🏥 Running health checks..."
    
    # Check Python environment
    python --version
    
    # Check required directories
    for dir in logs data models exports; do
        if [ ! -d "/app/$dir" ]; then
            echo "📁 Creating directory: $dir"
            mkdir -p "/app/$dir"
        fi
    done
    
    # Validate configuration
    python -c "
import sys
sys.path.insert(0, '/app')
from src.cryptosmarttrader.core.config import get_settings
settings = get_settings()
print(f'✅ Configuration loaded: {settings.environment}')
"
    
    echo "✅ Health checks completed"
}

# Service startup functions
start_dashboard() {
    echo "📊 Starting Streamlit dashboard on port 5000..."
    exec streamlit run secrets_dashboard.py \
        --server.port 5000 \
        --server.address 0.0.0.0 \
        --server.headless true \
        --server.enableCORS false \
        --server.enableXsrfProtection false
}

start_api() {
    echo "🔌 Starting FastAPI server on port 8001..."
    exec python -m uvicorn src.cryptosmarttrader.api.health_endpoints:health_app \
        --host 0.0.0.0 \
        --port 8001 \
        --workers 1 \
        --log-level info
}

start_metrics() {
    echo "📈 Starting Prometheus metrics server on port 8000..."
    exec python -c "
import sys
sys.path.insert(0, '/app')
from prometheus_client import start_http_server
from src.cryptosmarttrader.observability.metrics import get_metrics
import time

# Initialize metrics
metrics = get_metrics()
print('📊 Metrics initialized')

# Start Prometheus HTTP server
start_http_server(8000, addr='0.0.0.0')
print('🚀 Prometheus metrics server started on port 8000')

# Keep running
while True:
    time.sleep(30)
    print('📊 Metrics server heartbeat')
"
}

start_full_stack() {
    echo "🎯 Starting full CryptoSmartTrader V2 stack..."
    
    # Start services in background
    start_metrics &
    METRICS_PID=$!
    
    start_api &
    API_PID=$!
    
    # Start dashboard in foreground
    start_dashboard &
    DASHBOARD_PID=$!
    
    # Wait for any service to exit
    wait -n
    
    echo "⚠️ Service exited, shutting down..."
    kill $METRICS_PID $API_PID $DASHBOARD_PID 2>/dev/null || true
    exit 1
}

# Signal handlers
cleanup() {
    echo "🛑 Received shutdown signal..."
    # Kill background processes
    jobs -p | xargs -r kill
    echo "✅ Cleanup completed"
    exit 0
}

trap cleanup SIGTERM SIGINT

# Main execution
case "${1:-dashboard}" in
    "dashboard")
        health_check
        start_dashboard
        ;;
    "api")
        health_check
        start_api
        ;;
    "metrics")
        health_check
        start_metrics
        ;;
    "full")
        health_check
        start_full_stack
        ;;
    "health")
        health_check
        echo "✅ Health check completed successfully"
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo "Available commands: dashboard, api, metrics, full, health"
        exit 1
        ;;
esac