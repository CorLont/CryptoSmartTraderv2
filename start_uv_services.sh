#!/bin/bash
# Replit Multi-Service Startup Script using UV
# Based on the pattern: uv sync && (service1 & service2 & service3 & wait)

echo "🚀 CryptoSmartTrader V2 - UV Multi-Service Startup"
echo "=================================================="

# Ensure dependencies are synced
echo "📦 Syncing dependencies with uv..."
uv sync

# Check if sync was successful
if [ $? -ne 0 ]; then
    echo "❌ UV sync failed"
    exit 1
fi

echo "✅ Dependencies synced successfully"
echo ""

# Start all services in background and wait
echo "🏥 Starting Health API (port 8001)..."
echo "📊 Starting Metrics Server (port 8000)..."  
echo "🎯 Starting Main Dashboard (port 5000)..."
echo ""

# Start services with UV run
(
    uv run python api/health_endpoint.py --port 8001 &
    uv run python metrics/metrics_server.py --port 8000 &
    uv run streamlit run app_fixed_all_issues.py --server.port 5000 --server.headless true --server.address 0.0.0.0 &
    wait
)

echo "🛑 All services stopped"