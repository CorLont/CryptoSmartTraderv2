#!/usr/bin/env python3
"""
Metrics Server - Prometheus metrics endpoint
"""

import time
from datetime import datetime
from pathlib import Path

from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.core import CollectorRegistry
from fastapi import FastAPI, Response
import uvicorn


# Create custom registry
registry = CollectorRegistry()

# Define metrics
REQUEST_COUNT = Counter(
    'cryptotrader_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status'],
    registry=registry
)

REQUEST_DURATION = Histogram(
    'cryptotrader_request_duration_seconds',
    'Request duration in seconds',
    ['method', 'endpoint'],
    registry=registry
)

ACTIVE_TRADES = Gauge(
    'cryptotrader_active_trades',
    'Number of active trades',
    registry=registry
)

PORTFOLIO_VALUE = Gauge(
    'cryptotrader_portfolio_value_usd',
    'Total portfolio value in USD',
    registry=registry
)

CONFIDENCE_SCORE = Gauge(
    'cryptotrader_confidence_score',
    'Current model confidence score',
    registry=registry
)

SYSTEM_HEALTH_SCORE = Gauge(
    'cryptotrader_system_health_score',
    'Overall system health score (0-100)',
    registry=registry
)

API_CALLS_TOTAL = Counter(
    'cryptotrader_api_calls_total',
    'Total API calls to exchanges',
    ['exchange', 'endpoint'],
    registry=registry
)

PREDICTION_ACCURACY = Gauge(
    'cryptotrader_prediction_accuracy',
    'Model prediction accuracy',
    ['model_type', 'horizon'],
    registry=registry
)


app = FastAPI(
    title="CryptoSmartTrader V2 Metrics",
    description="Prometheus metrics for monitoring",
    version="2.0.0"
)


class MetricsCollector:
    """Collect and update metrics from system state"""
    
    def __init__(self):
        self.repo_root = Path.cwd()
        self.last_update = time.time()
    
    def update_metrics(self):
        """Update all metrics from current system state"""
        try:
            # Update portfolio metrics if health file exists
            health_file = self.repo_root / "health_status.json"
            if health_file.exists():
                import json
                with open(health_file) as f:
                    health_data = json.load(f)
                
                # Update system health score
                if 'overall_score' in health_data:
                    SYSTEM_HEALTH_SCORE.set(health_data['overall_score'])
                
                # Update trading metrics if available
                if 'trading_enabled' in health_data:
                    ACTIVE_TRADES.set(1 if health_data['trading_enabled'] else 0)
            
            # Set some example metrics (in production, these would come from real data)
            CONFIDENCE_SCORE.set(0.85)  # 85% confidence
            PORTFOLIO_VALUE.set(125000.0)  # $125K portfolio
            
            # Update prediction accuracy metrics
            PREDICTION_ACCURACY.labels(model_type='lstm', horizon='1h').set(0.82)
            PREDICTION_ACCURACY.labels(model_type='lstm', horizon='24h').set(0.78)
            PREDICTION_ACCURACY.labels(model_type='ensemble', horizon='7d').set(0.75)
            
            self.last_update = time.time()
            
        except Exception as e:
            print(f"Error updating metrics: {e}")


metrics_collector = MetricsCollector()


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    # Update metrics before serving
    metrics_collector.update_metrics()
    
    # Generate Prometheus format
    metrics_data = generate_latest(registry)
    
    return Response(
        content=metrics_data,
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/health")
async def health_check():
    """Health check for metrics service"""
    return {
        "status": "healthy",
        "service": "metrics",
        "timestamp": datetime.now().isoformat(),
        "last_metrics_update": datetime.fromtimestamp(metrics_collector.last_update).isoformat()
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "CryptoSmartTrader V2 Metrics",
        "version": "2.0.0",
        "endpoints": {
            "metrics": "/metrics",
            "health": "/health"
        },
        "prometheus_endpoint": "/metrics"
    }


def main():
    """Main entry point for the metrics server"""
    print("📊 Starting CryptoSmartTrader V2 Metrics Server on port 8000")
    
    # Update initial metrics
    metrics_collector.update_metrics()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()