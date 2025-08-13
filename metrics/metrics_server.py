#!/usr/bin/env python3
"""
Metrics Server for CryptoSmartTrader
Prometheus metrics endpoint server with enterprise monitoring.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import sys

sys.path.insert(0, ".")
import sys
import os

sys.path.append(".")
from src.cryptosmarttrader.observability import (
    get_metrics_collector,
    create_alert_manager,
    setup_metrics_collector,
)


# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "component": "metrics_server", "message": "%(message)s", "module": "%(module)s"}',
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CryptoSmartTrader Metrics Server",
    description="Enterprise metrics and monitoring endpoint",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize metrics system
metrics_collector = setup_metrics_collector()
alert_manager = create_alert_manager(metrics_collector)

logger.info("Metrics server initialized with observability system")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "metrics_server", "version": "2.0.0"}


@app.get("/metrics")
async def get_prometheus_metrics():
    """Prometheus metrics endpoint."""
    try:
        metrics_data = metrics_collector.get_metrics()
        return Response(content=metrics_data, media_type="text/plain; version=0.0.4; charset=utf-8")
    except Exception as e:
        logger.error(f"Error generating metrics: {e}")
        return Response(
            content=f"# Error generating metrics: {e}\n", media_type="text/plain", status_code=500
        )


@app.get("/metrics/summary")
async def get_metrics_summary():
    """Human-readable metrics summary."""
    try:
        summary = metrics_collector.get_metrics_summary()
        return summary
    except Exception as e:
        logger.error(f"Error generating metrics summary: {e}")
        return {"error": str(e)}


@app.get("/alerts")
async def get_active_alerts():
    """Get active alerts."""
    try:
        # Evaluate current alerts
        metrics_summary = metrics_collector.get_metrics_summary()
        alert_manager.evaluate_rules(metrics_summary)

        # Return alert data
        return {
            "active_alerts": alert_manager.get_active_alerts(),
            "alert_summary": alert_manager.get_alert_summary(),
        }
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        return {"error": str(e)}


@app.post("/alerts/{alert_name}/acknowledge")
async def acknowledge_alert(alert_name: str):
    """Acknowledge an active alert."""
    try:
        alert_manager.acknowledge_alert(alert_name)
        return {"status": "acknowledged", "alert": alert_name}
    except Exception as e:
        logger.error(f"Error acknowledging alert {alert_name}: {e}")
        return {"error": str(e)}


@app.post("/alerts/{alert_name}/suppress")
async def suppress_alert(alert_name: str, duration_minutes: int = 60):
    """Suppress an alert for specified duration."""
    try:
        alert_manager.suppress_alert(alert_name, duration_minutes)
        return {"status": "suppressed", "alert": alert_name, "duration_minutes": duration_minutes}
    except Exception as e:
        logger.error(f"Error suppressing alert {alert_name}: {e}")
        return {"error": str(e)}


@app.get("/system/status")
async def get_system_status():
    """Get comprehensive system status."""
    try:
        metrics_summary = metrics_collector.get_metrics_summary()
        alert_summary = alert_manager.get_alert_summary()

        # Determine overall health
        critical_alerts = alert_summary["severity_distribution"]["critical"]
        emergency_alerts = alert_summary["severity_distribution"]["emergency"]

        if emergency_alerts > 0:
            health_status = "emergency"
        elif critical_alerts > 0:
            health_status = "critical"
        elif alert_summary["active_alerts"] > 0:
            health_status = "warning"
        else:
            health_status = "healthy"

        return {
            "overall_health": health_status,
            "metrics": metrics_summary,
            "alerts": alert_summary,
            "timestamp": metrics_summary["timestamp"],
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {"error": str(e)}


@app.on_event("startup")
async def startup_event():
    """Server startup event."""
    logger.info("Metrics server starting up")

    # Record server startup
    metrics_collector.record_signal_received("system", "server_startup", service="metrics_server")


@app.on_event("shutdown")
async def shutdown_event():
    """Server shutdown event."""
    logger.info("Metrics server shutting down")


if __name__ == "__main__":
    try:
        logger.info("Starting CryptoSmartTrader Metrics Server on port 8000")
        uvicorn.run(
            "metrics_server:app",
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=True,
            reload=False,
        )
    except KeyboardInterrupt:
        logger.info("Metrics server stopped by user")
    except Exception as e:
        logger.error(f"Metrics server failed to start: {e}")
        sys.exit(1)
