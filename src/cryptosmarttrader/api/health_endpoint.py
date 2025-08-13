#!/usr/bin/env python3
"""
Health Endpoint API - Simple FastAPI service for health checks
"""

import json
import time
import sys
import os

sys.path.append(".")
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
import uvicorn
import psutil

# Import unified metrics system
try:
    from src.cryptosmarttrader.observability.unified_metrics import get_metrics

    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


app = FastAPI(
    title="CryptoSmartTrader V2 Health API",
    description="Health monitoring and status endpoints",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)


class HealthChecker:
    """System health checker"""

    def __init__(self):
        self.start_time = time.time()
        self.repo_root = Path.cwd()

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get basic system metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            return {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent,
                },
                "disk": {
                    "total": disk.total,
                    "free": disk.free,
                    "used": disk.used,
                    "percent": (disk.used / disk.total) * 100,
                },
            }
        except Exception:
            return {"error": "Could not retrieve system metrics"}

    def check_services(self) -> Dict[str, str]:
        """Check if other services are responsive"""
        services = {"dashboard": "http://localhost:5000", "metrics": "http://localhost:8000"}

        status = {}
        for service, url in services.items():
            try:
                import requests

                response = requests.get(f"{url}/health", timeout=2)
                status[service] = "healthy" if response.status_code == 200 else "unhealthy"
            except Exception:
                status[service] = "unavailable"

        return status

    def get_health_file_status(self) -> Dict[str, Any]:
        """Check health status from file if exists"""
        health_file = self.repo_root / "health_status.json"

        if health_file.exists():
            try:
                with open(health_file) as f:
                    return json.load(f)
            except Exception:
                return {"error": "Could not read health file"}

        return {"status": "no_health_file"}

    def get_uptime(self) -> float:
        """Get service uptime in seconds"""
        return time.time() - self.start_time


health_checker = HealthChecker()


@app.get("/health")
async def health_check():
    """Basic health check endpoint - returns 200 OK for Replit"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "cryptosmarttrader-api",
            "version": "2.0.0",
        },
    )


@app.get("/health/detailed")
async def detailed_health():
    """Detailed health information"""
    try:
        uptime = health_checker.get_uptime()
        system_metrics = health_checker.get_system_metrics()
        services_status = health_checker.check_services()
        health_file_status = health_checker.get_health_file_status()

        response = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "uptime_formatted": f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s",
            "system": system_metrics,
            "services": services_status,
            "application_health": health_file_status,
        }

        # Determine overall status
        if any(status == "unhealthy" for status in services_status.values()):
            response["status"] = "degraded"

        return JSONResponse(status_code=200, content=response)

    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            },
        )


@app.get("/status")
async def service_status():
    """Service status endpoint"""
    return {
        "service": "cryptosmarttrader-api",
        "status": "running",
        "port": 8001,
        "endpoints": {
            "health": "/health",
            "detailed_health": "/health/detailed",
            "status": "/status",
            "docs": "/api/docs",
        },
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CryptoSmartTrader V2 Health API",
        "version": "2.0.0",
        "docs": "/api/docs",
        "health": "/health",
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def get_metrics_endpoint():
    """Prometheus metrics endpoint for observability"""
    try:
        if METRICS_AVAILABLE:
            metrics = get_metrics()
            prometheus_output = metrics.get_prometheus_metrics()

            return PlainTextResponse(
                content=prometheus_output,
                headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"},
            )
        else:
            return PlainTextResponse(content="# Metrics system not available\n", status_code=503)

    except Exception as e:
        return PlainTextResponse(content=f"# Error: {str(e)}\n", status_code=500)


@app.get("/metrics/summary")
async def get_metrics_summary() -> JSONResponse:
    """Critical alerts summary for 500% target system"""
    try:
        summary = {
            "observability_status": "operational",
            "timestamp": datetime.now().isoformat(),
            "critical_alerts": {
                "order_error_rate": "< 5%",
                "slippage_bps_p95": "< 50 bps",
                "drawdown_pct": "< 8%",
                "signal_age_minutes": "< 30 min",
            },
            "monitoring_targets": [
                "HighOrderErrorRate",
                "ExcessiveSlippage",
                "HighDrawdown",
                "NoSignalsReceived",
            ],
        }

        if METRICS_AVAILABLE:
            metrics = get_metrics()
            summary.update(metrics.get_metrics_summary())

        return JSONResponse(content=summary)

    except Exception as e:
        return JSONResponse(content={"error": f"Failed to get summary: {str(e)}"}, status_code=500)


def main():
    """Main entry point for the health API"""
    print("🏥 Starting CryptoSmartTrader V2 Health API on port 8001")
    print("📊 Metrics endpoint: http://localhost:8001/metrics")
    print("📋 Summary endpoint: http://localhost:8001/metrics/summary")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False, log_level="info")


if __name__ == "__main__":
    main()
