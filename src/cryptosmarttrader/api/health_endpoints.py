#!/usr/bin/env python3
"""
FASE D - Health and Metrics API Endpoints  
Provides /health and /metrics endpoints for CI smoke tests and monitoring
"""

import time
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import PlainTextResponse
import uvicorn
from datetime import datetime
import asyncio

# Import observability components (with fallback for testing)
try:
    from ..observability.metrics import get_metrics
    from ..observability.fase_d_alerts import get_alert_manager
except ImportError:
    # For standalone testing
    import sys
    sys.path.insert(0, '/home/runner/workspace')
    from src.cryptosmarttrader.observability.metrics import get_metrics
    from src.cryptosmarttrader.observability.fase_d_alerts import get_alert_manager

logger = logging.getLogger(__name__)

# Create FastAPI app for health/metrics endpoints
health_app = FastAPI(
    title="CryptoSmartTrader V2 Health API",
    description="Health checks and metrics endpoints for FASE D observability",
    version="2.0.0"
)


@health_app.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for CI smoke tests
    Returns 200 OK if system is operational
    """
    try:
        # Get metrics instance
        metrics = get_metrics()
        alert_manager = get_alert_manager()
        
        # Basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "component": "cryptosmarttrader",
            "checks": {}
        }
        
        # Check 1: Metrics system operational
        try:
            metrics_summary = metrics.get_metrics_summary()
            health_status["checks"]["metrics"] = {
                "status": "ok",
                "total_metrics": len(metrics_summary),
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            health_status["checks"]["metrics"] = {
                "status": "error", 
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check 2: Alert system operational
        try:
            alert_status = alert_manager.get_alert_status()
            health_status["checks"]["alerts"] = {
                "status": "ok",
                "active_alerts": alert_status["active_alerts"],
                "total_conditions": alert_status["total_conditions"]
            }
        except Exception as e:
            health_status["checks"]["alerts"] = {
                "status": "error",
                "error": str(e)
            }
            health_status["status"] = "degraded"
        
        # Check 3: System resources
        try:
            import psutil
            health_status["checks"]["system"] = {
                "status": "ok", 
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            }
        except ImportError:
            health_status["checks"]["system"] = {
                "status": "warning",
                "message": "psutil not available"
            }
        except Exception as e:
            health_status["checks"]["system"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Overall health determination
        check_statuses = [check["status"] for check in health_status["checks"].values()]
        if "error" in check_statuses:
            health_status["status"] = "unhealthy"
            raise HTTPException(status_code=503, detail=health_status)
        elif "warning" in check_statuses or "degraded" in health_status["status"]:
            health_status["status"] = "degraded"
        
        return health_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


@health_app.get("/metrics", response_class=PlainTextResponse)
async def metrics_endpoint() -> str:
    """
    Prometheus metrics endpoint
    Returns metrics in Prometheus exposition format
    """
    try:
        metrics = get_metrics()
        
        # Export metrics in Prometheus format
        metrics_output = metrics.export_metrics()
        
        # Add custom headers for Prometheus
        return Response(
            content=metrics_output,
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )
        
    except Exception as e:
        logger.error(f"Metrics export failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Metrics export failed: {str(e)}"
        )


@health_app.get("/alerts")
async def alerts_endpoint() -> Dict[str, Any]:
    """
    Alert status endpoint
    Returns current alert status and active alerts
    """
    try:
        alert_manager = get_alert_manager()
        
        # Evaluate alerts before returning status
        evaluation_results = alert_manager.evaluate_alerts()
        alert_status = alert_manager.get_alert_status()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "evaluation_results": evaluation_results,
            "alert_status": alert_status
        }
        
    except Exception as e:
        logger.error(f"Alert status retrieval failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Alert status failed: {str(e)}"
        )


@health_app.get("/alerts/rules")
async def alert_rules_endpoint() -> PlainTextResponse:
    """
    Prometheus alert rules endpoint
    Returns alert rules in Prometheus format
    """
    try:
        alert_manager = get_alert_manager()
        rules_yaml = alert_manager.export_prometheus_rules()
        
        return PlainTextResponse(
            content=rules_yaml,
            media_type="text/plain"
        )
        
    except Exception as e:
        logger.error(f"Alert rules export failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Alert rules export failed: {str(e)}"
        )


@health_app.get("/status")
async def system_status() -> Dict[str, Any]:
    """
    Comprehensive system status endpoint
    Combines health, metrics, and alerts into single overview
    """
    try:
        # Get health status
        health_response = await health_check()
        
        # Get metrics summary
        metrics = get_metrics()
        metrics_summary = metrics.get_metrics_summary()
        
        # Get alert status
        alert_manager = get_alert_manager()
        alert_status = alert_manager.get_alert_status()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": health_response["status"],
            "health": health_response,
            "metrics_summary": {
                "total_orders": metrics_summary.get("total_orders_sent", 0),
                "order_errors": metrics_summary.get("total_order_errors", 0),
                "signals_received": metrics_summary.get("total_signals_received", 0),
                "portfolio_equity": metrics_summary.get("current_portfolio_equity", 0),
                "drawdown_pct": metrics_summary.get("current_drawdown_pct", 0),
                "alert_flags": {
                    "high_order_error_rate": metrics_summary.get("alert_high_order_error_rate", 0),
                    "drawdown_too_high": metrics_summary.get("alert_drawdown_too_high", 0),
                    "no_signals_timeout": metrics_summary.get("alert_no_signals_timeout", 0)
                }
            },
            "alerts": {
                "active_count": alert_status["active_alerts"],
                "total_conditions": alert_status["total_conditions"],
                "active_alerts": alert_status.get("active_alerts_detail", {})
            }
        }
        
    except Exception as e:
        logger.error(f"System status failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"System status failed: {str(e)}"
        )


def start_health_server(host: str = "0.0.0.0", port: int = 8001):
    """Start health and metrics API server"""
    logger.info(f"Starting health API server on {host}:{port}")
    
    # Configure uvicorn
    config = uvicorn.Config(
        app=health_app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
    
    server = uvicorn.Server(config)
    return server


async def run_health_server_async(host: str = "0.0.0.0", port: int = 8001):
    """Run health server asynchronously"""
    server = start_health_server(host, port)
    await server.serve()


if __name__ == "__main__":
    # For standalone testing
    import asyncio
    asyncio.run(run_health_server_async())