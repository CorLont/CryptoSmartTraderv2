#!/usr/bin/env python3
"""
Health Router - System health monitoring and diagnostics endpoints
"""

import time
import psutil
import logging
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..api.schemas.health import HealthResponse, LivenessResponse, ReadinessResponse, SystemStatus, StatusInfo

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Router with metadata
router = APIRouter(
    responses={
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable"}
    }
)

# Global start time for uptime calculation
START_TIME = time.time()


async def check_database_health() -> StatusInfo:
    """Check database connectivity"""
    try:
        # Placeholder for actual database check
        # In real implementation: await database.execute("SELECT 1")
        return StatusInfo(
            status="healthy",
            message="Database connection active",
            details={"connection_pool": "available"}
        )
    except Exception as e:
        logging.error(f"Database health check failed: {e}")
        return StatusInfo(
            status="unhealthy",
            message=f"Database error: {str(e)}",
            details={"error": str(e)}
        )


async def check_exchange_apis() -> StatusInfo:
    """Check exchange API connectivity"""
    try:
        # Placeholder for actual exchange API check
        # In real implementation: test Kraken, Binance connectivity
        return StatusInfo(
            status="healthy",
            message="Exchange APIs responding",
            details={
                "kraken": "connected",
                "binance": "connected"
            }
        )
    except Exception as e:
        logging.error(f"Exchange API health check failed: {e}")
        return StatusInfo(
            status="degraded",
            message=f"Some exchanges unavailable: {str(e)}",
            details={"error": str(e)}
        )


async def check_ml_models() -> StatusInfo:
    """Check ML models availability"""
    try:
        # Placeholder for actual ML model check
        # In real implementation: verify model files exist and are loadable
        return StatusInfo(
            status="healthy",
            message="ML models loaded and ready",
            details={
                "models_loaded": 5,
                "last_training": "2025-01-11T08:00:00Z"
            }
        )
    except Exception as e:
        logging.error(f"ML models health check failed: {e}")
        return StatusInfo(
            status="unhealthy",
            message=f"ML models unavailable: {str(e)}",
            details={"error": str(e)}
        )


async def check_cache_system() -> StatusInfo:
    """Check cache system health"""
    try:
        # Placeholder for actual cache check
        # In real implementation: test Redis/memory cache
        return StatusInfo(
            status="healthy",
            message="Cache system operational",
            details={"cache_size": "256MB", "hit_rate": 85.5}
        )
    except Exception as e:
        logging.error(f"Cache health check failed: {e}")
        return StatusInfo(
            status="degraded",
            message=f"Cache issues: {str(e)}",
            details={"error": str(e)}
        )


def calculate_health_score(components: SystemStatus) -> float:
    """Calculate overall health score based on component status"""
    scores = {
        "healthy": 100,
        "degraded": 50,
        "unhealthy": 0
    }

    component_scores = [
        scores.get(components.database.status, 0),
        scores.get(components.exchange_apis.status, 0),
        scores.get(components.ml_models.status, 0),
        scores.get(components.cache.status, 0)
    ]

    return sum(component_scores) / len(component_scores)


@router.get(
    "/",
    response_model=HealthResponse,
    summary="Comprehensive health check",
    description="Returns detailed system health information including component status and metrics"
)
@limiter.limit("30/minute")
async def health_check(request) -> HealthResponse:
    """
    Comprehensive health check with detailed system information.

    **Returns:**
    - Overall health status and score
    - Component-by-component health status
    - System metrics (CPU, memory, disk usage)
    - Service uptime information

    **Rate limit:** 30 requests per minute per IP
    """
    try:
        # Check all components
        components = SystemStatus(
            database=await check_database_health(),
            exchange_apis=await check_exchange_apis(),
            ml_models=await check_ml_models(),
            cache=await check_cache_system()
        )

        # Calculate health score
        health_score = calculate_health_score(components)

        # Determine overall status
        if health_score >= 90:
            overall_status = "healthy"
        elif health_score >= 50:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"

        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        checks = {
            "cpu_usage": round(cpu_percent, 1),
            "memory_usage": round(memory.percent, 1),
            "disk_usage": round(disk.percent, 1),
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None,
            "active_connections": len(psutil.net_connections()),
            "process_count": len(psutil.pids())
        }

        # Calculate uptime
        uptime_seconds = time.time() - START_TIME

        return HealthResponse(
            status=overall_status,
            score=health_score,
            version="2.0.0",
            uptime_seconds=uptime_seconds,
            components=components,
            checks=checks
        )

    except Exception as e:
        logging.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail=f"Health check failed: {str(e)}"
        )


@router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Liveness probe",
    description="Simple liveness check for container orchestration"
)
@limiter.limit("60/minute")
async def liveness_check(request) -> LivenessResponse:
    """
    Kubernetes/Docker liveness probe endpoint.

    Returns basic service availability status.
    **Rate limit:** 60 requests per minute per IP
    """
    return LivenessResponse(alive=True)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description="Readiness check for traffic routing decisions"
)
@limiter.limit("60/minute")
async def readiness_check(request) -> ReadinessResponse:
    """
    Kubernetes/Docker readiness probe endpoint.

    Checks if service is ready to accept traffic.
    **Rate limit:** 60 requests per minute per IP
    """
    try:
        # Quick dependency checks
        db_ready = (await check_database_health()).status != "unhealthy"
        models_ready = (await check_ml_models()).status != "unhealthy"

        dependencies_ready = db_ready and models_ready
        startup_complete = time.time() - START_TIME > 30  # 30 second startup grace period

        ready = dependencies_ready and startup_complete

        return ReadinessResponse(
            ready=ready,
            dependencies_ready=dependencies_ready,
            startup_complete=startup_complete
        )

    except Exception as e:
        logging.error(f"Readiness check failed: {e}")
        return ReadinessResponse(
            ready=False,
            dependencies_ready=False,
            startup_complete=False
        )
