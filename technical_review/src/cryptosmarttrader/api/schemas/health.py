#!/usr/bin/env python3
"""
Health API Schemas - System health and monitoring DTOs
"""

from datetime import datetime
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

from .common import BaseResponse, StatusInfo


class SystemStatus(BaseModel):
    """System component status"""
    database: StatusInfo = Field(description="Database connection status")
    exchange_apis: StatusInfo = Field(description="Exchange API status")
    ml_models: StatusInfo = Field(description="ML models status")
    cache: StatusInfo = Field(description="Cache system status")


class HealthResponse(BaseResponse):
    """Health check response"""
    status: str = Field(description="Overall system health", example="healthy")
    score: float = Field(description="Health score 0-100", example=95.5, ge=0, le=100)
    version: str = Field(description="API version", example="2.0.0")
    uptime_seconds: float = Field(description="Server uptime in seconds", example=3600.5)
    components: SystemStatus = Field(description="Component health status")
    checks: Dict[str, Any] = Field(description="Detailed health checks")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "timestamp": "2025-01-11T10:00:00Z",
                "status": "healthy",
                "score": 95.5,
                "version": "2.0.0",
                "uptime_seconds": 3600.5,
                "components": {
                    "database": {
                        "status": "healthy",
                        "message": "Connected",
                        "last_updated": "2025-01-11T10:00:00Z"
                    },
                    "exchange_apis": {
                        "status": "healthy",
                        "message": "All exchanges responding",
                        "last_updated": "2025-01-11T10:00:00Z"
                    },
                    "ml_models": {
                        "status": "healthy",
                        "message": "Models loaded and ready",
                        "last_updated": "2025-01-11T10:00:00Z"
                    },
                    "cache": {
                        "status": "healthy",
                        "message": "Cache operational",
                        "last_updated": "2025-01-11T10:00:00Z"
                    }
                },
                "checks": {
                    "disk_usage": 45.2,
                    "memory_usage": 67.8,
                    "cpu_usage": 23.1,
                    "active_connections": 42
                }
            }
        }


class LivenessResponse(BaseResponse):
    """Liveness probe response"""
    alive: bool = Field(description="Service is alive", example=True)


class ReadinessResponse(BaseResponse):
    """Readiness probe response"""
    ready: bool = Field(description="Service is ready to accept traffic", example=True)
    dependencies_ready: bool = Field(description="All dependencies are ready", example=True)
    startup_complete: bool = Field(description="Startup sequence completed", example=True)
