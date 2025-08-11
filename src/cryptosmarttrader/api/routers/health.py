"""Health Router - PR3 Style Implementation with Enterprise Extensions"""

from fastapi import APIRouter
from pydantic import BaseModel
import psutil
import time
from pathlib import Path
from typing import Dict, Any

router = APIRouter(tags=["health"])


class HealthOut(BaseModel):
    """PR3 Style Health Response Model"""
    status: str
    score: float


class HealthDetailOut(BaseModel):
    """Enterprise Extended Health Response"""
    status: str
    score: float
    timestamp: float
    details: Dict[str, Any]
    services: Dict[str, str]


@router.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    """PR3 Style Simple Health Check"""
    # TODO: haal echte metrics/health-score op uit orchestrator/metrics
    return HealthOut(status="ok", score=0.97)


@router.get("/health/detailed", response_model=HealthDetailOut)
async def health_detailed() -> HealthDetailOut:
    """Enterprise Detailed Health Check"""
    
    # System metrics
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('.')
    
    # Service checks
    services_status = {}
    
    # Check if key directories exist
    key_dirs = ["data", "models", "logs", "cache"]
    for dir_name in key_dirs:
        services_status[f"{dir_name}_dir"] = "ok" if Path(dir_name).exists() else "missing"
    
    # Calculate overall health score
    memory_score = 1.0 - (memory.percent / 100.0)
    disk_score = 1.0 - (disk.percent / 100.0)
    services_score = sum(1 for s in services_status.values() if s == "ok") / len(services_status)
    
    overall_score = (memory_score + disk_score + services_score) / 3.0
    
    # Determine status
    if overall_score >= 0.8:
        status = "ok"
    elif overall_score >= 0.5:
        status = "degraded"
    else:
        status = "fail"
    
    return HealthDetailOut(
        status=status,
        score=round(overall_score, 3),
        timestamp=time.time(),
        details={
            "memory_percent": memory.percent,
            "disk_percent": disk.percent,
            "cpu_count": psutil.cpu_count(),
            "memory_available_gb": round(memory.available / (1024**3), 2)
        },
        services=services_status
    )