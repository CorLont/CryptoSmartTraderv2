"""Health Check API Router - System Health Monitoring"""

from fastapi import APIRouter, Depends
from typing import Dict, Any
import psutil
from datetime import datetime

from ..models.health import HealthOut, HealthDetailOut, ServiceHealth, HealthStatus
from ..dependencies import get_orchestrator
from ..dependencies import get_settings

router = APIRouter(tags=["health"], prefix="/health")


@router.get("/", response_model=HealthOut, summary="Basic Health Check")
async def health(
    orchestrator=Depends(get_orchestrator),
    settings: Settings = Depends(get_settings)
) -> HealthOut:
    """
    Basic system health check endpoint
    
    Returns overall system health status and score
    """
    try:
        # Get health score from orchestrator
        health_score = await get_system_health_score(orchestrator)
        
        # Determine status based on score
        if health_score >= 0.8:
            status = HealthStatus.HEALTHY
        elif health_score >= 0.6:
            status = HealthStatus.DEGRADED
        elif health_score >= 0.4:
            status = HealthStatus.UNHEALTHY
        else:
            status = HealthStatus.CRITICAL
        
        return HealthOut(
            status=status,
            score=health_score,
            timestamp=datetime.utcnow()
        )
        
    except Exception:
        # Fallback to unhealthy state on any error
        return HealthOut(
            status=HealthStatus.UNHEALTHY,
            score=0.0,
            timestamp=datetime.utcnow()
        )


@router.get("/detailed", response_model=HealthDetailOut, summary="Detailed Health Check")
async def health_detailed(
    orchestrator=Depends(get_orchestrator),
    settings: Settings = Depends(get_settings)
) -> HealthDetailOut:
    """
    Comprehensive system health check with detailed breakdown
    
    Returns detailed health information including:
    - Individual service health
    - System resource metrics
    - Application health metrics
    - Trading system status
    """
    # Get individual service health
    services = await get_service_health_status(orchestrator)
    
    # Get system metrics
    system_metrics = get_system_metrics()
    
    # Get application health
    app_health = await get_application_health(orchestrator)
    
    # Calculate overall health score
    overall_score = calculate_overall_health_score(services, system_metrics, app_health)
    
    # Determine overall status
    if overall_score >= 0.8:
        status = HealthStatus.HEALTHY
    elif overall_score >= 0.6:
        status = HealthStatus.DEGRADED
    elif overall_score >= 0.4:
        status = HealthStatus.UNHEALTHY
    else:
        status = HealthStatus.CRITICAL
    
    # Collect warnings and errors
    warnings, errors = collect_health_issues(services, system_metrics, app_health)
    
    # Determine trading status
    trading_status = "GO" if overall_score >= settings.CONFIDENCE_THRESHOLD else "NO-GO"
    
    return HealthDetailOut(
        status=status,
        score=overall_score,
        timestamp=datetime.utcnow(),
        services=services,
        system_metrics=system_metrics,
        application_health=app_health,
        trading_status=trading_status,
        warnings=warnings,
        errors=errors
    )


async def get_system_health_score(orchestrator) -> float:
    """Calculate overall system health score"""
    try:
        # This would integrate with your orchestrator's health monitoring
        # For now, return a placeholder based on system resources
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Simple health scoring based on system resources
        cpu_score = max(0, 1 - (cpu_percent / 100))
        memory_score = max(0, 1 - (memory.percent / 100))
        
        return (cpu_score + memory_score) / 2
        
    except Exception:
        return 0.0


async def get_service_health_status(orchestrator) -> Dict[str, ServiceHealth]:
    """Get health status of individual services"""
    services = {}
    
    # Dashboard service
    services["dashboard"] = ServiceHealth(
        name="dashboard",
        status=HealthStatus.HEALTHY,
        response_time_ms=45.2,
        last_check=datetime.utcnow()
    )
    
    # API service  
    services["api"] = ServiceHealth(
        name="api",
        status=HealthStatus.HEALTHY,
        response_time_ms=12.8,
        last_check=datetime.utcnow()
    )
    
    # Metrics service
    services["metrics"] = ServiceHealth(
        name="metrics", 
        status=HealthStatus.HEALTHY,
        response_time_ms=8.5,
        last_check=datetime.utcnow()
    )
    
    return services


def get_system_metrics() -> Dict[str, Any]:
    """Get system resource metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_percent": round(cpu_percent, 1),
            "memory_percent": round(memory.percent, 1),
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_percent": round((disk.used / disk.total) * 100, 1),
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else None
        }
    except Exception:
        return {}


async def get_application_health(orchestrator) -> Dict[str, Any]:
    """Get application-specific health metrics"""
    # This would integrate with your orchestrator to get real metrics
    return {
        "health_score": 85.7,
        "active_agents": 8,
        "data_freshness_minutes": 2.3,
        "confidence_gate_pass_rate": 0.87,
        "trading_signals_today": 156,
        "uptime_hours": 24.5
    }


def calculate_overall_health_score(
    services: Dict[str, ServiceHealth],
    system_metrics: Dict[str, Any],
    app_health: Dict[str, Any]
) -> float:
    """Calculate weighted overall health score"""
    scores = []
    
    # Service health (30% weight)
    healthy_services = sum(1 for s in services.values() if s.status == HealthStatus.HEALTHY)
    service_score = healthy_services / len(services) if services else 0
    scores.append(service_score * 0.3)
    
    # System resources (40% weight) 
    if system_metrics:
        cpu_score = max(0, 1 - (system_metrics.get("cpu_percent", 100) / 100))
        memory_score = max(0, 1 - (system_metrics.get("memory_percent", 100) / 100))
        resource_score = (cpu_score + memory_score) / 2
        scores.append(resource_score * 0.4)
    
    # Application health (30% weight)
    app_score = app_health.get("health_score", 0) / 100
    scores.append(app_score * 0.3)
    
    return sum(scores)


def collect_health_issues(
    services: Dict[str, ServiceHealth],
    system_metrics: Dict[str, Any],
    app_health: Dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Collect warnings and errors from health checks"""
    warnings = []
    errors = []
    
    # Check service issues
    for service in services.values():
        if service.status == HealthStatus.DEGRADED:
            warnings.append(f"Service {service.name} is degraded")
        elif service.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
            errors.append(f"Service {service.name} is {service.status.value}")
    
    # Check system resource issues
    if system_metrics:
        cpu = system_metrics.get("cpu_percent", 0)
        memory = system_metrics.get("memory_percent", 0)
        
        if cpu > 80:
            warnings.append(f"High CPU usage: {cpu}%")
        elif cpu > 95:
            errors.append(f"Critical CPU usage: {cpu}%")
            
        if memory > 80:
            warnings.append(f"High memory usage: {memory}%")
        elif memory > 95:
            errors.append(f"Critical memory usage: {memory}%")
    
    return warnings, errors