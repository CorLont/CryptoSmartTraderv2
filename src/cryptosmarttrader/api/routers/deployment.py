"""
Deployment API router for go-live management.
Enterprise deployment endpoints with staging, canary, and production phases.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import asyncio

from cryptosmarttrader.deployment.go_live_system import (
    GoLiveManager,
    SLOMonitor,
    ChaosTestRunner,
    DeploymentStage,
    HealthStatus,
    DeploymentMetrics,
)

router = APIRouter()
logger = logging.getLogger(__name__)

# Global instances
go_live_manager = GoLiveManager()
slo_monitor = SLOMonitor()
chaos_tester = ChaosTestRunner()


# Pydantic models
class DeploymentRequest(BaseModel):
    """Deployment request model."""

    stage: str = Field(..., description="Deployment stage (staging/canary/production)")
    risk_budget_percentage: Optional[float] = Field(
        1.0, description="Risk budget for canary deployment"
    )
    duration_hours: Optional[int] = Field(72, description="Canary duration in hours")
    auto_rollback: bool = Field(True, description="Enable automatic rollback on failure")


class SLOCheckRequest(BaseModel):
    """SLO check request model."""

    environment: str = Field("production", description="Environment to check (staging/production)")
    slo_names: Optional[List[str]] = Field(None, description="Specific SLOs to check")


class ChaosTestRequest(BaseModel):
    """Chaos test request model."""

    test_types: Optional[List[str]] = Field(None, description="Specific chaos tests to run")
    severity: str = Field("moderate", description="Test severity (light/moderate/aggressive)")


class DeploymentStatusResponse(BaseModel):
    """Deployment status response model."""

    current_stage: str
    deployment_active: bool
    success_rate: float
    start_time: Optional[datetime]
    estimated_completion: Optional[datetime]
    issues: List[str]


class SLOComplianceResponse(BaseModel):
    """SLO compliance response model."""

    overall_compliance: bool
    slo_results: Dict[str, Any]
    violations: List[str]
    timestamp: datetime


class ChaosTestResponse(BaseModel):
    """Chaos test response model."""

    overall_success_rate: float
    passed_tests: int
    total_tests: int
    test_results: Dict[str, Any]
    recommendation: str


@router.get("/status", response_model=DeploymentStatusResponse, summary="Get deployment status")
async def get_deployment_status():
    """Get current deployment status and metrics."""
    try:
        status = go_live_manager.get_deployment_status()

        current_deployment = status.get("current_deployment")

        return DeploymentStatusResponse(
            current_stage=status["current_stage"],
            deployment_active=current_deployment is not None,
            success_rate=current_deployment.get("success_rate", 0.0) if current_deployment else 0.0,
            start_time=current_deployment.get("start_time") if current_deployment else None,
            estimated_completion=None,  # Calculate based on stage and duration
            issues=current_deployment.get("slo_violations", []) if current_deployment else [],
        )

    except Exception as e:
        logger.error(f"Failed to get deployment status: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {e}")


@router.post("/slo/check", response_model=SLOComplianceResponse, summary="Check SLO compliance")
async def check_slo_compliance(request: SLOCheckRequest):
    """Check Service Level Objective compliance for specified environment."""
    try:
        logger.info(f"Checking SLO compliance for {request.environment}")

        results = await slo_monitor.check_slo_compliance(request.environment)

        # Extract violations
        violations = []
        for slo_name, slo_result in results["slo_results"].items():
            if slo_result.get("status") == "violation":
                violations.append(f"{slo_name}: {slo_result.get('measurement', 'N/A')}")

        return SLOComplianceResponse(
            overall_compliance=results["overall_compliance"],
            slo_results=results["slo_results"],
            violations=violations,
            timestamp=datetime.fromisoformat(results["timestamp"]),
        )

    except Exception as e:
        logger.error(f"SLO compliance check failed: {e}")
        raise HTTPException(status_code=500, detail=f"SLO check failed: {e}")


@router.post("/chaos/test", response_model=ChaosTestResponse, summary="Run chaos engineering tests")
async def run_chaos_tests(request: ChaosTestRequest):
    """Execute chaos engineering tests to validate system resilience."""
    try:
        logger.info(f"Running chaos tests with severity: {request.severity}")

        results = await chaos_tester.run_chaos_tests()

        return ChaosTestResponse(
            overall_success_rate=results["overall_success_rate"],
            passed_tests=results["passed_tests"],
            total_tests=results["total_tests"],
            test_results=results["test_results"],
            recommendation=results["recommendation"],
        )

    except Exception as e:
        logger.error(f"Chaos tests failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chaos tests failed: {e}")


@router.post("/staging/validate", summary="Validate staging environment")
async def validate_staging(background_tasks: BackgroundTasks):
    """
    Validate staging environment for 7 days with alerts enabled.
    This endpoint starts the staging validation process.
    """
    try:
        logger.info("Starting staging validation")

        # Check if deployment is already running
        status = go_live_manager.get_deployment_status()
        if status["current_deployment"] is not None:
            raise HTTPException(status_code=409, detail="Deployment already in progress")

        # Start staging validation in background
        background_tasks.add_task(_run_staging_validation)

        return {
            "message": "Staging validation started",
            "duration_days": 7,
            "alerts_enabled": True,
            "monitoring_interval_minutes": 5,
            "status_endpoint": "/api/v1/deployment/status",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Staging validation failed to start: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start staging validation: {e}")


@router.post("/canary/deploy", summary="Deploy canary release")
async def deploy_canary(request: DeploymentRequest, background_tasks: BackgroundTasks):
    """
    Deploy canary release with specified risk budget and duration.
    Monitors performance and automatically rolls back if thresholds are exceeded.
    """
    try:
        logger.info(
            f"Starting canary deployment with {request.risk_budget_percentage}% risk budget"
        )

        # Validate request parameters
        if request.risk_budget_percentage > 5.0:
            raise HTTPException(
                status_code=400, detail="Risk budget cannot exceed 5% for canary deployment"
            )

        if request.duration_hours < 24 or request.duration_hours > 168:  # 1-7 days
            raise HTTPException(
                status_code=400, detail="Canary duration must be between 24 and 168 hours"
            )

        # Check if staging validation completed
        status = go_live_manager.get_deployment_status()
        if status["current_stage"] not in ["staging", "canary"]:
            raise HTTPException(
                status_code=409,
                detail="Staging validation must be completed before canary deployment",
            )

        # Update canary configuration
        go_live_manager.canary_config.risk_budget_percentage = request.risk_budget_percentage
        go_live_manager.canary_config.duration_hours = request.duration_hours

        # Start canary deployment in background
        background_tasks.add_task(_run_canary_deployment)

        return {
            "message": "Canary deployment started",
            "risk_budget_percentage": request.risk_budget_percentage,
            "duration_hours": request.duration_hours,
            "auto_rollback": request.auto_rollback,
            "monitoring": {
                "traffic_split": "5% initial",
                "success_threshold": "95%",
                "error_threshold": "2%",
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Canary deployment failed to start: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start canary deployment: {e}")


@router.post("/production/deploy", summary="Deploy to production")
async def deploy_production(background_tasks: BackgroundTasks):
    """
    Deploy to full production after successful canary validation.
    Includes comprehensive health checks and monitoring.
    """
    try:
        logger.info("Starting production deployment")

        # Check if canary completed successfully
        status = go_live_manager.get_deployment_status()
        last_deployment = status.get("current_deployment")

        if not last_deployment or last_deployment.get("stage") != "canary":
            raise HTTPException(
                status_code=409,
                detail="Successful canary deployment required before production deployment",
            )

        if last_deployment.get("success_rate", 0) < 0.95:
            raise HTTPException(
                status_code=409, detail="Canary success rate too low for production deployment"
            )

        # Start production deployment in background
        background_tasks.add_task(_run_production_deployment)

        return {
            "message": "Production deployment started",
            "canary_success_rate": last_deployment.get("success_rate", 0),
            "validation_steps": [
                "Pre-deployment health check",
                "Database migration",
                "Application deployment",
                "Configuration update",
                "Post-deployment validation",
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Production deployment failed to start: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start production deployment: {e}")


@router.post("/rollback", summary="Execute emergency rollback")
async def execute_rollback(background_tasks: BackgroundTasks):
    """
    Execute emergency rollback to previous stable version.
    Stops current deployment and reverts to last known good state.
    """
    try:
        logger.warning("Emergency rollback requested")

        # Check if deployment is active
        status = go_live_manager.get_deployment_status()
        if status["current_deployment"] is None:
            raise HTTPException(status_code=409, detail="No active deployment to rollback")

        # Start rollback in background
        background_tasks.add_task(_run_emergency_rollback)

        return {
            "message": "Emergency rollback initiated",
            "current_stage": status["current_stage"],
            "rollback_steps": [
                "Stop traffic to new deployment",
                "Revert to previous version",
                "Validate system health",
                "Restore service availability",
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Rollback failed to start: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start rollback: {e}")


@router.get("/history", summary="Get deployment history")
async def get_deployment_history(limit: int = 10):
    """Get deployment history with specified limit."""
    try:
        import json
        from pathlib import Path

        history_file = Path("deployment_history.json")

        if not history_file.exists():
            return {"deployments": [], "total_count": 0, "message": "No deployment history found"}

        with open(history_file, "r") as f:
            history = json.load(f)

        # Sort by start time (most recent first) and limit
        sorted_history = sorted(history, key=lambda x: x["start_time"], reverse=True)[:limit]

        return {
            "deployments": sorted_history,
            "total_count": len(history),
            "returned_count": len(sorted_history),
        }

    except Exception as e:
        logger.error(f"Failed to get deployment history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get deployment history: {e}")


@router.get("/slo/targets", summary="Get SLO targets")
async def get_slo_targets():
    """Get all configured SLO targets and thresholds."""
    try:
        targets = []

        for slo in slo_monitor.slo_targets:
            targets.append(
                {
                    "name": slo.name,
                    "target_percentage": slo.target_percentage,
                    "measurement_window_hours": slo.measurement_window_hours,
                    "alert_threshold": slo.alert_threshold,
                    "description": slo.description,
                }
            )

        return {"slo_targets": targets, "total_slos": len(targets)}

    except Exception as e:
        logger.error(f"Failed to get SLO targets: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get SLO targets: {e}")


@router.post("/go-live", summary="Execute complete go-live sequence")
async def execute_go_live_sequence(background_tasks: BackgroundTasks):
    """
    Execute complete go-live sequence: staging → canary → production.
    This is the main entry point for automated deployment.
    """
    try:
        logger.info("Starting complete go-live sequence")

        # Check system readiness
        slo_results = await slo_monitor.check_slo_compliance("staging")
        if not slo_results["overall_compliance"]:
            raise HTTPException(
                status_code=409, detail="SLO compliance violations prevent go-live sequence"
            )

        chaos_results = await chaos_tester.run_chaos_tests()
        if chaos_results["overall_success_rate"] < 0.8:
            raise HTTPException(
                status_code=409, detail="Chaos test failures prevent go-live sequence"
            )

        # Start complete sequence in background
        background_tasks.add_task(_run_complete_go_live_sequence)

        return {
            "message": "Complete go-live sequence started",
            "sequence_id": f"golive_{int(datetime.utcnow().timestamp())}",
            "phases": [
                "Staging validation (7 days)",
                "Canary deployment (48-72 hours)",
                "Production deployment",
            ],
            "estimated_duration": "7-10 days",
            "monitoring": {
                "slo_compliance": slo_results["overall_compliance"],
                "chaos_test_success": chaos_results["overall_success_rate"],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Go-live sequence failed to start: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start go-live sequence: {e}")


# Background task functions
async def _run_staging_validation():
    """Background task for staging validation."""
    try:
        result = await go_live_manager._execute_staging_phase()
        logger.info(f"Staging validation completed: {result['success']}")
    except Exception as e:
        logger.error(f"Staging validation background task failed: {e}")


async def _run_canary_deployment():
    """Background task for canary deployment."""
    try:
        result = await go_live_manager._execute_canary_phase()
        logger.info(f"Canary deployment completed: {result['success']}")

        if not result["success"]:
            # Automatic rollback
            await _run_emergency_rollback()
    except Exception as e:
        logger.error(f"Canary deployment background task failed: {e}")


async def _run_production_deployment():
    """Background task for production deployment."""
    try:
        result = await go_live_manager._execute_production_phase()
        logger.info(f"Production deployment completed: {result['success']}")

        if not result["success"]:
            # Emergency rollback
            await _run_emergency_rollback()
    except Exception as e:
        logger.error(f"Production deployment background task failed: {e}")


async def _run_emergency_rollback():
    """Background task for emergency rollback."""
    try:
        result = await go_live_manager._execute_rollback()
        logger.warning(f"Emergency rollback completed: {result['success']}")
    except Exception as e:
        logger.error(f"Emergency rollback background task failed: {e}")


async def _run_complete_go_live_sequence():
    """Background task for complete go-live sequence."""
    try:
        result = await go_live_manager.execute_go_live_sequence()
        logger.info(f"Complete go-live sequence finished: {result['result']}")
    except Exception as e:
        logger.error(f"Complete go-live sequence background task failed: {e}")
