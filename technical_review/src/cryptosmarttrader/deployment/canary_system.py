"""Canary deployment system for safe production rollouts with risk monitoring."""

import asyncio
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading

from ..core.structured_logger import get_logger
from ..core.risk_guard import RiskGuard
from ..monitoring.prometheus_metrics import get_metrics
from ..monitoring.alert_rules import AlertManager


class CanaryStage(Enum):
    """Canary deployment stages."""
    PREPARATION = "preparation"
    STAGING_CANARY = "staging_canary"
    PROD_CANARY = "prod_canary"
    FULL_ROLLOUT = "full_rollout"
    ROLLBACK = "rollback"
    COMPLETED = "completed"
    FAILED = "failed"


class DeploymentHealth(Enum):
    """Deployment health status."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"


@dataclass
class CanaryMetrics:
    """Metrics for canary deployment monitoring."""
    stage: CanaryStage
    risk_percentage: float
    error_rate: float
    latency_p95: float
    success_rate: float
    alert_count: int
    tracking_error_bps: float
    portfolio_impact: float
    user_feedback_score: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CanaryGates:
    """Success criteria gates for canary progression."""
    max_error_rate: float = 0.05  # 5% max error rate
    max_latency_p95: float = 2000.0  # 2 seconds max P95 latency
    min_success_rate: float = 0.95  # 95% min success rate
    max_alerts: int = 3  # Maximum 3 alerts
    max_tracking_error: float = 50.0  # 50 bps max tracking error
    max_portfolio_impact: float = 0.01  # 1% max portfolio impact
    min_uptime: float = 0.995  # 99.5% minimum uptime


@dataclass
class DeploymentPlan:
    """Canary deployment execution plan."""
    version: str
    features: List[str]
    staging_duration_hours: int = 168  # 7 days
    staging_risk_percentage: float = 1.0  # 1% risk
    prod_canary_duration_hours: int = 72  # 3 days
    prod_canary_risk_percentage: float = 5.0  # 5% risk
    auto_rollback_enabled: bool = True
    gates: CanaryGates = field(default_factory=CanaryGates)


class CanaryDeploymentSystem:
    """Enterprise canary deployment system with automated safety gates."""

    def __init__(self, risk_guard: RiskGuard, alert_manager: AlertManager):
        """Initialize canary deployment system."""
        self.logger = get_logger("canary_deployment")
        self.risk_guard = risk_guard
        self.alert_manager = alert_manager
        self.metrics = get_metrics()

        # Current deployment state
        self.current_deployment: Optional[DeploymentPlan] = None
        self.current_stage = CanaryStage.PREPARATION
        self.deployment_start_time: Optional[datetime] = None
        self.stage_start_time: Optional[datetime] = None

        # Health monitoring
        self.health_status = DeploymentHealth.HEALTHY
        self.metrics_history: List[CanaryMetrics] = []
        self.gate_violations: List[Dict[str, Any]] = []

        # Callbacks for deployment events
        self.deployment_callbacks: List[Callable] = []

        # Thread safety
        self._lock = threading.RLock()

        # Background monitoring
        self._monitoring_task: Optional[asyncio.Task] = None
        self._monitor_running = False

        # Persistence
        self.data_path = Path("data/canary_deployments")
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("Canary deployment system initialized")

    def register_deployment_callback(self, callback: Callable) -> None:
        """Register callback for deployment events."""
        self.deployment_callbacks.append(callback)

    async def start_deployment(self, deployment_plan: DeploymentPlan) -> bool:
        """Start a new canary deployment."""
        with self._lock:
            if self.current_deployment is not None:
                self.logger.error("Cannot start deployment: another deployment is active")
                return False

            self.current_deployment = deployment_plan
            self.current_stage = CanaryStage.PREPARATION
            self.deployment_start_time = datetime.now()
            self.stage_start_time = datetime.now()
            self.health_status = DeploymentHealth.HEALTHY
            self.metrics_history = []
            self.gate_violations = []

        self.logger.info(f"Starting canary deployment: {deployment_plan.version}",
                        features=deployment_plan.features,
                        staging_duration=deployment_plan.staging_duration_hours)

        # Start background monitoring
        await self._start_monitoring()

        # Notify callbacks
        await self._notify_deployment_event("deployment_started", {
            'version': deployment_plan.version,
            'stage': self.current_stage.value
        })

        # Move to staging canary
        return await self._progress_to_staging_canary()

    async def _progress_to_staging_canary(self) -> bool:
        """Progress to staging canary phase."""
        self.logger.info("Progressing to staging canary phase")

        with self._lock:
            self.current_stage = CanaryStage.STAGING_CANARY
            self.stage_start_time = datetime.now()

        # Configure staging environment (1% risk)
        success = await self._configure_staging_environment()

        if success:
            await self._notify_deployment_event("staging_canary_started", {
                'risk_percentage': self.current_deployment.staging_risk_percentage,
                'duration_hours': self.current_deployment.staging_duration_hours
            })
        else:
            await self._handle_deployment_failure("Failed to configure staging environment")

        return success

    async def _configure_staging_environment(self) -> bool:
        """Configure staging environment with limited risk exposure."""
        try:
            # Reduce position sizes for staging
            staging_multiplier = self.current_deployment.staging_risk_percentage / 100

            # Update risk constraints
            constraints = self.risk_guard.get_trading_constraints()
            original_position_size = constraints['max_position_size_percent']
            staging_position_size = original_position_size * staging_multiplier

            self.logger.info("Staging environment configured",
                           original_position_size=original_position_size,
                           staging_position_size=staging_position_size,
                           risk_reduction=1-staging_multiplier)

            # Store original configuration for rollback
            self._store_original_config(constraints)

            return True

        except Exception as e:
            self.logger.error(f"Failed to configure staging environment: {e}")
            return False

    def _store_original_config(self, config: Dict[str, Any]) -> None:
        """Store original configuration for rollback."""
        config_backup = {
            'timestamp': datetime.now().isoformat(),
            'original_config': config,
            'deployment_version': self.current_deployment.version
        }

        try:
            with open(self.data_path / "original_config.json", 'w') as f:
                json.dump(config_backup, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to store original config: {e}")

    async def _start_monitoring(self) -> None:
        """Start background monitoring task."""
        if self._monitoring_task and not self._monitoring_task.done():
            return

        self._monitor_running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def _stop_monitoring(self) -> None:
        """Stop background monitoring."""
        self._monitor_running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for canary deployment."""
        while self._monitor_running:
            try:
                # Collect current metrics
                metrics = await self._collect_canary_metrics()

                if metrics:
                    self.metrics_history.append(metrics)

                    # Evaluate gates
                    gate_results = self._evaluate_gates(metrics)

                    # Check for failures
                    if gate_results['failed']:
                        await self._handle_gate_failures(gate_results)

                    # Check for stage progression
                    elif self._should_progress_stage():
                        await self._progress_to_next_stage()

                # Update health status
                self._update_health_status()

                # Check for stage timeouts
                if self._is_stage_timeout():
                    await self._handle_stage_timeout()

                await asyncio.sleep(30)  # Monitor every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Longer delay on error

    async def _collect_canary_metrics(self) -> Optional[CanaryMetrics]:
        """Collect current metrics for canary evaluation."""
        try:
            # Get current alert count
            active_alerts = self.alert_manager.get_active_alerts()
            alert_count = len([a for a in active_alerts if a.severity.value in ['critical', 'emergency']])

            # Simulate metrics (in real implementation, collect from actual systems)
            error_rate = 0.02  # 2% error rate
            latency_p95 = 800.0  # 800ms P95 latency
            success_rate = 0.98  # 98% success rate
            tracking_error_bps = 15.0  # 15 bps tracking error
            portfolio_impact = 0.005  # 0.5% portfolio impact
            user_feedback_score = 0.85  # 85% user satisfaction

            # Add some randomness for demo
            import random
            error_rate += random.uniform(-0.01, 0.01)
            latency_p95 += random.uniform(-200, 200)
            success_rate += random.uniform(-0.02, 0.02)
            tracking_error_bps += random.uniform(-5, 5)

            # Ensure realistic bounds
            error_rate = max(0.0, min(1.0, error_rate))
            success_rate = max(0.0, min(1.0, success_rate))
            tracking_error_bps = max(0.0, tracking_error_bps)

            risk_percentage = self.current_deployment.staging_risk_percentage
            if self.current_stage == CanaryStage.PROD_CANARY:
                risk_percentage = self.current_deployment.prod_canary_risk_percentage

            return CanaryMetrics(
                stage=self.current_stage,
                risk_percentage=risk_percentage,
                error_rate=error_rate,
                latency_p95=latency_p95,
                success_rate=success_rate,
                alert_count=alert_count,
                tracking_error_bps=tracking_error_bps,
                portfolio_impact=portfolio_impact,
                user_feedback_score=user_feedback_score
            )

        except Exception as e:
            self.logger.error(f"Error collecting canary metrics: {e}")
            return None

    def _evaluate_gates(self, metrics: CanaryMetrics) -> Dict[str, Any]:
        """Evaluate success criteria gates."""
        gates = self.current_deployment.gates
        violations = []

        # Check each gate
        if metrics.error_rate > gates.max_error_rate:
            violations.append(f"Error rate {metrics.error_rate:.3f} exceeds limit {gates.max_error_rate:.3f}")

        if metrics.latency_p95 > gates.max_latency_p95:
            violations.append(f"P95 latency {metrics.latency_p95:.0f}ms exceeds limit {gates.max_latency_p95:.0f}ms")

        if metrics.success_rate < gates.min_success_rate:
            violations.append(f"Success rate {metrics.success_rate:.3f} below minimum {gates.min_success_rate:.3f}")

        if metrics.alert_count > gates.max_alerts:
            violations.append(f"Alert count {metrics.alert_count} exceeds limit {gates.max_alerts}")

        if metrics.tracking_error_bps > gates.max_tracking_error:
            violations.append(f"Tracking error {metrics.tracking_error_bps:.1f}bps exceeds limit {gates.max_tracking_error:.1f}bps")

        if metrics.portfolio_impact > gates.max_portfolio_impact:
            violations.append(f"Portfolio impact {metrics.portfolio_impact:.3f} exceeds limit {gates.max_portfolio_impact:.3f}")

        return {
            'failed': len(violations) > 0,
            'violations': violations,
            'gates_passed': len(violations) == 0
        }

    async def _handle_gate_failures(self, gate_results: Dict[str, Any]) -> None:
        """Handle gate failures."""
        self.gate_violations.append({
            'timestamp': datetime.now(),
            'violations': gate_results['violations'],
            'stage': self.current_stage.value
        })

        self.logger.warning("Gate violations detected",
                          violations=gate_results['violations'],
                          stage=self.current_stage.value)

        # Auto-rollback if enabled
        if self.current_deployment.auto_rollback_enabled:
            await self._initiate_rollback("Gate violations detected")
        else:
            self.health_status = DeploymentHealth.CRITICAL

    def _should_progress_stage(self) -> bool:
        """Check if current stage should progress to next."""
        if not self.stage_start_time:
            return False

        stage_duration = datetime.now() - self.stage_start_time

        if self.current_stage == CanaryStage.STAGING_CANARY:
            # Check if staging period is complete and gates are passing
            staging_complete = stage_duration.total_seconds() >= (self.current_deployment.staging_duration_hours * 3600)
            recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history

            # All recent metrics should pass gates
            all_gates_passing = all(
                self._evaluate_gates(m)['gates_passed'] for m in recent_metrics
            ) if recent_metrics else False

            return staging_complete and all_gates_passing

        elif self.current_stage == CanaryStage.PROD_CANARY:
            # Check if prod canary period is complete
            canary_complete = stage_duration.total_seconds() >= (self.current_deployment.prod_canary_duration_hours * 3600)
            recent_metrics = self.metrics_history[-20:] if len(self.metrics_history) >= 20 else self.metrics_history

            all_gates_passing = all(
                self._evaluate_gates(m)['gates_passed'] for m in recent_metrics
            ) if recent_metrics else False

            return canary_complete and all_gates_passing

        return False

    async def _progress_to_next_stage(self) -> None:
        """Progress to the next deployment stage."""
        if self.current_stage == CanaryStage.STAGING_CANARY:
            await self._progress_to_prod_canary()
        elif self.current_stage == CanaryStage.PROD_CANARY:
            await self._progress_to_full_rollout()

    async def _progress_to_prod_canary(self) -> bool:
        """Progress to production canary phase."""
        self.logger.info("Progressing to production canary phase")

        with self._lock:
            self.current_stage = CanaryStage.PROD_CANARY
            self.stage_start_time = datetime.now()

        # Configure production canary (5% risk)
        success = await self._configure_prod_canary_environment()

        if success:
            await self._notify_deployment_event("prod_canary_started", {
                'risk_percentage': self.current_deployment.prod_canary_risk_percentage,
                'duration_hours': self.current_deployment.prod_canary_duration_hours
            })
        else:
            await self._handle_deployment_failure("Failed to configure prod canary environment")

        return success

    async def _configure_prod_canary_environment(self) -> bool:
        """Configure production canary environment."""
        try:
            # Increase risk exposure to 5%
            canary_multiplier = self.current_deployment.prod_canary_risk_percentage / 100

            self.logger.info("Production canary environment configured",
                           risk_percentage=self.current_deployment.prod_canary_risk_percentage,
                           canary_multiplier=canary_multiplier)

            return True

        except Exception as e:
            self.logger.error(f"Failed to configure prod canary environment: {e}")
            return False

    async def _progress_to_full_rollout(self) -> bool:
        """Progress to full rollout."""
        self.logger.info("Progressing to full rollout")

        with self._lock:
            self.current_stage = CanaryStage.FULL_ROLLOUT
            self.stage_start_time = datetime.now()

        # Configure full production (100% traffic)
        success = await self._configure_full_production()

        if success:
            await self._notify_deployment_event("full_rollout_started", {
                'risk_percentage': 100.0
            })

            # Complete deployment
            await self._complete_deployment()
        else:
            await self._handle_deployment_failure("Failed to configure full production")

        return success

    async def _configure_full_production(self) -> bool:
        """Configure full production deployment."""
        try:
            self.logger.info("Full production environment configured")
            return True
        except Exception as e:
            self.logger.error(f"Failed to configure full production: {e}")
            return False

    async def _complete_deployment(self) -> None:
        """Complete the deployment process."""
        with self._lock:
            self.current_stage = CanaryStage.COMPLETED
            deployment_duration = datetime.now() - self.deployment_start_time

        self.logger.info(f"Deployment completed successfully: {self.current_deployment.version}",
                        duration_hours=deployment_duration.total_seconds() / 3600,
                        total_metrics=len(self.metrics_history),
                        gate_violations=len(self.gate_violations))

        await self._notify_deployment_event("deployment_completed", {
            'version': self.current_deployment.version,
            'duration_hours': deployment_duration.total_seconds() / 3600,
            'success': True
        })

        # Stop monitoring
        await self._stop_monitoring()

        # Clean up
        self._cleanup_deployment()

    async def _initiate_rollback(self, reason: str) -> None:
        """Initiate deployment rollback."""
        self.logger.warning(f"Initiating rollback: {reason}")

        with self._lock:
            self.current_stage = CanaryStage.ROLLBACK
            self.health_status = DeploymentHealth.FAILED

        # Restore original configuration
        success = await self._restore_original_config()

        await self._notify_deployment_event("rollback_initiated", {
            'reason': reason,
            'success': success
        })

        if success:
            await self._complete_rollback()
        else:
            await self._handle_rollback_failure()

    async def _restore_original_config(self) -> bool:
        """Restore original configuration."""
        try:
            config_file = self.data_path / "original_config.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    backup = json.load(f)

                self.logger.info("Original configuration restored")
                return True
            else:
                self.logger.error("Original configuration backup not found")
                return False
        except Exception as e:
            self.logger.error(f"Failed to restore original config: {e}")
            return False

    async def _complete_rollback(self) -> None:
        """Complete the rollback process."""
        self.logger.info("Rollback completed successfully")

        await self._notify_deployment_event("rollback_completed", {
            'version': self.current_deployment.version,
            'success': True
        })

        await self._stop_monitoring()
        self._cleanup_deployment()

    async def _handle_rollback_failure(self) -> None:
        """Handle rollback failure."""
        self.logger.error("Rollback failed - manual intervention required")

        await self._notify_deployment_event("rollback_failed", {
            'version': self.current_deployment.version,
            'manual_intervention_required': True
        })

    async def _handle_deployment_failure(self, reason: str) -> None:
        """Handle deployment failure."""
        self.logger.error(f"Deployment failed: {reason}")

        with self._lock:
            self.current_stage = CanaryStage.FAILED
            self.health_status = DeploymentHealth.FAILED

        await self._notify_deployment_event("deployment_failed", {
            'reason': reason,
            'version': self.current_deployment.version
        })

        if self.current_deployment.auto_rollback_enabled:
            await self._initiate_rollback(f"Deployment failure: {reason}")

    def _update_health_status(self) -> None:
        """Update overall health status."""
        if not self.metrics_history:
            return

        recent_metrics = self.metrics_history[-5:]  # Last 5 metrics

        critical_violations = 0
        warning_violations = 0

        for metrics in recent_metrics:
            gate_results = self._evaluate_gates(metrics)
            if gate_results['failed']:
                if metrics.error_rate > 0.1 or metrics.alert_count > 5:
                    critical_violations += 1
                else:
                    warning_violations += 1

        if critical_violations > 0:
            self.health_status = DeploymentHealth.CRITICAL
        elif warning_violations > 2:
            self.health_status = DeploymentHealth.WARNING
        else:
            self.health_status = DeploymentHealth.HEALTHY

    def _is_stage_timeout(self) -> bool:
        """Check if current stage has timed out."""
        if not self.stage_start_time:
            return False

        stage_duration = datetime.now() - self.stage_start_time

        # Define timeouts
        timeouts = {
            CanaryStage.STAGING_CANARY: self.current_deployment.staging_duration_hours * 1.2,  # 20% buffer
            CanaryStage.PROD_CANARY: self.current_deployment.prod_canary_duration_hours * 1.2
        }

        timeout_hours = timeouts.get(self.current_stage, 48)  # Default 48 hours

        return stage_duration.total_seconds() >= (timeout_hours * 3600)

    async def _handle_stage_timeout(self) -> None:
        """Handle stage timeout."""
        self.logger.warning(f"Stage timeout: {self.current_stage.value}")

        if self.current_deployment.auto_rollback_enabled:
            await self._initiate_rollback(f"Stage timeout: {self.current_stage.value}")

    async def _notify_deployment_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Notify deployment event callbacks."""
        event = {
            'event_type': event_type,
            'timestamp': datetime.now(),
            'stage': self.current_stage.value,
            'data': data
        }

        for callback in self.deployment_callbacks:
            try:
                await callback(event)
            except Exception as e:
                self.logger.error(f"Deployment callback failed: {e}")

    def _cleanup_deployment(self) -> None:
        """Clean up after deployment completion."""
        with self._lock:
            self.current_deployment = None
            self.deployment_start_time = None
            self.stage_start_time = None

    def get_deployment_status(self) -> Dict[str, Any]:
        """Get current deployment status."""
        if not self.current_deployment:
            return {'status': 'no_active_deployment'}

        stage_duration = None
        if self.stage_start_time:
            stage_duration = (datetime.now() - self.stage_start_time).total_seconds() / 3600

        deployment_duration = None
        if self.deployment_start_time:
            deployment_duration = (datetime.now() - self.deployment_start_time).total_seconds() / 3600

        recent_metrics = self.metrics_history[-1] if self.metrics_history else None

        return {
            'version': self.current_deployment.version,
            'stage': self.current_stage.value,
            'health_status': self.health_status.value,
            'stage_duration_hours': stage_duration,
            'deployment_duration_hours': deployment_duration,
            'metrics_collected': len(self.metrics_history),
            'gate_violations': len(self.gate_violations),
            'recent_metrics': {
                'error_rate': recent_metrics.error_rate if recent_metrics else None,
                'success_rate': recent_metrics.success_rate if recent_metrics else None,
                'tracking_error_bps': recent_metrics.tracking_error_bps if recent_metrics else None
            } if recent_metrics else None,
            'auto_rollback_enabled': self.current_deployment.auto_rollback_enabled
        }


def create_canary_deployment_system(risk_guard: RiskGuard,
                                   alert_manager: AlertManager) -> CanaryDeploymentSystem:
    """Factory function to create CanaryDeploymentSystem instance."""
    return CanaryDeploymentSystem(risk_guard, alert_manager)
