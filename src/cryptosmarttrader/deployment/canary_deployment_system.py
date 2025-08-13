"""
Canary Deployment System - Fase D Implementation  
Staging canary 7 dagen (≤1% risk) → prod canary 48-72 uur deployment pipeline.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import time

from ..core.structured_logger import get_logger
from ..observability.metrics_collector import MetricsCollector
from ..risk.risk_guard import RiskGuard, RiskLevel, TradingMode
from .health_checker import HealthChecker
from .environment_manager import EnvironmentManager


class DeploymentStage(Enum):
    """Deployment stage definitions."""
    
    DEVELOPMENT = "development"
    STAGING_CANARY = "staging_canary"
    PRODUCTION_CANARY = "production_canary" 
    PRODUCTION_FULL = "production_full"
    ROLLBACK = "rollback"


class DeploymentStatus(Enum):
    """Deployment status definitions."""
    
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


class HealthStatus(Enum):
    """Health status for SLO monitoring."""
    
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class SLOThresholds:
    """Service Level Objective thresholds."""
    
    # Uptime requirements
    min_uptime_percent: float = 99.5
    max_downtime_minutes_per_day: float = 7.2  # 0.5% of 1440 minutes
    
    # Performance requirements
    max_p95_latency_ms: float = 1000.0  # <1s P95 latency
    max_p99_latency_ms: float = 2000.0  # <2s P99 latency
    
    # Accuracy requirements
    max_tracking_error_bps: float = 20.0  # <20 bps tracking error
    min_hit_rate: float = 0.6  # >60% hit rate
    
    # Alert response requirements  
    max_alert_to_ack_minutes: float = 15.0  # <15min alert response
    max_incident_resolution_hours: float = 4.0  # <4h incident resolution


@dataclass
class CanaryConfig:
    """Configuration for canary deployments."""
    
    # Staging canary (7 days, ≤1% risk budget)
    staging_duration_days: int = 7
    staging_max_risk_percent: float = 1.0
    staging_traffic_percent: float = 10.0  # 10% traffic to canary
    
    # Production canary (48-72 hours, ≤5% risk budget) 
    prod_canary_min_hours: int = 48
    prod_canary_max_hours: int = 72
    prod_canary_max_risk_percent: float = 5.0
    prod_canary_traffic_percent: float = 25.0  # 25% traffic to canary
    
    # Auto-promotion thresholds
    min_success_rate_percent: float = 95.0
    max_error_rate_percent: float = 1.0
    min_trades_for_promotion: int = 100


@dataclass
class DeploymentMetrics:
    """Metrics tracked during canary deployment."""
    
    # Uptime metrics
    uptime_percent: float = 0.0
    downtime_minutes: float = 0.0
    availability_slo_met: bool = False
    
    # Performance metrics
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    avg_response_time_ms: float = 0.0
    performance_slo_met: bool = False
    
    # Trading metrics
    tracking_error_bps: float = 0.0
    hit_rate: float = 0.0
    total_trades: int = 0
    success_rate_percent: float = 0.0
    accuracy_slo_met: bool = False
    
    # Alert metrics
    total_alerts: int = 0
    avg_alert_to_ack_minutes: float = 0.0
    unresolved_incidents: int = 0
    alert_slo_met: bool = False
    
    # Overall SLO compliance
    all_slos_met: bool = False
    slo_compliance_percent: float = 0.0


@dataclass
class CanaryDeployment:
    """Canary deployment tracking."""
    
    deployment_id: str
    stage: DeploymentStage
    status: DeploymentStatus
    version: str
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Configuration
    config: CanaryConfig = None
    slo_thresholds: SLOThresholds = None
    
    # Metrics
    metrics: DeploymentMetrics = None
    
    # Risk tracking
    current_risk_percent: float = 0.0
    max_risk_exceeded: bool = False
    
    # Decision tracking
    promotion_eligible: bool = False
    auto_rollback_triggered: bool = False
    manual_intervention_required: bool = False
    
    # Health checks
    health_status: HealthStatus = HealthStatus.HEALTHY
    failed_health_checks: List[str] = None
    
    metadata: Dict[str, Any] = None


class CanaryDeploymentSystem:
    """
    Canary Deployment System for progressive rollouts.
    
    Pipeline:
    1. Staging Canary: 7 days, ≤1% risk budget, 10% traffic
    2. Production Canary: 48-72 hours, ≤5% risk budget, 25% traffic  
    3. Full Production: 100% traffic after SLO validation
    
    Features:
    - Automated SLO monitoring (uptime, latency, tracking-error, alert response)
    - Risk budget tracking with auto-rollback
    - Progressive traffic shifting
    - Comprehensive health monitoring
    - Manual and automatic promotion gates
    """
    
    def __init__(self):
        self.logger = get_logger("canary_deployment_system")
        
        # Configuration
        self.config = CanaryConfig()
        self.slo_thresholds = SLOThresholds()
        
        # Core components
        self.metrics_collector = MetricsCollector("canary_deployment")
        self.risk_guard = RiskGuard()
        self.health_checker = HealthChecker()
        self.env_manager = EnvironmentManager()
        
        # Deployment tracking
        self.active_deployments: Dict[str, CanaryDeployment] = {}
        self.deployment_history: List[CanaryDeployment] = []
        
        # State persistence
        self.state_file = Path("data/deployment/canary_deployments.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load previous state
        self._load_deployment_state()
        
        self.logger.info("CanaryDeploymentSystem initialized")
    
    async def start_staging_canary(self, version: str, deployment_config: Optional[CanaryConfig] = None) -> str:
        """
        Start staging canary deployment (7 days, ≤1% risk budget).
        
        Returns:
            deployment_id for tracking
        """
        deployment_id = f"staging_canary_{version}_{int(time.time())}"
        config = deployment_config or self.config
        
        self.logger.info(f"Starting staging canary deployment", version=version, deployment_id=deployment_id)
        
        try:
            # Create canary deployment
            canary = CanaryDeployment(
                deployment_id=deployment_id,
                stage=DeploymentStage.STAGING_CANARY,
                status=DeploymentStatus.RUNNING,
                version=version,
                start_time=datetime.now(),
                config=config,
                slo_thresholds=self.slo_thresholds,
                metrics=DeploymentMetrics(),
                failed_health_checks=[],
                metadata={
                    'target_end_time': (datetime.now() + timedelta(days=config.staging_duration_days)).isoformat(),
                    'traffic_percent': config.staging_traffic_percent,
                    'max_risk_percent': config.staging_max_risk_percent
                }
            )
            
            # Deploy to staging environment
            deployment_result = await self.env_manager.deploy_to_staging(version, config.staging_traffic_percent)
            
            if not deployment_result['success']:
                canary.status = DeploymentStatus.FAILED
                canary.end_time = datetime.now()
                self._save_deployment_state()
                raise Exception(f"Staging deployment failed: {deployment_result['error']}")
            
            # Start monitoring
            self.active_deployments[deployment_id] = canary
            asyncio.create_task(self._monitor_canary_deployment(deployment_id))
            
            # Save state
            self._save_deployment_state()
            
            self.logger.info(f"Staging canary deployment started successfully", deployment_id=deployment_id)
            return deployment_id
            
        except Exception as e:
            self.logger.error(f"Failed to start staging canary: {e}")
            raise
    
    async def promote_to_production_canary(self, staging_deployment_id: str) -> str:
        """
        Promote staging canary to production canary (48-72 hours, ≤5% risk budget).
        
        Returns:
            new production canary deployment_id
        """
        staging_canary = self.active_deployments.get(staging_deployment_id)
        
        if not staging_canary or staging_canary.stage != DeploymentStage.STAGING_CANARY:
            raise ValueError(f"Invalid staging canary deployment: {staging_deployment_id}")
        
        if not staging_canary.promotion_eligible:
            raise ValueError(f"Staging canary not eligible for promotion. Check SLO compliance.")
        
        prod_deployment_id = f"prod_canary_{staging_canary.version}_{int(time.time())}"
        
        self.logger.info(f"Promoting to production canary", 
                        staging_id=staging_deployment_id,
                        prod_id=prod_deployment_id,
                        version=staging_canary.version)
        
        try:
            # Create production canary deployment
            prod_canary = CanaryDeployment(
                deployment_id=prod_deployment_id,
                stage=DeploymentStage.PRODUCTION_CANARY,
                status=DeploymentStatus.RUNNING,
                version=staging_canary.version,
                start_time=datetime.now(),
                config=self.config,
                slo_thresholds=self.slo_thresholds,
                metrics=DeploymentMetrics(),
                failed_health_checks=[],
                metadata={
                    'min_end_time': (datetime.now() + timedelta(hours=self.config.prod_canary_min_hours)).isoformat(),
                    'max_end_time': (datetime.now() + timedelta(hours=self.config.prod_canary_max_hours)).isoformat(),
                    'traffic_percent': self.config.prod_canary_traffic_percent,
                    'max_risk_percent': self.config.prod_canary_max_risk_percent,
                    'promoted_from': staging_deployment_id
                }
            )
            
            # Deploy to production environment (canary)
            deployment_result = await self.env_manager.deploy_to_production_canary(
                staging_canary.version, 
                self.config.prod_canary_traffic_percent
            )
            
            if not deployment_result['success']:
                prod_canary.status = DeploymentStatus.FAILED
                prod_canary.end_time = datetime.now()
                self._save_deployment_state()
                raise Exception(f"Production canary deployment failed: {deployment_result['error']}")
            
            # Mark staging as complete
            staging_canary.status = DeploymentStatus.SUCCESS
            staging_canary.end_time = datetime.now()
            
            # Start monitoring production canary
            self.active_deployments[prod_deployment_id] = prod_canary
            asyncio.create_task(self._monitor_canary_deployment(prod_deployment_id))
            
            # Save state
            self._save_deployment_state()
            
            self.logger.info(f"Production canary deployment started", deployment_id=prod_deployment_id)
            return prod_deployment_id
            
        except Exception as e:
            self.logger.error(f"Failed to promote to production canary: {e}")
            raise
    
    async def promote_to_full_production(self, prod_canary_deployment_id: str) -> bool:
        """
        Promote production canary to full production (100% traffic).
        
        Returns:
            True if successful
        """
        prod_canary = self.active_deployments.get(prod_canary_deployment_id)
        
        if not prod_canary or prod_canary.stage != DeploymentStage.PRODUCTION_CANARY:
            raise ValueError(f"Invalid production canary deployment: {prod_canary_deployment_id}")
        
        if not prod_canary.promotion_eligible:
            raise ValueError(f"Production canary not eligible for promotion. Check SLO compliance.")
        
        # Check minimum canary duration
        min_end_time = datetime.fromisoformat(prod_canary.metadata['min_end_time'])
        if datetime.now() < min_end_time:
            raise ValueError(f"Production canary must run for minimum {self.config.prod_canary_min_hours} hours")
        
        self.logger.info(f"Promoting to full production", 
                        canary_id=prod_canary_deployment_id,
                        version=prod_canary.version)
        
        try:
            # Deploy to full production
            deployment_result = await self.env_manager.deploy_to_production_full(prod_canary.version)
            
            if not deployment_result['success']:
                raise Exception(f"Full production deployment failed: {deployment_result['error']}")
            
            # Mark canary as complete
            prod_canary.status = DeploymentStatus.SUCCESS
            prod_canary.end_time = datetime.now()
            prod_canary.stage = DeploymentStage.PRODUCTION_FULL
            
            # Move to history
            self.deployment_history.append(prod_canary)
            del self.active_deployments[prod_canary_deployment_id]
            
            # Save state
            self._save_deployment_state()
            
            self.logger.info(f"Full production deployment completed", version=prod_canary.version)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to promote to full production: {e}")
            raise
    
    async def rollback_deployment(self, deployment_id: str, reason: str) -> bool:
        """
        Rollback active deployment due to issues.
        
        Returns:
            True if rollback successful
        """
        deployment = self.active_deployments.get(deployment_id)
        
        if not deployment:
            raise ValueError(f"No active deployment found: {deployment_id}")
        
        self.logger.warning(f"Rolling back deployment", 
                           deployment_id=deployment_id,
                           version=deployment.version,
                           reason=reason)
        
        try:
            # Perform rollback based on stage
            if deployment.stage == DeploymentStage.STAGING_CANARY:
                rollback_result = await self.env_manager.rollback_staging(deployment.version)
            elif deployment.stage == DeploymentStage.PRODUCTION_CANARY:
                rollback_result = await self.env_manager.rollback_production_canary(deployment.version)
            else:
                raise ValueError(f"Cannot rollback deployment in stage: {deployment.stage}")
            
            if not rollback_result['success']:
                raise Exception(f"Rollback failed: {rollback_result['error']}")
            
            # Update deployment status
            deployment.status = DeploymentStatus.ROLLED_BACK
            deployment.end_time = datetime.now()
            deployment.auto_rollback_triggered = True
            deployment.metadata['rollback_reason'] = reason
            
            # Move to history
            self.deployment_history.append(deployment)
            del self.active_deployments[deployment_id]
            
            # Save state
            self._save_deployment_state()
            
            self.logger.info(f"Deployment rollback completed", deployment_id=deployment_id)
            return True
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            raise
    
    async def _monitor_canary_deployment(self, deployment_id: str) -> None:
        """
        Monitor canary deployment and assess SLO compliance.
        """
        deployment = self.active_deployments.get(deployment_id)
        if not deployment:
            return
        
        self.logger.info(f"Starting monitoring for deployment", deployment_id=deployment_id)
        
        monitoring_interval_seconds = 300  # Check every 5 minutes
        
        while deployment_id in self.active_deployments and deployment.status == DeploymentStatus.RUNNING:
            try:
                # Collect current metrics
                current_metrics = await self._collect_deployment_metrics(deployment)
                deployment.metrics = current_metrics
                
                # Assess SLO compliance
                slo_compliance = self._assess_slo_compliance(deployment)
                deployment.metrics.all_slos_met = slo_compliance['all_met']
                deployment.metrics.slo_compliance_percent = slo_compliance['compliance_percent']
                
                # Check risk budget
                risk_check = await self._check_risk_budget(deployment)
                deployment.current_risk_percent = risk_check['current_risk']
                deployment.max_risk_exceeded = risk_check['exceeded']
                
                # Health checks
                health_result = await self.health_checker.check_deployment_health(deployment.version)
                deployment.health_status = HealthStatus(health_result['status'])
                deployment.failed_health_checks = health_result.get('failed_checks', [])
                
                # Check for auto-rollback conditions
                if await self._should_auto_rollback(deployment):
                    await self.rollback_deployment(deployment_id, "Auto-rollback triggered due to SLO violations")
                    break
                
                # Check for promotion eligibility
                deployment.promotion_eligible = self._is_promotion_eligible(deployment)
                
                # Log status
                self.logger.info(f"Deployment monitoring update",
                               deployment_id=deployment_id,
                               stage=deployment.stage.value,
                               slo_compliance=f"{slo_compliance['compliance_percent']:.1f}%",
                               risk_percent=f"{deployment.current_risk_percent:.2f}%",
                               health=deployment.health_status.value,
                               promotion_eligible=deployment.promotion_eligible)
                
                # Save updated state
                self._save_deployment_state()
                
                await asyncio.sleep(monitoring_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Monitoring error for deployment {deployment_id}: {e}")
                deployment.manual_intervention_required = True
                await asyncio.sleep(monitoring_interval_seconds)
        
        self.logger.info(f"Monitoring stopped for deployment", deployment_id=deployment_id)
    
    async def _collect_deployment_metrics(self, deployment: CanaryDeployment) -> DeploymentMetrics:
        """Collect comprehensive deployment metrics."""
        
        try:
            # Collect metrics from various sources
            uptime_metrics = await self.metrics_collector.get_uptime_metrics()
            performance_metrics = await self.metrics_collector.get_performance_metrics() 
            trading_metrics = await self.metrics_collector.get_trading_metrics()
            alert_metrics = await self.metrics_collector.get_alert_metrics()
            
            # Create comprehensive metrics
            metrics = DeploymentMetrics(
                # Uptime
                uptime_percent=uptime_metrics.get('uptime_percent', 100.0),
                downtime_minutes=uptime_metrics.get('downtime_minutes', 0.0),
                availability_slo_met=uptime_metrics.get('uptime_percent', 100.0) >= self.slo_thresholds.min_uptime_percent,
                
                # Performance  
                p95_latency_ms=performance_metrics.get('p95_latency_ms', 100.0),
                p99_latency_ms=performance_metrics.get('p99_latency_ms', 200.0),
                avg_response_time_ms=performance_metrics.get('avg_response_time_ms', 50.0),
                performance_slo_met=performance_metrics.get('p95_latency_ms', 100.0) <= self.slo_thresholds.max_p95_latency_ms,
                
                # Trading
                tracking_error_bps=trading_metrics.get('tracking_error_bps', 15.0),
                hit_rate=trading_metrics.get('hit_rate', 0.7),
                total_trades=trading_metrics.get('total_trades', 0),
                success_rate_percent=trading_metrics.get('success_rate_percent', 95.0),
                accuracy_slo_met=trading_metrics.get('tracking_error_bps', 15.0) <= self.slo_thresholds.max_tracking_error_bps,
                
                # Alerts
                total_alerts=alert_metrics.get('total_alerts', 0),
                avg_alert_to_ack_minutes=alert_metrics.get('avg_alert_to_ack_minutes', 10.0),
                unresolved_incidents=alert_metrics.get('unresolved_incidents', 0),
                alert_slo_met=alert_metrics.get('avg_alert_to_ack_minutes', 10.0) <= self.slo_thresholds.max_alert_to_ack_minutes
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect deployment metrics: {e}")
            return DeploymentMetrics()  # Return empty metrics
    
    def _assess_slo_compliance(self, deployment: CanaryDeployment) -> Dict[str, Any]:
        """Assess overall SLO compliance."""
        
        metrics = deployment.metrics
        slos_met = [
            metrics.availability_slo_met,
            metrics.performance_slo_met,
            metrics.accuracy_slo_met,
            metrics.alert_slo_met
        ]
        
        compliance_percent = (sum(slos_met) / len(slos_met)) * 100
        all_met = all(slos_met)
        
        return {
            'all_met': all_met,
            'compliance_percent': compliance_percent,
            'slo_details': {
                'availability': metrics.availability_slo_met,
                'performance': metrics.performance_slo_met,
                'accuracy': metrics.accuracy_slo_met,
                'alerts': metrics.alert_slo_met
            }
        }
    
    async def _check_risk_budget(self, deployment: CanaryDeployment) -> Dict[str, Any]:
        """Check risk budget consumption."""
        
        # Calculate current risk based on deployment stage
        if deployment.stage == DeploymentStage.STAGING_CANARY:
            max_risk = deployment.config.staging_max_risk_percent
        elif deployment.stage == DeploymentStage.PRODUCTION_CANARY:
            max_risk = deployment.config.prod_canary_max_risk_percent
        else:
            max_risk = 100.0  # No risk limit for full production
        
        # Simulate risk calculation (in production, this would be real risk metrics)
        current_risk = 0.5  # Example: 0.5% current risk
        
        # Factor in actual performance
        if deployment.metrics:
            # Higher risk if SLOs not met
            if not deployment.metrics.all_slos_met:
                current_risk += (100 - deployment.metrics.slo_compliance_percent) / 100.0
            
            # Higher risk if many failed health checks
            if deployment.failed_health_checks:
                current_risk += len(deployment.failed_health_checks) * 0.1
        
        exceeded = current_risk > max_risk
        
        return {
            'current_risk': current_risk,
            'max_risk': max_risk,
            'exceeded': exceeded,
            'remaining_budget': max(0, max_risk - current_risk)
        }
    
    async def _should_auto_rollback(self, deployment: CanaryDeployment) -> bool:
        """Determine if deployment should be automatically rolled back."""
        
        # Rollback if risk budget exceeded
        if deployment.max_risk_exceeded:
            self.logger.warning(f"Risk budget exceeded for deployment {deployment.deployment_id}")
            return True
        
        # Rollback if health status is critical
        if deployment.health_status == HealthStatus.CRITICAL:
            self.logger.warning(f"Critical health status for deployment {deployment.deployment_id}")
            return True
        
        # Rollback if multiple SLOs failing
        if deployment.metrics and deployment.metrics.slo_compliance_percent < 60.0:
            self.logger.warning(f"SLO compliance too low ({deployment.metrics.slo_compliance_percent:.1f}%) for deployment {deployment.deployment_id}")
            return True
        
        # Rollback if too many unresolved incidents
        if deployment.metrics and deployment.metrics.unresolved_incidents >= 3:
            self.logger.warning(f"Too many unresolved incidents ({deployment.metrics.unresolved_incidents}) for deployment {deployment.deployment_id}")
            return True
        
        return False
    
    def _is_promotion_eligible(self, deployment: CanaryDeployment) -> bool:
        """Check if deployment is eligible for promotion."""
        
        # Must meet all SLOs
        if not deployment.metrics or not deployment.metrics.all_slos_met:
            return False
        
        # Must be healthy
        if deployment.health_status not in [HealthStatus.HEALTHY]:
            return False
        
        # Must be within risk budget
        if deployment.max_risk_exceeded:
            return False
        
        # Must have minimum runtime for production canary
        if deployment.stage == DeploymentStage.PRODUCTION_CANARY:
            min_end_time = datetime.fromisoformat(deployment.metadata['min_end_time'])
            if datetime.now() < min_end_time:
                return False
        
        # Must have sufficient trade data (for production canary)
        if deployment.stage == DeploymentStage.PRODUCTION_CANARY:
            if deployment.metrics.total_trades < self.config.min_trades_for_promotion:
                return False
        
        return True
    
    def _save_deployment_state(self) -> None:
        """Save deployment state to file."""
        try:
            state = {
                'active_deployments': {
                    k: self._serialize_deployment(v) for k, v in self.active_deployments.items()
                },
                'deployment_history': [
                    self._serialize_deployment(d) for d in self.deployment_history[-50:]  # Keep last 50
                ],
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save deployment state: {e}")
    
    def _load_deployment_state(self) -> None:
        """Load deployment state from file."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                # Load active deployments
                for k, v in state.get('active_deployments', {}).items():
                    self.active_deployments[k] = self._deserialize_deployment(v)
                
                # Load history
                for d in state.get('deployment_history', []):
                    self.deployment_history.append(self._deserialize_deployment(d))
                
                self.logger.info("Deployment state loaded successfully")
                
        except Exception as e:
            self.logger.warning(f"Could not load deployment state: {e}")
    
    def _serialize_deployment(self, deployment: CanaryDeployment) -> Dict[str, Any]:
        """Serialize deployment for JSON storage."""
        data = asdict(deployment)
        data['start_time'] = deployment.start_time.isoformat()
        if deployment.end_time:
            data['end_time'] = deployment.end_time.isoformat()
        return data
    
    def _deserialize_deployment(self, data: Dict[str, Any]) -> CanaryDeployment:
        """Deserialize deployment from JSON data."""
        data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data['end_time']:
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        
        # Handle enums
        data['stage'] = DeploymentStage(data['stage'])
        data['status'] = DeploymentStatus(data['status'])
        data['health_status'] = HealthStatus(data['health_status'])
        
        return CanaryDeployment(**data)
    
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get comprehensive status of deployment."""
        deployment = self.active_deployments.get(deployment_id)
        
        if not deployment:
            # Check history
            historical = next((d for d in self.deployment_history if d.deployment_id == deployment_id), None)
            if historical:
                return self._serialize_deployment(historical)
            return {'error': 'Deployment not found'}
        
        return self._serialize_deployment(deployment)
    
    def get_active_deployments(self) -> Dict[str, Any]:
        """Get all active deployments."""
        return {
            k: self._serialize_deployment(v) for k, v in self.active_deployments.items()
        }


# Convenience functions
async def start_staging_canary_deployment(version: str, config: Optional[CanaryConfig] = None) -> str:
    """Start a staging canary deployment."""
    system = CanaryDeploymentSystem()
    return await system.start_staging_canary(version, config)


async def promote_canary_to_production(staging_deployment_id: str) -> str:
    """Promote staging canary to production canary."""
    system = CanaryDeploymentSystem()
    return await system.promote_to_production_canary(staging_deployment_id)


if __name__ == "__main__":
    # Example usage
    async def example_canary_pipeline():
        system = CanaryDeploymentSystem()
        
        # Start staging canary
        staging_id = await system.start_staging_canary("v2.1.0")
        print(f"Started staging canary: {staging_id}")
        
        # Monitor would run automatically...
        # After 7 days and SLO compliance:
        
        # Promote to production canary  
        # prod_id = await system.promote_to_production_canary(staging_id)
        # print(f"Promoted to production canary: {prod_id}")
        
        # After 48-72 hours and SLO compliance:
        # success = await system.promote_to_full_production(prod_id)
        # print(f"Full production deployment: {'success' if success else 'failed'}")
    
    asyncio.run(example_canary_pipeline())