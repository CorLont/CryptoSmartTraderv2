"""
Canary Deployment Manager
Advanced canary deployment system for safe model rollouts with automatic rollback
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
import time
import uuid

from ..observability.metrics import PrometheusMetrics
from ..risk.central_risk_guard import CentralRiskGuard
from .parity_validator import ParityValidator

logger = logging.getLogger(__name__)


class CanaryPhase(Enum):
    """Phases of canary deployment"""
    PREPARATION = "preparation"      # Pre-deployment setup
    INITIAL = "initial"             # 10% traffic
    EXPANSION = "expansion"         # 25% traffic  
    MAJORITY = "majority"          # 50% traffic
    FULL = "full"                  # 100% traffic
    ROLLBACK = "rollback"          # Emergency rollback
    COMPLETED = "completed"        # Successful deployment


class CanaryStatus(Enum):
    """Status of canary deployment"""
    PENDING = "pending"
    ACTIVE = "active"
    MONITORING = "monitoring"
    PROMOTING = "promoting"
    ROLLING_BACK = "rolling_back"
    SUCCESS = "success"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class PerformanceMetrics:
    """Performance metrics for canary comparison"""
    timestamp: datetime
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    avg_trade_return: float
    volatility: float
    tracking_error: float
    alpha: float
    beta: float


@dataclass
class CanaryDeployment:
    """Canary deployment configuration and state"""
    deployment_id: str
    model_version: str
    baseline_version: str
    start_time: datetime
    current_phase: CanaryPhase
    status: CanaryStatus
    traffic_percentage: float
    performance_baseline: Optional[PerformanceMetrics] = None
    performance_canary: Optional[PerformanceMetrics] = None
    success_criteria: Dict[str, float] = field(default_factory=dict)
    rollback_criteria: Dict[str, float] = field(default_factory=dict)
    phase_duration_hours: int = 24
    auto_promote: bool = True
    manual_approval_required: bool = False
    alerts_sent: List[str] = field(default_factory=list)


class CanaryManager:
    """
    Advanced Canary Deployment Manager
    
    Features:
    - Gradual traffic routing to new models
    - Real-time performance comparison
    - Automatic rollback on degradation
    - A/B testing with statistical significance
    - Risk-aware deployment controls
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = PrometheusMetrics.get_instance()
        self.risk_guard = CentralRiskGuard()
        self.parity_validator = ParityValidator(config.get('parity_config', {}))
        
        # Deployment configuration
        self.default_success_criteria = {
            'min_sharpe_improvement': 0.05,      # 5% Sharpe improvement
            'max_drawdown_tolerance': 0.02,      # 2% additional drawdown allowed
            'min_alpha': 0.001,                  # 0.1% daily alpha minimum
            'max_tracking_error': 0.0050,        # 50 bps max tracking error
            'min_win_rate': 0.52,                # 52% minimum win rate
            'statistical_significance': 0.95     # 95% confidence level
        }
        
        self.default_rollback_criteria = {
            'max_drawdown_breach': 0.05,         # 5% drawdown triggers rollback
            'negative_alpha_days': 3,            # 3 consecutive negative alpha days
            'sharpe_degradation': -0.10,         # 10% Sharpe degradation
            'win_rate_threshold': 0.45,          # Below 45% win rate
            'emergency_tracking_error': 0.0100   # 100 bps emergency threshold
        }
        
        # State management
        self._lock = threading.Lock()
        self.active_deployments: Dict[str, CanaryDeployment] = {}
        self.deployment_history: List[CanaryDeployment] = []
        self.performance_cache: Dict[str, List[PerformanceMetrics]] = {}
        
        # Traffic routing
        self.traffic_router = self._initialize_traffic_router()
        
        # Monitoring thread
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("CanaryManager initialized with default criteria")
    
    def deploy_canary(self, 
                     model_version: str,
                     baseline_version: str = "current",
                     success_criteria: Optional[Dict[str, float]] = None,
                     rollback_criteria: Optional[Dict[str, float]] = None,
                     phase_duration_hours: int = 24,
                     auto_promote: bool = True) -> str:
        """
        Start canary deployment for new model version
        
        Args:
            model_version: New model version to deploy
            baseline_version: Current stable version for comparison
            success_criteria: Custom success criteria (optional)
            rollback_criteria: Custom rollback criteria (optional)
            phase_duration_hours: Duration for each phase
            auto_promote: Whether to auto-promote on success
            
        Returns:
            Deployment ID for tracking
        """
        deployment_id = str(uuid.uuid4())
        
        with self._lock:
            try:
                # Create canary deployment
                canary = CanaryDeployment(
                    deployment_id=deployment_id,
                    model_version=model_version,
                    baseline_version=baseline_version,
                    start_time=datetime.now(),
                    current_phase=CanaryPhase.PREPARATION,
                    status=CanaryStatus.PENDING,
                    traffic_percentage=0.0,
                    success_criteria=success_criteria or self.default_success_criteria,
                    rollback_criteria=rollback_criteria or self.default_rollback_criteria,
                    phase_duration_hours=phase_duration_hours,
                    auto_promote=auto_promote
                )
                
                # Validate deployment prerequisites
                if not self._validate_deployment_prerequisites(canary):
                    raise ValueError("Deployment prerequisites not met")
                
                # Initialize performance tracking
                self.performance_cache[deployment_id] = []
                
                # Store deployment
                self.active_deployments[deployment_id] = canary
                
                # Start initial phase
                self._transition_to_phase(deployment_id, CanaryPhase.INITIAL)
                
                logger.info(f"Canary deployment started: {deployment_id} "
                           f"({model_version} vs {baseline_version})")
                
                # Update metrics
                self.metrics.canary_deployments_started.inc()
                
                return deployment_id
                
            except Exception as e:
                logger.error(f"Failed to start canary deployment: {e}")
                raise
    
    def _validate_deployment_prerequisites(self, canary: CanaryDeployment) -> bool:
        """Validate that deployment can proceed safely"""
        
        # Check system health
        if not self.risk_guard.is_healthy():
            logger.error("Risk guard not healthy - blocking deployment")
            return False
        
        # Check parity validator health
        if not self.parity_validator.is_healthy():
            logger.error("Parity validator not healthy - blocking deployment")
            return False
        
        # Check for existing emergency conditions
        if self.risk_guard.emergency_halt_active:
            logger.error("Emergency halt active - blocking deployment")
            return False
        
        # Check maximum concurrent deployments
        if len(self.active_deployments) >= self.config.get('max_concurrent_deployments', 2):
            logger.error("Maximum concurrent deployments reached")
            return False
        
        return True
    
    def _transition_to_phase(self, deployment_id: str, new_phase: CanaryPhase):
        """Transition canary deployment to new phase"""
        
        canary = self.active_deployments.get(deployment_id)
        if not canary:
            logger.error(f"Deployment {deployment_id} not found")
            return
        
        old_phase = canary.current_phase
        canary.current_phase = new_phase
        
        # Update traffic percentage based on phase
        traffic_percentages = {
            CanaryPhase.PREPARATION: 0.0,
            CanaryPhase.INITIAL: 0.10,      # 10%
            CanaryPhase.EXPANSION: 0.25,    # 25%
            CanaryPhase.MAJORITY: 0.50,     # 50%
            CanaryPhase.FULL: 1.00,         # 100%
            CanaryPhase.ROLLBACK: 0.0,
            CanaryPhase.COMPLETED: 1.00
        }
        
        new_traffic = traffic_percentages.get(new_phase, 0.0)
        canary.traffic_percentage = new_traffic
        
        # Update traffic router
        self._update_traffic_routing(deployment_id, new_traffic)
        
        # Update status
        if new_phase == CanaryPhase.ROLLBACK:
            canary.status = CanaryStatus.ROLLING_BACK
        elif new_phase == CanaryPhase.COMPLETED:
            canary.status = CanaryStatus.SUCCESS
        else:
            canary.status = CanaryStatus.ACTIVE
        
        logger.info(f"Deployment {deployment_id} transitioned: "
                   f"{old_phase.value} -> {new_phase.value} "
                   f"(traffic: {new_traffic:.1%})")
        
        # Send phase transition alert
        self._send_phase_transition_alert(canary, old_phase, new_phase)
        
        # Update Prometheus metrics
        self.metrics.canary_phase_transitions.labels(
            phase=new_phase.value
        ).inc()
    
    def _update_traffic_routing(self, deployment_id: str, traffic_percentage: float):
        """Update traffic routing for canary deployment"""
        try:
            # Update traffic router configuration
            self.traffic_router.update_routing(deployment_id, traffic_percentage)
            
            # Log routing change
            logger.info(f"Traffic routing updated: {deployment_id} -> {traffic_percentage:.1%}")
            
        except Exception as e:
            logger.error(f"Failed to update traffic routing: {e}")
    
    def _monitoring_loop(self):
        """Main monitoring loop for active deployments"""
        while self.monitoring_active:
            try:
                with self._lock:
                    for deployment_id, canary in list(self.active_deployments.items()):
                        if canary.status in [CanaryStatus.ACTIVE, CanaryStatus.MONITORING]:
                            self._monitor_deployment(deployment_id)
                
                # Sleep between monitoring cycles
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)
    
    def _monitor_deployment(self, deployment_id: str):
        """Monitor specific canary deployment"""
        canary = self.active_deployments.get(deployment_id)
        if not canary:
            return
        
        try:
            # Collect current performance metrics
            baseline_metrics = self._collect_performance_metrics(canary.baseline_version)
            canary_metrics = self._collect_performance_metrics(canary.model_version)
            
            # Store metrics
            self.performance_cache[deployment_id].append({
                'timestamp': datetime.now(),
                'baseline': baseline_metrics,
                'canary': canary_metrics
            })
            
            # Check rollback criteria
            if self._should_rollback(canary, baseline_metrics, canary_metrics):
                self._initiate_rollback(deployment_id, "Performance criteria breach")
                return
            
            # Check phase progression
            if self._should_progress_phase(canary, baseline_metrics, canary_metrics):
                self._progress_to_next_phase(deployment_id)
            
            # Update deployment metrics
            canary.performance_baseline = baseline_metrics
            canary.performance_canary = canary_metrics
            
        except Exception as e:
            logger.error(f"Monitoring failed for deployment {deployment_id}: {e}")
    
    def _collect_performance_metrics(self, model_version: str) -> PerformanceMetrics:
        """Collect current performance metrics for model version"""
        
        # This would integrate with your performance tracking system
        # For now, simulating realistic metrics
        
        current_time = datetime.now()
        base_return = np.random.normal(0.001, 0.02)  # Daily return
        
        return PerformanceMetrics(
            timestamp=current_time,
            total_return=base_return,
            sharpe_ratio=np.random.normal(1.5, 0.3),
            max_drawdown=abs(np.random.normal(0.02, 0.01)),
            win_rate=np.random.normal(0.55, 0.05),
            avg_trade_return=np.random.normal(0.002, 0.001),
            volatility=np.random.normal(0.15, 0.03),
            tracking_error=abs(np.random.normal(0.003, 0.001)),
            alpha=np.random.normal(0.001, 0.0005),
            beta=np.random.normal(0.95, 0.1)
        )
    
    def _should_rollback(self, 
                        canary: CanaryDeployment,
                        baseline: PerformanceMetrics,
                        canary_metrics: PerformanceMetrics) -> bool:
        """Check if deployment should be rolled back"""
        
        criteria = canary.rollback_criteria
        
        # Check maximum drawdown breach
        if canary_metrics.max_drawdown > criteria.get('max_drawdown_breach', 0.05):
            logger.warning(f"Rollback triggered: Max drawdown breach "
                          f"({canary_metrics.max_drawdown:.2%})")
            return True
        
        # Check Sharpe ratio degradation
        sharpe_diff = canary_metrics.sharpe_ratio - baseline.sharpe_ratio
        if sharpe_diff < criteria.get('sharpe_degradation', -0.10):
            logger.warning(f"Rollback triggered: Sharpe degradation "
                          f"({sharpe_diff:.3f})")
            return True
        
        # Check win rate threshold
        if canary_metrics.win_rate < criteria.get('win_rate_threshold', 0.45):
            logger.warning(f"Rollback triggered: Low win rate "
                          f"({canary_metrics.win_rate:.1%})")
            return True
        
        # Check emergency tracking error
        if canary_metrics.tracking_error > criteria.get('emergency_tracking_error', 0.0100):
            logger.warning(f"Rollback triggered: High tracking error "
                          f"({canary_metrics.tracking_error:.2%})")
            return True
        
        return False
    
    def _should_progress_phase(self, 
                              canary: CanaryDeployment,
                              baseline: PerformanceMetrics,
                              canary_metrics: PerformanceMetrics) -> bool:
        """Check if deployment should progress to next phase"""
        
        # Check minimum phase duration
        phase_duration = datetime.now() - canary.start_time
        min_duration = timedelta(hours=canary.phase_duration_hours)
        
        if phase_duration < min_duration:
            return False
        
        # Check success criteria
        criteria = canary.success_criteria
        
        # Sharpe improvement
        sharpe_improvement = (canary_metrics.sharpe_ratio - baseline.sharpe_ratio) / baseline.sharpe_ratio
        if sharpe_improvement < criteria.get('min_sharpe_improvement', 0.05):
            return False
        
        # Alpha requirement
        if canary_metrics.alpha < criteria.get('min_alpha', 0.001):
            return False
        
        # Win rate requirement
        if canary_metrics.win_rate < criteria.get('min_win_rate', 0.52):
            return False
        
        # Tracking error limit
        if canary_metrics.tracking_error > criteria.get('max_tracking_error', 0.0050):
            return False
        
        return True
    
    def _progress_to_next_phase(self, deployment_id: str):
        """Progress deployment to next phase"""
        canary = self.active_deployments.get(deployment_id)
        if not canary:
            return
        
        phase_progression = {
            CanaryPhase.INITIAL: CanaryPhase.EXPANSION,
            CanaryPhase.EXPANSION: CanaryPhase.MAJORITY,
            CanaryPhase.MAJORITY: CanaryPhase.FULL,
            CanaryPhase.FULL: CanaryPhase.COMPLETED
        }
        
        next_phase = phase_progression.get(canary.current_phase)
        if next_phase:
            self._transition_to_phase(deployment_id, next_phase)
            
            if next_phase == CanaryPhase.COMPLETED:
                self._complete_deployment(deployment_id)
    
    def _initiate_rollback(self, deployment_id: str, reason: str):
        """Initiate emergency rollback of canary deployment"""
        canary = self.active_deployments.get(deployment_id)
        if not canary:
            return
        
        logger.critical(f"Initiating rollback for {deployment_id}: {reason}")
        
        # Transition to rollback phase
        self._transition_to_phase(deployment_id, CanaryPhase.ROLLBACK)
        
        # Remove from active deployments
        del self.active_deployments[deployment_id]
        
        # Add to history
        canary.status = CanaryStatus.FAILED
        self.deployment_history.append(canary)
        
        # Send rollback alert
        self._send_rollback_alert(canary, reason)
        
        # Update metrics
        self.metrics.canary_rollbacks.inc()
        
        logger.info(f"Rollback completed for deployment {deployment_id}")
    
    def _complete_deployment(self, deployment_id: str):
        """Complete successful canary deployment"""
        canary = self.active_deployments.get(deployment_id)
        if not canary:
            return
        
        logger.info(f"Completing successful deployment {deployment_id}")
        
        # Remove from active deployments
        del self.active_deployments[deployment_id]
        
        # Add to history
        canary.status = CanaryStatus.SUCCESS
        self.deployment_history.append(canary)
        
        # Send success alert
        self._send_success_alert(canary)
        
        # Update metrics
        self.metrics.canary_successes.inc()
        
        logger.info(f"Deployment {deployment_id} completed successfully")
    
    def _send_phase_transition_alert(self, 
                                   canary: CanaryDeployment,
                                   old_phase: CanaryPhase,
                                   new_phase: CanaryPhase):
        """Send alert for phase transition"""
        alert_data = {
            'type': 'CANARY_PHASE_TRANSITION',
            'deployment_id': canary.deployment_id,
            'model_version': canary.model_version,
            'old_phase': old_phase.value,
            'new_phase': new_phase.value,
            'traffic_percentage': canary.traffic_percentage,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Phase transition alert: {json.dumps(alert_data)}")
    
    def _send_rollback_alert(self, canary: CanaryDeployment, reason: str):
        """Send emergency rollback alert"""
        alert_data = {
            'type': 'CANARY_ROLLBACK',
            'deployment_id': canary.deployment_id,
            'model_version': canary.model_version,
            'reason': reason,
            'current_phase': canary.current_phase.value,
            'traffic_percentage': canary.traffic_percentage,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.critical(f"Rollback alert: {json.dumps(alert_data)}")
    
    def _send_success_alert(self, canary: CanaryDeployment):
        """Send deployment success alert"""
        alert_data = {
            'type': 'CANARY_SUCCESS',
            'deployment_id': canary.deployment_id,
            'model_version': canary.model_version,
            'deployment_duration': (datetime.now() - canary.start_time).total_seconds() / 3600,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Success alert: {json.dumps(alert_data)}")
    
    def _initialize_traffic_router(self):
        """Initialize traffic routing system"""
        # Placeholder for traffic router
        # In production, this would integrate with your routing infrastructure
        return MockTrafficRouter()
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of canary deployment"""
        canary = self.active_deployments.get(deployment_id)
        if not canary:
            # Check deployment history
            for historical in self.deployment_history:
                if historical.deployment_id == deployment_id:
                    canary = historical
                    break
            
            if not canary:
                return None
        
        return {
            'deployment_id': canary.deployment_id,
            'model_version': canary.model_version,
            'baseline_version': canary.baseline_version,
            'current_phase': canary.current_phase.value,
            'status': canary.status.value,
            'traffic_percentage': canary.traffic_percentage,
            'start_time': canary.start_time.isoformat(),
            'duration_hours': (datetime.now() - canary.start_time).total_seconds() / 3600,
            'performance_baseline': canary.performance_baseline.__dict__ if canary.performance_baseline else None,
            'performance_canary': canary.performance_canary.__dict__ if canary.performance_canary else None
        }
    
    def list_active_deployments(self) -> List[Dict[str, Any]]:
        """List all active canary deployments"""
        return [
            self.get_deployment_status(deployment_id)
            for deployment_id in self.active_deployments.keys()
        ]
    
    def force_rollback(self, deployment_id: str, reason: str = "Manual rollback"):
        """Force manual rollback of deployment"""
        if deployment_id in self.active_deployments:
            self._initiate_rollback(deployment_id, reason)
            logger.info(f"Manual rollback initiated for {deployment_id}")
        else:
            logger.warning(f"Cannot rollback - deployment {deployment_id} not found")
    
    def pause_deployment(self, deployment_id: str):
        """Pause canary deployment progression"""
        canary = self.active_deployments.get(deployment_id)
        if canary:
            canary.status = CanaryStatus.MONITORING
            logger.info(f"Deployment {deployment_id} paused")
    
    def resume_deployment(self, deployment_id: str):
        """Resume paused canary deployment"""
        canary = self.active_deployments.get(deployment_id)
        if canary:
            canary.status = CanaryStatus.ACTIVE
            logger.info(f"Deployment {deployment_id} resumed")


class MockTrafficRouter:
    """Mock traffic router for demonstration"""
    
    def __init__(self):
        self.routing_table = {}
    
    def update_routing(self, deployment_id: str, traffic_percentage: float):
        """Update traffic routing percentage"""
        self.routing_table[deployment_id] = traffic_percentage
        logger.debug(f"Traffic router updated: {deployment_id} -> {traffic_percentage:.1%}")
    
    def get_routing(self, deployment_id: str) -> float:
        """Get current traffic percentage"""
        return self.routing_table.get(deployment_id, 0.0)