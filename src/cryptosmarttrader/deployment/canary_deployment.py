#!/usr/bin/env python3
"""
FASE F - Canary Deployment System
Staging canary (≤1% risk budget) ≥7 dagen; daarna prod canary 48-72 uur
"""

import logging
import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class CanaryStage(Enum):
    """Canary deployment stages"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION_CANARY = "production_canary"
    PRODUCTION_FULL = "production_full"
    ROLLBACK = "rollback"


class CanaryStatus(Enum):
    """Canary deployment status"""
    PREPARING = "preparing"
    RUNNING = "running"
    MONITORING = "monitoring"
    PROMOTING = "promoting"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class RiskBudget:
    """Risk budget allocation for canary deployment"""
    total_capital_usd: float
    staging_allocation_percent: float = 1.0  # ≤1% for staging
    canary_allocation_percent: float = 5.0   # ≤5% for production canary
    max_daily_loss_percent: float = 0.5      # Max daily loss per canary
    max_drawdown_percent: float = 2.0        # Max drawdown per canary
    
    @property
    def staging_capital_usd(self) -> float:
        return self.total_capital_usd * (self.staging_allocation_percent / 100.0)
    
    @property
    def canary_capital_usd(self) -> float:
        return self.total_capital_usd * (self.canary_allocation_percent / 100.0)


@dataclass
class CanaryMetrics:
    """Canary deployment performance metrics"""
    stage: CanaryStage
    start_time: datetime
    end_time: Optional[datetime]
    duration_hours: float
    
    # Performance metrics
    total_return_percent: float
    sharpe_ratio: float
    max_drawdown_percent: float
    daily_vol_percent: float
    
    # Risk metrics
    risk_budget_used_percent: float
    daily_loss_percent: float
    breach_count: int
    
    # Operational metrics
    total_orders: int
    successful_orders: int
    error_rate_percent: float
    avg_latency_ms: float
    
    # Parity metrics
    tracking_error_bps: float
    parity_score: float


@dataclass
class CanaryConfig:
    """Canary deployment configuration"""
    version: str
    description: str
    staging_duration_days: int = 7
    canary_duration_hours: int = 48  # 48-72 hours
    promotion_criteria: Dict[str, float] = None
    rollback_criteria: Dict[str, float] = None
    
    def __post_init__(self):
        if self.promotion_criteria is None:
            self.promotion_criteria = {
                'min_sharpe_ratio': 1.0,
                'max_drawdown_percent': 2.0,
                'max_error_rate_percent': 1.0,
                'min_parity_score': 85.0,
                'max_tracking_error_bps': 20.0
            }
        
        if self.rollback_criteria is None:
            self.rollback_criteria = {
                'max_daily_loss_percent': 1.0,
                'max_drawdown_percent': 3.0,
                'max_error_rate_percent': 5.0,
                'min_parity_score': 70.0,
                'max_tracking_error_bps': 50.0
            }


@dataclass
class CanaryDeployment:
    """Canary deployment instance"""
    deployment_id: str
    config: CanaryConfig
    risk_budget: RiskBudget
    current_stage: CanaryStage
    status: CanaryStatus
    created_at: datetime
    
    # Stage tracking
    staging_start: Optional[datetime] = None
    staging_end: Optional[datetime] = None
    canary_start: Optional[datetime] = None
    canary_end: Optional[datetime] = None
    
    # Metrics per stage
    staging_metrics: Optional[CanaryMetrics] = None
    canary_metrics: Optional[CanaryMetrics] = None
    
    # Risk monitoring
    current_capital_used: float = 0.0
    current_daily_loss: float = 0.0
    breach_events: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.breach_events is None:
            self.breach_events = []


class CanaryDeploymentManager:
    """
    FASE F Canary Deployment Manager
    Manages staging and production canary deployments with risk budget controls
    """
    
    def __init__(self):
        self.deployments: Dict[str, CanaryDeployment] = {}
        self.active_deployment: Optional[str] = None
        
        # Storage for deployment data
        self.deployments_dir = Path("exports/canary_deployments")
        self.deployments_dir.mkdir(parents=True, exist_ok=True)
        
        # Monitoring thread
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        
        logger.info("Canary deployment manager initialized")
    
    def create_deployment(self, 
                         config: CanaryConfig, 
                         risk_budget: RiskBudget) -> str:
        """Create new canary deployment"""
        deployment_id = f"canary_{config.version}_{int(time.time())}"
        
        deployment = CanaryDeployment(
            deployment_id=deployment_id,
            config=config,
            risk_budget=risk_budget,
            current_stage=CanaryStage.DEVELOPMENT,
            status=CanaryStatus.PREPARING,
            created_at=datetime.now()
        )
        
        self.deployments[deployment_id] = deployment
        self.active_deployment = deployment_id
        
        logger.info(f"Created canary deployment: {deployment_id}")
        return deployment_id
    
    def start_staging(self, deployment_id: str) -> bool:
        """Start staging canary deployment (≤1% risk budget, ≥7 days)"""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            logger.error(f"Deployment not found: {deployment_id}")
            return False
        
        if deployment.current_stage != CanaryStage.DEVELOPMENT:
            logger.error(f"Cannot start staging from stage: {deployment.current_stage}")
            return False
        
        # Update deployment state
        deployment.current_stage = CanaryStage.STAGING
        deployment.status = CanaryStatus.RUNNING
        deployment.staging_start = datetime.now()
        deployment.current_capital_used = deployment.risk_budget.staging_capital_usd
        
        # Start monitoring
        self._start_monitoring(deployment_id)
        
        logger.info(f"Started staging canary: {deployment_id}")
        logger.info(f"Staging capital: ${deployment.risk_budget.staging_capital_usd:,.2f}")
        logger.info(f"Duration: {deployment.config.staging_duration_days} days")
        
        return True
    
    def start_production_canary(self, deployment_id: str) -> bool:
        """Start production canary deployment (≤5% risk budget, 48-72 hours)"""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            logger.error(f"Deployment not found: {deployment_id}")
            return False
        
        if deployment.current_stage != CanaryStage.STAGING:
            logger.error(f"Cannot start production canary from stage: {deployment.current_stage}")
            return False
        
        # Check staging completion requirements
        if not self._validate_staging_completion(deployment):
            logger.error("Staging validation failed - cannot promote to production canary")
            return False
        
        # Update deployment state
        deployment.current_stage = CanaryStage.PRODUCTION_CANARY
        deployment.status = CanaryStatus.RUNNING
        deployment.staging_end = datetime.now()
        deployment.canary_start = datetime.now()
        deployment.current_capital_used = deployment.risk_budget.canary_capital_usd
        
        logger.info(f"Started production canary: {deployment_id}")
        logger.info(f"Canary capital: ${deployment.risk_budget.canary_capital_usd:,.2f}")
        logger.info(f"Duration: {deployment.config.canary_duration_hours} hours")
        
        return True
    
    def promote_to_full_production(self, deployment_id: str) -> bool:
        """Promote canary to full production deployment"""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            logger.error(f"Deployment not found: {deployment_id}")
            return False
        
        if deployment.current_stage != CanaryStage.PRODUCTION_CANARY:
            logger.error(f"Cannot promote from stage: {deployment.current_stage}")
            return False
        
        # Check canary completion requirements
        if not self._validate_canary_completion(deployment):
            logger.error("Canary validation failed - cannot promote to full production")
            return False
        
        # Update deployment state
        deployment.current_stage = CanaryStage.PRODUCTION_FULL
        deployment.status = CanaryStatus.COMPLETED
        deployment.canary_end = datetime.now()
        deployment.current_capital_used = deployment.risk_budget.total_capital_usd
        
        # Stop monitoring
        self._stop_monitoring()
        
        logger.info(f"Promoted to full production: {deployment_id}")
        return True
    
    def rollback_deployment(self, deployment_id: str, reason: str) -> bool:
        """Rollback canary deployment due to risk breach or performance issues"""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            logger.error(f"Deployment not found: {deployment_id}")
            return False
        
        # Record rollback event
        rollback_event = {
            'timestamp': datetime.now().isoformat(),
            'stage': deployment.current_stage.value,
            'reason': reason,
            'capital_at_rollback': deployment.current_capital_used,
            'breach_count': len(deployment.breach_events)
        }
        
        deployment.breach_events.append(rollback_event)
        
        # Update deployment state
        deployment.current_stage = CanaryStage.ROLLBACK
        deployment.status = CanaryStatus.ROLLED_BACK
        
        if deployment.current_stage == CanaryStage.STAGING:
            deployment.staging_end = datetime.now()
        elif deployment.current_stage == CanaryStage.PRODUCTION_CANARY:
            deployment.canary_end = datetime.now()
        
        # Stop monitoring
        self._stop_monitoring()
        
        logger.critical(f"Rolled back deployment {deployment_id}: {reason}")
        return True
    
    def record_metrics(self, deployment_id: str, metrics: Dict[str, Any]) -> None:
        """Record performance metrics for active deployment"""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return
        
        current_time = datetime.now()
        
        # Create metrics object
        stage_metrics = CanaryMetrics(
            stage=deployment.current_stage,
            start_time=deployment.staging_start if deployment.current_stage == CanaryStage.STAGING else deployment.canary_start,
            end_time=None,
            duration_hours=self._calculate_duration_hours(deployment),
            total_return_percent=metrics.get('total_return_percent', 0.0),
            sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
            max_drawdown_percent=metrics.get('max_drawdown_percent', 0.0),
            daily_vol_percent=metrics.get('daily_vol_percent', 0.0),
            risk_budget_used_percent=metrics.get('risk_budget_used_percent', 0.0),
            daily_loss_percent=metrics.get('daily_loss_percent', 0.0),
            breach_count=len(deployment.breach_events),
            total_orders=metrics.get('total_orders', 0),
            successful_orders=metrics.get('successful_orders', 0),
            error_rate_percent=metrics.get('error_rate_percent', 0.0),
            avg_latency_ms=metrics.get('avg_latency_ms', 0.0),
            tracking_error_bps=metrics.get('tracking_error_bps', 0.0),
            parity_score=metrics.get('parity_score', 0.0)
        )
        
        # Store metrics
        if deployment.current_stage == CanaryStage.STAGING:
            deployment.staging_metrics = stage_metrics
        elif deployment.current_stage == CanaryStage.PRODUCTION_CANARY:
            deployment.canary_metrics = stage_metrics
        
        # Update current tracking
        deployment.current_daily_loss = metrics.get('daily_loss_percent', 0.0)
    
    def check_risk_breach(self, deployment_id: str) -> Optional[str]:
        """Check for risk budget breaches and return breach reason if any"""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return None
        
        config = deployment.config
        risk_budget = deployment.risk_budget
        
        # Get current metrics
        current_metrics = None
        if deployment.current_stage == CanaryStage.STAGING and deployment.staging_metrics:
            current_metrics = deployment.staging_metrics
        elif deployment.current_stage == CanaryStage.PRODUCTION_CANARY and deployment.canary_metrics:
            current_metrics = deployment.canary_metrics
        
        if not current_metrics:
            return None
        
        # Check rollback criteria
        rollback_criteria = config.rollback_criteria
        
        if current_metrics.daily_loss_percent > rollback_criteria['max_daily_loss_percent']:
            return f"Daily loss breach: {current_metrics.daily_loss_percent:.2f}% > {rollback_criteria['max_daily_loss_percent']:.2f}%"
        
        if current_metrics.max_drawdown_percent > rollback_criteria['max_drawdown_percent']:
            return f"Drawdown breach: {current_metrics.max_drawdown_percent:.2f}% > {rollback_criteria['max_drawdown_percent']:.2f}%"
        
        if current_metrics.error_rate_percent > rollback_criteria['max_error_rate_percent']:
            return f"Error rate breach: {current_metrics.error_rate_percent:.2f}% > {rollback_criteria['max_error_rate_percent']:.2f}%"
        
        if current_metrics.parity_score < rollback_criteria['min_parity_score']:
            return f"Parity score breach: {current_metrics.parity_score:.1f} < {rollback_criteria['min_parity_score']:.1f}"
        
        if current_metrics.tracking_error_bps > rollback_criteria['max_tracking_error_bps']:
            return f"Tracking error breach: {current_metrics.tracking_error_bps:.1f} bps > {rollback_criteria['max_tracking_error_bps']:.1f} bps"
        
        return None
    
    def _validate_staging_completion(self, deployment: CanaryDeployment) -> bool:
        """Validate staging completion requirements"""
        if not deployment.staging_start:
            return False
        
        # Check minimum duration
        duration = datetime.now() - deployment.staging_start
        if duration.days < deployment.config.staging_duration_days:
            logger.warning(f"Staging duration {duration.days} days < required {deployment.config.staging_duration_days} days")
            return False
        
        # Check staging metrics
        if not deployment.staging_metrics:
            logger.warning("No staging metrics available")
            return False
        
        metrics = deployment.staging_metrics
        criteria = deployment.config.promotion_criteria
        
        # Validate promotion criteria
        if metrics.sharpe_ratio < criteria['min_sharpe_ratio']:
            logger.warning(f"Sharpe ratio {metrics.sharpe_ratio:.2f} < required {criteria['min_sharpe_ratio']:.2f}")
            return False
        
        if metrics.max_drawdown_percent > criteria['max_drawdown_percent']:
            logger.warning(f"Max drawdown {metrics.max_drawdown_percent:.2f}% > limit {criteria['max_drawdown_percent']:.2f}%")
            return False
        
        if metrics.error_rate_percent > criteria['max_error_rate_percent']:
            logger.warning(f"Error rate {metrics.error_rate_percent:.2f}% > limit {criteria['max_error_rate_percent']:.2f}%")
            return False
        
        if metrics.parity_score < criteria['min_parity_score']:
            logger.warning(f"Parity score {metrics.parity_score:.1f} < required {criteria['min_parity_score']:.1f}")
            return False
        
        logger.info("Staging validation passed - ready for production canary")
        return True
    
    def _validate_canary_completion(self, deployment: CanaryDeployment) -> bool:
        """Validate canary completion requirements"""
        if not deployment.canary_start:
            return False
        
        # Check minimum duration
        duration = datetime.now() - deployment.canary_start
        if duration.total_seconds() < deployment.config.canary_duration_hours * 3600:
            logger.warning(f"Canary duration {duration.total_seconds()/3600:.1f}h < required {deployment.config.canary_duration_hours}h")
            return False
        
        # Check canary metrics
        if not deployment.canary_metrics:
            logger.warning("No canary metrics available")
            return False
        
        metrics = deployment.canary_metrics
        criteria = deployment.config.promotion_criteria
        
        # Validate promotion criteria (same as staging but stricter monitoring)
        if metrics.sharpe_ratio < criteria['min_sharpe_ratio']:
            logger.warning(f"Canary Sharpe ratio {metrics.sharpe_ratio:.2f} < required {criteria['min_sharpe_ratio']:.2f}")
            return False
        
        if metrics.max_drawdown_percent > criteria['max_drawdown_percent']:
            logger.warning(f"Canary max drawdown {metrics.max_drawdown_percent:.2f}% > limit {criteria['max_drawdown_percent']:.2f}%")
            return False
        
        if metrics.tracking_error_bps > criteria['max_tracking_error_bps']:
            logger.warning(f"Canary tracking error {metrics.tracking_error_bps:.1f} bps > limit {criteria['max_tracking_error_bps']:.1f} bps")
            return False
        
        logger.info("Canary validation passed - ready for full production")
        return True
    
    def _calculate_duration_hours(self, deployment: CanaryDeployment) -> float:
        """Calculate duration hours for current stage"""
        if deployment.current_stage == CanaryStage.STAGING and deployment.staging_start:
            return (datetime.now() - deployment.staging_start).total_seconds() / 3600
        elif deployment.current_stage == CanaryStage.PRODUCTION_CANARY and deployment.canary_start:
            return (datetime.now() - deployment.canary_start).total_seconds() / 3600
        return 0.0
    
    def _start_monitoring(self, deployment_id: str) -> None:
        """Start background monitoring thread"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(deployment_id,),
            daemon=True
        )
        self._monitoring_thread.start()
        logger.info("Started canary monitoring thread")
    
    def _stop_monitoring(self) -> None:
        """Stop background monitoring thread"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        logger.info("Stopped canary monitoring thread")
    
    def _monitoring_loop(self, deployment_id: str) -> None:
        """Background monitoring loop for risk breaches"""
        while self._monitoring_active:
            try:
                breach_reason = self.check_risk_breach(deployment_id)
                if breach_reason:
                    logger.critical(f"Risk breach detected: {breach_reason}")
                    self.rollback_deployment(deployment_id, breach_reason)
                    break
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive deployment status"""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            return None
        
        status = {
            'deployment_id': deployment_id,
            'stage': deployment.current_stage.value,
            'status': deployment.status.value,
            'created_at': deployment.created_at.isoformat(),
            'config': asdict(deployment.config),
            'risk_budget': asdict(deployment.risk_budget),
            'current_capital_used': deployment.current_capital_used,
            'breach_count': len(deployment.breach_events),
            'duration_info': {}
        }
        
        # Add stage durations
        if deployment.staging_start:
            staging_duration = (deployment.staging_end or datetime.now()) - deployment.staging_start
            status['duration_info']['staging_hours'] = staging_duration.total_seconds() / 3600
        
        if deployment.canary_start:
            canary_duration = (deployment.canary_end or datetime.now()) - deployment.canary_start
            status['duration_info']['canary_hours'] = canary_duration.total_seconds() / 3600
        
        # Add metrics
        if deployment.staging_metrics:
            status['staging_metrics'] = asdict(deployment.staging_metrics)
            status['staging_metrics']['stage'] = deployment.staging_metrics.stage.value
        
        if deployment.canary_metrics:
            status['canary_metrics'] = asdict(deployment.canary_metrics)
            status['canary_metrics']['stage'] = deployment.canary_metrics.stage.value
        
        return status
    
    def save_deployment(self, deployment_id: str) -> Path:
        """Save deployment state to file"""
        deployment = self.deployments.get(deployment_id)
        if not deployment:
            raise ValueError(f"Deployment not found: {deployment_id}")
        
        filename = f"canary_deployment_{deployment_id}.json"
        filepath = self.deployments_dir / filename
        
        status = self.get_deployment_status(deployment_id)
        
        with open(filepath, 'w') as f:
            json.dump(status, f, indent=2, default=str)
        
        logger.info(f"Deployment saved: {filepath}")
        return filepath


# Global canary deployment manager
_canary_manager: Optional[CanaryDeploymentManager] = None


def get_canary_manager() -> CanaryDeploymentManager:
    """Get global canary deployment manager"""
    global _canary_manager
    if _canary_manager is None:
        _canary_manager = CanaryDeploymentManager()
    return _canary_manager


def reset_canary_manager():
    """Reset canary manager (for testing)"""
    global _canary_manager
    _canary_manager = None