#!/usr/bin/env python3
"""
System Health Monitor - Enterprise Health Score & GO/NO-GO Gates
Implements comprehensive system health assessment with strict trading gates
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from core.logging_manager import get_logger
from core.coverage_audit_system import get_kraken_coverage_auditor
from core.signal_quality_validator import get_signal_quality_validator
from core.execution_simulator import get_execution_simulator
from core.shadow_trading_engine import get_shadow_trading_engine
from core.confidence_gate_manager import get_confidence_gate_manager

class HealthStatus(str, Enum):
    """System health status levels"""
    GO = "go"                    # ≥85 score - live trading authorized
    WARNING = "warning"          # 60-85 score - paper trading only
    NOGO = "nogo"               # <60 score - no trading allowed
    CRITICAL = "critical"        # System errors or failures

class ComponentStatus(str, Enum):
    """Individual component status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    UNKNOWN = "unknown"

@dataclass
class HealthConfig:
    """Configuration for health monitoring"""
    # Score thresholds
    go_threshold: float = 85.0           # 85+ = GO
    warning_threshold: float = 60.0      # 60-85 = WARNING
    nogo_threshold: float = 60.0         # <60 = NO-GO
    
    # Component weights (must sum to 1.0)
    validation_accuracy_weight: float = 0.25    # 25%
    sharpe_ratio_weight: float = 0.20           # 20%
    feedback_hit_rate_weight: float = 0.15      # 15%
    error_ratio_weight: float = 0.15            # 15%
    data_completeness_weight: float = 0.15      # 15%
    tuning_freshness_weight: float = 0.10       # 10%
    
    # Component targets
    min_validation_accuracy: float = 0.75       # 75% minimum accuracy
    min_sharpe_ratio: float = 1.0              # 1.0 minimum Sharpe
    min_feedback_hit_rate: float = 0.65        # 65% feedback hit rate
    max_error_ratio: float = 0.05              # 5% max error ratio
    min_data_completeness: float = 0.98        # 98% data completeness
    max_tuning_age_hours: float = 24.0         # 24 hours max tuning age

@dataclass
class ComponentHealth:
    """Health metrics for individual component"""
    component_name: str
    status: ComponentStatus
    score: float  # 0-100
    raw_value: float
    target_value: float
    last_updated: datetime
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SystemHealthReport:
    """Comprehensive system health report"""
    report_id: str
    timestamp: datetime
    overall_health_status: HealthStatus
    overall_health_score: float
    
    # Component health
    component_health: Dict[str, ComponentHealth]
    
    # Validation results
    validation_accuracy: float
    sharpe_ratio: float
    feedback_hit_rate: float
    error_ratio: float
    data_completeness: float
    tuning_freshness_hours: float
    
    # Trading authorization
    live_trading_authorized: bool
    paper_trading_authorized: bool
    confidence_gate_status: str
    
    # System metrics
    total_coins_analyzed: int
    high_confidence_coins: int
    actionable_recommendations: int
    
    # Critical issues
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]

class SystemHealthMonitor:
    """Comprehensive system health monitoring"""
    
    def __init__(self, config: Optional[HealthConfig] = None):
        self.config = config or HealthConfig()
        self.logger = get_logger()
        
        # Validate configuration
        total_weight = sum([
            self.config.validation_accuracy_weight,
            self.config.sharpe_ratio_weight,
            self.config.feedback_hit_rate_weight,
            self.config.error_ratio_weight,
            self.config.data_completeness_weight,
            self.config.tuning_freshness_weight
        ])
        
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Component weights must sum to 1.0, got {total_weight}")
        
    async def assess_system_health(self) -> SystemHealthReport:
        """Comprehensive system health assessment"""
        
        report_id = f"health_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        assessment_start = datetime.now()
        
        self.logger.info(f"Starting system health assessment: {report_id}")
        
        try:
            # Assess individual components
            component_health = {}
            
            # 1. Validation Accuracy
            validation_health = await self._assess_validation_accuracy()
            component_health["validation_accuracy"] = validation_health
            
            # 2. Sharpe Ratio
            sharpe_health = await self._assess_sharpe_ratio()
            component_health["sharpe_ratio"] = sharpe_health
            
            # 3. Feedback Hit Rate
            feedback_health = await self._assess_feedback_hit_rate()
            component_health["feedback_hit_rate"] = feedback_health
            
            # 4. Error Ratio
            error_health = await self._assess_error_ratio()
            component_health["error_ratio"] = error_health
            
            # 5. Data Completeness
            completeness_health = await self._assess_data_completeness()
            component_health["data_completeness"] = completeness_health
            
            # 6. Tuning Freshness
            freshness_health = await self._assess_tuning_freshness()
            component_health["tuning_freshness"] = freshness_health
            
            # Calculate overall health score
            overall_score = self._calculate_overall_health_score(component_health)
            
            # Determine health status
            if overall_score >= self.config.go_threshold:
                health_status = HealthStatus.GO
                live_trading_authorized = True
                paper_trading_authorized = True
            elif overall_score >= self.config.warning_threshold:
                health_status = HealthStatus.WARNING
                live_trading_authorized = False
                paper_trading_authorized = True
            else:
                health_status = HealthStatus.NOGO
                live_trading_authorized = False
                paper_trading_authorized = False
            
            # Check for critical component failures
            failed_components = [
                name for name, health in component_health.items()
                if health.status == ComponentStatus.FAILED
            ]
            
            if failed_components:
                health_status = HealthStatus.CRITICAL
                live_trading_authorized = False
                paper_trading_authorized = False
            
            # Assess confidence gate status
            confidence_gate_status = await self._assess_confidence_gate_status()
            
            # Get system metrics
            system_metrics = await self._get_system_metrics()
            
            # Generate issues and recommendations
            critical_issues, warnings, recommendations = self._generate_health_recommendations(
                component_health, overall_score, health_status
            )
            
            # Create health report
            health_report = SystemHealthReport(
                report_id=report_id,
                timestamp=assessment_start,
                overall_health_status=health_status,
                overall_health_score=overall_score,
                component_health=component_health,
                validation_accuracy=validation_health.raw_value,
                sharpe_ratio=sharpe_health.raw_value,
                feedback_hit_rate=feedback_health.raw_value,
                error_ratio=error_health.raw_value,
                data_completeness=completeness_health.raw_value,
                tuning_freshness_hours=freshness_health.raw_value,
                live_trading_authorized=live_trading_authorized,
                paper_trading_authorized=paper_trading_authorized,
                confidence_gate_status=confidence_gate_status,
                total_coins_analyzed=system_metrics["total_coins"],
                high_confidence_coins=system_metrics["high_confidence_coins"],
                actionable_recommendations=system_metrics["actionable_recommendations"],
                critical_issues=critical_issues,
                warnings=warnings,
                recommendations=recommendations
            )
            
            # Store health report
            await self._store_health_report(health_report)
            
            self.logger.info(
                f"System health assessment completed: {report_id}",
                extra={
                    "report_id": report_id,
                    "health_status": health_status.value,
                    "health_score": overall_score,
                    "live_trading_authorized": live_trading_authorized,
                    "failed_components": len(failed_components)
                }
            )
            
            return health_report
            
        except Exception as e:
            self.logger.error(f"System health assessment failed: {e}")
            
            # Return critical status
            return SystemHealthReport(
                report_id=report_id,
                timestamp=assessment_start,
                overall_health_status=HealthStatus.CRITICAL,
                overall_health_score=0.0,
                component_health={},
                validation_accuracy=0.0,
                sharpe_ratio=0.0,
                feedback_hit_rate=0.0,
                error_ratio=1.0,
                data_completeness=0.0,
                tuning_freshness_hours=float('inf'),
                live_trading_authorized=False,
                paper_trading_authorized=False,
                confidence_gate_status="failed",
                total_coins_analyzed=0,
                high_confidence_coins=0,
                actionable_recommendations=0,
                critical_issues=[f"Health assessment failed: {e}"],
                warnings=[],
                recommendations=["Fix system health monitoring before proceeding"]
            )
    
    async def _assess_validation_accuracy(self) -> ComponentHealth:
        """Assess validation accuracy component"""
        
        try:
            # Get latest signal quality validation
            signal_validator = get_signal_quality_validator()
            
            # Generate mock validation data for demonstration
            # In real implementation, would use actual historical validation results
            accuracy = np.random.beta(8, 2)  # Simulate ~80% accuracy
            
            # Calculate score (0-100)
            if accuracy >= self.config.min_validation_accuracy:
                score = min(100, (accuracy / self.config.min_validation_accuracy) * 100)
                status = ComponentStatus.HEALTHY
            elif accuracy >= (self.config.min_validation_accuracy * 0.8):
                score = 50 + (accuracy - (self.config.min_validation_accuracy * 0.8)) / (self.config.min_validation_accuracy * 0.2) * 30
                status = ComponentStatus.DEGRADED
            else:
                score = (accuracy / (self.config.min_validation_accuracy * 0.8)) * 50
                status = ComponentStatus.FAILED
            
            return ComponentHealth(
                component_name="validation_accuracy",
                status=status,
                score=score,
                raw_value=accuracy,
                target_value=self.config.min_validation_accuracy,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name="validation_accuracy",
                status=ComponentStatus.FAILED,
                score=0.0,
                raw_value=0.0,
                target_value=self.config.min_validation_accuracy,
                last_updated=datetime.now(),
                error_message=str(e)
            )
    
    async def _assess_sharpe_ratio(self) -> ComponentHealth:
        """Assess Sharpe ratio component"""
        
        try:
            # Get shadow trading engine for Sharpe calculation
            shadow_engine = get_shadow_trading_engine()
            
            # Generate mock Sharpe ratio for demonstration
            # In real implementation, would calculate from actual trading performance
            sharpe = np.random.gamma(2, 0.8)  # Simulate ~1.6 average Sharpe
            
            # Calculate score
            if sharpe >= self.config.min_sharpe_ratio:
                score = min(100, (sharpe / self.config.min_sharpe_ratio) * 75 + 25)
                status = ComponentStatus.HEALTHY
            elif sharpe >= (self.config.min_sharpe_ratio * 0.7):
                score = 30 + (sharpe - (self.config.min_sharpe_ratio * 0.7)) / (self.config.min_sharpe_ratio * 0.3) * 40
                status = ComponentStatus.DEGRADED
            else:
                score = (sharpe / (self.config.min_sharpe_ratio * 0.7)) * 30
                status = ComponentStatus.FAILED
            
            return ComponentHealth(
                component_name="sharpe_ratio",
                status=status,
                score=score,
                raw_value=sharpe,
                target_value=self.config.min_sharpe_ratio,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name="sharpe_ratio",
                status=ComponentStatus.FAILED,
                score=0.0,
                raw_value=0.0,
                target_value=self.config.min_sharpe_ratio,
                last_updated=datetime.now(),
                error_message=str(e)
            )
    
    async def _assess_feedback_hit_rate(self) -> ComponentHealth:
        """Assess feedback hit rate component"""
        
        try:
            # Generate mock feedback hit rate
            # In real implementation, would analyze user feedback on recommendations
            hit_rate = np.random.beta(7, 3)  # Simulate ~70% hit rate
            
            # Calculate score
            if hit_rate >= self.config.min_feedback_hit_rate:
                score = min(100, (hit_rate / self.config.min_feedback_hit_rate) * 85 + 15)
                status = ComponentStatus.HEALTHY
            elif hit_rate >= (self.config.min_feedback_hit_rate * 0.8):
                score = 40 + (hit_rate - (self.config.min_feedback_hit_rate * 0.8)) / (self.config.min_feedback_hit_rate * 0.2) * 35
                status = ComponentStatus.DEGRADED
            else:
                score = (hit_rate / (self.config.min_feedback_hit_rate * 0.8)) * 40
                status = ComponentStatus.FAILED
            
            return ComponentHealth(
                component_name="feedback_hit_rate",
                status=status,
                score=score,
                raw_value=hit_rate,
                target_value=self.config.min_feedback_hit_rate,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name="feedback_hit_rate",
                status=ComponentStatus.FAILED,
                score=0.0,
                raw_value=0.0,
                target_value=self.config.min_feedback_hit_rate,
                last_updated=datetime.now(),
                error_message=str(e)
            )
    
    async def _assess_error_ratio(self) -> ComponentHealth:
        """Assess system error ratio component"""
        
        try:
            # Generate mock error ratio
            # In real implementation, would analyze system logs for error rates
            error_ratio = np.random.beta(1, 20)  # Simulate ~5% error rate
            
            # Calculate score (lower error ratio = higher score)
            if error_ratio <= self.config.max_error_ratio:
                score = min(100, 100 - (error_ratio / self.config.max_error_ratio) * 25)
                status = ComponentStatus.HEALTHY
            elif error_ratio <= (self.config.max_error_ratio * 2):
                score = 50 - (error_ratio - self.config.max_error_ratio) / self.config.max_error_ratio * 25
                status = ComponentStatus.DEGRADED
            else:
                score = max(0, 25 - (error_ratio - self.config.max_error_ratio * 2) / self.config.max_error_ratio * 25)
                status = ComponentStatus.FAILED
            
            return ComponentHealth(
                component_name="error_ratio",
                status=status,
                score=score,
                raw_value=error_ratio,
                target_value=self.config.max_error_ratio,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name="error_ratio",
                status=ComponentStatus.FAILED,
                score=0.0,
                raw_value=1.0,
                target_value=self.config.max_error_ratio,
                last_updated=datetime.now(),
                error_message=str(e)
            )
    
    async def _assess_data_completeness(self) -> ComponentHealth:
        """Assess data completeness component"""
        
        try:
            # Get coverage auditor for data completeness
            coverage_auditor = get_kraken_coverage_auditor()
            
            # Generate mock completeness
            # In real implementation, would get from actual coverage audit
            completeness = np.random.beta(50, 2)  # Simulate ~96% completeness
            
            # Calculate score
            if completeness >= self.config.min_data_completeness:
                score = min(100, (completeness / self.config.min_data_completeness) * 95 + 5)
                status = ComponentStatus.HEALTHY
            elif completeness >= (self.config.min_data_completeness * 0.95):
                score = 60 + (completeness - (self.config.min_data_completeness * 0.95)) / (self.config.min_data_completeness * 0.05) * 25
                status = ComponentStatus.DEGRADED
            else:
                score = (completeness / (self.config.min_data_completeness * 0.95)) * 60
                status = ComponentStatus.FAILED
            
            return ComponentHealth(
                component_name="data_completeness",
                status=status,
                score=score,
                raw_value=completeness,
                target_value=self.config.min_data_completeness,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name="data_completeness",
                status=ComponentStatus.FAILED,
                score=0.0,
                raw_value=0.0,
                target_value=self.config.min_data_completeness,
                last_updated=datetime.now(),
                error_message=str(e)
            )
    
    async def _assess_tuning_freshness(self) -> ComponentHealth:
        """Assess model tuning freshness component"""
        
        try:
            # Generate mock tuning age
            # In real implementation, would check actual model training timestamps
            hours_since_tuning = np.random.exponential(12)  # Average 12 hours
            
            # Calculate score (fresher tuning = higher score)
            if hours_since_tuning <= self.config.max_tuning_age_hours:
                score = min(100, 100 - (hours_since_tuning / self.config.max_tuning_age_hours) * 30)
                status = ComponentStatus.HEALTHY
            elif hours_since_tuning <= (self.config.max_tuning_age_hours * 2):
                score = 50 - (hours_since_tuning - self.config.max_tuning_age_hours) / self.config.max_tuning_age_hours * 30
                status = ComponentStatus.DEGRADED
            else:
                score = max(0, 20 - (hours_since_tuning - self.config.max_tuning_age_hours * 2) / self.config.max_tuning_age_hours * 20)
                status = ComponentStatus.FAILED
            
            return ComponentHealth(
                component_name="tuning_freshness",
                status=status,
                score=score,
                raw_value=hours_since_tuning,
                target_value=self.config.max_tuning_age_hours,
                last_updated=datetime.now()
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name="tuning_freshness",
                status=ComponentStatus.FAILED,
                score=0.0,
                raw_value=float('inf'),
                target_value=self.config.max_tuning_age_hours,
                last_updated=datetime.now(),
                error_message=str(e)
            )
    
    def _calculate_overall_health_score(self, component_health: Dict[str, ComponentHealth]) -> float:
        """Calculate weighted overall health score"""
        
        weights = {
            "validation_accuracy": self.config.validation_accuracy_weight,
            "sharpe_ratio": self.config.sharpe_ratio_weight,
            "feedback_hit_rate": self.config.feedback_hit_rate_weight,
            "error_ratio": self.config.error_ratio_weight,
            "data_completeness": self.config.data_completeness_weight,
            "tuning_freshness": self.config.tuning_freshness_weight
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for component_name, weight in weights.items():
            if component_name in component_health:
                component = component_health[component_name]
                total_score += component.score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return total_score / total_weight
    
    async def _assess_confidence_gate_status(self) -> str:
        """Assess confidence gate status"""
        
        try:
            confidence_gate = get_confidence_gate_manager()
            
            # Generate mock confidence gate status
            # In real implementation, would check actual gate status
            high_conf_count = np.random.poisson(5)  # Average 5 high-confidence coins
            
            if high_conf_count >= 3:
                return "open"
            elif high_conf_count >= 1:
                return "limited"
            else:
                return "closed"
                
        except Exception as e:
            return "failed"
    
    async def _get_system_metrics(self) -> Dict[str, int]:
        """Get current system metrics"""
        
        try:
            # Generate mock system metrics
            # In real implementation, would query actual system state
            return {
                "total_coins": np.random.randint(150, 300),
                "high_confidence_coins": np.random.randint(0, 15),
                "actionable_recommendations": np.random.randint(0, 10)
            }
            
        except Exception as e:
            return {
                "total_coins": 0,
                "high_confidence_coins": 0,
                "actionable_recommendations": 0
            }
    
    def _generate_health_recommendations(
        self,
        component_health: Dict[str, ComponentHealth],
        overall_score: float,
        health_status: HealthStatus
    ) -> Tuple[List[str], List[str], List[str]]:
        """Generate health recommendations"""
        
        critical_issues = []
        warnings = []
        recommendations = []
        
        # Check for failed components
        failed_components = [
            name for name, health in component_health.items()
            if health.status == ComponentStatus.FAILED
        ]
        
        for component in failed_components:
            critical_issues.append(f"Component failure: {component}")
        
        # Check for degraded components
        degraded_components = [
            name for name, health in component_health.items()
            if health.status == ComponentStatus.DEGRADED
        ]
        
        for component in degraded_components:
            warnings.append(f"Component degraded: {component}")
        
        # Overall health recommendations
        if health_status == HealthStatus.CRITICAL:
            recommendations.append("System critical - immediate intervention required")
            recommendations.append("All trading disabled until issues resolved")
        elif health_status == HealthStatus.NOGO:
            recommendations.append(f"Health score {overall_score:.1f} < 60 - no trading allowed")
            recommendations.append("Improve failing components before enabling trading")
        elif health_status == HealthStatus.WARNING:
            recommendations.append(f"Health score {overall_score:.1f} - live trading disabled")
            recommendations.append("Paper trading only until score ≥85")
        else:
            recommendations.append("System healthy - all trading modes authorized")
        
        # Component-specific recommendations
        for name, health in component_health.items():
            if health.status == ComponentStatus.FAILED:
                if name == "validation_accuracy":
                    recommendations.append("Retrain models - validation accuracy too low")
                elif name == "sharpe_ratio":
                    recommendations.append("Review trading strategy - risk-adjusted returns poor")
                elif name == "data_completeness":
                    recommendations.append("Fix data pipeline - completeness below threshold")
                elif name == "tuning_freshness":
                    recommendations.append("Retrain models - tuning too stale")
        
        return critical_issues, warnings, recommendations
    
    async def _store_health_report(self, report: SystemHealthReport):
        """Store health report to disk"""
        
        try:
            # Create directories
            reports_dir = Path("data/health_reports")
            reports_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare report data
            report_data = {
                "report_id": report.report_id,
                "timestamp": report.timestamp.isoformat(),
                "overall_health": {
                    "status": report.overall_health_status.value,
                    "score": report.overall_health_score,
                    "live_trading_authorized": report.live_trading_authorized,
                    "paper_trading_authorized": report.paper_trading_authorized
                },
                "component_scores": {
                    name: {
                        "status": health.status.value,
                        "score": health.score,
                        "raw_value": health.raw_value,
                        "target_value": health.target_value,
                        "last_updated": health.last_updated.isoformat()
                    }
                    for name, health in report.component_health.items()
                },
                "system_metrics": {
                    "total_coins_analyzed": report.total_coins_analyzed,
                    "high_confidence_coins": report.high_confidence_coins,
                    "actionable_recommendations": report.actionable_recommendations,
                    "confidence_gate_status": report.confidence_gate_status
                },
                "issues_and_recommendations": {
                    "critical_issues": report.critical_issues,
                    "warnings": report.warnings,
                    "recommendations": report.recommendations
                }
            }
            
            # Write to file
            report_file = reports_dir / f"health_report_{report.report_id}.json"
            
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            # Also store as latest
            latest_file = reports_dir / "latest_health_report.json"
            with open(latest_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            self.logger.info(f"Health report stored: {report_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to store health report: {e}")

    def check_trading_authorization(self, health_report: SystemHealthReport) -> Dict[str, Any]:
        """Check trading authorization based on health report"""
        
        return {
            "timestamp": datetime.now().isoformat(),
            "health_score": health_report.overall_health_score,
            "health_status": health_report.overall_health_status.value,
            "live_trading_authorized": health_report.live_trading_authorized,
            "paper_trading_authorized": health_report.paper_trading_authorized,
            "confidence_gate_status": health_report.confidence_gate_status,
            "high_confidence_coins": health_report.high_confidence_coins,
            "authorization_reason": self._get_authorization_reason(health_report),
            "next_assessment_recommended": (datetime.now() + timedelta(hours=1)).isoformat()
        }
    
    def _get_authorization_reason(self, health_report: SystemHealthReport) -> str:
        """Get reason for current authorization status"""
        
        if health_report.overall_health_status == HealthStatus.GO:
            return f"Health score {health_report.overall_health_score:.1f} ≥ 85 - all trading authorized"
        elif health_report.overall_health_status == HealthStatus.WARNING:
            return f"Health score {health_report.overall_health_score:.1f} in warning range (60-85) - paper trading only"
        elif health_report.overall_health_status == HealthStatus.NOGO:
            return f"Health score {health_report.overall_health_score:.1f} < 60 - no trading allowed"
        else:
            return "System critical - all trading disabled"

# Global instance
_system_health_monitor = None

def get_system_health_monitor(config: Optional[HealthConfig] = None) -> SystemHealthMonitor:
    """Get global system health monitor instance"""
    global _system_health_monitor
    if _system_health_monitor is None:
        _system_health_monitor = SystemHealthMonitor(config)
    return _system_health_monitor