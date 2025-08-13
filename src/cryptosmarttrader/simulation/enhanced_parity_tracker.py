"""
Enhanced Parity Tracker - FASE D IMPLEMENTATION  
Advanced tracking error monitoring with component attribution and auto-disable
"""

import time
import json
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path
import logging

from ..core.structured_logger import get_logger
from ..observability.metrics import PrometheusMetrics
from .advanced_execution_simulator import SimulatedExecution, AdvancedExecutionSimulator

logger = get_logger(__name__)


class ParityStatus(Enum):
    """Parity tracking status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DISABLED = "disabled"


class DriftComponent(Enum):
    """Components contributing to tracking error drift"""
    SLIPPAGE = "slippage"
    FEES = "fees" 
    TIMING = "timing"
    LATENCY = "latency"
    MARKET_IMPACT = "market_impact"
    MODELING_ERROR = "modeling_error"


@dataclass
class ParityMetrics:
    """Comprehensive parity tracking metrics"""
    
    # Core tracking error metrics
    daily_tracking_error_bps: float = 0.0
    cumulative_tracking_error_bps: float = 0.0
    rolling_7d_tracking_error_bps: float = 0.0
    rolling_30d_tracking_error_bps: float = 0.0
    
    # Component attribution
    slippage_contribution_bps: float = 0.0
    fees_contribution_bps: float = 0.0
    timing_contribution_bps: float = 0.0
    latency_contribution_bps: float = 0.0
    impact_contribution_bps: float = 0.0
    modeling_error_bps: float = 0.0
    
    # Performance metrics
    total_simulations: int = 0
    total_volume_simulated: float = 0.0
    avg_execution_accuracy_pct: float = 100.0
    
    # Drift detection
    drift_trend_bps_per_day: float = 0.0
    consecutive_threshold_breaches: int = 0
    last_threshold_breach: Optional[datetime] = None
    
    # Status tracking
    parity_status: ParityStatus = ParityStatus.HEALTHY
    auto_disable_triggered: bool = False
    manual_intervention_required: bool = False


@dataclass
class TrackingErrorRecord:
    """Individual tracking error record"""
    timestamp: datetime
    tracking_error_bps: float
    component_breakdown: Dict[str, float]
    execution_count: int
    volume_usd: float
    strategy_id: str


@dataclass
class ParityReport:
    """Comprehensive parity tracking report"""
    report_id: str
    period_start: datetime
    period_end: datetime
    metrics: ParityMetrics
    daily_records: List[TrackingErrorRecord]
    component_analysis: Dict[str, Any]
    recommendations: List[str]
    status_summary: str


class EnhancedParityTracker:
    """
    ENHANCED PARITY TRACKER - FASE D IMPLEMENTATION
    
    Features:
    ✅ Advanced tracking error calculation in basis points
    ✅ Component attribution (slippage/fees/timing/latency) 
    ✅ Daily reporting with component breakdown
    ✅ Auto-disable when drift >100 bps
    ✅ Emergency alerting system
    ✅ Persistent state management
    ✅ Multi-strategy support with thresholds
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Tracking error thresholds
        self.warning_threshold_bps = self.config.get('warning_threshold_bps', 20.0)
        self.critical_threshold_bps = self.config.get('critical_threshold_bps', 50.0)
        self.auto_disable_threshold_bps = self.config.get('auto_disable_threshold_bps', 100.0)
        
        # Monitoring windows
        self.daily_window_hours = 24
        self.rolling_window_days = 7
        self.long_rolling_window_days = 30
        
        # State management
        self.metrics = ParityMetrics()
        self.tracking_records: List[TrackingErrorRecord] = []
        self.daily_reports: List[ParityReport] = []
        
        # Alert management
        self.alert_cooldown_minutes = 15
        self.last_alert_time: Optional[datetime] = None
        self.consecutive_breaches = 0
        
        # Integration components
        self.prometheus_metrics = PrometheusMetrics.get_instance()
        self.execution_simulator: Optional[AdvancedExecutionSimulator] = None
        
        # Persistence
        self.persistence_file = Path(self.config.get('persistence_file', 'data/parity_tracker_state.json'))
        self.persistence_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Load previous state
        self._load_state()
        
        logger.info("EnhancedParityTracker initialized", extra={
            'warning_threshold_bps': self.warning_threshold_bps,
            'critical_threshold_bps': self.critical_threshold_bps,
            'auto_disable_threshold_bps': self.auto_disable_threshold_bps,
            'persistence_enabled': self.persistence_file.exists()
        })
    
    def set_execution_simulator(self, simulator: AdvancedExecutionSimulator):
        """Set execution simulator for integrated monitoring"""
        self.execution_simulator = simulator
    
    def track_execution(
        self, 
        simulated_execution: SimulatedExecution,
        actual_execution: Optional[Dict[str, Any]] = None,
        strategy_id: str = "default"
    ):
        """Track execution and calculate tracking error"""
        
        with self._lock:
            # Calculate tracking error if actual execution provided
            if actual_execution:
                tracking_error_bps = self._calculate_tracking_error(simulated_execution, actual_execution)
                component_breakdown = self._calculate_component_breakdown(simulated_execution, actual_execution)
            else:
                # Use simulated execution as baseline (for testing)
                tracking_error_bps = simulated_execution.slippage_bps
                component_breakdown = simulated_execution.slippage_breakdown
            
            # Create tracking record
            record = TrackingErrorRecord(
                timestamp=datetime.now(),
                tracking_error_bps=tracking_error_bps,
                component_breakdown=component_breakdown,
                execution_count=1,
                volume_usd=simulated_execution.filled_quantity * simulated_execution.average_fill_price,
                strategy_id=strategy_id
            )
            
            self.tracking_records.append(record)
            
            # Update metrics
            self._update_metrics(record)
            
            # Check thresholds and alerts
            self._check_thresholds()
            
            # Persistence
            self._save_state()
            
            logger.debug("Execution tracked", extra={
                'tracking_error_bps': tracking_error_bps,
                'volume_usd': record.volume_usd,
                'strategy_id': strategy_id,
                'parity_status': self.metrics.parity_status.value
            })
    
    def _calculate_tracking_error(
        self, 
        simulated: SimulatedExecution, 
        actual: Dict[str, Any]
    ) -> float:
        """Calculate tracking error between simulated and actual execution"""
        
        # Get actual execution metrics
        actual_avg_price = actual.get('average_fill_price', simulated.average_fill_price)
        actual_total_fees = actual.get('total_fees', simulated.total_fees)
        actual_slippage_bps = actual.get('slippage_bps', simulated.slippage_bps)
        
        # Calculate price difference
        price_diff_bps = abs(actual_avg_price - simulated.average_fill_price) / simulated.average_fill_price * 10000
        
        # Calculate fee difference  
        fee_diff_bps = abs(actual_total_fees - simulated.total_fees) / (simulated.filled_quantity * simulated.average_fill_price) * 10000
        
        # Calculate slippage difference
        slippage_diff_bps = abs(actual_slippage_bps - simulated.slippage_bps)
        
        # Total tracking error
        total_tracking_error = price_diff_bps + fee_diff_bps + slippage_diff_bps
        
        return total_tracking_error
    
    def _calculate_component_breakdown(
        self, 
        simulated: SimulatedExecution, 
        actual: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate component-wise tracking error breakdown"""
        
        actual_avg_price = actual.get('average_fill_price', simulated.average_fill_price)
        actual_total_fees = actual.get('total_fees', simulated.total_fees)
        actual_execution_time = actual.get('execution_time_ms', simulated.execution_time_ms)
        
        breakdown = {}
        
        # Slippage component
        breakdown[DriftComponent.SLIPPAGE.value] = abs(
            actual.get('slippage_bps', simulated.slippage_bps) - simulated.slippage_bps
        )
        
        # Fees component
        fee_base = simulated.filled_quantity * simulated.average_fill_price
        if fee_base > 0:
            breakdown[DriftComponent.FEES.value] = abs(actual_total_fees - simulated.total_fees) / fee_base * 10000
        else:
            breakdown[DriftComponent.FEES.value] = 0.0
        
        # Timing component (price difference)
        breakdown[DriftComponent.TIMING.value] = abs(actual_avg_price - simulated.average_fill_price) / simulated.average_fill_price * 10000
        
        # Latency component  
        latency_diff_ms = abs(actual_execution_time - simulated.execution_time_ms)
        breakdown[DriftComponent.LATENCY.value] = min(latency_diff_ms / 100.0, 10.0)  # Cap at 10 bps
        
        # Market impact component
        breakdown[DriftComponent.MARKET_IMPACT.value] = simulated.price_impact_bps * 0.1  # 10% attribution to modeling error
        
        # Modeling error (residual)
        total_explained = sum(breakdown.values())
        total_actual = sum(breakdown.values()) + breakdown.get(DriftComponent.SLIPPAGE.value, 0)
        breakdown[DriftComponent.MODELING_ERROR.value] = max(0, total_actual - total_explained)
        
        return breakdown
    
    def _update_metrics(self, record: TrackingErrorRecord):
        """Update comprehensive parity metrics"""
        
        # Update core metrics
        self.metrics.total_simulations += record.execution_count
        self.metrics.total_volume_simulated += record.volume_usd
        
        # Calculate rolling metrics
        self.metrics.daily_tracking_error_bps = self._calculate_rolling_tracking_error(hours=24)
        self.metrics.rolling_7d_tracking_error_bps = self._calculate_rolling_tracking_error(days=7)
        self.metrics.rolling_30d_tracking_error_bps = self._calculate_rolling_tracking_error(days=30)
        
        # Update component contributions (daily)
        daily_records = self._get_records_in_window(hours=24)
        if daily_records:
            self.metrics.slippage_contribution_bps = np.mean([
                r.component_breakdown.get(DriftComponent.SLIPPAGE.value, 0) for r in daily_records
            ])
            self.metrics.fees_contribution_bps = np.mean([
                r.component_breakdown.get(DriftComponent.FEES.value, 0) for r in daily_records
            ])
            self.metrics.timing_contribution_bps = np.mean([
                r.component_breakdown.get(DriftComponent.TIMING.value, 0) for r in daily_records
            ])
            self.metrics.latency_contribution_bps = np.mean([
                r.component_breakdown.get(DriftComponent.LATENCY.value, 0) for r in daily_records
            ])
            self.metrics.impact_contribution_bps = np.mean([
                r.component_breakdown.get(DriftComponent.MARKET_IMPACT.value, 0) for r in daily_records
            ])
            self.metrics.modeling_error_bps = np.mean([
                r.component_breakdown.get(DriftComponent.MODELING_ERROR.value, 0) for r in daily_records
            ])
        
        # Calculate drift trend
        self.metrics.drift_trend_bps_per_day = self._calculate_drift_trend()
        
        # Update execution accuracy
        if self.metrics.total_simulations > 0:
            accurate_simulations = sum(1 for r in self.tracking_records if r.tracking_error_bps <= self.warning_threshold_bps)
            self.metrics.avg_execution_accuracy_pct = (accurate_simulations / self.metrics.total_simulations) * 100
        
        # Update Prometheus metrics
        self._update_prometheus_metrics()
    
    def _calculate_rolling_tracking_error(self, hours: int = None, days: int = None) -> float:
        """Calculate volume-weighted tracking error for rolling window"""
        
        if hours:
            window_records = self._get_records_in_window(hours=hours)
        else:
            window_records = self._get_records_in_window(days=days)
        
        if not window_records:
            return 0.0
        
        total_volume = sum(r.volume_usd for r in window_records)
        if total_volume == 0:
            return 0.0
        
        weighted_error = sum(
            r.tracking_error_bps * r.volume_usd for r in window_records
        ) / total_volume
        
        return weighted_error
    
    def _get_records_in_window(self, hours: int = None, days: int = None) -> List[TrackingErrorRecord]:
        """Get tracking records within specified time window"""
        
        if hours:
            cutoff = datetime.now() - timedelta(hours=hours)
        else:
            cutoff = datetime.now() - timedelta(days=days)
        
        return [r for r in self.tracking_records if r.timestamp >= cutoff]
    
    def _calculate_drift_trend(self) -> float:
        """Calculate tracking error drift trend in bps per day"""
        
        # Get last 7 days of data
        recent_records = self._get_records_in_window(days=7)
        
        if len(recent_records) < 2:
            return 0.0
        
        # Group by day and calculate daily averages
        daily_errors = {}
        for record in recent_records:
            day = record.timestamp.date()
            if day not in daily_errors:
                daily_errors[day] = []
            daily_errors[day].append(record.tracking_error_bps)
        
        # Calculate trend using linear regression
        if len(daily_errors) < 2:
            return 0.0
        
        days = sorted(daily_errors.keys())
        day_numbers = [(d - days[0]).days for d in days]
        avg_errors = [np.mean(daily_errors[d]) for d in days]
        
        if len(day_numbers) < 2:
            return 0.0
        
        # Simple linear regression
        x_mean = np.mean(day_numbers)
        y_mean = np.mean(avg_errors)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(day_numbers, avg_errors))
        denominator = sum((x - x_mean) ** 2 for x in day_numbers)
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope
    
    def _check_thresholds(self):
        """Check tracking error thresholds and trigger alerts"""
        
        current_error = self.metrics.daily_tracking_error_bps
        
        # Determine status
        if current_error >= self.auto_disable_threshold_bps:
            self.metrics.parity_status = ParityStatus.DISABLED
            self.metrics.auto_disable_triggered = True
            self.metrics.manual_intervention_required = True
            self.consecutive_breaches += 1
            
            # Disable execution simulator if available
            if self.execution_simulator:
                self.execution_simulator.simulation_enabled = False
            
            self._trigger_emergency_alert("Auto-disable triggered", current_error)
            
        elif current_error >= self.critical_threshold_bps:
            self.metrics.parity_status = ParityStatus.CRITICAL
            self.consecutive_breaches += 1
            
            if self.consecutive_breaches >= 3:
                self.metrics.manual_intervention_required = True
            
            self._trigger_alert("Critical tracking error", current_error)
            
        elif current_error >= self.warning_threshold_bps:
            self.metrics.parity_status = ParityStatus.WARNING
            self.consecutive_breaches = 0
            
            self._trigger_alert("Warning tracking error", current_error)
            
        else:
            self.metrics.parity_status = ParityStatus.HEALTHY
            self.consecutive_breaches = 0
        
        # Update breach tracking
        if current_error >= self.warning_threshold_bps:
            self.metrics.last_threshold_breach = datetime.now()
            self.metrics.consecutive_threshold_breaches = self.consecutive_breaches
        else:
            self.metrics.consecutive_threshold_breaches = 0
    
    def _trigger_alert(self, alert_type: str, current_error: float):
        """Trigger tracking error alert with cooldown"""
        
        now = datetime.now()
        
        # Check cooldown
        if (self.last_alert_time and 
            (now - self.last_alert_time).total_seconds() < self.alert_cooldown_minutes * 60):
            return
        
        self.last_alert_time = now
        
        # Update Prometheus alert metrics
        if "critical" in alert_type.lower():
            self.prometheus_metrics.drawdown_too_high.set(1)  # Reuse for tracking error alerts
        
        logger.warning(f"Parity tracking alert: {alert_type}", extra={
            'alert_type': alert_type,
            'current_tracking_error_bps': current_error,
            'threshold_warning': self.warning_threshold_bps,
            'threshold_critical': self.critical_threshold_bps,
            'consecutive_breaches': self.consecutive_breaches,
            'parity_status': self.metrics.parity_status.value
        })
    
    def _trigger_emergency_alert(self, alert_type: str, current_error: float):
        """Trigger emergency alert for auto-disable"""
        
        logger.critical(f"PARITY EMERGENCY: {alert_type}", extra={
            'alert_type': alert_type,
            'current_tracking_error_bps': current_error,
            'auto_disable_threshold_bps': self.auto_disable_threshold_bps,
            'total_simulations': self.metrics.total_simulations,
            'simulation_disabled': True,
            'manual_intervention_required': True
        })
    
    def _update_prometheus_metrics(self):
        """Update Prometheus metrics"""
        
        # Core tracking error metrics
        self.prometheus_metrics.estimated_slippage_bps.labels(
            symbol="parity_tracking_daily"
        ).observe(self.metrics.daily_tracking_error_bps)
        
        self.prometheus_metrics.estimated_slippage_bps.labels(
            symbol="parity_tracking_7d"
        ).observe(self.metrics.rolling_7d_tracking_error_bps)
        
        # Component metrics
        components = [
            ('slippage', self.metrics.slippage_contribution_bps),
            ('fees', self.metrics.fees_contribution_bps),
            ('timing', self.metrics.timing_contribution_bps),
            ('latency', self.metrics.latency_contribution_bps),
            ('impact', self.metrics.impact_contribution_bps),
            ('modeling', self.metrics.modeling_error_bps)
        ]
        
        for component, value in components:
            self.prometheus_metrics.estimated_slippage_bps.labels(
                symbol=f"parity_{component}"
            ).observe(value)
    
    def generate_daily_report(self) -> ParityReport:
        """Generate comprehensive daily parity report"""
        
        report_id = f"parity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        period_end = datetime.now()
        period_start = period_end - timedelta(days=1)
        
        # Get daily records
        daily_records = [
            r for r in self.tracking_records 
            if period_start <= r.timestamp <= period_end
        ]
        
        # Component analysis
        component_analysis = self._analyze_components(daily_records)
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        # Status summary
        status_summary = self._generate_status_summary()
        
        report = ParityReport(
            report_id=report_id,
            period_start=period_start,
            period_end=period_end,
            metrics=self.metrics,
            daily_records=daily_records,
            component_analysis=component_analysis,
            recommendations=recommendations,
            status_summary=status_summary
        )
        
        self.daily_reports.append(report)
        
        # Keep last 30 days
        if len(self.daily_reports) > 30:
            self.daily_reports = self.daily_reports[-30:]
        
        logger.info("Daily parity report generated", extra={
            'report_id': report_id,
            'daily_tracking_error_bps': self.metrics.daily_tracking_error_bps,
            'parity_status': self.metrics.parity_status.value,
            'executions_tracked': len(daily_records)
        })
        
        return report
    
    def _analyze_components(self, records: List[TrackingErrorRecord]) -> Dict[str, Any]:
        """Analyze component contributions to tracking error"""
        
        if not records:
            return {}
        
        analysis = {}
        
        for component in DriftComponent:
            values = [r.component_breakdown.get(component.value, 0) for r in records]
            
            analysis[component.value] = {
                'mean_bps': np.mean(values),
                'p95_bps': np.percentile(values, 95),
                'max_bps': np.max(values),
                'contribution_pct': np.mean(values) / max(self.metrics.daily_tracking_error_bps, 1) * 100
            }
        
        return analysis
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on current state"""
        
        recommendations = []
        
        if self.metrics.parity_status == ParityStatus.DISABLED:
            recommendations.append("URGENT: Manual intervention required - execution simulator disabled")
            recommendations.append("Review tracking error components and adjust simulation parameters")
            recommendations.append("Validate real execution data quality")
        
        elif self.metrics.parity_status == ParityStatus.CRITICAL:
            recommendations.append("Investigate primary tracking error drivers")
            recommendations.append("Consider reducing simulation complexity")
            recommendations.append("Review market microstructure parameters")
        
        elif self.metrics.parity_status == ParityStatus.WARNING:
            recommendations.append("Monitor tracking error trend closely")
            recommendations.append("Review recent execution performance")
        
        # Component-specific recommendations
        if self.metrics.slippage_contribution_bps > 10:
            recommendations.append("High slippage contribution - review market impact model")
        
        if self.metrics.fees_contribution_bps > 5:
            recommendations.append("Fee estimation errors - verify exchange fee structure")
        
        if self.metrics.timing_contribution_bps > 8:
            recommendations.append("Timing errors detected - review execution latency modeling")
        
        if self.metrics.modeling_error_bps > 15:
            recommendations.append("High modeling error - consider model recalibration")
        
        return recommendations
    
    def _generate_status_summary(self) -> str:
        """Generate human-readable status summary"""
        
        summary_parts = [
            f"Status: {self.metrics.parity_status.value.upper()}",
            f"Daily tracking error: {self.metrics.daily_tracking_error_bps:.1f} bps",
            f"7-day rolling: {self.metrics.rolling_7d_tracking_error_bps:.1f} bps",
            f"Simulations: {self.metrics.total_simulations}",
            f"Accuracy: {self.metrics.avg_execution_accuracy_pct:.1f}%"
        ]
        
        if self.metrics.auto_disable_triggered:
            summary_parts.append("⚠️ AUTO-DISABLED")
        
        if self.metrics.manual_intervention_required:
            summary_parts.append("🔴 MANUAL INTERVENTION REQUIRED")
        
        return " | ".join(summary_parts)
    
    def _save_state(self):
        """Save current state to persistence file"""
        
        try:
            state_data = {
                'metrics': {
                    'daily_tracking_error_bps': self.metrics.daily_tracking_error_bps,
                    'total_simulations': self.metrics.total_simulations,
                    'total_volume_simulated': self.metrics.total_volume_simulated,
                    'parity_status': self.metrics.parity_status.value,
                    'auto_disable_triggered': self.metrics.auto_disable_triggered,
                    'consecutive_threshold_breaches': self.metrics.consecutive_threshold_breaches
                },
                'tracking_records_count': len(self.tracking_records),
                'last_update': datetime.now().isoformat()
            }
            
            with open(self.persistence_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save parity tracker state: {e}")
    
    def _load_state(self):
        """Load previous state from persistence file"""
        
        try:
            if self.persistence_file.exists():
                with open(self.persistence_file, 'r') as f:
                    state_data = json.load(f)
                
                # Restore metrics
                metrics_data = state_data.get('metrics', {})
                self.metrics.total_simulations = metrics_data.get('total_simulations', 0)
                self.metrics.total_volume_simulated = metrics_data.get('total_volume_simulated', 0.0)
                self.metrics.auto_disable_triggered = metrics_data.get('auto_disable_triggered', False)
                self.metrics.consecutive_threshold_breaches = metrics_data.get('consecutive_threshold_breaches', 0)
                
                status_str = metrics_data.get('parity_status', 'healthy')
                self.metrics.parity_status = ParityStatus(status_str)
                
                logger.info("Parity tracker state loaded", extra={
                    'total_simulations': self.metrics.total_simulations,
                    'parity_status': self.metrics.parity_status.value,
                    'auto_disable_triggered': self.metrics.auto_disable_triggered
                })
                
        except Exception as e:
            logger.warning(f"Failed to load parity tracker state: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        return {
            'parity_status': self.metrics.parity_status.value,
            'daily_tracking_error_bps': self.metrics.daily_tracking_error_bps,
            'rolling_7d_tracking_error_bps': self.metrics.rolling_7d_tracking_error_bps,
            'rolling_30d_tracking_error_bps': self.metrics.rolling_30d_tracking_error_bps,
            'total_simulations': self.metrics.total_simulations,
            'execution_accuracy_pct': self.metrics.avg_execution_accuracy_pct,
            'auto_disable_triggered': self.metrics.auto_disable_triggered,
            'manual_intervention_required': self.metrics.manual_intervention_required,
            'consecutive_breaches': self.metrics.consecutive_threshold_breaches,
            'drift_trend_bps_per_day': self.metrics.drift_trend_bps_per_day,
            'component_contributions': {
                'slippage_bps': self.metrics.slippage_contribution_bps,
                'fees_bps': self.metrics.fees_contribution_bps,
                'timing_bps': self.metrics.timing_contribution_bps,
                'latency_bps': self.metrics.latency_contribution_bps,
                'impact_bps': self.metrics.impact_contribution_bps,
                'modeling_error_bps': self.metrics.modeling_error_bps
            },
            'thresholds': {
                'warning_bps': self.warning_threshold_bps,
                'critical_bps': self.critical_threshold_bps,
                'auto_disable_bps': self.auto_disable_threshold_bps
            }
        }
    
    def reset_auto_disable(self, operator: str) -> bool:
        """Reset auto-disable state (requires manual intervention)"""
        
        if self.metrics.auto_disable_triggered:
            self.metrics.auto_disable_triggered = False
            self.metrics.manual_intervention_required = False
            self.metrics.parity_status = ParityStatus.WARNING
            self.consecutive_breaches = 0
            
            # Re-enable execution simulator if available
            if self.execution_simulator:
                self.execution_simulator.enable_simulation(operator)
            
            self._save_state()
            
            logger.warning("Parity tracker auto-disable reset", extra={
                'operator': operator,
                'timestamp': datetime.now().isoformat()
            })
            
            return True
        
        return False


# Factory function
def create_parity_tracker(config: Dict[str, Any] = None) -> EnhancedParityTracker:
    """Create configured parity tracker"""
    return EnhancedParityTracker(config)