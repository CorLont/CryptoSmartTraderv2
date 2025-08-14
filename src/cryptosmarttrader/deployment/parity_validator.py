"""
Backtest-Live Parity Validator
Advanced system for tracking performance drift between backtesting and live trading
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import threading
import time

from ..observability.metrics import PrometheusMetrics
from ..risk.central_risk_guard import CentralRiskGuard

logger = logging.getLogger(__name__)


class DriftSeverity(Enum):
    """Classification of performance drift severity"""
    NORMAL = "normal"           # <20 bps
    WARNING = "warning"         # 20-50 bps  
    CRITICAL = "critical"       # 50-100 bps
    EMERGENCY = "emergency"     # >100 bps - Auto halt


@dataclass
class ParityMetrics:
    """Performance parity tracking metrics"""
    timestamp: datetime
    backtest_return: float
    live_return: float
    tracking_error_bps: float
    slippage_bps: float
    fees_bps: float
    timing_impact_bps: float
    latency_impact_bps: float
    drift_severity: DriftSeverity
    component_attribution: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExecutionCosts:
    """Detailed execution cost breakdown"""
    bid_ask_spread: float
    market_impact: float
    timing_costs: float
    fee_costs: float
    slippage_actual: float
    total_cost_bps: float


class ParityValidator:
    """
    Advanced Backtest-Live Parity Validation System
    
    Features:
    - Real-time tracking error calculation
    - Component attribution analysis
    - Automatic emergency halt triggers
    - Performance drift detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics = PrometheusMetrics.get_instance()
        self.risk_guard = CentralRiskGuard()
        
        # Thresholds
        self.warning_threshold_bps = config.get('warning_threshold_bps', 20)
        self.critical_threshold_bps = config.get('critical_threshold_bps', 50) 
        self.emergency_threshold_bps = config.get('emergency_threshold_bps', 100)
        
        # State management
        self._lock = threading.Lock()
        self.parity_history: List[ParityMetrics] = []
        self.emergency_halt_active = False
        self.last_validation_time = None
        
        # Attribution tracking
        self.component_weights = {
            'slippage': 0.4,
            'fees': 0.2,
            'timing': 0.2,
            'latency': 0.1,
            'other': 0.1
        }
        
        logger.info("ParityValidator initialized with thresholds: "
                   f"Warning={self.warning_threshold_bps}bps, "
                   f"Critical={self.critical_threshold_bps}bps, "
                   f"Emergency={self.emergency_threshold_bps}bps")
    
    def validate_parity(self, 
                       backtest_performance: Dict[str, float],
                       live_performance: Dict[str, float],
                       execution_details: Dict[str, Any]) -> ParityMetrics:
        """
        Validate performance parity between backtest and live trading
        
        Args:
            backtest_performance: Expected performance from backtesting
            live_performance: Actual live trading performance  
            execution_details: Detailed execution information
            
        Returns:
            ParityMetrics with drift analysis
        """
        with self._lock:
            try:
                # Calculate core metrics
                backtest_return = backtest_performance.get('total_return', 0.0)
                live_return = live_performance.get('total_return', 0.0)
                
                # Calculate tracking error in basis points
                tracking_error_bps = abs(live_return - backtest_return) * 10000
                
                # Calculate execution costs
                execution_costs = self._calculate_execution_costs(execution_details)
                
                # Component attribution
                attribution = self._calculate_component_attribution(
                    backtest_return, live_return, execution_costs
                )
                
                # Determine severity
                drift_severity = self._classify_drift_severity(tracking_error_bps)
                
                # Create parity metrics
                parity_metrics = ParityMetrics(
                    timestamp=datetime.now(),
                    backtest_return=backtest_return,
                    live_return=live_return,
                    tracking_error_bps=tracking_error_bps,
                    slippage_bps=execution_costs.slippage_actual,
                    fees_bps=execution_costs.fee_costs,
                    timing_impact_bps=execution_costs.timing_costs,
                    latency_impact_bps=execution_costs.market_impact,
                    drift_severity=drift_severity,
                    component_attribution=attribution
                )
                
                # Store metrics
                self.parity_history.append(parity_metrics)
                self._cleanup_old_metrics()
                
                # Update Prometheus metrics
                self._update_prometheus_metrics(parity_metrics)
                
                # Check for emergency conditions
                if drift_severity == DriftSeverity.EMERGENCY:
                    self._trigger_emergency_halt(parity_metrics)
                elif drift_severity == DriftSeverity.CRITICAL:
                    self._trigger_critical_alert(parity_metrics)
                
                self.last_validation_time = datetime.now()
                
                logger.info(f"Parity validation completed: "
                           f"Tracking error={tracking_error_bps:.2f}bps, "
                           f"Severity={drift_severity.value}")
                
                return parity_metrics
                
            except Exception as e:
                logger.error(f"Parity validation failed: {e}")
                raise
    
    def _calculate_execution_costs(self, execution_details: Dict[str, Any]) -> ExecutionCosts:
        """Calculate detailed execution cost breakdown"""
        try:
            # Extract execution data
            fills = execution_details.get('fills', [])
            orders = execution_details.get('orders', [])
            
            if not fills or not orders:
                logger.warning("Missing execution data for cost calculation")
                return ExecutionCosts(0, 0, 0, 0, 0, 0)
            
            # Calculate bid-ask spread impact
            bid_ask_spread = np.mean([f.get('spread_bps', 0) for f in fills])
            
            # Calculate market impact
            market_impact = np.mean([f.get('market_impact_bps', 0) for f in fills])
            
            # Calculate timing costs (delay between signal and execution)
            timing_costs = np.mean([o.get('timing_delay_bps', 0) for o in orders])
            
            # Calculate fee costs
            fee_costs = sum([f.get('fee_bps', 0) for f in fills])
            
            # Calculate actual slippage
            expected_prices = [o.get('expected_price', 0) for o in orders]
            actual_prices = [f.get('actual_price', 0) for f in fills]
            
            if expected_prices and actual_prices:
                slippage_actual = np.mean([
                    abs(actual - expected) / expected * 10000 
                    for actual, expected in zip(actual_prices, expected_prices)
                    if expected > 0
                ])
            else:
                slippage_actual = 0
            
            total_cost_bps = (bid_ask_spread + market_impact + 
                            timing_costs + fee_costs + slippage_actual)
            
            return ExecutionCosts(
                bid_ask_spread=bid_ask_spread,
                market_impact=market_impact,
                timing_costs=timing_costs,
                fee_costs=fee_costs,
                slippage_actual=slippage_actual,
                total_cost_bps=total_cost_bps
            )
            
        except Exception as e:
            logger.error(f"Execution cost calculation failed: {e}")
            return ExecutionCosts(0, 0, 0, 0, 0, 0)
    
    def _calculate_component_attribution(self, 
                                       backtest_return: float,
                                       live_return: float,
                                       execution_costs: ExecutionCosts) -> Dict[str, float]:
        """Calculate component attribution for performance difference"""
        
        total_diff_bps = (live_return - backtest_return) * 10000
        
        attribution = {
            'slippage_impact': -execution_costs.slippage_actual,
            'fee_impact': -execution_costs.fee_costs,
            'timing_impact': -execution_costs.timing_costs,
            'latency_impact': -execution_costs.market_impact,
            'spread_impact': -execution_costs.bid_ask_spread,
            'unexplained': total_diff_bps - (-execution_costs.total_cost_bps)
        }
        
        return attribution
    
    def _classify_drift_severity(self, tracking_error_bps: float) -> DriftSeverity:
        """Classify drift severity based on tracking error"""
        if tracking_error_bps >= self.emergency_threshold_bps:
            return DriftSeverity.EMERGENCY
        elif tracking_error_bps >= self.critical_threshold_bps:
            return DriftSeverity.CRITICAL
        elif tracking_error_bps >= self.warning_threshold_bps:
            return DriftSeverity.WARNING
        else:
            return DriftSeverity.NORMAL
    
    def _trigger_emergency_halt(self, parity_metrics: ParityMetrics):
        """Trigger emergency halt for severe parity drift"""
        if not self.emergency_halt_active:
            self.emergency_halt_active = True
            
            logger.critical(f"EMERGENCY HALT TRIGGERED - Parity drift: "
                           f"{parity_metrics.tracking_error_bps:.2f}bps")
            
            # Trigger risk guard emergency halt
            self.risk_guard.trigger_kill_switch(
                reason=f"Parity drift emergency: {parity_metrics.tracking_error_bps:.2f}bps",
                severity="CRITICAL"
            )
            
            # Record emergency halt metric
            self.metrics.emergency_halt_triggers.inc()
            
            # Send emergency alerts
            self._send_emergency_alert(parity_metrics)
    
    def _trigger_critical_alert(self, parity_metrics: ParityMetrics):
        """Trigger critical alert for significant parity drift"""
        logger.warning(f"CRITICAL PARITY DRIFT - Tracking error: "
                      f"{parity_metrics.tracking_error_bps:.2f}bps")
        
        # Update alert metrics
        self.metrics.parity_drift_alerts.inc()
        
        # Send critical alert
        self._send_critical_alert(parity_metrics)
    
    def _send_emergency_alert(self, parity_metrics: ParityMetrics):
        """Send emergency alert for parity violations"""
        alert_data = {
            'type': 'EMERGENCY_PARITY_VIOLATION',
            'timestamp': parity_metrics.timestamp.isoformat(),
            'tracking_error_bps': parity_metrics.tracking_error_bps,
            'backtest_return': parity_metrics.backtest_return,
            'live_return': parity_metrics.live_return,
            'component_attribution': parity_metrics.component_attribution,
            'action_taken': 'EMERGENCY_HALT_ACTIVATED'
        }
        
        # Log emergency alert
        logger.critical(f"Emergency parity alert: {json.dumps(alert_data, indent=2)}")
    
    def _send_critical_alert(self, parity_metrics: ParityMetrics):
        """Send critical alert for parity drift"""
        alert_data = {
            'type': 'CRITICAL_PARITY_DRIFT',
            'timestamp': parity_metrics.timestamp.isoformat(),
            'tracking_error_bps': parity_metrics.tracking_error_bps,
            'severity': parity_metrics.drift_severity.value,
            'component_attribution': parity_metrics.component_attribution
        }
        
        logger.warning(f"Critical parity alert: {json.dumps(alert_data, indent=2)}")
    
    def _update_prometheus_metrics(self, parity_metrics: ParityMetrics):
        """Update Prometheus metrics with parity data"""
        try:
            # Update tracking error metric
            self.metrics.tracking_error_bps.observe(parity_metrics.tracking_error_bps)
            
            # Update component metrics
            self.metrics.slippage_bps.observe(parity_metrics.slippage_bps)
            
            # Update drift severity counter
            self.metrics.parity_drift_severity.labels(
                severity=parity_metrics.drift_severity.value
            ).inc()
            
            # Update last validation timestamp
            self.metrics.last_parity_validation.set(
                parity_metrics.timestamp.timestamp()
            )
            
        except Exception as e:
            logger.error(f"Failed to update Prometheus metrics: {e}")
    
    def _cleanup_old_metrics(self, max_age_hours: int = 24):
        """Clean up old parity metrics to prevent memory bloat"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        self.parity_history = [
            m for m in self.parity_history 
            if m.timestamp > cutoff_time
        ]
    
    def get_parity_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get parity summary for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_metrics = [
            m for m in self.parity_history 
            if m.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {'status': 'no_data', 'period_hours': hours}
        
        tracking_errors = [m.tracking_error_bps for m in recent_metrics]
        
        return {
            'status': 'active',
            'period_hours': hours,
            'total_validations': len(recent_metrics),
            'avg_tracking_error_bps': np.mean(tracking_errors),
            'max_tracking_error_bps': np.max(tracking_errors),
            'min_tracking_error_bps': np.min(tracking_errors),
            'std_tracking_error_bps': np.std(tracking_errors),
            'emergency_halts': sum(1 for m in recent_metrics 
                                 if m.drift_severity == DriftSeverity.EMERGENCY),
            'critical_alerts': sum(1 for m in recent_metrics 
                                 if m.drift_severity == DriftSeverity.CRITICAL),
            'within_tolerance': sum(1 for m in recent_metrics 
                                  if m.drift_severity in [DriftSeverity.NORMAL, DriftSeverity.WARNING]),
            'last_validation': recent_metrics[-1].timestamp.isoformat() if recent_metrics else None
        }
    
    def reset_emergency_halt(self, reason: str = "Manual reset"):
        """Reset emergency halt status (manual intervention)"""
        with self._lock:
            if self.emergency_halt_active:
                self.emergency_halt_active = False
                logger.info(f"Emergency halt reset: {reason}")
                
                # Log reset event
                self.metrics.emergency_halt_resets.inc()
            else:
                logger.warning("Attempted to reset emergency halt when not active")
    
    def is_healthy(self) -> bool:
        """Check if parity validation system is healthy"""
        if self.emergency_halt_active:
            return False
        
        # Check if validations are recent
        if self.last_validation_time:
            time_since_last = datetime.now() - self.last_validation_time
            if time_since_last > timedelta(minutes=30):
                return False
        
        return True


class ParityReportGenerator:
    """Generate detailed parity reports for analysis"""
    
    def __init__(self, validator: ParityValidator):
        self.validator = validator
    
    def generate_daily_report(self, date: datetime = None) -> Dict[str, Any]:
        """Generate comprehensive daily parity report"""
        if date is None:
            date = datetime.now().date()
        
        start_time = datetime.combine(date, datetime.min.time())
        end_time = start_time + timedelta(days=1)
        
        # Filter metrics for the day
        daily_metrics = [
            m for m in self.validator.parity_history
            if start_time <= m.timestamp < end_time
        ]
        
        if not daily_metrics:
            return {'status': 'no_data', 'date': date.isoformat()}
        
        return self._create_detailed_report(daily_metrics, date)
    
    def _create_detailed_report(self, metrics: List[ParityMetrics], date) -> Dict[str, Any]:
        """Create detailed analysis report"""
        tracking_errors = [m.tracking_error_bps for m in metrics]
        
        # Component attribution analysis
        attributions = {}
        for component in ['slippage_impact', 'fee_impact', 'timing_impact', 'latency_impact']:
            values = [m.component_attribution.get(component, 0) for m in metrics]
            attributions[component] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'total_impact': np.sum(values)
            }
        
        return {
            'date': date.isoformat(),
            'status': 'complete',
            'summary': {
                'total_validations': len(metrics),
                'avg_tracking_error_bps': np.mean(tracking_errors),
                'max_tracking_error_bps': np.max(tracking_errors),
                'std_tracking_error_bps': np.std(tracking_errors),
                'within_20bps_pct': sum(1 for te in tracking_errors if te <= 20) / len(tracking_errors) * 100,
                'emergency_events': sum(1 for m in metrics if m.drift_severity == DriftSeverity.EMERGENCY),
                'critical_events': sum(1 for m in metrics if m.drift_severity == DriftSeverity.CRITICAL)
            },
            'component_attribution': attributions,
            'performance_analysis': {
                'best_tracking_error': min(tracking_errors),
                'worst_tracking_error': max(tracking_errors),
                'consistency_score': 100 - (np.std(tracking_errors) / np.mean(tracking_errors) * 100) if np.mean(tracking_errors) > 0 else 100
            }
        }