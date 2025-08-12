"""
Performance Monitor

Real-time performance monitoring with degradation detection
and automated alerting for faster MTTR on performance issues.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import deque

logger = logging.getLogger(__name__)

class AlertType(Enum):
    """Performance alert types"""
    SLIPPAGE_DRIFT = "slippage_drift"
    ALPHA_DECAY = "alpha_decay"
    FEE_SPIKE = "fee_spike"
    EXECUTION_QUALITY_DROP = "execution_quality_drop"
    REGIME_PERFORMANCE_DROP = "regime_performance_drop"
    CORRELATION_BREAK = "correlation_break"
    DRAWDOWN_WARNING = "drawdown_warning"
    VOLUME_ANOMALY = "volume_anomaly"
    LATENCY_SPIKE = "latency_spike"
    ERROR_RATE_INCREASE = "error_rate_increase"


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PerformanceAlert:
    """Performance monitoring alert"""
    alert_id: str
    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    
    # Alert details
    message: str
    current_value: float
    threshold_value: float
    deviation_pct: float
    
    # Context
    affected_pairs: List[str] = field(default_factory=list)
    affected_regimes: List[str] = field(default_factory=list)
    time_window: str = "1H"
    
    # Action recommendations
    recommendations: List[str] = field(default_factory=list)
    
    # Metadata
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    @property
    def age_minutes(self) -> float:
        """Alert age in minutes"""
        return (datetime.now() - self.timestamp).total_seconds() / 60
    
    @property
    def mttr_minutes(self) -> float:
        """Mean Time To Resolution in minutes"""
        if self.resolution_time:
            return (self.resolution_time - self.timestamp).total_seconds() / 60
        return 0.0


class PerformanceMonitor:
    """
    Real-time performance monitoring system with degradation detection
    """
    
    def __init__(self, 
                 monitoring_frequency_seconds: int = 60,
                 alert_cooldown_minutes: int = 15):
        
        self.monitoring_frequency = monitoring_frequency_seconds
        self.alert_cooldown = timedelta(minutes=alert_cooldown_minutes)
        
        # Monitoring buffers
        self.performance_buffer = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.slippage_buffer = deque(maxlen=1440)
        self.fee_buffer = deque(maxlen=1440)
        self.execution_buffer = deque(maxlen=1440)
        
        # Alert management
        self.active_alerts = {}
        self.alert_history = []
        self.alert_callbacks = []
        
        # Thresholds
        self.thresholds = {
            "slippage_drift_pct": 25,      # 25% increase in slippage
            "alpha_decay_pct": 20,         # 20% drop in alpha generation
            "fee_spike_pct": 50,           # 50% increase in fees
            "execution_quality_drop": 0.15, # 15 point drop in quality score
            "drawdown_warning_pct": 5,     # 5% drawdown warning
            "latency_spike_ms": 500,       # 500ms latency spike
            "error_rate_pct": 5            # 5% error rate
        }
        
        # Baseline metrics for comparison
        self.baselines = {}
        self.baseline_window_hours = 24
        
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback function for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def update_performance_metrics(self, metrics: Dict[str, Any]) -> List[PerformanceAlert]:
        """Update performance metrics and check for alerts"""
        try:
            timestamp = datetime.now()
            
            # Store metrics in buffers
            self.performance_buffer.append((timestamp, metrics))
            
            # Extract specific metrics
            if 'slippage_bps' in metrics:
                self.slippage_buffer.append((timestamp, metrics['slippage_bps']))
            
            if 'fees_bps' in metrics:
                self.fee_buffer.append((timestamp, metrics['fees_bps']))
            
            if 'execution_quality' in metrics:
                self.execution_buffer.append((timestamp, metrics['execution_quality']))
            
            # Update baselines
            self._update_baselines()
            
            # Check for alerts
            new_alerts = self._check_performance_alerts(metrics)
            
            # Process alerts
            for alert in new_alerts:
                self._process_alert(alert)
            
            return new_alerts
            
        except Exception as e:
            logger.error(f"Performance metrics update failed: {e}")
            return []
    
    def _update_baselines(self):
        """Update baseline metrics for comparison"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.baseline_window_hours)
            
            # Update slippage baseline
            if self.slippage_buffer:
                recent_slippage = [value for timestamp, value in self.slippage_buffer 
                                 if timestamp >= cutoff_time]
                if recent_slippage:
                    self.baselines['slippage_bps'] = np.mean(recent_slippage)
            
            # Update fee baseline
            if self.fee_buffer:
                recent_fees = [value for timestamp, value in self.fee_buffer 
                             if timestamp >= cutoff_time]
                if recent_fees:
                    self.baselines['fees_bps'] = np.mean(recent_fees)
            
            # Update execution quality baseline
            if self.execution_buffer:
                recent_execution = [value for timestamp, value in self.execution_buffer 
                                  if timestamp >= cutoff_time]
                if recent_execution:
                    self.baselines['execution_quality'] = np.mean(recent_execution)
            
            # Update performance baseline
            if self.performance_buffer:
                cutoff_time = datetime.now() - timedelta(hours=self.baseline_window_hours)
                recent_performance = [(timestamp, metrics) for timestamp, metrics in self.performance_buffer 
                                    if timestamp >= cutoff_time]
                
                if recent_performance:
                    # Calculate baseline alpha
                    alpha_values = [metrics.get('alpha_bps', 0) for _, metrics in recent_performance]
                    if alpha_values:
                        self.baselines['alpha_bps'] = np.mean(alpha_values)
                    
                    # Calculate baseline returns
                    return_values = [metrics.get('return_bps', 0) for _, metrics in recent_performance]
                    if return_values:
                        self.baselines['return_bps'] = np.mean(return_values)
            
        except Exception as e:
            logger.error(f"Baseline update failed: {e}")
    
    def _check_performance_alerts(self, current_metrics: Dict[str, Any]) -> List[PerformanceAlert]:
        """Check for performance alerts based on current metrics"""
        alerts = []
        
        try:
            # Slippage drift alert
            if 'slippage_bps' in current_metrics and 'slippage_bps' in self.baselines:
                current_slippage = current_metrics['slippage_bps']
                baseline_slippage = self.baselines['slippage_bps']
                
                if baseline_slippage > 0:
                    deviation_pct = (current_slippage - baseline_slippage) / baseline_slippage * 100
                    
                    if deviation_pct > self.thresholds['slippage_drift_pct']:
                        alert = self._create_slippage_alert(current_slippage, baseline_slippage, deviation_pct)
                        if self._should_create_alert(alert.alert_type):
                            alerts.append(alert)
            
            # Alpha decay alert
            if 'alpha_bps' in current_metrics and 'alpha_bps' in self.baselines:
                current_alpha = current_metrics['alpha_bps']
                baseline_alpha = self.baselines['alpha_bps']
                
                if baseline_alpha > 0:
                    decay_pct = (baseline_alpha - current_alpha) / baseline_alpha * 100
                    
                    if decay_pct > self.thresholds['alpha_decay_pct']:
                        alert = self._create_alpha_decay_alert(current_alpha, baseline_alpha, decay_pct)
                        if self._should_create_alert(alert.alert_type):
                            alerts.append(alert)
            
            # Fee spike alert
            if 'fees_bps' in current_metrics and 'fees_bps' in self.baselines:
                current_fees = current_metrics['fees_bps']
                baseline_fees = self.baselines['fees_bps']
                
                if baseline_fees > 0:
                    spike_pct = (current_fees - baseline_fees) / baseline_fees * 100
                    
                    if spike_pct > self.thresholds['fee_spike_pct']:
                        alert = self._create_fee_spike_alert(current_fees, baseline_fees, spike_pct)
                        if self._should_create_alert(alert.alert_type):
                            alerts.append(alert)
            
            # Execution quality drop alert
            if 'execution_quality' in current_metrics and 'execution_quality' in self.baselines:
                current_quality = current_metrics['execution_quality']
                baseline_quality = self.baselines['execution_quality']
                
                quality_drop = baseline_quality - current_quality
                
                if quality_drop > self.thresholds['execution_quality_drop']:
                    alert = self._create_execution_quality_alert(current_quality, baseline_quality, quality_drop)
                    if self._should_create_alert(alert.alert_type):
                        alerts.append(alert)
            
            # Drawdown warning
            if 'drawdown_pct' in current_metrics:
                current_drawdown = abs(current_metrics['drawdown_pct'])
                
                if current_drawdown > self.thresholds['drawdown_warning_pct']:
                    alert = self._create_drawdown_alert(current_drawdown)
                    if self._should_create_alert(alert.alert_type):
                        alerts.append(alert)
            
            # Latency spike alert
            if 'latency_ms' in current_metrics:
                current_latency = current_metrics['latency_ms']
                
                if current_latency > self.thresholds['latency_spike_ms']:
                    alert = self._create_latency_alert(current_latency)
                    if self._should_create_alert(alert.alert_type):
                        alerts.append(alert)
            
            # Error rate alert
            if 'error_rate_pct' in current_metrics:
                current_error_rate = current_metrics['error_rate_pct']
                
                if current_error_rate > self.thresholds['error_rate_pct']:
                    alert = self._create_error_rate_alert(current_error_rate)
                    if self._should_create_alert(alert.alert_type):
                        alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Performance alert check failed: {e}")
            return []
    
    def _should_create_alert(self, alert_type: AlertType) -> bool:
        """Check if alert should be created (cooldown logic)"""
        if alert_type not in self.active_alerts:
            return True
        
        last_alert_time = self.active_alerts[alert_type].timestamp
        return datetime.now() - last_alert_time > self.alert_cooldown
    
    def _create_slippage_alert(self, current: float, baseline: float, deviation: float) -> PerformanceAlert:
        """Create slippage drift alert"""
        severity = AlertSeverity.HIGH if deviation > 50 else \
                  AlertSeverity.MEDIUM if deviation > 35 else AlertSeverity.LOW
        
        return PerformanceAlert(
            alert_id=f"slippage_drift_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            alert_type=AlertType.SLIPPAGE_DRIFT,
            severity=severity,
            message=f"Slippage increased by {deviation:.1f}% from {baseline:.1f} to {current:.1f} bps",
            current_value=current,
            threshold_value=baseline,
            deviation_pct=deviation,
            time_window="1H",
            recommendations=[
                "Check order routing and liquidity providers",
                "Review execution algorithms and timing",
                "Analyze market microstructure changes",
                "Consider reducing order sizes temporarily"
            ]
        )
    
    def _create_alpha_decay_alert(self, current: float, baseline: float, decay: float) -> PerformanceAlert:
        """Create alpha decay alert"""
        severity = AlertSeverity.CRITICAL if decay > 40 else \
                  AlertSeverity.HIGH if decay > 30 else AlertSeverity.MEDIUM
        
        return PerformanceAlert(
            alert_id=f"alpha_decay_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            alert_type=AlertType.ALPHA_DECAY,
            severity=severity,
            message=f"Alpha generation declined by {decay:.1f}% from {baseline:.1f} to {current:.1f} bps",
            current_value=current,
            threshold_value=baseline,
            deviation_pct=decay,
            time_window="1H",
            recommendations=[
                "Review model performance and feature drift",
                "Check for regime changes in market conditions",
                "Analyze signal quality and confidence scores",
                "Consider model retraining or parameter updates"
            ]
        )
    
    def _create_fee_spike_alert(self, current: float, baseline: float, spike: float) -> PerformanceAlert:
        """Create fee spike alert"""
        severity = AlertSeverity.HIGH if spike > 75 else AlertSeverity.MEDIUM
        
        return PerformanceAlert(
            alert_id=f"fee_spike_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            alert_type=AlertType.FEE_SPIKE,
            severity=severity,
            message=f"Trading fees spiked by {spike:.1f}% from {baseline:.1f} to {current:.1f} bps",
            current_value=current,
            threshold_value=baseline,
            deviation_pct=spike,
            time_window="1H",
            recommendations=[
                "Check maker/taker ratio optimization",
                "Review fee tier utilization",
                "Analyze order type distribution",
                "Consider post-only orders during high fee periods"
            ]
        )
    
    def _create_execution_quality_alert(self, current: float, baseline: float, drop: float) -> PerformanceAlert:
        """Create execution quality drop alert"""
        severity = AlertSeverity.HIGH if drop > 0.25 else AlertSeverity.MEDIUM
        
        return PerformanceAlert(
            alert_id=f"execution_quality_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            alert_type=AlertType.EXECUTION_QUALITY_DROP,
            severity=severity,
            message=f"Execution quality dropped by {drop:.2f} points from {baseline:.2f} to {current:.2f}",
            current_value=current,
            threshold_value=baseline,
            deviation_pct=(drop / baseline * 100) if baseline > 0 else 0,
            time_window="1H",
            recommendations=[
                "Review execution algorithms and parameters",
                "Check market liquidity conditions",
                "Analyze order queue dynamics",
                "Consider switching execution venues"
            ]
        )
    
    def _create_drawdown_alert(self, current_drawdown: float) -> PerformanceAlert:
        """Create drawdown warning alert"""
        severity = AlertSeverity.CRITICAL if current_drawdown > 10 else \
                  AlertSeverity.HIGH if current_drawdown > 7 else AlertSeverity.MEDIUM
        
        return PerformanceAlert(
            alert_id=f"drawdown_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            alert_type=AlertType.DRAWDOWN_WARNING,
            severity=severity,
            message=f"Portfolio drawdown reached {current_drawdown:.1f}%",
            current_value=current_drawdown,
            threshold_value=self.thresholds['drawdown_warning_pct'],
            deviation_pct=((current_drawdown - self.thresholds['drawdown_warning_pct']) / 
                          self.thresholds['drawdown_warning_pct'] * 100),
            time_window="Current",
            recommendations=[
                "Activate risk reduction protocols",
                "Review position sizing and correlation",
                "Consider temporary trading halt",
                "Analyze drawdown attribution by strategy"
            ]
        )
    
    def _create_latency_alert(self, current_latency: float) -> PerformanceAlert:
        """Create latency spike alert"""
        severity = AlertSeverity.HIGH if current_latency > 1000 else AlertSeverity.MEDIUM
        
        return PerformanceAlert(
            alert_id=f"latency_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            alert_type=AlertType.LATENCY_SPIKE,
            severity=severity,
            message=f"Execution latency spiked to {current_latency:.0f}ms",
            current_value=current_latency,
            threshold_value=self.thresholds['latency_spike_ms'],
            deviation_pct=((current_latency - self.thresholds['latency_spike_ms']) / 
                          self.thresholds['latency_spike_ms'] * 100),
            time_window="Current",
            recommendations=[
                "Check network connectivity and routing",
                "Review exchange API status",
                "Consider switching to backup execution venues",
                "Monitor system resource utilization"
            ]
        )
    
    def _create_error_rate_alert(self, current_error_rate: float) -> PerformanceAlert:
        """Create error rate alert"""
        severity = AlertSeverity.HIGH if current_error_rate > 10 else AlertSeverity.MEDIUM
        
        return PerformanceAlert(
            alert_id=f"error_rate_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            alert_type=AlertType.ERROR_RATE_INCREASE,
            severity=severity,
            message=f"Error rate increased to {current_error_rate:.1f}%",
            current_value=current_error_rate,
            threshold_value=self.thresholds['error_rate_pct'],
            deviation_pct=((current_error_rate - self.thresholds['error_rate_pct']) / 
                          self.thresholds['error_rate_pct'] * 100),
            time_window="Current",
            recommendations=[
                "Review system logs and error patterns",
                "Check API rate limits and quotas",
                "Monitor system health and resources",
                "Consider fallback execution strategies"
            ]
        )
    
    def _process_alert(self, alert: PerformanceAlert):
        """Process and store alert"""
        try:
            # Store alert
            self.active_alerts[alert.alert_type] = alert
            self.alert_history.append(alert)
            
            # Log alert
            logger.warning(f"Performance Alert [{alert.severity.value.upper()}]: {alert.message}")
            
            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")
            
        except Exception as e:
            logger.error(f"Alert processing failed: {e}")
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert"""
        try:
            for alert in self.alert_history:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    logger.info(f"Alert {alert_id} acknowledged by {user}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Alert acknowledgment failed: {e}")
            return False
    
    def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Resolve an alert"""
        try:
            for alert in self.alert_history:
                if alert.alert_id == alert_id:
                    alert.resolved = True
                    alert.resolution_time = datetime.now()
                    logger.info(f"Alert {alert_id} resolved by {user} (MTTR: {alert.mttr_minutes:.1f} min)")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Alert resolution failed: {e}")
            return False
    
    def get_active_alerts(self, severity_filter: AlertSeverity = None) -> List[PerformanceAlert]:
        """Get currently active alerts"""
        active = [alert for alert in self.active_alerts.values() if not alert.resolved]
        
        if severity_filter:
            active = [alert for alert in active if alert.severity == severity_filter]
        
        return active
    
    def get_alert_metrics(self, days_back: int = 7) -> Dict[str, Any]:
        """Get alert metrics and MTTR statistics"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            period_alerts = [alert for alert in self.alert_history if alert.timestamp >= cutoff_time]
            
            if not period_alerts:
                return {"status": "no_alerts"}
            
            # Calculate MTTR
            resolved_alerts = [alert for alert in period_alerts if alert.resolved]
            mttr_minutes = np.mean([alert.mttr_minutes for alert in resolved_alerts]) if resolved_alerts else 0
            
            # Alert frequency by type
            alert_frequency = {}
            for alert in period_alerts:
                alert_type = alert.alert_type.value
                alert_frequency[alert_type] = alert_frequency.get(alert_type, 0) + 1
            
            # Severity distribution
            severity_dist = {}
            for alert in period_alerts:
                severity = alert.severity.value
                severity_dist[severity] = severity_dist.get(severity, 0) + 1
            
            return {
                "analysis_period_days": days_back,
                "total_alerts": len(period_alerts),
                "resolved_alerts": len(resolved_alerts),
                "unresolved_alerts": len(period_alerts) - len(resolved_alerts),
                "mttr_minutes": mttr_minutes,
                "alert_frequency": alert_frequency,
                "severity_distribution": severity_dist,
                "top_alert_types": sorted(alert_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
            }
            
        except Exception as e:
            logger.error(f"Alert metrics calculation failed: {e}")
            return {"status": "error", "error": str(e)}