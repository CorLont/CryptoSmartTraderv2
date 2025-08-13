#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Self-Healing & Auto-Disabling System
Detects performance degradation and automatically disables modules during anomalies
"""

import logging
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict, deque


class AlertLevel(Enum):
    """Alert severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SystemComponent(Enum):
    """System components that can be disabled"""

    TRADING_ENGINE = "trading_engine"
    ML_PREDICTIONS = "ml_predictions"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    TECHNICAL_ANALYSIS = "technical_analysis"
    WHALE_DETECTION = "whale_detection"
    PORTFOLIO_ALLOCATION = "portfolio_allocation"
    MARKET_SCANNING = "market_scanning"
    CAUSAL_INFERENCE = "causal_inference"
    RL_ALLOCATION = "rl_allocation"
    FEATURE_ENGINEERING = "feature_engineering"
    REGIME_DETECTION = "regime_detection"
    DATA_PIPELINE = "data_pipeline"


class DisableReason(Enum):
    """Reasons for disabling components"""

    BLACK_SWAN_EVENT = "black_swan_event"
    DATA_GAP = "data_gap"
    SECURITY_ALERT = "security_alert"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    API_FAILURE = "api_failure"
    MODEL_DRIFT = "model_drift"
    ANOMALY_DETECTION = "anomaly_detection"
    MANUAL_OVERRIDE = "manual_override"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class PerformanceMetric:
    """Performance metric tracking"""

    component: SystemComponent
    metric_name: str
    value: float
    timestamp: datetime
    threshold: float
    is_anomaly: bool = False


@dataclass
class SystemAlert:
    """System alert structure"""

    alert_id: str
    level: AlertLevel
    component: SystemComponent
    reason: DisableReason
    message: str
    timestamp: datetime
    auto_disabled: bool = False
    recovery_time: Optional[datetime] = None


@dataclass
class ComponentStatus:
    """Component status tracking"""

    component: SystemComponent
    is_enabled: bool
    last_disabled: Optional[datetime]
    disable_reason: Optional[DisableReason]
    performance_score: float
    consecutive_failures: int
    recovery_attempts: int
    auto_recovery_enabled: bool = True


class SelfHealingSystem:
    """Self-healing and auto-disabling system"""

    def __init__(self, config_path: str = "config/self_healing_config.json"):
        self.logger = logging.getLogger(__name__)
        self.config_path = Path(config_path)

        # Performance tracking
        self.performance_metrics: Dict[SystemComponent, deque] = defaultdict(
            lambda: deque(maxlen=100)
        self.component_status: Dict[SystemComponent, ComponentStatus] = {}
        self.system_alerts: deque = deque(maxlen=1000)

        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.last_check_time = datetime.now()

        # Black swan detection
        self.black_swan_indicators = deque(maxlen=50)
        self.market_volatility_history = deque(maxlen=100)

        # Data quality tracking
        self.data_quality_scores = defaultdict(lambda: deque(maxlen=50))
        self.api_response_times = defaultdict(lambda: deque(maxlen=100))

        # Load configuration
        self.config = self._load_config()

        # Initialize component status
        self._initialize_component_status()

        self.logger.info("Self-Healing System initialized")

    def _load_config(self) -> Dict[str, Any]:
        """Load self-healing configuration"""
        default_config = {
            "performance_thresholds": {
                "trading_engine": {"accuracy": 0.6, "sharpe_ratio": 0.5, "max_drawdown": 0.15},
                "ml_predictions": {"accuracy": 0.65, "confidence": 0.7, "drift_threshold": 0.1},
                "sentiment_analysis": {"confidence": 0.6, "response_time": 5.0},
                "technical_analysis": {"signal_quality": 0.7, "false_positive_rate": 0.3},
                "whale_detection": {"detection_rate": 0.8, "false_alarm_rate": 0.2},
                "portfolio_allocation": {"sharpe_ratio": 0.8, "max_drawdown": 0.1},
                "market_scanning": {"coverage": 0.95, "latency": 10.0},
                "causal_inference": {"confidence": 0.7, "effect_significance": 0.05},
                "rl_allocation": {"reward": 0.1, "convergence": 0.01},
                "feature_engineering": {"feature_importance": 0.05, "correlation": 0.9},
                "regime_detection": {"accuracy": 0.75, "regime_stability": 0.8},
                "data_pipeline": {"completeness": 0.98, "latency": 30.0},
            },
            "black_swan_thresholds": {
                "market_volatility": 0.05,  # 5% hourly volatility
                "price_deviation": 0.15,  # 15% deviation from trend
                "volume_spike": 10.0,  # 10x volume increase
                "correlation_breakdown": 0.3,  # Correlation drop below 0.3
            },
            "auto_disable_settings": {
                "consecutive_failure_threshold": 3,
                "performance_degradation_threshold": 0.3,
                "recovery_attempt_limit": 5,
                "auto_recovery_delay": 300,  # 5 minutes
                "critical_alert_immediate_disable": True,
            },
            "data_quality_thresholds": {
                "completeness": 0.95,
                "timeliness": 60,  # seconds
                "api_response_time": 10.0,
                "data_gap_threshold": 300,  # 5 minutes
            },
            "security_monitoring": {
                "unusual_api_activity": True,
                "data_integrity_checks": True,
                "performance_anomaly_detection": True,
            },
        }

        try:
            if self.config_path.exists():
                with open(self.config_path, "r") as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
        except Exception as e:
            self.logger.warning(f"Could not load config, using defaults: {e}")

        return default_config

    def _initialize_component_status(self):
        """Initialize status for all components"""
        for component in SystemComponent:
            self.component_status[component] = ComponentStatus(
                component=component,
                is_enabled=True,
                last_disabled=None,
                disable_reason=None,
                performance_score=1.0,
                consecutive_failures=0,
                recovery_attempts=0,
            )

    def start_monitoring(self):
        """Start continuous monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Self-healing monitoring started")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Self-healing monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                current_time = datetime.now()

                # Check system health every 30 seconds
                if (current_time - self.last_check_time).seconds >= 30:
                    self._perform_health_check()
                    self._check_black_swan_indicators()
                    self._check_data_quality()
                    self._check_security_alerts()
                    self._attempt_auto_recovery()
                    self.last_check_time = current_time

                time.sleep(5)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)

    def report_performance_metric(
        self, component: SystemComponent, metric_name: str, value: float, threshold: float = None
    ):
        """Report a performance metric for monitoring"""
        try:
            # Determine threshold from config if not provided
            if threshold is None:
                component_thresholds = self.config["performance_thresholds"].get(
                    component.value, {}
                )
                threshold = component_thresholds.get(metric_name, 0.5)

            # Create metric
            metric = PerformanceMetric(
                component=component,
                metric_name=metric_name,
                value=value,
                timestamp=datetime.now(),
                threshold=threshold,
                is_anomaly=value < threshold,
            )

            # Store metric
            self.performance_metrics[component].append(metric)

            # Update component performance score
            self._update_component_performance(component)

            # Check for immediate action if critical
            if metric.is_anomaly and value < threshold * 0.5:  # Critical threshold
                self._handle_critical_performance_issue(component, metric)

        except Exception as e:
            self.logger.error(f"Error reporting performance metric: {e}")

    def report_black_swan_indicator(self, indicator_type: str, severity: float, description: str):
        """Report a potential black swan indicator"""
        try:
            indicator = {
                "type": indicator_type,
                "severity": severity,
                "description": description,
                "timestamp": datetime.now(),
            }

            self.black_swan_indicators.append(indicator)

            # Check if this triggers black swan response
            if severity > 0.8 or self._is_black_swan_event():
                self._trigger_black_swan_response(indicator)

        except Exception as e:
            self.logger.error(f"Error reporting black swan indicator: {e}")

    def report_data_gap(self, component: SystemComponent, gap_duration: float, description: str):
        """Report a data gap"""
        try:
            gap_threshold = self.config["data_quality_thresholds"]["data_gap_threshold"]

            if gap_duration > gap_threshold:
                self._create_alert(
                    level=AlertLevel.HIGH
                    if gap_duration > gap_threshold * 2
                    else AlertLevel.MEDIUM,
                    component=component,
                    reason=DisableReason.DATA_GAP,
                    message=f"Data gap detected: {description} (duration: {gap_duration:.1f}s)",
                )

                # Disable component if gap is severe
                if gap_duration > gap_threshold * 3:
                    self.disable_component(component, DisableReason.DATA_GAP, auto_disable=True)

        except Exception as e:
            self.logger.error(f"Error reporting data gap: {e}")

    def report_security_alert(
        self,
        alert_type: str,
        severity: AlertLevel,
        description: str,
        affected_components: List[SystemComponent] = None,
    ):
        """Report a security alert"""
        try:
            for component in affected_components or [SystemComponent.TRADING_ENGINE]:
                self._create_alert(
                    level=severity,
                    component=component,
                    reason=DisableReason.SECURITY_ALERT,
                    message=f"Security alert: {alert_type} - {description}",
                )

                # Auto-disable on high/critical security alerts
                if severity in [AlertLevel.HIGH, AlertLevel.CRITICAL]:
                    self.disable_component(
                        component, DisableReason.SECURITY_ALERT, auto_disable=True
                    )

        except Exception as e:
            self.logger.error(f"Error reporting security alert: {e}")

    def disable_component(
        self,
        component: SystemComponent,
        reason: DisableReason,
        auto_disable: bool = False,
        recovery_delay: float = None,
    ):
        """Disable a system component"""
        try:
            status = self.component_status[component]

            if not status.is_enabled:
                return  # Already disabled

            # Update status
            status.is_enabled = False
            status.last_disabled = datetime.now()
            status.disable_reason = reason
            status.consecutive_failures += 1

            # Create alert
            self._create_alert(
                level=AlertLevel.CRITICAL if auto_disable else AlertLevel.HIGH,
                component=component,
                reason=reason,
                message=f"Component {component.value} disabled: {reason.value}",
                auto_disabled=auto_disable,
            )

            # Schedule recovery if auto-disabled
            if auto_disable and status.auto_recovery_enabled:
                delay = (
                    recovery_delay or self.config["auto_disable_settings"]["auto_recovery_delay"]
                )
                threading.Timer(delay, self._attempt_component_recovery, args=[component]).start()

            self.logger.warning(f"Component {component.value} disabled due to {reason.value}")

        except Exception as e:
            self.logger.error(f"Error disabling component: {e}")

    def enable_component(self, component: SystemComponent, force: bool = False):
        """Enable a system component"""
        try:
            status = self.component_status[component]

            if status.is_enabled and not force:
                return  # Already enabled

            # Reset status
            status.is_enabled = True
            status.last_disabled = None
            status.disable_reason = None
            status.consecutive_failures = 0

            self._create_alert(
                level=AlertLevel.LOW,
                component=component,
                reason=DisableReason.MANUAL_OVERRIDE,
                message=f"Component {component.value} enabled",
            )

            self.logger.info(f"Component {component.value} enabled")

        except Exception as e:
            self.logger.error(f"Error enabling component: {e}")

    def _perform_health_check(self):
        """Perform comprehensive health check"""
        try:
            for component in SystemComponent:
                if not self.component_status[component].is_enabled:
                    continue

                # Check recent performance metrics
                recent_metrics = [
                    m
                    for m in self.performance_metrics[component]
                    if (datetime.now() - m.timestamp).seconds < 300
                ]  # Last 5 minutes

                if not recent_metrics:
                    continue

                # Calculate performance degradation
                degradation = self._calculate_performance_degradation(recent_metrics)

                if (
                    degradation
                    > self.config["auto_disable_settings"]["performance_degradation_threshold"]
                ):
                    self._handle_performance_degradation(component, degradation)

        except Exception as e:
            self.logger.error(f"Error in health check: {e}")

    def _check_black_swan_indicators(self):
        """Check for black swan events"""
        try:
            if len(self.black_swan_indicators) < 3:
                return

            recent_indicators = [
                i
                for i in self.black_swan_indicators
                if (datetime.now() - i["timestamp"]).seconds < 1800
            ]  # Last 30 minutes

            if len(recent_indicators) >= 3:
                avg_severity = np.mean([i["severity"] for i in recent_indicators])

                if avg_severity > 0.7:
                    self._trigger_black_swan_response(
                        {
                            "type": "multiple_indicators",
                            "severity": avg_severity,
                            "description": f"Multiple black swan indicators detected (avg severity: {avg_severity:.2f})",
                            "timestamp": datetime.now(),
                        }
                    )

        except Exception as e:
            self.logger.error(f"Error checking black swan indicators: {e}")

    def _check_data_quality(self):
        """Check data quality across all sources"""
        try:
            for component in SystemComponent:
                if component not in self.data_quality_scores:
                    continue

                recent_scores = list(self.data_quality_scores[component])[-10:]  # Last 10 scores

                if (
                    recent_scores
                    and np.mean(recent_scores)
                    < self.config["data_quality_thresholds"]["completeness"]
                ):
                    self.report_data_gap(
                        component,
                        300,  # Assume 5-minute gap
                        f"Data quality degraded to {np.mean(recent_scores):.2%}",
                    )

        except Exception as e:
            self.logger.error(f"Error checking data quality: {e}")

    def _check_security_alerts(self):
        """Check for security-related issues"""
        try:
            # Check API response times for anomalies
            for component, response_times in self.api_response_times.items():
                if len(response_times) < 10:
                    continue

                recent_times = list(response_times)[-10:]
                avg_time = np.mean(recent_times)

                if avg_time > self.config["data_quality_thresholds"]["api_response_time"] * 3:
                    self.report_security_alert(
                        "unusual_api_response_time",
                        AlertLevel.MEDIUM,
                        f"API response time anomaly: {avg_time:.2f}s (component: {component})",
                    )

        except Exception as e:
            self.logger.error(f"Error checking security alerts: {e}")

    def _attempt_auto_recovery(self):
        """Attempt automatic recovery of disabled components"""
        try:
            recovery_delay = self.config["auto_disable_settings"]["auto_recovery_delay"]

            for component, status in self.component_status.items():
                if status.is_enabled or not status.auto_recovery_enabled:
                    continue

                if (
                    status.last_disabled
                    and (datetime.now() - status.last_disabled).seconds > recovery_delay
                ):
                    if (
                        status.recovery_attempts
                        < self.config["auto_disable_settings"]["recovery_attempt_limit"]
                    ):
                        self._attempt_component_recovery(component)

        except Exception as e:
            self.logger.error(f"Error in auto recovery: {e}")

    def _attempt_component_recovery(self, component: SystemComponent):
        """Attempt to recover a specific component"""
        try:
            status = self.component_status[component]
            status.recovery_attempts += 1

            # Simulate recovery check (in real implementation, this would test the component)
            recovery_success = status.consecutive_failures <= 2  # Simple heuristic

            if recovery_success:
                self.enable_component(component, force=True)

                self._create_alert(
                    level=AlertLevel.LOW,
                    component=component,
                    reason=DisableReason.MANUAL_OVERRIDE,
                    message=f"Component {component.value} auto-recovered after {status.recovery_attempts} attempts",
                )
            else:
                self.logger.warning(
                    f"Auto-recovery failed for {component.value} (attempt {status.recovery_attempts})"
                )

        except Exception as e:
            self.logger.error(f"Error in component recovery: {e}")

    def _update_component_performance(self, component: SystemComponent):
        """Update overall performance score for component"""
        try:
            recent_metrics = [
                m
                for m in self.performance_metrics[component]
                if (datetime.now() - m.timestamp).seconds < 600
            ]  # Last 10 minutes

            if not recent_metrics:
                return

            # Calculate weighted performance score
            performance_ratios = [min(m.value / m.threshold, 2.0) for m in recent_metrics]
            performance_score = np.mean(performance_ratios)

            self.component_status[component].performance_score = performance_score

        except Exception as e:
            self.logger.error(f"Error updating component performance: {e}")

    def _calculate_performance_degradation(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate performance degradation percentage"""
        try:
            if len(metrics) < 2:
                return 0.0

            # Sort by timestamp
            sorted_metrics = sorted(metrics, key=lambda m: m.timestamp)

            # Compare recent vs. earlier performance
            recent = sorted_metrics[-len(sorted_metrics) // 2 :]
            earlier = sorted_metrics[: len(sorted_metrics) // 2]

            recent_avg = np.mean([m.value / m.threshold for m in recent])
            earlier_avg = np.mean([m.value / m.threshold for m in earlier])

            if earlier_avg == 0:
                return 0.0

            degradation = (earlier_avg - recent_avg) / earlier_avg
            return max(0.0, degradation)

        except Exception as e:
            self.logger.error(f"Error calculating performance degradation: {e}")
            return 0.0

    def _handle_performance_degradation(self, component: SystemComponent, degradation: float):
        """Handle detected performance degradation"""
        try:
            status = self.component_status[component]

            if degradation > 0.5:  # Severe degradation
                self.disable_component(
                    component, DisableReason.PERFORMANCE_DEGRADATION, auto_disable=True
                )
            else:
                # Issue warning
                self._create_alert(
                    level=AlertLevel.MEDIUM,
                    component=component,
                    reason=DisableReason.PERFORMANCE_DEGRADATION,
                    message=f"Performance degradation detected: {degradation:.1%}",
                )

        except Exception as e:
            self.logger.error(f"Error handling performance degradation: {e}")

    def _handle_critical_performance_issue(
        self, component: SystemComponent, metric: PerformanceMetric
    ):
        """Handle critical performance issues immediately"""
        try:
            self.disable_component(
                component, DisableReason.PERFORMANCE_DEGRADATION, auto_disable=True
            )

            self._create_alert(
                level=AlertLevel.CRITICAL,
                component=component,
                reason=DisableReason.PERFORMANCE_DEGRADATION,
                message=f"Critical performance issue: {metric.metric_name}={metric.value:.3f} (threshold: {metric.threshold:.3f})",
            )

        except Exception as e:
            self.logger.error(f"Error handling critical performance issue: {e}")

    def _is_black_swan_event(self) -> bool:
        """Determine if recent indicators constitute a black swan event"""
        try:
            if len(self.black_swan_indicators) < 2:
                return False

            recent = [
                i
                for i in self.black_swan_indicators
                if (datetime.now() - i["timestamp"]).seconds < 900
            ]  # Last 15 minutes

            # Multiple high-severity indicators in short time = black swan
            high_severity_count = len([i for i in recent if i["severity"] > 0.8])

            return high_severity_count >= 2

        except Exception as e:
            self.logger.error(f"Error checking black swan event: {e}")
            return False

    def _trigger_black_swan_response(self, indicator: Dict[str, Any]):
        """Trigger emergency black swan response"""
        try:
            self.logger.critical(f"BLACK SWAN EVENT DETECTED: {indicator['description']}")

            # Disable all trading-related components immediately
            critical_components = [
                SystemComponent.TRADING_ENGINE,
                SystemComponent.PORTFOLIO_ALLOCATION,
                SystemComponent.RL_ALLOCATION,
            ]

            for component in critical_components:
                self.disable_component(
                    component,
                    DisableReason.BLACK_SWAN_EVENT,
                    auto_disable=True,
                    recovery_delay=1800,
                )

            # Create critical alert
            self._create_alert(
                level=AlertLevel.CRITICAL,
                component=SystemComponent.TRADING_ENGINE,
                reason=DisableReason.BLACK_SWAN_EVENT,
                message=f"BLACK SWAN EVENT: {indicator['description']} - Trading suspended",
            )

        except Exception as e:
            self.logger.error(f"Error in black swan response: {e}")

    def _create_alert(
        self,
        level: AlertLevel,
        component: SystemComponent,
        reason: DisableReason,
        message: str,
        auto_disabled: bool = False,
    ):
        """Create and store system alert"""
        try:
            alert = SystemAlert(
                alert_id=f"{component.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                level=level,
                component=component,
                reason=reason,
                message=message,
                timestamp=datetime.now(),
                auto_disabled=auto_disabled,
            )

            self.system_alerts.append(alert)

            # Log alert
            log_level = {
                AlertLevel.LOW: logging.INFO,
                AlertLevel.MEDIUM: logging.WARNING,
                AlertLevel.HIGH: logging.ERROR,
                AlertLevel.CRITICAL: logging.CRITICAL,
            }[level]

            self.logger.log(
                log_level, f"ALERT [{level.value.upper()}] {component.value}: {message}"
            )

        except Exception as e:
            self.logger.error(f"Error creating alert: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            enabled_components = [c.value for c, s in self.component_status.items() if s.is_enabled]
            disabled_components = [
                c.value for c, s in self.component_status.items() if not s.is_enabled
            ]

            recent_alerts = [asdict(alert) for alert in list(self.system_alerts)[-10:]]

            performance_summary = {}
            for component, status in self.component_status.items():
                performance_summary[component.value] = {
                    "enabled": status.is_enabled,
                    "performance_score": status.performance_score,
                    "consecutive_failures": status.consecutive_failures,
                    "last_disabled": status.last_disabled.isoformat()
                    if status.last_disabled
                    else None,
                    "disable_reason": status.disable_reason.value
                    if status.disable_reason
                    else None,
                }

            return {
                "monitoring_active": self.monitoring_active,
                "total_components": len(self.component_status),
                "enabled_components": enabled_components,
                "disabled_components": disabled_components,
                "total_alerts": len(self.system_alerts),
                "recent_alerts": recent_alerts,
                "performance_summary": performance_summary,
                "black_swan_indicators": len(self.black_swan_indicators),
                "last_check": self.last_check_time.isoformat(),
            }

        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}

    def get_component_health(self, component: SystemComponent) -> Dict[str, Any]:
        """Get detailed health info for specific component"""
        try:
            status = self.component_status[component]
            recent_metrics = [asdict(m) for m in self.performance_metrics[component]][-20:]

            return {
                "component": component.value,
                "status": asdict(status),
                "recent_metrics": recent_metrics,
                "data_quality_score": np.mean(list(self.data_quality_scores[component])[-10:])
                if self.data_quality_scores[component]
                else 1.0,
                "api_response_time": np.mean(list(self.api_response_times[component])[-10:])
                if self.api_response_times[component]
                else 0.0,
            }

        except Exception as e:
            self.logger.error(f"Error getting component health: {e}")
            return {"error": str(e)}


# Singleton instance
_self_healing_system = None


def get_self_healing_system() -> SelfHealingSystem:
    """Get or create self-healing system singleton"""
    global _self_healing_system
    if _self_healing_system is None:
        _self_healing_system = SelfHealingSystem()
    return _self_healing_system


def initialize_self_healing() -> SelfHealingSystem:
    """Initialize and start self-healing system"""
    system = get_self_healing_system()
    system.start_monitoring()
    return system


def report_performance(
    component: SystemComponent, metric_name: str, value: float, threshold: float = None
):
    """Convenient function to report performance metrics"""
    system = get_self_healing_system()
    system.report_performance_metric(component, metric_name, value, threshold)


def report_black_swan(indicator_type: str, severity: float, description: str):
    """Convenient function to report black swan indicators"""
    system = get_self_healing_system()
    system.report_black_swan_indicator(indicator_type, severity, description)


def report_data_gap(component: SystemComponent, gap_duration: float, description: str):
    """Convenient function to report data gaps"""
    system = get_self_healing_system()
    system.report_data_gap(component, gap_duration, description)


def report_security_alert(
    alert_type: str,
    severity: AlertLevel,
    description: str,
    affected_components: List[SystemComponent] = None,
):
    """Convenient function to report security alerts"""
    system = get_self_healing_system()
    system.report_security_alert(alert_type, severity, description, affected_components)


def disable_component(component: SystemComponent, reason: DisableReason):
    """Convenient function to manually disable component"""
    system = get_self_healing_system()
    system.disable_component(component, reason, auto_disable=False)


def enable_component(component: SystemComponent):
    """Convenient function to manually enable component"""
    system = get_self_healing_system()
    system.enable_component(component, force=True)


def get_system_health() -> Dict[str, Any]:
    """Convenient function to get system health status"""
    system = get_self_healing_system()
    return system.get_system_status()
