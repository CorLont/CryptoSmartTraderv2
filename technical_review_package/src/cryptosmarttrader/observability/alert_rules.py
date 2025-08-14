"""
Enterprise Alert Rules for CryptoSmartTrader
Comprehensive alerting system with severity levels and escalation policies.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum

from .metrics_collector import MetricsCollector, AlertSeverity


@dataclass
class AlertRule:
    """Alert rule configuration."""

    name: str
    description: str
    severity: AlertSeverity
    condition: Callable[[Dict[str, Any]], bool]
    threshold_config: Dict[str, Any]
    cooldown_seconds: int = 300  # 5 minutes default cooldown
    escalation_after_minutes: int = 15
    enabled: bool = True


@dataclass
class AlertState:
    """Alert state tracking."""

    rule_name: str
    severity: AlertSeverity
    first_triggered: datetime
    last_triggered: datetime
    trigger_count: int = 0
    acknowledged: bool = False
    escalated: bool = False
    context: Dict[str, Any] = field(default_factory=dict)


class AlertManager:
    """
    Enterprise alert manager with rule evaluation and escalation.

    Features:
    - Configurable alert rules with thresholds
    - Alert state tracking and cooldown periods
    - Severity-based escalation
    - Alert suppression and acknowledgment
    - Integration with metrics collector
    """

    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_states: Dict[str, AlertState] = {}
        self.suppressed_alerts: set = set()

        # Setup default alert rules
        self._setup_default_rules()

        self.metrics_collector.logger.info("AlertManager initialized with comprehensive rules")

    def _setup_default_rules(self):
        """Setup default alert rules for trading system."""

        # High Order Error Rate Alert
        self.add_alert_rule(
            AlertRule(
                name="HighOrderErrorRate",
                description="Order error rate exceeds threshold",
                severity=AlertSeverity.WARNING,
                condition=self._check_order_error_rate,
                threshold_config={
                    "error_rate_threshold": 0.10,  # 10%
                    "min_orders": 10,
                    "time_window_minutes": 15,
                },
                cooldown_seconds=300,
                escalation_after_minutes=15,
            )
        )

        # Drawdown Too High Alert
        self.add_alert_rule(
            AlertRule(
                name="DrawdownTooHigh",
                description="Portfolio drawdown exceeds safe limits",
                severity=AlertSeverity.CRITICAL,
                condition=self._check_drawdown_too_high,
                threshold_config={
                    "warning_threshold": 5.0,  # 5%
                    "critical_threshold": 10.0,  # 10%
                    "emergency_threshold": 15.0,  # 15%
                },
                cooldown_seconds=60,  # Short cooldown for critical alerts
                escalation_after_minutes=5,
            )
        )

        # No Signals Received Alert
        self.add_alert_rule(
            AlertRule(
                name="NoSignals",
                description="No trading signals received within time window",
                severity=AlertSeverity.WARNING,
                condition=self._check_no_signals,
                threshold_config={
                    "max_minutes_without_signal": 30,
                    "critical_minutes_without_signal": 60,
                },
                cooldown_seconds=600,  # 10 minutes cooldown
                escalation_after_minutes=30,
            )
        )

        # High Slippage Alert
        self.add_alert_rule(
            AlertRule(
                name="HighSlippage",
                description="Order slippage exceeds budget consistently",
                severity=AlertSeverity.WARNING,
                condition=self._check_high_slippage,
                threshold_config={
                    "slippage_threshold_bps": 50,  # 50 bps
                    "consecutive_orders": 3,
                    "time_window_minutes": 10,
                },
                cooldown_seconds=300,
                escalation_after_minutes=20,
            )
        )

        # Exchange Connectivity Lost Alert
        self.add_alert_rule(
            AlertRule(
                name="ExchangeConnectivityLost",
                description="Lost connectivity to critical exchange",
                severity=AlertSeverity.CRITICAL,
                condition=self._check_exchange_connectivity,
                threshold_config={
                    "critical_exchanges": ["kraken", "binance"],
                    "max_downtime_seconds": 30,
                },
                cooldown_seconds=60,
                escalation_after_minutes=2,
            )
        )

        # High API Error Rate Alert
        self.add_alert_rule(
            AlertRule(
                name="HighAPIErrorRate",
                description="API error rate exceeds threshold",
                severity=AlertSeverity.WARNING,
                condition=self._check_api_error_rate,
                threshold_config={
                    "error_rate_threshold": 0.15,  # 15%
                    "min_requests": 20,
                    "time_window_minutes": 5,
                },
                cooldown_seconds=180,
                escalation_after_minutes=10,
            )
        )

        # Low Liquidity Alert
        self.add_alert_rule(
            AlertRule(
                name="LowLiquidity",
                description="Market liquidity below safe trading levels",
                severity=AlertSeverity.WARNING,
                condition=self._check_low_liquidity,
                threshold_config={
                    "min_depth_usd": 5000,
                    "min_volume_24h_usd": 100000,
                    "max_spread_bps": 100,
                },
                cooldown_seconds=300,
                escalation_after_minutes=30,
            )
        )

        # System Resource Alert
        self.add_alert_rule(
            AlertRule(
                name="HighResourceUsage",
                description="System resource usage exceeds safe limits",
                severity=AlertSeverity.WARNING,
                condition=self._check_system_resources,
                threshold_config={
                    "max_memory_usage_pct": 85,
                    "max_cpu_usage_pct": 90,
                    "max_disk_usage_pct": 90,
                },
                cooldown_seconds=300,
                escalation_after_minutes=20,
            )
        )

    def add_alert_rule(self, rule: AlertRule):
        """Add new alert rule."""
        self.alert_rules[rule.name] = rule
        self.metrics_collector.logger.info(f"Alert rule added: {rule.name}")

    def remove_alert_rule(self, rule_name: str):
        """Remove alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            self.metrics_collector.logger.info(f"Alert rule removed: {rule_name}")

    def enable_rule(self, rule_name: str):
        """Enable alert rule."""
        if rule_name in self.alert_rules:
            self.alert_rules[rule_name].enabled = True
            self.metrics_collector.logger.info(f"Alert rule enabled: {rule_name}")

    def disable_rule(self, rule_name: str):
        """Disable alert rule."""
        if rule_name in self.alert_rules:
            self.alert_rules[rule_name].enabled = False
            self.metrics_collector.logger.info(f"Alert rule disabled: {rule_name}")

    def suppress_alert(self, rule_name: str, duration_minutes: int = 60):
        """Suppress alert for specified duration."""
        self.suppressed_alerts.add(rule_name)
        self.metrics_collector.logger.info(
            f"Alert suppressed: {rule_name} for {duration_minutes} minutes"
        )

        # TODO: Implement timer to remove suppression after duration

    def acknowledge_alert(self, rule_name: str):
        """Acknowledge active alert."""
        if rule_name in self.alert_states:
            self.alert_states[rule_name].acknowledged = True
            self.metrics_collector.logger.info(f"Alert acknowledged: {rule_name}")

    def evaluate_rules(self, metrics_data: Dict[str, Any]):
        """Evaluate all enabled alert rules against current metrics."""
        current_time = datetime.utcnow()

        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled or rule_name in self.suppressed_alerts:
                continue

            try:
                # Check if rule condition is met
                should_trigger = rule.condition(metrics_data)

                if should_trigger:
                    self._handle_alert_trigger(rule, current_time, metrics_data)
                else:
                    self._handle_alert_clear(rule_name, current_time)

            except Exception as e:
                self.metrics_collector.logger.error(f"Error evaluating rule {rule_name}: {e}")

    def _handle_alert_trigger(
        self, rule: AlertRule, current_time: datetime, context: Dict[str, Any]
    ):
        """Handle alert trigger with cooldown and escalation logic."""
        rule_name = rule.name

        # Check if alert is in cooldown
        if rule_name in self.alert_states:
            state = self.alert_states[rule_name]
            time_since_last = (current_time - state.last_triggered).total_seconds()

            if time_since_last < rule.cooldown_seconds:
                return  # Still in cooldown

            # Update existing alert
            state.last_triggered = current_time
            state.trigger_count += 1
            state.context.update(context)

            # Check for escalation
            if not state.escalated and not state.acknowledged:
                time_since_first = (current_time - state.first_triggered).total_seconds()
                if time_since_first > rule.escalation_after_minutes * 60:
                    self._escalate_alert(state, rule)
        else:
            # New alert
            state = AlertState(
                rule_name=rule_name,
                severity=rule.severity,
                first_triggered=current_time,
                last_triggered=current_time,
                trigger_count=1,
                context=context,
            )
            self.alert_states[rule_name] = state

        # Send alert notification
        self._send_alert_notification(rule, state)

    def _handle_alert_clear(self, rule_name: str, current_time: datetime):
        """Handle alert clearing."""
        if rule_name in self.alert_states:
            state = self.alert_states[rule_name]

            # Log alert cleared
            self.metrics_collector.logger.info(
                f"Alert cleared: {rule_name}",
                extra={
                    "extra_fields": {
                        "alert_name": rule_name,
                        "duration_minutes": (current_time - state.first_triggered).total_seconds()
                        / 60,
                        "trigger_count": state.trigger_count,
                        "metric": "alert_cleared",
                    }
                },
            )

            # Remove from active alerts
            del self.alert_states[rule_name]

    def _escalate_alert(self, state: AlertState, rule: AlertRule):
        """Escalate alert to higher severity."""
        state.escalated = True

        # Increase severity level
        if state.severity == AlertSeverity.WARNING:
            state.severity = AlertSeverity.CRITICAL
        elif state.severity == AlertSeverity.CRITICAL:
            state.severity = AlertSeverity.EMERGENCY

        self.metrics_collector.logger.error(
            f"Alert escalated: {state.rule_name}",
            extra={
                "extra_fields": {
                    "alert_name": state.rule_name,
                    "new_severity": state.severity.value,
                    "trigger_count": state.trigger_count,
                    "metric": "alert_escalated",
                }
            },
        )

    def _send_alert_notification(self, rule: AlertRule, state: AlertState):
        """Send alert notification."""
        self.metrics_collector.logger.warning(
            f"ALERT: {rule.name}",
            extra={
                "extra_fields": {
                    "alert_name": rule.name,
                    "severity": state.severity.value,
                    "description": rule.description,
                    "trigger_count": state.trigger_count,
                    "context": state.context,
                    "escalated": state.escalated,
                    "metric": "alert_triggered",
                }
            },
        )

    # Alert condition functions
    def _check_order_error_rate(self, metrics_data: Dict[str, Any]) -> bool:
        """Check order error rate condition."""
        orders = metrics_data.get("orders", {})
        total_orders = orders.get("total_sent", 0) + orders.get("total_filled", 0)
        total_errors = orders.get("total_errors", 0)

        if total_orders < 10:  # Minimum sample size
            return False

        error_rate = total_errors / total_orders if total_orders > 0 else 0
        return error_rate > 0.10  # 10% threshold

    def _check_drawdown_too_high(self, metrics_data: Dict[str, Any]) -> bool:
        """Check drawdown threshold condition."""
        trading = metrics_data.get("trading", {})
        drawdown_pct = trading.get("drawdown_percent", 0)

        return drawdown_pct > 10.0  # 10% threshold

    def _check_no_signals(self, metrics_data: Dict[str, Any]) -> bool:
        """Check no signals received condition."""
        system = metrics_data.get("system", {})
        minutes_since_signal = system.get("minutes_since_last_signal", 0)

        return minutes_since_signal > 30  # 30 minutes threshold

    def _check_high_slippage(self, metrics_data: Dict[str, Any]) -> bool:
        """Check high slippage condition."""
        # This would need more detailed slippage history
        # For demo, we'll use a simple check
        return False  # Placeholder

    def _check_exchange_connectivity(self, metrics_data: Dict[str, Any]) -> bool:
        """Check exchange connectivity condition."""
        # This would check actual exchange connectivity metrics
        # For demo, we'll simulate
        return False  # Placeholder

    def _check_api_error_rate(self, metrics_data: Dict[str, Any]) -> bool:
        """Check API error rate condition."""
        # This would check actual API metrics
        # For demo, we'll simulate
        return False  # Placeholder

    def _check_low_liquidity(self, metrics_data: Dict[str, Any]) -> bool:
        """Check low liquidity condition."""
        # This would check actual liquidity metrics
        # For demo, we'll simulate
        return False  # Placeholder

    def _check_system_resources(self, metrics_data: Dict[str, Any]) -> bool:
        """Check system resource usage condition."""
        # This would check actual resource metrics
        # For demo, we'll simulate
        return False  # Placeholder

    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts."""
        return [
            {
                "name": state.rule_name,
                "severity": state.severity.value,
                "first_triggered": state.first_triggered.isoformat(),
                "last_triggered": state.last_triggered.isoformat(),
                "trigger_count": state.trigger_count,
                "acknowledged": state.acknowledged,
                "escalated": state.escalated,
                "context": state.context,
            }
            for state in self.alert_states.values()
        ]

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert system status."""
        active_alerts = self.get_active_alerts()

        severity_counts = {"info": 0, "warning": 0, "critical": 0, "emergency": 0}

        for alert in active_alerts:
            severity_counts[alert["severity"]] += 1

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_rules": len(self.alert_rules),
            "enabled_rules": len([r for r in self.alert_rules.values() if r.enabled]),
            "active_alerts": len(active_alerts),
            "suppressed_alerts": len(self.suppressed_alerts),
            "severity_distribution": severity_counts,
            "escalated_alerts": len([a for a in active_alerts if a["escalated"]]),
            "unacknowledged_alerts": len([a for a in active_alerts if not a["acknowledged"]]),
        }


def create_alert_manager(metrics_collector: MetricsCollector) -> AlertManager:
    """Create and configure alert manager."""
    return AlertManager(metrics_collector)
