"""
Alert Manager System

24/7 alerting system with Slack/Email notifications, threshold management,
and silence scheduling for comprehensive system monitoring.
"""

import asyncio
import smtplib
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dt_time
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import json
from pathlib import Path
import threading
import schedule

try:
    import requests

    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

# Telegram will be primary alerting channel
try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False

    class WebClient:
        def __init__(self, token):
            pass

        def chat_postMessage(self, **kwargs):
            pass


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels"""

    TELEGRAM = "telegram"  # Primary channel
    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"
    SMS = "sms"  # Future implementation


@dataclass
class AlertRule:
    """Alert rule configuration"""

    name: str
    description: str
    metric_name: str
    threshold_value: float
    comparison: str  # "gt", "lt", "gte", "lte", "eq", "ne"
    severity: AlertSeverity
    channels: List[AlertChannel]

    # Timing
    evaluation_interval_seconds: int = 60
    for_duration_seconds: int = 300  # Must be true for 5 minutes

    # Suppression
    suppress_duration_seconds: int = 3600  # Don't repeat for 1 hour
    max_alerts_per_hour: int = 5

    # Labels and annotations
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    # State tracking
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0
    currently_firing: bool = False
    first_trigger_time: Optional[datetime] = None


@dataclass
class SilenceSchedule:
    """Silence schedule configuration"""

    name: str
    description: str
    start_time: dt_time  # Daily start time
    end_time: dt_time  # Daily end time
    days_of_week: List[int]  # 0=Monday, 6=Sunday
    severity_filter: Optional[List[AlertSeverity]] = None
    metric_filter: Optional[List[str]] = None
    active: bool = True


@dataclass
class Alert:
    """Alert instance"""

    timestamp: datetime
    rule_name: str
    severity: AlertSeverity
    title: str
    message: str
    metric_name: str
    current_value: float
    threshold_value: float

    # Context
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    # Delivery
    channels_sent: List[AlertChannel] = field(default_factory=list)
    delivery_attempts: int = 0
    successful_delivery: bool = False


class AlertManager:
    """
    Comprehensive 24/7 alert management system
    """

    def __init__(
        self,
        telegram_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        slack_token: Optional[str] = None,
        slack_channel: Optional[str] = None,
        smtp_server: Optional[str] = None,
        smtp_port: int = 587,
        email_username: Optional[str] = None,
        email_password: Optional[str] = None,
        email_recipients: Optional[List[str]] = None,
    ):
        # Configuration
        self.telegram_token = telegram_token
        self.telegram_chat_id = telegram_chat_id
        self.slack_token = slack_token
        self.slack_channel = slack_channel
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.email_username = email_username
        self.email_password = email_password
        self.email_recipients = email_recipients or []

        # Initialize clients
        self.telegram_api_url = None
        if self.telegram_token and TELEGRAM_AVAILABLE:
            self.telegram_api_url = f"https://api.telegram.org/bot{self.telegram_token}"

        self.slack_client = None
        if self.slack_token and SLACK_AVAILABLE:
            self.slack_client = WebClient(token=self.slack_token)

        # Alert rules and state
        self.alert_rules: Dict[str, AlertRule] = {}
        self.silence_schedules: Dict[str, SilenceSchedule] = {}
        self.alert_history: List[Alert] = []
        self.active_alerts: Dict[str, Alert] = {}

        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.metrics_source: Optional[Callable[[str], float]] = None

        # Statistics
        self.alert_stats = {
            "total_alerts": 0,
            "alerts_by_severity": {sev.value: 0 for sev in AlertSeverity},
            "alerts_by_channel": {ch.value: 0 for ch in AlertChannel},
            "delivery_failures": 0,
            "silenced_alerts": 0,
        }

        # Setup default alert rules
        self._setup_default_alert_rules()

        # Setup default silence schedules
        self._setup_default_silence_schedules()

    def _setup_default_alert_rules(self):
        """Setup default alert rules for trading system"""

        # High error rate
        self.add_alert_rule(
            AlertRule(
                name="high_error_rate",
                description="High error rate detected",
                metric_name="trading_errors_total",
                threshold_value=10.0,
                comparison="gte",
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
                evaluation_interval_seconds=60,
                for_duration_seconds=180,  # 3 minutes
                annotations={
                    "summary": "High error rate in trading system",
                    "description": "Error rate has exceeded 10 errors in the last 3 minutes",
                },
            )
        )

        # High API latency
        self.add_alert_rule(
            AlertRule(
                name="high_api_latency",
                description="High API latency detected",
                metric_name="api_request_duration_seconds",
                threshold_value=2.0,
                comparison="gt",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.SLACK],
                evaluation_interval_seconds=30,
                for_duration_seconds=300,  # 5 minutes
                annotations={
                    "summary": "High API latency detected",
                    "description": "API requests taking longer than 2 seconds",
                },
            )
        )

        # High slippage
        self.add_alert_rule(
            AlertRule(
                name="high_slippage",
                description="High trading slippage detected",
                metric_name="trading_slippage_bps",
                threshold_value=50.0,
                comparison="gt",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.SLACK],
                evaluation_interval_seconds=60,
                for_duration_seconds=300,
                annotations={
                    "summary": "High trading slippage",
                    "description": "Trading slippage exceeds 50 basis points",
                },
            )
        )

        # Large daily loss
        self.add_alert_rule(
            AlertRule(
                name="large_daily_loss",
                description="Large daily loss detected",
                metric_name="portfolio_pnl_usd",
                threshold_value=-5000.0,
                comparison="lt",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
                evaluation_interval_seconds=300,  # 5 minutes
                for_duration_seconds=0,  # Immediate
                max_alerts_per_hour=3,
                annotations={
                    "summary": "Large daily loss alert",
                    "description": "Daily P&L loss exceeds $5,000",
                },
            )
        )

        # High drawdown
        self.add_alert_rule(
            AlertRule(
                name="high_drawdown",
                description="High portfolio drawdown",
                metric_name="portfolio_drawdown_percent",
                threshold_value=8.0,
                comparison="gt",
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
                evaluation_interval_seconds=300,
                for_duration_seconds=600,  # 10 minutes
                annotations={
                    "summary": "High portfolio drawdown",
                    "description": "Portfolio drawdown exceeds 8%",
                },
            )
        )

        # System health critical
        self.add_alert_rule(
            AlertRule(
                name="system_health_critical",
                description="System health critically low",
                metric_name="system_health_score",
                threshold_value=30.0,
                comparison="lt",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
                evaluation_interval_seconds=30,
                for_duration_seconds=180,
                annotations={
                    "summary": "System health critical",
                    "description": "System health score below 30%",
                },
            )
        )

        # Circuit breaker triggered
        self.add_alert_rule(
            AlertRule(
                name="circuit_breaker_triggered",
                description="Circuit breaker activated",
                metric_name="circuit_breaker_triggers_total",
                threshold_value=0.0,
                comparison="gt",
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
                evaluation_interval_seconds=30,
                for_duration_seconds=0,  # Immediate
                annotations={
                    "summary": "Circuit breaker activated",
                    "description": "Trading circuit breaker has been triggered",
                },
            )
        )

        # Kill switch activation
        self.add_alert_rule(
            AlertRule(
                name="kill_switch_activation",
                description="Kill switch activated",
                metric_name="kill_switch_activations_total",
                threshold_value=0.0,
                comparison="gt",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.SLACK, AlertChannel.EMAIL],
                evaluation_interval_seconds=10,
                for_duration_seconds=0,  # Immediate
                max_alerts_per_hour=10,  # Important enough to repeat
                annotations={
                    "summary": "🚨 KILL SWITCH ACTIVATED 🚨",
                    "description": "Emergency kill switch has been activated - immediate action required",
                },
            )
        )

        # Data staleness
        self.add_alert_rule(
            AlertRule(
                name="stale_data",
                description="Market data is stale",
                metric_name="data_freshness_seconds",
                threshold_value=300.0,  # 5 minutes
                comparison="gt",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.SLACK],
                evaluation_interval_seconds=60,
                for_duration_seconds=180,
                annotations={
                    "summary": "Stale market data",
                    "description": "Market data is older than 5 minutes",
                },
            )
        )

        # High memory usage
        self.add_alert_rule(
            AlertRule(
                name="high_memory_usage",
                description="High system memory usage",
                metric_name="system_memory_usage_percent",
                threshold_value=85.0,
                comparison="gt",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.SLACK],
                evaluation_interval_seconds=120,
                for_duration_seconds=600,  # 10 minutes
                annotations={
                    "summary": "High memory usage",
                    "description": "System memory usage exceeds 85%",
                },
            )
        )

    def _setup_default_silence_schedules(self):
        """Setup default silence schedules"""

        # Night silence (reduce non-critical alerts)
        self.add_silence_schedule(
            SilenceSchedule(
                name="night_silence",
                description="Reduce alerts during night hours",
                start_time=dt_time(22, 0),  # 10 PM
                end_time=dt_time(6, 0),  # 6 AM
                days_of_week=[0, 1, 2, 3, 4, 5, 6],  # All days
                severity_filter=[AlertSeverity.INFO, AlertSeverity.WARNING],
                active=True,
            )
        )

        # Weekend silence (reduce non-critical alerts)
        self.add_silence_schedule(
            SilenceSchedule(
                name="weekend_silence",
                description="Reduce alerts during weekends",
                start_time=dt_time(0, 0),  # Midnight
                end_time=dt_time(23, 59),  # End of day
                days_of_week=[5, 6],  # Saturday, Sunday
                severity_filter=[AlertSeverity.INFO, AlertSeverity.WARNING],
                active=True,
            )
        )

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def add_silence_schedule(self, schedule: SilenceSchedule):
        """Add a silence schedule"""
        self.silence_schedules[schedule.name] = schedule
        logger.info(f"Added silence schedule: {schedule.name}")

    def set_metrics_source(self, metrics_source: Callable[[str], float]):
        """Set function to retrieve metric values"""
        self.metrics_source = metrics_source

    def start_monitoring(self):
        """Start alert monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Alert monitoring started")

    def stop_monitoring(self):
        """Stop alert monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        logger.info("Alert monitoring stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Evaluate all alert rules
                for rule_name, rule in self.alert_rules.items():
                    self._evaluate_rule(rule)

                # Clean up old alerts
                self._cleanup_old_alerts()

                # Sleep for minimum evaluation interval
                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"Alert monitoring loop error: {e}")
                time.sleep(60)

    def _evaluate_rule(self, rule: AlertRule):
        """Evaluate a single alert rule"""
        try:
            # Skip if no metrics source
            if not self.metrics_source:
                return

            # Get current metric value
            current_value = self.metrics_source(rule.metric_name)
            if current_value is None:
                return

            # Check if rule condition is met
            condition_met = self._evaluate_condition(
                current_value, rule.threshold_value, rule.comparison
            )

            now = datetime.now()

            if condition_met:
                # Update rule state
                if not rule.currently_firing:
                    rule.first_trigger_time = now

                # Check if condition has been true for required duration
                if (
                    rule.first_trigger_time
                    and (now - rule.first_trigger_time).total_seconds() >= rule.for_duration_seconds
                ):
                    # Check if we should fire the alert
                    if self._should_fire_alert(rule, now):
                        alert = self._create_alert(rule, current_value, now)
                        self._fire_alert(alert)

                        # Update rule state
                        rule.last_triggered = now
                        rule.trigger_count += 1
                        rule.currently_firing = True
            else:
                # Condition no longer met
                if rule.currently_firing:
                    rule.currently_firing = False
                    rule.first_trigger_time = None

                    # Send resolution alert for critical/error alerts
                    if rule.severity in [AlertSeverity.CRITICAL, AlertSeverity.ERROR]:
                        resolution_alert = self._create_resolution_alert(rule, current_value, now)
                        self._fire_alert(resolution_alert)

        except Exception as e:
            logger.error(f"Failed to evaluate rule {rule.name}: {e}")

    def _evaluate_condition(self, current_value: float, threshold: float, comparison: str) -> bool:
        """Evaluate if condition is met"""
        if comparison == "gt":
            return current_value > threshold
        elif comparison == "gte":
            return current_value >= threshold
        elif comparison == "lt":
            return current_value < threshold
        elif comparison == "lte":
            return current_value <= threshold
        elif comparison == "eq":
            return abs(current_value - threshold) < 1e-9
        elif comparison == "ne":
            return abs(current_value - threshold) >= 1e-9
        else:
            return False

    def _should_fire_alert(self, rule: AlertRule, now: datetime) -> bool:
        """Check if alert should be fired based on suppression rules"""

        # Check silence schedules
        if self._is_silenced(rule, now):
            self.alert_stats["silenced_alerts"] += 1
            return False

        # Check suppression duration
        if (
            rule.last_triggered
            and (now - rule.last_triggered).total_seconds() < rule.suppress_duration_seconds
        ):
            return False

        # Check max alerts per hour
        hour_ago = now - timedelta(hours=1)
        recent_alerts = [
            alert
            for alert in self.alert_history
            if alert.rule_name == rule.name and alert.timestamp >= hour_ago
        ]

        if len(recent_alerts) >= rule.max_alerts_per_hour:
            return False

        return True

    def _is_silenced(self, rule: AlertRule, now: datetime) -> bool:
        """Check if alert should be silenced based on schedules"""

        current_time = now.time()
        current_weekday = now.weekday()

        for schedule in self.silence_schedules.values():
            if not schedule.active:
                continue

            # Check day of week
            if current_weekday not in schedule.days_of_week:
                continue

            # Check time range
            if schedule.start_time <= schedule.end_time:
                # Same day range
                in_time_range = schedule.start_time <= current_time <= schedule.end_time
            else:
                # Overnight range
                in_time_range = (
                    current_time >= schedule.start_time or current_time <= schedule.end_time
                )

            if not in_time_range:
                continue

            # Check severity filter
            if schedule.severity_filter and rule.severity not in schedule.severity_filter:
                continue

            # Check metric filter
            if schedule.metric_filter and rule.metric_name not in schedule.metric_filter:
                continue

            return True

        return False

    def _create_alert(self, rule: AlertRule, current_value: float, timestamp: datetime) -> Alert:
        """Create alert object"""

        title = rule.annotations.get("summary", f"Alert: {rule.name}")
        message = rule.annotations.get("description", rule.description)

        # Add current value to message
        message += f"\n\nCurrent value: {current_value:.2f} (threshold: {rule.threshold_value:.2f})"

        return Alert(
            timestamp=timestamp,
            rule_name=rule.name,
            severity=rule.severity,
            title=title,
            message=message,
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold_value=rule.threshold_value,
            labels=rule.labels.copy(),
            annotations=rule.annotations.copy(),
        )

    def _create_resolution_alert(
        self, rule: AlertRule, current_value: float, timestamp: datetime
    ) -> Alert:
        """Create alert resolution notification"""

        title = f"RESOLVED: {rule.annotations.get('summary', rule.name)}"
        message = f"Alert condition for '{rule.name}' has been resolved.\n\n"
        message += f"Current value: {current_value:.2f} (threshold: {rule.threshold_value:.2f})"

        return Alert(
            timestamp=timestamp,
            rule_name=f"{rule.name}_resolved",
            severity=AlertSeverity.INFO,
            title=title,
            message=message,
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold_value=rule.threshold_value,
            labels=rule.labels.copy(),
            annotations={"resolution": "true"},
        )

    def _fire_alert(self, alert: Alert):
        """Fire an alert to configured channels"""

        rule = self.alert_rules.get(alert.rule_name.replace("_resolved", ""))
        channels = rule.channels if rule else [AlertChannel.SLACK]

        for channel in channels:
            try:
                success = False
                if channel == AlertChannel.SLACK:
                    success = self._send_slack_alert(alert)
                elif channel == AlertChannel.EMAIL:
                    success = self._send_email_alert(alert)

                if success:
                    alert.channels_sent.append(channel)
                    alert.successful_delivery = True
                    self.alert_stats["alerts_by_channel"][channel.value] += 1
                else:
                    self.alert_stats["delivery_failures"] += 1

                alert.delivery_attempts += 1

            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")
                self.alert_stats["delivery_failures"] += 1

        # Store alert in history
        self.alert_history.append(alert)
        self.active_alerts[alert.rule_name] = alert

        # Update statistics
        self.alert_stats["total_alerts"] += 1
        self.alert_stats["alerts_by_severity"][alert.severity.value] += 1

        logger.warning(f"Alert fired: {alert.title} ({alert.severity.value})")

    def _send_slack_alert(self, alert: Alert) -> bool:
        """Send alert to Slack"""
        if not self.slack_client or not self.slack_channel:
            return False

        try:
            # Choose color based on severity
            color_map = {
                AlertSeverity.INFO: "#36a64f",
                AlertSeverity.WARNING: "#ff9500",
                AlertSeverity.ERROR: "#ff4444",
                AlertSeverity.CRITICAL: "#ff0000",
            }
            color = color_map.get(alert.severity, "#ff9500")

            # Format message
            blocks = [
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"*{alert.title}*\n{alert.message}"},
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Severity:*\n{alert.severity.value.upper()}"},
                        {"type": "mrkdwn", "text": f"*Metric:*\n{alert.metric_name}"},
                        {
                            "type": "mrkdwn",
                            "text": f"*Time:*\n{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                        },
                        {"type": "mrkdwn", "text": f"*Value:*\n{alert.current_value:.2f}"},
                    ],
                },
            ]

            response = self.slack_client.chat_postMessage(
                channel=self.slack_channel,
                text=alert.title,
                blocks=blocks,
                attachments=[{"color": color, "fallback": alert.message}],
            )

            return response.get("ok", False)

        except Exception as e:
            logger.error(f"Slack alert delivery failed: {e}")
            return False

    def _send_email_alert(self, alert: Alert) -> bool:
        """Send alert via email"""
        if not all(
            [self.smtp_server, self.email_username, self.email_password, self.email_recipients]
        ):
            return False

        try:
            msg = MIMEMultipart()
            msg["From"] = self.email_username
            msg["To"] = ", ".join(self.email_recipients)
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"

            # Create email body
            body = f"""
Alert Details:
--------------
Severity: {alert.severity.value.upper()}
Metric: {alert.metric_name}
Current Value: {alert.current_value:.2f}
Threshold: {alert.threshold_value:.2f}
Time: {alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")}

Description:
{alert.message}

--
CryptoSmartTrader Alert System
            """

            msg.attach(MIMEText(body, "plain"))

            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.email_username, self.email_password)
                server.send_message(msg)

            return True

        except Exception as e:
            logger.error(f"Email alert delivery failed: {e}")
            return False

    def _cleanup_old_alerts(self):
        """Clean up old alerts from history"""
        cutoff_time = datetime.now() - timedelta(days=7)
        self.alert_history = [
            alert for alert in self.alert_history if alert.timestamp >= cutoff_time
        ]

        # Clean up resolved active alerts
        resolved_alerts = []
        for rule_name, alert in self.active_alerts.items():
            if alert.timestamp < cutoff_time - timedelta(hours=1):
                resolved_alerts.append(rule_name)

        for rule_name in resolved_alerts:
            del self.active_alerts[rule_name]

    def manual_alert(
        self, title: str, message: str, severity: AlertSeverity = AlertSeverity.INFO
    ) -> bool:
        """Send manual alert"""
        alert = Alert(
            timestamp=datetime.now(),
            rule_name="manual_alert",
            severity=severity,
            title=title,
            message=message,
            metric_name="manual",
            current_value=0.0,
            threshold_value=0.0,
        )

        self._fire_alert(alert)
        return alert.successful_delivery

    def test_alert_delivery(self) -> Dict[str, bool]:
        """Test alert delivery to all configured channels"""
        results = {}

        test_alert = Alert(
            timestamp=datetime.now(),
            rule_name="test_alert",
            severity=AlertSeverity.INFO,
            title="🧪 Test Alert",
            message="This is a test alert to verify delivery channels are working correctly.",
            metric_name="test",
            current_value=0.0,
            threshold_value=0.0,
        )

        # Test Slack
        if self.slack_client and self.slack_channel:
            results[AlertChannel.SLACK.value] = self._send_slack_alert(test_alert)

        # Test Email
        if self.email_recipients:
            results[AlertChannel.EMAIL.value] = self._send_email_alert(test_alert)

        return results

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get comprehensive alert summary"""

        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)

        # Recent alerts
        recent_alerts_1h = [a for a in self.alert_history if a.timestamp >= hour_ago]
        recent_alerts_24h = [a for a in self.alert_history if a.timestamp >= day_ago]

        return {
            "timestamp": now.isoformat(),
            "monitoring_active": self.monitoring_active,
            "total_rules": len(self.alert_rules),
            "active_alerts": len(self.active_alerts),
            "silence_schedules": len(self.silence_schedules),
            "recent_activity": {
                "alerts_last_hour": len(recent_alerts_1h),
                "alerts_last_24h": len(recent_alerts_24h),
                "currently_firing": sum(
                    1 for rule in self.alert_rules.values() if rule.currently_firing
                ),
            },
            "statistics": self.alert_stats,
            "channel_status": {
                "slack_configured": bool(self.slack_client and self.slack_channel),
                "email_configured": bool(self.email_recipients and self.smtp_server),
            },
        }
