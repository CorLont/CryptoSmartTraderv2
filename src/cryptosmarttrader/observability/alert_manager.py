"""
Alert Manager for Critical Trading System Alerts
Handles alert routing, escalation, en notification delivery
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json

from .centralized_prometheus import AlertEvent, AlertSeverity

logger = logging.getLogger(__name__)


class NotificationChannel(Enum):
    EMAIL = "email"
    SLACK = "slack"
    TELEGRAM = "telegram"
    WEBHOOK = "webhook"
    SMS = "sms"
    PAGERDUTY = "pagerduty"


@dataclass
class NotificationTarget:
    """Notification target configuration"""
    channel: NotificationChannel
    target: str  # email, webhook URL, etc.
    severity_filter: List[AlertSeverity] = field(default_factory=lambda: list(AlertSeverity))
    enabled: bool = True


@dataclass
class EscalationRule:
    """Alert escalation rule"""
    name: str
    alert_patterns: List[str]  # Alert name patterns to match
    escalation_delay: int  # Seconds before escalation
    escalation_targets: List[NotificationTarget]
    max_escalations: int = 3


@dataclass
class AlertContext:
    """Additional context for alerts"""
    alert: AlertEvent
    first_notification_time: float
    last_notification_time: float
    notification_count: int = 0
    escalation_level: int = 0
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None


class AlertManager:
    """
    Advanced alert manager voor critical trading system alerts
    Handles routing, escalation, deduplication, en delivery
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Alert tracking
        self.active_alert_contexts: Dict[str, AlertContext] = {}
        self.notification_targets: List[NotificationTarget] = []
        self.escalation_rules: List[EscalationRule] = []
        
        # Notification handlers
        self.notification_handlers: Dict[NotificationChannel, Callable] = {}
        
        # Configuration
        self.notification_cooldown = 300  # 5 minutes between repeated notifications
        self.max_notifications_per_alert = 10
        
        # Threading
        self.alert_lock = threading.Lock()
        self.running = True
        
        # Setup default notification targets
        self._setup_default_targets()
        
        # Setup escalation rules
        self._setup_escalation_rules()
        
        # Register default handlers
        self._register_default_handlers()
        
        # Start alert processor
        self._start_alert_processor()
        
        self.logger.info("✅ Alert Manager initialized")
    
    def _setup_default_targets(self):
        """Setup default notification targets"""
        
        # Critical alerts go to multiple channels
        self.notification_targets = [
            NotificationTarget(
                channel=NotificationChannel.SLACK,
                target="#trading-alerts",
                severity_filter=[AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
            ),
            NotificationTarget(
                channel=NotificationChannel.EMAIL,
                target="trading-team@company.com",
                severity_filter=[AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
            ),
            NotificationTarget(
                channel=NotificationChannel.TELEGRAM,
                target="@trading_bot",
                severity_filter=[AlertSeverity.EMERGENCY]
            ),
            NotificationTarget(
                channel=NotificationChannel.WEBHOOK,
                target="https://hooks.company.com/trading-alerts",
                severity_filter=list(AlertSeverity)
            )
        ]
    
    def _setup_escalation_rules(self):
        """Setup alert escalation rules"""
        
        self.escalation_rules = [
            EscalationRule(
                name="emergency_escalation",
                alert_patterns=["DrawdownTooHigh", "KillSwitchActivated"],
                escalation_delay=60,  # 1 minute
                escalation_targets=[
                    NotificationTarget(
                        channel=NotificationChannel.SMS,
                        target="+1234567890",  # CTO phone
                        severity_filter=[AlertSeverity.EMERGENCY]
                    ),
                    NotificationTarget(
                        channel=NotificationChannel.PAGERDUTY,
                        target="trading-incidents",
                        severity_filter=[AlertSeverity.EMERGENCY]
                    )
                ],
                max_escalations=3
            ),
            EscalationRule(
                name="critical_escalation", 
                alert_patterns=["HighOrderErrorRate", "SlippageP95ExceedsBudget"],
                escalation_delay=300,  # 5 minutes
                escalation_targets=[
                    NotificationTarget(
                        channel=NotificationChannel.SLACK,
                        target="#trading-escalation",
                        severity_filter=[AlertSeverity.CRITICAL]
                    )
                ],
                max_escalations=2
            )
        ]
    
    def _register_default_handlers(self):
        """Register default notification handlers"""
        
        # Slack handler
        def slack_handler(target: str, alert_context: AlertContext) -> bool:
            """Send Slack notification"""
            try:
                alert = alert_context.alert
                message = self._format_slack_message(alert)
                
                # In production, you'd use the Slack API
                self.logger.info(f"📱 SLACK → {target}: {alert.rule_name}")
                self.logger.info(f"📱 Message: {message}")
                
                return True
            except Exception as e:
                self.logger.error(f"❌ Slack notification failed: {e}")
                return False
        
        # Email handler
        def email_handler(target: str, alert_context: AlertContext) -> bool:
            """Send email notification"""
            try:
                alert = alert_context.alert
                subject = f"[{alert.severity.value.upper()}] Trading Alert: {alert.rule_name}"
                body = self._format_email_body(alert)
                
                # In production, you'd use SMTP
                self.logger.info(f"📧 EMAIL → {target}: {subject}")
                self.logger.info(f"📧 Body: {body}")
                
                return True
            except Exception as e:
                self.logger.error(f"❌ Email notification failed: {e}")
                return False
        
        # Webhook handler
        def webhook_handler(target: str, alert_context: AlertContext) -> bool:
            """Send webhook notification"""
            try:
                alert = alert_context.alert
                payload = self._format_webhook_payload(alert)
                
                # In production, you'd make HTTP POST request
                self.logger.info(f"🔗 WEBHOOK → {target}: {alert.rule_name}")
                self.logger.info(f"🔗 Payload: {json.dumps(payload, indent=2)}")
                
                return True
            except Exception as e:
                self.logger.error(f"❌ Webhook notification failed: {e}")
                return False
        
        # Telegram handler  
        def telegram_handler(target: str, alert_context: AlertContext) -> bool:
            """Send Telegram notification"""
            try:
                alert = alert_context.alert
                message = self._format_telegram_message(alert)
                
                # In production, you'd use Telegram Bot API
                self.logger.info(f"💬 TELEGRAM → {target}: {alert.rule_name}")
                self.logger.info(f"💬 Message: {message}")
                
                return True
            except Exception as e:
                self.logger.error(f"❌ Telegram notification failed: {e}")
                return False
        
        # Register handlers
        self.notification_handlers[NotificationChannel.SLACK] = slack_handler
        self.notification_handlers[NotificationChannel.EMAIL] = email_handler
        self.notification_handlers[NotificationChannel.WEBHOOK] = webhook_handler
        self.notification_handlers[NotificationChannel.TELEGRAM] = telegram_handler
    
    def _format_slack_message(self, alert: AlertEvent) -> str:
        """Format Slack message"""
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️", 
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "🆘"
        }
        
        emoji = severity_emoji.get(alert.severity, "❓")
        duration = time.time() - alert.started_at
        
        return f"""
{emoji} *{alert.severity.value.upper()}*: {alert.rule_name}

📊 *Current Value*: {alert.current_value:.2f}
🎯 *Threshold*: {alert.threshold:.2f}
⏱️ *Duration*: {duration:.1f}s
📝 *Description*: {alert.description}

🔗 [View Dashboard](http://localhost:5000) | [Runbook](https://wiki.internal/runbooks)
"""
    
    def _format_email_body(self, alert: AlertEvent) -> str:
        """Format email body"""
        duration = time.time() - alert.started_at
        timestamp = datetime.fromtimestamp(alert.started_at).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        return f"""
TRADING SYSTEM ALERT

Alert: {alert.rule_name}
Severity: {alert.severity.value.upper()}
Started: {timestamp}
Duration: {duration:.1f} seconds

Current Value: {alert.current_value:.2f}
Threshold: {alert.threshold:.2f}

Description: {alert.description}

Please investigate immediately and take appropriate action.

Trading System Monitoring
"""
    
    def _format_webhook_payload(self, alert: AlertEvent) -> Dict[str, Any]:
        """Format webhook payload"""
        return {
            "alert_name": alert.rule_name,
            "severity": alert.severity.value,
            "current_value": alert.current_value,
            "threshold": alert.threshold,
            "started_at": alert.started_at,
            "description": alert.description,
            "labels": alert.labels,
            "dashboard_url": "http://localhost:5000",
            "timestamp": time.time()
        }
    
    def _format_telegram_message(self, alert: AlertEvent) -> str:
        """Format Telegram message"""
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨", 
            AlertSeverity.EMERGENCY: "🆘"
        }
        
        emoji = severity_emoji.get(alert.severity, "❓")
        duration = time.time() - alert.started_at
        
        return f"""
{emoji} *TRADING ALERT*

🔴 {alert.rule_name}
📊 Value: {alert.current_value:.2f} (threshold: {alert.threshold:.2f})
⏱️ Duration: {duration:.1f}s

{alert.description}

[Dashboard](http://localhost:5000)
"""
    
    def _start_alert_processor(self):
        """Start alert processing thread"""
        def process_alerts():
            while self.running:
                try:
                    self._process_escalations()
                    time.sleep(30)  # Check every 30 seconds
                except Exception as e:
                    self.logger.error(f"❌ Alert processing error: {e}")
                    time.sleep(60)
        
        processor_thread = threading.Thread(target=process_alerts, daemon=True)
        processor_thread.start()
        self.logger.info("🚀 Alert processor started")
    
    def handle_alert(self, alert: AlertEvent):
        """Handle incoming alert"""
        with self.alert_lock:
            alert_key = alert.rule_name
            current_time = time.time()
            
            if alert_key not in self.active_alert_contexts:
                # New alert
                context = AlertContext(
                    alert=alert,
                    first_notification_time=current_time,
                    last_notification_time=current_time,
                    notification_count=0
                )
                
                self.active_alert_contexts[alert_key] = context
                
                # Send immediate notifications
                self._send_notifications(context)
                
                self.logger.info(f"🚨 New alert handled: {alert.rule_name}")
            else:
                # Update existing alert
                context = self.active_alert_contexts[alert_key]
                context.alert = alert  # Update with latest values
    
    def resolve_alert(self, alert_name: str):
        """Resolve alert"""
        with self.alert_lock:
            if alert_name in self.active_alert_contexts:
                context = self.active_alert_contexts.pop(alert_name)
                self._send_resolution_notifications(context)
                self.logger.info(f"✅ Alert resolved: {alert_name}")
    
    def acknowledge_alert(self, alert_name: str, acknowledged_by: str):
        """Acknowledge alert"""
        with self.alert_lock:
            if alert_name in self.active_alert_contexts:
                context = self.active_alert_contexts[alert_name]
                context.acknowledged = True
                context.acknowledged_by = acknowledged_by
                context.acknowledged_at = time.time()
                
                self.logger.info(f"👍 Alert acknowledged by {acknowledged_by}: {alert_name}")
    
    def _send_notifications(self, context: AlertContext):
        """Send notifications for alert"""
        alert = context.alert
        current_time = time.time()
        
        # Check notification cooldown
        if (current_time - context.last_notification_time < self.notification_cooldown and
            context.notification_count > 0):
            return
        
        # Check max notifications limit
        if context.notification_count >= self.max_notifications_per_alert:
            return
        
        # Send to configured targets
        for target in self.notification_targets:
            if not target.enabled:
                continue
            
            # Check severity filter
            if alert.severity not in target.severity_filter:
                continue
            
            # Send notification
            handler = self.notification_handlers.get(target.channel)
            if handler:
                try:
                    success = handler(target.target, context)
                    if success:
                        context.notification_count += 1
                        context.last_notification_time = current_time
                        self.logger.info(
                            f"📤 Notification sent via {target.channel.value} to {target.target}"
                        )
                except Exception as e:
                    self.logger.error(
                        f"❌ Notification failed via {target.channel.value}: {e}"
                    )
    
    def _send_resolution_notifications(self, context: AlertContext):
        """Send alert resolution notifications"""
        alert = context.alert
        duration = time.time() - alert.started_at
        
        for target in self.notification_targets:
            if not target.enabled or alert.severity not in target.severity_filter:
                continue
            
            handler = self.notification_handlers.get(target.channel)
            if handler:
                try:
                    # Create resolution alert context
                    resolution_alert = AlertEvent(
                        rule_name=f"{alert.rule_name}_RESOLVED",
                        severity=AlertSeverity.INFO,
                        current_value=alert.current_value,
                        threshold=alert.threshold,
                        started_at=alert.started_at,
                        description=f"Alert resolved after {duration:.1f} seconds"
                    )
                    
                    resolution_context = AlertContext(
                        alert=resolution_alert,
                        first_notification_time=time.time(),
                        last_notification_time=time.time()
                    )
                    
                    handler(target.target, resolution_context)
                    
                    self.logger.info(
                        f"✅ Resolution notification sent via {target.channel.value}"
                    )
                except Exception as e:
                    self.logger.error(f"❌ Resolution notification failed: {e}")
    
    def _process_escalations(self):
        """Process alert escalations"""
        current_time = time.time()
        
        with self.alert_lock:
            for context in list(self.active_alert_contexts.values()):
                if context.acknowledged:
                    continue  # Don't escalate acknowledged alerts
                
                # Check escalation rules
                for rule in self.escalation_rules:
                    if self._matches_escalation_rule(context.alert, rule):
                        time_since_first = current_time - context.first_notification_time
                        expected_escalations = int(time_since_first / rule.escalation_delay)
                        
                        if (expected_escalations > context.escalation_level and
                            context.escalation_level < rule.max_escalations):
                            
                            self._escalate_alert(context, rule)
                            context.escalation_level += 1
    
    def _matches_escalation_rule(self, alert: AlertEvent, rule: EscalationRule) -> bool:
        """Check if alert matches escalation rule"""
        for pattern in rule.alert_patterns:
            if pattern in alert.rule_name:
                return True
        return False
    
    def _escalate_alert(self, context: AlertContext, rule: EscalationRule):
        """Escalate alert to higher notification targets"""
        alert = context.alert
        
        self.logger.warning(
            f"📈 Escalating alert {alert.rule_name} (level {context.escalation_level + 1})"
        )
        
        for target in rule.escalation_targets:
            if alert.severity in target.severity_filter:
                handler = self.notification_handlers.get(target.channel)
                if handler:
                    try:
                        handler(target.target, context)
                        self.logger.info(
                            f"📈 Escalation sent via {target.channel.value} to {target.target}"
                        )
                    except Exception as e:
                        self.logger.error(f"❌ Escalation failed: {e}")
    
    def add_notification_target(self, target: NotificationTarget):
        """Add notification target"""
        self.notification_targets.append(target)
        self.logger.info(f"📤 Added notification target: {target.channel.value}")
    
    def register_notification_handler(
        self, 
        channel: NotificationChannel, 
        handler: Callable[[str, AlertContext], bool]
    ):
        """Register custom notification handler"""
        self.notification_handlers[channel] = handler
        self.logger.info(f"🔧 Registered handler for {channel.value}")
    
    def get_alert_status(self) -> Dict[str, Any]:
        """Get alert manager status"""
        with self.alert_lock:
            return {
                "active_alerts": len(self.active_alert_contexts),
                "notification_targets": len(self.notification_targets),
                "escalation_rules": len(self.escalation_rules),
                "active_alert_details": [
                    {
                        "name": context.alert.rule_name,
                        "severity": context.alert.severity.value,
                        "duration": time.time() - context.alert.started_at,
                        "notification_count": context.notification_count,
                        "escalation_level": context.escalation_level,
                        "acknowledged": context.acknowledged,
                        "acknowledged_by": context.acknowledged_by
                    }
                    for context in self.active_alert_contexts.values()
                ]
            }
    
    def shutdown(self):
        """Shutdown alert manager"""
        self.running = False
        self.logger.info("🛑 Alert Manager shutting down")


# Global alert manager instance
_global_alert_manager: Optional[AlertManager] = None


def get_global_alert_manager() -> AlertManager:
    """Get or create global alert manager"""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager()
        logger.info("✅ Global AlertManager initialized")
    return _global_alert_manager


def reset_global_alert_manager():
    """Reset global alert manager (for testing)"""
    global _global_alert_manager
    if _global_alert_manager:
        _global_alert_manager.shutdown()
    _global_alert_manager = None