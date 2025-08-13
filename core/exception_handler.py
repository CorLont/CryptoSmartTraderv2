#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Enterprise Exception Handler
Consistent exception handling with escalation to alerts and health score
"""

import logging
import traceback
import threading
import time
from typing import Dict, Any, List, Optional, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path


class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    DATA_COLLECTION = "data_collection"
    ML_MODEL = "ml_model"
    TRADING = "trading"
    SECURITY = "security"
    SYSTEM = "system"
    NETWORK = "network"
    EXTERNAL_API = "external_api"


@dataclass
class ErrorReport:
    """Comprehensive error report"""

    error_id: str
    timestamp: datetime
    category: ErrorCategory
    level: AlertLevel
    module: str
    function: str
    error_type: str
    message: str
    traceback_info: str
    context: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    resolved: bool = False
    health_impact: float = 0.0  # 0.0 to 1.0, impact on system health


@dataclass
class AlertConfig:
    """Alert configuration"""

    email_enabled: bool = False
    email_smtp_server: str = "smtp.gmail.com"
    email_smtp_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_to_addresses: List[str] = field(default_factory=list)

    discord_enabled: bool = False
    discord_webhook_url: str = ""

    slack_enabled: bool = False
    slack_webhook_url: str = ""

    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Alert throttling
    min_alert_interval: int = 300  # 5 minutes between same type alerts
    max_alerts_per_hour: int = 10
    critical_alert_bypass: bool = True  # Critical alerts always sent


class ExceptionHandler:
    """Enterprise exception handler with comprehensive error management"""

    def __init__(self, alert_config: Optional[AlertConfig] = None):
        self.alert_config = alert_config or AlertConfig()
        self.logger = logging.getLogger(f"{__name__}.ExceptionHandler")

        # Error tracking
        self.error_reports: List[ErrorReport] = []
        self.error_count_by_category: Dict[ErrorCategory, int] = {}
        self.last_alert_times: Dict[str, datetime] = {}
        self.alerts_sent_this_hour: int = 0
        self.last_hour_reset: datetime = datetime.now()

        # Thread safety
        self._lock = threading.RLock()

        # Error handlers by category
        self.category_handlers: Dict[ErrorCategory, Callable] = {
            ErrorCategory.DATA_COLLECTION: self._handle_data_error,
            ErrorCategory.ML_MODEL: self._handle_ml_error,
            ErrorCategory.TRADING: self._handle_trading_error,
            ErrorCategory.SECURITY: self._handle_security_error,
            ErrorCategory.SYSTEM: self._handle_system_error,
            ErrorCategory.NETWORK: self._handle_network_error,
            ErrorCategory.EXTERNAL_API: self._handle_api_error,
        }

        # Health impact weights
        self.health_impact_weights = {
            ErrorCategory.SECURITY: 1.0,
            ErrorCategory.TRADING: 0.8,
            ErrorCategory.ML_MODEL: 0.7,
            ErrorCategory.DATA_COLLECTION: 0.6,
            ErrorCategory.EXTERNAL_API: 0.5,
            ErrorCategory.NETWORK: 0.4,
            ErrorCategory.SYSTEM: 0.3,
        }

        self.logger.info("Enterprise Exception Handler initialized")

    def handle_exception(
        self,
        exception: Exception,
        category: ErrorCategory,
        module: str,
        function: str,
        context: Dict[str, Any] = None,
        level: AlertLevel = AlertLevel.ERROR,
    ) -> ErrorReport:
        """
        Handle an exception with comprehensive tracking and alerting

        Args:
            exception: The exception that occurred
            category: Error category for classification
            module: Module where error occurred
            function: Function where error occurred
            context: Additional context information
            level: Alert level for this error

        Returns:
            ErrorReport with tracking information
        """
        with self._lock:
            # Create error report
            error_id = f"{category.value}_{module}_{function}_{int(time.time())}"

            error_report = ErrorReport(
                error_id=error_id,
                timestamp=datetime.now(),
                category=category,
                level=level,
                module=module,
                function=function,
                error_type=type(exception).__name__,
                message=str(exception),
                traceback_info=traceback.format_exc(),
                context=context or {},
                health_impact=self.health_impact_weights.get(category, 0.5)
                * (
                    1.0
                    if level == AlertLevel.CRITICAL
                    else 0.8
                    if level == AlertLevel.ERROR
                    else 0.5
                    if level == AlertLevel.WARNING
                    else 0.2
                ),
            )

            # Store error report
            self.error_reports.append(error_report)

            # Update error counts
            self.error_count_by_category[category] = (
                self.error_count_by_category.get(category, 0) + 1
            )

            # Log the error
            self._log_error(error_report)

            # Apply category-specific handling
            try:
                handler = self.category_handlers.get(category)
                if handler:
                    handler(error_report)
            except Exception as e:
                self.logger.error(f"Error in category handler for {category}: {e}")

            # Send alerts if necessary
            if self._should_send_alert(error_report):
                self._send_alert(error_report)

            # Cleanup old error reports
            self._cleanup_old_errors()

            return error_report

    def _log_error(self, error_report: ErrorReport):
        """Log error with appropriate level"""
        log_message = (
            f"[{error_report.error_id}] {error_report.category.value.upper()} ERROR "
            f"in {error_report.module}.{error_report.function}: {error_report.message}"
        )

        if error_report.context:
            log_message += f" | Context: {json.dumps(error_report.context, default=str)}"

        if error_report.level == AlertLevel.CRITICAL:
            self.logger.critical(log_message)
            self.logger.critical(f"Traceback:\n{error_report.traceback_info}")
        elif error_report.level == AlertLevel.ERROR:
            self.logger.error(log_message)
            self.logger.debug(f"Traceback:\n{error_report.traceback_info}")
        elif error_report.level == AlertLevel.WARNING:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def _handle_data_error(self, error_report: ErrorReport):
        """Handle data collection errors"""
        # Check for fallback data issues
        if (
            "fallback" in error_report.message.lower()
            or "synthetic" in error_report.message.lower()
        ):
            error_report.level = AlertLevel.CRITICAL
            error_report.health_impact = 1.0
            self.logger.critical(
                "CRITICAL: Fallback/synthetic data detected - production violation!"
            )

        # Check for missing data
        elif "missing" in error_report.message.lower() or "empty" in error_report.message.lower():
            error_report.level = AlertLevel.ERROR
            error_report.health_impact = 0.8

    def _handle_ml_error(self, error_report: ErrorReport):
        """Handle ML model errors"""
        # Check for model training failures
        if "training" in error_report.message.lower() or "model" in error_report.message.lower():
            error_report.level = AlertLevel.ERROR
            error_report.health_impact = 0.9

        # Check for prediction failures
        elif (
            "prediction" in error_report.message.lower()
            or "inference" in error_report.message.lower()
        ):
            error_report.level = AlertLevel.WARNING
            error_report.health_impact = 0.6

    def _handle_trading_error(self, error_report: ErrorReport):
        """Handle trading-related errors"""
        # Trading errors are always high priority
        if error_report.level == AlertLevel.INFO:
            error_report.level = AlertLevel.WARNING

        # Check for order execution failures
        if "order" in error_report.message.lower() or "trade" in error_report.message.lower():
            error_report.level = AlertLevel.ERROR
            error_report.health_impact = 1.0

    def _handle_security_error(self, error_report: ErrorReport):
        """Handle security-related errors"""
        # Security errors are always critical
        error_report.level = AlertLevel.CRITICAL
        error_report.health_impact = 1.0

        # Additional logging for security events
        security_log_path = Path("logs/security_errors.log")
        security_log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(security_log_path, "a") as f:
            f.write(
                f"{error_report.timestamp.isoformat()} - SECURITY ERROR: {error_report.message}\n"
            )
            f.write(f"Module: {error_report.module}, Function: {error_report.function}\n")
            f.write(f"Context: {json.dumps(error_report.context, default=str)}\n")
            f.write(f"Traceback:\n{error_report.traceback_info}\n")
            f.write("-" * 80 + "\n")

    def _handle_system_error(self, error_report: ErrorReport):
        """Handle system-level errors"""
        # Check for memory/resource issues
        if any(
            keyword in error_report.message.lower()
            for keyword in ["memory", "disk", "cpu", "resource"]
        ):
            error_report.level = AlertLevel.CRITICAL
            error_report.health_impact = 0.9

    def _handle_network_error(self, error_report: ErrorReport):
        """Handle network-related errors"""
        # Network errors might be transient
        if (
            "timeout" in error_report.message.lower()
            or "connection" in error_report.message.lower()
        ):
            error_report.level = AlertLevel.WARNING
            error_report.health_impact = 0.4

    def _handle_api_error(self, error_report: ErrorReport):
        """Handle external API errors"""
        # Check for rate limiting
        if "rate limit" in error_report.message.lower() or "429" in error_report.message:
            error_report.level = AlertLevel.WARNING
            error_report.health_impact = 0.3

        # Check for authentication issues
        elif (
            "auth" in error_report.message.lower()
            or "401" in error_report.message
            or "403" in error_report.message
        ):
            error_report.level = AlertLevel.ERROR
            error_report.health_impact = 0.7

    def _should_send_alert(self, error_report: ErrorReport) -> bool:
        """Determine if an alert should be sent"""
        with self._lock:
            # Reset hourly alert counter
            now = datetime.now()
            if (now - self.last_hour_reset).total_seconds() >= 3600:
                self.alerts_sent_this_hour = 0
                self.last_hour_reset = now

            # Critical alerts bypass all limits
            if (
                error_report.level == AlertLevel.CRITICAL
                and self.alert_config.critical_alert_bypass
            ):
                return True

            # Check hourly limit
            if self.alerts_sent_this_hour >= self.alert_config.max_alerts_per_hour:
                return False

            # Check minimum interval
            alert_key = f"{error_report.category.value}_{error_report.error_type}"
            last_alert = self.last_alert_times.get(alert_key)

            if last_alert:
                time_since_last = (now - last_alert).total_seconds()
                if time_since_last < self.alert_config.min_alert_interval:
                    return False

            # Only send alerts for WARNING and above
            return error_report.level in (AlertLevel.WARNING, AlertLevel.ERROR, AlertLevel.CRITICAL)

    def _send_alert(self, error_report: ErrorReport):
        """Send alert via configured channels"""
        try:
            alert_message = self._format_alert_message(error_report)

            # Send email alert
            if self.alert_config.email_enabled:
                self._send_email_alert(error_report, alert_message)

            # Send Discord alert
            if self.alert_config.discord_enabled:
                self._send_discord_alert(error_report, alert_message)

            # Send Slack alert
            if self.alert_config.slack_enabled:
                self._send_slack_alert(error_report, alert_message)

            # Send Telegram alert
            if self.alert_config.telegram_enabled:
                self._send_telegram_alert(error_report, alert_message)

            # Update tracking
            alert_key = f"{error_report.category.value}_{error_report.error_type}"
            self.last_alert_times[alert_key] = datetime.now()
            self.alerts_sent_this_hour += 1

            self.logger.info(f"Alert sent for error {error_report.error_id}")

        except Exception as e:
            self.logger.error(f"Failed to send alert for {error_report.error_id}: {e}")

    def _format_alert_message(self, error_report: ErrorReport) -> str:
        """Format alert message"""
        level_emoji = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }

        message = f"{level_emoji.get(error_report.level, '⚠️')} **CryptoSmartTrader Alert**\n\n"
        message += f"**Level:** {error_report.level.value.upper()}\n"
        message += f"**Category:** {error_report.category.value.replace('_', ' ').title()}\n"
        message += f"**Module:** {error_report.module}\n"
        message += f"**Function:** {error_report.function}\n"
        message += f"**Error:** {error_report.error_type}\n"
        message += f"**Message:** {error_report.message}\n"
        message += f"**Time:** {error_report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        message += f"**Health Impact:** {error_report.health_impact:.1%}\n"

        if error_report.context:
            message += f"\n**Context:**\n```json\n{json.dumps(error_report.context, indent=2, default=str)}\n```"

        return message

    def _send_email_alert(self, error_report: ErrorReport, message: str):
        """Send email alert"""
        try:
            msg = MIMEMultipart()
            msg["From"] = self.alert_config.email_username
            msg["To"] = ", ".join(self.alert_config.email_to_addresses)
            msg["Subject"] = (
                f"CryptoSmartTrader {error_report.level.value.upper()} Alert - {error_report.category.value}"
            )

            msg.attach(MIMEText(message, "plain"))

            server = smtplib.SMTP(
                self.alert_config.email_smtp_server, self.alert_config.email_smtp_port
            )
            server.starttls()
            server.login(self.alert_config.email_username, self.alert_config.email_password)

            text = msg.as_string()
            server.sendmail(
                self.alert_config.email_username, self.alert_config.email_to_addresses, text
            )
            server.quit()

        except Exception as e:
            self.logger.error(f"Failed to send email alert: {e}")

    def _send_discord_alert(self, error_report: ErrorReport, message: str):
        """Send Discord webhook alert"""
        # Implementation would use aiohttp to send webhook
        # Placeholder for now
        pass

    def _send_slack_alert(self, error_report: ErrorReport, message: str):
        """Send Slack webhook alert"""
        # Implementation would use aiohttp to send webhook
        # Placeholder for now
        pass

    def _send_telegram_alert(self, error_report: ErrorReport, message: str):
        """Send Telegram bot alert"""
        # Implementation would use aiohttp to send via Telegram API
        # Placeholder for now
        pass

    def _cleanup_old_errors(self):
        """Clean up old error reports"""
        cutoff_time = datetime.now() - timedelta(days=7)

        self.error_reports = [
            report for report in self.error_reports if report.timestamp > cutoff_time
        ]

        # Keep only last 1000 errors
        if len(self.error_reports) > 1000:
            self.error_reports = self.error_reports[-1000:]

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for health scoring"""
        with self._lock:
            now = datetime.now()
            last_hour = now - timedelta(hours=1)
            last_day = now - timedelta(days=1)

            recent_errors = [r for r in self.error_reports if r.timestamp > last_hour]
            daily_errors = [r for r in self.error_reports if r.timestamp > last_day]

            stats = {
                "total_errors": len(self.error_reports),
                "errors_last_hour": len(recent_errors),
                "errors_last_day": len(daily_errors),
                "critical_errors_last_hour": len(
                    [r for r in recent_errors if r.level == AlertLevel.CRITICAL]
                ),
                "error_rate_per_hour": len(recent_errors),
                "average_health_impact": sum(r.health_impact for r in recent_errors)
                / max(len(recent_errors), 1),
                "error_count_by_category": dict(self.error_count_by_category),
                "alerts_sent_this_hour": self.alerts_sent_this_hour,
                "unresolved_critical_errors": len(
                    [
                        r
                        for r in self.error_reports
                        if r.level == AlertLevel.CRITICAL and not r.resolved
                    ]
                ),
            }

            return stats

    def resolve_error(self, error_id: str) -> bool:
        """Mark an error as resolved"""
        with self._lock:
            for report in self.error_reports:
                if report.error_id == error_id:
                    report.resolved = True
                    self.logger.info(f"Error {error_id} marked as resolved")
                    return True
            return False

    def get_health_impact_score(self) -> float:
        """Calculate health impact score for system health calculation"""
        with self._lock:
            now = datetime.now()
            last_hour = now - timedelta(hours=1)

            # Get recent errors
            recent_errors = [r for r in self.error_reports if r.timestamp > last_hour]

            if not recent_errors:
                return 1.0  # Perfect health

            # Calculate weighted impact
            total_impact = sum(r.health_impact for r in recent_errors)
            error_count = len(recent_errors)

            # Normalize impact (more errors = worse health)
            health_score = max(0.0, 1.0 - (total_impact / 10.0) - (error_count / 50.0))

            return health_score


# Singleton exception handler
_exception_handler = None
_handler_lock = threading.Lock()


def get_exception_handler(alert_config: Optional[AlertConfig] = None) -> ExceptionHandler:
    """Get the singleton exception handler instance"""
    global _exception_handler

    with _handler_lock:
        if _exception_handler is None:
            _exception_handler = ExceptionHandler(alert_config)
        return _exception_handler


def handle_error(
    exception: Exception,
    category: ErrorCategory,
    module: str,
    function: str,
    context: Dict[str, Any] = None,
    level: AlertLevel = AlertLevel.ERROR,
) -> ErrorReport:
    """Convenient function to handle an error"""
    handler = get_exception_handler()
    return handler.handle_exception(exception, category, module, function, context, level)


# Decorator for automatic exception handling
def exception_handler(category: ErrorCategory, level: AlertLevel = AlertLevel.ERROR):
    """Decorator for automatic exception handling"""

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handle_error(
                    exception=e,
                    category=category,
                    module=func.__module__,
                    function=func.__name__,
                    context={"args": args, "kwargs": kwargs},
                    level=level,
                )
                raise

        return wrapper

    return decorator
