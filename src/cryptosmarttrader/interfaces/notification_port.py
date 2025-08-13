"""
Notification Port - Interface for alert and reporting systems

Defines the contract for notification implementations enabling
swappable notification backends (email, SMS, webhooks, etc.).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
from dataclasses import dataclass


class NotificationSeverity(Enum):
    """Notification severity levels"""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """Available notification channels"""

    EMAIL = "email"
    SMS = "sms"
    SLACK = "slack"
    DISCORD = "discord"
    WEBHOOK = "webhook"
    CONSOLE = "console"
    FILE = "file"


@dataclass
class NotificationMessage:
    """Notification message structure"""

    title: str
    content: str
    severity: NotificationSeverity
    timestamp: datetime
    metadata: Optional[Dict] = None


@dataclass
class NotificationResult:
    """Result of notification send operation"""

    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    delivery_time: Optional[datetime] = None


class NotificationPort(ABC):
    """
    Abstract interface for notification systems

    This port defines the contract for sending alerts, reports,
    and other notifications through various channels.
    """

    @abstractmethod
    def send_alert(
        self,
        message: str,
        severity: NotificationSeverity = NotificationSeverity.INFO,
        channels: Optional[List[NotificationChannel]] = None,
        metadata: Optional[Dict] = None,
    ) -> NotificationResult:
        """
        Send alert notification

        Args:
            message: Alert message content
            severity: Severity level of the alert
            channels: Specific channels to use (None = use defaults)
            metadata: Additional metadata for the alert

        Returns:
            NotificationResult with send status
        """
        pass

    @abstractmethod
    def send_report(
        self,
        report_data: Dict[str, Any],
        report_type: str = "daily",
        channels: Optional[List[NotificationChannel]] = None,
    ) -> NotificationResult:
        """
        Send periodic report

        Args:
            report_data: Report content and data
            report_type: Type of report (daily, weekly, etc.)
            channels: Channels to send report to

        Returns:
            NotificationResult with send status
        """
        pass

    @abstractmethod
    def send_trade_notification(
        self, trade_data: Dict[str, Any], notification_type: str = "execution"
    ) -> NotificationResult:
        """
        Send trading-related notification

        Args:
            trade_data: Trade execution details
            notification_type: Type of trade notification

        Returns:
            NotificationResult with send status
        """
        pass

    @abstractmethod
    def get_notification_history(self, limit: int = 100) -> List[NotificationMessage]:
        """
        Get recent notification history

        Args:
            limit: Maximum number of notifications to return

        Returns:
            List of recent notifications
        """
        pass

    @abstractmethod
    def configure_channel(self, channel: NotificationChannel, config: Dict[str, Any]) -> bool:
        """
        Configure notification channel settings

        Args:
            channel: Channel to configure
            config: Channel configuration parameters

        Returns:
            True if configuration was successful
        """
        pass


# Registry for notification implementations
class NotificationRegistry:
    """Registry for managing notification implementations"""

    def __init__(self):
        self._notifiers: Dict[str, NotificationPort] = {}
        self._default_notifier: Optional[str] = None

    def register_notifier(self, name: str, notifier: NotificationPort, is_default: bool = False):
        """Register a notification implementation"""
        self._notifiers[name] = notifier
        if is_default or self._default_notifier is None:
            self._default_notifier = name

    def get_notifier(self, name: Optional[str] = None) -> NotificationPort:
        """Get specific notifier or default"""
        notifier_name = name or self._default_notifier
        if notifier_name not in self._notifiers:
            raise ValueError(f"Notifier '{notifier_name}' not found")
        return self._notifiers[notifier_name]


# Global registry
notification_registry = NotificationRegistry()
