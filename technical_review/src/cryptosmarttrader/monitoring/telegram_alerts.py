"""
Telegram Alert System

Dedicated Telegram alerting implementation with rich formatting,
priority routing, and 24/7 monitoring capabilities.
"""

import asyncio
import requests
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import json
import os

logger = logging.getLogger(__name__)

class TelegramMessageType(Enum):
    """Telegram message types"""
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"

class AlertPriority(Enum):
    """Alert priority levels"""
    LOW = "🟢"
    MEDIUM = "🟡"
    HIGH = "🟠"
    CRITICAL = "🔴"
    EMERGENCY = "🚨"

@dataclass
class TelegramAlert:
    """Telegram alert message"""
    title: str
    message: str
    priority: AlertPriority = AlertPriority.MEDIUM

    # Formatting
    message_type: TelegramMessageType = TelegramMessageType.MARKDOWN
    disable_notification: bool = False

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    alert_id: str = ""
    tags: List[str] = field(default_factory=list)

    # Delivery tracking
    sent: bool = False
    delivery_attempts: int = 0
    error_message: Optional[str] = None

class TelegramAlertsManager:
    """
    Advanced Telegram alerting system for CryptoSmartTrader
    """

    def __init__(self):
        # Telegram configuration from environment
        self.bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")

        # API configuration
        self.api_base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None
        self.timeout_seconds = 30
        self.max_retries = 3

        # Message queue and history
        self.alert_queue: List[TelegramAlert] = []
        self.alert_history: List[TelegramAlert] = []

        # Rate limiting
        self.last_message_time = 0
        self.min_message_interval = 1.0  # 1 second between messages
        self.burst_protection = True

        # Statistics
        self.stats = {
            'total_alerts': 0,
            'successful_deliveries': 0,
            'failed_deliveries': 0,
            'rate_limited': 0,
            'uptime_start': datetime.now()
        }

        # Validate configuration
        self._validate_configuration()

        logger.info(f"Telegram Alerts Manager initialized - Chat ID: {self.chat_id}")

    def _validate_configuration(self) -> bool:
        """Validate Telegram configuration"""

        if not self.bot_token:
            logger.error("TELEGRAM_BOT_TOKEN not found in environment variables")
            return False

        if not self.chat_id:
            logger.error("TELEGRAM_CHAT_ID not found in environment variables")
            return False

        # Test connection
        try:
            response = requests.get(
                f"{self.api_base_url}/getMe",
                timeout=10
            )

            if response.status_code == 200:
                bot_info = response.json()
                if bot_info.get('ok'):
                    logger.info(f"Telegram bot connected: {bot_info['result']['username']}")
                    return True
                else:
                    logger.error(f"Telegram API error: {bot_info.get('description')}")
            else:
                logger.error(f"Telegram connection failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")

        return False

    def send_alert(self,
                   title: str,
                   message: str,
                   priority: AlertPriority = AlertPriority.MEDIUM,
                   tags: Optional[List[str]] = None,
                   disable_notification: bool = False) -> bool:
        """Send immediate Telegram alert"""

        alert = TelegramAlert(
            title=title,
            message=message,
            priority=priority,
            tags=tags or [],
            disable_notification=disable_notification,
            alert_id=f"alert_{int(time.time())}"
        )

        return self._send_telegram_message(alert)

    def send_trading_alert(self,
                          symbol: str,
                          action: str,
                          details: Dict[str, Any],
                          priority: AlertPriority = AlertPriority.HIGH) -> bool:
        """Send specialized trading alert"""

        # Format trading message
        title = f"🔄 Trading Alert: {symbol}"

        message = f"*{title}*\n\n"
        message += f"**Action:** {action}\n"

        for key, value in details.items():
            formatted_key = key.replace('_', ' ').title()
            message += f"**{formatted_key}:** {value}\n"

        message += f"\n**Time:** {datetime.now().strftime('%H:%M:%S')}"

        return self.send_alert(
            title=title,
            message=message,
            priority=priority,
            tags=["trading", symbol.lower()]
        )

    def send_system_alert(self,
                         component: str,
                         status: str,
                         details: str,
                         priority: AlertPriority = AlertPriority.CRITICAL) -> bool:
        """Send system status alert"""

        status_emoji = {
            'healthy': '✅',
            'warning': '⚠️',
            'error': '❌',
            'critical': '🚨',
            'recovering': '🔄'
        }.get(status.lower(), '❓')

        title = f"{status_emoji} System Alert: {component}"

        message = f"*{title}*\n\n"
        message += f"**Component:** {component}\n"
        message += f"**Status:** {status.upper()}\n"
        message += f"**Details:** {details}\n"
        message += f"\n**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return self.send_alert(
            title=title,
            message=message,
            priority=priority,
            tags=["system", component.lower()]
        )

    def send_pnl_alert(self,
                       daily_pnl: float,
                       total_pnl: float,
                       drawdown: float,
                       priority: AlertPriority = AlertPriority.HIGH) -> bool:
        """Send P&L performance alert"""

        pnl_emoji = "📈" if daily_pnl >= 0 else "📉"
        title = f"{pnl_emoji} P&L Update"

        message = f"*{title}*\n\n"
        message += f"**Daily P&L:** ${daily_pnl:,.2f}\n"
        message += f"**Total P&L:** ${total_pnl:,.2f}\n"
        message += f"**Drawdown:** {drawdown:.2f}%\n"

        # Add performance context
        if daily_pnl > 1000:
            message += "\n🎯 **Excellent daily performance!**"
        elif daily_pnl < -1000:
            message += "\n⚠️ **Significant daily loss detected**"

        if drawdown > 5:
            message += f"\n🔸 **Drawdown above 5% threshold**"

        message += f"\n\n**Time:** {datetime.now().strftime('%H:%M:%S')}"

        return self.send_alert(
            title=title,
            message=message,
            priority=priority,
            tags=["pnl", "performance"]
        )

    def send_risk_alert(self,
                       alert_type: str,
                       current_value: float,
                       threshold: float,
                       action_taken: str,
                       priority: AlertPriority = AlertPriority.CRITICAL) -> bool:
        """Send risk management alert"""

        title = f"⚠️ Risk Alert: {alert_type}"

        message = f"*{title}*\n\n"
        message += f"**Alert Type:** {alert_type}\n"
        message += f"**Current Value:** {current_value:.2f}\n"
        message += f"**Threshold:** {threshold:.2f}\n"
        message += f"**Action Taken:** {action_taken}\n"
        message += f"\n**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        return self.send_alert(
            title=title,
            message=message,
            priority=priority,
            tags=["risk", "threshold"]
        )

    def send_chaos_test_alert(self,
                             test_name: str,
                             outcome: str,
                             recovery_time: float,
                             priority: AlertPriority = AlertPriority.MEDIUM) -> bool:
        """Send chaos test result alert"""

        outcome_emoji = "✅" if outcome == "passed" else "❌"
        title = f"🧪 Chaos Test: {test_name}"

        message = f"*{title}*\n\n"
        message += f"**Test Name:** {test_name}\n"
        message += f"**Outcome:** {outcome_emoji} {outcome.upper()}\n"
        message += f"**Recovery Time:** {recovery_time:.1f}s\n"

        if outcome == "passed":
            message += "\n✅ **System resilience validated**"
        else:
            message += "\n⚠️ **System resilience issue detected**"

        message += f"\n\n**Time:** {datetime.now().strftime('%H:%M:%S')}"

        return self.send_alert(
            title=title,
            message=message,
            priority=priority,
            tags=["chaos", "testing"]
        )

    def send_kill_switch_alert(self,
                              trigger_reason: str,
                              positions_closed: int,
                              priority: AlertPriority = AlertPriority.EMERGENCY) -> bool:
        """Send emergency kill switch alert"""

        title = "🚨 EMERGENCY KILL SWITCH ACTIVATED 🚨"

        message = f"*{title}*\n\n"
        message += f"**🔴 CRITICAL SYSTEM HALT 🔴**\n\n"
        message += f"**Trigger Reason:** {trigger_reason}\n"
        message += f"**Positions Closed:** {positions_closed}\n"
        message += f"**Status:** All trading STOPPED\n"
        message += f"\n**⚡ IMMEDIATE ACTION REQUIRED ⚡**"
        message += f"\n\n**Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        # Send multiple times for critical alerts
        success = False
        for _ in range(3):
            if self.send_alert(
                title=title,
                message=message,
                priority=priority,
                tags=["emergency", "kill_switch"]
            ):
                success = True
            time.sleep(1)  # 1 second between emergency messages

        return success

    def send_execution_alert(self,
                           symbol: str,
                           slippage_bps: float,
                           budget_remaining: float,
                           priority: AlertPriority = AlertPriority.MEDIUM) -> bool:
        """Send execution policy alert"""

        title = f"⚙️ Execution Alert: {symbol}"

        message = f"*{title}*\n\n"
        message += f"**Symbol:** {symbol}\n"
        message += f"**Realized Slippage:** {slippage_bps:.1f} bps\n"
        message += f"**Budget Remaining:** {budget_remaining:.1f} bps\n"

        if slippage_bps > 50:
            message += "\n⚠️ **High slippage detected**"

        if budget_remaining < 20:
            message += "\n🔸 **Slippage budget running low**"

        message += f"\n\n**Time:** {datetime.now().strftime('%H:%M:%S')}"

        return self.send_alert(
            title=title,
            message=message,
            priority=priority,
            tags=["execution", "slippage"]
        )

    def _send_telegram_message(self, alert: TelegramAlert) -> bool:
        """Send message to Telegram"""

        if not self.api_base_url or not self.chat_id:
            logger.error("Telegram not configured properly")
            return False

        # Rate limiting
        if self.burst_protection:
            current_time = time.time()
            time_since_last = current_time - self.last_message_time

            if time_since_last < self.min_message_interval:
                sleep_time = self.min_message_interval - time_since_last
                time.sleep(sleep_time)
                self.stats['rate_limited'] += 1

            self.last_message_time = time.time()

        # Format message
        formatted_message = self._format_message(alert)

        # Prepare request
        url = f"{self.api_base_url}/sendMessage"
        payload = {
            'chat_id': self.chat_id,
            'text': formatted_message,
            'parse_mode': 'Markdown' if alert.message_type == TelegramMessageType.MARKDOWN else None,
            'disable_notification': alert.disable_notification
        }

        # Send with retries
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout_seconds
                )

                if response.status_code == 200:
                    result = response.json()
                    if result.get('ok'):
                        alert.sent = True
                        self.stats['successful_deliveries'] += 1
                        self.alert_history.append(alert)

                        logger.info(f"Telegram alert sent: {alert.title}")
                        return True
                    else:
                        error_msg = result.get('description', 'Unknown error')
                        logger.error(f"Telegram API error: {error_msg}")
                        alert.error_message = error_msg
                else:
                    logger.error(f"Telegram HTTP error: {response.status_code}")
                    alert.error_message = f"HTTP {response.status_code}"

            except Exception as e:
                logger.error(f"Telegram send error (attempt {attempt + 1}): {e}")
                alert.error_message = str(e)

                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        # All attempts failed
        alert.delivery_attempts = self.max_retries
        self.stats['failed_deliveries'] += 1
        return False

    def _format_message(self, alert: TelegramAlert) -> str:
        """Format alert message for Telegram"""

        # Priority prefix
        priority_prefix = alert.priority.value

        # Build message
        if alert.message_type == TelegramMessageType.MARKDOWN:
            formatted = f"{priority_prefix} *{alert.title}*\n\n{alert.message}"
        else:
            formatted = f"{priority_prefix} {alert.title}\n\n{alert.message}"

        # Add metadata footer for high priority alerts
        if alert.priority in [AlertPriority.CRITICAL, AlertPriority.EMERGENCY]:
            formatted += f"\n\n🆔 Alert ID: `{alert.alert_id}`"

        # Add tags
        if alert.tags:
            tags_str = " ".join([f"#{tag}" for tag in alert.tags])
            formatted += f"\n\n{tags_str}"

        return formatted

    def send_test_alert(self) -> bool:
        """Send test alert to verify connection"""

        return self.send_alert(
            title="🧪 Test Alert",
            message="This is a test message from CryptoSmartTrader V2 alert system.\n\n"
                   "If you receive this message, Telegram alerts are working correctly!",
            priority=AlertPriority.LOW,
            tags=["test", "system_check"]
        )

    def send_startup_alert(self) -> bool:
        """Send system startup notification"""

        return self.send_system_alert(
            component="CryptoSmartTrader V2",
            status="healthy",
            details="System successfully started and ready for trading operations.",
            priority=AlertPriority.MEDIUM
        )

    def send_shutdown_alert(self) -> bool:
        """Send system shutdown notification"""

        return self.send_system_alert(
            component="CryptoSmartTrader V2",
            status="offline",
            details="System shutdown initiated. All trading operations stopped.",
            priority=AlertPriority.HIGH
        )

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get comprehensive alert statistics"""

        uptime = datetime.now() - self.stats['uptime_start']

        return {
            'telegram_config': {
                'configured': bool(self.bot_token and self.chat_id),
                'chat_id': self.chat_id,
                'connection_tested': self.stats['successful_deliveries'] > 0
            },

            'delivery_stats': {
                'total_alerts': self.stats['total_alerts'],
                'successful_deliveries': self.stats['successful_deliveries'],
                'failed_deliveries': self.stats['failed_deliveries'],
                'success_rate': (
                    self.stats['successful_deliveries'] /
                    max(self.stats['total_alerts'], 1) * 100
                ),
                'rate_limited_messages': self.stats['rate_limited']
            },

            'system_info': {
                'uptime_hours': uptime.total_seconds() / 3600,
                'alerts_per_hour': (
                    self.stats['total_alerts'] /
                    max(uptime.total_seconds() / 3600, 1)
                ),
                'last_alert_time': (
                    self.alert_history[-1].timestamp.isoformat()
                    if self.alert_history else None
                )
            },

            'recent_alerts': [
                {
                    'title': alert.title,
                    'priority': alert.priority.value,
                    'timestamp': alert.timestamp.isoformat(),
                    'tags': alert.tags,
                    'sent': alert.sent
                }
                for alert in self.alert_history[-10:]  # Last 10 alerts
            ]
        }

    def health_check(self) -> Dict[str, Any]:
        """Perform Telegram health check"""

        if not self.bot_token or not self.chat_id:
            return {
                'healthy': False,
                'error': 'Telegram not configured',
                'configured': False
            }

        try:
            # Test API connectivity
            response = requests.get(
                f"{self.api_base_url}/getMe",
                timeout=5
            )

            if response.status_code == 200:
                result = response.json()
                if result.get('ok'):
                    return {
                        'healthy': True,
                        'configured': True,
                        'bot_username': result['result'].get('username'),
                        'connection_time_ms': response.elapsed.total_seconds() * 1000
                    }

            return {
                'healthy': False,
                'error': f"API error: {response.status_code}",
                'configured': True
            }

        except Exception as e:
            return {
                'healthy': False,
                'error': str(e),
                'configured': True
            }
