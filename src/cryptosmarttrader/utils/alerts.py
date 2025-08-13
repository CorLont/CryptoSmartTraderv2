import threading
import time
import smtplib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from pathlib import Path
import logging

class AlertsManager:
    """Multi-channel alerts and notifications system"""

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # Alert storage
        self.alerts = []
        self.alert_rules = {}
        self.alert_history = {}
        self._lock = threading.Lock()

        # Alert channels
        self.alert_channels = {
            'console': self._send_console_alert,
            'email': self._send_email_alert,
            'file': self._send_file_alert,
            'webhook': self._send_webhook_alert
        }

        # Alert processing
        self.processing_active = False
        self.process_thread = None

        # Alert files path
        self.alerts_path = Path("alerts")
        self.alerts_path.mkdir(exist_ok=True)

        # Initialize default rules
        self._setup_default_rules()

        # Start alert processing
        self.start_processing()

    def _setup_default_rules(self):
        """Setup default alert rules"""
        self.alert_rules = {
            'system_health_critical': {
                'condition': lambda data: data.get('health_score', 100) < 30,
                'severity': 'critical',
                'channels': ['console', 'file', 'email'],
                'cooldown_minutes': 30,
                'description': 'System health critically low'
            },
            'system_health_warning': {
                'condition': lambda data: 30 <= data.get('health_score', 100) < 60,
                'severity': 'warning',
                'channels': ['console', 'file'],
                'cooldown_minutes': 60,
                'description': 'System health degraded'
            },
            'agent_failure': {
                'condition': lambda data: data.get('status') == 'failed',
                'severity': 'error',
                'channels': ['console', 'file', 'email'],
                'cooldown_minutes': 15,
                'description': 'Agent failure detected'
            },
            'high_error_rate': {
                'condition': lambda data: data.get('error_rate', 0) > 0.1,
                'severity': 'warning',
                'channels': ['console', 'file'],
                'cooldown_minutes': 45,
                'description': 'High error rate detected'
            },
            'data_quality_issue': {
                'condition': lambda data: data.get('data_coverage', 100) < 70,
                'severity': 'warning',
                'channels': ['console', 'file'],
                'cooldown_minutes': 30,
                'description': 'Data quality below threshold'
            },
            'whale_activity_critical': {
                'condition': lambda data: data.get('whale_score', 0) > 0.8,
                'severity': 'info',
                'channels': ['console', 'file'],
                'cooldown_minutes': 5,
                'description': 'Critical whale activity detected'
            },
            'api_failure': {
                'condition': lambda data: data.get('api_status') == 'failed',
                'severity': 'error',
                'channels': ['console', 'file'],
                'cooldown_minutes': 20,
                'description': 'API service failure'
            },
            'memory_usage_high': {
                'condition': lambda data: data.get('memory_percent', 0) > 90,
                'severity': 'warning',
                'channels': ['console', 'file'],
                'cooldown_minutes': 30,
                'description': 'High memory usage detected'
            },
            'disk_space_low': {
                'condition': lambda data: data.get('disk_percent', 0) > 90,
                'severity': 'critical',
                'channels': ['console', 'file', 'email'],
                'cooldown_minutes': 60,
                'description': 'Low disk space warning'
            }
        }

    def start_processing(self):
        """Start alert processing thread"""
        if not self.processing_active:
            self.processing_active = True
            self.process_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.process_thread.start()
            self.logger.info("Alerts processing started")

    def stop_processing(self):
        """Stop alert processing"""
        self.processing_active = False
        if self.process_thread:
            self.process_thread.join(timeout=5)
        self.logger.info("Alerts processing stopped")

    def _processing_loop(self):
        """Main alert processing loop"""
        while self.processing_active:
            try:
                # Process pending alerts
                self._process_pending_alerts()

                # Clean up old alerts
                self._cleanup_old_alerts()

                # Sleep for processing interval
                time.sleep(10)  # Process every 10 seconds

            except Exception as e:
                self.logger.error(f"Alert processing error: {str(e)}")
                time.sleep(30)

    def _process_pending_alerts(self):
        """Process alerts that are ready to be sent"""
        with self._lock:
            current_time = datetime.now()

            for alert in list(self.alerts):
                if alert.get('status') == 'pending':
                    try:
                        # Check if alert is ready (not in cooldown)
                        if self._is_alert_ready(alert, current_time):
                            self._send_alert(alert)
                            alert['status'] = 'sent'
                            alert['sent_at'] = current_time.isoformat()

                    except Exception as e:
                        self.logger.error(f"Error processing alert {alert.get('id')}: {str(e)}")
                        alert['status'] = 'failed'
                        alert['error'] = str(e)

    def _is_alert_ready(self, alert: Dict[str, Any], current_time: datetime) -> bool:
        """Check if alert is ready to be sent (not in cooldown)"""
        rule_name = alert.get('rule_name')
        if not rule_name or rule_name not in self.alert_rules:
            return True

        cooldown_minutes = self.alert_rules[rule_name].get('cooldown_minutes', 0)
        if cooldown_minutes <= 0:
            return True

        # Check last alert of same rule
        last_sent = self.alert_history.get(rule_name, {}).get('last_sent')
        if not last_sent:
            return True

        try:
            last_sent_time = datetime.fromisoformat(last_sent)
            time_diff = (current_time - last_sent_time).total_seconds() / 60
            return time_diff >= cooldown_minutes
        except Exception:
            return True

    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert through configured channels"""
        rule_name = alert.get('rule_name')
        channels = alert.get('channels', ['console'])

        for channel in channels:
            try:
                if channel in self.alert_channels:
                    self.alert_channels[channel](alert)
                    self.logger.debug(f"Alert sent via {channel}: {alert.get('title')}")

            except Exception as e:
                self.logger.error(f"Error sending alert via {channel}: {str(e)}")

        # Update alert history
        if rule_name:
            if rule_name not in self.alert_history:
                self.alert_history[rule_name] = {}

            self.alert_history[rule_name]['last_sent'] = datetime.now().isoformat()
            self.alert_history[rule_name]['count'] = self.alert_history[rule_name].get('count', 0) + 1

    def _send_console_alert(self, alert: Dict[str, Any]):
        """Send alert to console/log"""
        severity = alert.get('severity', 'info')
        title = alert.get('title', 'Alert')
        message = alert.get('message', '')

        log_message = f"🚨 ALERT [{severity.upper()}]: {title} - {message}"

        if severity == 'critical':
            self.logger.critical(log_message)
        elif severity == 'error':
            self.logger.error(log_message)
        elif severity == 'warning':
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def _send_email_alert(self, alert: Dict[str, Any]):
        """Send alert via email"""
        try:
            # Get email configuration
            smtp_server = os.getenv("SMTP_SERVER", "")
            smtp_port = int(os.getenv("SMTP_PORT", "587"))
            smtp_username = os.getenv("SMTP_USERNAME", "")
            smtp_password = os.getenv("SMTP_PASSWORD", "")
            from_email = os.getenv("ALERT_FROM_EMAIL", smtp_username)
            to_emails = os.getenv("ALERT_TO_EMAILS", "").split(",")

            if not all([smtp_server, smtp_username, smtp_password]) or not to_emails:
                self.logger.warning("Email configuration incomplete, skipping email alert")
                return

            # Create message
            msg = MimeMultipart()
            msg['From'] = from_email
            msg['To'] = ", ".join(to_emails)
            msg['Subject'] = f"CryptoSmartTrader Alert: {alert.get('title', 'System Alert')}"

            # Email body
            body = self._format_email_body(alert)
            msg.attach(MimeText(body, 'html'))

            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(smtp_username, smtp_password)
                server.send_message(msg)

            self.logger.info(f"Email alert sent: {alert.get('title')}")

        except Exception as e:
            self.logger.error(f"Email alert failed: {str(e)}")

    def _format_email_body(self, alert: Dict[str, Any]) -> str:
        """Format alert for email body"""
        severity = alert.get('severity', 'info')
        title = alert.get('title', 'Alert')
        message = alert.get('message', '')
        timestamp = alert.get('created_at', datetime.now().isoformat())
        data = alert.get('data', {})

        severity_colors = {
            'critical': '#dc3545',
            'error': '#fd7e14',
            'warning': '#ffc107',
            'info': '#17a2b8'
        }

        color = severity_colors.get(severity, '#17a2b8')

        html_body = f"""
        <html>
        <body style="font-family: Arial, sans-serif;">
            <div style="border-left: 5px solid {color}; padding: 20px; margin: 20px 0;">
                <h2 style="color: {color}; margin-top: 0;">
                    🚨 {title}
                </h2>
                <p><strong>Severity:</strong> {severity.upper()}</p>
                <p><strong>Time:</strong> {timestamp}</p>
                <p><strong>Message:</strong> {message}</p>

                {self._format_alert_data_html(data)}

                <hr style="margin: 20px 0;">
                <p style="font-size: 12px; color: #666;">
                    This alert was generated by CryptoSmartTrader V2 monitoring system.
                </p>
            </div>
        </body>
        </html>
        """

        return html_body

    def _format_alert_data_html(self, data: Dict[str, Any]) -> str:
        """Format alert data for HTML display"""
        if not data:
            return ""

        html = "<div style='margin: 15px 0;'><strong>Additional Data:</strong><ul>"

        for key, value in data.items():
            if isinstance(value, dict):
                html += f"<li><strong>{key}:</strong> {json.dumps(value, indent=2)}</li>"
            else:
                html += f"<li><strong>{key}:</strong> {value}</li>"

        html += "</ul></div>"
        return html

    def _send_file_alert(self, alert: Dict[str, Any]):
        """Send alert to file"""
        try:
            alert_file = self.alerts_path / "alerts.log"

            with open(alert_file, 'a', encoding='utf-8') as f:
                alert_line = {
                    'timestamp': alert.get('created_at', datetime.now().isoformat()),
                    'severity': alert.get('severity'),
                    'title': alert.get('title'),
                    'message': alert.get('message'),
                    'data': alert.get('data', {})
                }

                f.write(json.dumps(alert_line) + '\n')

        except Exception as e:
            self.logger.error(f"File alert failed: {str(e)}")

    def _send_webhook_alert(self, alert: Dict[str, Any]):
        """Send alert via webhook"""
        try:
            import requests

            webhook_url = os.getenv("ALERT_WEBHOOK_URL", "")
            if not webhook_url:
                return

            payload = {
                'alert_type': 'cryptosmarttrader',
                'severity': alert.get('severity'),
                'title': alert.get('title'),
                'message': alert.get('message'),
                'timestamp': alert.get('created_at'),
                'data': alert.get('data', {})
            }

            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()

            self.logger.info(f"Webhook alert sent: {alert.get('title')}")

        except Exception as e:
            self.logger.error(f"Webhook alert failed: {str(e)}")

    def _cleanup_old_alerts(self):
        """Clean up old alerts from memory"""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=24)

            # Remove old alerts
            self.alerts = [
                alert for alert in self.alerts
                if datetime.fromisoformat(alert.get('created_at', '1970-01-01')) > cutoff_time
            ]

    def create_alert(self, rule_name: str, title: str, message: str,
                    data: Dict[str, Any] = None, severity: str = None,
                    channels: List[str] = None) -> str:
        """Create a new alert"""
        alert_id = f"alert_{int(time.time() * 1000)}"

        # Get rule configuration
        rule_config = self.alert_rules.get(rule_name, {})

        alert = {
            'id': alert_id,
            'rule_name': rule_name,
            'title': title,
            'message': message,
            'severity': severity or rule_config.get('severity', 'info'),
            'channels': channels or rule_config.get('channels', ['console']),
            'data': data or {},
            'created_at': datetime.now().isoformat(),
            'status': 'pending'
        }

        with self._lock:
            self.alerts.append(alert)

        self.logger.debug(f"Alert created: {alert_id} - {title}")
        return alert_id

    def trigger_rule_check(self, rule_name: str, data: Dict[str, Any]):
        """Check if a rule condition is met and trigger alert if needed"""
        if rule_name not in self.alert_rules:
            return False

        rule = self.alert_rules[rule_name]
        condition = rule.get('condition')

        if not condition or not callable(condition):
            return False

        try:
            if condition(data):
                # Rule condition met, create alert
                title = f"{rule.get('description', rule_name)}"
                message = self._format_rule_message(rule_name, data)

                self.create_alert(
                    rule_name=rule_name,
                    title=title,
                    message=message,
                    data=data,
                    severity=rule.get('severity'),
                    channels=rule.get('channels')
                )

                return True

        except Exception as e:
            self.logger.error(f"Error checking rule {rule_name}: {str(e)}")

        return False

    def _format_rule_message(self, rule_name: str, data: Dict[str, Any]) -> str:
        """Format message for rule-based alerts"""
        messages = {
            'system_health_critical': f"System health score: {data.get('health_score', 0):.1f}%",
            'system_health_warning': f"System health score: {data.get('health_score', 0):.1f}%",
            'agent_failure': f"Agent {data.get('agent_name', 'unknown')} has failed",
            'high_error_rate': f"Error rate: {data.get('error_rate', 0):.1%}",
            'data_quality_issue': f"Data coverage: {data.get('data_coverage', 0):.1f}%",
            'whale_activity_critical': f"Whale score: {data.get('whale_score', 0):.2f} on {data.get('symbol', 'unknown')}",
            'api_failure': f"API {data.get('api_name', 'unknown')} is not responding",
            'memory_usage_high': f"Memory usage: {data.get('memory_percent', 0):.1f}%",
            'disk_space_low': f"Disk usage: {data.get('disk_percent', 0):.1f}%"
        }

        return messages.get(rule_name, f"Rule {rule_name} triggered")

    def add_custom_rule(self, rule_name: str, condition: Callable[[Dict], bool],
                       severity: str = 'info', channels: List[str] = None,
                       cooldown_minutes: int = 30, description: str = None):
        """Add custom alert rule"""
        self.alert_rules[rule_name] = {
            'condition': condition,
            'severity': severity,
            'channels': channels or ['console'],
            'cooldown_minutes': cooldown_minutes,
            'description': description or rule_name
        }

        self.logger.info(f"Custom alert rule added: {rule_name}")

    def get_alerts(self, hours: int = 24, severity: str = None,
                  status: str = None) -> List[Dict[str, Any]]:
        """Get alerts with filters"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        with self._lock:
            filtered_alerts = []

            for alert in self.alerts:
                try:
                    alert_time = datetime.fromisoformat(alert.get('created_at', '1970-01-01'))
                    if alert_time > cutoff_time:
                        if severity and alert.get('severity') != severity:
                            continue
                        if status and alert.get('status') != status:
                            continue
                        filtered_alerts.append(alert.copy())
                except Exception:
                    continue

            return sorted(filtered_alerts, key=lambda x: x.get('created_at', ''), reverse=True)

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert system statistics"""
        with self._lock:
            total_alerts = len(self.alerts)

            # Count by severity
            severity_counts = {}
            status_counts = {}

            for alert in self.alerts:
                severity = alert.get('severity', 'unknown')
                status = alert.get('status', 'unknown')

                severity_counts[severity] = severity_counts.get(severity, 0) + 1
                status_counts[status] = status_counts.get(status, 0) + 1

            # Recent alerts (last 24 hours)
            recent_alerts = self.get_alerts(hours=24)

            return {
                'total_alerts': total_alerts,
                'recent_alerts_24h': len(recent_alerts),
                'severity_distribution': severity_counts,
                'status_distribution': status_counts,
                'active_rules': len(self.alert_rules),
                'processing_active': self.processing_active,
                'alert_history_entries': len(self.alert_history)
            }

    def test_alert_channels(self) -> Dict[str, bool]:
        """Test all alert channels"""
        test_results = {}

        test_alert = {
            'id': 'test_alert',
            'title': 'Test Alert',
            'message': 'This is a test alert to verify channel functionality',
            'severity': 'info',
            'created_at': datetime.now().isoformat(),
            'data': {'test': True}
        }

        for channel_name, channel_func in self.alert_channels.items():
            try:
                channel_func(test_alert)
                test_results[channel_name] = True
                self.logger.info(f"Alert channel {channel_name} test: PASS")
            except Exception as e:
                test_results[channel_name] = False
                self.logger.error(f"Alert channel {channel_name} test: FAIL - {str(e)}")

        return test_results

    def get_rule_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all alert rules"""
        rule_status = {}

        for rule_name, rule_config in self.alert_rules.items():
            history = self.alert_history.get(rule_name, {})

            rule_status[rule_name] = {
                'description': rule_config.get('description', rule_name),
                'severity': rule_config.get('severity', 'info'),
                'channels': rule_config.get('channels', []),
                'cooldown_minutes': rule_config.get('cooldown_minutes', 0),
                'last_triggered': history.get('last_sent'),
                'trigger_count': history.get('count', 0)
            }

        return rule_status

    def disable_rule(self, rule_name: str):
        """Disable an alert rule"""
        if rule_name in self.alert_rules:
            self.alert_rules[rule_name]['disabled'] = True
            self.logger.info(f"Alert rule disabled: {rule_name}")

    def enable_rule(self, rule_name: str):
        """Enable an alert rule"""
        if rule_name in self.alert_rules:
            self.alert_rules[rule_name].pop('disabled', None)
            self.logger.info(f"Alert rule enabled: {rule_name}")
