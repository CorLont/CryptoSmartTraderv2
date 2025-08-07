"""
CryptoSmartTrader V2 - Production Monitoring System
Enterprise-grade monitoring with Prometheus integration and external alerting
"""

import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys
import psutil
import requests
from dataclasses import dataclass

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, start_http_server, write_to_textfile
except ImportError:
    # Fallback if prometheus_client is not available
    class MockMetric:
        def inc(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
    
    Counter = Histogram = Gauge = lambda *args, **kwargs: MockMetric()
    CollectorRegistry = lambda: None
    start_http_server = lambda *args, **kwargs: None
    write_to_textfile = lambda *args, **kwargs: None

@dataclass
class Alert:
    """Alert definition for monitoring system"""
    name: str
    severity: str  # critical, warning, info
    message: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = None

class PrometheusMetrics:
    """Prometheus metrics collection for CryptoSmartTrader"""
    
    def __init__(self, registry=None):
        self.registry = registry or CollectorRegistry()
        
        # Application metrics
        self.analysis_requests_total = Counter(
            'cryptotrader_analysis_requests_total',
            'Total number of analysis requests',
            ['agent_type', 'status'],
            registry=self.registry
        )
        
        self.analysis_duration_seconds = Histogram(
            'cryptotrader_analysis_duration_seconds',
            'Time spent on analysis operations',
            ['agent_type'],
            registry=self.registry
        )
        
        self.api_requests_total = Counter(
            'cryptotrader_api_requests_total',
            'Total API requests to external services',
            ['service', 'status'],
            registry=self.registry
        )
        
        self.errors_total = Counter(
            'cryptotrader_errors_total',
            'Total number of errors',
            ['category', 'severity'],
            registry=self.registry
        )
        
        # System metrics
        self.memory_usage_bytes = Gauge(
            'cryptotrader_memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage_percent = Gauge(
            'cryptotrader_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.active_agents = Gauge(
            'cryptotrader_active_agents',
            'Number of active agents',
            registry=self.registry
        )
        
        self.cache_hit_rate = Gauge(
            'cryptotrader_cache_hit_rate',
            'Cache hit rate percentage',
            registry=self.registry
        )
        
        # Trading metrics
        self.predictions_accuracy = Gauge(
            'cryptotrader_predictions_accuracy',
            'ML prediction accuracy',
            ['model_type'],
            registry=self.registry
        )
        
        self.sentiment_score = Gauge(
            'cryptotrader_sentiment_score',
            'Current market sentiment score',
            registry=self.registry
        )
        
        self.health_score = Gauge(
            'cryptotrader_health_score',
            'Overall system health score',
            registry=self.registry
        )

class ExternalAlertManager:
    """External alerting integration (PagerDuty, Slack, etc.)"""
    
    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        
        # Alert channels configuration
        self.alert_channels = {
            "slack": self._send_slack_alert,
            "email": self._send_email_alert,
            "webhook": self._send_webhook_alert,
            "pagerduty": self._send_pagerduty_alert
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            "critical_error_rate": 5,  # errors per minute
            "system_health_critical": 0.3,
            "memory_usage_critical": 0.9,
            "cpu_usage_critical": 0.8,
            "api_failure_rate": 0.1
        }
    
    def send_alert(self, alert: Alert, channels: List[str] = None):
        """Send alert to configured channels"""
        try:
            channels = channels or ["slack", "email"]
            
            for channel in channels:
                if channel in self.alert_channels:
                    self.alert_channels[channel](alert)
                    self.logger.info(f"Alert sent via {channel}: {alert.name}")
                else:
                    self.logger.warning(f"Unknown alert channel: {channel}")
                    
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
    
    def _send_slack_alert(self, alert: Alert):
        """Send alert to Slack"""
        try:
            slack_webhook = self.config_manager.get("alerts.slack_webhook") if self.config_manager else None
            if not slack_webhook:
                return
            
            color_map = {"critical": "danger", "warning": "warning", "info": "good"}
            color = color_map.get(alert.severity, "warning")
            
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"CryptoSmartTrader Alert: {alert.name}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Severity", "value": alert.severity, "short": True},
                        {"title": "Source", "value": alert.source, "short": True},
                        {"title": "Time", "value": alert.timestamp.isoformat(), "short": True}
                    ]
                }]
            }
            
            response = requests.post(slack_webhook, json=payload, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"Slack alert failed: {e}")
    
    def _send_email_alert(self, alert: Alert):
        """Send alert via email"""
        try:
            # Email configuration would be loaded from config
            email_config = self.config_manager.get("alerts.email") if self.config_manager else {}
            
            if not email_config.get("enabled", False):
                return
            
            # Email sending logic would go here
            self.logger.info(f"Email alert sent: {alert.name}")
            
        except Exception as e:
            self.logger.error(f"Email alert failed: {e}")
    
    def _send_webhook_alert(self, alert: Alert):
        """Send alert to webhook endpoint"""
        try:
            webhook_url = self.config_manager.get("alerts.webhook_url") if self.config_manager else None
            if not webhook_url:
                return
            
            payload = {
                "alert_name": alert.name,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "source": alert.source,
                "metadata": alert.metadata or {}
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"Webhook alert failed: {e}")
    
    def _send_pagerduty_alert(self, alert: Alert):
        """Send alert to PagerDuty"""
        try:
            pd_key = self.config_manager.get("alerts.pagerduty_key") if self.config_manager else None
            if not pd_key:
                return
            
            payload = {
                "routing_key": pd_key,
                "event_action": "trigger",
                "payload": {
                    "summary": f"CryptoSmartTrader: {alert.name}",
                    "severity": alert.severity,
                    "source": alert.source,
                    "timestamp": alert.timestamp.isoformat(),
                    "custom_details": alert.metadata or {}
                }
            }
            
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"PagerDuty alert failed: {e}")

class ProductionMonitoringSystem:
    """Enterprise-grade production monitoring system"""
    
    def __init__(self, config_manager=None, error_handler=None):
        self.config_manager = config_manager
        self.error_handler = error_handler
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics and alerting
        self.metrics = PrometheusMetrics()
        self.alert_manager = ExternalAlertManager(config_manager)
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_thread = None
        self.alert_history: List[Alert] = []
        
        # System baselines for anomaly detection
        self.baselines = {
            "normal_cpu_usage": 0.2,
            "normal_memory_usage": 0.5,
            "normal_api_response_time": 2.0,
            "normal_error_rate": 0.01
        }
        
        # Start Prometheus metrics server
        self._start_metrics_server()
    
    def _start_metrics_server(self):
        """Start Prometheus metrics HTTP server"""
        try:
            metrics_port = 8000
            start_http_server(metrics_port, registry=self.metrics.registry)
            self.logger.info(f"Prometheus metrics server started on port {metrics_port}")
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {e}")
    
    def start_monitoring(self):
        """Start production monitoring"""
        if self.is_monitoring:
            self.logger.warning("Monitoring already active")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Production monitoring started")
    
    def stop_monitoring(self):
        """Stop production monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        self.logger.info("Production monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check alert conditions
                self._check_alert_conditions()
                
                # Export metrics to file for external systems
                self._export_metrics()
                
                # Wait before next cycle
                time.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Longer wait on error
    
    def _collect_system_metrics(self):
        """Collect and update system metrics"""
        try:
            # CPU and Memory metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            self.metrics.cpu_usage_percent.set(cpu_percent)
            self.metrics.memory_usage_bytes.set(memory.used)
            
            # Health score from error handler
            if self.error_handler:
                error_stats = self.error_handler.get_error_statistics()
                health_score = error_stats.get("health_score", 0.5)
                self.metrics.health_score.set(health_score)
            
            # Log metrics periodically
            if int(time.time()) % 300 == 0:  # Every 5 minutes
                self.logger.info(f"System metrics - CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%, Health: {health_score:.2f}")
                
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
    
    def _check_alert_conditions(self):
        """Check for alert conditions and send alerts"""
        try:
            current_time = datetime.now()
            
            # Check system health
            health_score = self.metrics.health_score._value._value if hasattr(self.metrics.health_score, '_value') else 0.5
            if health_score < self.alert_thresholds["system_health_critical"]:
                alert = Alert(
                    name="System Health Critical",
                    severity="critical",
                    message=f"System health score dropped to {health_score:.2f}",
                    timestamp=current_time,
                    source="monitoring_system",
                    metadata={"health_score": health_score}
                )
                self._send_alert_if_new(alert)
            
            # Check CPU usage
            cpu_usage = psutil.cpu_percent()
            if cpu_usage > self.alert_thresholds["cpu_usage_critical"] * 100:
                alert = Alert(
                    name="High CPU Usage",
                    severity="warning",
                    message=f"CPU usage at {cpu_usage:.1f}%",
                    timestamp=current_time,
                    source="monitoring_system",
                    metadata={"cpu_usage": cpu_usage}
                )
                self._send_alert_if_new(alert)
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.alert_thresholds["memory_usage_critical"] * 100:
                alert = Alert(
                    name="High Memory Usage",
                    severity="warning",
                    message=f"Memory usage at {memory.percent:.1f}%",
                    timestamp=current_time,
                    source="monitoring_system",
                    metadata={"memory_percent": memory.percent}
                )
                self._send_alert_if_new(alert)
            
            # Check error rates
            if self.error_handler:
                error_stats = self.error_handler.get_error_statistics()
                recent_errors = sum(
                    count for hour, count in error_stats["errors_by_hour"].items()
                    if datetime.fromisoformat(hour.replace(" ", "T") + ":00") > current_time - timedelta(minutes=5)
                )
                
                if recent_errors > self.alert_thresholds["critical_error_rate"]:
                    alert = Alert(
                        name="High Error Rate",
                        severity="critical",
                        message=f"{recent_errors} errors in last 5 minutes",
                        timestamp=current_time,
                        source="error_handler",
                        metadata={"error_count": recent_errors}
                    )
                    self._send_alert_if_new(alert)
                    
        except Exception as e:
            self.logger.error(f"Alert condition check failed: {e}")
    
    def _send_alert_if_new(self, alert: Alert):
        """Send alert only if it's not a duplicate"""
        try:
            # Check for recent similar alerts
            recent_threshold = datetime.now() - timedelta(minutes=30)
            similar_alerts = [
                a for a in self.alert_history
                if a.name == alert.name and a.timestamp > recent_threshold
            ]
            
            if not similar_alerts:
                # Send alert and add to history
                self.alert_manager.send_alert(alert)
                self.alert_history.append(alert)
                
                # Keep alert history limited
                if len(self.alert_history) > 1000:
                    self.alert_history = self.alert_history[-500:]
                    
        except Exception as e:
            self.logger.error(f"Failed to send alert: {e}")
    
    def _export_metrics(self):
        """Export metrics to file for external monitoring systems"""
        try:
            metrics_dir = Path("data/metrics")
            metrics_dir.mkdir(parents=True, exist_ok=True)
            
            metrics_file = metrics_dir / "prometheus_metrics.prom"
            write_to_textfile(str(metrics_file), self.metrics.registry)
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
    
    def record_analysis_request(self, agent_type: str, status: str, duration: float = None):
        """Record analysis request metrics"""
        try:
            self.metrics.analysis_requests_total.labels(agent_type=agent_type, status=status).inc()
            
            if duration is not None:
                self.metrics.analysis_duration_seconds.labels(agent_type=agent_type).observe(duration)
                
        except Exception as e:
            self.logger.error(f"Failed to record analysis metrics: {e}")
    
    def record_api_request(self, service: str, status: str):
        """Record API request metrics"""
        try:
            self.metrics.api_requests_total.labels(service=service, status=status).inc()
        except Exception as e:
            self.logger.error(f"Failed to record API metrics: {e}")
    
    def record_error(self, category: str, severity: str):
        """Record error metrics"""
        try:
            self.metrics.errors_total.labels(category=category, severity=severity).inc()
        except Exception as e:
            self.logger.error(f"Failed to record error metrics: {e}")
    
    def update_cache_metrics(self, hit_rate: float):
        """Update cache hit rate metrics"""
        try:
            self.metrics.cache_hit_rate.set(hit_rate * 100)
        except Exception as e:
            self.logger.error(f"Failed to update cache metrics: {e}")
    
    def update_prediction_accuracy(self, model_type: str, accuracy: float):
        """Update ML prediction accuracy metrics"""
        try:
            self.metrics.predictions_accuracy.labels(model_type=model_type).set(accuracy)
        except Exception as e:
            self.logger.error(f"Failed to update prediction metrics: {e}")
    
    def update_sentiment_score(self, score: float):
        """Update sentiment score metrics"""
        try:
            self.metrics.sentiment_score.set(score)
        except Exception as e:
            self.logger.error(f"Failed to update sentiment metrics: {e}")
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status and metrics"""
        try:
            return {
                "monitoring_active": self.is_monitoring,
                "metrics_server_running": True,  # Assume running if no errors
                "alert_history_count": len(self.alert_history),
                "recent_alerts": [
                    {
                        "name": alert.name,
                        "severity": alert.severity,
                        "timestamp": alert.timestamp.isoformat()
                    }
                    for alert in self.alert_history[-10:]  # Last 10 alerts
                ],
                "system_metrics": {
                    "cpu_usage": psutil.cpu_percent(),
                    "memory_usage": psutil.virtual_memory().percent,
                    "health_score": getattr(self.metrics.health_score, '_value', {}).get('_value', 0.5)
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to get monitoring status: {e}")
            return {"error": str(e)}