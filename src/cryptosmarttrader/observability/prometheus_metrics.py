"""
Prometheus Metrics System

Enterprise observability with comprehensive metrics collection,
Prometheus integration, and 24/7 monitoring capabilities.
"""

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
    start_http_server
)
import time
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import psutil
from collections import defaultdict

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric_name: str
    condition: str      # e.g., "> 0.5", "< 100", "== 0"
    threshold: float
    duration_seconds: int = 60
    severity: str = "warning"  # warning, critical, emergency
    description: str = ""
    enabled: bool = True

    # State tracking
    triggered_at: Optional[datetime] = None
    notification_sent: bool = False


class PrometheusMetricsSystem:
    """
    Enterprise Prometheus metrics collection system
    """

    def __init__(self,
                 metrics_port: int = 8000,
                 registry: Optional[CollectorRegistry] = None):

        self.registry = registry or CollectorRegistry()
        self.metrics_port = metrics_port

        # Metric storage
        self.counters: Dict[str, Counter] = {}
        self.gauges: Dict[str, Gauge] = {}
        self.histograms: Dict[str, Histogram] = {}
        self.summaries: Dict[str, Summary] = {}
        self.info_metrics: Dict[str, Info] = {}

        # Alert system
        self.alert_rules: Dict[str, AlertRule] = {}
        self.alert_callbacks: List[Callable] = []

        # Monitoring
        self.monitoring_active = True
        self._setup_core_metrics()
        self._start_metrics_server()
        self._start_system_monitoring()
        self._start_alert_monitoring()

    def _setup_core_metrics(self):
        """Setup core trading system metrics"""

        # Trading metrics
        self.register_counter(
            "trades_total",
            "Total number of trades executed",
            ["symbol", "side", "status"]
        )

        self.register_counter(
            "orders_total",
            "Total number of orders placed",
            ["symbol", "order_type", "status"]
        )

        self.register_histogram(
            "trade_execution_time_seconds",
            "Time taken to execute trades",
            ["symbol", "order_type"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
        )

        self.register_histogram(
            "order_slippage",
            "Order execution slippage in basis points",
            ["symbol"],
            buckets=[1, 5, 10, 25, 50, 100, 250]
        )

        # PnL metrics
        self.register_gauge(
            "portfolio_value_usd",
            "Current portfolio value in USD"
        )

        self.register_gauge(
            "daily_pnl_pct",
            "Daily PnL percentage"
        )

        self.register_gauge(
            "max_drawdown_pct",
            "Maximum drawdown percentage"
        )

        self.register_counter(
            "pnl_total_usd",
            "Cumulative PnL in USD",
            ["strategy"]
        )

        # Risk metrics
        self.register_gauge(
            "position_count",
            "Current number of open positions"
        )

        self.register_gauge(
            "total_exposure_usd",
            "Total portfolio exposure in USD"
        )

        self.register_gauge(
            "risk_utilization_pct",
            "Risk limit utilization percentage",
            ["limit_type"]
        )

        self.register_counter(
            "risk_limit_breaches_total",
            "Total risk limit breaches",
            ["limit_type", "severity"]
        )

        # System metrics
        self.register_gauge(
            "data_latency_seconds",
            "Market data latency in seconds",
            ["exchange", "symbol"]
        )

        self.register_counter(
            "api_requests_total",
            "Total API requests",
            ["exchange", "endpoint", "status"]
        )

        self.register_histogram(
            "api_response_time_seconds",
            "API response time distribution",
            ["exchange", "endpoint"]
        )

        self.register_counter(
            "errors_total",
            "Total system errors",
            ["component", "error_type"]
        )

        # Model performance metrics
        self.register_gauge(
            "model_accuracy_score",
            "Model prediction accuracy score",
            ["model_name", "timeframe"]
        )

        self.register_gauge(
            "model_drift_score",
            "Model performance drift score",
            ["model_name"]
        )

        self.register_counter(
            "predictions_total",
            "Total number of predictions made",
            ["model_name", "confidence_level"]
        )

        # System health metrics
        self.register_gauge(
            "cpu_usage_percent",
            "CPU usage percentage"
        )

        self.register_gauge(
            "memory_usage_percent",
            "Memory usage percentage"
        )

        self.register_gauge(
            "disk_usage_percent",
            "Disk usage percentage"
        )

        self.register_gauge(
            "network_io_bytes",
            "Network I/O bytes",
            ["direction"]  # sent/received
        )

    def register_counter(self, name: str, description: str, labels: Optional[List[str]] = None):
        """Register a counter metric"""

        counter = Counter(
            name,
            description,
            labelnames=labels or [],
            registry=self.registry
        )
        self.counters[name] = counter
        logger.debug(f"Registered counter: {name}")

    def register_gauge(self, name: str, description: str, labels: Optional[List[str]] = None):
        """Register a gauge metric"""

        gauge = Gauge(
            name,
            description,
            labelnames=labels or [],
            registry=self.registry
        )
        self.gauges[name] = gauge
        logger.debug(f"Registered gauge: {name}")

    def register_histogram(self, name: str, description: str, labels: Optional[List[str]] = None,
                          buckets: Optional[List[float]] = None):
        """Register a histogram metric"""

        kwargs = {
            "name": name,
            "documentation": description,
            "labelnames": labels or [],
            "registry": self.registry
        }

        if buckets is not None:
            kwargs["buckets"] = buckets

        histogram = Histogram(**kwargs)
        self.histograms[name] = histogram
        logger.debug(f"Registered histogram: {name}")

    def register_summary(self, name: str, description: str, labels: Optional[List[str]] = None):
        """Register a summary metric"""

        summary = Summary(
            name,
            description,
            labelnames=labels or [],
            registry=self.registry
        )
        self.summaries[name] = summary
        logger.debug(f"Registered summary: {name}")

    def register_info(self, name: str, description: str, labels: Optional[List[str]] = None):
        """Register an info metric"""

        info = Info(
            name,
            description,
            labelnames=labels or [],
            registry=self.registry
        )
        self.info_metrics[name] = info
        logger.debug(f"Registered info: {name}")

    # Metric update methods
    def increment_counter(self, name: str, amount: float = 1, labels: Optional[Dict[str, str]] = None):
        """Increment a counter"""

        if name in self.counters:
            if labels:
                self.counters[name].labels(**labels).inc(amount)
            else:
                self.counters[name].inc(amount)

    def set_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge value"""

        if name in self.gauges:
            if labels:
                self.gauges[name].labels(**labels).set(value)
            else:
                self.gauges[name].set(value)

    def observe_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a histogram value"""

        if name in self.histograms:
            if labels:
                self.histograms[name].labels(**labels).observe(value)
            else:
                self.histograms[name].observe(value)

    def observe_summary(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Observe a summary value"""

        if name in self.summaries:
            if labels:
                self.summaries[name].labels(**labels).observe(value)
            else:
                self.summaries[name].observe(value)

    def set_info(self, name: str, info: Dict[str, str], labels: Optional[Dict[str, str]] = None):
        """Set info metric"""

        if name in self.info_metrics:
            if labels:
                self.info_metrics[name].labels(**labels).info(info)
            else:
                self.info_metrics[name].info(info)

    # High-level metric recording methods
    def record_trade(self, symbol: str, side: str, status: str, execution_time: float, slippage: float):
        """Record trade execution metrics"""

        self.increment_counter("trades_total", labels={
            "symbol": symbol,
            "side": side,
            "status": status
        })

        self.observe_histogram("trade_execution_time_seconds", execution_time, labels={
            "symbol": symbol,
            "order_type": "market"
        })

        self.observe_histogram("order_slippage", slippage, labels={
            "symbol": symbol
        })

    def record_order(self, symbol: str, order_type: str, status: str):
        """Record order metrics"""

        self.increment_counter("orders_total", labels={
            "symbol": symbol,
            "order_type": order_type,
            "status": status
        })

    def update_portfolio_metrics(self, portfolio_value: float, daily_pnl_pct: float,
                                max_drawdown_pct: float, position_count: int,
                                total_exposure: float):
        """Update portfolio-related metrics"""

        self.set_gauge("portfolio_value_usd", portfolio_value)
        self.set_gauge("daily_pnl_pct", daily_pnl_pct)
        self.set_gauge("max_drawdown_pct", max_drawdown_pct)
        self.set_gauge("position_count", position_count)
        self.set_gauge("total_exposure_usd", total_exposure)

    def record_api_request(self, exchange: str, endpoint: str, status: str, response_time: float):
        """Record API request metrics"""

        self.increment_counter("api_requests_total", labels={
            "exchange": exchange,
            "endpoint": endpoint,
            "status": status
        })

        self.observe_histogram("api_response_time_seconds", response_time, labels={
            "exchange": exchange,
            "endpoint": endpoint
        })

    def record_error(self, component: str, error_type: str):
        """Record error metrics"""

        self.increment_counter("errors_total", labels={
            "component": component,
            "error_type": error_type
        })

    def record_model_prediction(self, model_name: str, confidence_level: str,
                               accuracy_score: float, drift_score: float):
        """Record ML model metrics"""

        self.increment_counter("predictions_total", labels={
            "model_name": model_name,
            "confidence_level": confidence_level
        })

        self.set_gauge("model_accuracy_score", accuracy_score, labels={
            "model_name": model_name,
            "timeframe": "1h"
        })

        self.set_gauge("model_drift_score", drift_score, labels={
            "model_name": model_name
        })

    def record_risk_limit_breach(self, limit_type: str, severity: str, utilization_pct: float):
        """Record risk management metrics"""

        self.increment_counter("risk_limit_breaches_total", labels={
            "limit_type": limit_type,
            "severity": severity
        })

        self.set_gauge("risk_utilization_pct", utilization_pct, labels={
            "limit_type": limit_type
        })

    def update_data_latency(self, exchange: str, symbol: str, latency_seconds: float):
        """Update data latency metrics"""

        self.set_gauge("data_latency_seconds", latency_seconds, labels={
            "exchange": exchange,
            "symbol": symbol
        })

    def _start_metrics_server(self):
        """Start Prometheus metrics HTTP server"""

        try:
            start_http_server(self.metrics_port, registry=self.registry)
            logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")

    def _start_system_monitoring(self):
        """Start system resource monitoring"""

        def monitor_system_resources():
            while self.monitoring_active:
                try:
                    # CPU usage
                    cpu_percent = psutil.cpu_percent(interval=1)
                    self.set_gauge("cpu_usage_percent", cpu_percent)

                    # Memory usage
                    memory = psutil.virtual_memory()
                    self.set_gauge("memory_usage_percent", memory.percent)

                    # Disk usage
                    disk = psutil.disk_usage('/')
                    disk_percent = (disk.used / disk.total) * 100
                    self.set_gauge("disk_usage_percent", disk_percent)

                    # Network I/O
                    network = psutil.net_io_counters()
                    self.set_gauge("network_io_bytes", network.bytes_sent, {"direction": "sent"})
                    self.set_gauge("network_io_bytes", network.bytes_recv, {"direction": "received"})

                    time.sleep(10)  # Update every 10 seconds

                except Exception as e:
                    logger.error(f"System monitoring error: {e}")
                    time.sleep(30)  # Longer sleep on error

        monitor_thread = threading.Thread(target=monitor_system_resources, daemon=True)
        monitor_thread.start()
        logger.info("System resource monitoring started")

    def add_alert_rule(self, rule: AlertRule):
        """Add alert rule"""

        self.alert_rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")

    def _start_alert_monitoring(self):
        """Start alert monitoring"""

        def monitor_alerts():
            while self.monitoring_active:
                try:
                    for rule in self.alert_rules.values():
                        if rule.enabled:
                            self._check_alert_rule(rule)

                    time.sleep(30)  # Check alerts every 30 seconds

                except Exception as e:
                    logger.error(f"Alert monitoring error: {e}")
                    time.sleep(60)  # Longer sleep on error

        alert_thread = threading.Thread(target=monitor_alerts, daemon=True)
        alert_thread.start()
        logger.info("Alert monitoring started")

    def _check_alert_rule(self, rule: AlertRule):
        """Check individual alert rule"""

        try:
            # Get current metric value
            current_value = self._get_metric_value(rule.metric_name)
            if current_value is None:
                return

            # Evaluate condition
            condition_met = self._evaluate_condition(current_value, rule.condition, rule.threshold)

            if condition_met:
                if rule.triggered_at is None:
                    rule.triggered_at = datetime.now()

                # Check if alert should fire
                time_since_trigger = datetime.now() - rule.triggered_at
                if (time_since_trigger.total_seconds() >= rule.duration_seconds and
                    not rule.notification_sent):

                    self._fire_alert(rule, current_value)
                    rule.notification_sent = True

            else:
                # Reset alert state
                if rule.triggered_at is not None:
                    rule.triggered_at = None
                    rule.notification_sent = False

        except Exception as e:
            logger.error(f"Alert rule check failed for {rule.name}: {e}")

    def _get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get current value of a metric"""

        # Try gauges first
        if metric_name in self.gauges:
            try:
                # Get the metric value (this is a simplification)
                return self.gauges[metric_name]._value._value
            except:
                return None

        # Could extend to support other metric types
        return None

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""

        if condition.startswith(">"):
            return value > threshold
        elif condition.startswith("<"):
            return value < threshold
        elif condition.startswith(">="):
            return value >= threshold
        elif condition.startswith("<="):
            return value <= threshold
        elif condition.startswith("=="):
            return abs(value - threshold) < 0.0001  # Float equality with tolerance
        elif condition.startswith("!="):
            return abs(value - threshold) >= 0.0001

        return False

    def _fire_alert(self, rule: AlertRule, current_value: float):
        """Fire an alert"""

        alert_data = {
            "rule_name": rule.name,
            "metric_name": rule.metric_name,
            "current_value": current_value,
            "threshold": rule.threshold,
            "condition": rule.condition,
            "severity": rule.severity,
            "description": rule.description,
            "triggered_at": rule.triggered_at.isoformat() if rule.triggered_at else None
        }

        logger.warning(f"ALERT FIRED: {rule.name} - {rule.description}")
        logger.warning(f"Current value: {current_value}, Threshold: {rule.threshold}")

        # Notify alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def add_alert_callback(self, callback: Callable):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)

    def setup_default_alerts(self):
        """Setup default alert rules"""

        # High error rate alert
        self.add_alert_rule(AlertRule(
            name="high_error_rate",
            metric_name="errors_total",
            condition="> 10",
            threshold=10.0,
            duration_seconds=300,  # 5 minutes
            severity="critical",
            description="High system error rate detected"
        ))

        # High CPU usage alert
        self.add_alert_rule(AlertRule(
            name="high_cpu_usage",
            metric_name="cpu_usage_percent",
            condition="> 80",
            threshold=80.0,
            duration_seconds=600,  # 10 minutes
            severity="warning",
            description="High CPU usage detected"
        ))

        # High memory usage alert
        self.add_alert_rule(AlertRule(
            name="high_memory_usage",
            metric_name="memory_usage_percent",
            condition="> 85",
            threshold=85.0,
            duration_seconds=300,  # 5 minutes
            severity="warning",
            description="High memory usage detected"
        ))

        # Daily loss limit alert
        self.add_alert_rule(AlertRule(
            name="daily_loss_limit",
            metric_name="daily_pnl_pct",
            condition="< -3",
            threshold=-3.0,
            duration_seconds=60,  # 1 minute
            severity="critical",
            description="Daily loss limit approaching"
        ))

        # Max drawdown alert
        self.add_alert_rule(AlertRule(
            name="max_drawdown_alert",
            metric_name="max_drawdown_pct",
            condition="< -8",
            threshold=-8.0,
            duration_seconds=60,  # 1 minute
            severity="critical",
            description="Maximum drawdown threshold exceeded"
        ))

        logger.info("Default alert rules configured")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all metrics"""

        return {
            "counters": len(self.counters),
            "gauges": len(self.gauges),
            "histograms": len(self.histograms),
            "summaries": len(self.summaries),
            "info_metrics": len(self.info_metrics),
            "alert_rules": len(self.alert_rules),
            "active_alerts": len([r for r in self.alert_rules.values() if r.triggered_at is not None]),
            "metrics_port": self.metrics_port,
            "monitoring_active": self.monitoring_active
        }

    def shutdown(self):
        """Shutdown metrics system"""
        self.monitoring_active = False
        logger.info("Prometheus metrics system shutdown")
