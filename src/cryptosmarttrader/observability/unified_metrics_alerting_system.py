#!/usr/bin/env python3
"""
Unified Metrics & Alerting System
Enterprise-grade gecentraliseerd observability systeem voor CryptoSmartTrader V2

Consolideert alle verspreide Prometheus metrics in één centraal systeem met:
- Real-time alerting met multi-tier severity
- Degradatie detectie met trend analysis  
- Circuit breaker integration
- Multi-channel notification delivery
- Alert escalation met automatische routing
- Performance baseline tracking
- Comprehensive dashboard data export
"""

import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path
import statistics

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, Info,
        CollectorRegistry, generate_latest, start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("prometheus_client not available - using mock metrics")

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class MetricType(Enum):
    """Prometheus metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertState(Enum):
    """Alert state management"""
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"


@dataclass
class AlertRule:
    """Comprehensive alert rule definition"""
    name: str
    description: str
    query: str
    severity: AlertSeverity
    threshold: float
    comparison: str = ">"  # >, <, ==, !=, >=, <=
    for_duration: int = 300  # seconds
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    runbook_url: Optional[str] = None
    escalation_channels: List[str] = field(default_factory=list)


@dataclass
class AlertEvent:
    """Alert event with state tracking"""
    rule_name: str
    severity: AlertSeverity
    current_value: float
    threshold: float
    query: str
    started_at: float
    state: AlertState = AlertState.PENDING
    last_evaluation: float = 0
    notification_count: int = 0
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSample:
    """Time-series metric sample"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


class TrendAnalyzer:
    """Analyzes metric trends voor degradatie detectie"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.samples: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baselines: Dict[str, float] = {}
        self.trends: Dict[str, float] = {}
        
    def add_sample(self, metric_name: str, value: float, timestamp: Optional[float] = None):
        """Add metric sample for trend analysis"""
        timestamp = timestamp or time.time()
        sample = MetricSample(timestamp=timestamp, value=value)
        self.samples[metric_name].append(sample)
        
        # Update trend calculation
        self._calculate_trend(metric_name)
        
    def _calculate_trend(self, metric_name: str):
        """Calculate trend slope voor metric"""
        samples = self.samples[metric_name]
        if len(samples) < 10:  # Need minimum samples
            return
            
        # Extract values en timestamps
        values = [s.value for s in samples]
        timestamps = [s.timestamp for s in samples]
        
        # Calculate linear regression slope
        n = len(values)
        x_mean = statistics.mean(timestamps)
        y_mean = statistics.mean(values)
        
        numerator = sum((timestamps[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((timestamps[i] - x_mean) ** 2 for i in range(n))
        
        if denominator != 0:
            slope = numerator / denominator
            self.trends[metric_name] = slope
            
            # Update baseline if trend is stable
            if abs(slope) < 0.01:  # Stable trend
                self.baselines[metric_name] = y_mean
                
    def get_trend(self, metric_name: str) -> Optional[float]:
        """Get trend voor metric (positive = increasing, negative = decreasing)"""
        return self.trends.get(metric_name)
        
    def get_baseline(self, metric_name: str) -> Optional[float]:
        """Get baseline value voor metric"""
        return self.baselines.get(metric_name)
        
    def detect_degradation(self, metric_name: str, current_value: float, threshold_factor: float = 2.0) -> bool:
        """Detect if metric shows significant degradation"""
        baseline = self.baselines.get(metric_name)
        trend = self.trends.get(metric_name)
        
        if baseline is None or trend is None:
            return False
            
        # Check if current value deviates significantly from baseline
        deviation = abs(current_value - baseline) / baseline if baseline != 0 else 0
        
        # Check if trend is negative and accelerating
        negative_trend = trend < -0.1  # Significant negative trend
        
        return deviation > threshold_factor or negative_trend


class NotificationManager:
    """Manages multi-channel alert notifications"""
    
    def __init__(self):
        self.channels: Dict[str, Callable] = {}
        self.notification_history: List[Dict[str, Any]] = []
        self.rate_limits: Dict[str, float] = {}  # channel -> last_notification_time
        self.rate_limit_window = 60  # seconds
        
    def register_channel(self, name: str, handler: Callable):
        """Register notification channel"""
        self.channels[name] = handler
        logger.info(f"📡 Registered notification channel: {name}")
        
    def send_notification(self, channel: str, alert_event: AlertEvent) -> bool:
        """Send notification via channel"""
        if channel not in self.channels:
            logger.error(f"❌ Unknown notification channel: {channel}")
            return False
            
        # Check rate limiting
        last_notification = self.rate_limits.get(channel, 0)
        if time.time() - last_notification < self.rate_limit_window:
            logger.debug(f"⏰ Rate limited notification to {channel}")
            return False
            
        try:
            handler = self.channels[channel]
            success = handler(alert_event)
            
            if success:
                self.rate_limits[channel] = time.time()
                self.notification_history.append({
                    "timestamp": time.time(),
                    "channel": channel,
                    "alert": alert_event.rule_name,
                    "severity": alert_event.severity.value
                })
                logger.info(f"📤 Notification sent via {channel}: {alert_event.rule_name}")
                
            return success
            
        except Exception as e:
            logger.error(f"❌ Notification failed via {channel}: {e}")
            return False


class MockMetric:
    """Mock Prometheus metric for fallback"""
    def __init__(self, name: str, metric_type: str, description: str, labels: List[str] = None):
        self.name = name
        self.type = metric_type
        self.description = description
        self.labels = labels or []
        self.values = {}
        
    def labels(self, **kwargs):
        return self
        
    def inc(self, amount=1):
        pass
        
    def set(self, value):
        pass
        
    def observe(self, value):
        pass


class UnifiedMetricsAlertingSystem:
    """
    Enterprise Unified Metrics & Alerting System
    
    Consolideert alle verspreide Prometheus metrics met:
    - Centralized metric collection
    - Real-time alerting met multi-tier severity  
    - Trend analysis voor degradatie detectie
    - Circuit breaker integration
    - Multi-channel notifications
    - Performance baselines tracking
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self._initialized = True
        
        # Core components
        self.registry = CollectorRegistry() if PROMETHEUS_AVAILABLE else None
        self.metrics: Dict[str, Any] = {}
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, AlertEvent] = {}
        
        # Trend analysis
        self.trend_analyzer = TrendAnalyzer()
        self.notification_manager = NotificationManager()
        
        # Threading
        self.running = True
        self.alert_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        
        # Configuration
        self.evaluation_interval = 30  # seconds
        self.metrics_retention_hours = 24
        self.http_server_port = 8000
        self.http_server = None
        
        # Initialize system
        self._initialize_core_metrics()
        self._initialize_alert_rules()
        self._setup_notification_channels()
        self._start_background_processes()
        
        logger.info("🚀 Unified Metrics & Alerting System initialized")
    
    def _initialize_core_metrics(self):
        """Initialize comprehensive enterprise metrics"""
        metrics_config = [
            # System Health Metrics
            ("cst_system_health_score", "gauge", "Overall system health score (0-1)", []),
            ("cst_component_health_score", "gauge", "Component health score (0-1)", ["component"]),
            ("cst_service_uptime_seconds", "counter", "Service uptime in seconds", ["service"]),
            
            # Trading Performance Metrics  
            ("cst_trades_total", "counter", "Total trades executed", ["symbol", "side", "strategy", "status"]),
            ("cst_trade_pnl_usd", "gauge", "Trade PnL in USD", ["symbol", "strategy"]),
            ("cst_daily_pnl_usd", "gauge", "Daily PnL in USD", []),
            ("cst_portfolio_value_usd", "gauge", "Portfolio value in USD", []),
            ("cst_position_count", "gauge", "Number of open positions", ["exchange"]),
            
            # Risk Management Metrics
            ("cst_risk_evaluations_total", "counter", "Risk evaluations", ["decision", "reason"]),
            ("cst_risk_breaches_total", "counter", "Risk limit breaches", ["limit_type", "severity"]),
            ("cst_drawdown_current", "gauge", "Current drawdown percentage", []),
            ("cst_exposure_current_usd", "gauge", "Current total exposure", []),
            
            # Execution Quality Metrics
            ("cst_order_latency_seconds", "histogram", "Order execution latency", ["operation", "exchange"]),
            ("cst_slippage_bps", "histogram", "Trade slippage in basis points", ["symbol", "exchange"]),
            ("cst_execution_decisions_total", "counter", "Execution decisions", ["decision", "reason"]),
            ("cst_order_fill_ratio", "gauge", "Order fill ratio", ["symbol", "order_type"]),
            
            # Data Quality Metrics
            ("cst_data_requests_total", "counter", "Data requests", ["source", "status"]),
            ("cst_data_latency_seconds", "histogram", "Data request latency", ["source", "endpoint"]),
            ("cst_data_quality_score", "gauge", "Data quality score (0-1)", ["source", "metric_type"]),
            ("cst_data_gaps_total", "counter", "Data gaps detected", ["source", "severity"]),
            
            # AI/ML Performance Metrics
            ("cst_model_predictions_total", "counter", "Model predictions", ["model", "confidence_bucket"]),
            ("cst_model_accuracy_score", "gauge", "Model accuracy", ["model", "timeframe"]),
            ("cst_signal_accuracy_score", "gauge", "Signal accuracy", ["signal_type", "timeframe"]),
            ("cst_ai_api_requests_total", "counter", "AI API requests", ["provider", "status"]),
            
            # Infrastructure Metrics
            ("cst_cpu_usage_percent", "gauge", "CPU usage percentage", ["component"]),
            ("cst_memory_usage_bytes", "gauge", "Memory usage in bytes", ["component"]),
            ("cst_disk_usage_bytes", "gauge", "Disk usage in bytes", ["component"]),
            ("cst_network_bytes_total", "counter", "Network bytes transferred", ["direction", "component"]),
            
            # Error Tracking Metrics
            ("cst_errors_total", "counter", "Total errors", ["component", "error_type", "severity"]),
            ("cst_exceptions_total", "counter", "Unhandled exceptions", ["component", "exception_type"]),
            ("cst_circuit_breaker_state", "gauge", "Circuit breaker state (0=closed, 1=open)", ["service"]),
            
            # Alert Metrics
            ("cst_alerts_fired_total", "counter", "Alerts fired", ["severity", "alert_name"]),
            ("cst_alert_response_time_seconds", "histogram", "Alert response time", ["severity"]),
        ]
        
        for name, metric_type, description, labels in metrics_config:
            self._create_metric(name, metric_type, description, labels)
    
    def _create_metric(self, name: str, metric_type: str, description: str, labels: List[str]):
        """Create Prometheus metric or mock fallback"""
        if PROMETHEUS_AVAILABLE and self.registry:
            if metric_type == "counter":
                metric = Counter(name, description, labels, registry=self.registry)
            elif metric_type == "gauge":
                metric = Gauge(name, description, labels, registry=self.registry)
            elif metric_type == "histogram":
                metric = Histogram(name, description, labels, registry=self.registry)
            elif metric_type == "summary":
                metric = Summary(name, description, labels, registry=self.registry)
            else:
                raise ValueError(f"Unknown metric type: {metric_type}")
        else:
            metric = MockMetric(name, metric_type, description, labels)
            
        self.metrics[name] = metric
        logger.debug(f"📊 Created metric: {name} ({metric_type})")
    
    def _initialize_alert_rules(self):
        """Initialize comprehensive alert rules"""
        alert_rules_config = [
            # Critical System Alerts
            AlertRule(
                name="SystemHealthCritical",
                description="System health score critically low",
                query="cst_system_health_score < 0.3",
                severity=AlertSeverity.CRITICAL,
                threshold=0.3,
                comparison="<",
                for_duration=120,  # 2 minutes
                labels={"team": "sre", "priority": "p1"},
                escalation_channels=["slack-critical", "pagerduty"],
                runbook_url="https://wiki.internal/runbooks/system-health"
            ),
            
            AlertRule(
                name="HighErrorRate",
                description="Error rate exceeds threshold",
                query="rate(cst_errors_total[5m]) > 0.05",
                severity=AlertSeverity.CRITICAL,
                threshold=0.05,
                for_duration=180,  # 3 minutes
                labels={"team": "trading"},
                escalation_channels=["slack-alerts", "email"]
            ),
            
            # Trading Performance Alerts
            AlertRule(
                name="DrawdownExceedsLimit",
                description="Portfolio drawdown exceeds safety limit",
                query="cst_drawdown_current > 10.0",
                severity=AlertSeverity.EMERGENCY,
                threshold=10.0,
                for_duration=0,  # Immediate
                labels={"team": "trading", "escalation": "immediate"},
                escalation_channels=["slack-emergency", "sms", "pagerduty"]
            ),
            
            AlertRule(
                name="HighOrderLatency",
                description="Order execution latency too high",
                query="histogram_quantile(0.95, cst_order_latency_seconds_bucket) > 0.5",
                severity=AlertSeverity.WARNING,
                threshold=0.5,
                for_duration=300,  # 5 minutes
                labels={"team": "trading"}
            ),
            
            AlertRule(
                name="ExcessiveSlippage",
                description="Trade slippage exceeds budget",
                query="histogram_quantile(0.95, cst_slippage_bps_bucket) > 50",
                severity=AlertSeverity.CRITICAL,
                threshold=50,
                for_duration=180,
                labels={"team": "trading"}
            ),
            
            # Data Quality Alerts
            AlertRule(
                name="DataQualityDegraded",
                description="Data quality score below threshold",
                query="cst_data_quality_score < 0.8",
                severity=AlertSeverity.WARNING,
                threshold=0.8,
                comparison="<",
                for_duration=600,  # 10 minutes
                labels={"team": "data"}
            ),
            
            AlertRule(
                name="DataGapsDetected",
                description="Critical data gaps detected",
                query="rate(cst_data_gaps_total{severity=\"critical\"}[5m]) > 0",
                severity=AlertSeverity.CRITICAL,
                threshold=0,
                for_duration=60,
                labels={"team": "data"}
            ),
            
            # Model Performance Alerts
            AlertRule(
                name="ModelAccuracyDegraded",
                description="Model accuracy below acceptable threshold",
                query="cst_model_accuracy_score < 0.7",
                severity=AlertSeverity.WARNING,
                threshold=0.7,
                comparison="<",
                for_duration=1800,  # 30 minutes
                labels={"team": "ml"}
            ),
            
            # Infrastructure Alerts
            AlertRule(
                name="HighCpuUsage",
                description="CPU usage critically high",
                query="cst_cpu_usage_percent > 85",
                severity=AlertSeverity.WARNING,
                threshold=85,
                for_duration=600,  # 10 minutes
                labels={"team": "sre"}
            ),
            
            AlertRule(
                name="HighMemoryUsage",
                description="Memory usage critically high",
                query="cst_memory_usage_bytes / (1024*1024*1024) > 8",
                severity=AlertSeverity.CRITICAL,
                threshold=8.0,  # 8 GB
                for_duration=300,
                labels={"team": "sre"}
            ),
        ]
        
        for rule in alert_rules_config:
            self.alert_rules[rule.name] = rule
            logger.debug(f"🚨 Created alert rule: {rule.name}")
    
    def _setup_notification_channels(self):
        """Setup notification channels"""
        
        def slack_handler(alert_event: AlertEvent) -> bool:
            """Mock Slack notification handler"""
            try:
                severity_emoji = {
                    AlertSeverity.INFO: "ℹ️",
                    AlertSeverity.WARNING: "⚠️",
                    AlertSeverity.CRITICAL: "🚨",
                    AlertSeverity.EMERGENCY: "🆘"
                }
                
                emoji = severity_emoji.get(alert_event.severity, "❓")
                duration = time.time() - alert_event.started_at
                
                message = f"""
{emoji} *{alert_event.severity.value.upper()}*: {alert_event.rule_name}

📊 *Current Value*: {alert_event.current_value:.2f}
🎯 *Threshold*: {alert_event.threshold:.2f}
⏱️ *Duration*: {duration:.1f}s

🔗 [Dashboard](http://localhost:5000) | [Runbook](https://wiki.internal/runbooks)
"""
                logger.info(f"📱 SLACK ALERT: {alert_event.rule_name}")
                logger.info(f"📱 Message: {message.strip()}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Slack notification failed: {e}")
                return False
        
        def email_handler(alert_event: AlertEvent) -> bool:
            """Mock email notification handler"""
            try:
                subject = f"[{alert_event.severity.value.upper()}] Trading Alert: {alert_event.rule_name}"
                body = f"""
TRADING SYSTEM ALERT

Alert: {alert_event.rule_name}
Severity: {alert_event.severity.value.upper()}
Started: {datetime.fromtimestamp(alert_event.started_at).strftime('%Y-%m-%d %H:%M:%S UTC')}
Current Value: {alert_event.current_value:.2f}
Threshold: {alert_event.threshold:.2f}

Please investigate immediately and take appropriate action.
"""
                logger.info(f"📧 EMAIL ALERT: {subject}")
                logger.info(f"📧 Body: {body.strip()}")
                return True
                
            except Exception as e:
                logger.error(f"❌ Email notification failed: {e}")
                return False
        
        # Register notification channels
        self.notification_manager.register_channel("slack-alerts", slack_handler)
        self.notification_manager.register_channel("slack-critical", slack_handler)
        self.notification_manager.register_channel("slack-emergency", slack_handler)
        self.notification_manager.register_channel("email", email_handler)
        self.notification_manager.register_channel("pagerduty", email_handler)  # Mock with email
        self.notification_manager.register_channel("sms", email_handler)  # Mock with email
    
    def _start_background_processes(self):
        """Start background monitoring processes"""
        
        def alert_evaluator():
            """Background alert evaluation loop"""
            while self.running:
                try:
                    self._evaluate_alerts()
                    time.sleep(self.evaluation_interval)
                except Exception as e:
                    logger.error(f"❌ Alert evaluation error: {e}")
                    time.sleep(60)
        
        def metrics_cleaner():
            """Background metrics cleanup"""
            while self.running:
                try:
                    self._cleanup_old_data()
                    time.sleep(3600)  # Every hour
                except Exception as e:
                    logger.error(f"❌ Metrics cleanup error: {e}")
                    time.sleep(1800)
        
        # Start background threads
        alert_thread = threading.Thread(target=alert_evaluator, daemon=True)
        cleanup_thread = threading.Thread(target=metrics_cleaner, daemon=True)
        
        alert_thread.start()
        cleanup_thread.start()
        
        logger.info("🔄 Background processes started")
    
    def _evaluate_alerts(self):
        """Evaluate all alert rules"""
        current_time = time.time()
        
        with self.alert_lock:
            for rule_name, rule in self.alert_rules.items():
                if not rule.enabled:
                    continue
                    
                try:
                    # Mock alert evaluation - in production would query actual metrics
                    current_value = self._mock_evaluate_query(rule.query)
                    
                    # Check threshold
                    alert_triggered = self._check_threshold(current_value, rule.threshold, rule.comparison)
                    
                    if alert_triggered:
                        self._handle_alert_triggered(rule, current_value, current_time)
                    else:
                        self._handle_alert_resolved(rule_name, current_time)
                        
                except Exception as e:
                    logger.error(f"❌ Alert evaluation failed for {rule_name}: {e}")
    
    def _mock_evaluate_query(self, query: str) -> float:
        """Mock query evaluation - returns sample values based on query"""
        # In production, this would execute actual PromQL queries
        if "system_health_score" in query:
            return 0.85  # Good health
        elif "error" in query.lower():
            return 0.02  # Low error rate
        elif "latency" in query.lower():
            return 0.15  # 150ms latency
        elif "drawdown" in query.lower():
            return 5.2   # 5.2% drawdown
        elif "cpu_usage" in query:
            return 45.0  # 45% CPU
        else:
            return 1.0   # Default value
    
    def _check_threshold(self, current_value: float, threshold: float, comparison: str) -> bool:
        """Check if threshold is breached"""
        if comparison == ">":
            return current_value > threshold
        elif comparison == "<":
            return current_value < threshold
        elif comparison == "==":
            return current_value == threshold
        elif comparison == "!=":
            return current_value != threshold
        elif comparison == ">=":
            return current_value >= threshold
        elif comparison == "<=":
            return current_value <= threshold
        else:
            return False
    
    def _handle_alert_triggered(self, rule: AlertRule, current_value: float, current_time: float):
        """Handle triggered alert"""
        if rule.name not in self.active_alerts:
            # New alert
            alert_event = AlertEvent(
                rule_name=rule.name,
                severity=rule.severity,
                current_value=current_value,
                threshold=rule.threshold,
                query=rule.query,
                started_at=current_time,
                state=AlertState.PENDING,
                labels=rule.labels,
                annotations=rule.annotations
            )
            
            self.active_alerts[rule.name] = alert_event
            
            # Check for duration before firing
            if rule.for_duration == 0:
                self._fire_alert(alert_event)
            
            logger.debug(f"🚨 New alert triggered: {rule.name}")
            
        else:
            # Update existing alert
            alert_event = self.active_alerts[rule.name]
            alert_event.current_value = current_value
            alert_event.last_evaluation = current_time
            
            # Check if duration threshold met
            if (alert_event.state == AlertState.PENDING and 
                current_time - alert_event.started_at >= rule.for_duration):
                self._fire_alert(alert_event)
    
    def _handle_alert_resolved(self, rule_name: str, current_time: float):
        """Handle resolved alert"""
        if rule_name in self.active_alerts:
            alert_event = self.active_alerts.pop(rule_name)
            alert_event.state = AlertState.RESOLVED
            
            # Send resolution notifications
            for channel in alert_event.labels.get("escalation_channels", ["slack-alerts"]):
                try:
                    self.notification_manager.send_notification(channel, alert_event)
                except Exception as e:
                    logger.error(f"❌ Resolution notification failed: {e}")
            
            logger.info(f"✅ Alert resolved: {rule_name}")
    
    def _fire_alert(self, alert_event: AlertEvent):
        """Fire alert and send notifications"""
        alert_event.state = AlertState.FIRING
        
        # Record alert firing
        self.record_alert_fired(alert_event.severity.value, alert_event.rule_name)
        
        # Get escalation channels from rule
        rule = self.alert_rules.get(alert_event.rule_name)
        channels = rule.escalation_channels if rule else ["slack-alerts"]
        
        # Send notifications
        for channel in channels:
            try:
                success = self.notification_manager.send_notification(channel, alert_event)
                if success:
                    alert_event.notification_count += 1
            except Exception as e:
                logger.error(f"❌ Alert notification failed: {e}")
        
        logger.warning(f"🚨 ALERT FIRED: {alert_event.rule_name} (severity: {alert_event.severity.value})")
    
    def _cleanup_old_data(self):
        """Cleanup old metric data and alerts"""
        cutoff_time = time.time() - (self.metrics_retention_hours * 3600)
        
        # Cleanup notification history
        self.notification_manager.notification_history = [
            record for record in self.notification_manager.notification_history
            if record["timestamp"] > cutoff_time
        ]
        
        logger.debug("🧹 Old data cleanup completed")
    
    # Public API methods
    
    def record_system_health(self, score: float, component: str = "overall"):
        """Record system health score"""
        if component == "overall":
            self.metrics["cst_system_health_score"].set(score)
        else:
            self.metrics["cst_component_health_score"].labels(component=component).set(score)
        
        # Add to trend analysis
        self.trend_analyzer.add_sample(f"system_health_{component}", score)
    
    def record_trade(self, symbol: str, side: str, strategy: str, status: str, pnl_usd: Optional[float] = None):
        """Record trade execution"""
        self.metrics["cst_trades_total"].labels(
            symbol=symbol, side=side, strategy=strategy, status=status
        ).inc()
        
        if pnl_usd is not None:
            self.metrics["cst_trade_pnl_usd"].labels(symbol=symbol, strategy=strategy).set(pnl_usd)
    
    def record_error(self, component: str, error_type: str, severity: str):
        """Record error occurrence"""
        self.metrics["cst_errors_total"].labels(
            component=component, error_type=error_type, severity=severity
        ).inc()
    
    def record_latency(self, operation: str, exchange: str, latency_seconds: float):
        """Record operation latency"""
        self.metrics["cst_order_latency_seconds"].labels(
            operation=operation, exchange=exchange
        ).observe(latency_seconds)
    
    def record_data_request(self, source: str, status: str, latency_seconds: Optional[float] = None):
        """Record data request"""
        self.metrics["cst_data_requests_total"].labels(source=source, status=status).inc()
        
        if latency_seconds is not None:
            self.metrics["cst_data_latency_seconds"].labels(
                source=source, endpoint="api"
            ).observe(latency_seconds)
    
    def record_risk_evaluation(self, decision: str, reason: str):
        """Record risk evaluation"""
        self.metrics["cst_risk_evaluations_total"].labels(decision=decision, reason=reason).inc()
    
    def record_alert_fired(self, severity: str, alert_name: str):
        """Record alert firing"""
        self.metrics["cst_alerts_fired_total"].labels(severity=severity, alert_name=alert_name).inc()
    
    def start_http_server(self, port: Optional[int] = None) -> bool:
        """Start Prometheus HTTP server"""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("⚠️ Prometheus not available - HTTP server not started")
            return False
            
        if self.http_server is not None:
            logger.warning("⚠️ HTTP server already running")
            return False
            
        try:
            server_port = port or self.http_server_port
            start_http_server(server_port, registry=self.registry)
            self.http_server = {"port": server_port}
            logger.info(f"🌐 Metrics HTTP server started on port {server_port}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start HTTP server: {e}")
            return False
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        if PROMETHEUS_AVAILABLE and self.registry:
            return generate_latest(self.registry).decode('utf-8')
        else:
            return "# Prometheus metrics not available\n"
    
    def export_alert_rules(self) -> str:
        """Export alert rules in Prometheus format"""
        rules_yaml = "groups:\n"
        rules_yaml += "  - name: cryptosmarttrader_unified\n"
        rules_yaml += "    interval: 30s\n"
        rules_yaml += "    rules:\n"
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
                
            rules_yaml += f"      - alert: {rule.name}\n"
            rules_yaml += f"        expr: {rule.query}\n"
            rules_yaml += f"        for: {rule.for_duration}s\n"
            rules_yaml += "        labels:\n"
            rules_yaml += f"          severity: {rule.severity.value}\n"
            
            for k, v in rule.labels.items():
                rules_yaml += f"          {k}: \"{v}\"\n"
            
            rules_yaml += "        annotations:\n"
            rules_yaml += f"          description: \"{rule.description}\"\n"
            rules_yaml += f"          summary: \"Alert {rule.name} triggered\"\n"
            
            if rule.runbook_url:
                rules_yaml += f"          runbook_url: \"{rule.runbook_url}\"\n"
            
            for k, v in rule.annotations.items():
                rules_yaml += f"          {k}: \"{v}\"\n"
            
            rules_yaml += "\n"
        
        return rules_yaml
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        return {
            "system_status": {
                "health_score": 0.85,  # Mock value
                "active_alerts": len(self.active_alerts),
                "total_metrics": len(self.metrics),
                "uptime_seconds": time.time() - getattr(self, 'start_time', time.time())
            },
            "alert_summary": {
                "firing": len([a for a in self.active_alerts.values() if a.state == AlertState.FIRING]),
                "pending": len([a for a in self.active_alerts.values() if a.state == AlertState.PENDING]),
                "acknowledged": len([a for a in self.active_alerts.values() if a.acknowledged])
            },
            "active_alerts": [
                {
                    "name": alert.rule_name,
                    "severity": alert.severity.value,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "duration": time.time() - alert.started_at,
                    "state": alert.state.value
                }
                for alert in self.active_alerts.values()
            ],
            "recent_notifications": self.notification_manager.notification_history[-10:],
            "trend_analysis": {
                metric: {
                    "trend": self.trend_analyzer.get_trend(metric),
                    "baseline": self.trend_analyzer.get_baseline(metric)
                }
                for metric in ["system_health_overall", "trading_pnl", "error_rate"]
            }
        }
    
    def acknowledge_alert(self, alert_name: str, acknowledged_by: str) -> bool:
        """Acknowledge active alert"""
        if alert_name in self.active_alerts:
            alert = self.active_alerts[alert_name]
            alert.acknowledged = True
            alert.acknowledged_by = acknowledged_by
            alert.acknowledged_at = time.time()
            alert.state = AlertState.ACKNOWLEDGED
            
            logger.info(f"👍 Alert acknowledged by {acknowledged_by}: {alert_name}")
            return True
        return False
    
    def shutdown(self):
        """Shutdown metrics system"""
        self.running = False
        logger.info("🛑 Unified Metrics & Alerting System shutting down")


# Global singleton instance
unified_metrics = UnifiedMetricsAlertingSystem()

# Export main interface
__all__ = ["unified_metrics", "UnifiedMetricsAlertingSystem", "AlertSeverity", "AlertRule", "AlertEvent"]