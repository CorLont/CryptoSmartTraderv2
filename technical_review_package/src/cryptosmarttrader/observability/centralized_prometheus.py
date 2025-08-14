"""
Centralized Prometheus Metrics & Alerts System
All observability metrics in één module met critical alerts
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import numpy as np

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server
    )
except ImportError:
    # Fallback for environments without prometheus_client
    class MockMetric:
        def inc(self, amount=1): pass
        def dec(self, amount=1): pass
        def set(self, value): pass
        def observe(self, value): pass
        def info(self, info_dict): pass
    
    Counter = Histogram = Gauge = Summary = Info = lambda *args, **kwargs: MockMetric()
    CollectorRegistry = lambda: None
    generate_latest = lambda registry: b""
    CONTENT_TYPE_LATEST = "text/plain"
    start_http_server = lambda port, registry: None

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class AlertRule:
    """Prometheus alert rule definition"""
    name: str
    condition: str
    threshold: float
    duration: str  # "5m", "30s", etc.
    severity: AlertSeverity
    description: str
    runbook_url: Optional[str] = None


@dataclass
class AlertEvent:
    """Active alert event"""
    rule_name: str
    severity: AlertSeverity
    current_value: float
    threshold: float
    started_at: float
    description: str
    labels: Dict[str, str] = field(default_factory=dict)


class CentralizedPrometheusMetrics:
    """
    Centralized Prometheus metrics system voor complete observability
    Met critical alerts voor trading system health
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None, enable_server: bool = True, port: int = 8000):
        self.registry = registry or CollectorRegistry()
        self.logger = logging.getLogger(__name__)
        self.enable_server = enable_server
        self.port = port
        
        # Alert management
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, AlertEvent] = {}
        self.alert_history: List[AlertEvent] = []
        self.alert_lock = threading.Lock()
        
        # Metric storage for alert evaluation
        self.metric_values: Dict[str, List[Tuple[float, float]]] = {}  # metric_name -> [(timestamp, value)]
        self.metric_lock = threading.Lock()
        
        # Initialize all metrics
        self._initialize_trading_metrics()
        self._initialize_system_metrics()
        self._initialize_execution_metrics()
        self._initialize_risk_metrics()
        self._initialize_performance_metrics()
        
        # Setup alert rules
        self._setup_critical_alert_rules()
        
        # Start metrics server
        if self.enable_server:
            self._start_metrics_server()
        
        # Start alert evaluation thread
        self._start_alert_evaluator()
        
        self.logger.info(f"✅ Centralized Prometheus metrics initialized on port {port}")
    
    def _initialize_trading_metrics(self):
        """Initialize trading-specific metrics"""
        
        # Order metrics
        self.orders_total = Counter(
            'trading_orders_total',
            'Total number of orders placed',
            ['symbol', 'side', 'order_type', 'status'],
            registry=self.registry
        )
        
        self.order_errors_total = Counter(
            'trading_order_errors_total',
            'Total number of order errors',
            ['symbol', 'side', 'error_type'],
            registry=self.registry
        )
        
        self.order_execution_duration = Histogram(
            'trading_order_execution_duration_seconds',
            'Time taken to execute orders',
            ['symbol', 'side', 'order_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        # Fill metrics
        self.fills_total = Counter(
            'trading_fills_total',
            'Total number of fills',
            ['symbol', 'side', 'fill_type'],
            registry=self.registry
        )
        
        self.fill_size_usd = Histogram(
            'trading_fill_size_usd',
            'Fill size in USD',
            ['symbol', 'side'],
            buckets=[10, 100, 1000, 10000, 100000, 1000000],
            registry=self.registry
        )
        
        # Slippage metrics
        self.slippage_bps = Histogram(
            'trading_slippage_bps',
            'Slippage in basis points',
            ['symbol', 'side', 'order_type'],
            buckets=[0, 5, 10, 20, 50, 100, 200, 500],
            registry=self.registry
        )
        
        self.slippage_p95 = Gauge(
            'trading_slippage_p95_bps',
            '95th percentile slippage in basis points',
            ['symbol'],
            registry=self.registry
        )
        
        # Signal metrics
        self.signals_generated_total = Counter(
            'trading_signals_generated_total',
            'Total number of signals generated',
            ['strategy', 'symbol', 'signal_type'],
            registry=self.registry
        )
        
        self.signals_last_generated = Gauge(
            'trading_signals_last_generated_timestamp',
            'Timestamp of last signal generation',
            ['strategy'],
            registry=self.registry
        )
        
        self.signal_strength = Histogram(
            'trading_signal_strength',
            'Signal strength distribution',
            ['strategy', 'symbol'],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
    
    def _initialize_system_metrics(self):
        """Initialize system health metrics"""
        
        # System health
        self.system_health_score = Gauge(
            'system_health_score',
            'Overall system health score (0-100)',
            registry=self.registry
        )
        
        self.system_uptime_seconds = Counter(
            'system_uptime_seconds_total',
            'Total system uptime in seconds',
            registry=self.registry
        )
        
        # Data pipeline metrics
        self.data_updates_total = Counter(
            'data_updates_total',
            'Total number of data updates',
            ['source', 'symbol'],
            registry=self.registry
        )
        
        self.data_update_latency = Histogram(
            'data_update_latency_seconds',
            'Data update latency',
            ['source', 'symbol'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry
        )
        
        self.data_gaps_total = Counter(
            'data_gaps_total',
            'Total number of data gaps detected',
            ['source', 'symbol'],
            registry=self.registry
        )
        
        # Exchange connectivity
        self.exchange_connections = Gauge(
            'exchange_connections_active',
            'Number of active exchange connections',
            ['exchange'],
            registry=self.registry
        )
        
        self.exchange_errors_total = Counter(
            'exchange_errors_total',
            'Total exchange errors',
            ['exchange', 'error_type'],
            registry=self.registry
        )
        
        self.exchange_latency = Histogram(
            'exchange_latency_seconds',
            'Exchange API latency',
            ['exchange', 'endpoint'],
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            registry=self.registry
        )
    
    def _initialize_execution_metrics(self):
        """Initialize execution quality metrics"""
        
        # Execution quality
        self.execution_quality_score = Histogram(
            'execution_quality_score',
            'Execution quality score (0-100)',
            ['symbol', 'side'],
            buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            registry=self.registry
        )
        
        self.partial_fills_ratio = Gauge(
            'execution_partial_fills_ratio',
            'Ratio of partial fills to total orders',
            ['symbol'],
            registry=self.registry
        )
        
        self.rejection_rate = Gauge(
            'execution_rejection_rate',
            'Order rejection rate',
            ['symbol', 'exchange'],
            registry=self.registry
        )
        
        # Execution costs
        self.execution_fees_usd = Counter(
            'execution_fees_usd_total',
            'Total execution fees paid in USD',
            ['symbol', 'fee_type'],
            registry=self.registry
        )
        
        self.execution_cost_bps = Histogram(
            'execution_cost_bps',
            'Total execution cost in basis points',
            ['symbol', 'side'],
            buckets=[0, 5, 10, 15, 20, 30, 50, 100],
            registry=self.registry
        )
    
    def _initialize_risk_metrics(self):
        """Initialize risk management metrics"""
        
        # Portfolio risk
        self.portfolio_value_usd = Gauge(
            'portfolio_value_usd',
            'Total portfolio value in USD',
            registry=self.registry
        )
        
        self.portfolio_drawdown_pct = Gauge(
            'portfolio_drawdown_percent',
            'Current portfolio drawdown percentage',
            registry=self.registry
        )
        
        self.daily_pnl_usd = Gauge(
            'portfolio_daily_pnl_usd',
            'Daily PnL in USD',
            registry=self.registry
        )
        
        # Risk limits
        self.risk_limit_violations_total = Counter(
            'risk_limit_violations_total',
            'Total risk limit violations',
            ['limit_type', 'severity'],
            registry=self.registry
        )
        
        self.position_sizes_usd = Gauge(
            'position_sizes_usd',
            'Position sizes in USD',
            ['symbol'],
            registry=self.registry
        )
        
        self.exposure_utilization_pct = Gauge(
            'exposure_utilization_percent',
            'Exposure utilization percentage',
            ['limit_type'],
            registry=self.registry
        )
        
        # Kill switch
        self.kill_switch_active = Gauge(
            'kill_switch_active',
            'Kill switch status (1=active, 0=inactive)',
            registry=self.registry
        )
        
        self.kill_switch_triggers_total = Counter(
            'kill_switch_triggers_total',
            'Total kill switch triggers',
            ['trigger_reason'],
            registry=self.registry
        )
    
    def _initialize_performance_metrics(self):
        """Initialize performance tracking metrics"""
        
        # Returns
        self.strategy_returns_pct = Gauge(
            'strategy_returns_percent',
            'Strategy returns percentage',
            ['strategy', 'timeframe'],
            registry=self.registry
        )
        
        self.sharpe_ratio = Gauge(
            'strategy_sharpe_ratio',
            'Strategy Sharpe ratio',
            ['strategy', 'timeframe'],
            registry=self.registry
        )
        
        self.win_rate = Gauge(
            'strategy_win_rate',
            'Strategy win rate',
            ['strategy'],
            registry=self.registry
        )
        
        # Backtest parity
        self.tracking_error_bps = Gauge(
            'backtest_live_tracking_error_bps',
            'Backtest-live tracking error in basis points',
            registry=self.registry
        )
        
        self.parity_status = Gauge(
            'backtest_live_parity_status',
            'Parity status (0=healthy, 1=warning, 2=critical, 3=disabled)',
            registry=self.registry
        )
        
        # System performance
        self.cpu_usage_percent = Gauge(
            'system_cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage_mb = Gauge(
            'system_memory_usage_mb',
            'Memory usage in MB',
            registry=self.registry
        )
        
        self.disk_usage_percent = Gauge(
            'system_disk_usage_percent',
            'Disk usage percentage',
            registry=self.registry
        )
    
    def _setup_critical_alert_rules(self):
        """Setup critical alert rules"""
        
        # High order error rate alert
        self.add_alert_rule(AlertRule(
            name="HighOrderErrorRate",
            condition="rate(trading_order_errors_total[5m]) > 0.1",
            threshold=0.1,
            duration="5m",
            severity=AlertSeverity.CRITICAL,
            description="Order error rate too high (>10% over 5 minutes)",
            runbook_url="https://wiki.internal/runbooks/high-order-error-rate"
        ))
        
        # Drawdown too high alert
        self.add_alert_rule(AlertRule(
            name="DrawdownTooHigh",
            condition="portfolio_drawdown_percent > 10",
            threshold=10.0,
            duration="1m",
            severity=AlertSeverity.EMERGENCY,
            description="Portfolio drawdown exceeds 10%",
            runbook_url="https://wiki.internal/runbooks/high-drawdown"
        ))
        
        # No signals alert
        self.add_alert_rule(AlertRule(
            name="NoSignals30m",
            condition="time() - trading_signals_last_generated_timestamp > 1800",
            threshold=1800.0,  # 30 minutes
            duration="5m",
            severity=AlertSeverity.WARNING,
            description="No trading signals generated in the last 30 minutes",
            runbook_url="https://wiki.internal/runbooks/no-signals"
        ))
        
        # Slippage budget exceeded alert
        self.add_alert_rule(AlertRule(
            name="SlippageP95ExceedsBudget",
            condition="trading_slippage_p95_bps > 50",
            threshold=50.0,
            duration="10m",
            severity=AlertSeverity.CRITICAL,
            description="95th percentile slippage exceeds 50 bps budget",
            runbook_url="https://wiki.internal/runbooks/high-slippage"
        ))
        
        # System health alerts
        self.add_alert_rule(AlertRule(
            name="SystemHealthLow",
            condition="system_health_score < 70",
            threshold=70.0,
            duration="5m",
            severity=AlertSeverity.WARNING,
            description="System health score below 70",
            runbook_url="https://wiki.internal/runbooks/system-health"
        ))
        
        # Exchange connectivity alerts
        self.add_alert_rule(AlertRule(
            name="ExchangeDisconnected",
            condition="exchange_connections_active < 1",
            threshold=1.0,
            duration="30s",
            severity=AlertSeverity.CRITICAL,
            description="Exchange connection lost",
            runbook_url="https://wiki.internal/runbooks/exchange-connectivity"
        ))
        
        # Data gap alerts
        self.add_alert_rule(AlertRule(
            name="DataGapDetected",
            condition="increase(data_gaps_total[5m]) > 0",
            threshold=0.0,
            duration="1m",
            severity=AlertSeverity.WARNING,
            description="Data gaps detected in market data feed",
            runbook_url="https://wiki.internal/runbooks/data-gaps"
        ))
        
        # Kill switch alerts
        self.add_alert_rule(AlertRule(
            name="KillSwitchActivated",
            condition="kill_switch_active == 1",
            threshold=1.0,
            duration="0s",
            severity=AlertSeverity.EMERGENCY,
            description="Kill switch has been activated",
            runbook_url="https://wiki.internal/runbooks/kill-switch"
        ))
    
    def add_alert_rule(self, rule: AlertRule):
        """Add custom alert rule"""
        with self.alert_lock:
            self.alert_rules[rule.name] = rule
            self.logger.info(f"📊 Added alert rule: {rule.name}")
    
    def remove_alert_rule(self, rule_name: str):
        """Remove alert rule"""
        with self.alert_lock:
            if rule_name in self.alert_rules:
                del self.alert_rules[rule_name]
                # Clear any active alerts for this rule
                if rule_name in self.active_alerts:
                    del self.active_alerts[rule_name]
                self.logger.info(f"📊 Removed alert rule: {rule_name}")
    
    def _record_metric_value(self, metric_name: str, value: float):
        """Record metric value for alert evaluation"""
        current_time = time.time()
        
        with self.metric_lock:
            if metric_name not in self.metric_values:
                self.metric_values[metric_name] = []
            
            self.metric_values[metric_name].append((current_time, value))
            
            # Keep only last hour of data
            cutoff_time = current_time - 3600
            self.metric_values[metric_name] = [
                (ts, val) for ts, val in self.metric_values[metric_name]
                if ts > cutoff_time
            ]
    
    def _start_metrics_server(self):
        """Start Prometheus metrics HTTP server"""
        try:
            start_http_server(self.port, registry=self.registry)
            self.logger.info(f"🌐 Prometheus metrics server started on port {self.port}")
        except Exception as e:
            self.logger.error(f"❌ Failed to start metrics server: {e}")
    
    def _start_alert_evaluator(self):
        """Start alert evaluation thread"""
        def evaluate_alerts():
            while True:
                try:
                    self._evaluate_all_alerts()
                    time.sleep(30)  # Evaluate every 30 seconds
                except Exception as e:
                    self.logger.error(f"❌ Alert evaluation error: {e}")
                    time.sleep(60)  # Wait longer on error
        
        alert_thread = threading.Thread(target=evaluate_alerts, daemon=True)
        alert_thread.start()
        self.logger.info("🚨 Alert evaluator started")
    
    def _evaluate_all_alerts(self):
        """Evaluate all alert rules"""
        current_time = time.time()
        
        with self.alert_lock:
            for rule_name, rule in self.alert_rules.items():
                try:
                    should_fire = self._evaluate_alert_condition(rule, current_time)
                    
                    if should_fire and rule_name not in self.active_alerts:
                        # Fire new alert
                        alert_event = AlertEvent(
                            rule_name=rule_name,
                            severity=rule.severity,
                            current_value=should_fire,  # Value that triggered alert
                            threshold=rule.threshold,
                            started_at=current_time,
                            description=rule.description
                        )
                        
                        self.active_alerts[rule_name] = alert_event
                        self.alert_history.append(alert_event)
                        
                        self._fire_alert(alert_event)
                    
                    elif not should_fire and rule_name in self.active_alerts:
                        # Resolve alert
                        resolved_alert = self.active_alerts.pop(rule_name)
                        self._resolve_alert(resolved_alert)
                
                except Exception as e:
                    self.logger.error(f"❌ Error evaluating alert {rule_name}: {e}")
    
    def _evaluate_alert_condition(self, rule: AlertRule, current_time: float) -> Optional[float]:
        """Evaluate specific alert condition"""
        
        # Simple condition evaluation - in production, you'd use a proper query engine
        condition = rule.condition
        
        if "rate(trading_order_errors_total[5m])" in condition:
            # Calculate error rate over last 5 minutes
            error_data = self.metric_values.get("trading_order_errors_total", [])
            recent_errors = [v for t, v in error_data if current_time - t <= 300]  # 5 minutes
            if len(recent_errors) >= 2:
                error_rate = (recent_errors[-1] - recent_errors[0]) / 300  # per second
                if error_rate > rule.threshold:
                    return error_rate
        
        elif "portfolio_drawdown_percent" in condition:
            # Check current drawdown
            drawdown_data = self.metric_values.get("portfolio_drawdown_percent", [])
            if drawdown_data:
                current_drawdown = drawdown_data[-1][1]
                if current_drawdown > rule.threshold:
                    return current_drawdown
        
        elif "trading_signals_last_generated_timestamp" in condition:
            # Check time since last signal
            signal_data = self.metric_values.get("trading_signals_last_generated_timestamp", [])
            if signal_data:
                last_signal_time = signal_data[-1][1]
                time_since_signal = current_time - last_signal_time
                if time_since_signal > rule.threshold:
                    return time_since_signal
        
        elif "trading_slippage_p95_bps" in condition:
            # Check P95 slippage
            slippage_data = self.metric_values.get("trading_slippage_p95_bps", [])
            if slippage_data:
                current_slippage = slippage_data[-1][1]
                if current_slippage > rule.threshold:
                    return current_slippage
        
        # Add more condition evaluations as needed
        
        return None
    
    def _fire_alert(self, alert: AlertEvent):
        """Fire alert notification"""
        self.logger.critical(
            f"🚨 ALERT FIRED: {alert.rule_name} - {alert.description} "
            f"(current: {alert.current_value:.2f}, threshold: {alert.threshold:.2f})"
        )
        
        # Here you would integrate with your alerting system
        # e.g., PagerDuty, Slack, email, etc.
    
    def _resolve_alert(self, alert: AlertEvent):
        """Resolve alert notification"""
        duration = time.time() - alert.started_at
        self.logger.info(
            f"✅ ALERT RESOLVED: {alert.rule_name} after {duration:.1f} seconds"
        )
    
    # Convenience methods for recording metrics
    def record_order(self, symbol: str, side: str, order_type: str, status: str):
        """Record order placement"""
        self.orders_total.labels(symbol=symbol, side=side, order_type=order_type, status=status).inc()
    
    def record_order_error(self, symbol: str, side: str, error_type: str):
        """Record order error"""
        self.order_errors_total.labels(symbol=symbol, side=side, error_type=error_type).inc()
        self._record_metric_value("trading_order_errors_total", time.time())
    
    def record_execution_duration(self, symbol: str, side: str, order_type: str, duration: float):
        """Record order execution duration"""
        self.order_execution_duration.labels(symbol=symbol, side=side, order_type=order_type).observe(duration)
    
    def record_fill(self, symbol: str, side: str, fill_type: str, size_usd: float):
        """Record trade fill"""
        self.fills_total.labels(symbol=symbol, side=side, fill_type=fill_type).inc()
        self.fill_size_usd.labels(symbol=symbol, side=side).observe(size_usd)
    
    def record_slippage(self, symbol: str, side: str, order_type: str, slippage_bps: float):
        """Record slippage"""
        self.slippage_bps.labels(symbol=symbol, side=side, order_type=order_type).observe(slippage_bps)
        
        # Update P95 slippage for alert evaluation
        slippage_data = self.metric_values.get("trading_slippage_bps", [])
        if len(slippage_data) >= 20:  # Need sufficient data for P95
            recent_slippages = [v for t, v in slippage_data[-100:]]  # Last 100 data points
            p95_slippage = np.percentile(recent_slippages, 95)
            self.slippage_p95.labels(symbol=symbol).set(p95_slippage)
            self._record_metric_value("trading_slippage_p95_bps", p95_slippage)
    
    def record_signal(self, strategy: str, symbol: str, signal_type: str, strength: float):
        """Record trading signal"""
        self.signals_generated_total.labels(strategy=strategy, symbol=symbol, signal_type=signal_type).inc()
        self.signals_last_generated.labels(strategy=strategy).set(time.time())
        self.signal_strength.labels(strategy=strategy, symbol=symbol).observe(strength)
        self._record_metric_value("trading_signals_last_generated_timestamp", time.time())
    
    def update_portfolio_metrics(self, portfolio_value: float, drawdown_pct: float, daily_pnl: float):
        """Update portfolio metrics"""
        self.portfolio_value_usd.set(portfolio_value)
        self.portfolio_drawdown_pct.set(drawdown_pct)
        self.daily_pnl_usd.set(daily_pnl)
        self._record_metric_value("portfolio_drawdown_percent", drawdown_pct)
    
    def update_position_size(self, symbol: str, size_usd: float):
        """Update position size"""
        self.position_sizes_usd.labels(symbol=symbol).set(size_usd)
    
    def record_risk_violation(self, limit_type: str, severity: str):
        """Record risk limit violation"""
        self.risk_limit_violations_total.labels(limit_type=limit_type, severity=severity).inc()
    
    def update_kill_switch(self, active: bool, trigger_reason: Optional[str] = None):
        """Update kill switch status"""
        self.kill_switch_active.set(1 if active else 0)
        if active and trigger_reason:
            self.kill_switch_triggers_total.labels(trigger_reason=trigger_reason).inc()
    
    def update_tracking_error(self, tracking_error_bps: float, parity_status: int):
        """Update backtest-live parity metrics"""
        self.tracking_error_bps.set(tracking_error_bps)
        self.parity_status.set(parity_status)
    
    def record_exchange_error(self, exchange: str, error_type: str):
        """Record exchange error"""
        self.exchange_errors_total.labels(exchange=exchange, error_type=error_type).inc()
    
    def update_exchange_connection(self, exchange: str, connected: bool):
        """Update exchange connection status"""
        self.exchange_connections.labels(exchange=exchange).set(1 if connected else 0)
    
    def record_data_gap(self, source: str, symbol: str):
        """Record data gap"""
        self.data_gaps_total.labels(source=source, symbol=symbol).inc()
    
    def get_metrics_output(self) -> bytes:
        """Get Prometheus metrics output"""
        return generate_latest(self.registry)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        with self.alert_lock:
            return {
                "active_alerts": len(self.active_alerts),
                "total_rules": len(self.alert_rules),
                "alert_history_count": len(self.alert_history),
                "active_alert_details": [
                    {
                        "name": alert.rule_name,
                        "severity": alert.severity.value,
                        "duration": time.time() - alert.started_at,
                        "description": alert.description
                    }
                    for alert in self.active_alerts.values()
                ],
                "recent_alerts": [
                    {
                        "name": alert.rule_name,
                        "severity": alert.severity.value,
                        "started_at": alert.started_at,
                        "description": alert.description
                    }
                    for alert in self.alert_history[-10:]  # Last 10 alerts
                ]
            }


# Global centralized metrics instance
_global_prometheus_metrics: Optional[CentralizedPrometheusMetrics] = None


def get_global_prometheus_metrics() -> CentralizedPrometheusMetrics:
    """Get or create global Prometheus metrics instance"""
    global _global_prometheus_metrics
    if _global_prometheus_metrics is None:
        _global_prometheus_metrics = CentralizedPrometheusMetrics()
        logger.info("✅ Global CentralizedPrometheusMetrics initialized")
    return _global_prometheus_metrics


def reset_global_prometheus_metrics():
    """Reset global Prometheus metrics (for testing)"""
    global _global_prometheus_metrics
    _global_prometheus_metrics = None