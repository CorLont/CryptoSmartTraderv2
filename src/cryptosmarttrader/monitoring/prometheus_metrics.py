"""Prometheus metrics collection for CryptoSmartTrader V2."""

"""
DEPRECATED: This module has been superseded by centralized observability.
Use: from cryptosmarttrader.observability.metrics import get_metrics

This file is kept for backward compatibility only.
"""

# Redirect to centralized metrics
from cryptosmarttrader.observability.metrics import get_metrics

# Legacy compatibility - delegate to centralized metrics
def get_legacy_metrics():
    """Legacy compatibility function"""
    return get_metrics()

# Re-export centralized metrics for backward compatibility
metrics = get_metrics()



import time
from typing import Dict, Any, Optional
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Enum as PrometheusEnum,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from ..core.structured_logger import get_logger


class CryptoSmartTraderMetrics:
    """Comprehensive Prometheus metrics for trading system."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics with optional custom registry."""
        self.logger = get_logger("prometheus_metrics")
        self.registry = registry or CollectorRegistry()

        # Trading metrics
        self.orders_total = Counter(
            "cst_orders_total",
            "Total number of orders placed",
            ["exchange", "symbol", "side", "status"],
            registry=self.registry,
        )

        self.order_execution_time = Histogram(
            "cst_order_execution_seconds",
            "Order execution time in seconds",
            ["exchange", "symbol"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry,
        )

        self.slippage_percent = Histogram(
            "cst_slippage_percent",
            "Trading slippage percentage",
            ["exchange", "symbol"],
            buckets=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0],
            registry=self.registry,
        )

        self.position_pnl = Gauge(
            "cst_position_pnl_usd",
            "Current position PnL in USD",
            ["symbol"],
            registry=self.registry,
        )

        # Risk metrics
        self.portfolio_value = Gauge(
            "cst_portfolio_value_usd", "Total portfolio value in USD", registry=self.registry
        )

        self.daily_pnl_percent = Gauge(
            "cst_daily_pnl_percent", "Daily PnL percentage", registry=self.registry
        )

        self.max_drawdown_percent = Gauge(
            "cst_max_drawdown_percent", "Maximum drawdown percentage", registry=self.registry
        )

        self.risk_level = PrometheusEnum(
            "cst_risk_level",
            "Current risk level",
            states=["normal", "conservative", "defensive", "emergency", "shutdown"],
            registry=self.registry,
        )

        self.kill_switch_active = Gauge(
            "cst_kill_switch_active",
            "Kill switch status (1=active, 0=inactive)",
            registry=self.registry,
        )

        # Data quality metrics
        self.data_source_last_update = Gauge(
            "cst_data_source_last_update_timestamp",
            "Last update timestamp for data sources",
            ["source"],
            registry=self.registry,
        )

        self.data_quality_score = Gauge(
            "cst_data_quality_score", "Data quality score (0-1)", ["source"], registry=self.registry
        )

        self.api_requests_total = Counter(
            "cst_api_requests_total",
            "Total API requests",
            ["exchange", "endpoint", "status"],
            registry=self.registry,
        )

        self.api_request_duration = Histogram(
            "cst_api_request_duration_seconds",
            "API request duration",
            ["exchange", "endpoint"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry,
        )

        # Signal metrics
        self.signals_generated = Counter(
            "cst_signals_generated_total",
            "Total signals generated",
            ["strategy", "symbol", "direction"],
            registry=self.registry,
        )

        self.signal_confidence = Histogram(
            "cst_signal_confidence",
            "Signal confidence score",
            ["strategy"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry,
        )

        self.predictions_accuracy = Gauge(
            "cst_predictions_accuracy_percent",
            "Prediction accuracy percentage",
            ["strategy", "timeframe"],
            registry=self.registry,
        )

        # System health metrics
        self.agent_status = Gauge(
            "cst_agent_status",
            "Agent status (1=running, 0=stopped)",
            ["agent_name"],
            registry=self.registry,
        )

        self.memory_usage_bytes = Gauge(
            "cst_memory_usage_bytes", "Memory usage in bytes", ["component"], registry=self.registry
        )

        self.cpu_usage_percent = Gauge(
            "cst_cpu_usage_percent", "CPU usage percentage", ["component"], registry=self.registry
        )

        # Alert metrics
        self.alerts_total = Counter(
            "cst_alerts_total", "Total alerts fired", ["severity", "type"], registry=self.registry
        )

        self.error_rate = Gauge(
            "cst_error_rate_per_minute",
            "Error rate per minute",
            ["component"],
            registry=self.registry,
        )

        self.logger.info("Prometheus metrics initialized")

    def record_order(
        self,
        exchange: str,
        symbol: str,
        side: str,
        status: str,
        execution_time: float,
        slippage: float,
    ) -> None:
        """Record order execution metrics."""
        self.orders_total.labels(exchange=exchange, symbol=symbol, side=side, status=status).inc()

        self.order_execution_time.labels(exchange=exchange, symbol=symbol).observe(execution_time)

        if slippage >= 0:
            self.slippage_percent.labels(exchange=exchange, symbol=symbol).observe(slippage)

    def update_position_pnl(self, symbol: str, pnl: float) -> None:
        """Update position PnL."""
        self.position_pnl.labels(symbol=symbol).set(pnl)

    def update_portfolio_metrics(
        self, portfolio_value: float, daily_pnl_percent: float, max_drawdown_percent: float
    ) -> None:
        """Update portfolio-level metrics."""
        self.portfolio_value.set(portfolio_value)
        self.daily_pnl_percent.set(daily_pnl_percent)
        self.max_drawdown_percent.set(max_drawdown_percent)

    def update_risk_level(self, risk_level: str) -> None:
        """Update current risk level."""
        self.risk_level.state(risk_level)

    def set_kill_switch_status(self, active: bool) -> None:
        """Set kill switch status."""
        self.kill_switch_active.set(1 if active else 0)

    def update_data_source(self, source: str, quality_score: float) -> None:
        """Update data source metrics."""
        self.data_source_last_update.labels(source=source).set(time.time())
        self.data_quality_score.labels(source=source).set(quality_score)

    def record_api_request(
        self, exchange: str, endpoint: str, status: str, duration: float
    ) -> None:
        """Record API request metrics."""
        self.api_requests_total.labels(exchange=exchange, endpoint=endpoint, status=status).inc()

        self.api_request_duration.labels(exchange=exchange, endpoint=endpoint).observe(duration)

    def record_signal(self, strategy: str, symbol: str, direction: str, confidence: float) -> None:
        """Record signal generation."""
        self.signals_generated.labels(strategy=strategy, symbol=symbol, direction=direction).inc()

        self.signal_confidence.labels(strategy=strategy).observe(confidence)

    def update_prediction_accuracy(self, strategy: str, timeframe: str, accuracy: float) -> None:
        """Update prediction accuracy."""
        self.predictions_accuracy.labels(strategy=strategy, timeframe=timeframe).set(accuracy)

    def set_agent_status(self, agent_name: str, running: bool) -> None:
        """Set agent status."""
        self.agent_status.labels(agent_name=agent_name).set(1 if running else 0)

    def update_system_resources(
        self, component: str, memory_bytes: int, cpu_percent: float
    ) -> None:
        """Update system resource usage."""
        self.memory_usage_bytes.labels(component=component).set(memory_bytes)
        self.cpu_usage_percent.labels(component=component).set(cpu_percent)

    def record_alert(self, severity: str, alert_type: str) -> None:
        """Record alert firing."""
        self.alerts_total.labels(severity=severity, type=alert_type).inc()

    def update_error_rate(self, component: str, rate: float) -> None:
        """Update error rate."""
        self.error_rate.labels(component=component).set(rate)

    def get_metrics(self) -> str:
        """Get all metrics in Prometheus format."""
        return generate_latest(self.registry).decode("utf-8")

    def get_content_type(self) -> str:
        """Get Prometheus content type."""
        return CONTENT_TYPE_LATEST


# Global metrics instance
_metrics_instance: Optional[CryptoSmartTraderMetrics] = None


def get_metrics() -> CryptoSmartTraderMetrics:
    """Get global metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = CryptoSmartTraderMetrics()
    return _metrics_instance


def create_metrics_registry() -> CryptoSmartTraderMetrics:
    """Create new metrics registry for testing."""
    return CryptoSmartTraderMetrics(CollectorRegistry())
