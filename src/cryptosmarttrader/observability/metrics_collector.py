"""
Enterprise Metrics Collector for CryptoSmartTrader
Comprehensive Prometheus metrics for execution, performance, and system health monitoring.
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from prometheus_client import (
    Counter, Histogram, Gauge, Summary,
    CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST
)


class OrderState(Enum):
    """Order states for metrics tracking."""
    SENT = "sent"
    FILLED = "filled" 
    REJECTED = "rejected"
    CANCELED = "canceled"
    FAILED = "failed"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class RequestContext:
    """Request context for correlation tracking."""
    request_id: str
    correlation_id: str
    timestamp: datetime
    component: str
    operation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Enterprise metrics collector with comprehensive trading and system metrics.
    
    Features:
    - Order execution metrics (sent, filled, errors, latency)
    - Trading performance metrics (slippage, equity, drawdown)
    - System health metrics (signals, API health, resource usage)
    - Request correlation and structured logging
    - Alert rule evaluation
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.logger = self._setup_structured_logger()
        
        # Initialize Prometheus metrics
        self._init_order_metrics()
        self._init_trading_metrics()
        self._init_system_metrics()
        self._init_api_metrics()
        
        # Request tracking
        self.active_requests: Dict[str, RequestContext] = {}
        
        # Alert state tracking
        self.alert_states: Dict[str, bool] = {}
        self.last_signal_time = time.time()
        
        self.logger.info("MetricsCollector initialized with comprehensive monitoring")
    
    def _setup_structured_logger(self) -> logging.Logger:
        """Setup structured JSON logger with correlation IDs."""
        logger = logging.getLogger("cryptotrader.metrics")
        
        # Create custom formatter for JSON logging
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "level": record.levelname,
                    "component": "metrics_collector",
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno
                }
                
                # Add correlation ID if available
                if hasattr(record, 'correlation_id'):
                    log_entry["correlation_id"] = record.correlation_id
                
                # Add request ID if available
                if hasattr(record, 'request_id'):
                    log_entry["request_id"] = record.request_id
                
                # Add extra fields
                if hasattr(record, 'extra_fields'):
                    log_entry.update(record.extra_fields)
                
                return json.dumps(log_entry)
        
        # Configure handler with JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        return logger
    
    def _init_order_metrics(self):
        """Initialize order execution metrics."""
        # Order counters by state and symbol
        self.orders_total = Counter(
            'cryptotrader_orders_total',
            'Total number of orders by state and symbol',
            ['state', 'symbol', 'side'],
            registry=self.registry
        )
        
        # Order execution latency
        self.order_latency = Histogram(
            'cryptotrader_order_latency_seconds',
            'Order execution latency in seconds',
            ['symbol', 'order_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
            registry=self.registry
        )
        
        # Order errors by type
        self.order_errors = Counter(
            'cryptotrader_order_errors_total',
            'Total order errors by type and symbol',
            ['error_type', 'symbol'],
            registry=self.registry
        )
        
        # Slippage distribution
        self.slippage_bps = Histogram(
            'cryptotrader_slippage_bps',
            'Order slippage in basis points',
            ['symbol', 'side'],
            buckets=[0, 5, 10, 20, 30, 50, 100, 200, 500],
            registry=self.registry
        )
    
    def _init_trading_metrics(self):
        """Initialize trading performance metrics."""
        # Portfolio equity
        self.equity = Gauge(
            'cryptotrader_equity_usd',
            'Current portfolio equity in USD',
            registry=self.registry
        )
        
        # Drawdown percentage
        self.drawdown_pct = Gauge(
            'cryptotrader_drawdown_percent',
            'Current drawdown percentage from peak',
            registry=self.registry
        )
        
        # Daily PnL
        self.daily_pnl = Gauge(
            'cryptotrader_daily_pnl_usd',
            'Daily profit and loss in USD',
            registry=self.registry
        )
        
        # Position sizes by symbol
        self.position_size = Gauge(
            'cryptotrader_position_size',
            'Current position size by symbol',
            ['symbol'],
            registry=self.registry
        )
        
        # Risk metrics
        self.risk_score = Gauge(
            'cryptotrader_risk_score',
            'Current portfolio risk score (0-100)',
            registry=self.registry
        )
    
    def _init_system_metrics(self):
        """Initialize system health metrics."""
        # Signal reception
        self.signals_received = Counter(
            'cryptotrader_signals_received_total',
            'Total signals received by source and type',
            ['source', 'signal_type'],
            registry=self.registry
        )
        
        # Last signal timestamp
        self.last_signal_timestamp = Gauge(
            'cryptotrader_last_signal_timestamp',
            'Timestamp of last received signal',
            ['source'],
            registry=self.registry
        )
        
        # System uptime
        self.system_uptime = Gauge(
            'cryptotrader_uptime_seconds',
            'System uptime in seconds',
            registry=self.registry
        )
        
        # Memory usage
        self.memory_usage = Gauge(
            'cryptotrader_memory_usage_bytes',
            'Memory usage in bytes',
            ['component'],
            registry=self.registry
        )
        
        # Active connections
        self.active_connections = Gauge(
            'cryptotrader_active_connections',
            'Number of active connections by type',
            ['connection_type'],
            registry=self.registry
        )
    
    def _init_api_metrics(self):
        """Initialize API and external service metrics."""
        # API request duration
        self.api_request_duration = Histogram(
            'cryptotrader_api_request_duration_seconds',
            'API request duration by endpoint and method',
            ['endpoint', 'method', 'status_code'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        # API rate limit usage
        self.api_rate_limit_usage = Gauge(
            'cryptotrader_api_rate_limit_usage_percent',
            'API rate limit usage percentage',
            ['exchange'],
            registry=self.registry
        )
        
        # Exchange connectivity
        self.exchange_connectivity = Gauge(
            'cryptotrader_exchange_connectivity',
            'Exchange connectivity status (1=connected, 0=disconnected)',
            ['exchange'],
            registry=self.registry
        )
    
    def start_request(self, component: str, operation: str, **metadata) -> str:
        """Start tracking a new request with correlation ID."""
        request_id = str(uuid.uuid4())
        correlation_id = str(uuid.uuid4())
        
        context = RequestContext(
            request_id=request_id,
            correlation_id=correlation_id,
            timestamp=datetime.utcnow(),
            component=component,
            operation=operation,
            metadata=metadata
        )
        
        self.active_requests[request_id] = context
        
        # Log request start
        extra = logging.LoggerAdapter(self.logger, {
            'correlation_id': correlation_id,
            'request_id': request_id,
            'extra_fields': {
                'component': component,
                'operation': operation,
                'metadata': metadata
            }
        })
        extra.info(f"Request started: {component}.{operation}")
        
        return request_id
    
    def end_request(self, request_id: str, success: bool = True, **result_metadata):
        """End request tracking and log completion."""
        if request_id not in self.active_requests:
            self.logger.warning(f"Unknown request_id: {request_id}")
            return
        
        context = self.active_requests.pop(request_id)
        duration = (datetime.utcnow() - context.timestamp).total_seconds()
        
        # Log request completion
        extra = logging.LoggerAdapter(self.logger, {
            'correlation_id': context.correlation_id,
            'request_id': request_id,
            'extra_fields': {
                'component': context.component,
                'operation': context.operation,
                'duration_seconds': duration,
                'success': success,
                'result_metadata': result_metadata
            }
        })
        
        status = "completed successfully" if success else "failed"
        extra.info(f"Request {status}: {context.component}.{context.operation} ({duration:.3f}s)")
    
    def record_order_sent(self, symbol: str, side: str, order_type: str = "limit"):
        """Record order sent metric."""
        self.orders_total.labels(state=OrderState.SENT.value, symbol=symbol, side=side).inc()
        
        self.logger.info("Order sent", extra={
            'extra_fields': {
                'symbol': symbol,
                'side': side,
                'order_type': order_type,
                'metric': 'order_sent'
            }
        })
    
    def record_order_filled(self, symbol: str, side: str, quantity: float, price: float,
                           slippage_bps: float, latency_seconds: float, order_type: str = "limit"):
        """Record order filled with detailed metrics."""
        # Update counters
        self.orders_total.labels(state=OrderState.FILLED.value, symbol=symbol, side=side).inc()
        
        # Record latency
        self.order_latency.labels(symbol=symbol, order_type=order_type).observe(latency_seconds)
        
        # Record slippage
        self.slippage_bps.labels(symbol=symbol, side=side).observe(slippage_bps)
        
        self.logger.info("Order filled", extra={
            'extra_fields': {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'price': price,
                'slippage_bps': slippage_bps,
                'latency_seconds': latency_seconds,
                'order_type': order_type,
                'metric': 'order_filled'
            }
        })
    
    def record_order_error(self, symbol: str, error_type: str, error_message: str):
        """Record order error."""
        self.order_errors.labels(error_type=error_type, symbol=symbol).inc()
        
        self.logger.error("Order error", extra={
            'extra_fields': {
                'symbol': symbol,
                'error_type': error_type,
                'error_message': error_message,
                'metric': 'order_error'
            }
        })
    
    def update_equity(self, equity_usd: float):
        """Update portfolio equity."""
        self.equity.set(equity_usd)
        
        self.logger.info("Equity updated", extra={
            'extra_fields': {
                'equity_usd': equity_usd,
                'metric': 'equity_update'
            }
        })
    
    def update_drawdown(self, drawdown_pct: float):
        """Update drawdown percentage."""
        self.drawdown_pct.set(drawdown_pct)
        
        # Check for high drawdown alert
        if drawdown_pct > 10.0:  # 10% drawdown threshold
            self._trigger_alert("DrawdownTooHigh", AlertSeverity.CRITICAL, {
                'drawdown_percent': drawdown_pct,
                'threshold': 10.0
            })
        
        self.logger.info("Drawdown updated", extra={
            'extra_fields': {
                'drawdown_percent': drawdown_pct,
                'metric': 'drawdown_update'
            }
        })
    
    def record_signal_received(self, source: str, signal_type: str, **signal_data):
        """Record signal reception."""
        self.signals_received.labels(source=source, signal_type=signal_type).inc()
        self.last_signal_timestamp.labels(source=source).set(time.time())
        self.last_signal_time = time.time()
        
        self.logger.info("Signal received", extra={
            'extra_fields': {
                'source': source,
                'signal_type': signal_type,
                'signal_data': signal_data,
                'metric': 'signal_received'
            }
        })
    
    def record_api_request(self, endpoint: str, method: str, status_code: int, 
                          duration_seconds: float):
        """Record API request metrics."""
        self.api_request_duration.labels(
            endpoint=endpoint,
            method=method,
            status_code=str(status_code)
        ).observe(duration_seconds)
        
        # Check for high error rate
        if status_code >= 400:
            self._check_api_error_rate()
    
    def update_exchange_connectivity(self, exchange: str, connected: bool):
        """Update exchange connectivity status."""
        self.exchange_connectivity.labels(exchange=exchange).set(1 if connected else 0)
        
        self.logger.info("Exchange connectivity updated", extra={
            'extra_fields': {
                'exchange': exchange,
                'connected': connected,
                'metric': 'exchange_connectivity'
            }
        })
    
    def _check_api_error_rate(self):
        """Check API error rate and trigger alerts if necessary."""
        # This would implement actual error rate calculation
        # For demo purposes, we'll simulate the check
        error_rate = 0.15  # 15% error rate
        
        if error_rate > 0.10:  # 10% threshold
            self._trigger_alert("HighOrderErrorRate", AlertSeverity.WARNING, {
                'error_rate_percent': error_rate * 100,
                'threshold_percent': 10.0
            })
    
    def _trigger_alert(self, alert_name: str, severity: AlertSeverity, context: Dict[str, Any]):
        """Trigger alert with context."""
        alert_key = f"{alert_name}_{severity.value}"
        
        # Prevent alert spam
        if self.alert_states.get(alert_key, False):
            return
        
        self.alert_states[alert_key] = True
        
        self.logger.warning(f"ALERT: {alert_name}", extra={
            'extra_fields': {
                'alert_name': alert_name,
                'severity': severity.value,
                'context': context,
                'metric': 'alert_triggered'
            }
        })
    
    def check_no_signals_alert(self):
        """Check for no signals received alert."""
        time_since_last_signal = time.time() - self.last_signal_time
        
        if time_since_last_signal > 1800:  # 30 minutes
            self._trigger_alert("NoSignals", AlertSeverity.WARNING, {
                'minutes_since_last_signal': time_since_last_signal / 60,
                'threshold_minutes': 30
            })
    
    def update_position(self, symbol: str, size: float):
        """Update position size for symbol."""
        self.position_size.labels(symbol=symbol).set(size)
    
    def update_daily_pnl(self, pnl_usd: float):
        """Update daily PnL."""
        self.daily_pnl.set(pnl_usd)
    
    def update_risk_score(self, score: float):
        """Update risk score (0-100)."""
        self.risk_score.set(score)
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics in text format."""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get human-readable metrics summary."""
        # Check for recent signals
        self.check_no_signals_alert()
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'orders': {
                'total_sent': self._get_counter_value(self.orders_total, {'state': OrderState.SENT.value}),
                'total_filled': self._get_counter_value(self.orders_total, {'state': OrderState.FILLED.value}),
                'total_errors': self._get_counter_value(self.order_errors),
            },
            'trading': {
                'equity_usd': self.equity._value._value if hasattr(self.equity._value, '_value') else 0,
                'drawdown_percent': self.drawdown_pct._value._value if hasattr(self.drawdown_pct._value, '_value') else 0,
                'daily_pnl_usd': self.daily_pnl._value._value if hasattr(self.daily_pnl._value, '_value') else 0,
            },
            'system': {
                'signals_received': self._get_counter_value(self.signals_received),
                'minutes_since_last_signal': (time.time() - self.last_signal_time) / 60,
                'active_requests': len(self.active_requests),
            },
            'alerts': {
                'active_alerts': len([k for k, v in self.alert_states.items() if v]),
                'alert_names': [k for k, v in self.alert_states.items() if v]
            }
        }
    
    def _get_counter_value(self, counter, labels: Optional[Dict[str, str]] = None) -> float:
        """Get current value of a counter metric."""
        try:
            if labels:
                # Sum values for specific labels
                total = 0
                for sample in counter.collect()[0].samples:
                    if all(sample.labels.get(k) == v for k, v in labels.items()):
                        total += sample.value
                return total
            else:
                # Sum all values
                return sum(sample.value for sample in counter.collect()[0].samples)
        except:
            return 0.0


# Global metrics collector instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def setup_metrics_collector(registry: Optional[CollectorRegistry] = None) -> MetricsCollector:
    """Setup and configure global metrics collector."""
    global _metrics_collector
    _metrics_collector = MetricsCollector(registry)
    return _metrics_collector


# Convenience functions for common metrics
def record_order_sent(symbol: str, side: str, order_type: str = "limit"):
    """Record order sent metric."""
    get_metrics_collector().record_order_sent(symbol, side, order_type)


def record_order_filled(symbol: str, side: str, quantity: float, price: float,
                       slippage_bps: float, latency_seconds: float, order_type: str = "limit"):
    """Record order filled metric."""
    get_metrics_collector().record_order_filled(
        symbol, side, quantity, price, slippage_bps, latency_seconds, order_type
    )


def record_order_error(symbol: str, error_type: str, error_message: str):
    """Record order error metric."""
    get_metrics_collector().record_order_error(symbol, error_type, error_message)


def record_signal_received(source: str, signal_type: str, **signal_data):
    """Record signal received metric."""
    get_metrics_collector().record_signal_received(source, signal_type, **signal_data)


def update_equity(equity_usd: float):
    """Update portfolio equity metric."""
    get_metrics_collector().update_equity(equity_usd)


def update_drawdown(drawdown_pct: float):
    """Update drawdown percentage metric."""
    get_metrics_collector().update_drawdown(drawdown_pct)