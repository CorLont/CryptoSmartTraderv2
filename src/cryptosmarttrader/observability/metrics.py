"""
Centralized Prometheus Metrics for CryptoSmartTrader V2
All observability metrics consolidated in single module with consistent naming
"""

from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, generate_latest
from typing import Dict, Optional, Any
import time
import threading
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Standard metric types for consistent usage"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricDefinition:
    """Standardized metric definition"""
    name: str
    description: str
    labels: list
    metric_type: MetricType


class CryptoSmartTraderMetrics:
    """
    Centralized metrics registry for CryptoSmartTrader V2
    
    Standardized metric naming convention:
    - Trading: orders_sent, orders_filled, order_errors
    - Performance: latency_ms, slippage_bps, equity, drawdown_pct  
    - Signals: signals_received, signals_processed, signal_accuracy
    - System: api_calls_total, cache_hits, memory_usage_bytes
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._lock = threading.Lock()
        self._last_signal_time = time.time()  # Track last signal for NoSignals alert
        self._initialize_metrics()
        self._initialize_fase_d_alerts()
        
    def _initialize_fase_d_alerts(self):
        """Initialize FASE D alert conditions and thresholds"""
        # Alert thresholds for FASE D requirements
        self.alert_thresholds = {
            'high_order_error_rate': 0.05,  # 5% error rate threshold
            'drawdown_too_high': 0.10,      # 10% drawdown threshold  
            'no_signals_timeout': 1800,     # 30 minutes = 1800 seconds
        }
        # Initialize last signal timestamp
        self.last_signal_timestamp.set(time.time())
        logger.info("FASE D alert thresholds initialized")
    
    def _initialize_metrics(self):
        """Initialize all standard metrics with consistent naming"""
        
        # Trading Metrics
        self.orders_sent = Counter(
            'orders_sent_total',
            'Total number of orders sent to exchange',
            ['exchange', 'symbol', 'side', 'order_type'],
            registry=self.registry
        )
        
        self.orders_filled = Counter(
            'orders_filled_total', 
            'Total number of orders successfully filled',
            ['exchange', 'symbol', 'side', 'order_type'],
            registry=self.registry
        )
        
        self.order_errors = Counter(
            'order_errors_total',
            'Total number of order errors by type',
            ['exchange', 'symbol', 'error_type', 'error_code'],
            registry=self.registry
        )
        
        # Performance Metrics
        self.latency_ms = Histogram(
            'latency_ms',
            'Request latency in milliseconds',
            ['operation', 'exchange', 'endpoint'],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
            registry=self.registry
        )
        
        self.slippage_bps = Histogram(
            'slippage_bps', 
            'Order slippage in basis points',
            ['exchange', 'symbol', 'side'],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 25, 50, 100],
            registry=self.registry
        )
        
        self.equity = Gauge(
            'equity_usd',
            'Current portfolio equity in USD',
            ['strategy', 'account'],
            registry=self.registry
        )
        
        self.drawdown_pct = Gauge(
            'drawdown_pct',
            'Current drawdown percentage from peak',
            ['strategy', 'timeframe'],
            registry=self.registry
        )
        
        # Signal & ML Metrics
        self.signals_received = Counter(
            'signals_received_total',
            'Total number of trading signals received',
            ['agent', 'signal_type', 'symbol'],
            registry=self.registry
        )
        
        self.signals_processed = Counter(
            'signals_processed_total',
            'Total number of signals successfully processed',
            ['agent', 'signal_type', 'symbol', 'outcome'],
            registry=self.registry
        )
        
        self.signal_accuracy = Gauge(
            'signal_accuracy_pct',
            'Signal accuracy percentage by agent',
            ['agent', 'symbol', 'timeframe'],
            registry=self.registry
        )
        
        # System & Infrastructure Metrics
        self.api_calls_total = Counter(
            'api_calls_total',
            'Total number of API calls by service',
            ['service', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.cache_operations = Counter(
            'cache_operations_total',
            'Cache operations by type and result',
            ['operation', 'result'],
            registry=self.registry
        )
        
        self.memory_usage_bytes = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes by component',
            ['component'],
            registry=self.registry
        )
        
        # FASE C SPECIFIC METRICS - Guardrails & Observability
        
        # Execution Policy Metrics
        self.execution_decisions = Counter(
            'execution_decisions_total',
            'Execution policy decisions by outcome',
            ['symbol', 'side', 'decision'],
            registry=self.registry
        )
        
        self.execution_gates = Counter(
            'execution_gates_total',
            'Execution gate results by gate type',
            ['gate', 'result'],
            registry=self.registry
        )
        
        self.execution_latency_ms = Histogram(
            'execution_latency_ms',
            'Execution policy decision latency',
            ['operation'],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500],
            registry=self.registry
        )
        
        self.estimated_slippage_bps = Histogram(
            'estimated_slippage_bps',
            'Estimated slippage in basis points',
            ['symbol'],
            buckets=[0.1, 0.5, 1, 2, 5, 10, 25, 50, 100],
            registry=self.registry
        )
        
        # Risk Guard Metrics
        self.risk_violations = Counter(
            'risk_violations_total',
            'Risk violations by type and severity',
            ['risk_type', 'risk_level'],
            registry=self.registry
        )
        
        self.portfolio_risk_score = Gauge(
            'portfolio_risk_score',
            'Current portfolio risk score (0.0-1.0)',
            registry=self.registry
        )
        
        self.portfolio_equity = Gauge(
            'portfolio_equity_usd',
            'Current portfolio equity in USD',
            registry=self.registry
        )
        
        self.portfolio_drawdown_pct = Gauge(
            'portfolio_drawdown_pct',
            'Current portfolio drawdown percentage',
            registry=self.registry
        )
        
        self.portfolio_exposure = Gauge(
            'portfolio_exposure_usd',
            'Current total portfolio exposure',
            registry=self.registry
        )
        
        self.portfolio_positions = Gauge(
            'portfolio_positions_count',
            'Current number of open positions',
            registry=self.registry
        )
        
        self.kill_switch_triggers = Counter(
            'kill_switch_triggers_total',
            'Kill switch trigger events',
            registry=self.registry
        )
        
        # Alert-specific metrics for monitoring
        self.high_order_error_rate = Gauge(
            'high_order_error_rate',
            'Boolean indicator for high order error rate alert',
            registry=self.registry
        )
        
        self.drawdown_too_high = Gauge(
            'drawdown_too_high',
            'Boolean indicator for excessive drawdown alert',
            registry=self.registry
        )
        
        self.no_signals_timeout = Gauge(
            'no_signals_timeout',
            'Boolean indicator for signal timeout alert',
            registry=self.registry
        )
        
        self.last_signal_timestamp = Gauge(
            'last_signal_timestamp_seconds',
            'Unix timestamp of last received signal',
            registry=self.registry
        )
    
    # FASE C METHODS - Alert Logic Integration
    
    def record_execution_decision(self, symbol: str, side: str, decision: str):
        """Record execution policy decision"""
        self.execution_decisions.labels(symbol=symbol, side=side, decision=decision).inc()
    
    def record_execution_gate(self, gate: str, result: str):
        """Record execution gate result"""
        self.execution_gates.labels(gate=gate, result=result).inc()
    
    def record_risk_violation(self, risk_type: str, risk_level: str):
        """Record risk violation"""
        self.risk_violations.labels(risk_type=risk_type, risk_level=risk_level).inc()
    
    def update_portfolio_metrics(self, equity: float, drawdown_pct: float, exposure: float, positions: int):
        """Update portfolio metrics"""
        self.portfolio_equity.set(equity)
        self.portfolio_drawdown_pct.set(drawdown_pct)
        self.portfolio_exposure.set(exposure)
        self.portfolio_positions.set(positions)
    
    def check_alert_conditions(self):
        """Check and update alert condition metrics"""
        import time
        from datetime import datetime, timedelta
        
        # Check HighOrderErrorRate (>10% error rate in last 5 minutes)
        # This would require tracking order errors vs total orders
        # For now, set based on simple heuristic
        
        # Check DrawdownTooHigh (>3% drawdown)
        current_drawdown = self.portfolio_drawdown_pct._value._value if hasattr(self.portfolio_drawdown_pct, '_value') else 0
        self.drawdown_too_high.set(1 if current_drawdown > 3.0 else 0)
        
        # Check NoSignals (no signals for 30 minutes)
        current_time = time.time()
        last_signal = self.last_signal_timestamp._value._value if hasattr(self.last_signal_timestamp, '_value') else current_time
        signal_age_minutes = (current_time - last_signal) / 60.0
        self.no_signals_timeout.set(1 if signal_age_minutes > 30.0 else 0)
    
    def record_signal_received(self, agent: str, signal_type: str, symbol: str):
        """Record signal received and update timestamp"""
        import time
        self.signals_received.labels(agent=agent, signal_type=signal_type, symbol=symbol).inc()
        self.last_signal_timestamp.set(time.time())
        self._last_signal_time = time.time()
        # Reset no signals alert when signal received
        self.no_signals_timeout.set(0)
        
    def record_order_error(self, exchange: str, symbol: str, error_type: str, error_code: str):
        """Record order error and update error rate tracking"""
        self.order_errors.labels(exchange=exchange, symbol=symbol, error_type=error_type, error_code=error_code).inc()
        self._update_order_error_rate()
        
    def _update_order_error_rate(self):
        """Update order error rate alert based on recent activity"""
        # Calculate error rate from counters
        total_orders = self._get_counter_value(self.orders_sent)
        total_errors = self._get_counter_value(self.order_errors)
        
        if total_orders > 0:
            error_rate = total_errors / total_orders
            self.high_order_error_rate.set(1 if error_rate > self.alert_thresholds['high_order_error_rate'] else 0)
        else:
            self.high_order_error_rate.set(0)
            
    def update_drawdown(self, drawdown_pct: float):
        """Update portfolio drawdown and check alert threshold"""
        self.portfolio_drawdown_pct.set(drawdown_pct)
        self.drawdown_too_high.set(1 if drawdown_pct > self.alert_thresholds['drawdown_too_high'] * 100 else 0)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of key metrics for monitoring"""
        return {
            'total_orders_sent': self._get_counter_value(self.orders_sent),
            'total_orders_filled': self._get_counter_value(self.orders_filled),
            'total_order_errors': self._get_counter_value(self.order_errors),
            'total_signals_received': self._get_counter_value(self.signals_received),
            'total_risk_violations': self._get_counter_value(self.risk_violations),
            'current_portfolio_equity': self._get_gauge_value(self.portfolio_equity),
            'current_drawdown_pct': self._get_gauge_value(self.portfolio_drawdown_pct),
            'current_risk_score': self._get_gauge_value(self.portfolio_risk_score),
            'kill_switch_triggers': self._get_counter_value(self.kill_switch_triggers),
            'alert_high_order_error_rate': self._get_gauge_value(self.high_order_error_rate),
            'alert_drawdown_too_high': self._get_gauge_value(self.drawdown_too_high),
            'alert_no_signals_timeout': self._get_gauge_value(self.no_signals_timeout)
        }
    
    def _get_counter_value(self, counter) -> float:
        """Safely get counter value"""
        try:
            return sum(sample.value for sample in counter.collect()[0].samples)
        except (AttributeError, IndexError, TypeError) as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to get counter value: {e}")
            return 0.0
    
    def _get_gauge_value(self, gauge) -> float:
        """Safely get gauge value"""
        try:
            return gauge._value._value
        except (AttributeError, TypeError) as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to get gauge value: {e}")
            return 0.0
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')


class PrometheusMetrics:
    """
    Singleton wrapper for CryptoSmartTraderMetrics
    Provides global access to centralized metrics
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = CryptoSmartTraderMetrics()
            return cls._instance
    
    @classmethod
    def get_instance(cls) -> CryptoSmartTraderMetrics:
        """Get the singleton metrics instance"""
        return cls()
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton (for testing)"""
        cls._instance = None


# Context managers for automatic metric recording
class ExecutionTimer:
    """Context manager for automatic execution latency recording"""
    
    def __init__(self, operation: str, metrics: CryptoSmartTraderMetrics = None):
        self.operation = operation
        self.metrics = metrics or PrometheusMetrics.get_instance()
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            latency_ms = (time.time() - self.start_time) * 1000
            self.metrics.execution_latency_ms.labels(operation=self.operation).observe(latency_ms)


def record_order_metrics(symbol: str, side: str, order_type: str, exchange: str = "kraken"):
    """Decorator for automatic order metrics recording"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics = PrometheusMetrics.get_instance()
            
            # Record order sent
            metrics.orders_sent.labels(
                exchange=exchange, 
                symbol=symbol, 
                side=side, 
                order_type=order_type
            ).inc()
            
            try:
                result = func(*args, **kwargs)
                
                # Record order filled (assuming success means filled)
                metrics.orders_filled.labels(
                    exchange=exchange,
                    symbol=symbol, 
                    side=side,
                    order_type=order_type
                ).inc()
                
                return result
                
            except Exception as e:
                # Record order error
                metrics.order_errors.labels(
                    exchange=exchange,
                    symbol=symbol,
                    error_type=type(e).__name__,
                    error_code=getattr(e, 'code', 'unknown')
                ).inc()
                raise
                
        return wrapper
    return decorator


# Additional system metrics initialization (moved from duplicate location)
def _initialize_additional_system_metrics(registry):
    """Initialize additional system metrics"""
    api_calls_total = Counter(
        'api_calls_total',
        'Total API calls made',
        ['service', 'endpoint', 'method', 'status_code'],
        registry=registry
    )
    
    cache_hits = Counter(
        'cache_hits_total',
        'Cache hit/miss statistics',
        ['cache_type', 'hit_miss'],
        registry=registry
    )
    
    memory_usage_bytes = Gauge(
        'memory_usage_bytes',
        'Memory usage in bytes by component',
        ['component', 'memory_type'],
        registry=registry
    )
    
    # Risk Management Metrics
    risk_violations = Counter(
        'risk_violations_total',
        'Total risk limit violations',
        ['violation_type', 'symbol', 'strategy'],
        registry=registry
    )
    
    position_size_usd = Gauge(
        'position_size_usd',
        'Current position size in USD',
        ['symbol', 'strategy', 'side'],
        registry=registry
    )
    
    # Data Quality Metrics  
    data_points_received = Counter(
        'data_points_received_total',
        'Total data points received from sources',
        ['source', 'data_type', 'symbol'],
        registry=registry
    )
    
    return {
        'api_calls_total': api_calls_total,
        'cache_hits': cache_hits,
        'memory_usage_bytes': memory_usage_bytes,
        'risk_violations': risk_violations,
        'position_size_usd': position_size_usd,
        'data_points_received': data_points_received
    }
    
    # Convenience Methods for Common Operations
    
    def record_order_sent(self, exchange: str, symbol: str, side: str, order_type: str):
        """Record an order being sent"""
        self.orders_sent.labels(
            exchange=exchange, 
            symbol=symbol, 
            side=side, 
            order_type=order_type
        ).inc()
    
    def record_order_filled(self, exchange: str, symbol: str, side: str, order_type: str):
        """Record an order being filled"""
        self.orders_filled.labels(
            exchange=exchange,
            symbol=symbol,
            side=side, 
            order_type=order_type
        ).inc()
    
    def record_order_error(self, exchange: str, symbol: str, error_type: str, error_code: str = "unknown"):
        """Record an order error"""
        self.order_errors.labels(
            exchange=exchange,
            symbol=symbol,
            error_type=error_type,
            error_code=error_code
        ).inc()
    
    def record_latency(self, operation: str, exchange: str, endpoint: str, latency_ms: float):
        """Record operation latency"""
        self.latency_ms.labels(
            operation=operation,
            exchange=exchange,
            endpoint=endpoint
        ).observe(latency_ms)
    
    def record_slippage(self, exchange: str, symbol: str, side: str, slippage_bps: float):
        """Record trade slippage in basis points"""
        self.slippage_bps.labels(
            exchange=exchange,
            symbol=symbol,
            side=side
        ).observe(slippage_bps)
    
    def update_equity(self, strategy: str, account: str, equity_usd: float):
        """Update current equity"""
        self.equity.labels(strategy=strategy, account=account).set(equity_usd)
    
    def update_drawdown(self, strategy: str, timeframe: str, drawdown_pct: float):
        """Update current drawdown percentage"""
        self.drawdown_pct.labels(strategy=strategy, timeframe=timeframe).set(drawdown_pct)
    
    def record_signal(self, agent: str, signal_type: str, symbol: str):
        """Record a signal being received"""
        self.signals_received.labels(
            agent=agent,
            signal_type=signal_type,
            symbol=symbol
        ).inc()
    
    def record_api_call(self, service: str, endpoint: str, method: str, status_code: int):
        """Record an API call"""
        self.api_calls_total.labels(
            service=service,
            endpoint=endpoint,
            method=method,
            status_code=str(status_code)
        ).inc()
    
    def get_metrics(self) -> bytes:
        """Get all metrics in Prometheus format"""
        return generate_latest(self.registry)


# Global metrics instance
_metrics_instance: Optional[CryptoSmartTraderMetrics] = None
_metrics_lock = threading.Lock()


def get_metrics() -> CryptoSmartTraderMetrics:
    """Get global metrics instance (singleton)"""
    global _metrics_instance
    
    if _metrics_instance is None:
        with _metrics_lock:
            if _metrics_instance is None:
                _metrics_instance = CryptoSmartTraderMetrics()
    
    return _metrics_instance


def reset_metrics():
    """Reset global metrics instance (for testing)"""
    global _metrics_instance
    with _metrics_lock:
        _metrics_instance = None


# Context managers for timing operations
class timer:
    """Context manager for timing operations and recording latency"""
    
    def __init__(self, operation: str, exchange: str = "unknown", endpoint: str = "unknown"):
        self.operation = operation
        self.exchange = exchange
        self.endpoint = endpoint
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            latency_ms = (time.time() - self.start_time) * 1000
            get_metrics().record_latency(
                self.operation, 
                self.exchange, 
                self.endpoint, 
                latency_ms
            )


# Decorators for automatic metrics collection
def track_api_calls(service: str, endpoint: str = "unknown"):
    """Decorator to automatically track API calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                get_metrics().record_api_call(service, endpoint, func.__name__, 200)
                return result
            except Exception as e:
                get_metrics().record_api_call(service, endpoint, func.__name__, 500)
                raise
        return wrapper
    return decorator


def track_orders(exchange: str):
    """Decorator to automatically track order operations"""
    def decorator(func):
        def wrapper(symbol: str, side: str, order_type: str = "market", *args, **kwargs):
            try:
                get_metrics().record_order_sent(exchange, symbol, side, order_type)
                result = func(symbol, side, order_type, *args, **kwargs)
                get_metrics().record_order_filled(exchange, symbol, side, order_type)
                return result
            except Exception as e:
                get_metrics().record_order_error(exchange, symbol, type(e).__name__)
                raise
        return wrapper
    return decorator


# Standard metric labels for consistency
STANDARD_LABELS = {
    'exchanges': ['kraken', 'binance', 'coinbase', 'bybit'],
    'sides': ['buy', 'sell'],
    'order_types': ['market', 'limit', 'stop', 'stop_limit'],
    'signal_types': ['entry', 'exit', 'stop_loss', 'take_profit'],
    'error_types': ['timeout', 'rate_limit', 'invalid_order', 'insufficient_funds'],
    'operations': ['place_order', 'cancel_order', 'get_balance', 'get_orderbook'],
    'strategies': ['momentum', 'mean_reversion', 'arbitrage', 'market_making']
}


if __name__ == "__main__":
    # Example usage
    metrics = get_metrics()
    
    # Record some sample metrics
    metrics.record_order_sent("kraken", "BTC/USD", "buy", "market")
    metrics.record_latency("place_order", "kraken", "/api/orders", 45.2)
    metrics.update_equity("momentum", "main", 100000.0)
    
    print("Sample metrics recorded successfully")
    print(f"Metrics export size: {len(metrics.get_metrics())} bytes")
