"""
Prometheus Metrics System

Comprehensive metrics collection for error rates, latency, queue lengths,
slippage, PnL, drawdown and system health monitoring.
"""

import time
import threading
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from pathlib import Path

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, generate_latest, start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Mock classes for when prometheus_client is not available
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, amount=1): pass
        def labels(self, **labels): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, amount): pass
        def time(self): return MockTimer()
        def labels(self, **labels): return self
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, value): pass
        def inc(self, amount=1): pass
        def dec(self, amount=1): pass
        def labels(self, **labels): return self
    
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, amount): pass
        def labels(self, **labels): return self
    
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, data): pass
    
    class MockTimer:
        def __enter__(self): return self
        def __exit__(self, *args): pass

logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    HISTOGRAM = "histogram" 
    GAUGE = "gauge"
    SUMMARY = "summary"
    INFO = "info"

@dataclass
class MetricDefinition:
    """Metric definition"""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None

class PrometheusMetricsCollector:
    """
    Comprehensive Prometheus metrics collector for trading system
    """
    
    def __init__(self, metrics_port: int = 8000, enable_server: bool = True):
        self.metrics_port = metrics_port
        self.enable_server = enable_server
        
        # Registry for custom metrics
        self.registry = CollectorRegistry()
        
        # Metrics storage
        self.metrics: Dict[str, Any] = {}
        self.metric_definitions: Dict[str, MetricDefinition] = {}
        
        # Server state
        self.server_started = False
        self.last_collection_time = datetime.now()
        
        # Data buffers for calculations
        self.pnl_history: List[Tuple[datetime, float]] = []
        self.slippage_history: List[Tuple[datetime, float]] = []
        self.latency_history: List[Tuple[datetime, float]] = []
        
        # Setup core metrics
        self._setup_core_metrics()
        
        # Start metrics server
        if self.enable_server and PROMETHEUS_AVAILABLE:
            self._start_metrics_server()
    
    def _setup_core_metrics(self):
        """Setup core trading system metrics"""
        
        # Error rate metrics
        self.add_metric(MetricDefinition(
            name="trading_errors_total",
            metric_type=MetricType.COUNTER,
            description="Total number of trading errors",
            labels=["error_type", "component"]
        ))
        
        # API latency metrics
        self.add_metric(MetricDefinition(
            name="api_request_duration_seconds",
            metric_type=MetricType.HISTOGRAM,
            description="API request duration in seconds",
            labels=["endpoint", "exchange", "method"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        ))
        
        # Queue length metrics
        self.add_metric(MetricDefinition(
            name="queue_length",
            metric_type=MetricType.GAUGE,
            description="Current queue length",
            labels=["queue_type", "priority"]
        ))
        
        # Slippage metrics
        self.add_metric(MetricDefinition(
            name="trading_slippage_bps",
            metric_type=MetricType.HISTOGRAM,
            description="Trading slippage in basis points",
            labels=["symbol", "order_type", "side"],
            buckets=[0, 1, 2, 5, 10, 15, 20, 30, 50, 100, 200, 500]
        ))
        
        # PnL metrics
        self.add_metric(MetricDefinition(
            name="portfolio_pnl_usd",
            metric_type=MetricType.GAUGE,
            description="Portfolio P&L in USD",
            labels=["timeframe"]
        ))
        
        # Drawdown metrics
        self.add_metric(MetricDefinition(
            name="portfolio_drawdown_percent",
            metric_type=MetricType.GAUGE,
            description="Portfolio drawdown percentage"
        ))
        
        # Position metrics
        self.add_metric(MetricDefinition(
            name="position_value_usd",
            metric_type=MetricType.GAUGE,
            description="Position value in USD",
            labels=["symbol", "side"]
        ))
        
        # Trade execution metrics
        self.add_metric(MetricDefinition(
            name="trades_total",
            metric_type=MetricType.COUNTER,
            description="Total number of trades executed",
            labels=["symbol", "side", "order_type", "status"]
        ))
        
        # System health metrics
        self.add_metric(MetricDefinition(
            name="system_health_score",
            metric_type=MetricType.GAUGE,
            description="Overall system health score (0-100)"
        ))
        
        # Risk metrics
        self.add_metric(MetricDefinition(
            name="risk_limit_utilization",
            metric_type=MetricType.GAUGE,
            description="Risk limit utilization percentage",
            labels=["limit_type", "symbol"]
        ))
        
        # Circuit breaker metrics
        self.add_metric(MetricDefinition(
            name="circuit_breaker_triggers_total",
            metric_type=MetricType.COUNTER,
            description="Total circuit breaker triggers",
            labels=["breaker_name", "trigger_reason"]
        ))
        
        # Kill switch metrics
        self.add_metric(MetricDefinition(
            name="kill_switch_activations_total",
            metric_type=MetricType.COUNTER,
            description="Total kill switch activations",
            labels=["trigger_source", "severity"]
        ))
        
        # Model performance metrics
        self.add_metric(MetricDefinition(
            name="model_prediction_accuracy",
            metric_type=MetricType.GAUGE,
            description="Model prediction accuracy percentage",
            labels=["model_name", "timeframe"]
        ))
        
        # Data quality metrics
        self.add_metric(MetricDefinition(
            name="data_freshness_seconds",
            metric_type=MetricType.GAUGE,
            description="Data freshness in seconds",
            labels=["data_source", "symbol"]
        ))
        
        # Memory and CPU metrics
        self.add_metric(MetricDefinition(
            name="system_memory_usage_percent",
            metric_type=MetricType.GAUGE,
            description="System memory usage percentage"
        ))
        
        self.add_metric(MetricDefinition(
            name="system_cpu_usage_percent", 
            metric_type=MetricType.GAUGE,
            description="System CPU usage percentage"
        ))
    
    def add_metric(self, definition: MetricDefinition):
        """Add a metric definition and create the metric"""
        self.metric_definitions[definition.name] = definition
        
        if not PROMETHEUS_AVAILABLE:
            logger.warning(f"Prometheus not available, creating mock metric: {definition.name}")
        
        # Create the actual metric object
        if definition.metric_type == MetricType.COUNTER:
            self.metrics[definition.name] = Counter(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        elif definition.metric_type == MetricType.HISTOGRAM:
            self.metrics[definition.name] = Histogram(
                definition.name,
                definition.description,
                definition.labels,
                buckets=definition.buckets,
                registry=self.registry
            )
        elif definition.metric_type == MetricType.GAUGE:
            self.metrics[definition.name] = Gauge(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        elif definition.metric_type == MetricType.SUMMARY:
            self.metrics[definition.name] = Summary(
                definition.name,
                definition.description,
                definition.labels,
                registry=self.registry
            )
        elif definition.metric_type == MetricType.INFO:
            self.metrics[definition.name] = Info(
                definition.name,
                definition.description,
                registry=self.registry
            )
        
        logger.info(f"Added metric: {definition.name} ({definition.metric_type.value})")
    
    def _start_metrics_server(self):
        """Start Prometheus metrics HTTP server"""
        try:
            def start_server():
                start_http_server(self.metrics_port, registry=self.registry)
                logger.info(f"Prometheus metrics server started on port {self.metrics_port}")
                self.server_started = True
            
            # Start server in background thread
            server_thread = threading.Thread(target=start_server, daemon=True)
            server_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to start Prometheus metrics server: {e}")
    
    def record_error(self, error_type: str, component: str):
        """Record an error occurrence"""
        if "trading_errors_total" in self.metrics:
            self.metrics["trading_errors_total"].labels(
                error_type=error_type,
                component=component
            ).inc()
    
    def record_api_latency(self, endpoint: str, exchange: str, method: str, duration: float):
        """Record API request latency"""
        # Add to history
        self.latency_history.append((datetime.now(), duration))
        if len(self.latency_history) > 1000:
            self.latency_history = self.latency_history[-1000:]
        
        # Record metric
        if "api_request_duration_seconds" in self.metrics:
            self.metrics["api_request_duration_seconds"].labels(
                endpoint=endpoint,
                exchange=exchange,
                method=method
            ).observe(duration)
    
    def update_queue_length(self, queue_type: str, priority: str, length: int):
        """Update queue length metric"""
        if "queue_length" in self.metrics:
            self.metrics["queue_length"].labels(
                queue_type=queue_type,
                priority=priority
            ).set(length)
    
    def record_slippage(self, symbol: str, order_type: str, side: str, slippage_bps: float):
        """Record trading slippage"""
        # Add to history
        self.slippage_history.append((datetime.now(), slippage_bps))
        if len(self.slippage_history) > 1000:
            self.slippage_history = self.slippage_history[-1000:]
        
        # Record metric
        if "trading_slippage_bps" in self.metrics:
            self.metrics["trading_slippage_bps"].labels(
                symbol=symbol,
                order_type=order_type,
                side=side
            ).observe(slippage_bps)
    
    def update_portfolio_pnl(self, timeframe: str, pnl: float):
        """Update portfolio PnL metric"""
        # Add to history for daily PnL
        if timeframe == "daily":
            self.pnl_history.append((datetime.now(), pnl))
            if len(self.pnl_history) > 1000:
                self.pnl_history = self.pnl_history[-1000:]
        
        # Update metric
        if "portfolio_pnl_usd" in self.metrics:
            self.metrics["portfolio_pnl_usd"].labels(
                timeframe=timeframe
            ).set(pnl)
    
    def update_portfolio_drawdown(self, drawdown_percent: float):
        """Update portfolio drawdown metric"""
        if "portfolio_drawdown_percent" in self.metrics:
            self.metrics["portfolio_drawdown_percent"].set(drawdown_percent)
    
    def update_position_value(self, symbol: str, side: str, value_usd: float):
        """Update position value metric"""
        if "position_value_usd" in self.metrics:
            self.metrics["position_value_usd"].labels(
                symbol=symbol,
                side=side
            ).set(value_usd)
    
    def record_trade_execution(self, symbol: str, side: str, order_type: str, status: str):
        """Record trade execution"""
        if "trades_total" in self.metrics:
            self.metrics["trades_total"].labels(
                symbol=symbol,
                side=side,
                order_type=order_type,
                status=status
            ).inc()
    
    def update_system_health(self, health_score: float):
        """Update system health score (0-100)"""
        if "system_health_score" in self.metrics:
            self.metrics["system_health_score"].set(health_score)
    
    def update_risk_limit_utilization(self, limit_type: str, symbol: str, utilization_percent: float):
        """Update risk limit utilization"""
        if "risk_limit_utilization" in self.metrics:
            self.metrics["risk_limit_utilization"].labels(
                limit_type=limit_type,
                symbol=symbol
            ).set(utilization_percent)
    
    def record_circuit_breaker_trigger(self, breaker_name: str, trigger_reason: str):
        """Record circuit breaker trigger"""
        if "circuit_breaker_triggers_total" in self.metrics:
            self.metrics["circuit_breaker_triggers_total"].labels(
                breaker_name=breaker_name,
                trigger_reason=trigger_reason
            ).inc()
    
    def record_kill_switch_activation(self, trigger_source: str, severity: str):
        """Record kill switch activation"""
        if "kill_switch_activations_total" in self.metrics:
            self.metrics["kill_switch_activations_total"].labels(
                trigger_source=trigger_source,
                severity=severity
            ).inc()
    
    def update_model_accuracy(self, model_name: str, timeframe: str, accuracy_percent: float):
        """Update model prediction accuracy"""
        if "model_prediction_accuracy" in self.metrics:
            self.metrics["model_prediction_accuracy"].labels(
                model_name=model_name,
                timeframe=timeframe
            ).set(accuracy_percent)
    
    def update_data_freshness(self, data_source: str, symbol: str, age_seconds: float):
        """Update data freshness metric"""
        if "data_freshness_seconds" in self.metrics:
            self.metrics["data_freshness_seconds"].labels(
                data_source=data_source,
                symbol=symbol
            ).set(age_seconds)
    
    def update_system_resources(self, memory_percent: float, cpu_percent: float):
        """Update system resource metrics"""
        if "system_memory_usage_percent" in self.metrics:
            self.metrics["system_memory_usage_percent"].set(memory_percent)
        
        if "system_cpu_usage_percent" in self.metrics:
            self.metrics["system_cpu_usage_percent"].set(cpu_percent)
    
    def get_metric_value(self, metric_name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
        """Get current value of a metric (for gauges)"""
        try:
            if metric_name not in self.metrics:
                return None
            
            metric = self.metrics[metric_name]
            
            # For mock metrics, return dummy value
            if not PROMETHEUS_AVAILABLE:
                return 0.0
            
            # This is a simplified approach - in production, you'd use
            # the prometheus_client's metric value extraction
            return None
            
        except Exception as e:
            logger.error(f"Failed to get metric value for {metric_name}: {e}")
            return None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of key metrics"""
        
        # Calculate recent averages
        now = datetime.now()
        recent_cutoff = now - timedelta(minutes=5)
        
        # Recent latency
        recent_latencies = [lat for ts, lat in self.latency_history if ts >= recent_cutoff]
        avg_latency = sum(recent_latencies) / len(recent_latencies) if recent_latencies else 0
        
        # Recent slippage
        recent_slippages = [slip for ts, slip in self.slippage_history if ts >= recent_cutoff]
        avg_slippage = sum(recent_slippages) / len(recent_slippages) if recent_slippages else 0
        
        # Recent PnL
        recent_pnl = [pnl for ts, pnl in self.pnl_history if ts >= recent_cutoff]
        
        return {
            "timestamp": now.isoformat(),
            "metrics_server_running": self.server_started,
            "total_metrics": len(self.metrics),
            "recent_performance": {
                "avg_latency_ms": avg_latency * 1000,
                "avg_slippage_bps": avg_slippage,
                "recent_trades": len(recent_pnl),
            },
            "data_points": {
                "latency_history": len(self.latency_history),
                "slippage_history": len(self.slippage_history),
                "pnl_history": len(self.pnl_history)
            }
        }
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry).decode('utf-8')
        else:
            return "# Prometheus not available - mock metrics\n"
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for metrics system"""
        return {
            "prometheus_available": PROMETHEUS_AVAILABLE,
            "metrics_server_running": self.server_started,
            "metrics_port": self.metrics_port,
            "total_metrics": len(self.metrics),
            "last_collection": self.last_collection_time.isoformat(),
            "registry_collectors": len(self.registry._collector_to_names) if PROMETHEUS_AVAILABLE else 0
        }