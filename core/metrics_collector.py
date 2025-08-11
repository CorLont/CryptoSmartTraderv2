#!/usr/bin/env python3
"""
Enterprise Metrics Collection - Prometheus metrics with controlled cardinality
"""

import time
import logging
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from threading import Lock

from prometheus_client import Counter, Histogram, Gauge, Info, Summary, CollectorRegistry, generate_latest


class MetricType(Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    INFO = "info"


@dataclass
class MetricConfig:
    """Metric configuration"""
    name: str
    help_text: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None
    max_cardinality: int = 1000


class CardinalityController:
    """
    Control metric label cardinality to prevent unbounded growth
    """
    
    def __init__(self, max_global_cardinality: int = 10000):
        self.max_global_cardinality = max_global_cardinality
        self.label_sets: Dict[str, Set[tuple]] = defaultdict(set)
        self.cardinality_warnings: Dict[str, int] = defaultdict(int)
        self.lock = Lock()
        self.logger = logging.getLogger(__name__)
        
        # Allowed values for high-cardinality labels
        self.allowed_symbols = self._get_allowed_symbols()
        self.allowed_exchanges = {"kraken", "binance", "coinbase", "kucoin", "huobi"}
        self.allowed_signal_types = {"buy", "sell", "hold", "strong_buy", "strong_sell"}
    
    def _get_allowed_symbols(self) -> Set[str]:
        """Get allowed trading symbols (top coins + major pairs)"""
        # Top 100 cryptocurrencies by market cap + major fiat pairs
        top_symbols = {
            "BTC/USD", "ETH/USD", "BNB/USD", "XRP/USD", "ADA/USD",
            "SOL/USD", "DOGE/USD", "DOT/USD", "MATIC/USD", "LTC/USD",
            "SHIB/USD", "TRX/USD", "AVAX/USD", "UNI/USD", "ATOM/USD",
            "LINK/USD", "XLM/USD", "BCH/USD", "ALGO/USD", "VET/USD",
            "ICP/USD", "FIL/USD", "APE/USD", "MANA/USD", "SAND/USD",
            "CRO/USD", "LRC/USD", "FTM/USD", "AXS/USD", "THETA/USD",
            # Add EUR pairs for major coins
            "BTC/EUR", "ETH/EUR", "BNB/EUR", "XRP/EUR", "ADA/EUR",
            # Add BTC pairs
            "ETH/BTC", "BNB/BTC", "XRP/BTC", "ADA/BTC", "DOT/BTC"
        }
        
        return top_symbols
    
    def validate_labels(self, metric_name: str, labels: Dict[str, str]) -> Dict[str, str]:
        """
        Validate and sanitize metric labels to control cardinality
        
        Args:
            metric_name: Name of the metric
            labels: Label dictionary
            
        Returns:
            Sanitized label dictionary
        """
        
        with self.lock:
            # Sanitize labels
            sanitized_labels = {}
            
            for key, value in labels.items():
                if key == "symbol":
                    # Control symbol cardinality
                    if value in self.allowed_symbols:
                        sanitized_labels[key] = value
                    else:
                        sanitized_labels[key] = "OTHER"
                        
                elif key == "exchange":
                    # Control exchange cardinality
                    if value.lower() in self.allowed_exchanges:
                        sanitized_labels[key] = value.lower()
                    else:
                        sanitized_labels[key] = "other"
                        
                elif key == "signal_type":
                    # Control signal type cardinality
                    if value.lower() in self.allowed_signal_types:
                        sanitized_labels[key] = value.lower()
                    else:
                        sanitized_labels[key] = "other"
                        
                elif key in ["user_id", "session_id", "trade_id"]:
                    # Hash high-cardinality IDs to buckets
                    hash_bucket = hash(value) % 100
                    sanitized_labels[key] = f"bucket_{hash_bucket:02d}"
                    
                elif key == "status_code":
                    # Group HTTP status codes
                    if value.startswith("2"):
                        sanitized_labels[key] = "2xx"
                    elif value.startswith("4"):
                        sanitized_labels[key] = "4xx"
                    elif value.startswith("5"):
                        sanitized_labels[key] = "5xx"
                    else:
                        sanitized_labels[key] = "other"
                        
                else:
                    # For other labels, truncate long values
                    if len(str(value)) > 50:
                        sanitized_labels[key] = str(value)[:47] + "..."
                    else:
                        sanitized_labels[key] = str(value)
            
            # Check cardinality
            label_tuple = tuple(sorted(sanitized_labels.items()))
            self.label_sets[metric_name].add(label_tuple)
            
            current_cardinality = len(self.label_sets[metric_name])
            
            # Warn if cardinality is high
            if current_cardinality > 500 and current_cardinality % 100 == 0:
                if self.cardinality_warnings[metric_name] < 5:  # Limit warnings
                    self.logger.warning(
                        f"High cardinality for metric {metric_name}: {current_cardinality} series"
                    )
                    self.cardinality_warnings[metric_name] += 1
            
            # Reset if too high (emergency measure)
            if current_cardinality > 1000:
                self.logger.error(
                    f"Resetting metric {metric_name} due to excessive cardinality: {current_cardinality}"
                )
                self.label_sets[metric_name].clear()
                sanitized_labels = {"error": "cardinality_reset"}
            
            return sanitized_labels


class EnterpriseMetricsCollector:
    """
    Enterprise metrics collector with cardinality control
    """
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.cardinality_controller = CardinalityController()
        self.metrics: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize core business metrics
        self._init_trading_metrics()
        self._init_system_metrics()
        self._init_api_metrics()
        self._init_ml_metrics()
    
    def _init_trading_metrics(self):
        """Initialize trading-specific metrics"""
        
        # Trade execution metrics
        self.metrics["trades_total"] = Counter(
            "cryptotrader_trades_total",
            "Total number of trades executed",
            ["symbol", "exchange", "side", "status"],
            registry=self.registry
        )
        
        self.metrics["trade_value_usd"] = Histogram(
            "cryptotrader_trade_value_usd",
            "Trade value in USD",
            ["symbol", "exchange", "side"],
            buckets=[100, 500, 1000, 5000, 10000, 50000, 100000, 500000, float('inf')],
            registry=self.registry
        )
        
        self.metrics["slippage_bps"] = Histogram(
            "cryptotrader_slippage_bps",
            "Trade slippage in basis points",
            ["symbol", "exchange"],
            buckets=[0, 5, 10, 25, 50, 100, 250, 500, float('inf')],
            registry=self.registry
        )
        
        self.metrics["trading_fees_usd"] = Counter(
            "cryptotrader_trading_fees_usd_total",
            "Total trading fees paid in USD",
            ["exchange", "fee_type"],
            registry=self.registry
        )
        
        # Portfolio metrics
        self.metrics["portfolio_value_usd"] = Gauge(
            "cryptotrader_portfolio_value_usd",
            "Current portfolio value in USD",
            registry=self.registry
        )
        
        self.metrics["position_count"] = Gauge(
            "cryptotrader_positions_count",
            "Number of open positions",
            ["exchange"],
            registry=self.registry
        )
        
        self.metrics["unrealized_pnl_usd"] = Gauge(
            "cryptotrader_unrealized_pnl_usd",
            "Unrealized P&L in USD",
            ["symbol"],
            registry=self.registry
        )
        
        self.metrics["realized_pnl_usd"] = Counter(
            "cryptotrader_realized_pnl_usd_total",
            "Cumulative realized P&L in USD",
            ["symbol"],
            registry=self.registry
        )
        
        # Signal metrics
        self.metrics["signals_generated"] = Counter(
            "cryptotrader_signals_generated_total",
            "Total signals generated",
            ["symbol", "signal_type", "confidence_bucket"],
            registry=self.registry
        )
        
        self.metrics["signal_accuracy"] = Gauge(
            "cryptotrader_signal_accuracy_ratio",
            "Signal accuracy ratio (0-1)",
            ["symbol", "signal_type", "timeframe"],
            registry=self.registry
        )
    
    def _init_system_metrics(self):
        """Initialize system health metrics"""
        
        self.metrics["system_health_score"] = Gauge(
            "cryptotrader_system_health_score",
            "Overall system health score (0-1)",
            registry=self.registry
        )
        
        self.metrics["component_health"] = Gauge(
            "cryptotrader_component_health_score",
            "Component health score (0-1)",
            ["component"],
            registry=self.registry
        )
        
        self.metrics["data_drift_score"] = Gauge(
            "cryptotrader_data_drift_score",
            "Data drift score (0-1)",
            ["model", "feature_group"],
            registry=self.registry
        )
        
        self.metrics["model_inference_count"] = Counter(
            "cryptotrader_model_inference_total",
            "Model inference requests",
            ["model", "version", "status"],
            registry=self.registry
        )
        
        self.metrics["circuit_breaker_state"] = Gauge(
            "cryptotrader_circuit_breaker_state",
            "Circuit breaker state (0=closed, 1=open, 2=half-open)",
            ["service"],
            registry=self.registry
        )
    
    def _init_api_metrics(self):
        """Initialize API metrics"""
        
        self.metrics["http_requests_total"] = Counter(
            "cryptotrader_http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status_code"],
            registry=self.registry
        )
        
        self.metrics["http_request_duration"] = Histogram(
            "cryptotrader_http_request_duration_seconds",
            "HTTP request duration",
            ["method", "endpoint"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, float('inf')],
            registry=self.registry
        )
        
        self.metrics["api_rate_limit_hits"] = Counter(
            "cryptotrader_api_rate_limit_hits_total",
            "API rate limit violations",
            ["service", "endpoint"],
            registry=self.registry
        )
        
        self.metrics["cache_operations"] = Counter(
            "cryptotrader_cache_operations_total",
            "Cache operations",
            ["operation", "status"],
            registry=self.registry
        )
    
    def _init_ml_metrics(self):
        """Initialize ML/AI metrics"""
        
        self.metrics["model_training_duration"] = Histogram(
            "cryptotrader_model_training_duration_seconds",
            "Model training duration",
            ["model", "algorithm"],
            buckets=[60, 300, 900, 1800, 3600, 7200, 14400, float('inf')],
            registry=self.registry
        )
        
        self.metrics["model_accuracy"] = Gauge(
            "cryptotrader_model_accuracy_score",
            "Model accuracy score (0-1)",
            ["model", "version", "dataset"],
            registry=self.registry
        )
        
        self.metrics["prediction_confidence"] = Histogram(
            "cryptotrader_prediction_confidence",
            "Prediction confidence distribution",
            ["model", "symbol"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=self.registry
        )
    
    def record_trade(
        self,
        symbol: str,
        exchange: str,
        side: str,
        value_usd: float,
        slippage_bps: float,
        fees_usd: float,
        status: str = "executed"
    ):
        """Record trade execution metrics"""
        
        labels = self.cardinality_controller.validate_labels(
            "trades",
            {
                "symbol": symbol,
                "exchange": exchange,
                "side": side,
                "status": status
            }
        )
        
        self.metrics["trades_total"].labels(**labels).inc()
        
        value_labels = {k: v for k, v in labels.items() if k != "status"}
        self.metrics["trade_value_usd"].labels(**value_labels).observe(value_usd)
        
        slippage_labels = {k: v for k, v in labels.items() if k not in ["side", "status"]}
        self.metrics["slippage_bps"].labels(**slippage_labels).observe(slippage_bps)
        
        fee_labels = self.cardinality_controller.validate_labels(
            "fees",
            {"exchange": exchange, "fee_type": "trading"}
        )
        self.metrics["trading_fees_usd"].labels(**fee_labels).inc(fees_usd)
    
    def record_signal(
        self,
        symbol: str,
        signal_type: str,
        confidence: float,
        timeframe: str = "1h"
    ):
        """Record trading signal metrics"""
        
        # Bucket confidence for cardinality control
        if confidence >= 0.8:
            confidence_bucket = "high"
        elif confidence >= 0.6:
            confidence_bucket = "medium"
        else:
            confidence_bucket = "low"
        
        labels = self.cardinality_controller.validate_labels(
            "signals",
            {
                "symbol": symbol,
                "signal_type": signal_type,
                "confidence_bucket": confidence_bucket
            }
        )
        
        self.metrics["signals_generated"].labels(**labels).inc()
    
    def update_health_score(self, score: float, component: Optional[str] = None):
        """Update health score metrics"""
        
        if component:
            labels = self.cardinality_controller.validate_labels(
                "component_health",
                {"component": component}
            )
            self.metrics["component_health"].labels(**labels).set(score)
        else:
            self.metrics["system_health_score"].set(score)
    
    def record_http_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float
    ):
        """Record HTTP request metrics"""
        
        labels = self.cardinality_controller.validate_labels(
            "http_requests",
            {
                "method": method.upper(),
                "endpoint": endpoint,
                "status_code": str(status_code)
            }
        )
        
        self.metrics["http_requests_total"].labels(**labels).inc()
        
        duration_labels = {k: v for k, v in labels.items() if k != "status_code"}
        self.metrics["http_request_duration"].labels(**duration_labels).observe(duration_seconds)
    
    def record_model_inference(
        self,
        model: str,
        version: str,
        status: str = "success"
    ):
        """Record model inference metrics"""
        
        labels = self.cardinality_controller.validate_labels(
            "model_inference",
            {
                "model": model,
                "version": version,
                "status": status
            }
        )
        
        self.metrics["model_inference_count"].labels(**labels).inc()
    
    def set_circuit_breaker_state(self, service: str, state: str):
        """Set circuit breaker state"""
        
        state_value = {"closed": 0, "open": 1, "half_open": 2}.get(state, 0)
        
        labels = self.cardinality_controller.validate_labels(
            "circuit_breaker",
            {"service": service}
        )
        
        self.metrics["circuit_breaker_state"].labels(**labels).set(state_value)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_cardinality_stats(self) -> Dict[str, int]:
        """Get cardinality statistics"""
        return {
            metric_name: len(label_sets)
            for metric_name, label_sets in self.cardinality_controller.label_sets.items()
        }


# Global metrics collector
metrics_collector = EnterpriseMetricsCollector()