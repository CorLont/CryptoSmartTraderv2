#!/usr/bin/env python3
"""
CENTRALIZED METRICS SYSTEM
Consolidates ALL Prometheus metrics into single source of truth
Includes integrated alert rules and observability automation
"""

import time
import threading
import logging
from typing import Dict, Optional, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

from prometheus_client import (
    Counter, Histogram, Gauge, Summary, Info, Enum as PrometheusEnum,
    CollectorRegistry, generate_latest, start_http_server, CONTENT_TYPE_LATEST
)

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Supported metric types"""
    COUNTER = "counter"
    HISTOGRAM = "histogram" 
    GAUGE = "gauge"
    SUMMARY = "summary"
    INFO = "info"
    ENUM = "enum"


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class AlertRule:
    """Alert rule definition"""
    name: str
    description: str
    query: str
    severity: AlertSeverity
    threshold: float = 0.0
    for_duration: str = "1m"
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)


@dataclass 
class MetricDefinition:
    """Metric definition with metadata"""
    name: str
    help_text: str
    metric_type: MetricType
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None
    objectives: Optional[Dict[float, float]] = None
    enum_states: Optional[List[str]] = None
    alert_rules: List[AlertRule] = field(default_factory=list)


class CentralizedMetrics:
    """
    Centralized Prometheus metrics system for CryptoSmartTrader V2
    
    Consolidates ALL metrics from across the system into single registry
    Provides integrated alert rules and observability automation
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
            
        self.logger = logging.getLogger(__name__)
        self._initialized = True
        
        # Prometheus registry
        self.registry = CollectorRegistry()
        self.metrics: Dict[str, Any] = {}
        self.alert_rules: List[AlertRule] = []
        
        # HTTP server
        self.http_server_port: Optional[int] = None
        self.http_server_started = False
        
        # Initialize all metrics
        self._initialize_core_metrics()
        self._initialize_trading_metrics()
        self._initialize_risk_metrics()
        self._initialize_execution_metrics()
        self._initialize_ml_metrics()
        self._initialize_system_metrics()
        self._initialize_alert_rules()
        
        self.logger.info("🔍 Centralized Metrics System initialized with comprehensive observability")
    
    def _create_metric(self, definition: MetricDefinition) -> Any:
        """Create a Prometheus metric from definition"""
        
        kwargs = {
            'name': definition.name,
            'documentation': definition.help_text,
            'labelnames': definition.labels,
            'registry': self.registry
        }
        
        if definition.metric_type == MetricType.COUNTER:
            return Counter(**kwargs)
        
        elif definition.metric_type == MetricType.HISTOGRAM:
            if definition.buckets:
                kwargs['buckets'] = definition.buckets
            return Histogram(**kwargs)
        
        elif definition.metric_type == MetricType.GAUGE:
            return Gauge(**kwargs)
        
        elif definition.metric_type == MetricType.SUMMARY:
            if definition.objectives:
                kwargs['objectives'] = definition.objectives
            return Summary(**kwargs)
        
        elif definition.metric_type == MetricType.INFO:
            return Info(**kwargs)
        
        elif definition.metric_type == MetricType.ENUM:
            if definition.enum_states:
                kwargs['states'] = definition.enum_states
            return PrometheusEnum(**kwargs)
        
        else:
            raise ValueError(f"Unsupported metric type: {definition.metric_type}")
    
    def _initialize_core_metrics(self):
        """Initialize core system metrics"""
        
        core_metrics = [
            MetricDefinition(
                name="cst_system_health_status",
                help_text="Overall system health status (0=down, 1=degraded, 2=healthy)",
                metric_type=MetricType.GAUGE,
                alert_rules=[
                    AlertRule(
                        name="SystemDown",
                        description="CryptoSmartTrader system is down",
                        query="cst_system_health_status == 0",
                        severity=AlertSeverity.CRITICAL,
                        for_duration="30s"
                    ),
                    AlertRule(
                        name="SystemDegraded", 
                        description="CryptoSmartTrader system performance degraded",
                        query="cst_system_health_status == 1",
                        severity=AlertSeverity.HIGH,
                        for_duration="2m"
                    )
                ]
            ),
            
            MetricDefinition(
                name="cst_application_uptime_seconds", 
                help_text="Application uptime in seconds",
                metric_type=MetricType.GAUGE
            ),
            
            MetricDefinition(
                name="cst_errors_total",
                help_text="Total number of application errors",
                metric_type=MetricType.COUNTER,
                labels=["component", "error_type", "severity"],
                alert_rules=[
                    AlertRule(
                        name="HighErrorRate",
                        description="High error rate detected",
                        query="rate(cst_errors_total[5m]) > 0.1",
                        severity=AlertSeverity.HIGH,
                        threshold=0.1,
                        for_duration="2m"
                    )
                ]
            ),
            
            MetricDefinition(
                name="cst_request_duration_seconds",
                help_text="Request duration in seconds",
                metric_type=MetricType.HISTOGRAM,
                labels=["method", "endpoint", "status"],
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
            )
        ]
        
        for definition in core_metrics:
            metric = self._create_metric(definition)
            self.metrics[definition.name] = metric
            self.alert_rules.extend(definition.alert_rules)
    
    def _initialize_trading_metrics(self):
        """Initialize trading and portfolio metrics"""
        
        trading_metrics = [
            MetricDefinition(
                name="cst_trades_total",
                help_text="Total number of trades executed",
                metric_type=MetricType.COUNTER,
                labels=["symbol", "side", "strategy", "status"]
            ),
            
            MetricDefinition(
                name="cst_trade_pnl_usd",
                help_text="Trade P&L in USD",
                metric_type=MetricType.HISTOGRAM,
                labels=["symbol", "strategy"],
                buckets=[-1000, -500, -100, -50, -10, 0, 10, 50, 100, 500, 1000, 5000]
            ),
            
            MetricDefinition(
                name="cst_portfolio_value_usd",
                help_text="Current portfolio value in USD",
                metric_type=MetricType.GAUGE,
                alert_rules=[
                    AlertRule(
                        name="PortfolioDrawdownCritical",
                        description="Portfolio drawdown exceeds critical threshold",
                        query="(cst_portfolio_value_usd / cst_portfolio_peak_value_usd - 1) < -0.10",
                        severity=AlertSeverity.CRITICAL,
                        threshold=-0.10,
                        for_duration="1m"
                    )
                ]
            ),
            
            MetricDefinition(
                name="cst_portfolio_peak_value_usd",
                help_text="Portfolio peak value in USD",
                metric_type=MetricType.GAUGE
            ),
            
            MetricDefinition(
                name="cst_daily_pnl_usd",
                help_text="Daily P&L in USD",
                metric_type=MetricType.GAUGE,
                alert_rules=[
                    AlertRule(
                        name="DailyLossLimit",
                        description="Daily loss limit exceeded",
                        query="cst_daily_pnl_usd < -1000",
                        severity=AlertSeverity.CRITICAL,
                        threshold=-1000,
                        for_duration="30s"
                    )
                ]
            ),
            
            MetricDefinition(
                name="cst_position_count",
                help_text="Number of open positions",
                metric_type=MetricType.GAUGE,
                alert_rules=[
                    AlertRule(
                        name="TooManyPositions",
                        description="Position count exceeds limit",
                        query="cst_position_count > 10",
                        severity=AlertSeverity.HIGH,
                        threshold=10,
                        for_duration="1m"
                    )
                ]
            ),
            
            MetricDefinition(
                name="cst_total_exposure_usd",
                help_text="Total portfolio exposure in USD", 
                metric_type=MetricType.GAUGE
            )
        ]
        
        for definition in trading_metrics:
            metric = self._create_metric(definition)
            self.metrics[definition.name] = metric
            self.alert_rules.extend(definition.alert_rules)
    
    def _initialize_risk_metrics(self):
        """Initialize risk management metrics"""
        
        risk_metrics = [
            MetricDefinition(
                name="cst_risk_checks_total",
                help_text="Total number of risk checks performed",
                metric_type=MetricType.COUNTER,
                labels=["check_type", "result"]
            ),
            
            MetricDefinition(
                name="cst_risk_violations_total",
                help_text="Total number of risk violations detected",
                metric_type=MetricType.COUNTER,
                labels=["violation_type", "severity"],
                alert_rules=[
                    AlertRule(
                        name="RiskViolationSpike",
                        description="Spike in risk violations detected",
                        query="rate(cst_risk_violations_total[5m]) > 0.5",
                        severity=AlertSeverity.HIGH,
                        threshold=0.5,
                        for_duration="1m"
                    )
                ]
            ),
            
            MetricDefinition(
                name="cst_kill_switch_status",
                help_text="Kill switch status (0=off, 1=on)",
                metric_type=MetricType.GAUGE,
                alert_rules=[
                    AlertRule(
                        name="KillSwitchActivated",
                        description="Emergency kill switch has been activated",
                        query="cst_kill_switch_status == 1",
                        severity=AlertSeverity.CRITICAL,
                        for_duration="0s"
                    )
                ]
            ),
            
            MetricDefinition(
                name="cst_var_usd",
                help_text="Value at Risk in USD",
                metric_type=MetricType.GAUGE
            ),
            
            MetricDefinition(
                name="cst_correlation_limits_breached",
                help_text="Number of correlation limit breaches",
                metric_type=MetricType.COUNTER,
                labels=["asset_pair"]
            )
        ]
        
        for definition in risk_metrics:
            metric = self._create_metric(definition)
            self.metrics[definition.name] = metric
            self.alert_rules.extend(definition.alert_rules)
    
    def _initialize_execution_metrics(self):
        """Initialize execution and order metrics"""
        
        execution_metrics = [
            MetricDefinition(
                name="cst_orders_total",
                help_text="Total number of orders submitted",
                metric_type=MetricType.COUNTER,
                labels=["symbol", "side", "type", "status"]
            ),
            
            MetricDefinition(
                name="cst_order_latency_seconds",
                help_text="Order execution latency in seconds",
                metric_type=MetricType.HISTOGRAM,
                labels=["exchange", "order_type"],
                buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
                alert_rules=[
                    AlertRule(
                        name="HighOrderLatency",
                        description="Order execution latency is high",
                        query="histogram_quantile(0.95, rate(cst_order_latency_seconds_bucket[5m])) > 0.5",
                        severity=AlertSeverity.HIGH,
                        threshold=0.5,
                        for_duration="2m"
                    )
                ]
            ),
            
            MetricDefinition(
                name="cst_slippage_bps",
                help_text="Order slippage in basis points",
                metric_type=MetricType.HISTOGRAM,
                labels=["symbol", "side"],
                buckets=[0, 1, 5, 10, 25, 50, 100, 250, 500]
            ),
            
            MetricDefinition(
                name="cst_execution_policy_gates_total",
                help_text="ExecutionPolicy gate evaluations",
                metric_type=MetricType.COUNTER,
                labels=["gate_type", "result"]
            ),
            
            MetricDefinition(
                name="cst_fill_rate_ratio",
                help_text="Order fill rate (0.0 to 1.0)",
                metric_type=MetricType.GAUGE,
                labels=["symbol", "order_type"],
                alert_rules=[
                    AlertRule(
                        name="LowFillRate",
                        description="Order fill rate is low",
                        query="cst_fill_rate_ratio < 0.8",
                        severity=AlertSeverity.MEDIUM,
                        threshold=0.8,
                        for_duration="5m"
                    )
                ]
            )
        ]
        
        for definition in execution_metrics:
            metric = self._create_metric(definition)
            self.metrics[definition.name] = metric
            self.alert_rules.extend(definition.alert_rules)
    
    def _initialize_ml_metrics(self):
        """Initialize ML and prediction metrics"""
        
        ml_metrics = [
            MetricDefinition(
                name="cst_model_predictions_total",
                help_text="Total number of model predictions",
                metric_type=MetricType.COUNTER,
                labels=["model_name", "regime", "asset"]
            ),
            
            MetricDefinition(
                name="cst_model_accuracy",
                help_text="Model prediction accuracy",
                metric_type=MetricType.GAUGE,
                labels=["model_name", "timeframe"],
                alert_rules=[
                    AlertRule(
                        name="ModelAccuracyDegraded",
                        description="Model accuracy has degraded",
                        query="cst_model_accuracy < 0.6",
                        severity=AlertSeverity.MEDIUM,
                        threshold=0.6,
                        for_duration="10m"
                    )
                ]
            ),
            
            MetricDefinition(
                name="cst_feature_importance",
                help_text="Feature importance scores",
                metric_type=MetricType.GAUGE,
                labels=["feature_name", "model_name"]
            ),
            
            MetricDefinition(
                name="cst_model_training_duration_seconds",
                help_text="Model training duration",
                metric_type=MetricType.HISTOGRAM,
                labels=["model_name"],
                buckets=[1, 5, 15, 30, 60, 300, 600, 1800, 3600]
            ),
            
            MetricDefinition(
                name="cst_regime_classification",
                help_text="Current market regime classification",
                metric_type=MetricType.ENUM,
                labels=["asset"],
                enum_states=["bull", "bear", "sideways", "volatile", "low_vol", "trending"]
            )
        ]
        
        for definition in ml_metrics:
            metric = self._create_metric(definition)
            self.metrics[definition.name] = metric
            self.alert_rules.extend(definition.alert_rules)
    
    def _initialize_system_metrics(self):
        """Initialize system performance metrics"""
        
        system_metrics = [
            MetricDefinition(
                name="cst_memory_usage_bytes",
                help_text="Memory usage in bytes",
                metric_type=MetricType.GAUGE,
                labels=["component"]
            ),
            
            MetricDefinition(
                name="cst_cpu_usage_percent",
                help_text="CPU usage percentage",
                metric_type=MetricType.GAUGE,
                labels=["component"],
                alert_rules=[
                    AlertRule(
                        name="HighCPUUsage",
                        description="High CPU usage detected",
                        query="cst_cpu_usage_percent > 80",
                        severity=AlertSeverity.HIGH,
                        threshold=80,
                        for_duration="5m"
                    )
                ]
            ),
            
            MetricDefinition(
                name="cst_database_connections",
                help_text="Number of active database connections",
                metric_type=MetricType.GAUGE
            ),
            
            MetricDefinition(
                name="cst_api_requests_total",
                help_text="Total number of API requests",
                metric_type=MetricType.COUNTER,
                labels=["endpoint", "method", "status_code"]
            ),
            
            MetricDefinition(
                name="cst_queue_size",
                help_text="Queue size for various components",
                metric_type=MetricType.GAUGE,
                labels=["queue_name"],
                alert_rules=[
                    AlertRule(
                        name="QueueBacklog",
                        description="Queue backlog detected",
                        query="cst_queue_size > 1000",
                        severity=AlertSeverity.MEDIUM,
                        threshold=1000,
                        for_duration="3m"
                    )
                ]
            )
        ]
        
        for definition in system_metrics:
            metric = self._create_metric(definition)
            self.metrics[definition.name] = metric
            self.alert_rules.extend(definition.alert_rules)
    
    def _initialize_alert_rules(self):
        """Initialize comprehensive alert rules"""
        
        # Additional cross-metric alert rules
        additional_alerts = [
            AlertRule(
                name="SystemOverloaded",
                description="System is overloaded (high CPU + memory + errors)",
                query="(cst_cpu_usage_percent > 70) and (cst_memory_usage_bytes > 8e9) and (rate(cst_errors_total[5m]) > 0.05)",
                severity=AlertSeverity.CRITICAL,
                for_duration="2m",
                labels={"component": "system"},
                annotations={
                    "summary": "System is experiencing high load with elevated error rates",
                    "runbook": "Check system resources and scale if necessary"
                }
            ),
            
            AlertRule(
                name="TradingAnomalyDetected",
                description="Trading activity anomaly detected",
                query="(rate(cst_trades_total[5m]) > 10) and (rate(cst_risk_violations_total[5m]) > 0.1)",
                severity=AlertSeverity.HIGH,
                for_duration="1m",
                labels={"component": "trading"},
                annotations={
                    "summary": "High trading activity with elevated risk violations",
                    "runbook": "Review trading strategies and risk parameters"
                }
            ),
            
            AlertRule(
                name="DataIntegrityIssue",
                description="Data integrity issues detected",
                query="rate(cst_errors_total{error_type=\"data_integrity\"}[5m]) > 0",
                severity=AlertSeverity.CRITICAL,
                for_duration="30s",
                labels={"component": "data"},
                annotations={
                    "summary": "Data integrity violations detected",
                    "runbook": "Immediately investigate data sources and halt trading if necessary"
                }
            )
        ]
        
        self.alert_rules.extend(additional_alerts)
    
    # Metric Recording Methods
    
    def record_system_health(self, status: int):
        """Record system health status"""
        self.metrics["cst_system_health_status"].set(status)
    
    def record_error(self, component: str, error_type: str, severity: str):
        """Record an error occurrence"""
        self.metrics["cst_errors_total"].labels(
            component=component,
            error_type=error_type,
            severity=severity
        ).inc()
    
    def record_trade(self, symbol: str, side: str, strategy: str, status: str, pnl_usd: float = 0.0):
        """Record a trade execution"""
        self.metrics["cst_trades_total"].labels(
            symbol=symbol,
            side=side,
            strategy=strategy,
            status=status
        ).inc()
        
        if pnl_usd != 0.0:
            self.metrics["cst_trade_pnl_usd"].labels(
                symbol=symbol,
                strategy=strategy
            ).observe(pnl_usd)
    
    def record_portfolio_value(self, value_usd: float, daily_pnl: float):
        """Record portfolio values"""
        self.metrics["cst_portfolio_value_usd"].set(value_usd)
        self.metrics["cst_daily_pnl_usd"].set(daily_pnl)
    
    def record_risk_check(self, check_type: str, result: str):
        """Record a risk check"""
        self.metrics["cst_risk_checks_total"].labels(
            check_type=check_type,
            result=result
        ).inc()
    
    def record_order(self, symbol: str, side: str, order_type: str, status: str, latency_seconds: float = 0.0):
        """Record an order submission"""
        self.metrics["cst_orders_total"].labels(
            symbol=symbol,
            side=side,
            type=order_type,
            status=status
        ).inc()
        
        if latency_seconds > 0:
            self.metrics["cst_order_latency_seconds"].labels(
                exchange="kraken",
                order_type=order_type
            ).observe(latency_seconds)
    
    def record_execution_policy_gate(self, gate_type: str, result: str):
        """Record ExecutionPolicy gate evaluation"""
        self.metrics["cst_execution_policy_gates_total"].labels(
            gate_type=gate_type,
            result=result
        ).inc()
    
    def set_kill_switch_status(self, active: bool):
        """Set kill switch status"""
        self.metrics["cst_kill_switch_status"].set(1 if active else 0)
    
    def record_model_prediction(self, model_name: str, regime: str, asset: str, accuracy: float = None):
        """Record ML model prediction"""
        self.metrics["cst_model_predictions_total"].labels(
            model_name=model_name,
            regime=regime,
            asset=asset
        ).inc()
        
        if accuracy is not None:
            self.metrics["cst_model_accuracy"].labels(
                model_name=model_name,
                timeframe="1h"
            ).set(accuracy)
    
    # Alert Rules Export
    
    def export_alert_rules(self, format: str = "prometheus") -> str:
        """Export alert rules in specified format"""
        
        if format == "prometheus":
            return self._export_prometheus_alert_rules()
        elif format == "alertmanager":
            return self._export_alertmanager_config()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_prometheus_alert_rules(self) -> str:
        """Export alert rules in Prometheus format"""
        
        rules_yaml = "groups:\n"
        rules_yaml += "- name: cryptosmarttrader_alerts\n"
        rules_yaml += "  rules:\n"
        
        for rule in self.alert_rules:
            rules_yaml += f"  - alert: {rule.name}\n"
            rules_yaml += f"    expr: {rule.query}\n"
            rules_yaml += f"    for: {rule.for_duration}\n"
            rules_yaml += f"    labels:\n"
            rules_yaml += f"      severity: {rule.severity.value}\n"
            
            for label, value in rule.labels.items():
                rules_yaml += f"      {label}: {value}\n"
            
            rules_yaml += f"    annotations:\n"
            rules_yaml += f"      summary: {rule.description}\n"
            
            for annotation, value in rule.annotations.items():
                rules_yaml += f"      {annotation}: {value}\n"
            
            rules_yaml += "\n"
        
        return rules_yaml
    
    def _export_alertmanager_config(self) -> str:
        """Export AlertManager configuration"""
        
        config = {
            "global": {
                "smtp_smarthost": "localhost:587",
                "smtp_from": "alerts@cryptosmarttrader.com"
            },
            "route": {
                "group_by": ["alertname"],
                "group_wait": "10s",
                "group_interval": "10s",
                "repeat_interval": "1h",
                "receiver": "web.hook"
            },
            "receivers": [
                {
                    "name": "web.hook",
                    "webhook_configs": [
                        {
                            "url": "http://localhost:5001/webhook",
                            "send_resolved": True
                        }
                    ]
                }
            ]
        }
        
        return json.dumps(config, indent=2)
    
    # HTTP Server Management
    
    def start_metrics_server(self, port: int = 8000):
        """Start Prometheus metrics HTTP server"""
        
        if self.http_server_started:
            self.logger.warning(f"Metrics server already running on port {self.http_server_port}")
            return
        
        try:
            start_http_server(port, registry=self.registry)
            self.http_server_port = port
            self.http_server_started = True
            self.logger.info(f"🔍 Metrics server started on port {port}")
        except Exception as e:
            self.logger.error(f"Failed to start metrics server: {str(e)}")
            raise
    
    def get_metrics_text(self) -> str:
        """Get metrics in Prometheus text format"""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_metric_names(self) -> List[str]:
        """Get list of all registered metric names"""
        return list(self.metrics.keys())
    
    def get_alert_count(self) -> int:
        """Get total number of alert rules"""
        return len(self.alert_rules)
    
    def get_status(self) -> Dict[str, Any]:
        """Get centralized metrics system status"""
        return {
            "metrics_count": len(self.metrics),
            "alert_rules_count": len(self.alert_rules),
            "http_server_active": self.http_server_started,
            "http_server_port": self.http_server_port,
            "registry_size": len(list(self.registry._collector_to_names.keys()))
        }


# Global instance
centralized_metrics = CentralizedMetrics()


# Convenience functions for easy usage across the codebase
def record_system_health(status: int):
    """Record system health status"""
    centralized_metrics.record_system_health(status)


def record_error(component: str, error_type: str, severity: str = "medium"):
    """Record an error occurrence"""
    centralized_metrics.record_error(component, error_type, severity)


def record_trade(symbol: str, side: str, strategy: str, status: str, pnl_usd: float = 0.0):
    """Record a trade execution"""
    centralized_metrics.record_trade(symbol, side, strategy, status, pnl_usd)


def record_risk_check(check_type: str, result: str):
    """Record a risk check"""
    centralized_metrics.record_risk_check(check_type, result)


def record_order(symbol: str, side: str, order_type: str, status: str, latency_seconds: float = 0.0):
    """Record an order submission"""
    centralized_metrics.record_order(symbol, side, order_type, status, latency_seconds)


def record_execution_policy_gate(gate_type: str, result: str):
    """Record ExecutionPolicy gate evaluation"""
    centralized_metrics.record_execution_policy_gate(gate_type, result)


def set_kill_switch_status(active: bool):
    """Set kill switch status"""
    centralized_metrics.set_kill_switch_status(active)


def start_metrics_server(port: int = 8000):
    """Start Prometheus metrics HTTP server"""
    centralized_metrics.start_metrics_server(port)


def get_metrics_status() -> Dict[str, Any]:
    """Get centralized metrics system status"""
    return centralized_metrics.get_status()


def export_alert_rules(format: str = "prometheus") -> str:
    """Export alert rules in specified format"""
    return centralized_metrics.export_alert_rules(format)