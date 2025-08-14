#!/usr/bin/env python3
"""
Centralized Metrics System
Enterprise observability met Prometheus integration
"""

import time
import threading
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class AlertRule:
    """Prometheus alert rule structure"""
    name: str
    description: str
    query: str
    severity: AlertSeverity
    threshold: Optional[float] = None
    for_duration: str = "5m"
    labels: Optional[Dict[str, str]] = None
    annotations: Optional[Dict[str, str]] = None


class PrometheusMetric:
    """Prometheus metric wrapper"""
    
    def __init__(self, name: str, metric_type: MetricType, description: str, labels: Optional[List[str]] = None):
        self.name = name
        self.type = metric_type
        self.description = description
        self.labels = labels or []
        self.values: Dict[str, Union[float, int]] = {}
        self._lock = threading.Lock()
    
    def set(self, value: Union[float, int], label_values: Optional[Dict[str, str]] = None):
        """Set metric value"""
        with self._lock:
            label_key = self._make_label_key(label_values)
            self.values[label_key] = value
    
    def inc(self, amount: Union[float, int] = 1, label_values: Optional[Dict[str, str]] = None):
        """Increment counter metric"""
        with self._lock:
            label_key = self._make_label_key(label_values)
            self.values[label_key] = self.values.get(label_key, 0) + amount
    
    def observe(self, value: Union[float, int], label_values: Optional[Dict[str, str]] = None):
        """Observe value for histogram/summary"""
        with self._lock:
            label_key = self._make_label_key(label_values)
            if label_key not in self.values:
                self.values[label_key] = []
            self.values[label_key].append(value)
    
    def _make_label_key(self, label_values: Optional[Dict[str, str]]) -> str:
        """Create label key voor internal storage"""
        if not label_values:
            return ""
        
        return ",".join(f"{k}={v}" for k, v in sorted(label_values.items()))
    
    def export_prometheus(self) -> str:
        """Export metric in Prometheus format"""
        lines = []
        
        # Add HELP line
        lines.append(f"# HELP {self.name} {self.description}")
        
        # Add TYPE line
        lines.append(f"# TYPE {self.name} {self.type.value}")
        
        # Add metric values
        for label_key, value in self.values.items():
            if label_key:
                metric_line = f"{self.name}{{{label_key}}} {value}"
            else:
                metric_line = f"{self.name} {value}"
            lines.append(metric_line)
        
        return "\n".join(lines)


class CentralizedMetrics:
    """Centralized metrics system (Singleton)"""
    
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
        self.metrics: Dict[str, PrometheusMetric] = {}
        self.start_time = time.time()
        self._initialize_core_metrics()
        self._initialize_alert_rules()
        
        logger.info("CentralizedMetrics system initialized")
    
    def _initialize_core_metrics(self):
        """Initialize core enterprise metrics"""
        
        # System health metrics
        self.metrics["cst_system_health_status"] = PrometheusMetric(
            "cst_system_health_status",
            MetricType.GAUGE,
            "Overall system health status (1=healthy, 0=unhealthy)",
            ["component"]
        )
        
        # Error metrics
        self.metrics["cst_errors_total"] = PrometheusMetric(
            "cst_errors_total",
            MetricType.COUNTER,
            "Total number of errors by component and type",
            ["component", "error_type", "severity"]
        )
        
        # Trading metrics
        self.metrics["cst_trades_total"] = PrometheusMetric(
            "cst_trades_total",
            MetricType.COUNTER,
            "Total number of trades executed",
            ["symbol", "side", "strategy", "status"]
        )
        
        self.metrics["cst_trade_pnl_usd"] = PrometheusMetric(
            "cst_trade_pnl_usd",
            MetricType.GAUGE,
            "Trade PnL in USD",
            ["symbol", "strategy"]
        )
        
        # Risk metrics
        self.metrics["cst_risk_evaluations_total"] = PrometheusMetric(
            "cst_risk_evaluations_total",
            MetricType.COUNTER,
            "Total risk evaluations",
            ["decision", "reason"]
        )
        
        self.metrics["cst_risk_limits_breached_total"] = PrometheusMetric(
            "cst_risk_limits_breached_total",
            MetricType.COUNTER,
            "Risk limit breaches",
            ["limit_type", "severity"]
        )
        
        # Execution metrics
        self.metrics["cst_order_latency_seconds"] = PrometheusMetric(
            "cst_order_latency_seconds",
            MetricType.HISTOGRAM,
            "Order execution latency in seconds",
            ["operation", "exchange"]
        )
        
        self.metrics["cst_execution_decisions_total"] = PrometheusMetric(
            "cst_execution_decisions_total",
            MetricType.COUNTER,
            "Execution decisions by type",
            ["decision", "reason"]
        )
        
        # Data ingestion metrics
        self.metrics["cst_data_requests_total"] = PrometheusMetric(
            "cst_data_requests_total",
            MetricType.COUNTER,
            "Data requests by source and status",
            ["source", "status"]
        )
        
        self.metrics["cst_data_latency_seconds"] = PrometheusMetric(
            "cst_data_latency_seconds",
            MetricType.HISTOGRAM,
            "Data request latency",
            ["source", "endpoint"]
        )
        
        # Portfolio metrics
        self.metrics["cst_portfolio_value_usd"] = PrometheusMetric(
            "cst_portfolio_value_usd",
            MetricType.GAUGE,
            "Total portfolio value in USD"
        )
        
        self.metrics["cst_daily_pnl_usd"] = PrometheusMetric(
            "cst_daily_pnl_usd",
            MetricType.GAUGE,
            "Daily PnL in USD"
        )
        
        # Alert metrics
        self.metrics["cst_alerts_fired_total"] = PrometheusMetric(
            "cst_alerts_fired_total",
            MetricType.COUNTER,
            "Alerts fired by severity",
            ["severity", "alert_name"]
        )
    
    def _initialize_alert_rules(self):
        """Initialize critical alert rules"""
        self.alert_rules = [
            AlertRule(
                name="HighErrorRate",
                description="High error rate detected",
                query="rate(cst_errors_total[5m]) > 0.05",
                severity=AlertSeverity.CRITICAL,
                threshold=0.05,
                for_duration="2m",
                labels={"team": "trading"},
                annotations={"summary": "Error rate above 5% for 2 minutes"}
            ),
            
            AlertRule(
                name="HighOrderLatency",
                description="Order execution latency too high",
                query="histogram_quantile(0.95, cst_order_latency_seconds_bucket) > 0.5",
                severity=AlertSeverity.HIGH,
                threshold=0.5,
                for_duration="5m",
                labels={"team": "trading"},
                annotations={"summary": "95th percentile order latency above 500ms"}
            ),
            
            AlertRule(
                name="KillSwitchActivated",
                description="Trading kill switch has been activated",
                query="cst_system_health_status{component=\"risk_guard\"} == 0",
                severity=AlertSeverity.CRITICAL,
                threshold=0,
                for_duration="0s",
                labels={"team": "trading", "escalation": "immediate"},
                annotations={"summary": "Emergency kill switch activated - all trading halted"}
            ),
            
            AlertRule(
                name="SystemOverloaded",
                description="System showing signs of overload",
                query="rate(cst_errors_total[1m]) > 0.1 and histogram_quantile(0.95, cst_order_latency_seconds_bucket) > 1.0",
                severity=AlertSeverity.CRITICAL,
                threshold=0.1,
                for_duration="1m",
                labels={"team": "trading"},
                annotations={"summary": "System overloaded - high errors and latency"}
            ),
            
            AlertRule(
                name="DataIntegrityIssue",
                description="Data integrity issues detected",
                query="rate(cst_data_requests_total{status=\"error\"}[5m]) > 0.2",
                severity=AlertSeverity.HIGH,
                threshold=0.2,
                for_duration="3m",
                labels={"team": "data"},
                annotations={"summary": "High data request failure rate"}
            )
        ]
    
    def record_error(self, component: str, error_type: str, severity: str):
        """Record error occurrence"""
        self.metrics["cst_errors_total"].inc(
            label_values={
                "component": component,
                "error_type": error_type,
                "severity": severity
            }
        )
    
    def record_trade(self, symbol: str, side: str, strategy: str, status: str, pnl_usd: Optional[float] = None):
        """Record trade execution"""
        self.metrics["cst_trades_total"].inc(
            label_values={
                "symbol": symbol,
                "side": side,
                "strategy": strategy,
                "status": status
            }
        )
        
        if pnl_usd is not None:
            self.metrics["cst_trade_pnl_usd"].set(
                pnl_usd,
                label_values={"symbol": symbol, "strategy": strategy}
            )
    
    def record_risk_evaluation(self, decision: str, reason: str):
        """Record risk evaluation"""
        self.metrics["cst_risk_evaluations_total"].inc(
            label_values={"decision": decision, "reason": reason}
        )
    
    def record_risk_breach(self, limit_type: str, severity: str):
        """Record risk limit breach"""
        self.metrics["cst_risk_limits_breached_total"].inc(
            label_values={"limit_type": limit_type, "severity": severity}
        )
    
    def record_latency(self, operation: str, exchange: str, latency_seconds: float):
        """Record operation latency"""
        self.metrics["cst_order_latency_seconds"].observe(
            latency_seconds,
            label_values={"operation": operation, "exchange": exchange}
        )
    
    def record_execution_decision(self, decision: str, reason: str):
        """Record execution decision"""
        self.metrics["cst_execution_decisions_total"].inc(
            label_values={"decision": decision, "reason": reason}
        )
    
    def record_data_request(self, source: str, status: str, latency_seconds: Optional[float] = None):
        """Record data request"""
        self.metrics["cst_data_requests_total"].inc(
            label_values={"source": source, "status": status}
        )
        
        if latency_seconds is not None:
            self.metrics["cst_data_latency_seconds"].observe(
                latency_seconds,
                label_values={"source": source, "endpoint": "api"}
            )
    
    def record_system_health(self, status: int, component: str = "overall"):
        """Record system health status"""
        self.metrics["cst_system_health_status"].set(
            status,
            label_values={"component": component}
        )
    
    def record_portfolio_value(self, value_usd: float):
        """Record portfolio value"""
        self.metrics["cst_portfolio_value_usd"].set(value_usd)
    
    def record_daily_pnl(self, pnl_usd: float):
        """Record daily PnL"""
        self.metrics["cst_daily_pnl_usd"].set(pnl_usd)
    
    def fire_alert(self, severity: str, alert_name: str):
        """Record alert firing"""
        self.metrics["cst_alerts_fired_total"].inc(
            label_values={"severity": severity, "alert_name": alert_name}
        )
    
    def get_alert_rules(self) -> List[AlertRule]:
        """Get all configured alert rules"""
        return self.alert_rules
    
    def export_prometheus_rules(self) -> str:
        """Export alert rules in Prometheus format"""
        rules_yaml = "groups:\n"
        rules_yaml += "  - name: cryptosmarttrader\n"
        rules_yaml += "    rules:\n"
        
        for rule in self.alert_rules:
            rules_yaml += f"      - alert: {rule.name}\n"
            rules_yaml += f"        expr: {rule.query}\n"
            rules_yaml += f"        for: {rule.for_duration}\n"
            rules_yaml += "        labels:\n"
            rules_yaml += f"          severity: {rule.severity.value}\n"
            
            if rule.labels:
                for k, v in rule.labels.items():
                    rules_yaml += f"          {k}: {v}\n"
            
            rules_yaml += "        annotations:\n"
            rules_yaml += f"          description: {rule.description}\n"
            
            if rule.annotations:
                for k, v in rule.annotations.items():
                    rules_yaml += f"          {k}: {v}\n"
            
            rules_yaml += "\n"
        
        return rules_yaml
    
    def export_metrics(self) -> str:
        """Export all metrics in Prometheus format"""
        prometheus_output = []
        
        for metric in self.metrics.values():
            prometheus_output.append(metric.export_prometheus())
        
        return "\n\n".join(prometheus_output)
    
    def validate_prometheus_query(self, query: str) -> Optional[str]:
        """Validate Prometheus query syntax (basic validation)"""
        # Basic validation - in real implementation would use PromQL parser
        if not query or not isinstance(query, str):
            return None
        
        # Check for basic PromQL patterns
        valid_patterns = ['rate(', 'histogram_quantile(', '==', '>', '<', 'and', 'or']
        has_valid_pattern = any(pattern in query for pattern in valid_patterns)
        
        if has_valid_pattern or any(metric_name in query for metric_name in self.metrics.keys()):
            return query
        
        return None
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall metrics system health"""
        uptime_seconds = time.time() - self.start_time
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "uptime_seconds": uptime_seconds,
            "metrics_count": len(self.metrics),
            "alert_rules_count": len(self.alert_rules)
        }
    
    def start_http_server(self, port: int = 8000):
        """Start HTTP server voor metrics export"""
        # In real implementation, would start actual HTTP server
        # For now, return mock server object
        logger.info(f"Metrics HTTP server would start on port {port}")
        return {"port": port, "status": "started"}


# Export main class
__all__ = ['CentralizedMetrics', 'MetricType', 'AlertSeverity', 'AlertRule', 'PrometheusMetric']