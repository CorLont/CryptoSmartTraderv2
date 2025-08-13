#!/usr/bin/env python3
"""
Unified Observability System - Centralized metrics for 500% target trading system
Comprehensive monitoring with critical alerts for enterprise trading operations
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import threading
import asyncio
from contextlib import contextmanager

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info, Enum as PrometheusEnum,
        generate_latest, CollectorRegistry, REGISTRY, start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from ..core.structured_logger import get_logger

logger = get_logger("unified_metrics")

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    metric_name: str
    threshold: float
    operator: str  # >, <, >=, <=, ==
    severity: AlertSeverity
    duration_minutes: int = 5
    description: str = ""

class UnifiedMetrics:
    """Centralized metrics collection and alerting system"""
    
    def __init__(self, service_name: str = "cryptosmarttrader"):
        self.service_name = service_name
        self.logger = get_logger(f"unified_metrics_{service_name}")
        self.registry = CollectorRegistry()
        self.alerts_active = {}
        self._lock = threading.Lock()
        
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()
        else:
            self.logger.warning("Prometheus not available, using mock metrics")
            self._init_mock_metrics()
            
        self._init_alert_rules()
        
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        
        # Trading Performance Metrics
        self.orders_total = Counter(
            'trading_orders_total',
            'Total number of trading orders',
            ['status', 'symbol', 'side'],
            registry=self.registry
        )
        
        self.order_errors_total = Counter(
            'trading_order_errors_total', 
            'Total number of order errors',
            ['error_type', 'symbol'],
            registry=self.registry
        )
        
        self.slippage_bps = Histogram(
            'trading_slippage_bps',
            'Trading slippage in basis points',
            ['symbol', 'order_type'],
            buckets=[1, 5, 10, 20, 30, 50, 100, 200, 500],
            registry=self.registry
        )
        
        self.drawdown_pct = Gauge(
            'portfolio_drawdown_percent',
            'Current portfolio drawdown percentage',
            registry=self.registry
        )
        
        self.equity_usd = Gauge(
            'portfolio_equity_usd',
            'Current portfolio equity in USD',
            registry=self.registry
        )
        
        # Signal Quality Metrics
        self.signals_received = Counter(
            'ml_signals_received_total',
            'Total ML signals received',
            ['agent_name', 'confidence_bucket'],
            registry=self.registry
        )
        
        self.signal_age_minutes = Gauge(
            'ml_signal_age_minutes',
            'Age of last received signal in minutes',
            ['agent_name'],
            registry=self.registry
        )
        
        self.prediction_confidence = Histogram(
            'ml_prediction_confidence',
            'ML prediction confidence scores',
            ['model_name', 'symbol'],
            buckets=[0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99],
            registry=self.registry
        )
        
        # System Health Metrics
        self.api_requests_total = Counter(
            'api_requests_total',
            'Total API requests',
            ['endpoint', 'method', 'status_code'],
            registry=self.registry
        )
        
        self.api_duration_seconds = Histogram(
            'api_duration_seconds',
            'API request duration',
            ['endpoint', 'method'],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry
        )
        
        self.exchange_connectivity = Gauge(
            'exchange_connectivity_status',
            'Exchange connectivity status (1=connected, 0=disconnected)',
            ['exchange_name'],
            registry=self.registry
        )
        
        self.data_quality_score = Gauge(
            'data_quality_score',
            'Data quality score (0-1)',
            ['data_source', 'symbol'],
            registry=self.registry
        )
        
        # Risk Management Metrics
        self.risk_level = PrometheusEnum(
            'risk_management_level',
            'Current risk management level',
            ['component'],
            states=['normal', 'conservative', 'defensive', 'emergency', 'shutdown'],
            registry=self.registry
        )
        
        self.position_count = Gauge(
            'portfolio_position_count',
            'Number of open positions',
            registry=self.registry
        )
        
        self.largest_position_pct = Gauge(
            'portfolio_largest_position_percent',
            'Largest position as percentage of portfolio',
            registry=self.registry
        )
        
        # Alert Metrics
        self.alerts_fired = Counter(
            'alerts_fired_total',
            'Total alerts fired',
            ['alert_name', 'severity'],
            registry=self.registry
        )
        
        self.alerts_active_count = Gauge(
            'alerts_active_count',
            'Number of active alerts',
            ['severity'],
            registry=self.registry
        )
        
    def _init_mock_metrics(self):
        """Initialize mock metrics when Prometheus unavailable"""
        self.logger.warning("Using mock metrics - install prometheus_client for full functionality")
        
        # Create simple counters for mock mode
        self._mock_counters = {}
        self._mock_gauges = {}
        self._mock_histograms = {}
        
    def _init_alert_rules(self):
        """Initialize critical alert rules for 500% target system"""
        
        self.alert_rules = [
            # Trading Performance Alerts
            AlertRule(
                name="HighOrderErrorRate",
                metric_name="order_error_rate",
                threshold=0.05,  # 5%
                operator=">",
                severity=AlertSeverity.CRITICAL,
                duration_minutes=2,
                description="Order error rate above 5% - trading execution degraded"
            ),
            
            AlertRule(
                name="ExcessiveSlippage", 
                metric_name="slippage_bps_p95",
                threshold=50,  # 50 bps
                operator=">",
                severity=AlertSeverity.WARNING,
                duration_minutes=5,
                description="95th percentile slippage above 50 bps - execution quality poor"
            ),
            
            AlertRule(
                name="CriticalSlippage",
                metric_name="slippage_bps_p95", 
                threshold=100,  # 100 bps
                operator=">",
                severity=AlertSeverity.CRITICAL,
                duration_minutes=2,
                description="95th percentile slippage above 100 bps - critical execution degradation"
            ),
            
            # Portfolio Risk Alerts  
            AlertRule(
                name="HighDrawdown",
                metric_name="drawdown_pct",
                threshold=8.0,  # 8%
                operator=">", 
                severity=AlertSeverity.CRITICAL,
                duration_minutes=1,
                description="Portfolio drawdown above 8% - risk management intervention required"
            ),
            
            AlertRule(
                name="EmergencyDrawdown",
                metric_name="drawdown_pct",
                threshold=12.0,  # 12%
                operator=">",
                severity=AlertSeverity.EMERGENCY,
                duration_minutes=0,
                description="Portfolio drawdown above 12% - emergency stop required"
            ),
            
            # Signal Quality Alerts
            AlertRule(
                name="NoSignalsReceived", 
                metric_name="signal_age_minutes",
                threshold=30,  # 30 minutes
                operator=">",
                severity=AlertSeverity.WARNING,
                duration_minutes=5,
                description="No ML signals received for 30+ minutes - model pipeline issue"
            ),
            
            AlertRule(
                name="CriticalSignalGap",
                metric_name="signal_age_minutes", 
                threshold=60,  # 60 minutes
                operator=">",
                severity=AlertSeverity.CRITICAL,
                duration_minutes=2,
                description="No ML signals for 60+ minutes - critical model failure"
            ),
            
            # System Health Alerts
            AlertRule(
                name="ExchangeDisconnected",
                metric_name="exchange_connectivity",
                threshold=0.5,  # 50% exchanges down
                operator="<",
                severity=AlertSeverity.CRITICAL,
                duration_minutes=1,
                description="Exchange connectivity below 50% - trading capability compromised"
            ),
            
            AlertRule(
                name="LowDataQuality",
                metric_name="data_quality_score",
                threshold=0.8,  # 80%
                operator="<",
                severity=AlertSeverity.WARNING,
                duration_minutes=5,
                description="Data quality below 80% - decision accuracy degraded"
            )
        ]
        
        self.logger.info(f"Initialized {len(self.alert_rules)} critical alert rules")
        
    # Trading Metrics Methods
    def record_order(self, status: str, symbol: str, side: str, error_type: str = None):
        """Record trading order with status"""
        if PROMETHEUS_AVAILABLE:
            self.orders_total.labels(status=status, symbol=symbol, side=side).inc()
            if status == 'error' and error_type:
                self.order_errors_total.labels(error_type=error_type, symbol=symbol).inc()
        
        self.logger.info(
            f"Order recorded: {status}",
            symbol=symbol, side=side, error_type=error_type,
            metric="order_recorded"
        )
        
    def record_slippage(self, symbol: str, order_type: str, slippage_bps: float):
        """Record trading slippage in basis points"""
        if PROMETHEUS_AVAILABLE:
            self.slippage_bps.labels(symbol=symbol, order_type=order_type).observe(slippage_bps)
            
        # Check slippage alert
        self._check_slippage_alerts(slippage_bps)
        
    def update_drawdown(self, drawdown_pct: float):
        """Update portfolio drawdown percentage"""
        if PROMETHEUS_AVAILABLE:
            self.drawdown_pct.set(drawdown_pct)
            
        # Check drawdown alerts
        self._check_drawdown_alerts(drawdown_pct)
        
        self.logger.info(f"Drawdown updated: {drawdown_pct:.2f}%", metric="drawdown_update")
        
    def update_equity(self, equity_usd: float):
        """Update portfolio equity"""
        if PROMETHEUS_AVAILABLE:
            self.equity_usd.set(equity_usd)
            
        self.logger.info(f"Equity updated: ${equity_usd:,.2f}", metric="equity_update")
        
    # Signal Quality Methods
    def record_signal(self, agent_name: str, confidence: float, symbol: str = ""):
        """Record ML signal received"""
        confidence_bucket = self._get_confidence_bucket(confidence)
        
        if PROMETHEUS_AVAILABLE:
            self.signals_received.labels(
                agent_name=agent_name, 
                confidence_bucket=confidence_bucket
            ).inc()
            self.signal_age_minutes.labels(agent_name=agent_name).set(0)
            
            if symbol:
                self.prediction_confidence.labels(
                    model_name=agent_name, 
                    symbol=symbol
                ).observe(confidence)
        
        # Update signal timestamp for staleness monitoring
        with self._lock:
            self.alerts_active[f"signal_{agent_name}_timestamp"] = time.time()
            
        self.logger.info(
            f"Signal received from {agent_name}",
            confidence=confidence, symbol=symbol, metric="signal_received"
        )
        
    def update_signal_age(self, agent_name: str, age_minutes: float):
        """Update signal age for staleness monitoring"""
        if PROMETHEUS_AVAILABLE:
            self.signal_age_minutes.labels(agent_name=agent_name).set(age_minutes)
            
        # Check signal staleness alerts
        self._check_signal_alerts(agent_name, age_minutes)
        
    # System Health Methods  
    def record_api_request(self, endpoint: str, method: str, status_code: int, duration: float):
        """Record API request metrics"""
        if PROMETHEUS_AVAILABLE:
            self.api_requests_total.labels(
                endpoint=endpoint, 
                method=method, 
                status_code=str(status_code)
            ).inc()
            self.api_duration_seconds.labels(endpoint=endpoint, method=method).observe(duration)
            
    def update_exchange_connectivity(self, exchange_name: str, connected: bool):
        """Update exchange connectivity status"""
        if PROMETHEUS_AVAILABLE:
            self.exchange_connectivity.labels(exchange_name=exchange_name).set(1 if connected else 0)
            
        if not connected:
            self._fire_alert("ExchangeDisconnected", f"Exchange {exchange_name} disconnected")
            
    def update_data_quality(self, data_source: str, symbol: str, quality_score: float):
        """Update data quality score"""
        if PROMETHEUS_AVAILABLE:
            self.data_quality_score.labels(data_source=data_source, symbol=symbol).set(quality_score)
            
        # Check data quality alerts
        if quality_score < 0.8:
            self._fire_alert("LowDataQuality", f"Data quality {quality_score:.0%} for {data_source}/{symbol}")
            
    # Risk Management Methods
    def update_risk_level(self, component: str, level: str):
        """Update risk management level"""
        if PROMETHEUS_AVAILABLE:
            self.risk_level.labels(component=component).state(level.lower())
            
        self.logger.info(f"Risk level updated: {component} -> {level}", metric="risk_level_update")
        
    def update_position_metrics(self, position_count: int, largest_position_pct: float):
        """Update position-related metrics"""
        if PROMETHEUS_AVAILABLE:
            self.position_count.set(position_count)
            self.largest_position_pct.set(largest_position_pct)
            
    # Alert Management Methods
    def _check_slippage_alerts(self, slippage_bps: float):
        """Check slippage-based alerts"""
        for rule in self.alert_rules:
            if rule.metric_name == "slippage_bps_p95":
                # Simplified check - in production would use percentile calculation
                if self._evaluate_threshold(slippage_bps, rule.threshold, rule.operator):
                    self._fire_alert(rule.name, f"Slippage {slippage_bps:.1f} bps exceeds {rule.threshold}")
                    
    def _check_drawdown_alerts(self, drawdown_pct: float):
        """Check drawdown-based alerts"""
        for rule in self.alert_rules:
            if rule.metric_name == "drawdown_pct":
                if self._evaluate_threshold(drawdown_pct, rule.threshold, rule.operator):
                    self._fire_alert(rule.name, f"Drawdown {drawdown_pct:.1f}% exceeds {rule.threshold}%")
                    
    def _check_signal_alerts(self, agent_name: str, age_minutes: float):
        """Check signal staleness alerts"""
        for rule in self.alert_rules:
            if rule.metric_name == "signal_age_minutes":
                if self._evaluate_threshold(age_minutes, rule.threshold, rule.operator):
                    self._fire_alert(rule.name, f"Signal from {agent_name} stale for {age_minutes:.1f} minutes")
                    
    def _fire_alert(self, alert_name: str, message: str):
        """Fire an alert with proper logging and metrics"""
        alert_key = f"{alert_name}_{int(time.time() // 300)}"  # 5-minute buckets
        
        if alert_key not in self.alerts_active:
            # Find alert rule
            rule = next((r for r in self.alert_rules if r.name == alert_name), None)
            severity = rule.severity.value if rule else "warning"
            
            if PROMETHEUS_AVAILABLE:
                self.alerts_fired.labels(alert_name=alert_name, severity=severity).inc()
                
            self.alerts_active[alert_key] = {
                'timestamp': time.time(),
                'message': message,
                'severity': severity
            }
            
            # Log based on severity
            if severity == "emergency":
                self.logger.critical(f"EMERGENCY ALERT: {alert_name} - {message}")
            elif severity == "critical":
                self.logger.error(f"CRITICAL ALERT: {alert_name} - {message}")
            elif severity == "warning":
                self.logger.warning(f"WARNING ALERT: {alert_name} - {message}")
            else:
                self.logger.info(f"INFO ALERT: {alert_name} - {message}")
                
    def _evaluate_threshold(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate threshold condition"""
        if operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == "==":
            return abs(value - threshold) < 0.001
        return False
        
    def _get_confidence_bucket(self, confidence: float) -> str:
        """Get confidence bucket for labeling"""
        if confidence >= 0.95:
            return "very_high"
        elif confidence >= 0.85:
            return "high"
        elif confidence >= 0.7:
            return "medium"
        elif confidence >= 0.5:
            return "low"
        else:
            return "very_low"
            
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics"""
        return {
            'service_name': self.service_name,
            'prometheus_available': PROMETHEUS_AVAILABLE,
            'alert_rules_count': len(self.alert_rules),
            'active_alerts_count': len(self.alerts_active),
            'last_update': datetime.now().isoformat()
        }
        
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics output"""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry).decode('utf-8')
        else:
            return "# Prometheus not available\n"
            
    @contextmanager
    def timing_context(self, operation_name: str):
        """Context manager for timing operations"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            if PROMETHEUS_AVAILABLE:
                # Use a generic timing histogram
                timing_histogram = Histogram(
                    f'operation_duration_seconds',
                    'Operation duration in seconds',
                    ['operation_name'],
                    registry=self.registry
                )
                timing_histogram.labels(operation_name=operation_name).observe(duration)

# Global metrics instance
_global_metrics: Optional[UnifiedMetrics] = None

def get_metrics() -> UnifiedMetrics:
    """Get global metrics instance"""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = UnifiedMetrics()
    return _global_metrics

def start_metrics_server(port: int = 8000) -> bool:
    """Start Prometheus metrics HTTP server"""
    try:
        if PROMETHEUS_AVAILABLE:
            start_http_server(port, registry=get_metrics().registry)
            logger.info(f"Metrics server started on port {port}")
            return True
        else:
            logger.warning("Cannot start metrics server - Prometheus not available")
            return False
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
        return False

if __name__ == "__main__":
    # Test the unified metrics system
    print("🔍 TESTING UNIFIED METRICS SYSTEM")
    print("=" * 50)
    
    metrics = UnifiedMetrics("test_service")
    
    # Test trading metrics
    print("\n📊 Testing trading metrics...")
    metrics.record_order("filled", "BTC/USD", "buy")
    metrics.record_order("error", "ETH/USD", "sell", "insufficient_balance")
    metrics.record_slippage("BTC/USD", "market", 25.5)
    metrics.update_drawdown(3.2)
    metrics.update_equity(100000.50)
    
    # Test signal metrics
    print("📈 Testing signal metrics...")
    metrics.record_signal("ensemble_voting", 0.92, "BTC/USD")
    metrics.record_signal("technical_analyzer", 0.78, "ETH/USD")
    metrics.update_signal_age("sentiment_agent", 45.0)
    
    # Test system health
    print("🏥 Testing system health metrics...")
    metrics.record_api_request("/health", "GET", 200, 0.123)
    metrics.update_exchange_connectivity("kraken", True)
    metrics.update_data_quality("kraken", "BTC/USD", 0.95)
    
    # Test risk management
    print("🛡️ Testing risk management metrics...")
    metrics.update_risk_level("portfolio", "normal")
    metrics.update_position_metrics(5, 15.2)
    
    # Generate high slippage alert
    print("🚨 Testing alert system...")
    metrics.record_slippage("ETH/USD", "market", 75.0)  # Should trigger alert
    metrics.update_drawdown(9.5)  # Should trigger critical alert
    
    # Show summary
    print("\n📋 Metrics Summary:")
    summary = metrics.get_metrics_summary()
    for key, value in summary.items():
        print(f"   {key}: {value}")
        
    print(f"\n✅ Unified metrics system operational")
    print(f"🎯 Ready for 500% target monitoring")