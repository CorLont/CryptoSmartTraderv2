#!/usr/bin/env python3
"""
Centralized Observability API voor CryptoSmartTrader V2
Centraliseerde metrics, alerts en health monitoring voor 500% target trading system
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, 
        CollectorRegistry, generate_latest, 
        start_http_server, REGISTRY
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from .metrics import get_metrics, CryptoSmartTraderMetrics
from ..core.structured_logger import get_logger

logger = get_logger("centralized_observability")


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning" 
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Alert:
    """Alert data structure"""
    name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    metric_value: float
    threshold: float
    resolved: bool = False
    resolved_at: Optional[float] = None


class CentralizedObservabilityService:
    """
    Gecentraliseerde observability service voor alle metrics en alerts
    Handles metrics collection, alert evaluation, en notification routing
    """
    
    def __init__(self):
        self.logger = get_logger("centralized_observability")
        self.metrics = get_metrics()
        
        # Alert tracking
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_lock = threading.Lock()
        
        # Alert thresholds configuratie
        self.alert_thresholds = {
            'high_order_error_rate': 0.05,  # 5% error rate
            'drawdown_too_high': 0.10,      # 10% drawdown
            'no_signals_timeout': 1800,     # 30 minutes (1800 seconds)
        }
        
        # Metrics tracking voor rate calculations
        self.metrics_history = {
            'orders_sent': 0,
            'order_errors': 0,
            'last_signal_time': time.time(),
            'current_drawdown': 0.0,
            'current_equity': 100000.0,  # Default starting equity
        }
        
        # Background monitoring
        self.monitoring_active = True
        self.monitoring_thread = None
        
        self._start_background_monitoring()
        self.logger.info("✅ Centralized Observability Service geïnitialiseerd")
    
    def _start_background_monitoring(self):
        """Start background thread voor alert monitoring"""
        def monitor():
            while self.monitoring_active:
                try:
                    self._check_alert_conditions()
                    time.sleep(30)  # Check elke 30 seconden
                except Exception as e:
                    self.logger.error(f"Error in alert monitoring: {e}")
                    time.sleep(60)
        
        self.monitoring_thread = threading.Thread(target=monitor, daemon=True)
        self.monitoring_thread.start()
        self.logger.info("Background alert monitoring gestart")
    
    def _check_alert_conditions(self):
        """Check alle alert conditions en trigger alerts waar nodig"""
        current_time = time.time()
        
        # 1. HighOrderErrorRate Alert
        self._check_high_order_error_rate()
        
        # 2. DrawdownTooHigh Alert  
        self._check_drawdown_too_high()
        
        # 3. NoSignals Alert (30 minutes)
        self._check_no_signals_timeout(current_time)
    
    def _check_high_order_error_rate(self):
        """Check voor hoge order error rate (>5%)"""
        try:
            # Get current order counts
            total_orders = self._get_counter_value('orders_sent')
            total_errors = self._get_counter_value('order_errors')
            
            if total_orders > 10:  # Minimum threshold voor meaningful rate
                error_rate = total_errors / total_orders
                
                if error_rate > self.alert_thresholds['high_order_error_rate']:
                    self._trigger_alert(
                        name="HighOrderErrorRate",
                        severity=AlertSeverity.CRITICAL,
                        message=f"Order error rate {error_rate:.2%} exceeds threshold {self.alert_thresholds['high_order_error_rate']:.1%}",
                        metric_value=error_rate,
                        threshold=self.alert_thresholds['high_order_error_rate']
                    )
                else:
                    self._resolve_alert("HighOrderErrorRate")
        except Exception as e:
            self.logger.error(f"Error checking order error rate: {e}")
    
    def _check_drawdown_too_high(self):
        """Check voor te hoge drawdown (>10%)"""
        try:
            current_drawdown = self.metrics_history.get('current_drawdown', 0.0)
            
            if current_drawdown > self.alert_thresholds['drawdown_too_high']:
                self._trigger_alert(
                    name="DrawdownTooHigh", 
                    severity=AlertSeverity.EMERGENCY,
                    message=f"Portfolio drawdown {current_drawdown:.2%} exceeds threshold {self.alert_thresholds['drawdown_too_high']:.1%}",
                    metric_value=current_drawdown,
                    threshold=self.alert_thresholds['drawdown_too_high']
                )
            else:
                self._resolve_alert("DrawdownTooHigh")
        except Exception as e:
            self.logger.error(f"Error checking drawdown: {e}")
    
    def _check_no_signals_timeout(self, current_time: float):
        """Check voor geen signals ontvangen (>30 min)"""
        try:
            last_signal_time = self.metrics_history.get('last_signal_time', current_time)
            signal_age_seconds = current_time - last_signal_time
            
            if signal_age_seconds > self.alert_thresholds['no_signals_timeout']:
                self._trigger_alert(
                    name="NoSignals",
                    severity=AlertSeverity.WARNING,
                    message=f"No signals received for {signal_age_seconds/60:.1f} minutes (threshold: 30 min)",
                    metric_value=signal_age_seconds/60,
                    threshold=self.alert_thresholds['no_signals_timeout']/60
                )
            else:
                self._resolve_alert("NoSignals")
        except Exception as e:
            self.logger.error(f"Error checking signal timeout: {e}")
    
    def _trigger_alert(self, name: str, severity: AlertSeverity, message: str, 
                      metric_value: float, threshold: float):
        """Trigger een alert"""
        with self.alert_lock:
            current_time = time.time()
            
            # Check als alert al actief is
            if name in self.active_alerts and not self.active_alerts[name].resolved:
                return  # Alert al actief, skip duplicate
            
            alert = Alert(
                name=name,
                severity=severity,
                message=message,
                timestamp=current_time,
                metric_value=metric_value,
                threshold=threshold
            )
            
            self.active_alerts[name] = alert
            self.alert_history.append(alert)
            
            # Log alert
            self.logger.warning(f"🚨 ALERT TRIGGERED: {name} - {message}")
            
            # Update metrics
            if hasattr(self.metrics, 'alerts_triggered'):
                self.metrics.alerts_triggered.labels(
                    alert_name=name, 
                    severity=severity.value
                ).inc()
    
    def _resolve_alert(self, name: str):
        """Resolve een alert"""
        with self.alert_lock:
            if name in self.active_alerts and not self.active_alerts[name].resolved:
                self.active_alerts[name].resolved = True
                self.active_alerts[name].resolved_at = time.time()
                self.logger.info(f"✅ ALERT RESOLVED: {name}")
    
    def _get_counter_value(self, metric_name: str) -> float:
        """Safely get counter value from metrics"""
        try:
            # Use metrics_history as primary source for tracking
            if metric_name in self.metrics_history:
                return self.metrics_history[metric_name]
            
            # Fallback to prometheus metrics
            if hasattr(self.metrics, metric_name):
                metric = getattr(self.metrics, metric_name)
                if hasattr(metric, 'collect'):
                    samples = metric.collect()[0].samples
                    return sum(sample.value for sample in samples)
            return 0.0
        except Exception as e:
            self.logger.error(f"Error getting counter value for {metric_name}: {e}")
            return 0.0
    
    def get_centralized_metrics(self) -> Dict[str, Any]:
        """Get alle gecentraliseerde metrics"""
        return {
            'timestamp': datetime.now().isoformat(),
            'trading_metrics': {
                'orders_sent': self._get_counter_value('orders_sent'),
                'orders_filled': self._get_counter_value('orders_filled'), 
                'order_errors': self._get_counter_value('order_errors'),
                'signals_received': self._get_counter_value('signals_received'),
            },
            'performance_metrics': {
                'current_equity_usd': self.metrics_history.get('current_equity', 100000.0),
                'current_drawdown_pct': self.metrics_history.get('current_drawdown', 0.0) * 100,
                'last_signal_age_minutes': (time.time() - self.metrics_history.get('last_signal_time', time.time())) / 60,
            },
            'alert_metrics': {
                'active_alerts_count': len([a for a in self.active_alerts.values() if not a.resolved]),
                'total_alerts_count': len(self.alert_history),
            }
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get alle actieve alerts"""
        with self.alert_lock:
            return [
                {
                    'name': alert.name,
                    'severity': alert.severity.value,
                    'message': alert.message,
                    'timestamp': alert.timestamp,
                    'metric_value': alert.metric_value,
                    'threshold': alert.threshold,
                    'age_minutes': (time.time() - alert.timestamp) / 60
                }
                for alert in self.active_alerts.values()
                if not alert.resolved
            ]
    
    def get_prometheus_metrics(self) -> str:
        """Export alle metrics in Prometheus format"""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.metrics.registry).decode('utf-8')
        else:
            return "# Prometheus niet beschikbaar\n"
    
    # Public API methods voor metric updates
    def record_order_sent(self, exchange: str, symbol: str, side: str, order_type: str):
        """Record order sent"""
        self.metrics.orders_sent.labels(
            exchange=exchange, symbol=symbol, side=side, order_type=order_type
        ).inc()
        self.metrics_history['orders_sent'] = self.metrics_history.get('orders_sent', 0) + 1
    
    def record_order_filled(self, exchange: str, symbol: str, side: str, order_type: str):
        """Record order filled"""
        self.metrics.orders_filled.labels(
            exchange=exchange, symbol=symbol, side=side, order_type=order_type
        ).inc()
    
    def record_order_error(self, exchange: str, symbol: str, error_type: str, error_code: str):
        """Record order error"""
        self.metrics.order_errors.labels(
            exchange=exchange, symbol=symbol, error_type=error_type, error_code=error_code
        ).inc()
        self.metrics_history['order_errors'] = self.metrics_history.get('order_errors', 0) + 1
    
    def record_latency(self, operation: str, exchange: str, endpoint: str, latency_ms: float):
        """Record operation latency"""
        self.metrics.latency_ms.labels(
            operation=operation, exchange=exchange, endpoint=endpoint
        ).observe(latency_ms)
    
    def record_slippage(self, exchange: str, symbol: str, side: str, slippage_bps: float):
        """Record trade slippage"""
        self.metrics.slippage_bps.labels(
            exchange=exchange, symbol=symbol, side=side
        ).observe(slippage_bps)
    
    def update_equity(self, equity_usd: float):
        """Update current portfolio equity"""
        self.metrics.equity.labels(strategy="main", account="default").set(equity_usd)
        self.metrics_history['current_equity'] = equity_usd
    
    def update_drawdown(self, drawdown_pct: float):
        """Update current portfolio drawdown"""
        self.metrics.drawdown_pct.labels(strategy="main", timeframe="1d").set(drawdown_pct)
        self.metrics_history['current_drawdown'] = drawdown_pct / 100.0  # Convert to ratio
    
    def record_signal_received(self, agent: str, signal_type: str, symbol: str):
        """Record signal ontvangen"""
        self.metrics.signals_received.labels(
            agent=agent, signal_type=signal_type, symbol=symbol
        ).inc()
        self.metrics_history['last_signal_time'] = time.time()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        active_alerts = self.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a['severity'] in ['critical', 'emergency']]
        
        return {
            'status': 'unhealthy' if critical_alerts else 'healthy',
            'timestamp': datetime.now().isoformat(),
            'service': 'cryptosmarttrader-observability',
            'version': '2.0.0',
            'uptime_seconds': time.time() - (self.metrics_history.get('start_time', time.time())),
            'metrics_available': PROMETHEUS_AVAILABLE,
            'alerts': {
                'active_count': len(active_alerts),
                'critical_count': len(critical_alerts),
                'active_alerts': active_alerts[:5]  # Top 5 most recent
            },
            'monitoring': {
                'background_monitoring_active': self.monitoring_active,
                'last_check_age_seconds': 30,  # Since we check every 30s
            }
        }


# Global singleton instance
_observability_service: Optional[CentralizedObservabilityService] = None
_service_lock = threading.Lock()


def get_observability_service() -> CentralizedObservabilityService:
    """Get global observability service singleton"""
    global _observability_service
    
    if _observability_service is None:
        with _service_lock:
            if _observability_service is None:
                _observability_service = CentralizedObservabilityService()
    
    return _observability_service


# FastAPI application
app = FastAPI(
    title="CryptoSmartTrader V2 Observability API",
    description="Centralized observability, metrics and alerts for 500% target trading system",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint - FASE 3 requirement"""
    try:
        service = get_observability_service()
        health_status = service.get_health_status()
        
        status_code = 200 if health_status['status'] == 'healthy' else 503
        
        return JSONResponse(
            status_code=status_code,
            content=health_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        )


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics_endpoint():
    """Prometheus metrics endpoint - FASE 3 requirement"""
    try:
        service = get_observability_service()
        prometheus_metrics = service.get_prometheus_metrics()
        
        return PlainTextResponse(
            content=prometheus_metrics,
            headers={"Content-Type": "text/plain; version=0.0.4; charset=utf-8"}
        )
    except Exception as e:
        logger.error(f"Metrics export failed: {e}")
        return PlainTextResponse(
            content=f"# Error exporting metrics: {str(e)}\n",
            status_code=500
        )


@app.get("/metrics/summary")
async def metrics_summary():
    """Centralized metrics summary"""
    try:
        service = get_observability_service()
        return JSONResponse(content=service.get_centralized_metrics())
    except Exception as e:
        logger.error(f"Metrics summary failed: {e}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@app.get("/alerts")
async def get_alerts():
    """Get active alerts - FASE 3 requirement"""
    try:
        service = get_observability_service()
        active_alerts = service.get_active_alerts()
        
        return JSONResponse(content={
            'active_alerts': active_alerts,
            'count': len(active_alerts),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Get alerts failed: {e}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@app.get("/alerts/summary")
async def alerts_summary():
    """Alert summary met kritieke alerts"""
    try:
        service = get_observability_service()
        active_alerts = service.get_active_alerts()
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_active_alerts': len(active_alerts),
            'alerts_by_severity': {},
            'critical_alerts': []
        }
        
        # Group by severity
        for alert in active_alerts:
            severity = alert['severity']
            if severity not in summary['alerts_by_severity']:
                summary['alerts_by_severity'][severity] = 0
            summary['alerts_by_severity'][severity] += 1
            
            # Track critical alerts
            if severity in ['critical', 'emergency']:
                summary['critical_alerts'].append(alert)
        
        return JSONResponse(content=summary)
    except Exception as e:
        logger.error(f"Alerts summary failed: {e}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@app.post("/metrics/record/order")
async def record_order_metric(
    action: str,  # 'sent', 'filled', 'error'
    exchange: str = "kraken",
    symbol: str = "BTC/USD", 
    side: str = "buy",
    order_type: str = "market",
    error_type: str = "",
    error_code: str = ""
):
    """Record order metrics programmatically"""
    try:
        service = get_observability_service()
        
        if action == "sent":
            service.record_order_sent(exchange, symbol, side, order_type)
        elif action == "filled":
            service.record_order_filled(exchange, symbol, side, order_type)
        elif action == "error":
            service.record_order_error(exchange, symbol, error_type, error_code)
        
        return JSONResponse(content={'status': 'recorded', 'action': action})
    except Exception as e:
        logger.error(f"Record order metric failed: {e}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@app.post("/metrics/update/portfolio")
async def update_portfolio_metrics(
    equity_usd: float,
    drawdown_pct: float
):
    """Update portfolio metrics"""
    try:
        service = get_observability_service()
        service.update_equity(equity_usd)
        service.update_drawdown(drawdown_pct)
        
        return JSONResponse(content={
            'status': 'updated',
            'equity_usd': equity_usd,
            'drawdown_pct': drawdown_pct
        })
    except Exception as e:
        logger.error(f"Update portfolio metrics failed: {e}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


@app.post("/metrics/record/signal")
async def record_signal_metric(
    agent: str,
    signal_type: str,
    symbol: str
):
    """Record signal received"""
    try:
        service = get_observability_service()
        service.record_signal_received(agent, signal_type, symbol)
        
        return JSONResponse(content={
            'status': 'recorded',
            'agent': agent,
            'signal_type': signal_type,
            'symbol': symbol
        })
    except Exception as e:
        logger.error(f"Record signal failed: {e}")
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )


def main():
    """Start observability API server"""
    print("🔍 Starting CryptoSmartTrader V2 Centralized Observability API")
    print("📊 Health endpoint: http://localhost:8002/health")
    print("📈 Metrics endpoint: http://localhost:8002/metrics") 
    print("🚨 Alerts endpoint: http://localhost:8002/alerts")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()