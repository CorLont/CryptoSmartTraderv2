"""
Metrics Integration Layer
Integrates centralized Prometheus metrics with trading system components
"""

import time
import logging
from typing import Dict, List, Optional, Any
from contextlib import contextmanager
import threading

from .centralized_prometheus import get_global_prometheus_metrics, CentralizedPrometheusMetrics
from .alert_manager import get_global_alert_manager, AlertManager

logger = logging.getLogger(__name__)


class MetricsIntegration:
    """
    High-level integration layer voor metrics collection
    Simplifies metric recording across trading system components
    """
    
    def __init__(self):
        self.metrics = get_global_prometheus_metrics()
        self.alert_manager = get_global_alert_manager()
        self.logger = logging.getLogger(__name__)
        
        # Connect alert manager to metrics
        self._connect_alert_manager()
        
        self.logger.info("✅ Metrics integration initialized")
    
    def _connect_alert_manager(self):
        """Connect alert manager to receive alerts from metrics"""
        # In a real implementation, you'd set up a proper connection
        # For now, we'll use a simple polling mechanism
        pass
    
    @contextmanager
    def measure_execution_time(self, symbol: str, side: str, order_type: str):
        """Context manager to measure order execution time"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics.record_execution_duration(symbol, side, order_type, duration)
    
    @contextmanager
    def measure_api_call(self, exchange: str, endpoint: str):
        """Context manager to measure API call latency"""
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self.metrics.exchange_latency.labels(exchange=exchange, endpoint=endpoint).observe(duration)
    
    # Trading Operations
    def record_successful_order(self, symbol: str, side: str, order_type: str, size_usd: float):
        """Record successful order placement"""
        self.metrics.record_order(symbol, side, order_type, "filled")
        self.logger.debug(f"📊 Recorded successful order: {symbol} {side} ${size_usd:.2f}")
    
    def record_failed_order(self, symbol: str, side: str, error_type: str, error_message: str):
        """Record failed order"""
        self.metrics.record_order_error(symbol, side, error_type)
        self.logger.warning(f"📊 Recorded order error: {symbol} {side} - {error_type}")
    
    def record_trade_execution(
        self, 
        symbol: str, 
        side: str, 
        fill_type: str, 
        size_usd: float, 
        slippage_bps: float,
        execution_quality: float
    ):
        """Record complete trade execution"""
        self.metrics.record_fill(symbol, side, fill_type, size_usd)
        self.metrics.record_slippage(symbol, side, "market", slippage_bps)
        self.metrics.execution_quality_score.labels(symbol=symbol, side=side).observe(execution_quality)
        
        self.logger.debug(
            f"📊 Recorded trade execution: {symbol} {side} ${size_usd:.2f} "
            f"(slippage: {slippage_bps:.1f} bps, quality: {execution_quality:.1f})"
        )
    
    def record_trading_signal(self, strategy: str, symbol: str, signal_type: str, strength: float):
        """Record trading signal generation"""
        self.metrics.record_signal(strategy, symbol, signal_type, strength)
        self.logger.debug(f"📊 Recorded signal: {strategy} {symbol} {signal_type} ({strength:.2f})")
    
    # Portfolio Management
    def update_portfolio_state(
        self, 
        total_value: float, 
        positions: Dict[str, float], 
        daily_pnl: float,
        drawdown_pct: float
    ):
        """Update complete portfolio state"""
        self.metrics.update_portfolio_metrics(total_value, drawdown_pct, daily_pnl)
        
        # Update individual position sizes
        for symbol, size_usd in positions.items():
            self.metrics.update_position_size(symbol, size_usd)
        
        self.logger.debug(
            f"📊 Updated portfolio: ${total_value:.0f} value, {drawdown_pct:.1f}% drawdown, "
            f"${daily_pnl:.0f} daily PnL"
        )
    
    def record_risk_violation(self, violation_type: str, severity: str, details: Dict[str, Any]):
        """Record risk limit violation"""
        self.metrics.record_risk_violation(violation_type, severity)
        
        self.logger.warning(
            f"📊 Risk violation: {violation_type} ({severity}) - {details}"
        )
    
    def update_kill_switch_status(self, active: bool, reason: Optional[str] = None):
        """Update kill switch status"""
        self.metrics.update_kill_switch(active, reason)
        
        if active:
            self.logger.critical(f"🚨 Kill switch ACTIVATED: {reason}")
        else:
            self.logger.info("✅ Kill switch deactivated")
    
    # System Health
    def update_system_health(self, health_score: float, component_scores: Dict[str, float]):
        """Update system health metrics"""
        self.metrics.system_health_score.set(health_score)
        
        # Record individual component health as custom metrics
        for component, score in component_scores.items():
            # You could add component-specific metrics here
            pass
        
        self.logger.debug(f"📊 System health updated: {health_score:.1f}/100")
    
    def record_data_update(self, source: str, symbol: str, latency_seconds: float):
        """Record data pipeline update"""
        self.metrics.data_updates_total.labels(source=source, symbol=symbol).inc()
        self.metrics.data_update_latency.labels(source=source, symbol=symbol).observe(latency_seconds)
        
        if latency_seconds > 1.0:  # Log slow updates
            self.logger.warning(f"📊 Slow data update: {source} {symbol} ({latency_seconds:.2f}s)")
    
    def record_data_gap(self, source: str, symbol: str, gap_duration: float):
        """Record data gap"""
        self.metrics.record_data_gap(source, symbol)
        
        self.logger.warning(
            f"📊 Data gap detected: {source} {symbol} ({gap_duration:.1f}s gap)"
        )
    
    # Exchange Operations
    def update_exchange_status(self, exchange: str, connected: bool, latency_ms: Optional[float] = None):
        """Update exchange connection status"""
        self.metrics.update_exchange_connection(exchange, connected)
        
        if latency_ms is not None:
            self.metrics.exchange_latency.labels(exchange=exchange, endpoint="heartbeat").observe(latency_ms / 1000)
        
        status = "connected" if connected else "disconnected"
        self.logger.info(f"📊 Exchange {exchange}: {status}")
    
    def record_exchange_error(self, exchange: str, error_type: str, error_details: str):
        """Record exchange error"""
        self.metrics.record_exchange_error(exchange, error_type)
        
        self.logger.error(f"📊 Exchange error: {exchange} {error_type} - {error_details}")
    
    # Backtest-Live Parity
    def update_parity_metrics(self, tracking_error_bps: float, parity_status: str):
        """Update backtest-live parity metrics"""
        status_mapping = {
            "healthy": 0,
            "warning": 1, 
            "critical": 2,
            "disabled": 3
        }
        
        status_code = status_mapping.get(parity_status, 0)
        self.metrics.update_tracking_error(tracking_error_bps, status_code)
        
        self.logger.info(
            f"📊 Parity metrics: {tracking_error_bps:.1f} bps tracking error, "
            f"status: {parity_status}"
        )
    
    # Performance Tracking
    def update_strategy_performance(
        self, 
        strategy: str, 
        timeframe: str, 
        returns_pct: float,
        sharpe_ratio: float,
        win_rate: float
    ):
        """Update strategy performance metrics"""
        self.metrics.strategy_returns_pct.labels(strategy=strategy, timeframe=timeframe).set(returns_pct)
        self.metrics.sharpe_ratio.labels(strategy=strategy, timeframe=timeframe).set(sharpe_ratio)
        self.metrics.win_rate.labels(strategy=strategy).set(win_rate)
        
        self.logger.debug(
            f"📊 Strategy performance: {strategy} {timeframe} - "
            f"{returns_pct:.1f}% returns, {sharpe_ratio:.2f} Sharpe, {win_rate:.1%} win rate"
        )
    
    # Utility Methods
    def get_current_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of current metrics"""
        return {
            "metrics_server_running": True,
            "alert_manager_status": self.alert_manager.get_alert_status(),
            "metrics_registry_size": len(self.metrics.registry._collector_to_names),
        }
    
    def export_metrics_for_prometheus(self) -> bytes:
        """Export metrics in Prometheus format"""
        return self.metrics.get_metrics_output()
    
    def force_alert_evaluation(self):
        """Force immediate alert evaluation (for testing)"""
        self.metrics._evaluate_all_alerts()
    
    # Batch Operations
    def record_trading_session_summary(self, session_data: Dict[str, Any]):
        """Record summary of trading session"""
        summary = session_data
        
        # Total orders
        total_orders = summary.get("total_orders", 0)
        successful_orders = summary.get("successful_orders", 0)
        failed_orders = summary.get("failed_orders", 0)
        
        # Execution metrics
        avg_execution_time = summary.get("avg_execution_time", 0)
        avg_slippage = summary.get("avg_slippage_bps", 0)
        total_fees = summary.get("total_fees_usd", 0)
        
        # Performance
        session_pnl = summary.get("session_pnl_usd", 0)
        max_drawdown = summary.get("max_drawdown_pct", 0)
        
        self.logger.info(
            f"📊 Trading session summary: {total_orders} orders, "
            f"${session_pnl:.0f} PnL, {avg_slippage:.1f} bps avg slippage"
        )
    
    def cleanup_old_metrics(self, retention_hours: int = 24):
        """Cleanup old metric data (for memory management)"""
        cutoff_time = time.time() - (retention_hours * 3600)
        
        with self.metrics.metric_lock:
            for metric_name in list(self.metrics.metric_values.keys()):
                self.metrics.metric_values[metric_name] = [
                    (ts, val) for ts, val in self.metrics.metric_values[metric_name]
                    if ts > cutoff_time
                ]
        
        self.logger.debug(f"📊 Cleaned up metrics older than {retention_hours} hours")


# Global metrics integration instance
_global_metrics_integration: Optional[MetricsIntegration] = None


def get_global_metrics_integration() -> MetricsIntegration:
    """Get or create global metrics integration"""
    global _global_metrics_integration
    if _global_metrics_integration is None:
        _global_metrics_integration = MetricsIntegration()
        logger.info("✅ Global MetricsIntegration initialized")
    return _global_metrics_integration


def reset_global_metrics_integration():
    """Reset global metrics integration (for testing)"""
    global _global_metrics_integration
    _global_metrics_integration = None


# Convenience decorators
def record_trading_operation(symbol: str, side: str, order_type: str):
    """Decorator to automatically record trading operations"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics = get_global_metrics_integration()
            
            with metrics.measure_execution_time(symbol, side, order_type):
                try:
                    result = func(*args, **kwargs)
                    
                    # Assume successful if no exception
                    if hasattr(result, 'get') and result.get('success', True):
                        metrics.record_successful_order(
                            symbol, side, order_type, 
                            result.get('size_usd', 0)
                        )
                    
                    return result
                    
                except Exception as e:
                    metrics.record_failed_order(symbol, side, "exception", str(e))
                    raise
        
        return wrapper
    return decorator


def record_api_call(exchange: str, endpoint: str):
    """Decorator to automatically record API calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            metrics = get_global_metrics_integration()
            
            with metrics.measure_api_call(exchange, endpoint):
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    metrics.record_exchange_error(exchange, "api_error", str(e))
                    raise
        
        return wrapper
    return decorator