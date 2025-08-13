"""
Enterprise Observability Module
Comprehensive monitoring, metrics, and alerting for CryptoSmartTrader.
"""

from .metrics_collector import (
    MetricsCollector,
    RequestContext,
    OrderState,
    AlertSeverity,
    get_metrics_collector,
    setup_metrics_collector,
    record_order_sent,
    record_order_filled,
    record_order_error,
    record_signal_received,
    update_equity,
    update_drawdown
)

from .alert_rules import (
    AlertRule,
    AlertState,
    AlertManager,
    create_alert_manager
)

__all__ = [
    'MetricsCollector',
    'RequestContext',
    'OrderState',
    'AlertSeverity',
    'AlertRule',
    'AlertState',
    'AlertManager',
    'get_metrics_collector',
    'setup_metrics_collector',
    'create_alert_manager',
    'record_order_sent',
    'record_order_filled',
    'record_order_error',
    'record_signal_received',
    'update_equity',
    'update_drawdown'
]