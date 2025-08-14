"""
CryptoSmartTrader V2 Observability Package

Centralized metrics, monitoring, and observability for the trading system.
All Prometheus metrics are consolidated in metrics.py for consistency.
"""

from .metrics import get_metrics, timer, track_api_calls, track_orders

__all__ = [
    'get_metrics',
    'timer', 
    'track_api_calls',
    'track_orders'
]

# Version info
__version__ = '2.0.0'
__title__ = 'CryptoSmartTrader Observability'
__description__ = 'Centralized observability and metrics collection'
