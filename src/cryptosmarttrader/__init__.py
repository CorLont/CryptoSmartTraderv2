"""
CryptoSmartTrader V2 - Enterprise Cryptocurrency Trading Intelligence System

A sophisticated multi-agent system for cryptocurrency trading with:
- Advanced ML ensemble optimization for 500% returns
- Enterprise-grade risk management and safety systems  
- Real-time market analysis and regime detection
- Comprehensive observability and monitoring
"""

__version__ = "2.0.0"
__author__ = "CryptoSmartTrader Team"

# Core system imports for convenient access
from .risk.risk_guard import RiskGuard, RiskLevel, RiskMetrics
from .execution.execution_policy import ExecutionPolicy
try:
    from .observability.metrics import PrometheusMetrics
except ImportError:
    # Fallback for package reorganization
    PrometheusMetrics = None

__all__ = [
    'RiskGuard',
    'RiskLevel', 
    'RiskMetrics',
    'ExecutionPolicy',
    'PrometheusMetrics'
]