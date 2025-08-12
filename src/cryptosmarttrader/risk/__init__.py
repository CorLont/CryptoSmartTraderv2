"""
Risk Management Module

Enterprise-grade risk management system with circuit breakers,
kill switches, and comprehensive safety mechanisms.
"""

from .risk_limits import RiskLimitManager, RiskLimit, LimitType
from .kill_switch import KillSwitchSystem, KillSwitchTrigger, EmergencyStop
from .circuit_breaker import CircuitBreakerSystem, CircuitState, BreakReason
from .order_deduplication import OrderDeduplicator, OrderState, DuplicateCheck

__all__ = [
    "RiskLimitManager",
    "RiskLimit", 
    "LimitType",
    "KillSwitchSystem",
    "KillSwitchTrigger",
    "EmergencyStop",
    "CircuitBreakerSystem",
    "CircuitState",
    "BreakReason",
    "OrderDeduplicator",
    "OrderState",
    "DuplicateCheck"
]