"""
Backtest-Live Parity Module
Advanced simulation and monitoring for backtest-live parity validation.
"""

from .execution_simulator import (
    ExecutionSimulator,
    ExecutionResult,
    OrderBook,
    SimulationConfig,
    OrderType,
    OrderSide,
    create_execution_simulator
)

from .parity_analyzer import (
    ParityAnalyzer,
    ParityMetrics,
    DriftDetection,
    ParityStatus,
    create_parity_analyzer
)

__all__ = [
    'ExecutionSimulator',
    'ExecutionResult',
    'OrderBook',
    'SimulationConfig',
    'OrderType',
    'OrderSide',
    'ParityAnalyzer',
    'ParityMetrics',
    'DriftDetection',
    'ParityStatus',
    'create_execution_simulator',
    'create_parity_analyzer'
]