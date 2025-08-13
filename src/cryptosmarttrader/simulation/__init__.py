"""
CryptoSmartTrader V2 Backtest-Live Parity System

Advanced execution simulation and parity tracking for ensuring
backtest results match live trading performance.
"""

from .execution_simulator import (
    ExecutionSimulator, MarketConditions, OrderType, FillType, FeeStructure,
    OrderStatus, SimulatedOrder, OrderFill, LatencyModel, LiquidityModel,
    SlippageModel, get_execution_simulator
)

from .parity_tracker import (
    ParityTracker, ParityThresholds, ParityStatus, DriftType, TradeExecution,
    DailyParityReport, get_parity_tracker, calculate_tracking_error
)

__all__ = [
    # Execution Simulation
    'ExecutionSimulator', 'MarketConditions', 'OrderType', 'FillType', 'FeeStructure',
    'OrderStatus', 'SimulatedOrder', 'OrderFill', 'LatencyModel', 'LiquidityModel',
    'SlippageModel', 'get_execution_simulator',
    
    # Parity Tracking  
    'ParityTracker', 'ParityThresholds', 'ParityStatus', 'DriftType', 'TradeExecution',
    'DailyParityReport', 'get_parity_tracker', 'calculate_tracking_error'
]

# Version info
__version__ = '2.0.0'
__title__ = 'CryptoSmartTrader Backtest-Live Parity'
__description__ = 'Execution simulation and parity tracking system'