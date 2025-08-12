"""
Execution Module

Implements realistic execution simulation and live-backtest parity
to eliminate performance illusions and ensure accurate backtesting.
"""

from .execution_simulator import ExecutionSimulator, OrderResult, ExecutionMetrics
from .market_microstructure import MarketMicrostructure, OrderBookSimulator, LiquidityProvider
from .slippage_analyzer import SlippageAnalyzer, SlippageAttribution, SlippageSource
from .execution_quality_monitor import ExecutionQualityMonitor, ExecutionQuality, QualityGrade
from .live_backtest_comparator import LiveBacktestComparator, ParityMetrics, PerformanceGap

__all__ = [
    "ExecutionSimulator",
    "OrderResult", 
    "ExecutionMetrics",
    "MarketMicrostructure",
    "OrderBookSimulator",
    "LiquidityProvider",
    "SlippageAnalyzer",
    "SlippageAttribution",
    "SlippageSource",
    "ExecutionQualityMonitor",
    "ExecutionQuality",
    "QualityGrade",
    "LiveBacktestComparator",
    "ParityMetrics",
    "PerformanceGap"
]