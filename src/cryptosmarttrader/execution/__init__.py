"""
Execution Module

Implements realistic execution simulation and live-backtest parity
to eliminate performance illusions and ensure accurate backtesting.
"""

# from .execution_simulator import ExecutionSimulator, ExecutionResult, Fill  # Module not yet implemented
# from .market_microstructure import MarketMicrostructure, OrderBookSimulator, LiquidityProvider  # Module not yet implemented
# from .slippage_analyzer import SlippageAnalyzer, SlippageAttribution, SlippageSource  # Module not yet implemented
# from .execution_quality_monitor import ExecutionQualityMonitor, ExecutionQuality, QualityGrade  # Module not yet implemented
# from .live_backtest_comparator import LiveBacktestComparator, ParityMetrics, PerformanceGap  # Module not yet implemented

# Order idempotency and deduplication modules
from .order_deduplication import OrderDeduplicationEngine, OrderSubmission, ClientOrderId, OrderStatus
from .idempotent_executor import IdempotentOrderExecutor, ExecutionContext, ExecutionMode

__all__ = [
    # Order idempotency and deduplication
    "OrderDeduplicationEngine",
    "OrderSubmission", 
    "ClientOrderId",
    "OrderStatus",
    "IdempotentOrderExecutor",
    "ExecutionContext",
    "ExecutionMode"
]