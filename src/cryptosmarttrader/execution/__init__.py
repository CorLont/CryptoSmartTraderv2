"""
Execution Quality Management Module

Implements advanced execution strategies to preserve alpha through:
- Post-only orders where possible (maker rebates)
- TWAP for thin order books
- Time-in-force optimization
- Fee-tier awareness (maker vs taker)
- Regime-aware execution policies
- Partial fill handling
- Price improvement ladders
"""

from .execution_policy import ExecutionPolicy, ExecutionMode, TimeInForce
from .fee_optimizer import FeeOptimizer, FeeStructure
from .twap_executor import TWAPExecutor, TWAPConfig
from .partial_fill_handler import PartialFillHandler, FillStatus
from .price_improvement import PriceImprovementLadder, ImprovementStrategy
from .execution_analytics import ExecutionAnalytics, ExecutionMetrics

__all__ = [
    "ExecutionPolicy",
    "ExecutionMode", 
    "TimeInForce",
    "FeeOptimizer",
    "FeeStructure",
    "TWAPExecutor",
    "TWAPConfig",
    "PartialFillHandler",
    "FillStatus",
    "PriceImprovementLadder",
    "ImprovementStrategy",
    "ExecutionAnalytics",
    "ExecutionMetrics"
]