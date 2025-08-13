"""
API Schemas - Pydantic DTOs for request/response validation
"""

from .common import *
from .health import *
from .market_data import *
from .predictions import *
from .signals import *

__all__ = [
    # Common
    "BaseResponse", "ErrorResponse", "PaginatedResponse",
    # Health
    "HealthResponse", "SystemStatus",
    # Market Data
    "CoinData", "MarketDataResponse", "PriceData",
    # Predictions
    "PredictionRequest", "PredictionResponse", "MLMetrics",
    # Signals
    "TradingSignal", "SignalResponse", "SignalMetrics"
]