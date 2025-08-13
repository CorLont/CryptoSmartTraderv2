#!/usr/bin/env python3
"""
Signals Schemas - DTOs for trading signals
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

from .common import BaseResponse


class TradingSignal(BaseModel):
    """Trading signal"""
    symbol: str = Field(description="Trading symbol")
    signal_type: str = Field(description="Signal type", example="BUY")
    strength: float = Field(description="Signal strength", ge=0, le=1)
    price: float = Field(description="Signal price", gt=0)
    timestamp: datetime = Field(description="Signal timestamp")


class SignalMetrics(BaseModel):
    """Signal metrics"""
    total_signals: int = Field(description="Total signals", ge=0)
    success_rate: float = Field(description="Success rate", ge=0, le=1)
    avg_return: float = Field(description="Average return percentage")


class SignalResponse(BaseResponse):
    """Signal response"""
    signals: List[TradingSignal] = Field(description="Trading signals")
    metrics: SignalMetrics = Field(description="Signal metrics")
