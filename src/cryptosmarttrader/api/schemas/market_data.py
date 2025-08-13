#!/usr/bin/env python3
"""
Market Data Schemas - DTOs for cryptocurrency market data
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

from .common import BaseResponse


class PriceData(BaseModel):
    """Price data point"""
    timestamp: datetime = Field(description="Price timestamp")
    open: float = Field(description="Opening price", gt=0)
    high: float = Field(description="Highest price", gt=0)
    low: float = Field(description="Lowest price", gt=0)
    close: float = Field(description="Closing price", gt=0)
    volume: float = Field(description="Trading volume", ge=0)


class CoinData(BaseModel):
    """Cryptocurrency data"""
    symbol: str = Field(description="Trading symbol", example="BTC/USD")
    name: str = Field(description="Full coin name", example="Bitcoin")
    current_price: float = Field(description="Current price", gt=0)
    market_cap: Optional[float] = Field(description="Market capitalization", ge=0)
    volume_24h: float = Field(description="24h trading volume", ge=0)
    price_change_24h: float = Field(description="24h price change percentage")


class MarketDataResponse(BaseResponse):
    """Market data response"""
    data: List[CoinData] = Field(description="Market data")
