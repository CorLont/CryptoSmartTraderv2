"""
Market data API router.
Provides endpoints for market data, analysis, and predictions.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


class MarketOverviewResponse(BaseModel):
    """Market overview response model."""
    total_pairs: int = Field(..., description="Total number of trading pairs")
    active_signals: int = Field(..., description="Number of active trading signals")
    last_update: str = Field(..., description="Last data update timestamp")
    market_status: str = Field(..., description="Overall market status")
    top_movers: List[Dict[str, Any]] = Field(..., description="Top market movers")


class PairDataResponse(BaseModel):
    """Trading pair data response model."""
    symbol: str = Field(..., description="Trading pair symbol")
    price: float = Field(..., description="Current price")
    change_24h: float = Field(..., description="24-hour price change percentage")
    volume_24h: float = Field(..., description="24-hour trading volume")
    high_24h: float = Field(..., description="24-hour high price")
    low_24h: float = Field(..., description="24-hour low price")
    last_update: str = Field(..., description="Last update timestamp")


@router.get("/overview", 
           response_model=MarketOverviewResponse,
           summary="Get market overview")
async def get_market_overview():
    """Get comprehensive market overview and statistics."""
    try:
        # Return demo data for now
        return MarketOverviewResponse(
            total_pairs=471,
            active_signals=23,
            last_update=datetime.utcnow().isoformat(),
            market_status="operational",
            top_movers=[
                {"symbol": "BTC-USD", "change_24h": 2.45},
                {"symbol": "ETH-USD", "change_24h": 3.21},
                {"symbol": "ADA-USD", "change_24h": -1.87}
            ]
        )
        
    except Exception as e:
        logger.error(f"Failed to get market overview: {e}")
        raise HTTPException(status_code=500, detail=f"Market overview failed: {e}")


@router.get("/pair/{symbol}",
           response_model=PairDataResponse,
           summary="Get pair data")
async def get_pair_data(symbol: str):
    """Get detailed data for a specific trading pair."""
    try:
        # Return demo data for now
        demo_prices = {
            "BTC-USD": 45000.0,
            "ETH-USD": 2800.0,
            "ADA-USD": 0.85
        }
        
        base_price = demo_prices.get(symbol, 1000.0)
        
        return PairDataResponse(
            symbol=symbol,
            price=base_price,
            change_24h=2.45,
            volume_24h=125000000.0,
            high_24h=base_price * 1.05,
            low_24h=base_price * 0.95,
            last_update=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to get pair data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Pair data failed: {e}")


@router.get("/top_performers",
           summary="Get top performers")
async def get_top_performers(limit: int = 10, timeframe: str = "24h"):
    """Get top performing cryptocurrencies."""
    try:
        # Return demo data
        performers = [
            {"symbol": "SOL-USD", "change_24h": 8.45, "volume": 89000000},
            {"symbol": "AVAX-USD", "change_24h": 6.23, "volume": 67000000},
            {"symbol": "MATIC-USD", "change_24h": 5.12, "volume": 45000000},
            {"symbol": "DOT-USD", "change_24h": 4.87, "volume": 38000000},
            {"symbol": "LINK-USD", "change_24h": 3.94, "volume": 29000000}
        ]
        
        return {
            "pairs": performers[:limit],
            "timeframe": timeframe,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get top performers: {e}")
        raise HTTPException(status_code=500, detail=f"Top performers failed: {e}")


@router.get("/predictions",
           summary="Get ML predictions")
async def get_predictions(confidence_threshold: float = 0.7):
    """Get machine learning predictions for market movements."""
    try:
        # Return demo predictions
        predictions = [
            {
                "symbol": "BTC-USD",
                "direction": "bullish",
                "confidence": 0.85,
                "target_price": 47000.0,
                "timeframe": "24h"
            },
            {
                "symbol": "ETH-USD", 
                "direction": "bearish",
                "confidence": 0.72,
                "target_price": 2750.0,
                "timeframe": "24h"
            },
            {
                "symbol": "ADA-USD",
                "direction": "neutral",
                "confidence": 0.68,
                "target_price": 0.86,
                "timeframe": "24h"
            }
        ]
        
        # Filter by confidence threshold
        filtered_predictions = [
            p for p in predictions 
            if p["confidence"] >= confidence_threshold
        ]
        
        return {
            "predictions": filtered_predictions,
            "confidence_threshold": confidence_threshold,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get predictions: {e}")
        raise HTTPException(status_code=500, detail=f"Predictions failed: {e}")


@router.get("/technical/{symbol}",
           summary="Get technical analysis")
async def get_technical_analysis(symbol: str):
    """Get technical analysis indicators for a trading pair."""
    try:
        # Return demo technical analysis
        return {
            "symbol": symbol,
            "indicators": {
                "rsi": 65.4,
                "macd": 0.23,
                "bb_upper": 46500.0,
                "bb_lower": 43500.0,
                "sma_20": 45000.0,
                "ema_12": 45200.0,
                "volume_sma": 125000000
            },
            "signals": {
                "trend": "bullish",
                "momentum": "strong",
                "volatility": "normal"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get technical analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Technical analysis failed: {e}")