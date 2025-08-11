"""Market Data API Router - Cryptocurrency Market Information"""

from fastapi import APIRouter, Depends, Query, HTTPException
from typing import List, Optional
from datetime import datetime

from ..models.market import MarketDataOut, CoinInfo, PriceData
from ..dependencies import get_orchestrator, get_settings
from ...config import Settings

router = APIRouter(tags=["market"], prefix="/market")


@router.get("/data", response_model=MarketDataOut, summary="Get Market Data")
async def get_market_data(
    limit: int = Query(default=100, ge=1, le=1000, description="Number of coins to return"),
    sort_by: str = Query(default="market_cap", description="Sort field (market_cap, volume, change)"),
    orchestrator=Depends(get_orchestrator),
    settings: Settings = Depends(get_settings)
) -> MarketDataOut:
    """
    Get cryptocurrency market data
    
    Returns real-time market data for cryptocurrencies including:
    - Current prices
    - 24-hour volume
    - Price changes
    - Market data confidence score
    """
    try:
        # Get market data from orchestrator
        market_data = await orchestrator.get_market_data(limit=limit, sort_by=sort_by)
        
        # Convert to API response format
        coins = [
            PriceData(
                symbol=coin["symbol"],
                price=coin["price"],
                volume_24h=coin["volume_24h"],
                change_24h=coin["change_24h"],
                timestamp=coin["timestamp"]
            )
            for coin in market_data["coins"]
        ]
        
        return MarketDataOut(
            coins=coins,
            total_count=market_data["total_count"],
            last_updated=market_data["last_updated"],
            data_source=market_data["data_source"],
            confidence_score=market_data["confidence_score"]
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve market data: {str(e)}"
        )


@router.get("/coin/{symbol}", response_model=PriceData, summary="Get Coin Data")
async def get_coin_data(
    symbol: str,
    orchestrator=Depends(get_orchestrator)
) -> PriceData:
    """
    Get detailed data for a specific cryptocurrency
    
    Returns current price, volume, and change information for the specified symbol
    """
    try:
        # Get specific coin data from orchestrator
        coin_data = await orchestrator.get_coin_data(symbol.upper())
        
        if not coin_data:
            raise HTTPException(
                status_code=404,
                detail=f"Coin {symbol} not found"
            )
        
        return PriceData(
            symbol=coin_data["symbol"],
            price=coin_data["price"], 
            volume_24h=coin_data["volume_24h"],
            change_24h=coin_data["change_24h"],
            timestamp=coin_data["timestamp"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve coin data: {str(e)}"
        )


@router.get("/top-gainers", response_model=List[PriceData], summary="Get Top Gainers")
async def get_top_gainers(
    limit: int = Query(default=20, ge=1, le=100, description="Number of top gainers to return"),
    min_volume: Optional[float] = Query(default=None, description="Minimum 24h volume filter"),
    orchestrator=Depends(get_orchestrator)
) -> List[PriceData]:
    """
    Get top gaining cryptocurrencies by 24-hour price change
    
    Returns list of cryptocurrencies with highest positive price changes
    """
    try:
        # Get top gainers from orchestrator
        gainers_data = await orchestrator.get_top_gainers(
            limit=limit,
            min_volume=min_volume
        )
        
        return [
            PriceData(
                symbol=coin["symbol"],
                price=coin["price"],
                volume_24h=coin["volume_24h"],
                change_24h=coin["change_24h"],
                timestamp=coin["timestamp"]
            )
            for coin in gainers_data
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve top gainers: {str(e)}"
        )


@router.get("/top-losers", response_model=List[PriceData], summary="Get Top Losers")
async def get_top_losers(
    limit: int = Query(default=20, ge=1, le=100, description="Number of top losers to return"),
    min_volume: Optional[float] = Query(default=None, description="Minimum 24h volume filter"),
    orchestrator=Depends(get_orchestrator)
) -> List[PriceData]:
    """
    Get top losing cryptocurrencies by 24-hour price change
    
    Returns list of cryptocurrencies with highest negative price changes
    """
    try:
        # Get top losers from orchestrator
        losers_data = await orchestrator.get_top_losers(
            limit=limit,
            min_volume=min_volume
        )
        
        return [
            PriceData(
                symbol=coin["symbol"],
                price=coin["price"],
                volume_24h=coin["volume_24h"],
                change_24h=coin["change_24h"],
                timestamp=coin["timestamp"]
            )
            for coin in losers_data
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve top losers: {str(e)}"
        )