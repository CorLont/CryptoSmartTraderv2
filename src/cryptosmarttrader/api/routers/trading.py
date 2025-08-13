"""Trading API Router - Trading Signals and Portfolio Management"""

from fastapi import APIRouter, Depends, Query, HTTPException
from typing import List, Optional
from datetime import datetime, timedelta

from ..models.trading import SignalOut, PositionOut, PortfolioOut
from ..dependencies import get_orchestrator, get_settings
from ...config import Settings

router = APIRouter(tags=["trading"], prefix="/trading")


@router.get("/signals", response_model=List[SignalOut], summary="Get Trading Signals")
async def get_trading_signals(
    limit: int = Query(default=50, ge=1, le=200, description="Number of signals to return"),
    min_confidence: float = Query(
        default=0.7, ge=0.0, le=1.0, description="Minimum confidence threshold"
    ),
    symbol: Optional[str] = Query(default=None, description="Filter by specific symbol"),
    orchestrator=Depends(get_orchestrator),
    settings: Settings = Depends(get_settings),
) -> List[SignalOut]:
    """
    Get current trading signals from all agents

    Returns trading signals with confidence scores above the threshold
    """
    try:
        # Get signals from orchestrator
        signals_data = await orchestrator.get_trading_signals(
            limit=limit, min_confidence=min_confidence, symbol=symbol
        )

        return [
            SignalOut(
                symbol=signal["symbol"],
                signal_type=signal["signal_type"],
                confidence=signal["confidence"],
                strength=signal["strength"],
                price_target=signal.get("price_target"),
                stop_loss=signal.get("stop_loss"),
                reasoning=signal["reasoning"],
                timestamp=signal["timestamp"],
                agent_source=signal["agent_source"],
            )
            for signal in signals_data
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve trading signals: {str(e)}")


@router.get("/signals/{symbol}", response_model=List[SignalOut], summary="Get Symbol Signals")
async def get_symbol_signals(
    symbol: str,
    hours: int = Query(default=24, ge=1, le=168, description="Hours of signal history"),
    orchestrator=Depends(get_orchestrator),
) -> List[SignalOut]:
    """
    Get trading signals for a specific symbol

    Returns historical signals for the specified cryptocurrency
    """
    try:
        # Get symbol-specific signals from orchestrator
        signals_data = await orchestrator.get_symbol_signals(
            symbol=symbol.upper(), start_time=datetime.utcnow() - timedelta(hours=hours)
        )

        return [
            SignalOut(
                symbol=signal["symbol"],
                signal_type=signal["signal_type"],
                confidence=signal["confidence"],
                strength=signal["strength"],
                price_target=signal.get("price_target"),
                stop_loss=signal.get("stop_loss"),
                reasoning=signal["reasoning"],
                timestamp=signal["timestamp"],
                agent_source=signal["agent_source"],
            )
            for signal in signals_data
        ]

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve signals for {symbol}: {str(e)}"
        )


@router.get("/portfolio", response_model=PortfolioOut, summary="Get Portfolio Summary")
async def get_portfolio(
    orchestrator=Depends(get_orchestrator), settings: Settings = Depends(get_settings)
) -> PortfolioOut:
    """
    Get current portfolio summary

    Returns portfolio value, positions, and P&L information
    """
    try:
        # Get portfolio data from orchestrator
        portfolio_data = await orchestrator.get_portfolio_summary()

        # Convert positions to API format
        positions = [
            PositionOut(
                symbol=pos["symbol"],
                position_type=pos["position_type"],
                entry_price=pos["entry_price"],
                current_price=pos["current_price"],
                quantity=pos["quantity"],
                unrealized_pnl=pos["unrealized_pnl"],
                unrealized_pnl_percent=pos["unrealized_pnl_percent"],
                entry_timestamp=pos["entry_timestamp"],
            )
            for pos in portfolio_data["positions"]
        ]

        return PortfolioOut(
            total_value=portfolio_data["total_value"],
            available_balance=portfolio_data["available_balance"],
            invested_amount=portfolio_data["invested_amount"],
            unrealized_pnl=portfolio_data["unrealized_pnl"],
            unrealized_pnl_percent=portfolio_data["unrealized_pnl_percent"],
            positions=positions,
            position_count=len(positions),
            last_updated=portfolio_data["last_updated"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve portfolio: {str(e)}")


@router.get("/positions", response_model=List[PositionOut], summary="Get Active Positions")
async def get_positions(orchestrator=Depends(get_orchestrator)) -> List[PositionOut]:
    """
    Get all active trading positions

    Returns list of currently open positions with P&L information
    """
    try:
        # Get positions from orchestrator
        positions_data = await orchestrator.get_active_positions()

        return [
            PositionOut(
                symbol=pos["symbol"],
                position_type=pos["position_type"],
                entry_price=pos["entry_price"],
                current_price=pos["current_price"],
                quantity=pos["quantity"],
                unrealized_pnl=pos["unrealized_pnl"],
                unrealized_pnl_percent=pos["unrealized_pnl_percent"],
                entry_timestamp=pos["entry_timestamp"],
            )
            for pos in positions_data
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve positions: {str(e)}")
