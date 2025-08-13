#!/usr/bin/env python3
"""
Backtesting Engine - Realistic order execution with slippage and fees
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Trade:
    """Trade execution record"""
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    executed_price: float
    slippage: float
    fees: float
    order_type: OrderType
    trade_id: str


@dataclass
class Position:
    """Portfolio position"""
    symbol: str
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0


@dataclass
class BacktestMetrics:
    """Comprehensive backtest metrics"""
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    total_fees: float
    total_slippage: float
    avg_trade_duration_hours: float
    profit_factor: float
    calmar_ratio: float
    sortino_ratio: float


class SlippageModel:
    """
    Realistic slippage model based on market conditions
    """
    
    def __init__(self):
        # Exchange-specific slippage parameters
        self.exchange_params = {
            "kraken": {
                "base_slippage_bps": 2.0,  # Base slippage in basis points
                "volume_impact": 0.1,      # Volume impact coefficient
                "volatility_impact": 0.5,   # Volatility impact coefficient
                "liquidity_threshold": 10000  # USD threshold for liquidity impact
            },
            "binance": {
                "base_slippage_bps": 1.5,
                "volume_impact": 0.08,
                "volatility_impact": 0.4,
                "liquidity_threshold": 20000
            },
            "coinbase": {
                "base_slippage_bps": 3.0,
                "volume_impact": 0.12,
                "volatility_impact": 0.6,
                "liquidity_threshold": 5000
            }
        }
    
    def calculate_slippage(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        exchange: str = "kraken",
        volatility: float = 0.02,
        spread_bps: float = 5.0
    ) -> float:
        """
        Calculate realistic slippage for order execution
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            quantity: Order quantity
            price: Current market price
            exchange: Exchange name
            volatility: Recent volatility (24h)
            spread_bps: Current bid-ask spread in basis points
            
        Returns:
            Slippage in price units
        """
        
        params = self.exchange_params.get(exchange, self.exchange_params["kraken"])
        
        # Order value in USD
        order_value = quantity * price
        
        # Base slippage
        base_slippage = params["base_slippage_bps"] / 10000
        
        # Volume impact - larger orders have more slippage
        volume_impact = params["volume_impact"] * np.log(1 + order_value / params["liquidity_threshold"])
        
        # Volatility impact - higher volatility = more slippage
        volatility_impact = params["volatility_impact"] * volatility
        
        # Spread impact - tighter spreads = less slippage
        spread_impact = spread_bps / 10000
        
        # Combine impacts
        total_slippage_pct = base_slippage + volume_impact + volatility_impact + spread_impact
        
        # Direction matters - buying moves price up, selling moves down
        direction = 1 if side == OrderSide.BUY else -1
        
        # Add randomness for realistic variation
        random_factor = np.random.normal(1.0, 0.2)
        
        slippage = price * total_slippage_pct * direction * random_factor
        
        return max(slippage, 0.0)  # Slippage cannot be negative


class FeeModel:
    """
    Realistic fee model for different exchanges and order types
    """
    
    def __init__(self):
        # Exchange fee structures (maker/taker)
        self.exchange_fees = {
            "kraken": {
                "maker": 0.0016,  # 0.16%
                "taker": 0.0026,  # 0.26%
                "minimum_fee": 0.01  # USD
            },
            "binance": {
                "maker": 0.001,   # 0.10%
                "taker": 0.001,   # 0.10%
                "minimum_fee": 0.01
            },
            "coinbase": {
                "maker": 0.005,   # 0.50%
                "taker": 0.005,   # 0.50%
                "minimum_fee": 0.01
            }
        }
    
    def calculate_fee(
        self,
        order_value: float,
        order_type: OrderType,
        exchange: str = "kraken",
        is_maker: bool = False
    ) -> float:
        """
        Calculate trading fees
        
        Args:
            order_value: Order value in USD
            order_type: Order type
            exchange: Exchange name
            is_maker: Whether order is maker (limit) or taker (market)
            
        Returns:
            Fee amount in USD
        """
        
        fees = self.exchange_fees.get(exchange, self.exchange_fees["kraken"])
        
        # Determine fee rate
        if order_type == OrderType.LIMIT and is_maker:
            fee_rate = fees["maker"]
        else:
            fee_rate = fees["taker"]
        
        # Calculate fee
        fee = order_value * fee_rate
        
        # Apply minimum fee
        return max(fee, fees["minimum_fee"])


class BacktestingEngine:
    """
    Comprehensive backtesting engine with realistic execution
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        
        self.slippage_model = SlippageModel()
        self.fee_model = FeeModel()
        
        self.logger = logging.getLogger(__name__)
        
        # Risk management parameters
        self.max_position_size = 0.1  # 10% of capital per position
        self.max_portfolio_exposure = 0.8  # 80% total exposure
        self.max_drawdown_limit = 0.2  # 20% max drawdown
        self.stop_loss_pct = 0.05  # 5% stop loss
        
        # Kill switch parameters
        self.min_health_score = 0.7  # Minimum health score to trade
        self.max_drift_score = 0.3   # Maximum drift score to trade
        
        # State tracking
        self.is_trading_enabled = True
        self.last_health_score = 1.0
        self.last_drift_score = 0.0
        self.current_drawdown = 0.0
        self.peak_equity = initial_capital
    
    def execute_order(
        self,
        timestamp: datetime,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        order_type: OrderType = OrderType.MARKET,
        exchange: str = "kraken"
    ) -> Optional[Trade]:
        """
        Execute order with realistic slippage and fees
        
        Args:
            timestamp: Execution timestamp
            symbol: Trading symbol
            side: Order side
            quantity: Order quantity
            price: Market price
            order_type: Order type
            exchange: Exchange name
            
        Returns:
            Trade object if executed, None if rejected
        """
        
        # Check if trading is enabled
        if not self.is_trading_enabled:
            self.logger.warning("Trading disabled - order rejected")
            return None
        
        # Calculate order value
        order_value = quantity * price
        
        # Risk management checks
        if not self._risk_check(symbol, side, order_value):
            self.logger.warning(f"Risk check failed for {symbol} {side.value}")
            return None
        
        # Calculate slippage
        volatility = self._estimate_volatility(symbol, timestamp)
        spread_bps = self._estimate_spread(symbol, exchange)
        
        slippage = self.slippage_model.calculate_slippage(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            exchange=exchange,
            volatility=volatility,
            spread_bps=spread_bps
        )
        
        # Calculate execution price
        executed_price = price + slippage
        executed_value = quantity * executed_price
        
        # Calculate fees
        is_maker = (order_type == OrderType.LIMIT)
        fees = self.fee_model.calculate_fee(
            order_value=executed_value,
            order_type=order_type,
            exchange=exchange,
            is_maker=is_maker
        )
        
        # Check if we have enough capital
        if side == OrderSide.BUY:
            required_capital = executed_value + fees
            if required_capital > self.current_capital:
                self.logger.warning(f"Insufficient capital: need {required_capital}, have {self.current_capital}")
                return None
        
        # Execute trade
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            executed_price=executed_price,
            slippage=slippage,
            fees=fees,
            order_type=order_type,
            trade_id=f"{timestamp.isoformat()}_{symbol}_{side.value}"
        )
        
        # Update positions and capital
        self._update_position(trade)
        self._update_capital(trade)
        
        # Record trade
        self.trades.append(trade)
        
        self.logger.info(f"Executed: {symbol} {side.value} {quantity} @ {executed_price:.4f}")
        return trade
    
    def _risk_check(self, symbol: str, side: OrderSide, order_value: float) -> bool:
        """Risk management checks"""
        
        # Position size check
        if side == OrderSide.BUY:
            position_size = order_value / self.current_capital
            if position_size > self.max_position_size:
                return False
        
        # Portfolio exposure check
        total_exposure = sum(
            abs(pos.quantity * pos.avg_price) 
            for pos in self.positions.values()
        )
        
        if side == OrderSide.BUY:
            total_exposure += order_value
        
        if total_exposure / self.current_capital > self.max_portfolio_exposure:
            return False
        
        # Drawdown check
        if self.current_drawdown > self.max_drawdown_limit:
            return False
        
        return True
    
    def _update_position(self, trade: Trade):
        """Update position based on trade"""
        symbol = trade.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0.0,
                avg_price=0.0
            )
        
        position = self.positions[symbol]
        
        if trade.side == OrderSide.BUY:
            # Update average price for buys
            total_quantity = position.quantity + trade.quantity
            if total_quantity > 0:
                total_cost = (position.quantity * position.avg_price + 
                             trade.quantity * trade.executed_price)
                position.avg_price = total_cost / total_quantity
            position.quantity = total_quantity
            
        else:  # SELL
            # Calculate realized P&L
            if position.quantity > 0:
                sell_quantity = min(trade.quantity, position.quantity)
                realized_pnl = sell_quantity * (trade.executed_price - position.avg_price)
                position.realized_pnl += realized_pnl
                position.quantity -= sell_quantity
                
                # If position is closed, reset average price
                if position.quantity == 0:
                    position.avg_price = 0.0
    
    def _update_capital(self, trade: Trade):
        """Update capital based on trade"""
        if trade.side == OrderSide.BUY:
            self.current_capital -= (trade.quantity * trade.executed_price + trade.fees)
        else:
            self.current_capital += (trade.quantity * trade.executed_price - trade.fees)
    
    def _estimate_volatility(self, symbol: str, timestamp: datetime) -> float:
        """Estimate recent volatility (placeholder)"""
        # In real implementation, calculate from historical data
        return 0.02  # 2% daily volatility
    
    def _estimate_spread(self, symbol: str, exchange: str) -> float:
        """Estimate current spread (placeholder)"""
        # In real implementation, get from order book data
        spread_estimates = {
            "kraken": 5.0,   # 5 bps
            "binance": 3.0,  # 3 bps
            "coinbase": 8.0  # 8 bps
        }
        return spread_estimates.get(exchange, 5.0)
    
    def update_health_score(self, health_score: float):
        """Update system health score"""
        self.last_health_score = health_score
        
        # Disable trading if health is too low
        if health_score < self.min_health_score:
            self.is_trading_enabled = False
            self.logger.warning(f"Trading disabled: health score {health_score} < {self.min_health_score}")
        elif not self.is_trading_enabled and health_score > self.min_health_score + 0.1:
            # Re-enable with hysteresis
            self.is_trading_enabled = True
            self.logger.info(f"Trading re-enabled: health score {health_score}")
    
    def update_drift_score(self, drift_score: float):
        """Update data drift score"""
        self.last_drift_score = drift_score
        
        # Disable trading if drift is too high
        if drift_score > self.max_drift_score:
            self.is_trading_enabled = False
            self.logger.warning(f"Trading disabled: drift score {drift_score} > {self.max_drift_score}")
        elif not self.is_trading_enabled and drift_score < self.max_drift_score - 0.05:
            # Re-enable with hysteresis
            self.is_trading_enabled = True
            self.logger.info(f"Trading re-enabled: drift score {drift_score}")
    
    def update_equity_curve(self, timestamp: datetime, market_prices: Dict[str, float]):
        """Update equity curve with current market values"""
        # Calculate current portfolio value
        portfolio_value = self.current_capital
        
        for symbol, position in self.positions.items():
            if symbol in market_prices and position.quantity > 0:
                market_value = position.quantity * market_prices[symbol]
                portfolio_value += market_value
                
                # Update unrealized P&L
                position.unrealized_pnl = (market_prices[symbol] - position.avg_price) * position.quantity
        
        # Track peak and drawdown
        if portfolio_value > self.peak_equity:
            self.peak_equity = portfolio_value
        
        self.current_drawdown = (self.peak_equity - portfolio_value) / self.peak_equity
        
        # Record equity point
        self.equity_curve.append((timestamp, portfolio_value))
    
    def calculate_metrics(self) -> BacktestMetrics:
        """Calculate comprehensive backtest metrics"""
        if len(self.equity_curve) < 2:
            return BacktestMetrics(
                total_return=0.0, annual_return=0.0, sharpe_ratio=0.0,
                max_drawdown=0.0, win_rate=0.0, total_trades=0,
                total_fees=0.0, total_slippage=0.0, avg_trade_duration_hours=0.0,
                profit_factor=0.0, calmar_ratio=0.0, sortino_ratio=0.0
            )
        
        # Calculate returns
        final_value = self.equity_curve[-1][1]
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate time-based metrics
        start_date = self.equity_curve[0][0]
        end_date = self.equity_curve[-1][0]
        years = (end_date - start_date).days / 365.25
        
        annual_return = (1 + total_return) ** (1 / max(years, 1/365)) - 1 if years > 0 else 0
        
        # Calculate Sharpe ratio
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev_value = self.equity_curve[i-1][1]
            curr_value = self.equity_curve[i][1]
            daily_return = (curr_value - prev_value) / prev_value
            returns.append(daily_return)
        
        if returns:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = avg_return / std_return * np.sqrt(252) if std_return > 0 else 0
            
            # Sortino ratio (downside deviation)
            negative_returns = [r for r in returns if r < 0]
            downside_std = np.std(negative_returns) if negative_returns else 0
            sortino_ratio = avg_return / downside_std * np.sqrt(252) if downside_std > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Calculate max drawdown
        max_drawdown = 0
        peak = self.initial_capital
        
        for _, value in self.equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Trade-based metrics
        winning_trades = 0
        total_profit = 0
        total_loss = 0
        total_fees = sum(trade.fees for trade in self.trades)
        total_slippage = sum(abs(trade.slippage) for trade in self.trades)
        
        for position in self.positions.values():
            if position.realized_pnl > 0:
                winning_trades += 1
                total_profit += position.realized_pnl
            else:
                total_loss += abs(position.realized_pnl)
        
        win_rate = winning_trades / len(self.positions) if self.positions else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calmar ratio
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        return BacktestMetrics(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=len(self.trades),
            total_fees=total_fees,
            total_slippage=total_slippage,
            avg_trade_duration_hours=0.0,  # Would need trade matching logic
            profit_factor=profit_factor,
            calmar_ratio=calmar_ratio,
            sortino_ratio=sortino_ratio
        )


# Global backtesting engine
backtesting_engine = BacktestingEngine()