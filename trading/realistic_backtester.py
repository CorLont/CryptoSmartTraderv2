#!/usr/bin/env python3
"""
Realistic Backtester
Advanced backtesting with fees, slippage, partial fills, latency, and market impact
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import warnings

warnings.filterwarnings("ignore")

from trading.orderbook_simulator import (
    OrderBookSimulator,
    ExchangeConfig,
    OrderSide,
    OrderType,
    OrderStatus,
)


@dataclass
class BacktestConfig:
    """Backtesting configuration"""

    initial_capital: float = 100000.0
    max_position_size: float = 0.1  # 10% of capital per position
    risk_per_trade: float = 0.02  # 2% risk per trade
    enable_slippage: bool = True
    enable_fees: bool = True
    enable_latency: bool = True
    enable_partial_fills: bool = True
    max_latency_ms: float = 1000.0  # Max acceptable latency
    min_liquidity_ratio: float = 0.1  # Min ratio of order size to available liquidity


@dataclass
class Position:
    """Trading position"""

    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    fees_paid: float = 0.0
    entry_timestamp: datetime = field(default_factory=lambda: datetime.utcnow())
    last_update: datetime = field(default_factory=lambda: datetime.utcnow())


@dataclass
class Trade:
    """Executed trade record"""

    trade_id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    entry_timestamp: datetime = field(default_factory=lambda: datetime.utcnow())
    exit_timestamp: Optional[datetime] = None
    pnl: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    latency_ms: float = 0.0
    partial_fill_ratio: float = 1.0
    is_closed: bool = False


@dataclass
class BacktestResult:
    """Complete backtesting results"""

    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    total_fees: float
    total_slippage: float
    avg_latency_ms: float
    partial_fill_rate: float
    liquidity_impact_events: int
    daily_returns: pd.Series
    equity_curve: pd.Series
    trade_history: List[Trade]
    position_history: List[Position]


class RealisticBacktester:
    """Advanced backtester with realistic market frictions"""

    def __init__(
        self, config: BacktestConfig, exchange_config: ExchangeConfig, price_data: pd.DataFrame
    ):
        self.config = config
        self.exchange_config = exchange_config
        self.price_data = price_data.copy()

        # Initialize orderbook simulator
        initial_price = price_data["close"].iloc[0]
        self.orderbook_sim = OrderBookSimulator(exchange_config, initial_price)

        # Portfolio state
        self.capital = config.initial_capital
        self.available_capital = config.initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []

        # Performance tracking
        self.equity_history = []
        self.daily_returns = []
        self.max_equity = config.initial_capital
        self.max_drawdown = 0.0

        # Friction tracking
        self.total_fees = 0.0
        self.total_slippage = 0.0
        self.latency_events = []
        self.partial_fill_events = 0
        self.liquidity_impact_events = 0

        self.next_trade_id = 1
        self.logger = logging.getLogger(__name__)

    def run_backtest(
        self, signal_data: pd.DataFrame, volatility_data: Optional[pd.DataFrame] = None
    ) -> BacktestResult:
        """Run complete backtest with realistic execution"""

        if "signal" not in signal_data.columns:
            raise ValueError("Signal data must contain 'signal' column")

        # Align data
        common_index = self.price_data.index.intersection(signal_data.index)
        price_aligned = self.price_data.loc[common_index]
        signal_aligned = signal_data.loc[common_index]

        if volatility_data is not None:
            vol_aligned = volatility_data.loc[common_index]
        else:
            # Calculate volatility from price data
            returns = price_aligned["close"].pct_change()
            vol_aligned = pd.DataFrame(
                {"volatility": returns.rolling(20).std()}, index=common_index
            )

        self.logger.info(f"Running backtest on {len(common_index)} periods")

        # Process each time period
        for i, (timestamp, price_row) in enumerate(price_aligned.iterrows()):
            if i == 0:
                continue  # Skip first row (no signal)

            # Get current market state
            current_price = price_row["close"]
            current_volume = price_row.get("volume", 1000000)  # Default volume
            current_volatility = (
                vol_aligned.loc[timestamp, "volatility"] if timestamp in vol_aligned.index else 0.02
            )

            # Update orderbook with price movement
            if i > 1:
                prev_price = price_aligned["close"].iloc[i - 1]
                price_change = (current_price - prev_price) / prev_price
                self.orderbook_sim.update_orderbook(price_change, current_volatility)

            # Get trading signal
            signal = signal_aligned.loc[timestamp, "signal"]

            # Process trading logic
            self._process_trading_signal(
                timestamp=timestamp,
                signal=signal,
                current_price=current_price,
                current_volume=current_volume,
                volatility=current_volatility,
            )

            # Update positions
            self._update_positions(current_price, timestamp)

            # Record equity
            total_equity = self._calculate_total_equity(current_price)
            self.equity_history.append(
                {
                    "timestamp": timestamp,
                    "equity": total_equity,
                    "available_capital": self.available_capital,
                }
            )

            # Track maximum equity and drawdown
            if total_equity > self.max_equity:
                self.max_equity = total_equity

            current_drawdown = (self.max_equity - total_equity) / self.max_equity
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown

        # Calculate final results
        return self._calculate_backtest_results()

    def _process_trading_signal(
        self,
        timestamp: datetime,
        signal: float,
        current_price: float,
        current_volume: float,
        volatility: float,
    ):
        """Process trading signal with realistic execution"""

        symbol = "BTC/USD"  # Assuming single asset for now

        # Determine action based on signal
        if signal > 0.5:  # Buy signal
            if symbol not in self.positions or self.positions[symbol].quantity <= 0:
                self._execute_buy_order(
                    symbol, current_price, current_volume, volatility, timestamp
                )

        elif signal < -0.5:  # Sell signal
            if symbol in self.positions and self.positions[symbol].quantity > 0:
                self._execute_sell_order(
                    symbol, current_price, current_volume, volatility, timestamp
                )

    def _execute_buy_order(
        self,
        symbol: str,
        current_price: float,
        current_volume: float,
        volatility: float,
        timestamp: datetime,
    ):
        """Execute buy order with realistic frictions"""

        # Calculate position size
        position_value = self.available_capital * self.config.max_position_size
        quantity = position_value / current_price

        # Check minimum order size
        if quantity < self.exchange_config.min_order_size:
            return

        # Check liquidity constraints
        estimated_volume = current_volume * 0.01  # Assume 1% of daily volume available
        liquidity_ratio = quantity / estimated_volume

        if liquidity_ratio > (1.0 / self.config.min_liquidity_ratio):
            # Reduce order size due to liquidity constraints
            quantity = estimated_volume * self.config.min_liquidity_ratio
            self.liquidity_impact_events += 1
            self.logger.warning(f"Order size reduced due to liquidity constraints")

        # Submit order to simulator
        order = self.orderbook_sim.submit_order(
            symbol=symbol, side=OrderSide.BUY, order_type=OrderType.MARKET, quantity=quantity
        )

        # Process execution results
        if order.status in [OrderStatus.FILLED, OrderStatus.PARTIAL]:
            fills = self.orderbook_sim.get_fills_for_order(order.order_id)

            if fills:
                # Calculate execution metrics
                total_quantity = sum(fill.quantity for fill in fills)
                total_cost = sum(fill.price * fill.quantity for fill in fills)
                total_fees = sum(fill.fee for fill in fills)
                avg_price = total_cost / total_quantity

                # Track frictions
                self.total_fees += total_fees
                self.total_slippage += order.slippage * total_cost
                self.latency_events.append(order.latency_ms)

                if order.status == OrderStatus.PARTIAL:
                    self.partial_fill_events += 1

                # Check latency impact
                if order.latency_ms > self.config.max_latency_ms:
                    self.logger.warning(f"High latency detected: {order.latency_ms:.1f}ms")

                # Update position
                if symbol in self.positions:
                    # Add to existing position
                    existing_pos = self.positions[symbol]
                    total_quantity_new = existing_pos.quantity + total_quantity
                    total_cost_new = (
                        existing_pos.quantity * existing_pos.avg_entry_price
                    ) + total_cost
                    new_avg_price = total_cost_new / total_quantity_new

                    existing_pos.quantity = total_quantity_new
                    existing_pos.avg_entry_price = new_avg_price
                    existing_pos.fees_paid += total_fees
                    existing_pos.last_update = timestamp
                else:
                    # Create new position
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        quantity=total_quantity,
                        avg_entry_price=avg_price,
                        current_price=current_price,
                        fees_paid=total_fees,
                        entry_timestamp=timestamp,
                        last_update=timestamp,
                    )

                # Update available capital
                self.available_capital -= total_cost + total_fees

                # Create trade record
                trade = Trade(
                    trade_id=f"trade_{self.next_trade_id}",
                    symbol=symbol,
                    side="buy",
                    quantity=total_quantity,
                    entry_price=avg_price,
                    entry_timestamp=timestamp,
                    fees=total_fees,
                    slippage=order.slippage,
                    latency_ms=order.latency_ms,
                    partial_fill_ratio=total_quantity / quantity,
                )

                self.trades.append(trade)
                self.next_trade_id += 1

                self.logger.debug(
                    f"Buy executed: {total_quantity:.6f} @ {avg_price:.2f}, "
                    f"slippage: {order.slippage:.4f}, fees: {total_fees:.2f}"
                )

    def _execute_sell_order(
        self,
        symbol: str,
        current_price: float,
        current_volume: float,
        volatility: float,
        timestamp: datetime,
    ):
        """Execute sell order with realistic frictions"""

        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        quantity = position.quantity

        if quantity <= 0:
            return

        # Submit order to simulator
        order = self.orderbook_sim.submit_order(
            symbol=symbol, side=OrderSide.SELL, order_type=OrderType.MARKET, quantity=quantity
        )

        # Process execution results
        if order.status in [OrderStatus.FILLED, OrderStatus.PARTIAL]:
            fills = self.orderbook_sim.get_fills_for_order(order.order_id)

            if fills:
                # Calculate execution metrics
                total_quantity = sum(fill.quantity for fill in fills)
                total_proceeds = sum(fill.price * fill.quantity for fill in fills)
                total_fees = sum(fill.fee for fill in fills)
                avg_price = total_proceeds / total_quantity

                # Track frictions
                self.total_fees += total_fees
                self.total_slippage += order.slippage * total_proceeds
                self.latency_events.append(order.latency_ms)

                if order.status == OrderStatus.PARTIAL:
                    self.partial_fill_events += 1

                # Calculate P&L
                entry_cost = total_quantity * position.avg_entry_price
                pnl = total_proceeds - entry_cost - total_fees

                # Update position
                position.quantity -= total_quantity
                position.realized_pnl += pnl
                position.fees_paid += total_fees
                position.last_update = timestamp

                # Update available capital
                self.available_capital += total_proceeds - total_fees

                # Find corresponding trade and close it
                for trade in reversed(self.trades):
                    if trade.symbol == symbol and trade.side == "buy" and not trade.is_closed:
                        trade.exit_price = avg_price
                        trade.exit_timestamp = timestamp
                        trade.pnl = pnl
                        trade.fees += total_fees
                        trade.is_closed = True
                        break

                # Remove position if fully closed
                if position.quantity <= 1e-8:  # Essentially zero
                    del self.positions[symbol]

                self.logger.debug(
                    f"Sell executed: {total_quantity:.6f} @ {avg_price:.2f}, "
                    f"P&L: {pnl:.2f}, slippage: {order.slippage:.4f}"
                )

    def _update_positions(self, current_price: float, timestamp: datetime):
        """Update position valuations"""

        for position in self.positions.values():
            position.current_price = current_price
            position.unrealized_pnl = (current_price - position.avg_entry_price) * position.quantity
            position.last_update = timestamp

    def _calculate_total_equity(self, current_price: float) -> float:
        """Calculate total portfolio equity"""

        total_equity = self.available_capital

        for position in self.positions.values():
            position_value = position.quantity * current_price
            total_equity += position_value

        return total_equity

    def _calculate_backtest_results(self) -> BacktestResult:
        """Calculate comprehensive backtest results"""

        # Convert equity history to pandas Series
        equity_df = pd.DataFrame(self.equity_history)
        equity_series = equity_df.set_index("timestamp")["equity"]

        # Calculate returns
        returns = equity_series.pct_change().dropna()

        # Performance metrics
        total_return = (
            equity_series.iloc[-1] - self.config.initial_capital
        ) / self.config.initial_capital

        # Sharpe ratio (assuming daily returns)
        if len(returns) > 1 and returns.std() > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0

        # Trade statistics
        closed_trades = [t for t in self.trades if t.is_closed]
        total_trades = len(closed_trades)

        if total_trades > 0:
            winning_trades = [t for t in closed_trades if t.pnl > 0]
            losing_trades = [t for t in closed_trades if t.pnl <= 0]

            win_rate = len(winning_trades) / total_trades

            total_wins = sum(t.pnl for t in winning_trades)
            total_losses = abs(sum(t.pnl for t in losing_trades))

            profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
        else:
            win_rate = 0.0
            profit_factor = 0.0

        # Friction statistics
        avg_latency = np.mean(self.latency_events) if self.latency_events else 0.0
        partial_fill_rate = self.partial_fill_events / max(total_trades, 1)

        return BacktestResult(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=self.max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            total_fees=self.total_fees,
            total_slippage=self.total_slippage,
            avg_latency_ms=avg_latency,
            partial_fill_rate=partial_fill_rate,
            liquidity_impact_events=self.liquidity_impact_events,
            daily_returns=returns,
            equity_curve=equity_series,
            trade_history=self.trades,
            position_history=list(self.positions.values()),
        )


def run_realistic_backtest(
    price_data: pd.DataFrame,
    signal_data: pd.DataFrame,
    initial_capital: float = 100000.0,
    exchange: str = "kraken",
    enable_all_frictions: bool = True,
) -> BacktestResult:
    """High-level function to run realistic backtest"""

    # Create configurations
    config = BacktestConfig(
        initial_capital=initial_capital,
        enable_slippage=enable_all_frictions,
        enable_fees=enable_all_frictions,
        enable_latency=enable_all_frictions,
        enable_partial_fills=enable_all_frictions,
    )

    exchange_config = ExchangeConfig(exchange)

    # Create backtester
    backtester = RealisticBacktester(config, exchange_config, price_data)

    # Run backtest
    return backtester.run_backtest(signal_data)


def compare_friction_impact(
    price_data: pd.DataFrame,
    signal_data: pd.DataFrame,
    initial_capital: float = 100000.0,
    exchange: str = "kraken",
) -> Dict[str, BacktestResult]:
    """Compare backtest results with and without market frictions"""

    results = {}

    # Perfect execution (no frictions)
    results["perfect"] = run_realistic_backtest(
        price_data, signal_data, initial_capital, exchange, enable_all_frictions=False
    )

    # Realistic execution (with frictions)
    results["realistic"] = run_realistic_backtest(
        price_data, signal_data, initial_capital, exchange, enable_all_frictions=True
    )

    # Calculate friction impact
    perfect_return = results["perfect"].total_return
    realistic_return = results["realistic"].total_return
    friction_impact = perfect_return - realistic_return

    results["friction_analysis"] = {
        "perfect_return": perfect_return,
        "realistic_return": realistic_return,
        "friction_impact": friction_impact,
        "friction_impact_pct": friction_impact / abs(perfect_return) if perfect_return != 0 else 0,
        "total_fees": results["realistic"].total_fees,
        "total_slippage": results["realistic"].total_slippage,
        "avg_latency_ms": results["realistic"].avg_latency_ms,
        "partial_fill_rate": results["realistic"].partial_fill_rate,
    }

    return results
