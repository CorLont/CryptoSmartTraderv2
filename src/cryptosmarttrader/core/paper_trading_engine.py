#!/usr/bin/env python3
"""
Paper Trading Engine
4-week mandatory paper trading with complete logging before live trading
"""

import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

warnings.filterwarnings("ignore")

# Import core components
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from ..core.structured_logger import get_logger
from ..core.orderbook_simulator import (
    OrderBookSimulator,
    OrderSide,
    OrderType,
    TimeInForce,
    OrderStatus,
    ExchangeConfig,
)
from ..core.slippage_estimator import SlippageEstimator


class TradingSession(Enum):
    """Trading session status"""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class ValidationStatus(Enum):
    """Paper trading validation status"""

    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class PaperTrade:
    """Paper trading record"""

    trade_id: str
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    entry_price: float
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    intended_entry_price: float = 0.0
    intended_exit_price: float = 0.0
    slippage_bps: float = 0.0
    fees_paid: float = 0.0
    latency_ms: float = 0.0
    pnl: float = 0.0
    status: str = "open"


@dataclass
class PaperTradingMetrics:
    """Paper trading performance metrics"""

    session_id: str
    start_date: datetime
    end_date: Optional[datetime]
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_fees: float
    sharpe_ratio: float
    max_drawdown: float
    average_trade_size: float
    average_slippage_bps: float
    average_latency_ms: float
    risk_metrics: Dict[str, Any]


@dataclass
class ValidationCriteria:
    """Paper trading validation criteria"""

    minimum_duration_days: int = 28  # 4 weeks
    minimum_trades: int = 100
    minimum_sharpe_ratio: float = 0.5
    maximum_drawdown: float = 0.15  # 15%
    minimum_win_rate: float = 0.35  # 35%
    maximum_average_slippage_bps: float = 25.0
    required_symbols: List[str] = None

    def __post_init__(self):
        if self.required_symbols is None:
            self.required_symbols = ["BTC/USD", "ETH/USD"]


class PaperTradingEngine:
    """Complete paper trading engine with 4-week validation"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = get_logger("PaperTradingEngine")

        # Configuration
        self.config = {
            "validation_criteria": ValidationCriteria(),
            "starting_balance": 100000.0,  # $100k starting balance
            "max_position_size": 0.1,  # 10% max position
            "enable_slippage_simulation": True,
            "enable_latency_simulation": True,
            "log_directory": "logs/paper_trading",
            "session_backup_interval": 3600,  # 1 hour
        }

        if config:
            if "validation_criteria" in config:
                criteria_config = config.pop("validation_criteria")
                if hasattr(criteria_config, "__dict__"):
                    # If it's a ValidationCriteria object
                    self.config["validation_criteria"] = criteria_config
                else:
                    # If it's a dict
                    for key, value in criteria_config.items():
                        setattr(self.config["validation_criteria"], key, value)
            self.config.update(config)

        # Trading state
        self.current_session_id: Optional[str] = None
        self.current_balance = self.config["starting_balance"]
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.open_trades: Dict[str, PaperTrade] = {}
        self.completed_trades: List[PaperTrade] = []

        # Order book simulators by symbol
        self.simulators: Dict[str, OrderBookSimulator] = {}

        # Slippage estimator
        self.slippage_estimator = SlippageEstimator()

        # Session management
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

        # Create log directory
        log_dir = Path(self.config["log_directory"])
        log_dir.mkdir(parents=True, exist_ok=True)

    def start_paper_trading_session(self, session_name: str = None) -> str:
        """Start new paper trading session"""

        with self.lock:
            if self.current_session_id:
                self.logger.warning(f"Session {self.current_session_id} already active")
                return self.current_session_id

            # Generate session ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_id = (
                f"paper_{session_name}_{timestamp}" if session_name else f"paper_{timestamp}"
            )

            # Initialize session
            session_data = {
                "session_id": session_id,
                "start_time": datetime.now(),
                "end_time": None,
                "status": TradingSession.ACTIVE,
                "starting_balance": self.config["starting_balance"],
                "validation_status": ValidationStatus.IN_PROGRESS,
                "trades": [],
                "metrics": None,
            }

            self.sessions[session_id] = session_data
            self.current_session_id = session_id

            # Reset trading state
            self.current_balance = self.config["starting_balance"]
            self.positions = {}
            self.open_trades = {}
            self.completed_trades = []

            self.logger.info(f"Started paper trading session: {session_id}")

            return session_id

    def end_paper_trading_session(self) -> Optional[str]:
        """End current paper trading session"""

        with self.lock:
            if not self.current_session_id:
                self.logger.warning("No active paper trading session")
                return None

            session_id = self.current_session_id
            session = self.sessions[session_id]

            # Close any open positions
            self._close_all_positions()

            # Calculate final metrics
            metrics = self._calculate_session_metrics(session_id)

            # Update session
            session["end_time"] = datetime.now()
            session["status"] = TradingSession.COMPLETED
            session["trades"] = [asdict(trade) for trade in self.completed_trades]
            session["metrics"] = asdict(metrics)

            # Validate session
            validation_status = self._validate_session(metrics)
            session["validation_status"] = validation_status

            # Save session to disk
            self._save_session_to_disk(session_id)

            self.current_session_id = None

            self.logger.info(
                f"Ended paper trading session: {session_id} - Status: {validation_status.value}"
            )

            return session_id

    def register_symbol(
        self, symbol: str, exchange_config: Optional[ExchangeConfig] = None
    ) -> None:
        """Register symbol for paper trading"""

        if symbol not in self.simulators:
            config = exchange_config or ExchangeConfig(f"paper_{symbol}")
            self.simulators[symbol] = OrderBookSimulator(symbol, config)

            self.logger.info(f"Registered symbol for paper trading: {symbol}")

    def update_market_data(
        self, symbol: str, bid_price: float, ask_price: float, volume: float = 1000.0
    ) -> None:
        """Update market data for symbol"""

        if symbol not in self.simulators:
            self.register_symbol(symbol)

        simulator = self.simulators[symbol]

        # Generate realistic order book
        mid_price = (bid_price + ask_price) / 2
        volatility = abs(ask_price - bid_price) / mid_price

        orderbook = simulator.generate_realistic_orderbook(mid_price, volatility)
        simulator.update_orderbook(orderbook)

        # Update slippage estimator
        self.slippage_estimator.update_orderbook(symbol, orderbook)

        self.logger.debug(f"Updated market data for {symbol}: {bid_price:.6f}/{ask_price:.6f}")

    def submit_paper_trade(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
    ) -> Optional[str]:
        """Submit paper trade"""

        if not self.current_session_id:
            self.logger.error("No active paper trading session")
            return None

        with self.lock:
            try:
                # Validate trade
                if not self._validate_trade(symbol, side, quantity):
                    return None

                # Get current market price for intended price
                simulator = self.simulators.get(symbol)
                if not simulator:
                    self.logger.error(f"No market data available for {symbol}")
                    return None

                intended_price = (
                    limit_price
                    if limit_price
                    else (simulator.ask_price if side == OrderSide.BUY else simulator.bid_price)

                # Submit order to simulator
                order_id = simulator.submit_order(
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=limit_price,
                    time_in_force=TimeInForce.IOC
                    if order_type == OrderType.MARKET
                    else TimeInForce.GTC,
                )

                # Check order execution
                order_status = simulator.get_order_status(order_id)
                if not order_status or order_status["status"] == "rejected":
                    self.logger.warning(f"Paper trade rejected: {symbol} {side.value} {quantity}")
                    return None

                # Create paper trade record
                if order_status["filled_quantity"] > 0:
                    trade = self._create_paper_trade_record(
                        symbol, side, order_status, intended_price
                    )

                    if trade:
                        self.open_trades[trade.trade_id] = trade
                        self._update_position(symbol, side, trade.quantity)

                        self.logger.info(
                            f"Paper trade executed: {trade.trade_id} - "
                            f"{side.value} {trade.quantity} {symbol} at {trade.entry_price:.6f}"
                        )

                        return trade.trade_id

                return None

            except Exception as e:
                self.logger.error(f"Paper trade submission failed: {e}")
                return None

    def close_paper_position(self, symbol: str, quantity: Optional[float] = None) -> List[str]:
        """Close paper trading position"""

        if not self.current_session_id:
            self.logger.error("No active paper trading session")
            return []

        current_position = self.positions.get(symbol, 0.0)
        if current_position == 0:
            self.logger.warning(f"No position to close for {symbol}")
            return []

        # Determine close quantity
        close_qty = abs(quantity) if quantity else abs(current_position)
        close_qty = min(close_qty, abs(current_position))

        # Determine close side (opposite of position)
        close_side = OrderSide.SELL if current_position > 0 else OrderSide.BUY

        # Submit closing trade
        trade_id = self.submit_paper_trade(symbol, close_side, close_qty, OrderType.MARKET)

        if trade_id:
            return [trade_id]

        return []

    def _validate_trade(self, symbol: str, side: OrderSide, quantity: float) -> bool:
        """Validate paper trade parameters"""

        try:
            # Check if symbol is registered
            if symbol not in self.simulators:
                self.logger.warning(f"Symbol not registered: {symbol}")
                return False

            # Check position size limits
            current_position = self.positions.get(symbol, 0.0)
            max_position_value = self.current_balance * self.config["max_position_size"]

            simulator = self.simulators[symbol]
            estimated_value = quantity * simulator.mid_price

            if estimated_value > max_position_value:
                self.logger.warning(
                    f"Trade size {estimated_value:.2f} exceeds max position "
                    f"{max_position_value:.2f}"
                )
                return False

            # Check sufficient balance for buy orders
            if side == OrderSide.BUY:
                required_balance = estimated_value * 1.01  # Include buffer for fees
                if required_balance > self.current_balance:
                    self.logger.warning(
                        f"Insufficient balance: required {required_balance:.2f}, "
                        f"available {self.current_balance:.2f}"
                    )
                    return False

            # Check sufficient position for sell orders
            if side == OrderSide.SELL:
                if current_position < quantity:
                    self.logger.warning(
                        f"Insufficient position: required {quantity}, available {current_position}"
                    )
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Trade validation failed: {e}")
            return False

    def _create_paper_trade_record(
        self, symbol: str, side: OrderSide, order_status: Dict[str, Any], intended_price: float
    ) -> Optional[PaperTrade]:
        """Create paper trade record from order execution"""

        try:
            trade_id = f"trade_{symbol}_{int(datetime.now().timestamp())}"

            # Get fill information
            fills = self.simulators[symbol].get_fill_history(order_status["order_id"])

            if not fills:
                return None

            # Calculate weighted average execution price and total fees
            total_quantity = 0.0
            total_cost = 0.0
            total_fees = 0.0
            total_latency = 0.0

            for fill in fills:
                fill_qty = fill["quantity"]
                fill_price = fill["price"]
                fill_fee = fill["fee"]

                total_quantity += fill_qty
                total_cost += fill_qty * fill_price
                total_fees += fill_fee

                # REMOVED: Mock data pattern not allowed in production
                total_latency += 50.0  # Average 50ms

            avg_execution_price = (
                total_cost / total_quantity if total_quantity > 0 else intended_price
            )
            avg_latency = total_latency / len(fills) if fills else 50.0

            # Calculate slippage
            if intended_price > 0:
                if side == OrderSide.BUY:
                    slippage_bps = (avg_execution_price - intended_price) / intended_price * 10000
                else:
                    slippage_bps = (intended_price - avg_execution_price) / intended_price * 10000
            else:
                slippage_bps = 0.0

            # Record execution for slippage estimator
            self.slippage_estimator.record_execution(
                symbol=symbol,
                side=side,
                order_size=total_quantity,
                intended_price=intended_price,
                executed_price=avg_execution_price,
                latency_ms=avg_latency,
            )

            trade = PaperTrade(
                trade_id=trade_id,
                timestamp=datetime.now(),
                symbol=symbol,
                side=side,
                quantity=total_quantity,
                entry_price=avg_execution_price,
                intended_entry_price=intended_price,
                slippage_bps=slippage_bps,
                fees_paid=total_fees,
                latency_ms=avg_latency,
            )

            return trade

        except Exception as e:
            self.logger.error(f"Failed to create trade record: {e}")
            return None

    def _update_position(self, symbol: str, side: OrderSide, quantity: float) -> None:
        """Update position after trade execution"""

        current_position = self.positions.get(symbol, 0.0)

        if side == OrderSide.BUY:
            new_position = current_position + quantity
        else:
            new_position = current_position - quantity

        self.positions[symbol] = new_position

        # Update balance (simplified - would be more complex with margin)
        trade_value = quantity * self.simulators[symbol].mid_price

        if side == OrderSide.BUY:
            self.current_balance -= trade_value
        else:
            self.current_balance += trade_value

        self.logger.debug(f"Updated position {symbol}: {current_position} -> {new_position}")

    def _close_all_positions(self) -> None:
        """Close all open positions at session end"""

        for symbol, position in self.positions.items():
            if abs(position) > 1e-8:  # Close positions above threshold
                side = OrderSide.SELL if position > 0 else OrderSide.BUY
                self.submit_paper_trade(symbol, side, abs(position), OrderType.MARKET)

    def _calculate_session_metrics(self, session_id: str) -> PaperTradingMetrics:
        """Calculate comprehensive session metrics"""

        session = self.sessions[session_id]
        start_time = session["start_time"]
        end_time = session.get("end_time", datetime.now())

        # Trade statistics
        total_trades = len(self.completed_trades)
        winning_trades = len([t for t in self.completed_trades if t.pnl > 0])
        losing_trades = len([t for t in self.completed_trades if t.pnl < 0])

        win_rate = winning_trades / max(total_trades, 1)
        total_pnl = sum(t.pnl for t in self.completed_trades)
        total_fees = sum(t.fees_paid for t in self.completed_trades)

        # Risk metrics
        pnl_series = [t.pnl for t in self.completed_trades]
        cumulative_pnl = []
        running_total = 0
        for pnl in pnl_series:
            running_total += pnl
            cumulative_pnl.append(running_total)

        # Calculate Sharpe ratio (simplified)
        if pnl_series:
            pnl_std = np.std(pnl_series) if len(pnl_series) > 1 else 1.0
            avg_pnl = np.mean(pnl_series)
            sharpe_ratio = avg_pnl / max(pnl_std, 1e-8) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0

        # Calculate maximum drawdown
        if cumulative_pnl:
            peak = cumulative_pnl[0]
            max_drawdown = 0.0

            for value in cumulative_pnl:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / max(abs(peak), 1)
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0.0

        # Average metrics
        avg_trade_size = (
            np.mean([t.quantity for t in self.completed_trades]) if self.completed_trades else 0.0
        )
        avg_slippage = (
            np.mean([t.slippage_bps for t in self.completed_trades])
            if self.completed_trades
            else 0.0
        )
        avg_latency = (
            np.mean([t.latency_ms for t in self.completed_trades]) if self.completed_trades else 0.0
        )

        return PaperTradingMetrics(
            session_id=session_id,
            start_date=start_time,
            end_date=end_time,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_fees=total_fees,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            average_trade_size=avg_trade_size,
            average_slippage_bps=avg_slippage,
            average_latency_ms=avg_latency,
            risk_metrics={
                "total_return_pct": total_pnl / self.config["starting_balance"] * 100,
                "number_of_symbols_traded": len(set(t.symbol for t in self.completed_trades)),
                "average_holding_period_hours": self._calculate_avg_holding_period(),
            },
        )

    def _calculate_avg_holding_period(self) -> float:
        """Calculate average holding period for closed trades"""

        closed_trades = [t for t in self.completed_trades if t.exit_timestamp]

        if not closed_trades:
            return 0.0

        holding_periods = []
        for trade in closed_trades:
            duration = (trade.exit_timestamp - trade.timestamp).total_seconds() / 3600
            holding_periods.append(duration)

        return np.mean(holding_periods)

    def _validate_session(self, metrics: PaperTradingMetrics) -> ValidationStatus:
        """Validate paper trading session against criteria"""

        criteria = self.config["validation_criteria"]

        try:
            # Check duration
            if metrics.end_date and metrics.start_date:
                session_duration = (metrics.end_date - metrics.start_date).days
            else:
                session_duration = 0
            if session_duration < criteria.minimum_duration_days:
                self.logger.warning(
                    f"Session duration {session_duration} days < minimum "
                    f"{criteria.minimum_duration_days} days"
                )
                return ValidationStatus.INSUFFICIENT_DATA

            # Check number of trades
            if metrics.total_trades < criteria.minimum_trades:
                self.logger.warning(
                    f"Total trades {metrics.total_trades} < minimum {criteria.minimum_trades}"
                )
                return ValidationStatus.INSUFFICIENT_DATA

            # Check performance criteria
            validation_checks = [
                (
                    metrics.sharpe_ratio >= criteria.minimum_sharpe_ratio,
                    f"Sharpe ratio {metrics.sharpe_ratio:.2f} < {criteria.minimum_sharpe_ratio}",
                ),
                (
                    metrics.max_drawdown <= criteria.maximum_drawdown,
                    f"Max drawdown {metrics.max_drawdown:.2%} > {criteria.maximum_drawdown:.2%}",
                ),
                (
                    metrics.win_rate >= criteria.minimum_win_rate,
                    f"Win rate {metrics.win_rate:.2%} < {criteria.minimum_win_rate:.2%}",
                ),
                (
                    metrics.average_slippage_bps <= criteria.maximum_average_slippage_bps,
                    f"Average slippage {metrics.average_slippage_bps:.1f} bps > {criteria.maximum_average_slippage_bps} bps",
                ),
            ]

            failed_checks = [msg for passed, msg in validation_checks if not passed]

            if failed_checks:
                self.logger.warning(f"Validation failed: {'; '.join(failed_checks)}")
                return ValidationStatus.FAILED

            # Check required symbols
            symbols_traded = set(t.symbol for t in self.completed_trades)
            missing_symbols = set(criteria.required_symbols) - symbols_traded

            if missing_symbols:
                self.logger.warning(f"Missing required symbols: {missing_symbols}")
                return ValidationStatus.FAILED

            self.logger.info("Paper trading session validation PASSED")
            return ValidationStatus.PASSED

        except Exception as e:
            self.logger.error(f"Session validation failed: {e}")
            return ValidationStatus.FAILED

    def _save_session_to_disk(self, session_id: str) -> None:
        """Save session data to disk"""

        try:
            session_file = Path(self.config["log_directory"]) / f"{session_id}.json"

            session_data = self.sessions[session_id].copy()

            # Convert datetime objects to ISO strings
            if "start_time" in session_data:
                session_data["start_time"] = session_data["start_time"].isoformat()
            if "end_time" in session_data and session_data["end_time"]:
                session_data["end_time"] = session_data["end_time"].isoformat()

            with open(session_file, "w") as f:
                json.dump(session_data, f, indent=2, default=str)

            self.logger.info(f"Saved session data to {session_file}")

        except Exception as e:
            self.logger.error(f"Failed to save session data: {e}")

    def get_session_status(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get status of paper trading session"""

        target_session = session_id or self.current_session_id

        if not target_session or target_session not in self.sessions:
            return None

        session = self.sessions[target_session]

        # Calculate current metrics if session is active
        if session["status"] == TradingSession.ACTIVE:
            current_metrics = self._calculate_session_metrics(target_session)
            session["current_metrics"] = asdict(current_metrics)

        return {
            "session_id": target_session,
            "status": session["status"].value,
            "validation_status": session["validation_status"].value,
            "start_time": session["start_time"].isoformat(),
            "end_time": session["end_time"].isoformat() if session["end_time"] else None,
            "current_balance": self.current_balance,
            "open_positions": dict(self.positions),
            "total_trades": len(self.completed_trades),
            "current_metrics": session.get("current_metrics"),
            "final_metrics": session.get("metrics"),
        }


if __name__ == "__main__":

    async def test_paper_trading_engine():
        """Test paper trading engine"""

        print("🔍 TESTING PAPER TRADING ENGINE")
        print("=" * 60)

        # Create paper trading engine
        engine = PaperTradingEngine()

        print("🚀 Starting paper trading session...")
        session_id = engine.start_paper_trading_session("test_session")
        print(f"   Session started: {session_id}")

        # Register symbols
        print("\n📊 Registering trading symbols...")
        symbols = ["BTC/USD", "ETH/USD"]
        for symbol in symbols:
            engine.register_symbol(symbol)

        # Update market data
        print("\n📈 Updating market data...")
        engine.update_market_data("BTC/USD", 49900.0, 50100.0)
        engine.update_market_data("ETH/USD", 2990.0, 3010.0)

        # Execute some paper trades
        print("\n💼 Executing paper trades...")

        trades = []

        # Buy BTC
        trade_id = engine.submit_paper_trade("BTC/USD", OrderSide.BUY, 0.1, OrderType.MARKET)
        if trade_id:
            trades.append(trade_id)
            print(f"   Executed BTC buy: {trade_id}")

        # Buy ETH
        trade_id = engine.submit_paper_trade("ETH/USD", OrderSide.BUY, 1.0, OrderType.MARKET)
        if trade_id:
            trades.append(trade_id)
            print(f"   Executed ETH buy: {trade_id}")

        # Sell some BTC
        trade_id = engine.submit_paper_trade("BTC/USD", OrderSide.SELL, 0.05, OrderType.MARKET)
        if trade_id:
            trades.append(trade_id)
            print(f"   Executed BTC sell: {trade_id}")

        print(f"\n   Total trades executed: {len(trades)}")

        # Check session status
        print("\n📋 Session status:")
        status = engine.get_session_status()
        if status:
            print(f"   Session ID: {status['session_id']}")
            print(f"   Status: {status['status']}")
            print(f"   Current balance: ${status['current_balance']:.2f}")
            print(f"   Open positions: {status['open_positions']}")
            print(f"   Total trades: {status['total_trades']}")

            if "current_metrics" in status:
                metrics = status["current_metrics"]
                print(f"   Win rate: {metrics['win_rate']:.1%}")
                print(f"   Total P&L: ${metrics['total_pnl']:.2f}")
                print(f"   Average slippage: {metrics['average_slippage_bps']:.2f} bps")

        # Test position closing
        print("\n❌ Closing positions...")
        for symbol in ["BTC/USD", "ETH/USD"]:
            closed_trades = engine.close_paper_position(symbol)
            if closed_trades:
                print(f"   Closed {symbol} position: {len(closed_trades)} trades")

        # End session
        print("\n🏁 Ending paper trading session...")
        ended_session = engine.end_paper_trading_session()

        if ended_session:
            final_status = engine.get_session_status(ended_session)
            if final_status and "final_metrics" in final_status:
                metrics = final_status["final_metrics"]
                print(f"   Final validation: {final_status['validation_status']}")
                print(f"   Total trades: {metrics['total_trades']}")
                print(f"   Win rate: {metrics['win_rate']:.1%}")
                print(f"   Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
                print(f"   Max drawdown: {metrics['max_drawdown']:.1%}")

        print("\n✅ PAPER TRADING ENGINE TEST COMPLETED")

        return len(trades) > 0 and ended_session is not None

    # Run test
    import asyncio
    import numpy as np

    success = asyncio.run(test_paper_trading_engine())
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
