"""
Enhanced Backtest Agent - Addresses slippage, latency, smart order routing
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
import random
import time

from utils.daily_logger import get_daily_logger


@dataclass
class RealWorldConstraints:
    """Real-world trading constraints"""

    slippage_base: float = 0.0005  # 0.05% base slippage
    slippage_impact: float = 0.0001  # Additional slippage per $1000
    latency_ms: float = 50  # Average execution latency
    api_failure_rate: float = 0.001  # 0.1% API failure rate
    max_order_size: float = 0.01  # Max 1% of daily volume
    liquidity_penalty: float = 0.002  # 0.2% penalty for large orders


@dataclass
class BacktestResult:
    """Enhanced backtest result with realistic metrics"""

    strategy_name: str
    symbol: str
    period: str
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_duration: float
    slippage_cost: float
    latency_impact: float
    api_failures: int
    liquidity_score: float
    stress_test_results: Dict[str, float]
    timestamp: datetime


class MarketImpactModel:
    """Model market impact and slippage"""

    def __init__(self):
        self.logger = get_daily_logger().get_logger("trading_opportunities")

    def calculate_slippage(
        self,
        order_size_usd: float,
        daily_volume_usd: float,
        volatility: float,
        constraints: RealWorldConstraints,
    ) -> float:
        """Calculate realistic slippage based on order size and market conditions"""

        # Base slippage
        slippage = constraints.slippage_base

        # Volume impact
        volume_ratio = order_size_usd / max(daily_volume_usd, 1)
        if volume_ratio > constraints.max_order_size:
            # Heavy liquidity penalty for oversized orders
            slippage += constraints.liquidity_penalty * (volume_ratio / constraints.max_order_size)

        # Impact based on order size
        size_impact = (order_size_usd / 1000) * constraints.slippage_impact
        slippage += size_impact

        # Volatility adjustment
        volatility_multiplier = 1 + (volatility * 2)  # Higher vol = more slippage
        slippage *= volatility_multiplier

        # Market hours (higher slippage outside main hours)
        hour = datetime.now().hour
        if hour < 8 or hour > 20:  # Outside main trading hours
            slippage *= 1.5

        return min(slippage, 0.05)  # Cap at 5%

    def simulate_latency_impact(
        self, price_at_signal: float, price_at_execution: float, latency_ms: float
    ) -> Tuple[float, float]:
        """Simulate impact of execution latency"""

        # Price movement during latency period
        price_change = (price_at_execution - price_at_signal) / price_at_signal

        # Latency cost (missed opportunity)
        latency_cost = abs(price_change) * (latency_ms / 1000)  # Scale by latency

        return price_at_execution, latency_cost

    def simulate_api_failure(self, failure_rate: float) -> bool:
        """Simulate API failures and connection issues"""
        return random.random() < failure_rate


class SmartOrderRouter:
    """Smart order routing with anti-manipulation features"""

    def __init__(self):
        self.logger = get_daily_logger().get_logger("trading_opportunities")
        self.order_history = []
        self.pump_detection_window = 300  # 5 minutes

    def route_order(
        self, symbol: str, order_size: float, order_type: str, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Route order with anti-pump protection"""

        # Check for pump conditions
        if self._detect_pump_risk(symbol, order_type):
            return {
                "status": "rejected",
                "reason": "pump_risk_detected",
                "suggested_delay": 600,  # 10 minutes
            }

        # Check for front-running risk
        if self._detect_frontrun_risk(symbol, order_size):
            return {
                "status": "modified",
                "reason": "frontrun_protection",
                "suggested_chunks": self._calculate_order_chunks(order_size),
                "suggested_delay_between_chunks": 30,
            }

        # Normal routing
        routing_result = self._select_optimal_routing(symbol, order_size, market_data)

        # Record order for future analysis
        self._record_order(symbol, order_size, order_type)

        return routing_result

    def _detect_pump_risk(self, symbol: str, order_type: str) -> bool:
        """Detect if placing order might contribute to pump"""

        # Check recent order frequency
        recent_orders = [
            order
            for order in self.order_history
            if order["symbol"] == symbol
            and order["timestamp"] > time.time() - self.pump_detection_window
        ]

        # Too many buy orders in short period
        if order_type == "buy" and len(recent_orders) > 3:
            buy_orders = [o for o in recent_orders if o["order_type"] == "buy"]
            if len(buy_orders) >= len(recent_orders) * 0.8:  # 80% buy orders
                return True

        return False

    def _detect_frontrun_risk(self, symbol: str, order_size: float) -> bool:
        """Detect front-running risk for large orders"""

        # Large orders are more susceptible to front-running
        return order_size > 10000  # $10,000 threshold

    def _calculate_order_chunks(self, total_size: float) -> List[float]:
        """Calculate optimal order chunking"""

        num_chunks = min(int(total_size / 2000), 10)  # Max 10 chunks
        chunk_size = total_size / max(num_chunks, 1)

        # Add randomization to avoid pattern detection
        chunks = []
        for i in range(num_chunks):
            variation = random.uniform(0.8, 1.2)
            chunks.append(chunk_size * variation)

        # Normalize to total size
        actual_total = sum(chunks)
        chunks = [chunk * (total_size / actual_total) for chunk in chunks]

        return chunks

    def _select_optimal_routing(
        self, symbol: str, order_size: float, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Select optimal exchange/routing"""

        # For now, return simplified routing
        return {
            "status": "accepted",
            "exchange": "kraken",  # Would be dynamic based on liquidity
            "estimated_slippage": 0.0005,
            "estimated_execution_time": 2.5,
        }

    def _record_order(self, symbol: str, order_size: float, order_type: str):
        """Record order for pattern analysis"""

        self.order_history.append(
            {
                "symbol": symbol,
                "order_size": order_size,
                "order_type": order_type,
                "timestamp": time.time(),
            }
        )

        # Keep only recent history
        cutoff_time = time.time() - 3600  # 1 hour
        self.order_history = [
            order for order in self.order_history if order["timestamp"] > cutoff_time
        ]


class StressTester:
    """Stress test strategies under extreme conditions"""

    def __init__(self):
        self.logger = get_daily_logger().get_logger("trading_opportunities")

    def run_stress_tests(
        self, strategy_results: Dict[str, Any], historical_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Run comprehensive stress tests"""

        stress_results = {}

        # Flash crash simulation
        stress_results["flash_crash"] = self._simulate_flash_crash(
            strategy_results, historical_data
        )

        # High volatility periods
        stress_results["high_volatility"] = self._simulate_high_volatility(
            strategy_results, historical_data
        )

        # Liquidity crisis
        stress_results["liquidity_crisis"] = self._simulate_liquidity_crisis(
            strategy_results, historical_data
        )

        # Exchange outages
        stress_results["exchange_outage"] = self._simulate_exchange_outage(
            strategy_results, historical_data
        )

        # Regulatory shutdown
        stress_results["regulatory_risk"] = self._simulate_regulatory_shutdown(
            strategy_results, historical_data
        )

        return stress_results

    def _simulate_flash_crash(self, results: Dict[str, Any], data: pd.DataFrame) -> float:
        """Simulate flash crash scenario"""

        # Simulate 30% drop in 15 minutes
        crash_impact = -0.30
        recovery_time = 24  # hours

        # Calculate how strategy would perform
        portfolio_value = results.get("final_portfolio_value", 10000)

        # Assume some positions would be stopped out
        stop_loss_triggered = 0.7  # 70% of positions

        # Calculate losses
        direct_loss = portfolio_value * crash_impact * stop_loss_triggered

        # Normalize to percentage impact
        return direct_loss / portfolio_value

    def _simulate_high_volatility(self, results: Dict[str, Any], data: pd.DataFrame) -> float:
        """Simulate sustained high volatility"""

        # Increased slippage and false signals
        normal_performance = results.get("total_return", 0)
        volatility_penalty = 0.15  # 15% performance degradation

        return normal_performance * (1 - volatility_penalty)

    def _simulate_liquidity_crisis(self, results: Dict[str, Any], data: pd.DataFrame) -> float:
        """Simulate liquidity crisis"""

        # Higher slippage, failed executions
        normal_performance = results.get("total_return", 0)
        liquidity_penalty = 0.25  # 25% performance degradation

        return normal_performance * (1 - liquidity_penalty)

    def _simulate_exchange_outage(self, results: Dict[str, Any], data: pd.DataFrame) -> float:
        """Simulate exchange outages"""

        # Missed opportunities, forced hodling
        normal_performance = results.get("total_return", 0)
        outage_penalty = 0.10  # 10% performance degradation

        return normal_performance * (1 - outage_penalty)

    def _simulate_regulatory_shutdown(self, results: Dict[str, Any], data: pd.DataFrame) -> float:
        """Simulate regulatory shutdown scenario"""

        # Forced liquidation at unfavorable prices
        portfolio_value = results.get("final_portfolio_value", 10000)
        liquidation_penalty = 0.20  # 20% forced liquidation loss

        return -liquidation_penalty


class EnhancedBacktestAgent:
    """Professional backtesting with realistic constraints"""

    def __init__(self):
        self.logger = get_daily_logger().get_logger("trading_opportunities")
        self.market_impact_model = MarketImpactModel()
        self.smart_router = SmartOrderRouter()
        self.stress_tester = StressTester()
        self.constraints = RealWorldConstraints()

    async def backtest_strategy(
        self,
        strategy_signals: pd.DataFrame,
        market_data: pd.DataFrame,
        symbol: str,
        strategy_name: str,
    ) -> BacktestResult:
        """Run comprehensive backtest with real-world constraints"""

        self.logger.info(f"Starting enhanced backtest for {strategy_name} on {symbol}")

        # Initialize portfolio
        initial_capital = 10000
        portfolio = {
            "cash": initial_capital,
            "position": 0,
            "entry_price": 0,
            "total_value": initial_capital,
        }

        # Trading metrics
        trades = []
        equity_curve = []
        total_slippage_cost = 0
        total_latency_impact = 0
        api_failures = 0

        # Process each signal
        for i, signal_row in strategy_signals.iterrows():
            try:
                # Get market data at signal time
                market_row = market_data.iloc[i] if i < len(market_data) else market_data.iloc[-1]

                # Process signal with realistic constraints
                trade_result = await self._process_signal_with_constraints(
                    signal_row, market_row, portfolio, symbol
                )

                if trade_result:
                    trades.append(trade_result)
                    total_slippage_cost += trade_result.get("slippage_cost", 0)
                    total_latency_impact += trade_result.get("latency_impact", 0)

                    if trade_result.get("api_failure", False):
                        api_failures += 1

                # Record portfolio value
                current_value = self._calculate_portfolio_value(portfolio, market_row["close"])
                equity_curve.append({"timestamp": signal_row["timestamp"], "value": current_value})

            except Exception as e:
                self.logger.error(f"Error processing signal at index {i}: {e}")
                continue

        # Calculate performance metrics
        performance_metrics = self._calculate_enhanced_metrics(
            trades, equity_curve, initial_capital
        )

        # Run stress tests
        stress_results = self.stress_tester.run_stress_tests(performance_metrics, market_data)

        # Calculate liquidity score
        liquidity_score = self._calculate_liquidity_score(trades, market_data)

        result = BacktestResult(
            strategy_name=strategy_name,
            symbol=symbol,
            period=f"{len(strategy_signals)} periods",
            total_return=performance_metrics["total_return"],
            sharpe_ratio=performance_metrics["sharpe_ratio"],
            max_drawdown=performance_metrics["max_drawdown"],
            win_rate=performance_metrics["win_rate"],
            profit_factor=performance_metrics["profit_factor"],
            total_trades=len(trades),
            avg_trade_duration=performance_metrics["avg_trade_duration"],
            slippage_cost=total_slippage_cost,
            latency_impact=total_latency_impact,
            api_failures=api_failures,
            liquidity_score=liquidity_score,
            stress_test_results=stress_results,
            timestamp=datetime.now(),
        )

        self.logger.info(
            f"Enhanced backtest complete: {performance_metrics['total_return']:.2%} return"
        )

        return result

    async def _process_signal_with_constraints(
        self, signal_row: pd.Series, market_row: pd.Series, portfolio: Dict, symbol: str
    ) -> Optional[Dict]:
        """Process trading signal with realistic constraints"""

        signal = signal_row.get("signal", 0)
        current_price = market_row["close"]
        volume_usd = market_row.get("volume", 1000000)  # Default volume

        if signal == 0:  # No signal
            return None

        # Calculate order size
        if signal > 0 and portfolio["position"] == 0:  # Buy signal
            order_size_usd = portfolio["cash"] * 0.95  # Use 95% of cash
            order_type = "buy"
        elif signal < 0 and portfolio["position"] > 0:  # Sell signal
            order_size_usd = portfolio["position"] * current_price
            order_type = "sell"
        else:
            return None

        # Check smart routing
        routing_result = self.smart_router.route_order(
            symbol, order_size_usd, order_type, {"volume": volume_usd}
        )

        if routing_result["status"] == "rejected":
            return {
                "status": "rejected",
                "reason": routing_result["reason"],
                "timestamp": signal_row["timestamp"],
            }

        # Calculate slippage
        volatility = self._calculate_volatility(market_row)
        slippage = self.market_impact_model.calculate_slippage(
            order_size_usd, volume_usd, volatility, self.constraints
        )

        # Simulate API failure
        if self.market_impact_model.simulate_api_failure(self.constraints.api_failure_rate):
            return {
                "status": "failed",
                "reason": "api_failure",
                "api_failure": True,
                "timestamp": signal_row["timestamp"],
            }

        # Simulate latency
        execution_delay = random.uniform(20, 100)  # 20-100ms
        price_drift = random.uniform(-0.0005, 0.0005)  # Small price movement
        execution_price = current_price * (1 + price_drift)

        final_price, latency_cost = self.market_impact_model.simulate_latency_impact(
            current_price, execution_price, execution_delay
        )

        # Apply slippage
        if order_type == "buy":
            final_price *= 1 + slippage
            shares = order_size_usd / final_price
            portfolio["position"] = shares
            portfolio["entry_price"] = final_price
            portfolio["cash"] -= order_size_usd
        else:  # sell
            final_price *= 1 - slippage
            proceeds = portfolio["position"] * final_price
            portfolio["cash"] = proceeds
            portfolio["position"] = 0
            portfolio["entry_price"] = 0

        return {
            "status": "executed",
            "order_type": order_type,
            "price": final_price,
            "size": order_size_usd,
            "slippage": slippage,
            "slippage_cost": order_size_usd * slippage,
            "latency_impact": latency_cost,
            "timestamp": signal_row["timestamp"],
            "api_failure": False,
        }

    def _calculate_portfolio_value(self, portfolio: Dict, current_price: float) -> float:
        """Calculate current portfolio value"""
        if portfolio["position"] > 0:
            return portfolio["cash"] + (portfolio["position"] * current_price)
        return portfolio["cash"]

    def _calculate_volatility(self, market_row: pd.Series) -> float:
        """Calculate simple volatility estimate"""
        high = market_row.get("high", market_row["close"])
        low = market_row.get("low", market_row["close"])
        close = market_row["close"]

        # Simple volatility estimate
        return (high - low) / close if close > 0 else 0.01

    def _calculate_enhanced_metrics(
        self, trades: List[Dict], equity_curve: List[Dict], initial_capital: float
    ) -> Dict[str, float]:
        """Calculate enhanced performance metrics"""

        if not equity_curve:
            return {
                "total_return": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "profit_factor": 0,
                "avg_trade_duration": 0,
                "final_portfolio_value": initial_capital,
            }

        # Calculate returns
        values = [point["value"] for point in equity_curve]
        final_value = values[-1]
        total_return = (final_value - initial_capital) / initial_capital

        # Calculate Sharpe ratio
        if len(values) > 1:
            returns = np.diff(values) / values[:-1]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        # Calculate max drawdown
        running_max = np.maximum.accumulate(values)
        drawdowns = (values - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))

        # Trade statistics
        executed_trades = [t for t in trades if t.get("status") == "executed"]

        if executed_trades:
            # Win rate (simplified)
            profitable_trades = len([t for t in executed_trades if t.get("profit", 0) > 0])
            win_rate = profitable_trades / len(executed_trades)

            # Average trade duration (simplified)
            avg_trade_duration = 24  # Default 24 hours

            # Profit factor (simplified)
            total_profit = sum(max(0, t.get("profit", 0)) for t in executed_trades)
            total_loss = sum(abs(min(0, t.get("profit", 0))) for t in executed_trades)
            profit_factor = total_profit / max(total_loss, 1)
        else:
            win_rate = 0
            avg_trade_duration = 0
            profit_factor = 0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_trade_duration": avg_trade_duration,
            "final_portfolio_value": final_value,
        }

    def _calculate_liquidity_score(self, trades: List[Dict], market_data: pd.DataFrame) -> float:
        """Calculate liquidity score based on execution quality"""

        if not trades:
            return 0.5

        executed_trades = [t for t in trades if t.get("status") == "executed"]

        if not executed_trades:
            return 0.0

        # Average slippage as proxy for liquidity
        avg_slippage = np.mean([t.get("slippage", 0) for t in executed_trades])

        # Convert to score (lower slippage = higher score)
        liquidity_score = max(0, 1 - (avg_slippage / 0.01))  # Normalize by 1% slippage

        return liquidity_score

    def get_status(self) -> Dict:
        """Get agent status"""
        return {
            "agent": "enhanced_backtest",
            "status": "operational",
            "market_impact_modeling": True,
            "smart_order_routing": True,
            "stress_testing": True,
            "realistic_constraints": True,
        }


# Global instance
backtest_agent = EnhancedBacktestAgent()
