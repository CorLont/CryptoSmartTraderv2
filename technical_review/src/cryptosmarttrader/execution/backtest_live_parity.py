"""
Backtest-Live Parity System

Advanced execution simulation ensuring perfect parity between
backtest and live trading performance through comprehensive modeling.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import statistics
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class ExecutionStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"


class SlippageComponent(Enum):
    """Components of execution slippage"""
    MARKET_IMPACT = "market_impact"
    SPREAD_COST = "spread_cost"
    TIMING_DELAY = "timing_delay"
    PARTIAL_FILLS = "partial_fills"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_SHORTAGE = "liquidity_shortage"
    NETWORK_LATENCY = "network_latency"
    QUEUE_PROCESSING = "queue_processing"
    FEE_IMPACT = "fee_impact"
    PRICE_IMPROVEMENT = "price_improvement"


@dataclass
class MarketDepth:
    """Order book depth at execution time"""
    bids: List[Tuple[float, float]]  # [(price, quantity), ...]
    asks: List[Tuple[float, float]]
    timestamp: datetime

    @property
    def spread_bps(self) -> float:
        """Calculate bid-ask spread in basis points"""
        if self.bids and self.asks:
            best_bid = self.bids[0][0]
            best_ask = self.asks[0][0]
            mid_price = (best_bid + best_ask) / 2
            spread = best_ask - best_bid
            return (spread / mid_price) * 10000  # basis points
        return 0.0

    @property
    def depth_usd(self) -> Dict[str, float]:
        """Calculate order book depth in USD"""
        bid_depth = sum(price * quantity for price, quantity in self.bids[:10])
        ask_depth = sum(price * quantity for price, quantity in self.asks[:10])
        return {"bid_depth": bid_depth, "ask_depth": ask_depth}


@dataclass
class ExecutionResult:
    """Result of order execution (real or simulated)"""
    order_id: str
    symbol: str
    side: str
    requested_quantity: float

    # Execution details
    filled_quantity: float
    average_price: float
    execution_time_ms: float
    status: ExecutionStatus

    # Market context
    market_depth: Optional[MarketDepth] = None
    mid_price: Optional[float] = None

    # Slippage analysis
    slippage_bps: float = 0.0
    slippage_components: Dict[str, float] = field(default_factory=dict)

    # Fees
    maker_fee_usd: float = 0.0
    taker_fee_usd: float = 0.0
    total_fees_usd: float = 0.0

    # Timing
    submitted_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def fill_rate(self) -> float:
        """Calculate fill rate percentage"""
        if self.requested_quantity == 0:
            return 0.0
        return (self.filled_quantity / abs(self.requested_quantity)) * 100

    @property
    def total_slippage_bps(self) -> float:
        """Calculate total slippage in basis points"""
        return sum(self.slippage_components.values())


class ExecutionSimulator:
    """
    Advanced execution simulator for backtest-live parity
    """

    def __init__(self):
        # Market microstructure parameters
        self.typical_spreads = {
            "BTC/USD": 2.0,   # basis points
            "ETH/USD": 3.0,
            "default": 5.0
        }

        self.market_impact_coefficients = {
            "BTC/USD": 0.1,   # Impact per $1M notional
            "ETH/USD": 0.15,
            "default": 0.25
        }

        # Latency modeling
        self.network_latency_ms = {
            "p50": 50,
            "p95": 150,
            "p99": 300
        }

        self.exchange_processing_ms = {
            "p50": 20,
            "p95": 80,
            "p99": 200
        }

        # Fee structures (maker/taker)
        self.fee_structures = {
            "kraken": {
                "maker": 0.0016,  # 0.16%
                "taker": 0.0026   # 0.26%
            },
            "default": {
                "maker": 0.0010,
                "taker": 0.0025
            }
        }

        # Historical execution data for calibration
        self.execution_history: List[ExecutionResult] = []
        self.calibration_data = {}

    def simulate_execution(self,
                          symbol: str,
                          side: str,
                          quantity: float,
                          order_type: str,
                          limit_price: Optional[float] = None,
                          market_depth: Optional[MarketDepth] = None,
                          market_conditions: Optional[Dict[str, float]] = None) -> ExecutionResult:
        """Simulate order execution with realistic market microstructure"""

        try:
            # Generate order ID
            order_id = f"SIM_{int(datetime.now().timestamp() * 1000000)}"

            # Determine execution parameters
            mid_price = self._get_mid_price(symbol, market_depth)
            spread_bps = self._get_spread_bps(symbol, market_depth)

            # Calculate execution components
            execution_time_ms = self._simulate_execution_time(quantity)
            slippage_components = self._calculate_slippage_components(
                symbol, side, quantity, order_type, mid_price, spread_bps, market_conditions
            )

            # Determine fill characteristics
            fill_rate, average_price = self._simulate_fill_execution(
                symbol, side, quantity, order_type, limit_price, mid_price, slippage_components
            )

            filled_quantity = quantity * (fill_rate / 100)

            # Calculate fees
            fees = self._calculate_fees(symbol, filled_quantity, average_price, order_type)

            # Determine execution status
            status = self._determine_execution_status(fill_rate, order_type)

            # Create execution result
            result = ExecutionResult(
                order_id=order_id,
                symbol=symbol,
                side=side,
                requested_quantity=quantity,
                filled_quantity=filled_quantity,
                average_price=average_price,
                execution_time_ms=execution_time_ms,
                status=status,
                market_depth=market_depth,
                mid_price=mid_price,
                slippage_bps=sum(slippage_components.values()),
                slippage_components=slippage_components,
                maker_fee_usd=fees["maker"],
                taker_fee_usd=fees["taker"],
                total_fees_usd=fees["total"],
                submitted_at=datetime.now(),
                completed_at=datetime.now() + timedelta(milliseconds=execution_time_ms)
            )

            # Store for calibration
            if len(self.execution_history) >= 10000:  # Limit history size
                self.execution_history = self.execution_history[-5000:]
            self.execution_history.append(result)

            return result

        except Exception as e:
            logger.error(f"Execution simulation failed: {e}")
            # Return failed execution
            return ExecutionResult(
                order_id=f"FAILED_{int(datetime.now().timestamp())}",
                symbol=symbol,
                side=side,
                requested_quantity=quantity,
                filled_quantity=0.0,
                average_price=0.0,
                execution_time_ms=0.0,
                status=ExecutionStatus.REJECTED
            )

    def _get_mid_price(self, symbol: str, market_depth: Optional[MarketDepth]) -> float:
        """Get current mid price"""

        if market_depth and market_depth.bids and market_depth.asks:
            best_bid = market_depth.bids[0][0]
            best_ask = market_depth.asks[0][0]
            return (best_bid + best_ask) / 2

        # Fallback - would typically get from market data
        return 50000.0  # Default BTC price for simulation

    def _get_spread_bps(self, symbol: str, market_depth: Optional[MarketDepth]) -> float:
        """Get current spread in basis points"""

        if market_depth:
            return market_depth.spread_bps

        return self.typical_spreads.get(symbol, self.typical_spreads["default"])

    def _simulate_execution_time(self, quantity: float) -> float:
        """Simulate realistic execution time"""

        # Base latency
        network_latency = np.random.lognormal(
            np.log(self.network_latency_ms["p50"]),
            0.5
        )

        processing_latency = np.random.lognormal(
            np.log(self.exchange_processing_ms["p50"]),
            0.3
        )

        # Quantity impact on processing time
        quantity_factor = min(1.0 + np.log10(abs(quantity) + 1) * 0.1, 2.0)

        total_time = (network_latency + processing_latency) * quantity_factor

        return max(10.0, total_time)  # Minimum 10ms

    def _calculate_slippage_components(self,
                                     symbol: str,
                                     side: str,
                                     quantity: float,
                                     order_type: str,
                                     mid_price: float,
                                     spread_bps: float,
                                     market_conditions: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Calculate detailed slippage components"""

        components = {}
        notional = abs(quantity) * mid_price

        # 1. Spread cost (always present for market orders)
        if order_type == "market":
            components[SlippageComponent.SPREAD_COST.value] = spread_bps / 2

        # 2. Market impact
        impact_coef = self.market_impact_coefficients.get(symbol, self.market_impact_coefficients["default"])
        market_impact = impact_coef * np.sqrt(notional / 1000000)  # Impact per $1M
        components[SlippageComponent.MARKET_IMPACT.value] = market_impact

        # 3. Timing delay (random walk during execution)
        if market_conditions and "volatility" in market_conditions:
            volatility = market_conditions["volatility"]
            execution_seconds = self._simulate_execution_time(quantity) / 1000
            timing_cost = volatility * np.sqrt(execution_seconds / 86400) * 10000  # Daily vol to execution period
            components[SlippageComponent.TIMING_DELAY.value] = max(0, np.random.normal(0, timing_cost))

        # 4. Partial fills (for large orders)
        if notional > 100000:  # $100k threshold
            partial_fill_prob = min(0.3, notional / 1000000 * 0.1)
            if np.random.random() < partial_fill_prob:
                components[SlippageComponent.PARTIAL_FILLS.value] = spread_bps * 0.25

        # 5. Volatility spikes
        if market_conditions and "volatility_spike" in market_conditions:
            spike_factor = market_conditions["volatility_spike"]
            if spike_factor > 2.0:  # 2x normal volatility
                components[SlippageComponent.VOLATILITY_SPIKE.value] = spread_bps * (spike_factor - 1)

        # 6. Liquidity shortage
        if market_conditions and "liquidity_ratio" in market_conditions:
            liquidity_ratio = market_conditions["liquidity_ratio"]
            if liquidity_ratio < 0.5:  # Below normal liquidity
                shortage_cost = (1 - liquidity_ratio) * spread_bps
                components[SlippageComponent.LIQUIDITY_SHORTAGE.value] = shortage_cost

        # 7. Network latency impact
        latency_impact = max(0, (self._simulate_execution_time(quantity) - 50) / 1000 * spread_bps * 0.1)
        components[SlippageComponent.NETWORK_LATENCY.value] = latency_impact

        # 8. Positive slippage (price improvement) possibility
        improvement_prob = 0.15  # 15% chance of price improvement
        if np.random.random() < improvement_prob:
            improvement = -np.random.exponential(spread_bps * 0.3)
            components[SlippageComponent.PRICE_IMPROVEMENT.value] = improvement

        return components

    def _simulate_fill_execution(self,
                               symbol: str,
                               side: str,
                               quantity: float,
                               order_type: str,
                               limit_price: Optional[float],
                               mid_price: float,
                               slippage_components: Dict[str, float]) -> Tuple[float, float]:
        """Simulate fill rate and execution price"""

        total_slippage_bps = sum(slippage_components.values())

        # Calculate execution price
        slippage_factor = total_slippage_bps / 10000  # Convert bps to decimal

        if side.lower() == "buy":
            execution_price = mid_price * (1 + slippage_factor)
        else:  # sell
            execution_price = mid_price * (1 - slippage_factor)

        # Determine fill rate
        fill_rate = 100.0  # Default 100% fill

        if order_type == "limit" and limit_price:
            # Check if limit price allows execution
            if side.lower() == "buy" and limit_price < execution_price:
                fill_rate = 0.0  # No fill
            elif side.lower() == "sell" and limit_price > execution_price:
                fill_rate = 0.0  # No fill
            else:
                # Partial fill possibility for limit orders
                if abs(quantity) * mid_price > 50000:  # Large orders
                    fill_rate = np.random.uniform(70, 100)

        # Market orders can have partial fills in extreme conditions
        elif order_type == "market":
            if total_slippage_bps > 50:  # High slippage scenario
                fill_rate = max(50, np.random.normal(90, 10))

        return min(100.0, max(0.0, fill_rate)), execution_price

    def _calculate_fees(self, symbol: str, quantity: float, price: float, order_type: str) -> Dict[str, float]:
        """Calculate trading fees"""

        notional = abs(quantity) * price

        # Determine exchange (simplified)
        exchange = "kraken"  # Default
        fee_struct = self.fee_structures.get(exchange, self.fee_structures["default"])

        # Assume market orders are taker, limit orders are maker
        if order_type == "market":
            fee_rate = fee_struct["taker"]
            taker_fee = notional * fee_rate
            maker_fee = 0.0
        else:
            fee_rate = fee_struct["maker"]
            maker_fee = notional * fee_rate
            taker_fee = 0.0

        return {
            "maker": maker_fee,
            "taker": taker_fee,
            "total": maker_fee + taker_fee
        }

    def _determine_execution_status(self, fill_rate: float, order_type: str) -> ExecutionStatus:
        """Determine final execution status"""

        if fill_rate >= 99.9:
            return ExecutionStatus.FILLED
        elif fill_rate > 0:
            return ExecutionStatus.PARTIAL
        else:
            return ExecutionStatus.REJECTED


class BacktestLiveParityTracker:
    """
    Track and analyze parity between backtest and live execution
    """

    def __init__(self, max_history: int = 10000):
        self.max_history = max_history

        # Execution tracking
        self.backtest_executions: deque = deque(maxlen=max_history)
        self.live_executions: deque = deque(maxlen=max_history)

        # Parity metrics
        self.parity_reports: List[Dict[str, Any]] = []

        # Thresholds for parity alerts
        self.tracking_error_threshold_bps = 20  # 20 bps per day
        self.execution_time_threshold_pct = 50   # 50% difference
        self.slippage_difference_threshold_bps = 10  # 10 bps difference

    def add_backtest_execution(self, execution: ExecutionResult):
        """Add backtest execution result"""
        # Add execution_type attribute dynamically
        setattr(execution, 'execution_type', 'backtest')
        self.backtest_executions.append(execution)

    def add_live_execution(self, execution: ExecutionResult):
        """Add live execution result"""
        # Add execution_type attribute dynamically
        setattr(execution, 'execution_type', 'live')
        self.live_executions.append(execution)

    def generate_parity_report(self, lookback_days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive parity analysis report"""

        cutoff_time = datetime.now() - timedelta(days=lookback_days)

        # Filter recent executions
        recent_backtest = [e for e in self.backtest_executions if e.completed_at and e.completed_at >= cutoff_time]
        recent_live = [e for e in self.live_executions if e.completed_at and e.completed_at >= cutoff_time]

        if not recent_backtest or not recent_live:
            return {
                "status": "insufficient_data",
                "message": "Not enough execution data for meaningful comparison"
            }

        # Calculate comparative metrics
        parity_metrics = self._calculate_parity_metrics(recent_backtest, recent_live)

        # Analyze slippage differences
        slippage_analysis = self._analyze_slippage_differences(recent_backtest, recent_live)

        # Execution time analysis
        timing_analysis = self._analyze_execution_timing(recent_backtest, recent_live)

        # Fill rate analysis
        fill_analysis = self._analyze_fill_rates(recent_backtest, recent_live)

        # Calculate overall parity score
        parity_score = self._calculate_parity_score(parity_metrics, slippage_analysis, timing_analysis, fill_analysis)

        report = {
            "report_timestamp": datetime.now().isoformat(),
            "lookback_days": lookback_days,
            "data_summary": {
                "backtest_executions": len(recent_backtest),
                "live_executions": len(recent_live),
                "total_backtest_volume": sum(abs(e.filled_quantity) * e.average_price for e in recent_backtest),
                "total_live_volume": sum(abs(e.filled_quantity) * e.average_price for e in recent_live)
            },
            "parity_score": parity_score,
            "parity_metrics": parity_metrics,
            "slippage_analysis": slippage_analysis,
            "timing_analysis": timing_analysis,
            "fill_analysis": fill_analysis,
            "recommendations": self._generate_parity_recommendations(parity_score, parity_metrics),
            "alerts": self._check_parity_alerts(parity_metrics, slippage_analysis)
        }

        self.parity_reports.append(report)
        return report

    def _calculate_parity_metrics(self, backtest_execs: List[ExecutionResult],
                                live_execs: List[ExecutionResult]) -> Dict[str, Any]:
        """Calculate core parity metrics"""

        # Average slippage comparison
        bt_slippage = [e.slippage_bps for e in backtest_execs]
        live_slippage = [e.slippage_bps for e in live_execs]

        bt_avg_slippage = statistics.mean(bt_slippage) if bt_slippage else 0
        live_avg_slippage = statistics.mean(live_slippage) if live_slippage else 0

        slippage_difference = live_avg_slippage - bt_avg_slippage

        # Fee comparison
        bt_fees = [e.total_fees_usd for e in backtest_execs]
        live_fees = [e.total_fees_usd for e in live_execs]

        bt_avg_fees = statistics.mean(bt_fees) if bt_fees else 0
        live_avg_fees = statistics.mean(live_fees) if live_fees else 0

        # Execution success rates
        bt_success_rate = sum(1 for e in backtest_execs if e.status == ExecutionStatus.FILLED) / len(backtest_execs) * 100
        live_success_rate = sum(1 for e in live_execs if e.status == ExecutionStatus.FILLED) / len(live_execs) * 100

        # Tracking error calculation
        bt_returns = self._calculate_execution_returns(backtest_execs)
        live_returns = self._calculate_execution_returns(live_execs)
        tracking_error = self._calculate_tracking_error(bt_returns, live_returns)

        return {
            "avg_slippage_bps": {
                "backtest": bt_avg_slippage,
                "live": live_avg_slippage,
                "difference": slippage_difference
            },
            "avg_fees_usd": {
                "backtest": bt_avg_fees,
                "live": live_avg_fees,
                "difference": live_avg_fees - bt_avg_fees
            },
            "success_rates_pct": {
                "backtest": bt_success_rate,
                "live": live_success_rate,
                "difference": live_success_rate - bt_success_rate
            },
            "tracking_error_bps_daily": tracking_error
        }

    def _analyze_slippage_differences(self, backtest_execs: List[ExecutionResult],
                                    live_execs: List[ExecutionResult]) -> Dict[str, Any]:
        """Analyze slippage component differences"""

        # Group by slippage components
        bt_components = defaultdict(list)
        live_components = defaultdict(list)

        for exec in backtest_execs:
            for component, value in exec.slippage_components.items():
                bt_components[component].append(value)

        for exec in live_execs:
            for component, value in exec.slippage_components.items():
                live_components[component].append(value)

        component_analysis = {}

        for component in set(bt_components.keys()) | set(live_components.keys()):
            bt_values = bt_components.get(component, [0])
            live_values = live_components.get(component, [0])

            bt_avg = statistics.mean(bt_values)
            live_avg = statistics.mean(live_values)

            component_analysis[component] = {
                "backtest_avg": bt_avg,
                "live_avg": live_avg,
                "difference": live_avg - bt_avg,
                "samples": {"backtest": len(bt_values), "live": len(live_values)}
            }

        return {
            "component_breakdown": component_analysis,
            "largest_differences": sorted(
                [(comp, data["difference"]) for comp, data in component_analysis.items()],
                key=lambda x: abs(x[1]), reverse=True
            )[:5]
        }

    def _analyze_execution_timing(self, backtest_execs: List[ExecutionResult],
                                live_execs: List[ExecutionResult]) -> Dict[str, Any]:
        """Analyze execution timing differences"""

        bt_times = [e.execution_time_ms for e in backtest_execs]
        live_times = [e.execution_time_ms for e in live_execs]

        bt_avg_time = statistics.mean(bt_times) if bt_times else 0
        live_avg_time = statistics.mean(live_times) if live_times else 0

        time_difference_pct = ((live_avg_time - bt_avg_time) / bt_avg_time * 100) if bt_avg_time > 0 else 0

        return {
            "avg_execution_time_ms": {
                "backtest": bt_avg_time,
                "live": live_avg_time,
                "difference_ms": live_avg_time - bt_avg_time,
                "difference_pct": time_difference_pct
            },
            "timing_distribution": {
                "backtest_p95": np.percentile(bt_times, 95) if bt_times else 0,
                "live_p95": np.percentile(live_times, 95) if live_times else 0
            }
        }

    def _analyze_fill_rates(self, backtest_execs: List[ExecutionResult],
                          live_execs: List[ExecutionResult]) -> Dict[str, Any]:
        """Analyze fill rate differences"""

        bt_fill_rates = [e.fill_rate for e in backtest_execs]
        live_fill_rates = [e.fill_rate for e in live_execs]

        bt_avg_fill = statistics.mean(bt_fill_rates) if bt_fill_rates else 0
        live_avg_fill = statistics.mean(live_fill_rates) if live_fill_rates else 0

        return {
            "avg_fill_rates_pct": {
                "backtest": bt_avg_fill,
                "live": live_avg_fill,
                "difference": live_avg_fill - bt_avg_fill
            },
            "partial_fill_rates": {
                "backtest": sum(1 for e in backtest_execs if e.status == ExecutionStatus.PARTIAL) / len(backtest_execs) * 100,
                "live": sum(1 for e in live_execs if e.status == ExecutionStatus.PARTIAL) / len(live_execs) * 100
            }
        }

    def _calculate_execution_returns(self, executions: List[ExecutionResult]) -> List[float]:
        """Calculate returns from executions for tracking error"""

        returns = []
        for exec in executions:
            if exec.mid_price and exec.average_price:
                # Calculate return as execution price vs mid price
                if exec.side.lower() == "buy":
                    return_pct = (exec.mid_price - exec.average_price) / exec.average_price
                else:  # sell
                    return_pct = (exec.average_price - exec.mid_price) / exec.mid_price

                returns.append(return_pct * 10000)  # Convert to bps

        return returns

    def _calculate_tracking_error(self, bt_returns: List[float], live_returns: List[float]) -> float:
        """Calculate tracking error between backtest and live returns"""

        if not bt_returns or not live_returns:
            return 0.0

        # Align returns by taking minimum length
        min_len = min(len(bt_returns), len(live_returns))
        bt_aligned = bt_returns[:min_len]
        live_aligned = live_returns[:min_len]

        # Calculate return differences
        differences = [live - bt for live, bt in zip(live_aligned, bt_aligned)]

        if not differences:
            return 0.0

        # Tracking error is standard deviation of return differences
        tracking_error = statistics.stdev(differences) if len(differences) > 1 else 0.0

        return tracking_error

    def _calculate_parity_score(self, parity_metrics: Dict, slippage_analysis: Dict,
                              timing_analysis: Dict, fill_analysis: Dict) -> Dict[str, Any]:
        """Calculate overall parity score"""

        scores = []

        # Slippage parity score (0-100)
        slippage_diff = abs(parity_metrics["avg_slippage_bps"]["difference"])
        slippage_score = max(0, 100 - (slippage_diff * 10))  # -10 points per bps difference
        scores.append(("slippage", slippage_score))

        # Timing parity score
        timing_diff_pct = abs(timing_analysis["avg_execution_time_ms"]["difference_pct"])
        timing_score = max(0, 100 - (timing_diff_pct / 10))  # -10 points per 10% difference
        scores.append(("timing", timing_score))

        # Fill rate parity score
        fill_diff = abs(fill_analysis["avg_fill_rates_pct"]["difference"])
        fill_score = max(0, 100 - (fill_diff * 2))  # -2 points per % difference
        scores.append(("fill_rate", fill_score))

        # Tracking error score
        tracking_error = parity_metrics["tracking_error_bps_daily"]
        tracking_score = max(0, 100 - (tracking_error * 2))  # -2 points per bps tracking error
        scores.append(("tracking_error", tracking_score))

        # Calculate weighted average
        weights = {"slippage": 0.4, "timing": 0.2, "fill_rate": 0.2, "tracking_error": 0.2}
        weighted_score = sum(weights[name] * score for name, score in scores)

        return {
            "overall_score": weighted_score,
            "component_scores": dict(scores),
            "grade": self._score_to_grade(weighted_score)
        }

    def _score_to_grade(self, score: float) -> str:
        """Convert numerical score to grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"

    def _generate_parity_recommendations(self, parity_score: Dict, parity_metrics: Dict) -> List[str]:
        """Generate recommendations for improving parity"""

        recommendations = []

        if parity_score["overall_score"] < 80:
            recommendations.append("Overall parity score is below acceptable threshold - investigate major differences")

        slippage_diff = abs(parity_metrics["avg_slippage_bps"]["difference"])
        if slippage_diff > self.slippage_difference_threshold_bps:
            recommendations.append(f"High slippage difference ({slippage_diff:.1f} bps) - recalibrate market impact model")

        tracking_error = parity_metrics["tracking_error_bps_daily"]
        if tracking_error > self.tracking_error_threshold_bps:
            recommendations.append(f"High tracking error ({tracking_error:.1f} bps) - review execution timing model")

        if parity_score["component_scores"]["fill_rate"] < 90:
            recommendations.append("Low fill rate parity - adjust partial fill modeling")

        if not recommendations:
            recommendations.append("Parity metrics are within acceptable ranges - maintain current calibration")

        return recommendations

    def _check_parity_alerts(self, parity_metrics: Dict, slippage_analysis: Dict) -> List[Dict[str, Any]]:
        """Check for parity alert conditions"""

        alerts = []

        # Tracking error alert
        tracking_error = parity_metrics["tracking_error_bps_daily"]
        if tracking_error > self.tracking_error_threshold_bps:
            alerts.append({
                "type": "tracking_error_high",
                "severity": "warning",
                "message": f"Daily tracking error ({tracking_error:.1f} bps) exceeds threshold ({self.tracking_error_threshold_bps} bps)",
                "value": tracking_error,
                "threshold": self.tracking_error_threshold_bps
            })

        # Slippage difference alert
        slippage_diff = abs(parity_metrics["avg_slippage_bps"]["difference"])
        if slippage_diff > self.slippage_difference_threshold_bps:
            alerts.append({
                "type": "slippage_difference_high",
                "severity": "warning",
                "message": f"Slippage difference ({slippage_diff:.1f} bps) exceeds threshold ({self.slippage_difference_threshold_bps} bps)",
                "value": slippage_diff,
                "threshold": self.slippage_difference_threshold_bps
            })

        return alerts

    def get_parity_summary(self) -> Dict[str, Any]:
        """Get summary of parity tracking"""

        if not self.parity_reports:
            return {"status": "no_reports"}

        latest_report = self.parity_reports[-1]

        return {
            "latest_parity_score": latest_report["parity_score"]["overall_score"],
            "latest_grade": latest_report["parity_score"]["grade"],
            "total_reports": len(self.parity_reports),
            "backtest_executions": len(self.backtest_executions),
            "live_executions": len(self.live_executions),
            "recent_alerts": latest_report.get("alerts", []),
            "improvement_trend": self._calculate_improvement_trend()
        }

    def _calculate_improvement_trend(self) -> str:
        """Calculate trend in parity scores"""

        if len(self.parity_reports) < 2:
            return "insufficient_data"

        recent_scores = [r["parity_score"]["overall_score"] for r in self.parity_reports[-5:]]

        if len(recent_scores) < 2:
            return "stable"

        # Simple trend calculation
        score_diff = recent_scores[-1] - recent_scores[0]

        if score_diff > 5:
            return "improving"
        elif score_diff < -5:
            return "declining"
        else:
            return "stable"
