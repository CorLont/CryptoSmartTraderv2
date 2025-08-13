"""Backtest-Live parity system for execution simulation and tracking error analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from pathlib import Path
import asyncio
import threading

from ..core.structured_logger import get_logger


@dataclass
class ExecutionSlippage:
    """Detailed slippage analysis."""
    symbol: str
    order_size: float
    market_impact: float
    bid_ask_spread: float
    timing_slippage: float
    total_slippage: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradeExecution:
    """Complete trade execution record."""
    trade_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    intended_price: float
    executed_price: float
    slippage_bps: float
    fees_bps: float
    latency_ms: int
    market_conditions: Dict[str, float]
    execution_type: str  # 'backtest' or 'live'
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceAttribution:
    """Performance attribution analysis."""
    period_start: datetime
    period_end: datetime
    total_return: float
    alpha_return: float
    fees_impact: float
    slippage_impact: float
    timing_impact: float
    sizing_impact: float
    market_impact: float
    attribution_confidence: float


@dataclass
class ParityMetrics:
    """Backtest-live parity tracking metrics."""
    tracking_error_bps: float
    correlation: float
    alpha_difference: float
    slippage_difference: float
    fee_difference: float
    execution_quality_score: float
    sample_size: int
    confidence_interval: Tuple[float, float]
    timestamp: datetime = field(default_factory=datetime.now)


class BacktestParityAnalyzer:
    """Advanced backtest-live parity analysis and execution simulation."""

    def __init__(self, target_tracking_error_bps: float = 20.0):
        """Initialize parity analyzer."""
        self.logger = get_logger("backtest_parity")
        self.target_tracking_error_bps = target_tracking_error_bps

        # Execution tracking
        self.backtest_executions: List[TradeExecution] = []
        self.live_executions: List[TradeExecution] = []
        self.slippage_history: List[ExecutionSlippage] = []

        # Performance tracking
        self.backtest_returns: pd.Series = pd.Series(dtype=float)
        self.live_returns: pd.Series = pd.Series(dtype=float)
        self.attribution_history: List[PerformanceAttribution] = []

        # Market microstructure simulation
        self.latency_model = {
            'mean_latency_ms': 50,
            'latency_std_ms': 20,
            'network_delay_ms': 10,
            'exchange_processing_ms': 15
        }

        self.slippage_model = {
            'base_spread_bps': 8,  # Base bid-ask spread
            'impact_coefficient': 0.5,  # Price impact per $1M trade
            'volatility_multiplier': 1.5,  # Volatility impact on slippage
            'liquidity_threshold': 10000  # Minimum liquidity for normal execution
        }

        # Fee structures
        self.fee_schedule = {
            'maker_fee_bps': 10,  # 0.1% maker fee
            'taker_fee_bps': 15,  # 0.15% taker fee
            'funding_rate_bps': 5,  # Average funding rate
            'gas_fee_fixed': 2.0   # Fixed gas fee in USD
        }

        # Thread safety
        self._lock = threading.RLock()

        # Persistence
        self.data_path = Path("data/backtest_parity")
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("Backtest parity analyzer initialized",
                        target_tracking_error_bps=target_tracking_error_bps)

    def simulate_execution(self, symbol: str, quantity: float, side: str,
                          market_conditions: Dict[str, float],
                          execution_type: str = 'backtest') -> TradeExecution:
        """Simulate realistic order execution with market microstructure."""

        # Extract market conditions
        bid_price = market_conditions.get('bid', 0.0)
        ask_price = market_conditions.get('ask', 0.0)
        mid_price = (bid_price + ask_price) / 2 if bid_price > 0 and ask_price > 0 else market_conditions.get('price', 0.0)
        volume_24h = market_conditions.get('volume_24h', 1000000)
        volatility = market_conditions.get('volatility', 0.02)
        orderbook_depth = market_conditions.get('orderbook_depth', 50000)

        # Calculate intended price
        if side == 'buy':
            intended_price = ask_price if ask_price > 0 else mid_price
        else:
            intended_price = bid_price if bid_price > 0 else mid_price

        # Simulate latency (more realistic for live trading)
        if execution_type == 'live':
            latency_ms = max(5, np.random.normal(
                self.latency_model['mean_latency_ms'],
                self.latency_model['latency_std_ms']
            ))
        else:
            latency_ms = 0  # Perfect execution in backtest

        # Calculate market impact
        trade_value = quantity * mid_price
        impact_factor = min(0.01, trade_value / (orderbook_depth * 1000))  # Liquidity impact
        market_impact = impact_factor * self.slippage_model['impact_coefficient']

        # Volatility impact during execution delay
        volatility_impact = volatility * np.sqrt(latency_ms / (1000 * 60 * 60 * 24 * 365))  # Annualized vol

        # Bid-ask spread cost
        spread = ask_price - bid_price if ask_price > bid_price else mid_price * 0.001
        spread_cost = spread / 2 / mid_price  # Half spread cost

        # Total slippage calculation
        if execution_type == 'live':
            # Live execution: more realistic slippage
            direction_multiplier = 1 if side == 'buy' else -1
            price_movement = (market_impact + volatility_impact) * direction_multiplier
            total_slippage = spread_cost + abs(price_movement)
        else:
            # Backtest execution: idealized slippage
            total_slippage = spread_cost * 0.5  # Assume better execution

        # Calculate executed price
        if side == 'buy':
            executed_price = intended_price * (1 + total_slippage)
        else:
            executed_price = intended_price * (1 - total_slippage)

        # Calculate fees
        fee_bps = self.fee_schedule['taker_fee_bps']  # Assume taker orders
        if trade_value < 1000:  # Small trade
            fee_bps += 5  # Additional 0.05% for small trades

        slippage_bps = abs(executed_price - intended_price) / intended_price * 10000

        # Create execution record
        execution = TradeExecution(
            trade_id=f"{execution_type}_{int(datetime.now().timestamp())}_{symbol}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            intended_price=intended_price,
            executed_price=executed_price,
            slippage_bps=slippage_bps,
            fees_bps=fee_bps,
            latency_ms=int(latency_ms),
            market_conditions=market_conditions,
            execution_type=execution_type
        )

        return execution

    def record_execution(self, execution: TradeExecution) -> None:
        """Record a trade execution for parity analysis."""
        with self._lock:
            if execution.execution_type == 'backtest':
                self.backtest_executions.append(execution)
            else:
                self.live_executions.append(execution)

            # Calculate detailed slippage breakdown
            slippage_breakdown = ExecutionSlippage(
                symbol=execution.symbol,
                order_size=execution.quantity * execution.executed_price,
                market_impact=execution.slippage_bps * 0.4,  # Estimate 40% market impact
                bid_ask_spread=execution.slippage_bps * 0.3,  # Estimate 30% spread
                timing_slippage=execution.slippage_bps * 0.3,  # Estimate 30% timing
                total_slippage=execution.slippage_bps
            )

            self.slippage_history.append(slippage_breakdown)

            self.logger.debug(f"Recorded {execution.execution_type} execution",
                            symbol=execution.symbol,
                            slippage_bps=execution.slippage_bps,
                            fees_bps=execution.fees_bps)

    def calculate_parity_metrics(self, lookback_hours: int = 24) -> Optional[ParityMetrics]:
        """Calculate backtest-live parity metrics."""
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)

        # Filter recent executions
        recent_backtest = [e for e in self.backtest_executions if e.timestamp >= cutoff_time]
        recent_live = [e for e in self.live_executions if e.timestamp >= cutoff_time]

        if len(recent_backtest) < 10 or len(recent_live) < 10:
            self.logger.warning("Insufficient executions for parity analysis")
            return None

        try:
            # Convert to DataFrames for analysis
            bt_df = pd.DataFrame([
                {
                    'timestamp': e.timestamp,
                    'symbol': e.symbol,
                    'slippage_bps': e.slippage_bps,
                    'fees_bps': e.fees_bps,
                    'return': (e.executed_price - e.intended_price) / e.intended_price * 10000
                }
                for e in recent_backtest
            ])

            live_df = pd.DataFrame([
                {
                    'timestamp': e.timestamp,
                    'symbol': e.symbol,
                    'slippage_bps': e.slippage_bps,
                    'fees_bps': e.fees_bps,
                    'return': (e.executed_price - e.intended_price) / e.intended_price * 10000
                }
                for e in recent_live
            ])

            # Align executions by symbol and time (simplified)
            aligned_bt = bt_df.groupby('symbol')['return'].mean()
            aligned_live = live_df.groupby('symbol')['return'].mean()

            # Find common symbols
            common_symbols = set(aligned_bt.index) & set(aligned_live.index)
            if len(common_symbols) < 3:
                return None

            bt_returns = aligned_bt[list(common_symbols)]
            live_returns = aligned_live[list(common_symbols)]

            # Calculate tracking error
            return_diff = live_returns - bt_returns
            tracking_error_bps = return_diff.std()

            # Calculate correlation
            correlation = bt_returns.corr(live_returns)

            # Alpha difference (systematic bias)
            alpha_difference = return_diff.mean()

            # Slippage difference
            bt_slippage = bt_df['slippage_bps'].mean()
            live_slippage = live_df['slippage_bps'].mean()
            slippage_difference = live_slippage - bt_slippage

            # Fee difference
            bt_fees = bt_df['fees_bps'].mean()
            live_fees = live_df['fees_bps'].mean()
            fee_difference = live_fees - bt_fees

            # Execution quality score (0-100)
            quality_score = max(0, 100 - (tracking_error_bps / self.target_tracking_error_bps * 50))

            # Confidence interval for tracking error
            n_obs = len(return_diff)
            std_error = tracking_error_bps / np.sqrt(n_obs)
            confidence_interval = (
                tracking_error_bps - 1.96 * std_error,
                tracking_error_bps + 1.96 * std_error
            )

            return ParityMetrics(
                tracking_error_bps=tracking_error_bps,
                correlation=correlation if not np.isnan(correlation) else 0.0,
                alpha_difference=alpha_difference,
                slippage_difference=slippage_difference,
                fee_difference=fee_difference,
                execution_quality_score=quality_score,
                sample_size=len(common_symbols),
                confidence_interval=confidence_interval
            )

        except Exception as e:
            self.logger.error(f"Error calculating parity metrics: {e}")
            return None

    def analyze_return_attribution(self, portfolio_returns: pd.Series,
                                 benchmark_returns: pd.Series,
                                 period_days: int = 7) -> PerformanceAttribution:
        """Analyze return attribution with component breakdown."""

        if len(portfolio_returns) < period_days or len(benchmark_returns) < period_days:
            # Return default attribution for insufficient data
            return PerformanceAttribution(
                period_start=datetime.now() - timedelta(days=period_days),
                period_end=datetime.now(),
                total_return=0.0,
                alpha_return=0.0,
                fees_impact=0.0,
                slippage_impact=0.0,
                timing_impact=0.0,
                sizing_impact=0.0,
                market_impact=0.0,
                attribution_confidence=0.0
            )

        # Calculate period returns
        period_start = datetime.now() - timedelta(days=period_days)
        period_end = datetime.now()

        portfolio_period_return = portfolio_returns.tail(period_days).sum()
        benchmark_period_return = benchmark_returns.tail(period_days).sum()

        # Calculate alpha (excess return)
        alpha_return = portfolio_period_return - benchmark_period_return

        # Estimate component impacts
        recent_executions = [
            e for e in (self.backtest_executions + self.live_executions)
            if e.timestamp >= period_start
        ]

        if recent_executions:
            # Fees impact
            avg_fees_bps = np.mean([e.fees_bps for e in recent_executions])
            fees_impact = -avg_fees_bps / 10000  # Convert to decimal return

            # Slippage impact
            avg_slippage_bps = np.mean([e.slippage_bps for e in recent_executions])
            slippage_impact = -avg_slippage_bps / 10000

            # Timing impact (estimated from execution latency)
            avg_latency = np.mean([e.latency_ms for e in recent_executions])
            timing_impact = -min(0.001, avg_latency / 100000)  # Rough estimate

        else:
            fees_impact = -0.0015  # Default 15 bps
            slippage_impact = -0.0008  # Default 8 bps
            timing_impact = -0.0002  # Default 2 bps

        # Market impact (estimated)
        market_impact = benchmark_period_return

        # Sizing impact (residual)
        explained_return = alpha_return + fees_impact + slippage_impact + timing_impact
        sizing_impact = portfolio_period_return - explained_return - market_impact

        # Attribution confidence (based on sample size and correlation)
        attribution_confidence = min(1.0, len(recent_executions) / 50)

        return PerformanceAttribution(
            period_start=period_start,
            period_end=period_end,
            total_return=portfolio_period_return,
            alpha_return=alpha_return,
            fees_impact=fees_impact,
            slippage_impact=slippage_impact,
            timing_impact=timing_impact,
            sizing_impact=sizing_impact,
            market_impact=market_impact,
            attribution_confidence=attribution_confidence
        )

    def get_slippage_analysis(self, symbol: Optional[str] = None,
                            hours: int = 24) -> Dict[str, Any]:
        """Get detailed slippage analysis."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        filtered_slippage = [
            s for s in self.slippage_history
            if s.timestamp >= cutoff_time and (symbol is None or s.symbol == symbol)
        ]

        if not filtered_slippage:
            return {'error': 'No slippage data available'}

        # Calculate statistics
        total_slippages = [s.total_slippage for s in filtered_slippage]
        market_impacts = [s.market_impact for s in filtered_slippage]
        spread_costs = [s.bid_ask_spread for s in filtered_slippage]
        timing_costs = [s.timing_slippage for s in filtered_slippage]

        return {
            'period_hours': hours,
            'sample_size': len(filtered_slippage),
            'total_slippage': {
                'mean_bps': np.mean(total_slippages),
                'median_bps': np.median(total_slippages),
                'p95_bps': np.percentile(total_slippages, 95),
                'std_bps': np.std(total_slippages)
            },
            'component_breakdown': {
                'market_impact_bps': np.mean(market_impacts),
                'spread_cost_bps': np.mean(spread_costs),
                'timing_cost_bps': np.mean(timing_costs)
            },
            'within_budget': np.percentile(total_slippages, 95) <= 30.0  # 30 bps budget
        }

    def is_tracking_error_acceptable(self) -> Tuple[bool, Optional[ParityMetrics]]:
        """Check if current tracking error is within acceptable limits."""
        metrics = self.calculate_parity_metrics()

        if metrics is None:
            return False, None

        acceptable = metrics.tracking_error_bps <= self.target_tracking_error_bps

        return acceptable, metrics

    def get_execution_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive execution quality report."""
        parity_metrics = self.calculate_parity_metrics()
        slippage_analysis = self.get_slippage_analysis()

        # Recent performance
        recent_executions = [
            e for e in (self.backtest_executions + self.live_executions)
            if e.timestamp >= datetime.now() - timedelta(hours=24)
        ]

        if recent_executions:
            avg_latency = np.mean([e.latency_ms for e in recent_executions if e.execution_type == 'live'])
            execution_success_rate = len([e for e in recent_executions if e.slippage_bps < 50]) / len(recent_executions)
        else:
            avg_latency = 0
            execution_success_rate = 0

        return {
            'parity_metrics': {
                'tracking_error_bps': parity_metrics.tracking_error_bps if parity_metrics else None,
                'correlation': parity_metrics.correlation if parity_metrics else None,
                'within_target': parity_metrics.tracking_error_bps <= self.target_tracking_error_bps if parity_metrics else False
            },
            'slippage_analysis': slippage_analysis,
            'execution_performance': {
                'avg_latency_ms': avg_latency,
                'success_rate': execution_success_rate,
                'total_executions_24h': len(recent_executions)
            },
            'quality_score': parity_metrics.execution_quality_score if parity_metrics else 0,
            'report_timestamp': datetime.now().isoformat()
        }

    def save_parity_state(self) -> None:
        """Save parity analysis state."""
        # Save recent executions (last 1000)
        recent_bt = self.backtest_executions[-1000:] if len(self.backtest_executions) > 1000 else self.backtest_executions
        recent_live = self.live_executions[-1000:] if len(self.live_executions) > 1000 else self.live_executions

        state = {
            'backtest_executions': len(self.backtest_executions),
            'live_executions': len(self.live_executions),
            'last_parity_check': datetime.now().isoformat(),
            'target_tracking_error_bps': self.target_tracking_error_bps
        }

        try:
            with open(self.data_path / "parity_state.json", 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save parity state: {e}")


def create_backtest_parity_analyzer(target_tracking_error_bps: float = 20.0) -> BacktestParityAnalyzer:
    """Factory function to create BacktestParityAnalyzer instance."""
    return BacktestParityAnalyzer(target_tracking_error_bps=target_tracking_error_bps)
