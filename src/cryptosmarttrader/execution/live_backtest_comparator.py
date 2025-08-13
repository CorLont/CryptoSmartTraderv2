"""
Live-Backtest Comparator

System to compare backtest results with live trading performance
to identify and eliminate performance illusions and biases.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PerformanceGap(Enum):
    """Types of performance gaps between backtest and live"""
    EXECUTION_COSTS = "execution_costs"           # Higher execution costs in live
    SLIPPAGE_UNDERESTIMATE = "slippage_underestimate"  # Slippage worse than expected
    LATENCY_IMPACT = "latency_impact"             # Latency causing performance drag
    PARTIAL_FILL_BIAS = "partial_fill_bias"       # Backtest assumes full fills
    BAR_CLOSE_BIAS = "bar_close_bias"             # Backtest uses bar close prices
    LIQUIDITY_OVERESTIMATE = "liquidity_overestimate"  # Overestimated available liquidity
    MARKET_IMPACT_UNDERESTIMATE = "market_impact_underestimate"  # Market impact worse than expected
    FEE_STRUCTURE_MISMATCH = "fee_structure_mismatch"  # Fee structure differences
    TIMING_PRECISION = "timing_precision"         # Timing precision differences
    SURVIVORSHIP_BIAS = "survivorship_bias"       # Backtest survivorship bias


@dataclass
class ParityCheck:
    """Individual parity check result"""
    check_type: str
    backtest_value: float
    live_value: float
    difference: float
    difference_pct: float
    significance_score: float  # How significant this difference is (0-1)
    explanation: str

    @property
    def is_significant(self) -> bool:
        """Check if difference is statistically significant"""
        return self.significance_score > 0.7


@dataclass
class ParityMetrics:
    """Comprehensive backtest-live parity metrics"""
    analysis_period_days: int
    total_trades_compared: int

    # Return metrics
    backtest_total_return_pct: float
    live_total_return_pct: float
    return_gap_pct: float

    # Risk metrics
    backtest_sharpe_ratio: float
    live_sharpe_ratio: float
    sharpe_gap: float

    backtest_max_drawdown_pct: float
    live_max_drawdown_pct: float
    drawdown_gap_pct: float

    # Execution metrics
    backtest_avg_slippage_bps: float
    live_avg_slippage_bps: float
    slippage_gap_bps: float

    backtest_fill_rate: float
    live_fill_rate: float
    fill_rate_gap: float

    # Individual parity checks
    parity_checks: List[ParityCheck]

    # Overall assessment
    overall_parity_score: float  # 0-1, 1 = perfect parity
    major_gaps_identified: List[PerformanceGap]

    @property
    def has_significant_gaps(self) -> bool:
        """Check if there are significant performance gaps"""
        return len([check for check in self.parity_checks if check.is_significant]) > 0


class LiveBacktestComparator:
    """
    Advanced system for comparing backtest and live performance
    """

    def __init__(self):
        self.backtest_results = []
        self.live_results = []
        self.parity_history = []

        # Significance thresholds
        self.return_significance_threshold = 0.05    # 5% return difference
        self.slippage_significance_threshold = 10    # 10 bps slippage difference
        self.fill_rate_significance_threshold = 0.1  # 10% fill rate difference

        # Statistical parameters
        self.min_trades_for_analysis = 20            # Minimum trades for meaningful comparison
        self.confidence_level = 0.95                 # Statistical confidence level

    def compare_performance(self,
                          backtest_data: Dict[str, Any],
                          live_data: Dict[str, Any],
                          analysis_period_days: int = 30) -> ParityMetrics:
        """Comprehensive backtest vs live performance comparison"""
        try:
            logger.info(f"Comparing backtest vs live performance over {analysis_period_days} days")

            # Extract and align data
            bt_trades = self._extract_backtest_trades(backtest_data, analysis_period_days)
            live_trades = self._extract_live_trades(live_data, analysis_period_days)

            if len(bt_trades) < self.min_trades_for_analysis or len(live_trades) < self.min_trades_for_analysis:
                logger.warning(f"Insufficient trades for analysis: BT={len(bt_trades)}, Live={len(live_trades)}")
                return self._create_insufficient_data_metrics()

            # Perform individual parity checks
            parity_checks = self._perform_parity_checks(bt_trades, live_trades)

            # Calculate aggregate metrics
            bt_metrics = self._calculate_aggregate_metrics(bt_trades)
            live_metrics = self._calculate_aggregate_metrics(live_trades)

            # Identify major performance gaps
            major_gaps = self._identify_major_gaps(parity_checks)

            # Calculate overall parity score
            parity_score = self._calculate_overall_parity_score(parity_checks)

            metrics = ParityMetrics(
                analysis_period_days=analysis_period_days,
                total_trades_compared=min(len(bt_trades), len(live_trades)),
                backtest_total_return_pct=bt_metrics["total_return_pct"],
                live_total_return_pct=live_metrics["total_return_pct"],
                return_gap_pct=live_metrics["total_return_pct"] - bt_metrics["total_return_pct"],
                backtest_sharpe_ratio=bt_metrics["sharpe_ratio"],
                live_sharpe_ratio=live_metrics["sharpe_ratio"],
                sharpe_gap=live_metrics["sharpe_ratio"] - bt_metrics["sharpe_ratio"],
                backtest_max_drawdown_pct=bt_metrics["max_drawdown_pct"],
                live_max_drawdown_pct=live_metrics["max_drawdown_pct"],
                drawdown_gap_pct=live_metrics["max_drawdown_pct"] - bt_metrics["max_drawdown_pct"],
                backtest_avg_slippage_bps=bt_metrics["avg_slippage_bps"],
                live_avg_slippage_bps=live_metrics["avg_slippage_bps"],
                slippage_gap_bps=live_metrics["avg_slippage_bps"] - bt_metrics["avg_slippage_bps"],
                backtest_fill_rate=bt_metrics["fill_rate"],
                live_fill_rate=live_metrics["fill_rate"],
                fill_rate_gap=live_metrics["fill_rate"] - bt_metrics["fill_rate"],
                parity_checks=parity_checks,
                overall_parity_score=parity_score,
                major_gaps_identified=major_gaps
            )

            # Store in history
            self.parity_history.append(metrics)

            # Log significant findings
            if metrics.has_significant_gaps:
                logger.warning(f"Significant performance gaps detected: {[gap.value for gap in major_gaps]}")

            return metrics

        except Exception as e:
            logger.error(f"Performance comparison failed: {e}")
            return self._create_error_metrics()

    def _extract_backtest_trades(self, backtest_data: Dict[str, Any], days_back: int) -> List[Dict[str, Any]]:
        """Extract and normalize backtest trade data"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)

            trades = backtest_data.get("trades", [])

            # Filter to analysis period
            filtered_trades = [
                trade for trade in trades
                if datetime.fromisoformat(trade.get("timestamp", "1970-01-01")) >= cutoff_time
            ]

            # Normalize trade format
            normalized_trades = []
            for trade in filtered_trades:
                normalized = {
                    "timestamp": datetime.fromisoformat(trade.get("timestamp", "1970-01-01")),
                    "pair": trade.get("pair", ""),
                    "side": trade.get("side", ""),
                    "size": trade.get("size", 0),
                    "price": trade.get("price", 0),
                    "expected_price": trade.get("expected_price", trade.get("price", 0)),
                    "slippage_bps": trade.get("slippage_bps", 0),
                    "fees": trade.get("fees", 0),
                    "fill_rate": trade.get("fill_rate", 1.0),  # Backtest assumes full fills
                    "execution_time_ms": trade.get("execution_time_ms", 0),  # Backtest assumes instant
                    "pnl": trade.get("pnl", 0),
                    "source": "backtest"
                }
                normalized_trades.append(normalized)

            return normalized_trades

        except Exception as e:
            logger.error(f"Backtest trade extraction failed: {e}")
            return []

    def _extract_live_trades(self, live_data: Dict[str, Any], days_back: int) -> List[Dict[str, Any]]:
        """Extract and normalize live trade data"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)

            trades = live_data.get("trades", [])

            # Filter to analysis period
            filtered_trades = [
                trade for trade in trades
                if datetime.fromisoformat(trade.get("timestamp", "1970-01-01")) >= cutoff_time
            ]

            # Normalize trade format
            normalized_trades = []
            for trade in filtered_trades:
                normalized = {
                    "timestamp": datetime.fromisoformat(trade.get("timestamp", "1970-01-01")),
                    "pair": trade.get("pair", ""),
                    "side": trade.get("side", ""),
                    "size": trade.get("size", 0),
                    "price": trade.get("avg_fill_price", trade.get("price", 0)),
                    "expected_price": trade.get("expected_price", 0),
                    "slippage_bps": trade.get("realized_slippage_bps", 0),
                    "fees": trade.get("total_fees", 0),
                    "fill_rate": trade.get("fill_rate", 1.0),
                    "execution_time_ms": trade.get("execution_time_seconds", 0) * 1000,
                    "pnl": trade.get("realized_pnl", 0),
                    "source": "live"
                }
                normalized_trades.append(normalized)

            return normalized_trades

        except Exception as e:
            logger.error(f"Live trade extraction failed: {e}")
            return []

    def _perform_parity_checks(self,
                              bt_trades: List[Dict[str, Any]],
                              live_trades: List[Dict[str, Any]]) -> List[ParityCheck]:
        """Perform individual parity checks"""
        checks = []

        try:
            # Return comparison
            bt_returns = [t["pnl"] for t in bt_trades]
            live_returns = [t["pnl"] for t in live_trades]

            bt_total_return = sum(bt_returns) / len(bt_trades) if bt_returns else 0
            live_total_return = sum(live_returns) / len(live_returns) if live_returns else 0

            return_diff = live_total_return - bt_total_return
            return_diff_pct = (return_diff / abs(bt_total_return)) * 100 if bt_total_return != 0 else 0

            checks.append(ParityCheck(
                check_type="returns",
                backtest_value=bt_total_return,
                live_value=live_total_return,
                difference=return_diff,
                difference_pct=return_diff_pct,
                significance_score=min(1.0, abs(return_diff_pct) / (self.return_significance_threshold * 100)),
                explanation=f"Live returns {'underperformed' if return_diff < 0 else 'outperformed'} backtest by {abs(return_diff_pct):.2f}%"
            ))

            # Slippage comparison
            bt_slippage = [abs(t["slippage_bps"]) for t in bt_trades]
            live_slippage = [abs(t["slippage_bps"]) for t in live_trades]

            bt_avg_slippage = np.mean(bt_slippage) if bt_slippage else 0
            live_avg_slippage = np.mean(live_slippage) if live_slippage else 0

            slippage_diff = live_avg_slippage - bt_avg_slippage
            slippage_diff_pct = (slippage_diff / bt_avg_slippage) * 100 if bt_avg_slippage != 0 else 0

            checks.append(ParityCheck(
                check_type="slippage",
                backtest_value=bt_avg_slippage,
                live_value=live_avg_slippage,
                difference=slippage_diff,
                difference_pct=slippage_diff_pct,
                significance_score=min(1.0, abs(slippage_diff) / self.slippage_significance_threshold),
                explanation=f"Live slippage was {slippage_diff:.1f} bps {'higher' if slippage_diff > 0 else 'lower'} than backtest"
            ))

            # Fill rate comparison
            bt_fill_rates = [t["fill_rate"] for t in bt_trades]
            live_fill_rates = [t["fill_rate"] for t in live_trades]

            bt_avg_fill_rate = np.mean(bt_fill_rates) if bt_fill_rates else 1.0
            live_avg_fill_rate = np.mean(live_fill_rates) if live_fill_rates else 1.0

            fill_rate_diff = live_avg_fill_rate - bt_avg_fill_rate
            fill_rate_diff_pct = (fill_rate_diff / bt_avg_fill_rate) * 100 if bt_avg_fill_rate != 0 else 0

            checks.append(ParityCheck(
                check_type="fill_rate",
                backtest_value=bt_avg_fill_rate,
                live_value=live_avg_fill_rate,
                difference=fill_rate_diff,
                difference_pct=fill_rate_diff_pct,
                significance_score=min(1.0, abs(fill_rate_diff) / self.fill_rate_significance_threshold),
                explanation=f"Live fill rate was {abs(fill_rate_diff_pct):.1f}% {'lower' if fill_rate_diff < 0 else 'higher'} than backtest"
            ))

            # Execution time comparison
            bt_exec_times = [t["execution_time_ms"] for t in bt_trades]
            live_exec_times = [t["execution_time_ms"] for t in live_trades]

            bt_avg_exec_time = np.mean(bt_exec_times) if bt_exec_times else 0
            live_avg_exec_time = np.mean(live_exec_times) if live_exec_times else 0

            exec_time_diff = live_avg_exec_time - bt_avg_exec_time
            exec_time_diff_pct = (exec_time_diff / max(bt_avg_exec_time, 1)) * 100

            checks.append(ParityCheck(
                check_type="execution_time",
                backtest_value=bt_avg_exec_time,
                live_value=live_avg_exec_time,
                difference=exec_time_diff,
                difference_pct=exec_time_diff_pct,
                significance_score=min(1.0, exec_time_diff / 1000),  # Significance if >1 second difference
                explanation=f"Live execution was {exec_time_diff:.0f}ms {'slower' if exec_time_diff > 0 else 'faster'} than backtest"
            ))

            # Fee comparison
            bt_fees = [t["fees"] for t in bt_trades if t["fees"] > 0]
            live_fees = [t["fees"] for t in live_trades if t["fees"] > 0]

            if bt_fees and live_fees:
                bt_avg_fee_rate = np.mean([f/(t["size"]*t["price"]) for t, f in zip(bt_trades, bt_fees) if t["size"]*t["price"] > 0])
                live_avg_fee_rate = np.mean([f/(t["size"]*t["price"]) for t, f in zip(live_trades, live_fees) if t["size"]*t["price"] > 0])

                fee_diff = live_avg_fee_rate - bt_avg_fee_rate
                fee_diff_pct = (fee_diff / bt_avg_fee_rate) * 100 if bt_avg_fee_rate != 0 else 0

                checks.append(ParityCheck(
                    check_type="fees",
                    backtest_value=bt_avg_fee_rate * 10000,  # Convert to bps
                    live_value=live_avg_fee_rate * 10000,
                    difference=fee_diff * 10000,
                    difference_pct=fee_diff_pct,
                    significance_score=min(1.0, abs(fee_diff * 10000) / 5),  # Significance if >5 bps difference
                    explanation=f"Live fees were {abs(fee_diff_pct):.1f}% {'higher' if fee_diff > 0 else 'lower'} than backtest"
                ))

            # Volatility impact comparison
            bt_volatility_impact = self._calculate_volatility_impact(bt_trades)
            live_volatility_impact = self._calculate_volatility_impact(live_trades)

            vol_impact_diff = live_volatility_impact - bt_volatility_impact
            vol_impact_diff_pct = (vol_impact_diff / abs(bt_volatility_impact)) * 100 if bt_volatility_impact != 0 else 0

            checks.append(ParityCheck(
                check_type="volatility_impact",
                backtest_value=bt_volatility_impact,
                live_value=live_volatility_impact,
                difference=vol_impact_diff,
                difference_pct=vol_impact_diff_pct,
                significance_score=min(1.0, abs(vol_impact_diff) / 10),  # Significance if >10 bps difference
                explanation=f"Live volatility impact was {abs(vol_impact_diff_pct):.1f}% {'higher' if vol_impact_diff > 0 else 'lower'} than backtest"
            ))

            return checks

        except Exception as e:
            logger.error(f"Parity checks failed: {e}")
            return []

    def _calculate_aggregate_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate aggregate performance metrics"""
        try:
            if not trades:
                return self._get_empty_metrics()

            # Extract metrics
            returns = [t["pnl"] for t in trades]
            slippages = [abs(t["slippage_bps"]) for t in trades]
            fill_rates = [t["fill_rate"] for t in trades]

            # Return metrics
            total_return = sum(returns)
            avg_return = np.mean(returns)
            return_volatility = np.std(returns) if len(returns) > 1 else 0

            # Calculate Sharpe ratio (simplified)
            sharpe_ratio = avg_return / return_volatility if return_volatility > 0 else 0

            # Calculate max drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0
            max_drawdown_pct = (max_drawdown / max(running_max)) * 100 if max(running_max) > 0 else 0

            return {
                "total_return": total_return,
                "total_return_pct": (total_return / len(trades)) * 100,  # Simplified
                "avg_return": avg_return,
                "return_volatility": return_volatility,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "max_drawdown_pct": max_drawdown_pct,
                "avg_slippage_bps": np.mean(slippages),
                "fill_rate": np.mean(fill_rates),
                "trade_count": len(trades)
            }

        except Exception as e:
            logger.error(f"Aggregate metrics calculation failed: {e}")
            return self._get_empty_metrics()

    def _calculate_volatility_impact(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate volatility impact on execution"""
        try:
            if not trades:
                return 0.0

            # Simplified volatility impact calculation
            # In practice, this would analyze price movements during execution
            execution_times = [t["execution_time_ms"] for t in trades]
            slippages = [abs(t["slippage_bps"]) for t in trades]

            if not execution_times or not slippages:
                return 0.0

            # Correlation between execution time and slippage
            # Higher correlation suggests volatility impact
            correlation = np.corrcoef(execution_times, slippages)[0, 1] if len(execution_times) > 1 else 0

            # Estimate volatility impact as proportion of slippage correlated with timing
            avg_slippage = np.mean(slippages)
            volatility_impact = avg_slippage * abs(correlation) * 0.5  # Simplified attribution

            return volatility_impact

        except Exception as e:
            logger.error(f"Volatility impact calculation failed: {e}")
            return 0.0

    def _identify_major_gaps(self, parity_checks: List[ParityCheck]) -> List[PerformanceGap]:
        """Identify major performance gaps from parity checks"""
        gaps = []

        try:
            for check in parity_checks:
                if not check.is_significant:
                    continue

                if check.check_type == "slippage" and check.difference > 0:
                    if check.difference > 20:  # >20 bps higher slippage
                        gaps.append(PerformanceGap.SLIPPAGE_UNDERESTIMATE)
                    else:
                        gaps.append(PerformanceGap.EXECUTION_COSTS)

                elif check.check_type == "fill_rate" and check.difference < -0.1:  # 10% lower fill rate
                    gaps.append(PerformanceGap.PARTIAL_FILL_BIAS)

                elif check.check_type == "execution_time" and check.difference > 500:  # >500ms slower
                    gaps.append(PerformanceGap.LATENCY_IMPACT)

                elif check.check_type == "fees" and check.difference > 5:  # >5 bps higher fees
                    gaps.append(PerformanceGap.FEE_STRUCTURE_MISMATCH)

                elif check.check_type == "volatility_impact" and check.difference > 15:  # >15 bps higher impact
                    gaps.append(PerformanceGap.MARKET_IMPACT_UNDERESTIMATE)

                elif check.check_type == "returns" and check.difference < -0.05:  # 5% underperformance
                    # Could be multiple causes - identify most likely
                    if any(c.check_type == "slippage" and c.is_significant for c in parity_checks):
                        gaps.append(PerformanceGap.EXECUTION_COSTS)
                    else:
                        gaps.append(PerformanceGap.BAR_CLOSE_BIAS)

            # Remove duplicates
            return list(set(gaps))

        except Exception as e:
            logger.error(f"Major gap identification failed: {e}")
            return []

    def _calculate_overall_parity_score(self, parity_checks: List[ParityCheck]) -> float:
        """Calculate overall parity score (0-1, 1 = perfect parity)"""
        try:
            if not parity_checks:
                return 0.0

            # Weight different check types
            weights = {
                "returns": 0.3,
                "slippage": 0.25,
                "fill_rate": 0.2,
                "execution_time": 0.1,
                "fees": 0.1,
                "volatility_impact": 0.05
            }

            weighted_score = 0.0
            total_weight = 0.0

            for check in parity_checks:
                weight = weights.get(check.check_type, 0.05)
                # Convert significance score to parity score (inverse)
                check_parity_score = 1.0 - min(1.0, check.significance_score)

                weighted_score += check_parity_score * weight
                total_weight += weight

            overall_score = weighted_score / total_weight if total_weight > 0 else 0.0

            return overall_score

        except Exception as e:
            logger.error(f"Overall parity score calculation failed: {e}")
            return 0.0

    def get_parity_recommendations(self, parity_metrics: ParityMetrics) -> List[str]:
        """Generate recommendations to improve backtest-live parity"""
        recommendations = []

        try:
            if PerformanceGap.SLIPPAGE_UNDERESTIMATE in parity_metrics.major_gaps_identified:
                recommendations.append(
                    "Implement more realistic slippage modeling in backtest using historical execution data"
                )

            if PerformanceGap.PARTIAL_FILL_BIAS in parity_metrics.major_gaps_identified:
                recommendations.append(
                    "Add partial fill simulation to backtest based on historical fill rates by pair and order size"
                )

            if PerformanceGap.LATENCY_IMPACT in parity_metrics.major_gaps_identified:
                recommendations.append(
                    "Include latency delays in backtest execution simulation with realistic timing models"
                )

            if PerformanceGap.EXECUTION_COSTS in parity_metrics.major_gaps_identified:
                recommendations.append(
                    "Calibrate backtest execution costs using live execution analytics and fee structures"
                )

            if PerformanceGap.BAR_CLOSE_BIAS in parity_metrics.major_gaps_identified:
                recommendations.append(
                    "Avoid using bar close prices in backtest; implement intra-bar execution simulation"
                )

            if PerformanceGap.MARKET_IMPACT_UNDERESTIMATE in parity_metrics.major_gaps_identified:
                recommendations.append(
                    "Enhance market impact modeling using order book depth and historical impact measurements"
                )

            if PerformanceGap.FEE_STRUCTURE_MISMATCH in parity_metrics.major_gaps_identified:
                recommendations.append(
                    "Align backtest fee calculations with actual exchange fee structures and volume tiers"
                )

            # General recommendations if parity score is low
            if parity_metrics.overall_parity_score < 0.7:
                recommendations.append(
                    "Implement comprehensive execution simulation with order book replay and market microstructure modeling"
                )
                recommendations.append(
                    "Regularly calibrate backtest parameters using rolling windows of live execution data"
                )

            return recommendations

        except Exception as e:
            logger.error(f"Parity recommendations generation failed: {e}")
            return ["Review and enhance backtest execution simulation to better match live conditions"]

    def get_parity_analytics(self, days_back: int = 90) -> Dict[str, Any]:
        """Get historical parity analytics"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            recent_metrics = [
                metrics for metrics in self.parity_history
                if datetime.now() - timedelta(days=metrics.analysis_period_days) >= cutoff_time
            ]

            if not recent_metrics:
                return {"status": "no_data"}

            # Aggregate statistics
            avg_parity_score = np.mean([m.overall_parity_score for m in recent_metrics])
            avg_return_gap = np.mean([abs(m.return_gap_pct) for m in recent_metrics])
            avg_slippage_gap = np.mean([abs(m.slippage_gap_bps) for m in recent_metrics])

            # Gap frequency analysis
            gap_frequency = {}
            for gap_type in PerformanceGap:
                frequency = sum(1 for m in recent_metrics if gap_type in m.major_gaps_identified) / len(recent_metrics)
                gap_frequency[gap_type.value] = frequency

            # Trend analysis
            parity_scores = [m.overall_parity_score for m in recent_metrics]
            parity_trend = np.polyfit(range(len(parity_scores)), parity_scores, 1)[0] if len(parity_scores) > 1 else 0

            analytics = {
                "analysis_period_days": days_back,
                "total_comparisons": len(recent_metrics),
                "parity_quality": {
                    "avg_parity_score": avg_parity_score,
                    "parity_trend": "improving" if parity_trend > 0.01 else "declining" if parity_trend < -0.01 else "stable",
                    "best_parity_score": max(parity_scores),
                    "worst_parity_score": min(parity_scores)
                },
                "performance_gaps": {
                    "avg_return_gap_pct": avg_return_gap,
                    "avg_slippage_gap_bps": avg_slippage_gap,
                    "gap_frequency": gap_frequency
                },
                "improvement_trend": {
                    "parity_improving": parity_trend > 0.01,
                    "trend_strength": abs(parity_trend),
                    "consistency": 1 - np.std(parity_scores) if len(parity_scores) > 1 else 0
                }
            }

            return analytics

        except Exception as e:
            logger.error(f"Parity analytics calculation failed: {e}")
            return {"status": "error", "error": str(e)}

    def _get_empty_metrics(self) -> Dict[str, Any]:
        """Get empty metrics dictionary"""
        return {
            "total_return": 0.0,
            "total_return_pct": 0.0,
            "avg_return": 0.0,
            "return_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_slippage_bps": 0.0,
            "fill_rate": 1.0,
            "trade_count": 0
        }

    def _create_insufficient_data_metrics(self) -> ParityMetrics:
        """Create metrics for insufficient data case"""
        return ParityMetrics(
            analysis_period_days=0,
            total_trades_compared=0,
            backtest_total_return_pct=0.0,
            live_total_return_pct=0.0,
            return_gap_pct=0.0,
            backtest_sharpe_ratio=0.0,
            live_sharpe_ratio=0.0,
            sharpe_gap=0.0,
            backtest_max_drawdown_pct=0.0,
            live_max_drawdown_pct=0.0,
            drawdown_gap_pct=0.0,
            backtest_avg_slippage_bps=0.0,
            live_avg_slippage_bps=0.0,
            slippage_gap_bps=0.0,
            backtest_fill_rate=1.0,
            live_fill_rate=1.0,
            fill_rate_gap=0.0,
            parity_checks=[],
            overall_parity_score=0.0,
            major_gaps_identified=[]
        )

    def _create_error_metrics(self) -> ParityMetrics:
        """Create error metrics"""
        return self._create_insufficient_data_metrics()
