"""
Return Attribution System

Comprehensive PnL decomposition into alpha, fees, slippage, timing, and sizing
to understand what drives trading performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AttributionSource(Enum):
    """Return attribution sources"""

    ALPHA = "alpha"  # Pure alpha generation
    FEES = "fees"  # Trading fees impact
    SLIPPAGE = "slippage"  # Execution slippage
    TIMING = "timing"  # Entry/exit timing
    SIZING = "sizing"  # Position sizing decisions
    MARKET_IMPACT = "market_impact"  # Market impact costs
    FUNDING = "funding"  # Funding rate impact
    REGIME_SHIFT = "regime_shift"  # Regime change impact
    VOLATILITY = "volatility"  # Volatility timing
    CORRELATION = "correlation"  # Cross-asset correlation


@dataclass
class AttributionComponent:
    """Individual attribution component"""

    source: AttributionSource
    value: float  # Attribution value in basis points
    contribution_pct: float  # Percentage contribution to total PnL
    confidence: float  # Confidence in this attribution (0-1)
    description: str  # Human-readable description

    # Supporting metrics
    trade_count: int = 0
    avg_per_trade: float = 0.0
    volatility: float = 0.0
    trend: str = "stable"  # "improving", "degrading", "stable"

    # Context data
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttributionResult:
    """Complete return attribution result"""

    period_start: datetime
    period_end: datetime

    # Overall metrics
    total_pnl_bps: float
    total_return_pct: float
    sharpe_ratio: float

    # Attribution components
    components: List[AttributionComponent] = field(default_factory=list)

    # Breakdown by dimensions
    pair_attribution: Dict[str, float] = field(default_factory=dict)
    regime_attribution: Dict[str, float] = field(default_factory=dict)
    time_attribution: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Quality metrics
    attribution_accuracy: float = 0.0  # How well we explained total PnL
    unexplained_pnl: float = 0.0  # PnL we couldn't attribute

    @property
    def attributed_pnl(self) -> float:
        """Total attributed PnL"""
        return sum(comp.value for comp in self.components)

    @property
    def alpha_contribution(self) -> float:
        """Pure alpha contribution"""
        alpha_comps = [c for c in self.components if c.source == AttributionSource.ALPHA]
        return sum(c.value for c in alpha_comps)

    @property
    def cost_drag(self) -> float:
        """Total cost drag (negative contributors)"""
        cost_sources = [
            AttributionSource.FEES,
            AttributionSource.SLIPPAGE,
            AttributionSource.MARKET_IMPACT,
        ]
        cost_comps = [c for c in self.components if c.source in cost_sources]
        return sum(c.value for c in cost_comps)


class ReturnAttributor:
    """
    Advanced return attribution system for comprehensive PnL analysis
    """

    def __init__(self):
        self.attribution_history = []

        # Attribution parameters
        self.lookback_periods = {"1D": 1, "1W": 7, "1M": 30, "3M": 90, "1Y": 365}

        # Benchmark costs for attribution
        self.benchmark_costs = {
            "fees_bps": {
                "maker": 25,  # 0.25% maker fee
                "taker": 40,  # 0.40% taker fee
                "average": 32.5,  # Average fee
            },
            "slippage_bps": {
                "major": 5,  # Major pairs
                "minor": 15,  # Minor pairs
                "micro": 30,  # Micro cap pairs
            },
            "market_impact_bps": {
                "small": 2,  # Small orders
                "medium": 8,  # Medium orders
                "large": 20,  # Large orders
            },
        }

    def attribute_returns(
        self, trade_data: pd.DataFrame, market_data: pd.DataFrame = None, period_days: int = 30
    ) -> AttributionResult:
        """
        Perform comprehensive return attribution analysis
        """
        try:
            logger.info(f"Performing return attribution for {period_days} days")

            # Filter data to analysis period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)

            period_trades = trade_data[
                (trade_data["timestamp"] >= start_date) & (trade_data["timestamp"] <= end_date)
            ].copy()

            if len(period_trades) == 0:
                logger.warning("No trades found for attribution period")
                return self._create_empty_attribution(start_date, end_date)

            # Calculate overall metrics
            total_pnl = period_trades["realized_pnl"].sum()
            total_return_pct = (total_pnl / period_trades["notional_value"].sum()) * 100

            returns = period_trades.groupby("date")["realized_pnl"].sum()
            returns_daily = returns / period_trades.groupby("date")["notional_value"].sum()
            sharpe = (
                np.mean(returns_daily) / np.std(returns_daily) * np.sqrt(252)
                if len(returns_daily) > 1
                else 0
            )

            # Perform attribution analysis
            components = self._analyze_attribution_components(period_trades, market_data)

            # Multi-dimensional attribution
            pair_attribution = self._attribute_by_pair(period_trades)
            regime_attribution = self._attribute_by_regime(period_trades, market_data)
            time_attribution = self._attribute_by_time(period_trades)

            # Calculate attribution quality
            attributed_total = sum(comp.value for comp in components)
            attribution_accuracy = min(1.0, abs(attributed_total) / max(abs(total_pnl), 1))
            unexplained = total_pnl - attributed_total

            result = AttributionResult(
                period_start=start_date,
                period_end=end_date,
                total_pnl_bps=total_pnl,
                total_return_pct=total_return_pct,
                sharpe_ratio=sharpe,
                components=components,
                pair_attribution=pair_attribution,
                regime_attribution=regime_attribution,
                time_attribution=time_attribution,
                attribution_accuracy=attribution_accuracy,
                unexplained_pnl=unexplained,
            )

            # Store result
            self.attribution_history.append(result)

            logger.info(
                f"Attribution completed: {len(components)} components, {attribution_accuracy:.1%} accuracy"
            )

            return result

        except Exception as e:
            logger.error(f"Return attribution failed: {e}")
            return self._create_empty_attribution(
                datetime.now() - timedelta(days=period_days), datetime.now()

    def _analyze_attribution_components(
        self, trade_data: pd.DataFrame, market_data: pd.DataFrame = None
    ) -> List[AttributionComponent]:
        """Analyze individual attribution components"""
        components = []

        try:
            # Alpha Attribution
            alpha_component = self._analyze_alpha_attribution(trade_data, market_data)
            if alpha_component:
                components.append(alpha_component)

            # Fee Attribution
            fee_component = self._analyze_fee_attribution(trade_data)
            if fee_component:
                components.append(fee_component)

            # Slippage Attribution
            slippage_component = self._analyze_slippage_attribution(trade_data)
            if slippage_component:
                components.append(slippage_component)

            # Timing Attribution
            timing_component = self._analyze_timing_attribution(trade_data, market_data)
            if timing_component:
                components.append(timing_component)

            # Sizing Attribution
            sizing_component = self._analyze_sizing_attribution(trade_data)
            if sizing_component:
                components.append(sizing_component)

            # Market Impact Attribution
            impact_component = self._analyze_market_impact_attribution(trade_data)
            if impact_component:
                components.append(impact_component)

            # Additional components based on available data
            if market_data is not None:
                # Volatility Attribution
                vol_component = self._analyze_volatility_attribution(trade_data, market_data)
                if vol_component:
                    components.append(vol_component)

                # Regime Shift Attribution
                regime_component = self._analyze_regime_attribution(trade_data, market_data)
                if regime_component:
                    components.append(regime_component)

            return components

        except Exception as e:
            logger.error(f"Component analysis failed: {e}")
            return []

    def _analyze_alpha_attribution(
        self, trade_data: pd.DataFrame, market_data: pd.DataFrame = None
    ) -> Optional[AttributionComponent]:
        """Analyze pure alpha generation"""
        try:
            # Alpha = Total PnL - All Identifiable Costs
            total_pnl = trade_data["realized_pnl"].sum()

            # Subtract identifiable costs
            fees = trade_data["fees"].sum() if "fees" in trade_data.columns else 0
            slippage = (
                trade_data["slippage_cost"].sum() if "slippage_cost" in trade_data.columns else 0
            )

            # Estimate alpha as residual after costs
            estimated_alpha = total_pnl - fees - slippage

            # Calculate metrics
            trade_count = len(trade_data)
            avg_per_trade = estimated_alpha / trade_count if trade_count > 0 else 0

            # Volatility of alpha per trade
            if "expected_return" in trade_data.columns:
                alpha_per_trade = (
                    trade_data["realized_pnl"]
                    - trade_data.get("fees", 0)
                    - trade_data.get("slippage_cost", 0)
                alpha_volatility = np.std(alpha_per_trade) if len(alpha_per_trade) > 1 else 0
            else:
                alpha_volatility = np.std(trade_data["realized_pnl"]) if len(trade_data) > 1 else 0

            # Determine trend
            if len(trade_data) >= 10:
                # Recent vs older performance
                half_point = len(trade_data) // 2
                recent_alpha = trade_data.iloc[half_point:]["realized_pnl"].mean()
                older_alpha = trade_data.iloc[:half_point]["realized_pnl"].mean()

                if recent_alpha > older_alpha * 1.1:
                    trend = "improving"
                elif recent_alpha < older_alpha * 0.9:
                    trend = "degrading"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            contribution_pct = (estimated_alpha / total_pnl * 100) if total_pnl != 0 else 0

            return AttributionComponent(
                source=AttributionSource.ALPHA,
                value=estimated_alpha,
                contribution_pct=contribution_pct,
                confidence=0.7,  # Medium confidence as it's residual
                description=f"Pure alpha generation: {estimated_alpha:.1f} bps across {trade_count} trades",
                trade_count=trade_count,
                avg_per_trade=avg_per_trade,
                volatility=alpha_volatility,
                trend=trend,
                context={
                    "total_pnl": total_pnl,
                    "identified_costs": fees + slippage,
                    "alpha_per_trade_std": alpha_volatility,
                },
            )

        except Exception as e:
            logger.error(f"Alpha attribution failed: {e}")
            return None

    def _analyze_fee_attribution(self, trade_data: pd.DataFrame) -> Optional[AttributionComponent]:
        """Analyze trading fees impact"""
        try:
            if "fees" not in trade_data.columns:
                return None

            total_fees = trade_data["fees"].sum()
            trade_count = len(trade_data)
            avg_fee_per_trade = total_fees / trade_count if trade_count > 0 else 0

            # Calculate fee efficiency vs benchmark
            notional_volume = trade_data["notional_value"].sum()
            actual_fee_rate = abs(total_fees) / notional_volume if notional_volume > 0 else 0
            benchmark_fee_rate = self.benchmark_costs["fees_bps"]["average"] / 10000

            fee_efficiency = (benchmark_fee_rate - actual_fee_rate) * notional_volume

            # Analyze maker vs taker rates if available
            if "fill_type" in trade_data.columns:
                maker_trades = trade_data[trade_data["fill_type"] == "maker"]
                taker_trades = trade_data[trade_data["fill_type"] == "taker"]

                maker_rate = len(maker_trades) / len(trade_data) if len(trade_data) > 0 else 0

                context = {
                    "maker_rate": maker_rate,
                    "fee_efficiency_bps": fee_efficiency,
                    "actual_fee_rate_bps": actual_fee_rate * 10000,
                    "benchmark_fee_rate_bps": benchmark_fee_rate * 10000,
                }
            else:
                context = {
                    "fee_efficiency_bps": fee_efficiency,
                    "actual_fee_rate_bps": actual_fee_rate * 10000,
                }

            total_pnl = trade_data["realized_pnl"].sum()
            contribution_pct = (total_fees / total_pnl * 100) if total_pnl != 0 else 0

            return AttributionComponent(
                source=AttributionSource.FEES,
                value=total_fees,  # Negative impact
                contribution_pct=contribution_pct,
                confidence=0.95,  # High confidence - fees are exact
                description=f"Trading fees: {total_fees:.1f} bps across {trade_count} trades ({actual_fee_rate * 10000:.1f} bps rate)",
                trade_count=trade_count,
                avg_per_trade=avg_fee_per_trade,
                volatility=np.std(trade_data["fees"]) if len(trade_data) > 1 else 0,
                trend="stable",  # Fees typically stable
                context=context,
            )

        except Exception as e:
            logger.error(f"Fee attribution failed: {e}")
            return None

    def _analyze_slippage_attribution(
        self, trade_data: pd.DataFrame
    ) -> Optional[AttributionComponent]:
        """Analyze execution slippage impact"""
        try:
            # Calculate slippage from available data
            if "slippage_bps" in trade_data.columns:
                slippage_values = trade_data["slippage_bps"]
            elif "expected_price" in trade_data.columns and "actual_price" in trade_data.columns:
                # Calculate slippage from price difference
                price_diff = (
                    trade_data["actual_price"] - trade_data["expected_price"]
                ) / trade_data["expected_price"]
                slippage_values = price_diff * 10000  # Convert to bps
            else:
                return None

            total_slippage = slippage_values.sum()
            trade_count = len(trade_data)
            avg_slippage = np.mean(slippage_values)
            slippage_volatility = np.std(slippage_values) if len(slippage_values) > 1 else 0

            # Analyze slippage trend
            if len(slippage_values) >= 10:
                recent_slippage = slippage_values.tail(len(slippage_values) // 2).mean()
                older_slippage = slippage_values.head(len(slippage_values) // 2).mean()

                if recent_slippage > older_slippage * 1.2:
                    trend = "degrading"
                elif recent_slippage < older_slippage * 0.8:
                    trend = "improving"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            # Compare to benchmark
            pair_types = trade_data.get("pair_type", ["minor"] * len(trade_data))
            benchmark_slippage = np.mean(
                [self.benchmark_costs["slippage_bps"].get(pt, 15) for pt in pair_types]
            )

            total_pnl = trade_data["realized_pnl"].sum()
            contribution_pct = (total_slippage / total_pnl * 100) if total_pnl != 0 else 0

            return AttributionComponent(
                source=AttributionSource.SLIPPAGE,
                value=total_slippage,
                contribution_pct=contribution_pct,
                confidence=0.8,
                description=f"Execution slippage: {avg_slippage:.1f} bps avg, {slippage_volatility:.1f} volatility",
                trade_count=trade_count,
                avg_per_trade=avg_slippage,
                volatility=slippage_volatility,
                trend=trend,
                context={
                    "benchmark_slippage_bps": benchmark_slippage,
                    "excess_slippage_bps": avg_slippage - benchmark_slippage,
                    "worst_slippage_bps": max(slippage_values) if len(slippage_values) > 0 else 0,
                    "slippage_trend": trend,
                },
            )

        except Exception as e:
            logger.error(f"Slippage attribution failed: {e}")
            return None

    def _analyze_timing_attribution(
        self, trade_data: pd.DataFrame, market_data: pd.DataFrame = None
    ) -> Optional[AttributionComponent]:
        """Analyze entry/exit timing impact"""
        try:
            if "entry_time" not in trade_data.columns or "exit_time" not in trade_data.columns:
                return None

            timing_impacts = []

            for _, trade in trade_data.iterrows():
                # Simple timing attribution based on holding period performance
                entry_price = trade.get("entry_price", 0)
                exit_price = trade.get("exit_price", 0)
                expected_return = trade.get("expected_return", 0)
                actual_return = (exit_price - entry_price) / entry_price if entry_price > 0 else 0

                # Timing impact = actual - expected (simplified)
                timing_impact = (actual_return - expected_return) * 10000  # bps
                timing_impacts.append(timing_impact)

            if not timing_impacts:
                return None

            total_timing = sum(timing_impacts)
            avg_timing = np.mean(timing_impacts)
            timing_volatility = np.std(timing_impacts) if len(timing_impacts) > 1 else 0

            # Analyze timing trend
            if len(timing_impacts) >= 10:
                recent_timing = np.mean(timing_impacts[-len(timing_impacts) // 2 :])
                older_timing = np.mean(timing_impacts[: len(timing_impacts) // 2])

                if recent_timing > older_timing * 1.1:
                    trend = "improving"
                elif recent_timing < older_timing * 0.9:
                    trend = "degrading"
                else:
                    trend = "stable"
            else:
                trend = "stable"

            total_pnl = trade_data["realized_pnl"].sum()
            contribution_pct = (total_timing / total_pnl * 100) if total_pnl != 0 else 0

            return AttributionComponent(
                source=AttributionSource.TIMING,
                value=total_timing,
                contribution_pct=contribution_pct,
                confidence=0.6,  # Lower confidence as timing is hard to measure
                description=f"Entry/exit timing: {avg_timing:.1f} bps avg impact",
                trade_count=len(timing_impacts),
                avg_per_trade=avg_timing,
                volatility=timing_volatility,
                trend=trend,
                context={
                    "best_timing_bps": max(timing_impacts) if timing_impacts else 0,
                    "worst_timing_bps": min(timing_impacts) if timing_impacts else 0,
                    "timing_hit_rate": len([t for t in timing_impacts if t > 0])
                    / len(timing_impacts)
                    if timing_impacts
                    else 0,
                },
            )

        except Exception as e:
            logger.error(f"Timing attribution failed: {e}")
            return None

    def _analyze_sizing_attribution(
        self, trade_data: pd.DataFrame
    ) -> Optional[AttributionComponent]:
        """Analyze position sizing impact"""
        try:
            if (
                "position_size" not in trade_data.columns
                or "optimal_size" not in trade_data.columns
            ):
                return None

            sizing_impacts = []

            for _, trade in trade_data.iterrows():
                actual_size = trade["position_size"]
                optimal_size = trade["optimal_size"]
                trade_return = trade.get("return_pct", 0)

                # Sizing impact = (actual_size - optimal_size) * trade_return
                size_ratio = actual_size / optimal_size if optimal_size > 0 else 1
                sizing_impact = (size_ratio - 1) * trade_return * 10000  # bps
                sizing_impacts.append(sizing_impact)

            if not sizing_impacts:
                return None

            total_sizing = sum(sizing_impacts)
            avg_sizing = np.mean(sizing_impacts)
            sizing_volatility = np.std(sizing_impacts) if len(sizing_impacts) > 1 else 0

            total_pnl = trade_data["realized_pnl"].sum()
            contribution_pct = (total_sizing / total_pnl * 100) if total_pnl != 0 else 0

            return AttributionComponent(
                source=AttributionSource.SIZING,
                value=total_sizing,
                contribution_pct=contribution_pct,
                confidence=0.7,
                description=f"Position sizing: {avg_sizing:.1f} bps avg impact from sizing decisions",
                trade_count=len(sizing_impacts),
                avg_per_trade=avg_sizing,
                volatility=sizing_volatility,
                trend="stable",
                context={
                    "avg_size_ratio": np.mean(
                        trade_data["position_size"] / trade_data["optimal_size"]
                    ),
                    "oversized_trades": len(
                        trade_data[trade_data["position_size"] > trade_data["optimal_size"]]
                    ),
                    "undersized_trades": len(
                        trade_data[trade_data["position_size"] < trade_data["optimal_size"]]
                    ),
                },
            )

        except Exception as e:
            logger.error(f"Sizing attribution failed: {e}")
            return None

    def _analyze_market_impact_attribution(
        self, trade_data: pd.DataFrame
    ) -> Optional[AttributionComponent]:
        """Analyze market impact costs"""
        try:
            if "market_impact_bps" not in trade_data.columns:
                # Estimate market impact based on order size
                if "order_size" not in trade_data.columns:
                    return None

                impact_estimates = []
                for _, trade in trade_data.iterrows():
                    size = trade["order_size"]
                    if size < 1000:
                        impact = self.benchmark_costs["market_impact_bps"]["small"]
                    elif size < 10000:
                        impact = self.benchmark_costs["market_impact_bps"]["medium"]
                    else:
                        impact = self.benchmark_costs["market_impact_bps"]["large"]
                    impact_estimates.append(impact)

                market_impacts = np.array(impact_estimates)
            else:
                market_impacts = trade_data["market_impact_bps"].values

            total_impact = np.sum(market_impacts)
            avg_impact = np.mean(market_impacts)
            impact_volatility = np.std(market_impacts) if len(market_impacts) > 1 else 0

            total_pnl = trade_data["realized_pnl"].sum()
            contribution_pct = (total_impact / total_pnl * 100) if total_pnl != 0 else 0

            return AttributionComponent(
                source=AttributionSource.MARKET_IMPACT,
                value=total_impact,
                contribution_pct=contribution_pct,
                confidence=0.6,  # Medium confidence for estimates
                description=f"Market impact: {avg_impact:.1f} bps avg cost",
                trade_count=len(market_impacts),
                avg_per_trade=avg_impact,
                volatility=impact_volatility,
                trend="stable",
                context={
                    "max_impact_bps": np.max(market_impacts) if len(market_impacts) > 0 else 0,
                    "impact_per_size": avg_impact / np.mean(trade_data.get("order_size", [1000]))
                    if len(trade_data) > 0
                    else 0,
                },
            )

        except Exception as e:
            logger.error(f"Market impact attribution failed: {e}")
            return None

    def _analyze_volatility_attribution(
        self, trade_data: pd.DataFrame, market_data: pd.DataFrame
    ) -> Optional[AttributionComponent]:
        """Analyze volatility timing impact"""
        # Implementation would analyze how volatility changes affected performance
        # This is a placeholder for the complex volatility attribution logic
        return None

    def _analyze_regime_attribution(
        self, trade_data: pd.DataFrame, market_data: pd.DataFrame
    ) -> Optional[AttributionComponent]:
        """Analyze regime change impact"""
        # Implementation would analyze how regime shifts affected performance
        # This is a placeholder for the complex regime attribution logic
        return None

    def _attribute_by_pair(self, trade_data: pd.DataFrame) -> Dict[str, float]:
        """Attribute returns by trading pair"""
        try:
            if "pair" not in trade_data.columns:
                return {}

            pair_attribution = trade_data.groupby("pair")["realized_pnl"].sum().to_dict()
            return pair_attribution

        except Exception as e:
            logger.error(f"Pair attribution failed: {e}")
            return {}

    def _attribute_by_regime(
        self, trade_data: pd.DataFrame, market_data: pd.DataFrame = None
    ) -> Dict[str, float]:
        """Attribute returns by market regime"""
        try:
            if "regime" not in trade_data.columns:
                return {}

            regime_attribution = trade_data.groupby("regime")["realized_pnl"].sum().to_dict()
            return regime_attribution

        except Exception as e:
            logger.error(f"Regime attribution failed: {e}")
            return {}

    def _attribute_by_time(self, trade_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Attribute returns by time dimensions"""
        try:
            time_attribution = {}

            if "timestamp" in trade_data.columns:
                trade_data_copy = trade_data.copy()
                trade_data_copy["hour"] = pd.to_datetime(trade_data_copy["timestamp"]).dt.hour
                trade_data_copy["day_of_week"] = pd.to_datetime(
                    trade_data_copy["timestamp"]
                ).dt.day_name()
                trade_data_copy["month"] = pd.to_datetime(trade_data_copy["timestamp"]).dt.month

                # Hourly attribution
                time_attribution["hourly"] = (
                    trade_data_copy.groupby("hour")["realized_pnl"].sum().to_dict()

                # Daily attribution
                time_attribution["daily"] = (
                    trade_data_copy.groupby("day_of_week")["realized_pnl"].sum().to_dict()

                # Monthly attribution
                time_attribution["monthly"] = (
                    trade_data_copy.groupby("month")["realized_pnl"].sum().to_dict()

            return time_attribution

        except Exception as e:
            logger.error(f"Time attribution failed: {e}")
            return {}

    def _create_empty_attribution(
        self, start_date: datetime, end_date: datetime
    ) -> AttributionResult:
        """Create empty attribution result"""
        return AttributionResult(
            period_start=start_date,
            period_end=end_date,
            total_pnl_bps=0.0,
            total_return_pct=0.0,
            sharpe_ratio=0.0,
            components=[],
            pair_attribution={},
            regime_attribution={},
            time_attribution={},
            attribution_accuracy=0.0,
            unexplained_pnl=0.0,
        )

    def get_attribution_trends(self, days_back: int = 90) -> Dict[str, Any]:
        """Analyze attribution trends over time"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_results = [r for r in self.attribution_history if r.period_start >= cutoff_date]

            if not recent_results:
                return {"status": "no_data"}

            # Track component trends
            component_trends = {}
            for source in AttributionSource:
                source_values = []
                for result in recent_results:
                    source_comps = [c for c in result.components if c.source == source]
                    total_value = sum(c.value for c in source_comps)
                    source_values.append(total_value)

                if source_values:
                    component_trends[source.value] = {
                        "recent_avg": np.mean(source_values),
                        "trend": "improving"
                        if source_values[-1] > np.mean(source_values[:-1])
                        else "stable",
                        "volatility": np.std(source_values),
                        "contribution": np.mean([abs(v) for v in source_values]),
                    }

            return {
                "analysis_period_days": days_back,
                "results_count": len(recent_results),
                "component_trends": component_trends,
                "overall_attribution_quality": np.mean(
                    [r.attribution_accuracy for r in recent_results]
                ),
                "top_contributors": sorted(
                    component_trends.items(), key=lambda x: x[1]["contribution"], reverse=True
                )[:5],
            }

        except Exception as e:
            logger.error(f"Attribution trends analysis failed: {e}")
            return {"status": "error", "error": str(e)}
