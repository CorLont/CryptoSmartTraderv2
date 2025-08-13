"""
Return Attribution System for CryptoSmartTrader
Advanced return decomposition: alpha vs fees vs slippage vs timing vs sizing.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging


class ReturnComponent(Enum):
    """Return attribution components."""

    ALPHA = "alpha"
    FEES = "fees"
    SLIPPAGE = "slippage"
    TIMING = "timing"
    SIZING = "sizing"
    MARKET_IMPACT = "market_impact"
    FUNDING = "funding"
    REBALANCING = "rebalancing"


@dataclass
class ReturnAttribution:
    """Detailed return attribution breakdown."""

    symbol: str
    period_start: datetime
    period_end: datetime
    total_return: float
    components: Dict[ReturnComponent, float]
    confidence_scores: Dict[ReturnComponent, float]
    benchmark_return: float
    excess_return: float
    attribution_quality: float
    metadata: Dict[str, Any]


@dataclass
class PortfolioAttribution:
    """Portfolio-level return attribution."""

    period_start: datetime
    period_end: datetime
    total_portfolio_return: float
    benchmark_return: float
    excess_return: float
    asset_attributions: Dict[str, ReturnAttribution]
    component_summary: Dict[ReturnComponent, float]
    risk_adjusted_metrics: Dict[str, float]
    performance_quality: float
    timestamp: datetime


class ReturnAttributor:
    """
    Enterprise return attribution system with comprehensive decomposition.

    Features:
    - Alpha decomposition from execution costs
    - Transaction cost analysis (fees, slippage, market impact)
    - Timing attribution (entry/exit optimization)
    - Sizing attribution (Kelly vs realized sizing)
    - Funding rate attribution
    - Rebalancing cost attribution
    - Statistical confidence scoring
    """

    def __init__(
        self,
        benchmark_return: float = 0.0,
        min_attribution_period: int = 24,  # hours
        confidence_threshold: float = 0.7,
    ):
        self.benchmark_return = benchmark_return
        self.min_attribution_period = min_attribution_period
        self.confidence_threshold = confidence_threshold

        # Transaction cost assumptions
        self.base_trading_fee = 0.001  # 0.1% base fee
        self.slippage_threshold = 0.002  # 0.2% expected slippage
        self.market_impact_factor = 0.0001  # Market impact per 1% of volume

        self.logger = logging.getLogger(__name__)
        self.logger.info("ReturnAttributor initialized with enterprise attribution")

    def attribute_returns(
        self,
        portfolio_data: Dict[str, Dict[str, Any]],
        execution_data: Dict[str, List[Dict[str, Any]]],
        market_data: Dict[str, pd.DataFrame],
        period_start: datetime,
        period_end: datetime,
    ) -> PortfolioAttribution:
        """
        Perform comprehensive return attribution analysis.

        Args:
            portfolio_data: Portfolio positions and returns
            execution_data: Trade execution details
            market_data: Market price and volume data
            period_start: Attribution period start
            period_end: Attribution period end

        Returns:
            PortfolioAttribution with detailed breakdown
        """

        asset_attributions = {}

        # Attribute returns for each asset
        for symbol in portfolio_data:
            if symbol not in market_data:
                continue

            asset_attribution = self._attribute_asset_returns(
                symbol,
                portfolio_data[symbol],
                execution_data.get(symbol, []),
                market_data[symbol],
                period_start,
                period_end,
            )

            asset_attributions[symbol] = asset_attribution

        # Calculate portfolio-level attribution
        portfolio_attribution = self._calculate_portfolio_attribution(
            asset_attributions, period_start, period_end
        )

        return portfolio_attribution

    def _attribute_asset_returns(
        self,
        symbol: str,
        portfolio_data: Dict[str, Any],
        executions: List[Dict[str, Any]],
        price_data: pd.DataFrame,
        period_start: datetime,
        period_end: datetime,
    ) -> ReturnAttribution:
        """Attribute returns for single asset."""

        # Calculate total return
        start_price = self._get_price_at_time(price_data, period_start)
        end_price = self._get_price_at_time(price_data, period_end)

        if start_price is None or end_price is None or start_price <= 0:
            return self._empty_attribution(symbol, period_start, period_end)

        # Price return (before costs)
        price_return = (end_price / start_price) - 1

        # Initialize components
        components = {comp: 0.0 for comp in ReturnComponent}
        confidence_scores = {comp: 1.0 for comp in ReturnComponent}

        # 1. Calculate fees attribution
        fees_component, fees_confidence = self._calculate_fees_attribution(
            executions, portfolio_data.get("position_size", 0.0)
        components[ReturnComponent.FEES] = fees_component
        confidence_scores[ReturnComponent.FEES] = fees_confidence

        # 2. Calculate slippage attribution
        slippage_component, slippage_confidence = self._calculate_slippage_attribution(
            executions, price_data
        )
        components[ReturnComponent.SLIPPAGE] = slippage_component
        confidence_scores[ReturnComponent.SLIPPAGE] = slippage_confidence

        # 3. Calculate timing attribution
        timing_component, timing_confidence = self._calculate_timing_attribution(
            executions, price_data, period_start, period_end
        )
        components[ReturnComponent.TIMING] = timing_component
        confidence_scores[ReturnComponent.TIMING] = timing_confidence

        # 4. Calculate sizing attribution
        sizing_component, sizing_confidence = self._calculate_sizing_attribution(
            portfolio_data, price_return
        )
        components[ReturnComponent.SIZING] = sizing_component
        confidence_scores[ReturnComponent.SIZING] = sizing_confidence

        # 5. Calculate market impact attribution
        impact_component, impact_confidence = self._calculate_market_impact_attribution(
            executions, price_data
        )
        components[ReturnComponent.MARKET_IMPACT] = impact_component
        confidence_scores[ReturnComponent.MARKET_IMPACT] = impact_confidence

        # 6. Calculate funding attribution (for leveraged positions)
        funding_component, funding_confidence = self._calculate_funding_attribution(
            portfolio_data, period_start, period_end
        )
        components[ReturnComponent.FUNDING] = funding_component
        confidence_scores[ReturnComponent.FUNDING] = funding_confidence

        # 7. Calculate rebalancing attribution
        rebalancing_component, rebalancing_confidence = self._calculate_rebalancing_attribution(
            executions, price_data
        )
        components[ReturnComponent.REBALANCING] = rebalancing_component
        confidence_scores[ReturnComponent.REBALANCING] = rebalancing_confidence

        # 8. Calculate alpha (residual after all costs)
        total_costs = sum(components[comp] for comp in components if comp != ReturnComponent.ALPHA)
        alpha_component = price_return - total_costs
        components[ReturnComponent.ALPHA] = alpha_component
        confidence_scores[ReturnComponent.ALPHA] = self._calculate_alpha_confidence(
            components, executions
        )

        # Calculate total realized return
        total_return = sum(components.values())

        # Calculate attribution quality
        attribution_quality = self._calculate_attribution_quality(
            components, confidence_scores, total_return
        )

        return ReturnAttribution(
            symbol=symbol,
            period_start=period_start,
            period_end=period_end,
            total_return=total_return,
            components=components,
            confidence_scores=confidence_scores,
            benchmark_return=self.benchmark_return,
            excess_return=total_return - self.benchmark_return,
            attribution_quality=attribution_quality,
            metadata={
                "price_return": price_return,
                "execution_count": len(executions),
                "start_price": start_price,
                "end_price": end_price,
            },
        )

    def _calculate_fees_attribution(
        self, executions: List[Dict[str, Any]], position_size: float
    ) -> Tuple[float, float]:
        """Calculate fees attribution component."""
        if not executions or position_size <= 0:
            return 0.0, 1.0

        total_fees = 0.0
        total_volume = 0.0

        for execution in executions:
            trade_value = execution.get("quantity", 0) * execution.get("price", 0)
            fee_rate = execution.get("fee_rate", self.base_trading_fee)
            fees = trade_value * fee_rate

            total_fees += fees
            total_volume += trade_value

        # Calculate fees as percentage of position value
        position_value = position_size  # Assuming position_size is in value terms
        if position_value > 0:
            fees_attribution = -total_fees / position_value  # Negative because fees reduce returns
        else:
            fees_attribution = 0.0

        # Confidence based on execution data quality
        confidence = 0.9 if total_volume > 0 else 0.5

        return fees_attribution, confidence

    def _calculate_slippage_attribution(
        self, executions: List[Dict[str, Any]], price_data: pd.DataFrame
    ) -> Tuple[float, float]:
        """Calculate slippage attribution component."""
        if not executions:
            return 0.0, 1.0

        total_slippage = 0.0
        total_volume = 0.0
        slippage_count = 0

        for execution in executions:
            execution_time = execution.get("timestamp")
            execution_price = execution.get("price", 0)
            quantity = execution.get("quantity", 0)
            side = execution.get("side", "buy")

            if not execution_time or execution_price <= 0:
                continue

            # Get reference price at execution time
            reference_price = self._get_price_at_time(price_data, execution_time)
            if reference_price is None or reference_price <= 0:
                continue

            # Calculate slippage
            if side == "buy":
                slippage = (execution_price - reference_price) / reference_price
            else:  # sell
                slippage = (reference_price - execution_price) / reference_price

            # Weight by trade size
            trade_value = quantity * execution_price
            total_slippage += slippage * trade_value
            total_volume += trade_value
            slippage_count += 1

        # Calculate weighted average slippage
        if total_volume > 0:
            avg_slippage = total_slippage / total_volume
        else:
            avg_slippage = 0.0

        # Confidence based on sample size
        confidence = min(0.95, 0.5 + (slippage_count * 0.1))

        return -abs(avg_slippage), confidence  # Negative because slippage reduces returns

    def _calculate_timing_attribution(
        self,
        executions: List[Dict[str, Any]],
        price_data: pd.DataFrame,
        period_start: datetime,
        period_end: datetime,
    ) -> Tuple[float, float]:
        """Calculate timing attribution component."""
        if not executions or len(price_data) < 24:  # Need sufficient price history
            return 0.0, 0.5

        timing_value = 0.0
        total_trades = 0

        # Calculate VWAP for the period as benchmark
        if "volume" in price_data.columns:
            vwap = (price_data["close"] * price_data["volume"]).sum() / price_data["volume"].sum()
        else:
            vwap = price_data["close"].mean()

        for execution in executions:
            execution_price = execution.get("price", 0)
            quantity = execution.get("quantity", 0)
            side = execution.get("side", "buy")

            if execution_price <= 0 or quantity <= 0:
                continue

            # Compare execution price to VWAP
            if side == "buy":
                timing_benefit = (vwap - execution_price) / vwap  # Better if bought below VWAP
            else:  # sell
                timing_benefit = (execution_price - vwap) / vwap  # Better if sold above VWAP

            timing_value += timing_benefit
            total_trades += 1

        # Average timing benefit
        if total_trades > 0:
            avg_timing = timing_value / total_trades
        else:
            avg_timing = 0.0

        # Confidence based on number of executions
        confidence = min(0.8, 0.3 + (total_trades * 0.1))

        return avg_timing, confidence

    def _calculate_sizing_attribution(
        self, portfolio_data: Dict[str, Any], price_return: float
    ) -> Tuple[float, float]:
        """Calculate sizing attribution component."""

        optimal_size = portfolio_data.get("optimal_size", 0.0)
        actual_size = portfolio_data.get("actual_size", 0.0)

        if optimal_size <= 0:
            return 0.0, 0.3  # Low confidence without optimal sizing

        # Calculate sizing efficiency
        size_ratio = actual_size / optimal_size

        # If undersized and positive return, lost opportunity
        # If oversized and negative return, avoided loss
        if price_return > 0:
            # Positive return - undersizing is bad, oversizing is good (up to a point)
            if size_ratio < 1.0:
                sizing_attribution = (size_ratio - 1.0) * price_return  # Negative
            else:
                sizing_attribution = min(0.02, (size_ratio - 1.0) * price_return * 0.5)
        else:
            # Negative return - undersizing is good, oversizing is bad
            sizing_attribution = (1.0 - size_ratio) * abs(price_return)

        # Confidence based on data availability
        confidence = 0.7 if optimal_size > 0 else 0.3

        return sizing_attribution, confidence

    def _calculate_market_impact_attribution(
        self, executions: List[Dict[str, Any]], price_data: pd.DataFrame
    ) -> Tuple[float, float]:
        """Calculate market impact attribution component."""
        if not executions:
            return 0.0, 1.0

        total_impact = 0.0
        impact_count = 0

        for execution in executions:
            quantity = execution.get("quantity", 0)
            price = execution.get("price", 0)
            timestamp = execution.get("timestamp")

            if quantity <= 0 or price <= 0 or not timestamp:
                continue

            # Estimate daily volume from price data
            if "volume" in price_data.columns:
                daily_volume = price_data["volume"].mean()
            else:
                # Estimate volume based on price volatility
                volatility = price_data["close"].pct_change().std()
                daily_volume = price * 1000 * volatility  # Rough estimate

            if daily_volume <= 0:
                continue

            # Calculate trade size as percentage of daily volume
            trade_value = quantity * price
            volume_percentage = trade_value / daily_volume

            # Market impact = impact_factor * sqrt(volume_percentage)
            market_impact = self.market_impact_factor * np.sqrt(volume_percentage)

            total_impact += market_impact
            impact_count += 1

        # Average market impact
        if impact_count > 0:
            avg_impact = total_impact / impact_count
        else:
            avg_impact = 0.0

        # Confidence based on volume data availability
        has_volume_data = "volume" in price_data.columns
        confidence = 0.8 if has_volume_data else 0.4

        return -avg_impact, confidence  # Negative because impact reduces returns

    def _calculate_funding_attribution(
        self, portfolio_data: Dict[str, Any], period_start: datetime, period_end: datetime
    ) -> Tuple[float, float]:
        """Calculate funding attribution for leveraged positions."""

        leverage = portfolio_data.get("leverage", 1.0)
        funding_rate = portfolio_data.get("funding_rate", 0.0)

        if leverage <= 1.0 or funding_rate == 0.0:
            return 0.0, 1.0

        # Calculate funding period
        period_hours = (period_end - period_start).total_seconds() / 3600
        funding_periods = period_hours / 8  # Funding every 8 hours typically

        # Funding cost = funding_rate * leveraged_amount * periods
        leveraged_exposure = leverage - 1.0  # Only leveraged portion pays funding
        funding_cost = funding_rate * leveraged_exposure * funding_periods

        # Confidence based on data availability
        confidence = 0.9 if funding_rate != 0.0 else 0.5

        return -funding_cost, confidence  # Negative because funding reduces returns

    def _calculate_rebalancing_attribution(
        self, executions: List[Dict[str, Any]], price_data: pd.DataFrame
    ) -> Tuple[float, float]:
        """Calculate rebalancing cost attribution."""
        if len(executions) <= 1:
            return 0.0, 1.0  # No rebalancing if only one trade

        rebalancing_cost = 0.0
        rebalancing_count = 0

        # Look for paired buy/sell trades that indicate rebalancing
        buy_trades = [e for e in executions if e.get("side") == "buy"]
        sell_trades = [e for e in executions if e.get("side") == "sell"]

        # Estimate rebalancing frequency
        if len(buy_trades) > 1 or len(sell_trades) > 1:
            # Multiple trades suggest rebalancing
            total_trades = len(executions)

            # Estimate cost as percentage of excess trading
            base_trades = 2  # Assume 2 trades (buy and sell) for basic position
            excess_trades = max(0, total_trades - base_trades)

            # Each excess trade costs approximately the spread
            estimated_spread = self.slippage_threshold * 2  # Bid-ask spread
            rebalancing_cost = excess_trades * estimated_spread * 0.1  # 10% of spread per trade

            rebalancing_count = excess_trades

        # Confidence based on trade pattern analysis
        confidence = 0.6 if rebalancing_count > 0 else 0.8

        return -rebalancing_cost, confidence

    def _calculate_alpha_confidence(
        self, components: Dict[ReturnComponent, float], executions: List[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in alpha attribution."""

        # Base confidence
        confidence = 0.7

        # Reduce confidence if large unexplained components
        total_costs = sum(
            abs(components[comp]) for comp in components if comp != ReturnComponent.ALPHA
        )
        alpha_magnitude = abs(components[ReturnComponent.ALPHA])

        if total_costs > 0:
            cost_ratio = alpha_magnitude / total_costs
            if cost_ratio < 0.5:  # Alpha much smaller than costs
                confidence *= 0.8
            elif cost_ratio > 5.0:  # Alpha much larger than costs
                confidence *= 0.9

        # Increase confidence with more execution data
        execution_bonus = min(0.2, len(executions) * 0.02)
        confidence += execution_bonus

        return min(0.95, confidence)

    def _calculate_attribution_quality(
        self,
        components: Dict[ReturnComponent, float],
        confidence_scores: Dict[ReturnComponent, float],
        total_return: float,
    ) -> float:
        """Calculate overall attribution quality score."""

        # Weighted average of component confidences
        component_weights = {}
        for comp, value in components.items():
            if total_return != 0:
                component_weights[comp] = abs(value) / abs(total_return)
            else:
                component_weights[comp] = 1.0 / len(components)

        # Normalize weights
        total_weight = sum(component_weights.values())
        if total_weight > 0:
            for comp in component_weights:
                component_weights[comp] /= total_weight

        # Calculate weighted confidence
        weighted_confidence = sum(
            confidence_scores[comp] * component_weights[comp] for comp in components
        )

        return weighted_confidence

    def _calculate_portfolio_attribution(
        self,
        asset_attributions: Dict[str, ReturnAttribution],
        period_start: datetime,
        period_end: datetime,
    ) -> PortfolioAttribution:
        """Calculate portfolio-level attribution from asset attributions."""

        if not asset_attributions:
            return self._empty_portfolio_attribution(period_start, period_end)

        # Aggregate component returns (assuming equal weighting for simplicity)
        n_assets = len(asset_attributions)
        component_summary = {comp: 0.0 for comp in ReturnComponent}

        total_portfolio_return = 0.0
        total_excess_return = 0.0

        for attribution in asset_attributions.values():
            total_portfolio_return += attribution.total_return / n_assets
            total_excess_return += attribution.excess_return / n_assets

            for comp, value in attribution.components.items():
                component_summary[comp] += value / n_assets

        # Calculate portfolio risk-adjusted metrics
        risk_adjusted_metrics = self._calculate_portfolio_risk_metrics(
            asset_attributions, total_portfolio_return
        )

        # Calculate performance quality
        quality_scores = [attr.attribution_quality for attr in asset_attributions.values()]
        performance_quality = np.mean(quality_scores)

        return PortfolioAttribution(
            period_start=period_start,
            period_end=period_end,
            total_portfolio_return=total_portfolio_return,
            benchmark_return=self.benchmark_return,
            excess_return=total_excess_return,
            asset_attributions=asset_attributions,
            component_summary=component_summary,
            risk_adjusted_metrics=risk_adjusted_metrics,
            performance_quality=performance_quality,
            timestamp=datetime.utcnow(),
        )

    def _calculate_portfolio_risk_metrics(
        self, asset_attributions: Dict[str, ReturnAttribution], portfolio_return: float
    ) -> Dict[str, float]:
        """Calculate portfolio risk-adjusted performance metrics."""

        returns = [attr.total_return for attr in asset_attributions.values()]

        if not returns:
            return {}

        # Portfolio volatility (simple average for demonstration)
        portfolio_vol = np.std(returns) if len(returns) > 1 else 0.0

        # Sharpe ratio
        sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0.0

        # Information ratio (excess return / tracking error)
        excess_returns = [attr.excess_return for attr in asset_attributions.values()]
        tracking_error = np.std(excess_returns) if len(excess_returns) > 1 else 0.0
        info_ratio = np.mean(excess_returns) / tracking_error if tracking_error > 0 else 0.0

        # Alpha quality score
        alpha_returns = [
            attr.components[ReturnComponent.ALPHA] for attr in asset_attributions.values()
        ]
        alpha_consistency = 1.0 - (
            np.std(alpha_returns) / max(0.01, np.mean(np.abs(alpha_returns)))

        return {
            "portfolio_volatility": portfolio_vol,
            "sharpe_ratio": sharpe,
            "information_ratio": info_ratio,
            "alpha_consistency": max(0.0, min(1.0, alpha_consistency)),
            "cost_efficiency": self._calculate_cost_efficiency(asset_attributions),
        }

    def _calculate_cost_efficiency(self, asset_attributions: Dict[str, ReturnAttribution]) -> float:
        """Calculate cost efficiency score."""

        total_alpha = 0.0
        total_costs = 0.0

        cost_components = [
            ReturnComponent.FEES,
            ReturnComponent.SLIPPAGE,
            ReturnComponent.MARKET_IMPACT,
            ReturnComponent.REBALANCING,
        ]

        for attribution in asset_attributions.values():
            total_alpha += attribution.components[ReturnComponent.ALPHA]

            for comp in cost_components:
                total_costs += abs(attribution.components[comp])

        # Cost efficiency = alpha / total_costs
        if total_costs > 0:
            efficiency = total_alpha / total_costs
        else:
            efficiency = 1.0 if total_alpha >= 0 else 0.0

        # Normalize to 0-1 scale
        return max(0.0, min(1.0, (efficiency + 1.0) / 2.0))

    def _get_price_at_time(self, price_data: pd.DataFrame, timestamp: datetime) -> Optional[float]:
        """Get price at specific timestamp."""
        if price_data.empty:
            return None

        try:
            # Find closest timestamp
            time_diff = abs(price_data.index - timestamp)
            closest_idx = time_diff.argmin()

            # Check if within reasonable time window (1 hour)
            if time_diff.iloc[closest_idx] <= timedelta(hours=1):
                return float(price_data.iloc[closest_idx]["close"])
            else:
                return None

        except Exception as e:
            self.logger.warning(f"Error getting price at time {timestamp}: {e}")
            return None

    def _empty_attribution(
        self, symbol: str, period_start: datetime, period_end: datetime
    ) -> ReturnAttribution:
        """Return empty attribution for asset."""
        return ReturnAttribution(
            symbol=symbol,
            period_start=period_start,
            period_end=period_end,
            total_return=0.0,
            components={comp: 0.0 for comp in ReturnComponent},
            confidence_scores={comp: 0.0 for comp in ReturnComponent},
            benchmark_return=self.benchmark_return,
            excess_return=-self.benchmark_return,
            attribution_quality=0.0,
            metadata={},
        )

    def _empty_portfolio_attribution(
        self, period_start: datetime, period_end: datetime
    ) -> PortfolioAttribution:
        """Return empty portfolio attribution."""
        return PortfolioAttribution(
            period_start=period_start,
            period_end=period_end,
            total_portfolio_return=0.0,
            benchmark_return=self.benchmark_return,
            excess_return=-self.benchmark_return,
            asset_attributions={},
            component_summary={comp: 0.0 for comp in ReturnComponent},
            risk_adjusted_metrics={},
            performance_quality=0.0,
            timestamp=datetime.utcnow(),
        )


def create_return_attributor(
    benchmark_return: float = 0.0, confidence_threshold: float = 0.7
) -> ReturnAttributor:
    """Create return attributor with specified parameters."""
    return ReturnAttributor(benchmark_return, confidence_threshold=confidence_threshold)
