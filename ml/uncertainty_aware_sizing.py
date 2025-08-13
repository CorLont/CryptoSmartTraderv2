#!/usr/bin/env python3
"""
Uncertainty-Aware Position Sizing (Kelly-lite)
More capital in high-conviction trades, less in uncertain trades with correlation caps
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore")


@dataclass
class PositionSizeRecommendation:
    """Position sizing recommendation with risk metrics"""

    symbol: str
    recommended_size: float  # Fraction of portfolio (0-1)
    max_size: float  # Maximum allowed size
    confidence_adjusted_size: float  # Size adjusted for uncertainty
    risk_adjusted_size: float  # Final size after all adjustments
    kelly_fraction: float  # Theoretical Kelly fraction
    uncertainty_discount: float  # Discount factor due to uncertainty
    correlation_penalty: float  # Penalty due to correlation with existing positions
    liquidity_adjustment: float  # Adjustment for liquidity constraints
    reasoning: str  # Explanation of sizing decision


@dataclass
class PortfolioRiskMetrics:
    """Portfolio-level risk metrics"""

    total_exposure: float
    concentration_risk: float  # Herfindahl index
    correlation_risk: float  # Average pairwise correlation
    uncertainty_risk: float  # Portfolio-weighted uncertainty
    liquidity_risk: float  # Illiquidity exposure
    var_95: float  # 95% Value at Risk
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float


class KellyCalculator:
    """Calculate Kelly fraction for position sizing"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_kelly_fraction(
        self,
        expected_return: float,
        win_probability: float,
        volatility: float,
        uncertainty: float = 0.0,
    ) -> Dict[str, float]:
        """Calculate Kelly fraction with uncertainty adjustment"""

        # Basic Kelly formula: f = (bp - q) / b
        # where b = odds, p = win probability, q = lose probability

        if volatility <= 0 or win_probability <= 0 or win_probability >= 1:
            return {
                "kelly_fraction": 0.0,
                "adjusted_fraction": 0.0,
                "uncertainty_discount": 1.0,
                "confidence": 0.0,
            }

        # Convert to Kelly parameters
        # For continuous returns, approximate with win/loss scenario
        if expected_return > 0:
            # Bullish scenario
            win_amount = abs(expected_return) + volatility
            loss_amount = volatility

            # Kelly fraction
            kelly_f = (
                win_probability * win_amount - (1 - win_probability) * loss_amount
            ) / win_amount
        else:
            # Bearish scenario (short)
            win_amount = abs(expected_return) + volatility
            loss_amount = volatility

            # Reverse probabilities for short
            short_win_prob = 1 - win_probability
            kelly_f = (short_win_prob * win_amount - win_probability * loss_amount) / win_amount

        # Ensure Kelly fraction is reasonable
        kelly_f = max(0, min(kelly_f, 0.25))  # Cap at 25%

        # Uncertainty discount (reduce position when uncertain)
        uncertainty_discount = 1 - uncertainty
        adjusted_fraction = kelly_f * uncertainty_discount

        # Confidence in the calculation
        confidence = win_probability * (1 - uncertainty)

        return {
            "kelly_fraction": kelly_f,
            "adjusted_fraction": adjusted_fraction,
            "uncertainty_discount": uncertainty_discount,
            "confidence": confidence,
        }

    def calculate_fractional_kelly(self, kelly_fraction: float, fraction: float = 0.25) -> float:
        """Calculate fractional Kelly (more conservative)"""

        # Fractional Kelly reduces position size to reduce volatility
        return kelly_fraction * fraction


class CorrelationManager:
    """Manages portfolio correlation constraints"""

    def __init__(self, max_correlation: float = 0.7, correlation_window: int = 30):
        self.max_correlation = max_correlation
        self.correlation_window = correlation_window
        self.correlation_matrix = None
        self.logger = logging.getLogger(__name__)

    def update_correlations(self, returns_data: pd.DataFrame):
        """Update correlation matrix from returns data"""

        if len(returns_data) < self.correlation_window:
            self.logger.warning(
                f"Insufficient data for correlation calculation: {len(returns_data)} < {self.correlation_window}"
            )
            self.correlation_matrix = pd.DataFrame()
            return

        # Calculate rolling correlation matrix
        self.correlation_matrix = (
            returns_data.rolling(self.correlation_window, min_periods=self.correlation_window)
            .corr()
            .iloc[-len(returns_data.columns) :, :]
        )

    def calculate_correlation_penalty(
        self,
        new_symbol: str,
        existing_positions: Dict[str, float],
        returns_data: pd.DataFrame = None,
    ) -> float:
        """Calculate penalty factor for adding correlated position"""

        if (
            not existing_positions
            or self.correlation_matrix is None
            or self.correlation_matrix.empty
        ):
            return 0.0  # No penalty if no existing positions or correlation data

        if new_symbol not in self.correlation_matrix.index:
            return 0.0  # No penalty if correlation data not available

        # Calculate weighted correlation with existing positions
        weighted_correlation = 0.0
        total_weight = 0.0

        for symbol, position_size in existing_positions.items():
            if symbol in self.correlation_matrix.columns and symbol != new_symbol:
                correlation = self.correlation_matrix.loc[new_symbol, symbol]

                if not np.isnan(correlation):
                    weighted_correlation += abs(correlation) * position_size
                    total_weight += position_size

        if total_weight == 0:
            return 0.0

        avg_correlation = weighted_correlation / total_weight

        # Penalty increases non-linearly with correlation
        if avg_correlation > self.max_correlation:
            penalty = (avg_correlation - self.max_correlation) / (1 - self.max_correlation)
            penalty = min(penalty, 0.8)  # Cap penalty at 80%
        else:
            penalty = 0.0

        return penalty

    def get_portfolio_correlation_risk(self, positions: Dict[str, float]) -> float:
        """Calculate overall portfolio correlation risk"""

        if not positions or self.correlation_matrix is None or self.correlation_matrix.empty:
            return 0.0

        # Calculate weighted average correlation
        total_correlation = 0.0
        total_pairs = 0

        symbols = list(positions.keys())

        for i, symbol1 in enumerate(symbols):
            for j, symbol2 in enumerate(symbols):
                if (
                    i < j
                    and symbol1 in self.correlation_matrix.index
                    and symbol2 in self.correlation_matrix.columns
                ):
                    correlation = self.correlation_matrix.loc[symbol1, symbol2]

                    if not np.isnan(correlation):
                        weight = positions[symbol1] * positions[symbol2]
                        total_correlation += abs(correlation) * weight
                        total_pairs += weight

        return total_correlation / total_pairs if total_pairs > 0 else 0.0


class LiquidityManager:
    """Manages liquidity constraints for position sizing"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_liquidity_adjustment(
        self,
        symbol: str,
        position_size: float,
        volume_24h: float = None,
        market_cap: float = None,
        bid_ask_spread: float = None,
    ) -> Dict[str, float]:
        """Calculate position size adjustment for liquidity constraints"""

        # Default to no adjustment if no liquidity data
        if volume_24h is None:
            return {
                "liquidity_score": 0.5,
                "size_adjustment": 1.0,
                "max_position_pct": 0.1,
                "liquidity_warning": True,
            }

        # Volume-based liquidity score (higher is better)
        volume_score = min(1.0, volume_24h / 1000000)  # Normalize by $1M daily volume

        # Market cap based adjustment
        if market_cap is not None:
            mcap_score = min(1.0, market_cap / 100000000)  # Normalize by $100M market cap
        else:
            mcap_score = 0.5

        # Spread-based adjustment
        if bid_ask_spread is not None:
            spread_score = max(0.1, 1 - bid_ask_spread / 0.01)  # Penalize spreads > 1%
        else:
            spread_score = 0.5

        # Combined liquidity score
        liquidity_score = volume_score * 0.5 + mcap_score * 0.3 + spread_score * 0.2

        # Maximum position percentage based on liquidity
        if liquidity_score > 0.8:
            max_position_pct = 0.15  # 15% for highly liquid assets
        elif liquidity_score > 0.5:
            max_position_pct = 0.10  # 10% for moderately liquid
        elif liquidity_score > 0.2:
            max_position_pct = 0.05  # 5% for low liquidity
        else:
            max_position_pct = 0.02  # 2% for very illiquid

        # Size adjustment factor
        if position_size > max_position_pct:
            size_adjustment = max_position_pct / position_size
        else:
            size_adjustment = 1.0

        return {
            "liquidity_score": liquidity_score,
            "size_adjustment": size_adjustment,
            "max_position_pct": max_position_pct,
            "liquidity_warning": liquidity_score < 0.3,
        }


class UncertaintyAwarePortfolioSizer:
    """Main portfolio sizing system with uncertainty awareness"""

    def __init__(
        self,
        max_portfolio_risk: float = 0.15,
        max_single_position: float = 0.10,
        kelly_fraction: float = 0.25,
        min_confidence: float = 0.6,
    ):
        self.max_portfolio_risk = max_portfolio_risk
        self.max_single_position = max_single_position
        self.kelly_fraction = kelly_fraction
        self.min_confidence = min_confidence

        # Component managers
        self.kelly_calculator = KellyCalculator()
        self.correlation_manager = CorrelationManager()
        self.liquidity_manager = LiquidityManager()

        # Current portfolio state
        self.current_positions = {}
        self.current_returns_data = None

        self.logger = logging.getLogger(__name__)

    def update_portfolio_state(
        self, positions: Dict[str, float], returns_data: pd.DataFrame = None
    ):
        """Update current portfolio state"""

        self.current_positions = positions.copy()

        if returns_data is not None:
            self.current_returns_data = returns_data
            self.correlation_manager.update_correlations(returns_data)

    def calculate_position_size(
        self,
        symbol: str,
        expected_return: float,
        confidence: float,
        volatility: float,
        volume_24h: float = None,
        market_cap: float = None,
        bid_ask_spread: float = None,
    ) -> PositionSizeRecommendation:
        """Calculate optimal position size with uncertainty awareness"""

        # Step 1: Calculate Kelly fraction
        uncertainty = 1 - confidence
        kelly_result = self.kelly_calculator.calculate_kelly_fraction(
            expected_return, confidence, volatility, uncertainty
        )

        # Step 2: Apply fractional Kelly
        fractional_kelly = self.kelly_calculator.calculate_fractional_kelly(
            kelly_result["kelly_fraction"], self.kelly_fraction
        )

        # Step 3: Apply confidence adjustment
        confidence_adjusted_size = fractional_kelly * confidence

        # Step 4: Calculate correlation penalty
        correlation_penalty = self.correlation_manager.calculate_correlation_penalty(
            symbol, self.current_positions, self.current_returns_data
        )

        correlation_adjusted_size = confidence_adjusted_size * (1 - correlation_penalty)

        # Step 5: Apply liquidity constraints
        liquidity_result = self.liquidity_manager.calculate_liquidity_adjustment(
            symbol, correlation_adjusted_size, volume_24h, market_cap, bid_ask_spread
        )

        liquidity_adjusted_size = correlation_adjusted_size * liquidity_result["size_adjustment"]

        # Step 6: Apply portfolio constraints
        max_allowed = min(self.max_single_position, liquidity_result["max_position_pct"])
        final_size = min(liquidity_adjusted_size, max_allowed)

        # Step 7: Check minimum confidence threshold
        if confidence < self.min_confidence:
            final_size = 0.0
            reasoning = (
                f"Below minimum confidence threshold ({confidence:.3f} < {self.min_confidence})"
            )
        elif final_size == 0:
            reasoning = "Position size reduced to zero due to risk constraints"
        else:
            reasoning = f"Kelly: {fractional_kelly:.3f} → Conf: {confidence_adjusted_size:.3f} → Corr: {correlation_adjusted_size:.3f} → Liq: {liquidity_adjusted_size:.3f} → Final: {final_size:.3f}"

        recommendation = PositionSizeRecommendation(
            symbol=symbol,
            recommended_size=fractional_kelly,
            max_size=max_allowed,
            confidence_adjusted_size=confidence_adjusted_size,
            risk_adjusted_size=final_size,
            kelly_fraction=kelly_result["kelly_fraction"],
            uncertainty_discount=kelly_result["uncertainty_discount"],
            correlation_penalty=correlation_penalty,
            liquidity_adjustment=liquidity_result["size_adjustment"],
            reasoning=reasoning,
        )

        return recommendation

    def optimize_portfolio_allocation(
        self, opportunities: List[Dict[str, Any]], max_total_exposure: float = None
    ) -> Tuple[Dict[str, PositionSizeRecommendation], PortfolioRiskMetrics]:
        """Optimize allocation across multiple opportunities"""

        if max_total_exposure is None:
            max_total_exposure = self.max_portfolio_risk

        # Calculate individual position sizes
        recommendations = {}

        for opp in opportunities:
            symbol = opp["symbol"]
            expected_return = opp.get("expected_return", 0)
            confidence = opp.get("confidence", 0.5)
            volatility = opp.get("volatility", 0.05)
            volume_24h = opp.get("volume_24h")
            market_cap = opp.get("market_cap")
            bid_ask_spread = opp.get("bid_ask_spread")

            rec = self.calculate_position_size(
                symbol,
                expected_return,
                confidence,
                volatility,
                volume_24h,
                market_cap,
                bid_ask_spread,
            )

            recommendations[symbol] = rec

        # Portfolio-level optimization
        total_exposure = sum(rec.risk_adjusted_size for rec in recommendations.values())

        # Scale down if over-exposed
        if total_exposure > max_total_exposure:
            scale_factor = max_total_exposure / total_exposure

            for symbol, rec in recommendations.items():
                rec.risk_adjusted_size *= scale_factor
                rec.reasoning += f" → Scaled by {scale_factor:.3f} for portfolio limits"

        # Calculate portfolio risk metrics
        portfolio_metrics = self._calculate_portfolio_metrics(recommendations, opportunities)

        return recommendations, portfolio_metrics

    def _calculate_portfolio_metrics(
        self,
        recommendations: Dict[str, PositionSizeRecommendation],
        opportunities: List[Dict[str, Any]],
    ) -> PortfolioRiskMetrics:
        """Calculate portfolio-level risk metrics"""

        # Create opportunity lookup
        opp_lookup = {opp["symbol"]: opp for opp in opportunities}

        # Portfolio weights and properties
        weights = np.array([rec.risk_adjusted_size for rec in recommendations.values()])
        symbols = list(recommendations.keys())

        if len(weights) == 0 or np.sum(weights) == 0:
            return PortfolioRiskMetrics(
                total_exposure=0,
                concentration_risk=0,
                correlation_risk=0,
                uncertainty_risk=0,
                liquidity_risk=0,
                var_95=0,
                expected_return=0,
                expected_volatility=0,
                sharpe_ratio=0,
            )

        # Expected returns and volatilities
        returns = np.array([opp_lookup[symbol].get("expected_return", 0) for symbol in symbols])
        volatilities = np.array([opp_lookup[symbol].get("volatility", 0.05) for symbol in symbols])
        confidences = np.array([opp_lookup[symbol].get("confidence", 0.5) for symbol in symbols])

        # Portfolio metrics
        total_exposure = np.sum(weights)

        # Concentration risk (Herfindahl index)
        normalized_weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights
        concentration_risk = np.sum(normalized_weights**2)

        # Correlation risk
        correlation_risk = self.correlation_manager.get_portfolio_correlation_risk(
            dict(zip(symbols, weights))

        # Uncertainty risk (weighted average of uncertainties)
        uncertainty_risk = np.average(1 - confidences, weights=weights)

        # Liquidity risk (simplified)
        liquidity_scores = []
        for symbol in symbols:
            volume_24h = opp_lookup[symbol].get("volume_24h", 1000000)
            liq_result = self.liquidity_manager.calculate_liquidity_adjustment(
                symbol, 0.1, volume_24h
            )
            liquidity_scores.append(1 - liq_result["liquidity_score"])

        liquidity_risk = np.average(liquidity_scores, weights=weights)

        # Portfolio return and volatility
        portfolio_return = np.sum(weights * returns)

        # Simplified portfolio volatility (assuming some correlation)
        avg_correlation = min(0.5, correlation_risk)  # Cap correlation assumption
        portfolio_variance = np.sum((weights * volatilities) ** 2) + 2 * avg_correlation * np.sum(
            np.outer(weights * volatilities, weights * volatilities)
        portfolio_volatility = np.sqrt(max(0, portfolio_variance))

        # 95% VaR (assuming normal distribution)
        var_95 = portfolio_return - 1.645 * portfolio_volatility

        # Sharpe ratio
        sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0

        return PortfolioRiskMetrics(
            total_exposure=total_exposure,
            concentration_risk=concentration_risk,
            correlation_risk=correlation_risk,
            uncertainty_risk=uncertainty_risk,
            liquidity_risk=liquidity_risk,
            var_95=var_95,
            expected_return=portfolio_return,
            expected_volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
        )


def create_uncertainty_aware_sizer(
    max_portfolio_risk: float = 0.15,
    max_single_position: float = 0.10,
    kelly_fraction: float = 0.25,
) -> UncertaintyAwarePortfolioSizer:
    """Create uncertainty-aware portfolio sizer"""

    return UncertaintyAwarePortfolioSizer(
        max_portfolio_risk=max_portfolio_risk,
        max_single_position=max_single_position,
        kelly_fraction=kelly_fraction,
    )
