"""
Kelly Sizing & Portfolio Optimization for CryptoSmartTrader
Fractional Kelly with volatility targeting and advanced portfolio optimization.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

try:
    import scipy.optimize as opt
    from scipy.linalg import inv

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class PositionSizing:
    """Position sizing recommendation."""

    symbol: str
    kelly_fraction: float
    fractional_kelly: float
    volatility_adjusted: float
    final_allocation: float
    confidence: float
    risk_budget: float
    expected_return: float
    volatility: float
    sharpe_ratio: float
    metadata: Dict[str, Any]


@dataclass
class PortfolioAllocation:
    """Complete portfolio allocation."""

    allocations: Dict[str, PositionSizing]
    total_allocation: float
    portfolio_volatility: float
    portfolio_sharpe: float
    diversification_ratio: float
    risk_budget_used: float
    timestamp: datetime
    constraints_applied: List[str]


class KellyOptimizer:
    """
    Enterprise Kelly sizing with volatility targeting and portfolio optimization.

    Features:
    - Fractional Kelly with safety factors
    - Volatility targeting and scaling
    - Correlation-aware position sizing
    - Risk budgeting and cluster caps
    - Dynamic leverage adjustment
    - Uncertainty-aware sizing
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        volatility_target: float = 0.15,
        max_position_size: float = 0.20,
        max_leverage: float = 1.0,
        lookback_periods: int = 252,
    ):
        self.kelly_fraction = kelly_fraction  # Fractional Kelly multiplier
        self.volatility_target = volatility_target  # Portfolio vol target
        self.max_position_size = max_position_size  # Max single position
        self.max_leverage = max_leverage  # Maximum portfolio leverage
        self.lookback_periods = lookback_periods

        # Risk management parameters
        self.min_sharpe_threshold = 0.5
        self.max_correlation = 0.7
        self.uncertainty_penalty = 0.8

        self.logger = logging.getLogger(__name__)
        self.logger.info("KellyOptimizer initialized with enterprise risk controls")

    def calculate_kelly_sizing(
        self,
        expected_returns: Dict[str, float],
        volatilities: Dict[str, float],
        correlations: Optional[pd.DataFrame] = None,
        confidence_scores: Optional[Dict[str, float]] = None,
        current_positions: Optional[Dict[str, float]] = None,
    ) -> PortfolioAllocation:
        """
        Calculate optimal Kelly-based position sizing for portfolio.

        Args:
            expected_returns: Expected returns for each asset (annualized)
            volatilities: Volatilities for each asset (annualized)
            correlations: Correlation matrix between assets
            confidence_scores: Confidence in predictions (0-1)
            current_positions: Current position sizes (for rebalancing)

        Returns:
            PortfolioAllocation with optimal position sizes
        """
        symbols = list(expected_returns.keys())
        n_assets = len(symbols)

        if n_assets == 0:
            return self._empty_allocation()

        # Initialize confidence scores if not provided
        if confidence_scores is None:
            confidence_scores = {symbol: 1.0 for symbol in symbols}

        # Create correlation matrix if not provided
        if correlations is None:
            correlations = pd.DataFrame(np.eye(n_assets), index=symbols, columns=symbols)

        # Calculate individual Kelly fractions
        kelly_fractions = {}
        for symbol in symbols:
            kelly_fractions[symbol] = self._calculate_single_kelly(
                expected_returns[symbol], volatilities[symbol], confidence_scores[symbol]
            )

        # Calculate portfolio-level optimization
        position_sizings = self._optimize_portfolio(
            symbols,
            expected_returns,
            volatilities,
            correlations,
            kelly_fractions,
            confidence_scores,
            current_positions,
        )

        # Apply constraints and risk limits
        constrained_sizings, constraints = self._apply_constraints(position_sizings)

        # Calculate portfolio metrics
        portfolio_metrics = self._calculate_portfolio_metrics(
            constrained_sizings, volatilities, correlations
        )

        return PortfolioAllocation(
            allocations=constrained_sizings,
            total_allocation=sum(pos.final_allocation for pos in constrained_sizings.values()),
            portfolio_volatility=portfolio_metrics["volatility"],
            portfolio_sharpe=portfolio_metrics["sharpe"],
            diversification_ratio=portfolio_metrics["diversification_ratio"],
            risk_budget_used=portfolio_metrics["risk_budget_used"],
            timestamp=datetime.utcnow(),
            constraints_applied=constraints,
        )

    def _calculate_single_kelly(
        self, expected_return: float, volatility: float, confidence: float
    ) -> float:
        """Calculate Kelly fraction for single asset."""
        if volatility <= 0:
            return 0.0

        # Basic Kelly formula: f = μ / σ²
        kelly_raw = expected_return / (volatility**2)

        # Apply fractional Kelly
        kelly_fractional = kelly_raw * self.kelly_fraction

        # Apply confidence adjustment
        kelly_confident = kelly_fractional * confidence

        # Apply uncertainty penalty
        kelly_final = kelly_confident * self.uncertainty_penalty

        # Clamp to reasonable range
        return max(0.0, min(self.max_position_size, kelly_final))

    def _optimize_portfolio(
        self,
        symbols: List[str],
        expected_returns: Dict[str, float],
        volatilities: Dict[str, float],
        correlations: pd.DataFrame,
        kelly_fractions: Dict[str, float],
        confidence_scores: Dict[str, float],
        current_positions: Optional[Dict[str, float]],
    ) -> Dict[str, PositionSizing]:
        """Optimize portfolio allocations considering correlations."""

        position_sizings = {}

        for symbol in symbols:
            # Start with Kelly fraction
            kelly_size = kelly_fractions[symbol]

            # Adjust for volatility targeting
            vol_adjusted = self._volatility_target_adjustment(kelly_size, volatilities[symbol])

            # Calculate correlation penalty
            corr_penalty = self._calculate_correlation_penalty(
                symbol, correlations, kelly_fractions
            )

            # Apply correlation adjustment
            final_size = vol_adjusted * (1 - corr_penalty)

            # Calculate risk metrics
            expected_ret = expected_returns[symbol]
            vol = volatilities[symbol]
            sharpe = expected_ret / vol if vol > 0 else 0.0

            position_sizings[symbol] = PositionSizing(
                symbol=symbol,
                kelly_fraction=kelly_size,
                fractional_kelly=kelly_size * self.kelly_fraction,
                volatility_adjusted=vol_adjusted,
                final_allocation=final_size,
                confidence=confidence_scores[symbol],
                risk_budget=final_size * vol,
                expected_return=expected_ret,
                volatility=vol,
                sharpe_ratio=sharpe,
                metadata={
                    "correlation_penalty": corr_penalty,
                    "volatility_target_used": self.volatility_target,
                },
            )

        return position_sizings

    def _volatility_target_adjustment(self, position_size: float, asset_volatility: float) -> float:
        """Adjust position size for volatility targeting."""
        if asset_volatility <= 0:
            return 0.0

        # Scale position to achieve target portfolio volatility
        vol_scalar = self.volatility_target / asset_volatility

        # Apply scaling with reasonable limits
        vol_scalar = max(0.1, min(2.0, vol_scalar))

        return position_size * vol_scalar

    def _calculate_correlation_penalty(
        self, symbol: str, correlations: pd.DataFrame, kelly_fractions: Dict[str, float]
    ) -> float:
        """Calculate penalty for high correlations with other positions."""
        if symbol not in correlations.index:
            return 0.0

        total_penalty = 0.0
        for other_symbol, other_kelly in kelly_fractions.items():
            if symbol == other_symbol:
                continue

            if other_symbol in correlations.columns:
                correlation = correlations.loc[symbol, other_symbol]

                # Penalty increases with correlation and size of other position
                if abs(correlation) > self.max_correlation:
                    penalty = (abs(correlation) - self.max_correlation) * other_kelly
                    total_penalty += penalty

        # Cap maximum penalty
        return min(0.5, total_penalty)

    def _apply_constraints(
        self, position_sizings: Dict[str, PositionSizing]
    ) -> Tuple[Dict[str, PositionSizing], List[str]]:
        """Apply portfolio constraints and risk limits."""
        constrained_sizings = {}
        constraints_applied = []

        # Sort by Sharpe ratio for constraint application
        sorted_positions = sorted(
            position_sizings.items(), key=lambda x: x[1].sharpe_ratio, reverse=True
        )

        total_allocation = 0.0
        total_risk_budget = 0.0

        for symbol, sizing in sorted_positions:
            current_size = sizing.final_allocation

            # Apply position size limit
            if current_size > self.max_position_size:
                current_size = self.max_position_size
                constraints_applied.append(f"max_position_size_{symbol}")

            # Apply Sharpe threshold
            if sizing.sharpe_ratio < self.min_sharpe_threshold:
                current_size *= 0.5  # Reduce low Sharpe positions
                constraints_applied.append(f"low_sharpe_{symbol}")

            # Check portfolio leverage limit
            if total_allocation + current_size > self.max_leverage:
                current_size = max(0.0, self.max_leverage - total_allocation)
                constraints_applied.append("max_leverage")

            # Check volatility budget
            risk_contribution = current_size * sizing.volatility
            if total_risk_budget + risk_contribution > self.volatility_target * 2:
                # Scale down to fit risk budget
                available_risk = max(0.0, self.volatility_target * 2 - total_risk_budget)
                if sizing.volatility > 0:
                    current_size = min(current_size, available_risk / sizing.volatility)
                constraints_applied.append(f"risk_budget_{symbol}")

            # Update sizing with constraints
            constrained_sizing = PositionSizing(
                symbol=sizing.symbol,
                kelly_fraction=sizing.kelly_fraction,
                fractional_kelly=sizing.fractional_kelly,
                volatility_adjusted=sizing.volatility_adjusted,
                final_allocation=current_size,
                confidence=sizing.confidence,
                risk_budget=current_size * sizing.volatility,
                expected_return=sizing.expected_return,
                volatility=sizing.volatility,
                sharpe_ratio=sizing.sharpe_ratio,
                metadata=sizing.metadata,
            )

            constrained_sizings[symbol] = constrained_sizing
            total_allocation += current_size
            total_risk_budget += constrained_sizing.risk_budget

        return constrained_sizings, list(set(constraints_applied))

    def _calculate_portfolio_metrics(
        self,
        position_sizings: Dict[str, PositionSizing],
        volatilities: Dict[str, float],
        correlations: pd.DataFrame,
    ) -> Dict[str, float]:
        """Calculate portfolio-level risk and return metrics."""
        if not position_sizings:
            return {
                "volatility": 0.0,
                "sharpe": 0.0,
                "diversification_ratio": 1.0,
                "risk_budget_used": 0.0,
            }

        symbols = list(position_sizings.keys())
        weights = np.array([pos.final_allocation for pos in position_sizings.values()])

        # Portfolio expected return
        expected_returns = np.array([pos.expected_return for pos in position_sizings.values()])
        portfolio_return = np.sum(weights * expected_returns)

        # Portfolio volatility calculation
        if len(symbols) == 1:
            portfolio_vol = weights[0] * list(volatilities.values())[0]
        else:
            # Create covariance matrix
            vol_array = np.array([volatilities[s] for s in symbols])
            corr_matrix = correlations.loc[symbols, symbols].values
            cov_matrix = np.outer(vol_array, vol_array) * corr_matrix

            # Portfolio variance
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_vol = np.sqrt(portfolio_variance)

        # Portfolio Sharpe ratio
        portfolio_sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0.0

        # Diversification ratio
        weighted_vol = np.sum(weights * vol_array)
        diversification_ratio = weighted_vol / portfolio_vol if portfolio_vol > 0 else 1.0

        # Risk budget utilization
        total_risk_budget = sum(pos.risk_budget for pos in position_sizings.values())
        risk_budget_used = (
            total_risk_budget / self.volatility_target if self.volatility_target > 0 else 0.0
        )

        return {
            "volatility": float(portfolio_vol),
            "sharpe": float(portfolio_sharpe),
            "diversification_ratio": float(diversification_ratio),
            "risk_budget_used": float(risk_budget_used),
        }

    def _empty_allocation(self) -> PortfolioAllocation:
        """Return empty portfolio allocation."""
        return PortfolioAllocation(
            allocations={},
            total_allocation=0.0,
            portfolio_volatility=0.0,
            portfolio_sharpe=0.0,
            diversification_ratio=1.0,
            risk_budget_used=0.0,
            timestamp=datetime.utcnow(),
            constraints_applied=[],
        )

    def calculate_dynamic_leverage(
        self, market_regime: str, portfolio_sharpe: float, market_volatility: float
    ) -> float:
        """Calculate dynamic leverage based on market conditions."""

        base_leverage = self.max_leverage

        # Regime-based adjustment
        regime_multipliers = {
            "trending_up": 1.2,
            "trending_down": 0.8,
            "mean_reverting": 1.0,
            "choppy": 0.6,
            "high_volatility": 0.4,
            "low_volatility": 1.3,
        }

        regime_mult = regime_multipliers.get(market_regime, 1.0)

        # Sharpe-based adjustment
        sharpe_mult = min(1.5, max(0.5, 0.5 + portfolio_sharpe))

        # Volatility-based adjustment
        vol_mult = min(1.2, max(0.3, self.volatility_target / market_volatility))

        # Combined dynamic leverage
        dynamic_leverage = base_leverage * regime_mult * sharpe_mult * vol_mult

        return max(0.1, min(2.0, dynamic_leverage))

    def rebalance_portfolio(
        self,
        current_allocation: PortfolioAllocation,
        new_expected_returns: Dict[str, float],
        new_volatilities: Dict[str, float],
        transaction_cost: float = 0.001,
    ) -> PortfolioAllocation:
        """Rebalance portfolio considering transaction costs."""

        # Calculate new optimal allocation
        new_allocation = self.calculate_kelly_sizing(
            new_expected_returns,
            new_volatilities,
            current_positions={
                pos.symbol: pos.final_allocation for pos in current_allocation.allocations.values()
            },
        )

        # Calculate transaction costs for rebalancing
        rebalance_costs = {}
        for symbol in new_allocation.allocations:
            current_pos = current_allocation.allocations.get(symbol)
            current_size = current_pos.final_allocation if current_pos else 0.0
            new_size = new_allocation.allocations[symbol].final_allocation

            turnover = abs(new_size - current_size)
            cost = turnover * transaction_cost
            rebalance_costs[symbol] = cost

        # Adjust allocations for transaction costs
        for symbol, sizing in new_allocation.allocations.items():
            cost_penalty = rebalance_costs[symbol]
            # Reduce allocation slightly to account for transaction costs
            adjusted_allocation = max(0.0, sizing.final_allocation - cost_penalty)

            # Update sizing
            new_allocation.allocations[symbol] = PositionSizing(
                symbol=sizing.symbol,
                kelly_fraction=sizing.kelly_fraction,
                fractional_kelly=sizing.fractional_kelly,
                volatility_adjusted=sizing.volatility_adjusted,
                final_allocation=adjusted_allocation,
                confidence=sizing.confidence,
                risk_budget=adjusted_allocation * sizing.volatility,
                expected_return=sizing.expected_return,
                volatility=sizing.volatility,
                sharpe_ratio=sizing.sharpe_ratio,
                metadata={**sizing.metadata, "transaction_cost": cost_penalty},
            )

        return new_allocation


def create_kelly_optimizer(
    kelly_fraction: float = 0.25, volatility_target: float = 0.15, max_position_size: float = 0.20
) -> KellyOptimizer:
    """Create Kelly optimizer with specified parameters."""
    return KellyOptimizer(kelly_fraction, volatility_target, max_position_size)
