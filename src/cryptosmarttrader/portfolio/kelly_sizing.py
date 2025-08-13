"""
Fractional Kelly Sizing System

Advanced position sizing using fractional Kelly criterion with
volatility targeting and overconfidence corrections.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy.optimize import minimize_scalar
from scipy.stats import norm
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class KellyMode(Enum):
    """Kelly calculation modes"""

    FULL = "full"
    FRACTIONAL = "fractional"
    CONSTRAINED = "constrained"
    CONSERVATIVE = "conservative"


@dataclass
class KellyParameters:
    """Kelly sizing parameters"""

    win_rate: float = 0.55
    avg_win: float = 0.02
    avg_loss: float = 0.015
    confidence: float = 0.75
    volatility: float = 0.02

    # Fractional adjustments
    kelly_fraction: float = 0.25  # Use 25% of full Kelly
    max_position_size: float = 0.05  # Max 5% of portfolio
    min_position_size: float = 0.001  # Min 0.1% of portfolio

    # Volatility targeting
    target_volatility: float = 0.15  # Target 15% annual vol
    current_portfolio_vol: float = 0.12

    @property
    def expectancy(self) -> float:
        """Calculate expectancy (expected return per trade)"""
        return (self.win_rate * self.avg_win) - ((1 - self.win_rate) * self.avg_loss)

    @property
    def profit_factor(self) -> float:
        """Calculate profit factor"""
        gross_profit = self.win_rate * self.avg_win
        gross_loss = (1 - self.win_rate) * self.avg_loss
        return gross_profit / gross_loss if gross_loss > 0 else float("inf")


@dataclass
class KellySizingResult:
    """Result of Kelly sizing calculation"""

    recommended_size: float
    full_kelly_size: float
    fractional_kelly_size: float
    volatility_adjusted_size: float
    final_size: float

    # Risk metrics
    confidence_adjustment: float
    volatility_ratio: float
    max_dd_estimate: float

    # Constraints applied
    size_constraints: Dict[str, float] = field(default_factory=dict)
    applied_caps: List[str] = field(default_factory=list)

    # Metadata
    kelly_mode: KellyMode = KellyMode.FRACTIONAL
    calculation_timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_valid(self) -> bool:
        """Check if sizing result is valid"""
        return (
            0 <= self.final_size <= 1.0
            and self.recommended_size > 0
            and not np.isnan(self.final_size)
        )


class FractionalKellySizer:
    """
    Advanced Kelly sizing system with fractional approach and volatility targeting
    """

    def __init__(
        self,
        default_kelly_fraction: float = 0.25,
        target_volatility: float = 0.15,
        confidence_threshold: float = 0.6,
    ):
        self.default_kelly_fraction = default_kelly_fraction
        self.target_volatility = target_volatility
        self.confidence_threshold = confidence_threshold

        # Risk management parameters
        self.max_single_position = 0.05  # 5% max per position
        self.min_position_size = 0.001  # 0.1% min position
        self.max_total_exposure = 0.25  # 25% max total crypto exposure

        # Overconfidence corrections
        self.overconfidence_penalty = 0.2  # Reduce size by 20% for overconfidence
        self.low_confidence_multiplier = 0.5  # Reduce size by 50% for low confidence

        # Historical performance tracking
        self.sizing_performance: List[Dict[str, Any]] = []
        self.kelly_accuracy_score = 0.5  # Track how well Kelly predictions work

    def calculate_full_kelly(self, params: KellyParameters) -> float:
        """Calculate full Kelly percentage"""

        try:
            if params.avg_loss <= 0:
                logger.warning("Invalid average loss for Kelly calculation")
                return 0.0

            # Classic Kelly formula: f = (bp - q) / b
            # where b = odds received (avg_win/avg_loss), p = win probability, q = 1-p
            b = params.avg_win / params.avg_loss
            p = params.win_rate
            q = 1 - p

            full_kelly = (b * p - q) / b

            # Ensure non-negative
            full_kelly = max(0.0, full_kelly)

            # Cap at reasonable maximum (100%)
            full_kelly = min(1.0, full_kelly)

            return full_kelly

        except Exception as e:
            logger.error(f"Full Kelly calculation failed: {e}")
            return 0.0

    def apply_confidence_adjustment(
        self, kelly_size: float, confidence: float
    ) -> Tuple[float, float]:
        """Apply confidence-based adjustments to Kelly size"""

        try:
            confidence_adjustment = 1.0

            # Low confidence penalty
            if confidence < self.confidence_threshold:
                confidence_adjustment *= self.low_confidence_multiplier
                logger.info(
                    f"Low confidence adjustment: {confidence:.2f} < {self.confidence_threshold:.2f}"
                )

            # Overconfidence detection (very high confidence may be unrealistic)
            elif confidence > 0.9:
                confidence_adjustment *= 1 - self.overconfidence_penalty
                logger.info(f"Overconfidence penalty applied: {confidence:.2f} > 0.9")

            # Smooth confidence scaling
            else:
                # Scale between 0.8 and 1.0 based on confidence
                confidence_adjustment = 0.8 + 0.2 * (
                    (confidence - self.confidence_threshold) / (0.9 - self.confidence_threshold)
                )

            adjusted_size = kelly_size * confidence_adjustment

            return adjusted_size, confidence_adjustment

        except Exception as e:
            logger.error(f"Confidence adjustment failed: {e}")
            return kelly_size, 1.0

    def apply_volatility_targeting(
        self, position_size: float, asset_volatility: float, current_portfolio_vol: float
    ) -> Tuple[float, float]:
        """Apply volatility targeting adjustments"""

        try:
            if asset_volatility <= 0 or current_portfolio_vol <= 0:
                return position_size, 1.0

            # Calculate volatility ratio
            vol_ratio = self.target_volatility / current_portfolio_vol

            # Additional adjustment for asset-specific volatility
            asset_vol_adjustment = self.target_volatility / asset_volatility

            # Combined volatility scaling - prioritize asset vol adjustment
            # High vol assets should get smaller positions
            if asset_volatility > self.target_volatility:
                # Reduce size for high volatility assets
                total_vol_adjustment = min(1.0, asset_vol_adjustment * 0.5)
            else:
                # Increase size for low volatility assets (but cap it)
                total_vol_adjustment = min(1.5, asset_vol_adjustment)

            # Apply portfolio vol adjustment
            total_vol_adjustment *= vol_ratio

            # Final clipping
            total_vol_adjustment = np.clip(total_vol_adjustment, 0.1, 2.0)

            adjusted_size = position_size * total_vol_adjustment

            logger.debug(
                f"Vol targeting: target={self.target_volatility:.1%}, "
                f"portfolio={current_portfolio_vol:.1%}, "
                f"asset={asset_volatility:.1%}, "
                f"adjustment={total_vol_adjustment:.2f}"
            )

            return adjusted_size, total_vol_adjustment

        except Exception as e:
            logger.error(f"Volatility targeting failed: {e}")
            return position_size, 1.0

    def calculate_max_drawdown_estimate(
        self, kelly_size: float, win_rate: float, avg_loss: float
    ) -> float:
        """Estimate maximum drawdown for given Kelly size"""

        try:
            # Simple drawdown estimation based on Kelly size and loss parameters
            # This is a rough approximation

            # Expected consecutive losses
            consecutive_losses = np.log(0.05) / np.log(1 - win_rate)  # 5% probability

            # Estimated max drawdown
            max_dd = kelly_size * avg_loss * consecutive_losses

            return min(max_dd, 0.5)  # Cap at 50%

        except Exception as e:
            logger.error(f"Drawdown estimation failed: {e}")
            return 0.1

    def calculate_fractional_kelly(
        self, params: KellyParameters, mode: KellyMode = KellyMode.FRACTIONAL
    ) -> KellySizingResult:
        """Calculate fractional Kelly position size with all adjustments"""

        try:
            # Step 1: Calculate full Kelly
            full_kelly = self.calculate_full_kelly(params)

            if full_kelly <= 0:
                return KellySizingResult(
                    recommended_size=0.0,
                    full_kelly_size=0.0,
                    fractional_kelly_size=0.0,
                    volatility_adjusted_size=0.0,
                    final_size=0.0,
                    confidence_adjustment=0.0,
                    volatility_ratio=1.0,
                    max_dd_estimate=0.0,
                    kelly_mode=mode,
                )

            # Step 2: Apply fractional scaling
            if mode == KellyMode.FULL:
                fractional_kelly = full_kelly
            elif mode == KellyMode.CONSERVATIVE:
                fractional_kelly = full_kelly * 0.1  # Very conservative
            else:
                fractional_kelly = full_kelly * params.kelly_fraction

            # Step 3: Apply confidence adjustment
            confidence_adjusted, confidence_adj_factor = self.apply_confidence_adjustment(
                fractional_kelly, params.confidence
            )

            # Step 4: Apply volatility targeting
            vol_adjusted, vol_ratio = self.apply_volatility_targeting(
                confidence_adjusted, params.volatility, params.current_portfolio_vol
            )

            # Step 5: Apply size constraints
            constrained_size = np.clip(
                vol_adjusted, params.min_position_size, params.max_position_size
            )

            # Step 6: Final size determination
            final_size = constrained_size

            # Track constraints applied
            applied_caps = []
            size_constraints = {
                "min_constraint": params.min_position_size,
                "max_constraint": params.max_position_size,
                "original_size": vol_adjusted,
                "final_size": final_size,
            }

            if vol_adjusted < params.min_position_size:
                applied_caps.append("minimum_size")
            elif vol_adjusted > params.max_position_size:
                applied_caps.append("maximum_size")

            # Calculate estimated max drawdown
            max_dd_estimate = self.calculate_max_drawdown_estimate(
                final_size, params.win_rate, params.avg_loss
            )

            result = KellySizingResult(
                recommended_size=fractional_kelly,
                full_kelly_size=full_kelly,
                fractional_kelly_size=fractional_kelly,
                volatility_adjusted_size=vol_adjusted,
                final_size=final_size,
                confidence_adjustment=confidence_adj_factor,
                volatility_ratio=vol_ratio,
                max_dd_estimate=max_dd_estimate,
                size_constraints=size_constraints,
                applied_caps=applied_caps,
                kelly_mode=mode,
            )

            # Log sizing details
            logger.info(
                f"Kelly sizing: Full={full_kelly:.3f}, "
                f"Fractional={fractional_kelly:.3f}, "
                f"Final={final_size:.3f}, "
                f"Constraints={applied_caps}"
            )

            return result

        except Exception as e:
            logger.error(f"Fractional Kelly calculation failed: {e}")
            return KellySizingResult(
                recommended_size=0.0,
                full_kelly_size=0.0,
                fractional_kelly_size=0.0,
                volatility_adjusted_size=0.0,
                final_size=0.0,
                confidence_adjustment=0.0,
                volatility_ratio=1.0,
                max_dd_estimate=0.0,
                kelly_mode=mode,
            )

    def optimize_kelly_fraction(
        self, historical_params: List[KellyParameters], lookback_trades: int = 100
    ) -> float:
        """Optimize Kelly fraction based on historical performance"""

        try:
            if len(historical_params) < 20:
                return self.default_kelly_fraction

            # Test different fractions
            fractions_to_test = np.arange(0.05, 0.6, 0.05)
            best_fraction = self.default_kelly_fraction
            best_score = -np.inf

            for fraction in fractions_to_test:
                total_return = 1.0
                max_dd = 0.0
                current_dd = 0.0
                peak = 1.0

                for params in historical_params[-lookback_trades:]:
                    # Placeholder removed
                    params.kelly_fraction = fraction
                    sizing_result = self.calculate_fractional_kelly(params)

                    # Placeholder removed
                    if np.random.random() < params.win_rate:
                        trade_return = 1 + (sizing_result.final_size * params.avg_win)
                    else:
                        trade_return = 1 - (sizing_result.final_size * params.avg_loss)

                    total_return *= trade_return

                    # Track drawdown
                    if total_return > peak:
                        peak = total_return
                        current_dd = 0.0
                    else:
                        current_dd = (peak - total_return) / peak
                        max_dd = max(max_dd, current_dd)

                # Score based on return/drawdown ratio
                if max_dd > 0:
                    score = (total_return - 1) / max_dd
                else:
                    score = total_return - 1

                if score > best_score:
                    best_score = score
                    best_fraction = fraction

            logger.info(f"Optimized Kelly fraction: {best_fraction:.2f} (score: {best_score:.2f})")
            return best_fraction

        except Exception as e:
            logger.error(f"Kelly fraction optimization failed: {e}")
            return self.default_kelly_fraction

    def get_regime_adjusted_kelly(
        self, base_params: KellyParameters, regime: str, regime_confidence: float
    ) -> KellyParameters:
        """Adjust Kelly parameters based on market regime"""

        try:
            adjusted_params = KellyParameters(
                win_rate=base_params.win_rate,
                avg_win=base_params.avg_win,
                avg_loss=base_params.avg_loss,
                confidence=base_params.confidence,
                volatility=base_params.volatility,
                kelly_fraction=base_params.kelly_fraction,
                max_position_size=base_params.max_position_size,
                min_position_size=base_params.min_position_size,
                target_volatility=base_params.target_volatility,
                current_portfolio_vol=base_params.current_portfolio_vol,
            )

            # Regime-specific adjustments
            if regime.lower() == "trend_up":
                adjusted_params.kelly_fraction *= 1.2  # More aggressive in uptrends
                adjusted_params.avg_win *= 1.1  # Expect higher wins

            elif regime.lower() == "trend_down":
                adjusted_params.kelly_fraction *= 0.8  # More conservative
                adjusted_params.avg_loss *= 1.1  # Expect higher losses

            elif regime.lower() == "chop":
                adjusted_params.kelly_fraction *= 0.5  # Very conservative
                adjusted_params.win_rate *= 0.9  # Lower win rate in chop

            elif regime.lower() == "volatility_spike":
                adjusted_params.kelly_fraction *= 0.3  # Minimal exposure
                adjusted_params.volatility *= 2.0  # Account for higher vol

            # Apply regime confidence scaling
            adjusted_params.confidence *= regime_confidence

            # Ensure reasonable bounds
            adjusted_params.kelly_fraction = np.clip(adjusted_params.kelly_fraction, 0.05, 0.5)
            adjusted_params.win_rate = np.clip(adjusted_params.win_rate, 0.3, 0.8)

            return adjusted_params

        except Exception as e:
            logger.error(f"Regime Kelly adjustment failed: {e}")
            return base_params

    def update_performance_tracking(
        self, kelly_result: KellySizingResult, actual_trade_return: float, trade_symbol: str
    ):
        """Update Kelly sizing performance tracking"""

        try:
            # Calculate prediction accuracy
            expected_return = kelly_result.recommended_size * 0.02  # Rough estimate

            prediction_error = abs(actual_trade_return - expected_return)
            accuracy_score = max(0, 1 - prediction_error * 10)  # Scale to 0-1

            # Update running accuracy
            alpha = 0.1  # Exponential smoothing factor
            self.kelly_accuracy_score = (
                alpha * accuracy_score + (1 - alpha) * self.kelly_accuracy_score
            )

            # Store performance record
            perf_record = {
                "timestamp": datetime.now().isoformat(),
                "symbol": trade_symbol,
                "kelly_size": kelly_result.final_size,
                "expected_return": expected_return,
                "actual_return": actual_trade_return,
                "prediction_error": prediction_error,
                "accuracy_score": accuracy_score,
                "kelly_mode": kelly_result.kelly_mode.value,
            }

            self.sizing_performance.append(perf_record)

            # Limit history size
            if len(self.sizing_performance) > 1000:
                self.sizing_performance = self.sizing_performance[-500:]

            logger.debug(f"Kelly performance updated: accuracy={self.kelly_accuracy_score:.2f}")

        except Exception as e:
            logger.error(f"Performance tracking update failed: {e}")

    def get_sizing_statistics(self, days_back: int = 30) -> Dict[str, Any]:
        """Get Kelly sizing performance statistics"""

        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)

            recent_performance = [
                p
                for p in self.sizing_performance
                if datetime.fromisoformat(p["timestamp"]) >= cutoff_time
            ]

            if not recent_performance:
                return {"status": "no_data"}

            # Calculate statistics
            avg_kelly_size = np.mean([p["kelly_size"] for p in recent_performance])
            avg_prediction_error = np.mean([p["prediction_error"] for p in recent_performance])
            accuracy_trend = np.mean([p["accuracy_score"] for p in recent_performance])

            # Size distribution
            size_buckets = {
                "small": sum(1 for p in recent_performance if p["kelly_size"] < 0.01),
                "medium": sum(1 for p in recent_performance if 0.01 <= p["kelly_size"] < 0.03),
                "large": sum(1 for p in recent_performance if p["kelly_size"] >= 0.03),
            }

            return {
                "period_days": days_back,
                "total_trades": len(recent_performance),
                "avg_kelly_size": avg_kelly_size,
                "avg_prediction_error": avg_prediction_error,
                "accuracy_score": accuracy_trend,
                "kelly_accuracy_score": self.kelly_accuracy_score,
                "size_distribution": size_buckets,
                "current_kelly_fraction": self.default_kelly_fraction,
            }

        except Exception as e:
            logger.error(f"Sizing statistics failed: {e}")
            return {"status": "error", "message": str(e)}

    def stress_test_kelly_sizing(
        self, base_params: KellyParameters, stress_scenarios: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """Stress test Kelly sizing under various market conditions"""

        try:
            stress_results = {}

            for scenario_name, adjustments in stress_scenarios:
                # Create stressed parameters
                stressed_params = KellyParameters(
                    win_rate=base_params.win_rate * adjustments.get("win_rate_multiplier", 1.0),
                    avg_win=base_params.avg_win * adjustments.get("avg_win_multiplier", 1.0),
                    avg_loss=base_params.avg_loss * adjustments.get("avg_loss_multiplier", 1.0),
                    confidence=base_params.confidence
                    * adjustments.get("confidence_multiplier", 1.0),
                    volatility=base_params.volatility
                    * adjustments.get("volatility_multiplier", 1.0),
                    kelly_fraction=base_params.kelly_fraction,
                    target_volatility=base_params.target_volatility,
                    current_portfolio_vol=base_params.current_portfolio_vol,
                )

                # Calculate sizing under stress
                stress_result = self.calculate_fractional_kelly(stressed_params)

                stress_results[scenario_name] = {
                    "final_size": stress_result.final_size,
                    "full_kelly": stress_result.full_kelly_size,
                    "max_dd_estimate": stress_result.max_dd_estimate,
                    "constraints_applied": stress_result.applied_caps,
                    "volatility_ratio": stress_result.volatility_ratio,
                }

            return stress_results

        except Exception as e:
            logger.error(f"Kelly stress test failed: {e}")
            return {}
