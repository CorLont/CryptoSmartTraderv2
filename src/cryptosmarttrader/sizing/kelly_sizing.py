"""
Fractional Kelly Criterion Position Sizing

Implements optimal position sizing using Kelly criterion with risk controls:
f* = p - (1-p)/R  (Kelly formula)
size = k * f*     (fractional Kelly with k in [0.25..0.5])

Where:
- p = calibrated win probability
- R = expected payoff ratio (win/loss)
- k = fraction factor for risk control
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class KellyMode(Enum):
    """Kelly calculation modes"""

    CONSERVATIVE = "conservative"  # k = 0.25
    MODERATE = "moderate"  # k = 0.375
    AGGRESSIVE = "aggressive"  # k = 0.5


@dataclass
class KellyParameters:
    """Parameters for Kelly sizing calculation"""

    win_probability: float  # Calibrated probability of winning
    payoff_ratio: float  # Average win / Average loss
    fraction_factor: float  # Risk control factor (0.25-0.5)
    max_position_size: float  # Maximum allowed position (%)
    min_position_size: float  # Minimum position for execution (%)
    regime_adjustment: float  # Regime-based sizing adjustment
    confidence_threshold: float  # Minimum confidence to size position


class KellySizer:
    """
    Fractional Kelly position sizing with risk controls
    """

    def __init__(
        self,
        mode: KellyMode = KellyMode.MODERATE,
        max_position: float = 20.0,
        min_position: float = 0.5,
        confidence_threshold: float = 0.6,
    ):
        """
        Initialize Kelly sizer

        Args:
            mode: Risk level (conservative/moderate/aggressive)
            max_position: Maximum position size (%)
            min_position: Minimum position size for execution (%)
            confidence_threshold: Minimum confidence to size positions
        """
        self.mode = mode
        self.max_position = max_position
        self.min_position = min_position
        self.confidence_threshold = confidence_threshold

        # Set fraction factors by mode
        self.fraction_factors = {
            KellyMode.CONSERVATIVE: 0.25,
            KellyMode.MODERATE: 0.375,
            KellyMode.AGGRESSIVE: 0.5,
        }

        self.default_fraction = self.fraction_factors[mode]

        # Track sizing history for analysis
        self.sizing_history = []

    def calculate_kelly_size(
        self,
        win_probability: float,
        payoff_ratio: float,
        confidence: float = 1.0,
        regime_factor: float = 1.0,
        custom_fraction: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Calculate optimal position size using fractional Kelly

        Args:
            win_probability: Calibrated probability of winning (0-1)
            payoff_ratio: Expected win amount / Expected loss amount
            confidence: Model confidence in the prediction (0-1)
            regime_factor: Regime-based sizing adjustment (0.5-2.0)
            custom_fraction: Override default fraction factor

        Returns:
            Dict with sizing recommendation and rationale
        """
        try:
            # Input validation
            if not (0 <= win_probability <= 1):
                return self._invalid_input_response("Win probability must be 0-1")

            if payoff_ratio <= 0:
                return self._invalid_input_response("Payoff ratio must be positive")

            if not (0 <= confidence <= 1):
                confidence = max(0, min(1, confidence))
                logger.warning(f"Confidence clamped to valid range: {confidence}")

            # Check confidence threshold
            if confidence < self.confidence_threshold:
                return {
                    "position_size": 0.0,
                    "kelly_fraction": 0.0,
                    "raw_kelly": 0.0,
                    "reason": f"Confidence {confidence:.3f} below threshold {self.confidence_threshold}",
                    "should_trade": False,
                    "risk_level": "high",
                }

            # Calculate raw Kelly fraction
            raw_kelly = self._calculate_raw_kelly(win_probability, payoff_ratio)

            # Apply fractional Kelly for risk control
            fraction_factor = custom_fraction or self.default_fraction
            kelly_fraction = raw_kelly * fraction_factor

            # Apply regime and confidence adjustments
            adjusted_kelly = kelly_fraction * regime_factor * confidence

            # Apply position size limits
            final_size = self._apply_size_limits(adjusted_kelly)

            # Generate recommendation
            recommendation = self._generate_recommendation(
                final_size,
                raw_kelly,
                kelly_fraction,
                adjusted_kelly,
                win_probability,
                payoff_ratio,
                confidence,
                regime_factor,
            )

            # Store in history
            self._update_history(recommendation)

            return recommendation

        except Exception as e:
            logger.error(f"Kelly sizing calculation failed: {e}")
            return self._error_response(str(e))

    def calculate_batch_sizes(self, predictions: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Calculate position sizes for multiple predictions

        Args:
            predictions: List of dicts with keys: 'win_prob', 'payoff_ratio', 'confidence'

        Returns:
            List of sizing recommendations
        """
        recommendations = []

        for i, pred in enumerate(predictions):
            try:
                rec = self.calculate_kelly_size(
                    win_probability=pred.get("win_prob", 0.5),
                    payoff_ratio=pred.get("payoff_ratio", 1.0),
                    confidence=pred.get("confidence", 0.5),
                    regime_factor=pred.get("regime_factor", 1.0),
                )
                rec["prediction_id"] = i
                recommendations.append(rec)

            except Exception as e:
                logger.error(f"Batch sizing failed for prediction {i}: {e}")
                recommendations.append(self._error_response(f"Prediction {i}: {e}"))

        return recommendations

    def estimate_payoff_ratio(
        self, historical_returns: pd.Series, take_profit_pct: float, stop_loss_pct: float
    ) -> float:
        """
        Estimate payoff ratio from historical data and risk parameters

        Args:
            historical_returns: Historical trade returns (%)
            take_profit_pct: Take profit level (%)
            stop_loss_pct: Stop loss level (%)

        Returns:
            Estimated payoff ratio (average win / average loss)
        """
        try:
            if len(historical_returns) < 10:
                logger.warning("Limited historical data for payoff estimation")
                return take_profit_pct / stop_loss_pct  # Theoretical ratio

            # Separate wins and losses
            wins = historical_returns[historical_returns > 0]
            losses = historical_returns[historical_returns < 0]

            if len(wins) == 0 or len(losses) == 0:
                logger.warning("No wins or losses in historical data")
                return take_profit_pct / stop_loss_pct

            # Calculate average win and loss (absolute values)
            avg_win = wins.mean()
            avg_loss = abs(losses.mean())

            payoff_ratio = avg_win / avg_loss

            # Sanity check and bounds
            payoff_ratio = max(0.1, min(10.0, payoff_ratio))

            logger.info(
                f"Estimated payoff ratio: {payoff_ratio:.2f} "
                f"(avg win: {avg_win:.2f}%, avg loss: {avg_loss:.2f}%)"
            )

            return payoff_ratio

        except Exception as e:
            logger.error(f"Payoff ratio estimation failed: {e}")
            return take_profit_pct / stop_loss_pct  # Fallback to theoretical

    def optimize_fraction_factor(
        self,
        historical_predictions: List[Dict],
        historical_outcomes: List[float],
        test_fractions: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """
        Optimize fraction factor based on historical performance

        Args:
            historical_predictions: List of prediction dicts
            historical_outcomes: Corresponding trade outcomes (%)
            test_fractions: Fraction factors to test (default: [0.1, 0.25, 0.375, 0.5])

        Returns:
            Optimal fraction factor and performance metrics
        """
        try:
            if len(historical_predictions) != len(historical_outcomes):
                raise ValueError("Predictions and outcomes must have same length")

            if len(historical_predictions) < 20:
                logger.warning("Limited data for fraction optimization")

            test_fractions = test_fractions or [0.1, 0.25, 0.375, 0.5, 0.75]
            results = {}

            for fraction in test_fractions:
                # Calculate sizes for all historical predictions
                total_return = 0.0
                sharpe_numerator = 0.0
                returns = []

                for pred, outcome in zip(historical_predictions, historical_outcomes):
                    size_result = self.calculate_kelly_size(
                        win_probability=pred.get("win_prob", 0.5),
                        payoff_ratio=pred.get("payoff_ratio", 1.0),
                        confidence=pred.get("confidence", 0.5),
                        custom_fraction=fraction,
                    )

                    position_size = size_result["position_size"] / 100  # Convert to fraction
                    trade_return = position_size * outcome

                    total_return += trade_return
                    returns.append(trade_return)

                # Calculate performance metrics
                returns_array = np.array(returns)
                avg_return = np.mean(returns_array)
                volatility = np.std(returns_array)
                sharpe = avg_return / volatility if volatility > 0 else 0
                max_drawdown = self._calculate_max_drawdown(returns_array)

                results[fraction] = {
                    "total_return": total_return,
                    "avg_return": avg_return,
                    "volatility": volatility,
                    "sharpe_ratio": sharpe,
                    "max_drawdown": max_drawdown,
                    "win_rate": np.mean([r > 0 for r in returns]),
                }

            # Select best fraction (optimize Sharpe ratio with drawdown constraint)
            best_fraction = None
            best_score = -np.inf

            for fraction, metrics in results.items():
                # Composite score: Sharpe ratio penalized by drawdown
                score = metrics["sharpe_ratio"] * (1 - min(0.5, metrics["max_drawdown"]))

                if score > best_score:
                    best_score = score
                    best_fraction = fraction

            logger.info(
                f"Optimal fraction factor: {best_fraction} "
                f"(Sharpe: {results[best_fraction]['sharpe_ratio']:.3f})"
            )

            return {
                "optimal_fraction": best_fraction,
                "optimal_metrics": results[best_fraction],
                "all_results": results,
                "n_trades": len(historical_predictions),
            }

        except Exception as e:
            logger.error(f"Fraction optimization failed: {e}")
            return {"optimal_fraction": self.default_fraction, "error": str(e)}

    def _calculate_raw_kelly(self, win_prob: float, payoff_ratio: float) -> float:
        """Calculate raw Kelly fraction: f* = p - (1-p)/R"""
        try:
            # Kelly formula: f* = p - (1-p)/R
            kelly_fraction = win_prob - (1 - win_prob) / payoff_ratio

            # Kelly can be negative (don't trade) or > 1 (high leverage)
            # We'll handle limits in the fractional application
            return kelly_fraction

        except Exception as e:
            logger.error(f"Raw Kelly calculation failed: {e}")
            return 0.0

    def _apply_size_limits(self, kelly_size: float) -> float:
        """Apply position size limits"""
        try:
            # No negative positions (Kelly says don't trade)
            if kelly_size <= 0:
                return 0.0

            # Convert to percentage
            size_pct = kelly_size * 100

            # Apply limits
            size_pct = max(self.min_position, min(self.max_position, size_pct))

            return size_pct

        except Exception as e:
            logger.error(f"Size limit application failed: {e}")
            return 0.0

    def _generate_recommendation(
        self,
        final_size: float,
        raw_kelly: float,
        kelly_fraction: float,
        adjusted_kelly: float,
        win_prob: float,
        payoff_ratio: float,
        confidence: float,
        regime_factor: float,
    ) -> Dict[str, Any]:
        """Generate comprehensive sizing recommendation"""

        # Determine if should trade
        should_trade = final_size >= self.min_position

        # Risk assessment
        if raw_kelly <= 0:
            risk_level = "very_high"
            reason = "Negative Kelly - unfavorable odds"
        elif final_size == 0:
            risk_level = "high"
            reason = "Position size below minimum threshold"
        elif final_size >= self.max_position * 0.8:
            risk_level = "high"
            reason = "Large position near maximum limit"
        elif confidence < 0.7:
            risk_level = "medium"
            reason = "Moderate model confidence"
        else:
            risk_level = "low"
            reason = "Favorable Kelly sizing conditions"

        return {
            "position_size": final_size,
            "kelly_fraction": kelly_fraction,
            "raw_kelly": raw_kelly,
            "adjusted_kelly": adjusted_kelly,
            "should_trade": should_trade,
            "risk_level": risk_level,
            "reason": reason,
            "parameters": {
                "win_probability": win_prob,
                "payoff_ratio": payoff_ratio,
                "confidence": confidence,
                "regime_factor": regime_factor,
                "fraction_factor": self.default_fraction,
            },
            "limits": {
                "max_position": self.max_position,
                "min_position": self.min_position,
                "confidence_threshold": self.confidence_threshold,
            },
        }

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns"""
        try:
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = running_max - cumulative
            return np.max(drawdown)
        except Exception:
            return 0.0

    def _update_history(self, recommendation: Dict[str, Any]) -> None:
        """Update sizing history for analysis"""
        try:
            history_entry = {
                "timestamp": pd.Timestamp.now(),
                "position_size": recommendation["position_size"],
                "kelly_fraction": recommendation["kelly_fraction"],
                "should_trade": recommendation["should_trade"],
                "risk_level": recommendation["risk_level"],
                "win_probability": recommendation["parameters"]["win_probability"],
                "payoff_ratio": recommendation["parameters"]["payoff_ratio"],
                "confidence": recommendation["parameters"]["confidence"],
            }

            self.sizing_history.append(history_entry)

            # Keep only recent history (last 1000 entries)
            if len(self.sizing_history) > 1000:
                self.sizing_history = self.sizing_history[-1000:]

        except Exception as e:
            logger.error(f"History update failed: {e}")

    def _invalid_input_response(self, message: str) -> Dict[str, Any]:
        """Generate response for invalid inputs"""
        return {
            "position_size": 0.0,
            "kelly_fraction": 0.0,
            "raw_kelly": 0.0,
            "reason": f"Invalid input: {message}",
            "should_trade": False,
            "risk_level": "high",
        }

    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """Generate response for calculation errors"""
        return {
            "position_size": 0.0,
            "kelly_fraction": 0.0,
            "raw_kelly": 0.0,
            "reason": f"Calculation error: {error_msg}",
            "should_trade": False,
            "risk_level": "high",
            "error": error_msg,
        }

    def get_sizing_analytics(self) -> Dict[str, Any]:
        """Get analytics from sizing history"""
        if not self.sizing_history:
            return {"status": "No sizing history available"}

        try:
            df = pd.DataFrame(self.sizing_history)

            return {
                "total_recommendations": len(df),
                "avg_position_size": df["position_size"].mean(),
                "max_position_size": df["position_size"].max(),
                "trade_rate": df["should_trade"].mean(),
                "avg_kelly_fraction": df["kelly_fraction"].mean(),
                "risk_distribution": df["risk_level"].value_counts().to_dict(),
                "avg_confidence": df["confidence"].mean(),
                "avg_win_probability": df["win_probability"].mean(),
                "avg_payoff_ratio": df["payoff_ratio"].mean(),
                "recent_trend": {
                    "last_10_avg_size": df["position_size"].tail(10).mean(),
                    "last_10_trade_rate": df["should_trade"].tail(10).mean(),
                },
            }

        except Exception as e:
            logger.error(f"Analytics generation failed: {e}")
            return {"status": "Error", "error": str(e)}
