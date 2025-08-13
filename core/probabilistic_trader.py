#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Probabilistic Trading Engine
Integration of uncertainty modeling with trading decisions
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import warnings

warnings.filterwarnings("ignore")

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .uncertainty_engine import UncertaintyEngine, UncertaintyPrediction, ConfidenceLevel


class TradeDecision(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    NO_TRADE = "no_trade"


class RiskLevel(Enum):
    VERY_LOW = 0.01  # 1% risk
    LOW = 0.02  # 2% risk
    MEDIUM = 0.05  # 5% risk
    HIGH = 0.10  # 10% risk
    VERY_HIGH = 0.20  # 20% risk


@dataclass
class TradingSignal:
    """Probabilistic trading signal with uncertainty"""

    symbol: str
    decision: TradeDecision
    confidence: float
    uncertainty: float
    predicted_return: float
    prediction_interval: Tuple[float, float]
    position_size: float
    risk_level: RiskLevel
    stop_loss: float
    take_profit: float
    reasoning: str
    timestamp: datetime
    model_consensus: float = 0.0

    # Uncertainty components
    epistemic_uncertainty: float = 0.0  # Model uncertainty
    aleatoric_uncertainty: float = 0.0  # Data noise
    total_uncertainty: float = 0.0

    # Risk metrics
    value_at_risk_95: float = 0.0
    expected_shortfall: float = 0.0
    sharpe_ratio_estimate: float = 0.0


@dataclass
class ProbabilisticTradingConfig:
    """Configuration for probabilistic trading"""

    # Confidence thresholds for different decisions
    strong_buy_confidence: float = 0.9
    buy_confidence: float = 0.75
    sell_confidence: float = 0.75
    strong_sell_confidence: float = 0.9

    # Uncertainty thresholds
    max_uncertainty_for_trade: float = 0.3
    uncertainty_penalty_factor: float = 2.0

    # Risk management
    base_position_size: float = 0.02  # 2% of portfolio
    max_position_size: float = 0.10  # 10% max
    min_position_size: float = 0.005  # 0.5% min

    # Stop loss and take profit
    dynamic_stop_loss: bool = True
    uncertainty_based_stops: bool = True
    min_risk_reward_ratio: float = 2.0

    # Portfolio level
    max_portfolio_uncertainty: float = 0.5
    correlation_penalty: bool = True

    # Model ensemble
    min_models_for_consensus: int = 3
    consensus_threshold: float = 0.7


class ProbabilisticTrader:
    """Probabilistic trading engine with uncertainty-aware decisions"""

    def __init__(self, config: Optional[ProbabilisticTradingConfig] = None):
        self.config = config or ProbabilisticTradingConfig()
        self.logger = logging.getLogger(f"{__name__}.ProbabilisticTrader")

        # Uncertainty engine
        self.uncertainty_engine = UncertaintyEngine()

        # Trading state
        self.active_signals: Dict[str, TradingSignal] = {}
        self.signal_history: List[TradingSignal] = []
        self.portfolio_uncertainty: float = 0.0

        # Model ensemble for consensus
        self.model_predictions: Dict[str, List[UncertaintyPrediction]] = {}

        # Performance tracking
        self.performance_metrics: Dict[str, float] = {}
        self.uncertainty_performance: Dict[str, List] = {
            "high_confidence_trades": [],
            "low_confidence_trades": [],
            "uncertainty_predictions": [],
            "actual_errors": [],
        }

        self._lock = threading.RLock()

        self.logger.info("Probabilistic Trading Engine initialized")

    def generate_trading_signal(
        self,
        symbol: str,
        predictions: List[UncertaintyPrediction],
        current_price: float,
        portfolio_context: Optional[Dict] = None,
    ) -> TradingSignal:
        """
        Generate trading signal from multiple uncertainty predictions

        Args:
            symbol: Trading symbol
            predictions: List of uncertainty predictions from different models
            current_price: Current market price
            portfolio_context: Portfolio state for risk management

        Returns:
            Probabilistic trading signal
        """
        with self._lock:
            try:
                # Aggregate predictions for consensus
                consensus_prediction = self._calculate_model_consensus(predictions)

                # Determine trade decision
                decision = self._determine_trade_decision(consensus_prediction)

                # Calculate position size with uncertainty adjustment
                position_size = self._calculate_position_size(
                    consensus_prediction, portfolio_context
                )

                # Calculate risk metrics
                risk_metrics = self._calculate_risk_metrics(consensus_prediction, current_price)

                # Generate stop loss and take profit levels
                stop_loss, take_profit = self._calculate_stop_levels(
                    consensus_prediction, current_price, decision
                )

                # Generate reasoning
                reasoning = self._generate_reasoning(
                    consensus_prediction, decision, len(predictions)

                # Create trading signal
                signal = TradingSignal(
                    symbol=symbol,
                    decision=decision,
                    confidence=consensus_prediction.confidence,
                    uncertainty=consensus_prediction.uncertainty,
                    predicted_return=consensus_prediction.mean_prediction,
                    prediction_interval=consensus_prediction.prediction_interval,
                    position_size=position_size,
                    risk_level=self._classify_risk_level(consensus_prediction.uncertainty),
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    reasoning=reasoning,
                    timestamp=datetime.now(),
                    model_consensus=len(predictions) / max(self.config.min_models_for_consensus, 1),
                    epistemic_uncertainty=consensus_prediction.epistemic_uncertainty,
                    aleatoric_uncertainty=consensus_prediction.aleatoric_uncertainty,
                    total_uncertainty=consensus_prediction.uncertainty,
                    value_at_risk_95=risk_metrics["var_95"],
                    expected_shortfall=risk_metrics["expected_shortfall"],
                    sharpe_ratio_estimate=risk_metrics["sharpe_estimate"],
                )

                # Store signal
                self.active_signals[symbol] = signal
                self.signal_history.append(signal)

                # Update portfolio uncertainty
                self._update_portfolio_uncertainty()

                return signal

            except Exception as e:
                self.logger.error(f"Signal generation failed for {symbol}: {e}")
                return self._create_no_trade_signal(symbol)

    def _calculate_model_consensus(
        self, predictions: List[UncertaintyPrediction]
    ) -> UncertaintyPrediction:
        """Calculate consensus from multiple model predictions"""
        if not predictions:
            raise ValueError("No predictions provided")

        # Extract values
        mean_preds = [p.mean_prediction for p in predictions]
        confidences = [p.confidence for p in predictions]
        uncertainties = [p.uncertainty for p in predictions]

        # Weighted average by confidence
        weights = np.array(confidences)
        weights = weights / np.sum(weights)

        consensus_mean = np.average(mean_preds, weights=weights)
        consensus_confidence = np.mean(confidences)

        # Uncertainty includes both average uncertainty and prediction disagreement
        avg_uncertainty = np.mean(uncertainties)
        prediction_disagreement = np.std(mean_preds)
        consensus_uncertainty = avg_uncertainty + prediction_disagreement

        # Prediction interval from ensemble
        lower_bounds = [p.prediction_interval[0] for p in predictions]
        upper_bounds = [p.prediction_interval[1] for p in predictions]
        consensus_interval = (np.min(lower_bounds), np.max(upper_bounds))

        # Aggregate quantiles if available
        consensus_quantiles = {}
        if predictions[0].quantile_predictions:
            for quantile in predictions[0].quantile_predictions.keys():
                quantile_values = [
                    p.quantile_predictions.get(quantile, consensus_mean) for p in predictions
                ]
                consensus_quantiles[quantile] = np.average(quantile_values, weights=weights)

        # Uncertainty decomposition
        epistemic_unc = np.mean([p.epistemic_uncertainty for p in predictions])
        aleatoric_unc = np.mean([p.aleatoric_uncertainty for p in predictions])

        return UncertaintyPrediction(
            mean_prediction=consensus_mean,
            confidence=consensus_confidence,
            uncertainty=consensus_uncertainty,
            prediction_interval=consensus_interval,
            quantile_predictions=consensus_quantiles,
            method_used=predictions[0].method_used,  # Use first method as representative
            timestamp=datetime.now(),
            epistemic_uncertainty=epistemic_unc,
            aleatoric_uncertainty=aleatoric_unc,
        )

    def _determine_trade_decision(self, prediction: UncertaintyPrediction) -> TradeDecision:
        """Determine trade decision based on prediction and uncertainty"""
        # Check if uncertainty is too high for any trade
        if prediction.uncertainty > self.config.max_uncertainty_for_trade:
            return TradeDecision.NO_TRADE

        predicted_return = prediction.mean_prediction
        confidence = prediction.confidence

        # Strong buy/sell require high confidence and significant predicted return
        if predicted_return > 0.05 and confidence >= self.config.strong_buy_confidence:
            return TradeDecision.STRONG_BUY
        elif predicted_return < -0.05 and confidence >= self.config.strong_sell_confidence:
            return TradeDecision.STRONG_SELL

        # Regular buy/sell with moderate confidence
        elif predicted_return > 0.02 and confidence >= self.config.buy_confidence:
            return TradeDecision.BUY
        elif predicted_return < -0.02 and confidence >= self.config.sell_confidence:
            return TradeDecision.SELL

        # Hold for uncertain or small predicted returns
        elif abs(predicted_return) <= 0.02 or confidence < 0.6:
            return TradeDecision.HOLD

        # Default to no trade if conditions don't match
        return TradeDecision.NO_TRADE

    def _calculate_position_size(
        self, prediction: UncertaintyPrediction, portfolio_context: Optional[Dict] = None
    ) -> float:
        """Calculate position size based on uncertainty and risk management"""
        base_size = self.config.base_position_size

        # Confidence scaling
        confidence_multiplier = prediction.confidence**2  # Square for more conservative scaling

        # Uncertainty penalty
        uncertainty_penalty = 1.0 / (
            1.0 + self.config.uncertainty_penalty_factor * prediction.uncertainty
        )

        # Prediction magnitude scaling
        return_magnitude = abs(prediction.mean_prediction)
        magnitude_multiplier = min(2.0, return_magnitude * 10)  # Cap at 2x

        # Kelly criterion approximation
        if prediction.uncertainty > 0:
            win_prob = prediction.confidence
            avg_win = abs(prediction.mean_prediction)
            avg_loss = prediction.uncertainty  # Approximate loss as uncertainty

            if avg_loss > 0:
                kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win
                kelly_fraction = max(0, min(0.25, kelly_fraction))  # Cap Kelly at 25%
            else:
                kelly_fraction = 0.1
        else:
            kelly_fraction = 0.05

        # Combine all factors
        size = (
            base_size
            * confidence_multiplier
            * uncertainty_penalty
            * magnitude_multiplier
            * kelly_fraction
        )

        # Apply portfolio-level constraints
        if portfolio_context:
            max_size = portfolio_context.get("max_position_size", self.config.max_position_size)
            portfolio_risk = portfolio_context.get("current_risk", 0.0)

            # Reduce size if portfolio risk is high
            if portfolio_risk > 0.5:
                size *= 1.0 - portfolio_risk

        # Enforce limits
        size = max(self.config.min_position_size, min(self.config.max_position_size, size))

        return size

    def _calculate_risk_metrics(
        self, prediction: UncertaintyPrediction, current_price: float
    ) -> Dict[str, float]:
        """Calculate risk metrics for the trade"""
        # Value at Risk (95% confidence level)
        if 0.05 in prediction.quantile_predictions:
            var_95 = abs(prediction.quantile_predictions[0.05] - prediction.mean_prediction)
        else:
            # Approximate from prediction interval
            var_95 = abs(prediction.prediction_interval[0] - prediction.mean_prediction)

        # Expected Shortfall (conditional VaR)
        expected_shortfall = var_95 * 1.3  # Approximation

        # Sharpe ratio estimate
        expected_return = prediction.mean_prediction
        volatility = prediction.uncertainty
        sharpe_estimate = expected_return / max(volatility, 0.01)

        return {
            "var_95": var_95,
            "expected_shortfall": expected_shortfall,
            "sharpe_estimate": sharpe_estimate,
        }

    def _calculate_stop_levels(
        self, prediction: UncertaintyPrediction, current_price: float, decision: TradeDecision
    ) -> Tuple[float, float]:
        """Calculate dynamic stop loss and take profit levels"""
        if decision in [TradeDecision.NO_TRADE, TradeDecision.HOLD]:
            return 0.0, 0.0

        # Base stop loss from uncertainty
        uncertainty_stop = prediction.uncertainty * 2.0  # 2 standard deviations

        # Minimum stop loss
        min_stop = 0.02  # 2%

        # Direction-based calculations
        is_long = decision in [TradeDecision.BUY, TradeDecision.STRONG_BUY]

        if is_long:
            # Long position
            stop_loss_pct = max(min_stop, uncertainty_stop)
            stop_loss = current_price * (1 - stop_loss_pct)

            # Take profit based on prediction and risk-reward ratio
            expected_return = prediction.mean_prediction
            take_profit_pct = max(
                expected_return, stop_loss_pct * self.config.min_risk_reward_ratio
            )
            take_profit = current_price * (1 + take_profit_pct)

        else:
            # Short position
            stop_loss_pct = max(min_stop, uncertainty_stop)
            stop_loss = current_price * (1 + stop_loss_pct)

            # Take profit for short
            expected_return = abs(prediction.mean_prediction)
            take_profit_pct = max(
                expected_return, stop_loss_pct * self.config.min_risk_reward_ratio
            )
            take_profit = current_price * (1 - take_profit_pct)

        return stop_loss, take_profit

    def _classify_risk_level(self, uncertainty: float) -> RiskLevel:
        """Classify risk level based on uncertainty"""
        if uncertainty < 0.1:
            return RiskLevel.VERY_LOW
        elif uncertainty < 0.2:
            return RiskLevel.LOW
        elif uncertainty < 0.3:
            return RiskLevel.MEDIUM
        elif uncertainty < 0.5:
            return RiskLevel.HIGH
        else:
            return RiskLevel.VERY_HIGH

    def _generate_reasoning(
        self, prediction: UncertaintyPrediction, decision: TradeDecision, num_models: int
    ) -> str:
        """Generate human-readable reasoning for the trade decision"""
        confidence_desc = (
            "high"
            if prediction.confidence > 0.8
            else "medium"
            if prediction.confidence > 0.6
            else "low"
        )
        uncertainty_desc = (
            "low"
            if prediction.uncertainty < 0.2
            else "medium"
            if prediction.uncertainty < 0.4
            else "high"
        )

        reasoning_parts = [
            f"{decision.value.replace('_', ' ').title()} decision based on:",
            f"• {confidence_desc} confidence ({prediction.confidence:.2f})",
            f"• {uncertainty_desc} uncertainty ({prediction.uncertainty:.2f})",
            f"• Predicted return: {prediction.mean_prediction:.2%}",
            f"• Model consensus from {num_models} models",
        ]

        if prediction.prediction_interval:
            interval_width = prediction.prediction_interval[1] - prediction.prediction_interval[0]
            reasoning_parts.append(f"• Prediction interval width: {interval_width:.2%}")

        return " ".join(reasoning_parts)

    def _create_no_trade_signal(self, symbol: str) -> TradingSignal:
        """Create a no-trade signal for error cases"""
        return TradingSignal(
            symbol=symbol,
            decision=TradeDecision.NO_TRADE,
            confidence=0.0,
            uncertainty=1.0,
            predicted_return=0.0,
            prediction_interval=(0.0, 0.0),
            position_size=0.0,
            risk_level=RiskLevel.VERY_HIGH,
            stop_loss=0.0,
            take_profit=0.0,
            reasoning="No trade due to insufficient or invalid predictions",
            timestamp=datetime.now(),
        )

    def _update_portfolio_uncertainty(self):
        """Update portfolio-level uncertainty"""
        if not self.active_signals:
            self.portfolio_uncertainty = 0.0
            return

        # Weight uncertainties by position size
        total_uncertainty = 0.0
        total_weight = 0.0

        for signal in self.active_signals.values():
            if signal.decision != TradeDecision.NO_TRADE:
                weight = signal.position_size
                total_uncertainty += signal.uncertainty * weight
                total_weight += weight

        self.portfolio_uncertainty = total_uncertainty / max(total_weight, 1e-8)

    def evaluate_signal_performance(
        self, symbol: str, actual_return: float, time_horizon_hours: int = 24
    ) -> Dict[str, Any]:
        """Evaluate trading signal performance against actual outcome"""
        if symbol not in self.active_signals:
            return {}

        signal = self.active_signals[symbol]

        # Prediction accuracy
        prediction_error = abs(signal.predicted_return - actual_return)

        # Was the prediction within the uncertainty bounds?
        within_interval = (
            signal.prediction_interval[0] <= actual_return <= signal.prediction_interval[1]
        )

        # Direction accuracy
        predicted_direction = (
            1 if signal.predicted_return > 0 else -1 if signal.predicted_return < 0 else 0
        )
        actual_direction = 1 if actual_return > 0 else -1 if actual_return < 0 else 0
        direction_correct = predicted_direction == actual_direction

        # Confidence calibration
        confidence_calibrated = (
            signal.confidence > 0.5 if direction_correct else signal.confidence <= 0.5
        )

        # Risk-adjusted performance
        if signal.uncertainty > 0:
            risk_adjusted_error = prediction_error / signal.uncertainty
        else:
            risk_adjusted_error = prediction_error

        # Store performance data
        performance = {
            "symbol": symbol,
            "predicted_return": signal.predicted_return,
            "actual_return": actual_return,
            "prediction_error": prediction_error,
            "confidence": signal.confidence,
            "uncertainty": signal.uncertainty,
            "within_interval": within_interval,
            "direction_correct": direction_correct,
            "confidence_calibrated": confidence_calibrated,
            "risk_adjusted_error": risk_adjusted_error,
            "time_horizon_hours": time_horizon_hours,
            "evaluation_time": datetime.now(),
        }

        # Update uncertainty performance tracking
        if signal.confidence > 0.8:
            self.uncertainty_performance["high_confidence_trades"].append(performance)
        else:
            self.uncertainty_performance["low_confidence_trades"].append(performance)

        self.uncertainty_performance["uncertainty_predictions"].append(signal.uncertainty)
        self.uncertainty_performance["actual_errors"].append(prediction_error)

        return performance

    def get_trading_summary(self) -> Dict[str, Any]:
        """Get comprehensive trading performance summary"""
        with self._lock:
            # Active signals summary
            active_summary = {
                "total_signals": len(self.active_signals),
                "by_decision": {},
                "average_confidence": 0.0,
                "average_uncertainty": 0.0,
                "portfolio_uncertainty": self.portfolio_uncertainty,
            }

            if self.active_signals:
                confidences = [s.confidence for s in self.active_signals.values()]
                uncertainties = [s.uncertainty for s in self.active_signals.values()]

                active_summary["average_confidence"] = np.mean(confidences)
                active_summary["average_uncertainty"] = np.mean(uncertainties)

                # Count by decision type
                for signal in self.active_signals.values():
                    decision = signal.decision.value
                    active_summary["by_decision"][decision] = (
                        active_summary["by_decision"].get(decision, 0) + 1
                    )

            # Historical performance
            historical_summary = {
                "total_signals_generated": len(self.signal_history),
                "high_confidence_trades": len(
                    self.uncertainty_performance["high_confidence_trades"]
                ),
                "low_confidence_trades": len(self.uncertainty_performance["low_confidence_trades"]),
            }

            # Performance metrics
            if self.uncertainty_performance["high_confidence_trades"]:
                high_conf_errors = [
                    t["prediction_error"]
                    for t in self.uncertainty_performance["high_confidence_trades"]
                ]
                historical_summary["high_confidence_avg_error"] = np.mean(high_conf_errors)

            if self.uncertainty_performance["low_confidence_trades"]:
                low_conf_errors = [
                    t["prediction_error"]
                    for t in self.uncertainty_performance["low_confidence_trades"]
                ]
                historical_summary["low_confidence_avg_error"] = np.mean(low_conf_errors)

            return {
                "active_signals": active_summary,
                "historical_performance": historical_summary,
                "uncertainty_performance": self.uncertainty_performance,
                "config": {
                    "trade_confidence_threshold": self.config.buy_confidence,
                    "max_uncertainty": self.config.max_uncertainty_for_trade,
                    "base_position_size": self.config.base_position_size,
                },
            }


# Singleton probabilistic trader
_probabilistic_trader = None
_trader_lock = threading.Lock()


def get_probabilistic_trader(
    config: Optional[ProbabilisticTradingConfig] = None,
) -> ProbabilisticTrader:
    """Get the singleton probabilistic trader"""
    global _probabilistic_trader

    with _trader_lock:
        if _probabilistic_trader is None:
            _probabilistic_trader = ProbabilisticTrader(config)
        return _probabilistic_trader


def generate_uncertainty_signal(
    symbol: str, predictions: List[UncertaintyPrediction], current_price: float
) -> TradingSignal:
    """Convenient function to generate trading signal with uncertainty"""
    trader = get_probabilistic_trader()
    return trader.generate_trading_signal(symbol, predictions, current_price)
