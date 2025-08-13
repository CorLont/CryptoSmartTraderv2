#!/usr/bin/env python3
"""
Ensemble Optimizer - Advanced ensemble learning optimization for 500% target returns
Implements dynamic model weighting, regime-adaptive strategies, and enhanced confidence gating
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestRegressor, VotingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications"""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"


@dataclass
class ModelPerformance:
    """Track individual model performance metrics"""

    model_name: str
    accuracy_score: float
    confidence_score: float
    recent_returns: float
    regime_performance: Dict[str, float] = field(default_factory=dict)
    prediction_count: int = 0
    success_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0

    def update_performance(self, prediction_correct: bool, actual_return: float):
        """Update performance metrics with new prediction result"""
        self.prediction_count += 1
        self.success_rate = (
            self.success_rate * (self.prediction_count - 1) + (1.0 if prediction_correct else 0.0) / self.prediction_count

        # Update rolling returns
        self.recent_returns = (self.recent_returns * 0.9) + (actual_return * 0.1)


@dataclass
class EnsembleWeights:
    """Dynamic ensemble weights for different market conditions"""

    regime: MarketRegime
    model_weights: Dict[str, float]
    confidence_threshold: float
    volatility_adjustment: float
    timestamp: datetime

    def normalize_weights(self):
        """Normalize weights to sum to 1.0"""
        total = sum(self.model_weights.values())
        if total > 0:
            self.model_weights = {k: v / total for k, v in self.model_weights.items()}


class EnsembleOptimizer:
    """Advanced ensemble optimization for maximum returns"""

    def __init__(self, target_return: float = 5.0, max_risk: float = 0.15):
        """
        Initialize optimizer for 500% annual target (5x)

        Args:
            target_return: Target annual return multiple (5.0 = 500%)
            max_risk: Maximum acceptable portfolio volatility
        """
        self.target_return = target_return
        self.max_risk = max_risk
        self.logger = logger

        # Model performance tracking
        self.model_performances: Dict[str, ModelPerformance] = {}

        # Regime-specific weights
        self.regime_weights: Dict[MarketRegime, EnsembleWeights] = {}

        # Current market state
        self.current_regime = MarketRegime.SIDEWAYS
        self.current_volatility = 0.0

        # Optimization parameters
        self.min_confidence_threshold = 0.90  # Raised from 0.85 for higher quality
        self.dynamic_confidence_scaling = True
        self.performance_decay_factor = 0.95  # Weekly decay

        # Initialize regime-specific configurations
        self._initialize_regime_configurations()

        self.logger.info(f"EnsembleOptimizer initialized for {target_return}x target return")

    def _initialize_regime_configurations(self):
        """Initialize optimized configurations for each market regime"""

        # High-confidence momentum regime (best for crypto growth)
        self.regime_weights[MarketRegime.MOMENTUM] = EnsembleWeights(
            regime=MarketRegime.MOMENTUM,
            model_weights={
                "random_forest": 0.35,
                "xgboost": 0.30,
                "technical_analysis": 0.20,
                "sentiment_model": 0.15,
            },
            confidence_threshold=0.92,  # Very high confidence for momentum
            volatility_adjustment=1.2,  # Increase sizing in momentum
            timestamp=datetime.now(),
        )

        # Trending markets - follow the trend
        self.regime_weights[MarketRegime.TRENDING_UP] = EnsembleWeights(
            regime=MarketRegime.TRENDING_UP,
            model_weights={
                "random_forest": 0.30,
                "xgboost": 0.35,
                "technical_analysis": 0.25,
                "sentiment_model": 0.10,
            },
            confidence_threshold=0.88,
            volatility_adjustment=1.1,
            timestamp=datetime.now(),
        )

        # High volatility - focus on mean reversion and sentiment
        self.regime_weights[MarketRegime.HIGH_VOLATILITY] = EnsembleWeights(
            regime=MarketRegime.HIGH_VOLATILITY,
            model_weights={
                "random_forest": 0.25,
                "xgboost": 0.25,
                "technical_analysis": 0.20,
                "sentiment_model": 0.30,  # Higher weight on sentiment in volatile markets
            },
            confidence_threshold=0.95,  # Extra cautious in volatile markets
            volatility_adjustment=0.7,  # Reduce sizing
            timestamp=datetime.now(),
        )

        # Low volatility - careful momentum plays
        self.regime_weights[MarketRegime.LOW_VOLATILITY] = EnsembleWeights(
            regime=MarketRegime.LOW_VOLATILITY,
            model_weights={
                "random_forest": 0.40,
                "xgboost": 0.30,
                "technical_analysis": 0.25,
                "sentiment_model": 0.05,
            },
            confidence_threshold=0.85,  # Lower threshold in stable markets
            volatility_adjustment=1.3,  # Can size up in stable conditions
            timestamp=datetime.now(),
        )

        # Sideways/choppy markets - mean reversion focused
        self.regime_weights[MarketRegime.SIDEWAYS] = EnsembleWeights(
            regime=MarketRegime.SIDEWAYS,
            model_weights={
                "random_forest": 0.35,
                "xgboost": 0.30,
                "technical_analysis": 0.30,
                "sentiment_model": 0.05,
            },
            confidence_threshold=0.90,
            volatility_adjustment=0.8,
            timestamp=datetime.now(),
        )

        # Normalize all weights
        for regime_weight in self.regime_weights.values():
            regime_weight.normalize_weights()

    def detect_market_regime(self, market_data: pd.DataFrame) -> MarketRegime:
        """
        Detect current market regime from price data

        Args:
            market_data: DataFrame with OHLCV data

        Returns:
            Detected market regime
        """
        if len(market_data) < 50:
            return MarketRegime.SIDEWAYS

        try:
            # Calculate technical indicators for regime detection
            prices = market_data["close"].values
            returns = np.diff(np.log(prices))

            # Volatility measures
            volatility = np.std(returns) * np.sqrt(252)  # Annualized

            # Trend strength (using simple moving averages)
            short_ma = np.mean(prices[-10:])
            long_ma = np.mean(prices[-50:])
            trend_strength = (short_ma - long_ma) / long_ma

            # Momentum measure
            momentum_20 = (prices[-1] - prices[-20]) / prices[-20]
            momentum_5 = (prices[-1] - prices[-5]) / prices[-5]

            # Update current volatility
            self.current_volatility = volatility

            # Regime classification logic
            if volatility > 0.8:  # High volatility threshold for crypto
                regime = MarketRegime.HIGH_VOLATILITY
            elif volatility < 0.3:  # Low volatility
                regime = MarketRegime.LOW_VOLATILITY
            elif momentum_5 > 0.05 and momentum_20 > 0.1:  # Strong momentum up
                regime = MarketRegime.MOMENTUM
            elif trend_strength > 0.02:  # Trending up
                regime = MarketRegime.TRENDING_UP
            elif trend_strength < -0.02:  # Trending down
                regime = MarketRegime.TRENDING_DOWN
            else:  # Sideways/choppy
                regime = MarketRegime.SIDEWAYS

            self.current_regime = regime
            self.logger.info(f"Market regime detected: {regime.value} (vol: {volatility:.2f})")

            return regime

        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return MarketRegime.SIDEWAYS

    def calculate_dynamic_weights(
        self, model_predictions: Dict[str, Dict[str, Any]], market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate dynamic ensemble weights based on regime and performance

        Args:
            model_predictions: Dict of model predictions with confidence scores
            market_data: Recent market data for regime detection

        Returns:
            Optimized model weights
        """
        # Detect current market regime
        regime = self.detect_market_regime(market_data)

        # Get base weights for current regime
        base_weights = self.regime_weights.get(regime, self.regime_weights[MarketRegime.SIDEWAYS])

        # Adjust weights based on recent model performance
        adjusted_weights = {}

        for model_name, base_weight in base_weights.model_weights.items():
            if model_name in self.model_performances:
                performance = self.model_performances[model_name]

                # Performance adjustment factor
                perf_factor = 1.0

                # Recent accuracy boost/penalty
                if performance.success_rate > 0.6:
                    perf_factor *= 1.0 + (performance.success_rate - 0.6) * 0.5
                else:
                    perf_factor *= 0.5 + performance.success_rate

                # Confidence alignment
                if model_name in model_predictions:
                    model_confidence = model_predictions[model_name].get("confidence", 0.5)
                    if model_confidence > base_weights.confidence_threshold:
                        perf_factor *= 1.2  # Boost high-confidence predictions

                # Regime-specific performance
                regime_perf = performance.regime_performance.get(regime.value, 0.5)
                if regime_perf > 0.6:
                    perf_factor *= 1.0 + (regime_perf - 0.6) * 0.3

                adjusted_weights[model_name] = base_weight * perf_factor
            else:
                adjusted_weights[model_name] = base_weight

        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}

        return adjusted_weights

    def optimize_prediction_confidence(
        self, model_predictions: Dict[str, Dict[str, Any]], market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate optimized ensemble prediction with enhanced confidence scoring

        Args:
            model_predictions: Individual model predictions
            market_data: Market data for regime detection

        Returns:
            Optimized ensemble prediction
        """
        if not model_predictions:
            return {"prediction": 0.0, "confidence": 0.0, "regime": "unknown"}

        # Get dynamic weights
        weights = self.calculate_dynamic_weights(model_predictions, market_data)

        # Calculate weighted ensemble prediction
        weighted_prediction = 0.0
        weighted_confidence = 0.0
        total_valid_weight = 0.0

        for model_name, weight in weights.items():
            if model_name in model_predictions:
                pred_data = model_predictions[model_name]

                prediction = pred_data.get("prediction", 0.0)
                confidence = pred_data.get("confidence", 0.0)

                # Only include high-quality predictions
                min_threshold = self.regime_weights[self.current_regime].confidence_threshold

                if confidence >= min_threshold:
                    weighted_prediction += prediction * weight * confidence
                    weighted_confidence += confidence * weight
                    total_valid_weight += weight

        # Normalize by valid weights
        if total_valid_weight > 0:
            final_prediction = weighted_prediction / total_valid_weight
            final_confidence = weighted_confidence / total_valid_weight
        else:
            final_prediction = 0.0
            final_confidence = 0.0

        # Apply volatility-based position sizing adjustment
        volatility_adj = self.regime_weights[self.current_regime].volatility_adjustment

        # Enhanced confidence scoring for 500% target
        enhanced_confidence = self._calculate_enhanced_confidence(
            final_confidence, model_predictions, market_data
        )

        return {
            "prediction": final_prediction,
            "confidence": enhanced_confidence,
            "raw_confidence": final_confidence,
            "regime": self.current_regime.value,
            "volatility_adjustment": volatility_adj,
            "position_sizing_multiplier": volatility_adj * enhanced_confidence,
            "model_weights": weights,
            "models_used": len([m for m in model_predictions if weights.get(m, 0) > 0]),
        }

    def _calculate_enhanced_confidence(
        self,
        base_confidence: float,
        model_predictions: Dict[str, Dict[str, Any]],
        market_data: pd.DataFrame,
    ) -> float:
        """Calculate enhanced confidence for 500% target optimization"""

        if base_confidence < 0.85:
            return 0.0  # Hard filter for quality

        # Start with base confidence
        enhanced = base_confidence

        # Consensus bonus - reward model agreement
        predictions = [pred.get("prediction", 0) for pred in model_predictions.values()]
        if len(predictions) > 1:
            pred_std = np.std(predictions)
            pred_mean = np.mean(np.abs(predictions))

            if pred_mean > 0:
                consensus_factor = 1.0 - (pred_std / pred_mean)
                enhanced *= 1.0 + consensus_factor * 0.2  # Up to 20% bonus

        # Volume/momentum confirmation
        if len(market_data) > 5:
            try:
                recent_volume = market_data["volume"].iloc[-5:].mean()
                historical_volume = market_data["volume"].iloc[-50:-5].mean()

                if recent_volume > historical_volume * 1.5:  # Volume surge
                    enhanced *= 1.15  # 15% confidence boost

            except Exception:
                pass  # Skip volume analysis if data unavailable

        # Regime alignment bonus
        regime_threshold = self.regime_weights[self.current_regime].confidence_threshold
        if enhanced > regime_threshold:
            regime_bonus = (enhanced - regime_threshold) * 0.1
            enhanced += regime_bonus

        # Cap at reasonable maximum
        return min(enhanced, 0.98)

    def update_model_performance(
        self, model_name: str, prediction_correct: bool, actual_return: float, regime: MarketRegime
    ):
        """Update model performance tracking"""

        if model_name not in self.model_performances:
            self.model_performances[model_name] = ModelPerformance(
                model_name=model_name, accuracy_score=0.5, confidence_score=0.5, recent_returns=0.0
            )

        performance = self.model_performances[model_name]
        performance.update_performance(prediction_correct, actual_return)

        # Update regime-specific performance
        if regime.value not in performance.regime_performance:
            performance.regime_performance[regime.value] = 0.5

        # Rolling update for regime performance
        current_regime_perf = performance.regime_performance[regime.value]
        performance.regime_performance[regime.value] = (
            current_regime_perf * 0.9 + (1.0 if prediction_correct else 0.0) * 0.1
        )

        self.logger.info(f"Updated {model_name} performance: {performance.success_rate:.2%}")

    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization status report"""

        report = {
            "timestamp": datetime.now().isoformat(),
            "target_return": self.target_return,
            "current_regime": self.current_regime.value,
            "current_volatility": self.current_volatility,
            "confidence_threshold": self.regime_weights[self.current_regime].confidence_threshold,
            "model_performances": {},
        }

        # Add model performance details
        for model_name, performance in self.model_performances.items():
            report["model_performances"][model_name] = {
                "success_rate": performance.success_rate,
                "prediction_count": performance.prediction_count,
                "recent_returns": performance.recent_returns,
                "regime_performance": performance.regime_performance,
            }

        # Add regime configurations
        report["regime_configurations"] = {}
        for regime, weights in self.regime_weights.items():
            report["regime_configurations"][regime.value] = {
                "model_weights": weights.model_weights,
                "confidence_threshold": weights.confidence_threshold,
                "volatility_adjustment": weights.volatility_adjustment,
            }

        return report


def main():
    """Demo ensemble optimization"""
    print("🎯 ENSEMBLE OPTIMIZER FOR 500% TARGET")
    print("=" * 45)

    optimizer = EnsembleOptimizer(target_return=5.0)

    # Sample market data
    sample_data = pd.DataFrame(
        {
            "close": np.random.randn(100).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 100),
        }
    )

    # Sample predictions
    sample_predictions = {
        "random_forest": {"prediction": 0.05, "confidence": 0.88},
        "xgboost": {"prediction": 0.07, "confidence": 0.92},
        "technical_analysis": {"prediction": 0.04, "confidence": 0.85},
        "sentiment_model": {"prediction": 0.08, "confidence": 0.90},
    }

    # Get optimized prediction
    result = optimizer.optimize_prediction_confidence(sample_predictions, sample_data)

    print(f"Optimized Prediction: {result['prediction']:.2%}")
    print(f"Enhanced Confidence: {result['confidence']:.2%}")
    print(f"Market Regime: {result['regime']}")
    print(f"Position Sizing Multiplier: {result['position_sizing_multiplier']:.2f}")

    # Generate report
    report = optimizer.get_optimization_report()
    print(f"\nOptimization Report Generated: {len(report)} sections")


if __name__ == "__main__":
    main()
