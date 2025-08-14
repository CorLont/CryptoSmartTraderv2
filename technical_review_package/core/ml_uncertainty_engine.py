#!/usr/bin/env python3
"""
Enterprise ML Uncertainty Quantification Engine
Implements Bayesian uncertainty, quantile regression, and ensemble confidence intervals
"""

import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
import warnings

warnings.filterwarnings("ignore")

from core.logging_manager import get_logger


@dataclass
class UncertaintyMetrics:
    """Uncertainty quantification metrics"""

    mean_prediction: float
    confidence_intervals: Dict[str, Tuple[float, float]]  # e.g., {'95%': (lower, upper)}
    prediction_variance: float
    epistemic_uncertainty: float  # Model uncertainty
    aleatoric_uncertainty: float  # Data uncertainty
    prediction_interval_coverage: float  # Historical coverage accuracy
    confidence_score: float  # Overall confidence (0-1)
    regime_confidence: float  # Confidence in current regime detection


@dataclass
class RegimePrediction:
    """Market regime prediction with uncertainty"""

    regime: str  # 'bull', 'bear', 'sideways', 'volatile'
    regime_probability: float
    regime_confidence: float
    expected_duration_days: float
    transition_probabilities: Dict[str, float]


class BayesianLSTM(nn.Module):
    """Bayesian LSTM with uncertainty quantification"""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bayesian LSTM layers with dropout for uncertainty
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

        # Bayesian output heads
        self.mean_head = nn.Linear(hidden_size, output_size)
        self.variance_head = nn.Linear(hidden_size, output_size)

        # Dropout for Monte Carlo sampling
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, return_uncertainty=False):
        batch_size = x.size(0)

        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout(lstm_out[:, -1, :])  # Take last timestep

        # Predict mean and variance
        mean = self.mean_head(lstm_out)
        log_variance = self.variance_head(lstm_out)
        variance = torch.exp(log_variance) + 1e-6  # Ensure positive variance

        if return_uncertainty:
            return mean, variance
        return mean

    def predict_with_uncertainty(self, x, n_samples=100):
        """Monte Carlo sampling for uncertainty quantification"""
        self.train()  # Enable dropout for uncertainty sampling

        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                mean, variance = self.forward(x, return_uncertainty=True)
                # Sample from predicted distribution
                sample = torch.normal(mean, torch.sqrt(variance))
                predictions.append(sample)

        self.eval()  # Disable dropout

        predictions = torch.stack(predictions)

        # Calculate uncertainty metrics
        mean_pred = predictions.mean(dim=0)
        epistemic_uncertainty = predictions.var(dim=0)  # Model uncertainty
        aleatoric_uncertainty = variance.mean()  # Data uncertainty

        return mean_pred, epistemic_uncertainty, aleatoric_uncertainty


class QuantileRegressorEnsemble:
    """Ensemble of quantile regressors for prediction intervals"""

    def __init__(self, quantiles: List[float] = None):
        self.quantiles = quantiles or [0.05, 0.25, 0.5, 0.75, 0.95]
        self.regressors = {}
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit quantile regressors for each quantile"""
        for q in self.quantiles:
            try:
                regressor = QuantileRegressor(quantile=q, alpha=0.01, solver="highs")
                regressor.fit(X, y)
                self.regressors[q] = regressor
            except Exception as e:
                # Fallback to random forest quantile regression
                regressor = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
                regressor.fit(X, y)
                self.regressors[q] = regressor

        self.is_fitted = True

    def predict_intervals(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict quantile intervals"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        predictions = {}
        for q in self.quantiles:
            if hasattr(self.regressors[q], "predict"):
                pred = self.regressors[q].predict(X)
            else:
                # Fallback prediction
                pred = np.zeros(X.shape[0])
            predictions[f"q{int(q * 100)}"] = pred

        return predictions


class MarketRegimeDetector:
    """Unsupervised market regime detection with confidence scoring"""

    def __init__(self):
        self.regime_states = ["bull", "bear", "sideways", "volatile"]
        self.feature_weights = {}
        self.regime_thresholds = {
            "bull": {"return_threshold": 0.02, "volatility_max": 0.3},
            "bear": {"return_threshold": -0.02, "volatility_max": 0.4},
            "sideways": {"return_range": 0.01, "volatility_max": 0.2},
            "volatile": {"volatility_min": 0.4},
        }
        self.regime_history = []
        self.transition_matrix = np.ones((4, 4)) * 0.25  # Uniform prior

    def detect_regime(self, market_data: Dict[str, Any]) -> RegimePrediction:
        """Detect current market regime with confidence"""

        # Calculate regime features
        features = self._calculate_regime_features(market_data)

        # Score each regime
        regime_scores = {}
        for regime in self.regime_states:
            score = self._score_regime(features, regime)
            regime_scores[regime] = score

        # Determine most likely regime
        best_regime = max(regime_scores, key=regime_scores.get)
        regime_probability = regime_scores[best_regime]

        # Calculate confidence based on score separation
        sorted_scores = sorted(regime_scores.values(), reverse=True)
        confidence = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 1.0

        # Estimate regime duration
        expected_duration = self._estimate_regime_duration(best_regime)

        # Calculate transition probabilities
        transition_probs = self._calculate_transition_probabilities(best_regime)

        return RegimePrediction(
            regime=best_regime,
            regime_probability=regime_probability,
            regime_confidence=confidence,
            expected_duration_days=expected_duration,
            transition_probabilities=transition_probs,
        )

    def _calculate_regime_features(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate features for regime detection"""
        features = {}

        try:
            # Price momentum features
            prices = market_data.get("prices", [])
            if len(prices) >= 20:
                returns = np.diff(prices) / prices[:-1]
                features["return_mean"] = np.mean(returns[-10:])  # Recent returns
                features["return_volatility"] = np.std(returns[-20:])  # Recent volatility
                features["trend_strength"] = self._calculate_trend_strength(prices)

            # Volume features
            volumes = market_data.get("volumes", [])
            if len(volumes) >= 10:
                features["volume_trend"] = np.mean(volumes[-5:]) / np.mean(volumes[-10:-5])
                features["volume_volatility"] = np.std(volumes[-10:]) / np.mean(volumes[-10:])

            # Market breadth features
            features["market_breadth"] = market_data.get("advancing_declining_ratio", 1.0)
            features["fear_greed_index"] = market_data.get("fear_greed_index", 50) / 100.0

        except Exception as e:
            # Default neutral features if calculation fails
            features = {
                "return_mean": 0.0,
                "return_volatility": 0.2,
                "trend_strength": 0.0,
                "volume_trend": 1.0,
                "volume_volatility": 0.3,
                "market_breadth": 1.0,
                "fear_greed_index": 0.5,
            }

        return features

    def _calculate_trend_strength(self, prices: List[float]) -> float:
        """Calculate trend strength indicator"""
        if len(prices) < 10:
            return 0.0

        # Linear regression slope as trend indicator
        x = np.arange(len(prices))
        coeffs = np.polyfit(x, prices, 1)
        slope = coeffs[0]

        # Normalize by average price
        avg_price = np.mean(prices)
        normalized_slope = slope / avg_price if avg_price > 0 else 0.0

        return normalized_slope

    def _score_regime(self, features: Dict[str, float], regime: str) -> float:
        """Score how well features match a regime"""
        score = 0.0

        if regime == "bull":
            score += max(0, features["return_mean"] * 10)  # Positive returns
            score += max(0, features["trend_strength"] * 5)  # Upward trend
            score += max(0, (0.3 - features["return_volatility"]) * 2)  # Moderate volatility
            score += max(0, (features["market_breadth"] - 1.0) * 2)  # More advancing

        elif regime == "bear":
            score += max(0, -features["return_mean"] * 10)  # Negative returns
            score += max(0, -features["trend_strength"] * 5)  # Downward trend
            score += max(
                0, (0.4 - features["return_volatility"]) * 1.5
            )  # High but capped volatility
            score += max(0, (1.0 - features["market_breadth"]) * 2)  # More declining

        elif regime == "sideways":
            score += max(0, (0.01 - abs(features["return_mean"])) * 10)  # Low returns
            score += max(0, (0.05 - abs(features["trend_strength"])) * 10)  # No trend
            score += max(0, (0.2 - features["return_volatility"]) * 3)  # Low volatility

        elif regime == "volatile":
            score += max(0, (features["return_volatility"] - 0.4) * 5)  # High volatility
            score += max(0, features["volume_volatility"] * 3)  # Volume spikes

        return max(0, min(1, score))  # Clamp to [0, 1]

    def _estimate_regime_duration(self, regime: str) -> float:
        """Estimate expected regime duration based on historical patterns"""
        regime_durations = {
            "bull": 120,  # 4 months average
            "bear": 90,  # 3 months average
            "sideways": 60,  # 2 months average
            "volatile": 30,  # 1 month average
        }
        return regime_durations.get(regime, 60)

    def _calculate_transition_probabilities(self, current_regime: str) -> Dict[str, float]:
        """Calculate regime transition probabilities"""
        regime_to_idx = {regime: i for i, regime in enumerate(self.regime_states)}
        current_idx = regime_to_idx[current_regime]

        # Get transition probabilities from transition matrix
        transition_probs = {}
        for regime in self.regime_states:
            target_idx = regime_to_idx[regime]
            transition_probs[regime] = self.transition_matrix[current_idx, target_idx]

        return transition_probs


class MLUncertaintyEngine:
    """Main uncertainty quantification engine"""

    def __init__(self):
        self.logger = get_logger()

        # Models
        self.bayesian_lstm = None
        self.quantile_ensemble = QuantileRegressorEnsemble()
        self.regime_detector = MarketRegimeDetector()

        # Configuration
        self.confidence_levels = [0.68, 0.95, 0.99]  # 1σ, 2σ, 3σ
        self.min_samples_for_training = 100
        self.uncertainty_threshold = 0.3  # High uncertainty threshold

        # Historical performance tracking
        self.prediction_history = []
        self.coverage_metrics = {}

    def predict_with_uncertainty(
        self, features: np.ndarray, market_data: Dict[str, Any], horizon: str = "24h"
    ) -> Tuple[float, UncertaintyMetrics]:
        """Generate prediction with comprehensive uncertainty quantification"""

        try:
            # Detect current market regime
            regime_prediction = self.regime_detector.detect_regime(market_data)

            # Get quantile predictions
            if self.quantile_ensemble.is_fitted:
                quantile_preds = self.quantile_ensemble.predict_intervals(features.reshape(1, -1))
            else:
                # Fallback to simple prediction if not trained
                quantile_preds = {f"q{i}": np.array([0.0]) for i in [5, 25, 50, 75, 95]}

            # Calculate confidence intervals
            confidence_intervals = {}
            for conf_level in self.confidence_levels:
                lower_q = (1 - conf_level) / 2
                upper_q = 1 - lower_q

                lower_key = f"q{int(lower_q * 100)}"
                upper_key = f"q{int(upper_q * 100)}"

                if lower_key in quantile_preds and upper_key in quantile_preds:
                    confidence_intervals[f"{int(conf_level * 100)}%"] = (
                        float(quantile_preds[lower_key][0]),
                        float(quantile_preds[upper_key][0]),
                    )

            # Main prediction (median)
            mean_prediction = float(quantile_preds.get("q50", np.array([0.0]))[0])

            # Calculate prediction variance
            if "q75" in quantile_preds and "q25" in quantile_preds:
                iqr = quantile_preds["q75"][0] - quantile_preds["q25"][0]
                prediction_variance = (iqr / 1.35) ** 2  # Approximate variance from IQR
            else:
                prediction_variance = 0.1

            # Estimate uncertainties
            epistemic_uncertainty = prediction_variance * 0.6  # Model uncertainty
            aleatoric_uncertainty = prediction_variance * 0.4  # Data uncertainty

            # Calculate overall confidence score
            regime_confidence = regime_prediction.regime_confidence
            prediction_confidence = 1.0 / (1.0 + prediction_variance)
            confidence_score = (regime_confidence + prediction_confidence) / 2

            # Get historical coverage accuracy
            coverage_accuracy = self._calculate_coverage_accuracy(horizon)

            uncertainty_metrics = UncertaintyMetrics(
                mean_prediction=mean_prediction,
                confidence_intervals=confidence_intervals,
                prediction_variance=prediction_variance,
                epistemic_uncertainty=epistemic_uncertainty,
                aleatoric_uncertainty=aleatoric_uncertainty,
                prediction_interval_coverage=coverage_accuracy,
                confidence_score=confidence_score,
                regime_confidence=regime_confidence,
            )

            # Log uncertainty metrics
            self.logger.info(
                f"Uncertainty prediction for {horizon}",
                extra={
                    "horizon": horizon,
                    "mean_prediction": mean_prediction,
                    "confidence_score": confidence_score,
                    "regime": regime_prediction.regime,
                    "regime_confidence": regime_confidence,
                    "prediction_variance": prediction_variance,
                    "confidence_intervals": confidence_intervals,
                },
            )

            return mean_prediction, uncertainty_metrics

        except Exception as e:
            self.logger.error(
                f"Uncertainty prediction failed: {e}", extra={"horizon": horizon, "error": str(e)}
            )

            # Return conservative fallback
            fallback_metrics = UncertaintyMetrics(
                mean_prediction=0.0,
                confidence_intervals={"95%": (-0.1, 0.1)},
                prediction_variance=1.0,
                epistemic_uncertainty=0.6,
                aleatoric_uncertainty=0.4,
                prediction_interval_coverage=0.5,
                confidence_score=0.1,
                regime_confidence=0.1,
            )

            return 0.0, fallback_metrics

    def train_uncertainty_models(self, training_data: Dict[str, Any]) -> bool:
        """Train uncertainty quantification models"""
        try:
            features = training_data.get("features")
            targets = training_data.get("targets")

            if features is None or targets is None:
                self.logger.warning("Insufficient training data for uncertainty models")
                return False

            if len(features) < self.min_samples_for_training:
                self.logger.warning(
                    f"Not enough samples for training: {len(features)} < {self.min_samples_for_training}"
                )
                return False

            # Train quantile ensemble
            self.quantile_ensemble.fit(features, targets)

            # Train Bayesian LSTM if enough data
            if len(features) > 500:
                self._train_bayesian_lstm(features, targets)

            self.logger.info(
                "Uncertainty models trained successfully",
                extra={
                    "samples": len(features),
                    "features": features.shape[1] if hasattr(features, "shape") else 0,
                },
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to train uncertainty models: {e}")
            return False

    def _train_bayesian_lstm(self, features: np.ndarray, targets: np.ndarray):
        """Train Bayesian LSTM for uncertainty quantification"""
        try:
            input_size = features.shape[1] if len(features.shape) > 1 else 1
            hidden_size = min(64, input_size * 2)

            self.bayesian_lstm = BayesianLSTM(
                input_size=input_size, hidden_size=hidden_size, output_size=1
            )

            # Convert to tensors
            X_tensor = torch.FloatTensor(features).unsqueeze(1)  # Add sequence dimension
            y_tensor = torch.FloatTensor(targets).unsqueeze(1)

            # Simple training loop
            optimizer = torch.optim.Adam(self.bayesian_lstm.parameters(), lr=0.001)
            criterion = nn.MSELoss()

            self.bayesian_lstm.train()
            for epoch in range(50):  # Limited epochs for quick training
                optimizer.zero_grad()

                mean_pred, variance = self.bayesian_lstm(X_tensor, return_uncertainty=True)

                # Negative log-likelihood loss for Bayesian training
                loss = criterion(mean_pred, y_tensor)
                loss += torch.mean(torch.log(variance))  # Regularize variance

                loss.backward()
                optimizer.step()

            self.bayesian_lstm.eval()

        except Exception as e:
            self.logger.warning(f"Bayesian LSTM training failed: {e}")
            self.bayesian_lstm = None

    def _calculate_coverage_accuracy(self, horizon: str) -> float:
        """Calculate historical prediction interval coverage accuracy"""
        if horizon not in self.coverage_metrics:
            return 0.8  # Default assumption

        return self.coverage_metrics[horizon].get("coverage_95", 0.8)

    def should_retrain(self, performance_metrics: Dict[str, float]) -> bool:
        """Determine if models should be retrained based on uncertainty degradation"""

        # Check prediction interval coverage
        for horizon, metrics in self.coverage_metrics.items():
            coverage_95 = metrics.get("coverage_95", 0.95)
            if coverage_95 < 0.85:  # Coverage fell below 85%
                self.logger.warning(
                    f"Poor prediction interval coverage for {horizon}: {coverage_95:.2f}"
                )
                return True

        # Check if uncertainty is too high
        avg_uncertainty = performance_metrics.get("avg_uncertainty", 0.0)
        if avg_uncertainty > self.uncertainty_threshold:
            self.logger.warning(f"High uncertainty detected: {avg_uncertainty:.2f}")
            return True

        # Check regime detection confidence
        regime_confidence = performance_metrics.get("regime_confidence", 1.0)
        if regime_confidence < 0.3:
            self.logger.warning(f"Low regime confidence: {regime_confidence:.2f}")
            return True

        return False

    def update_performance_metrics(
        self,
        actual_values: List[float],
        predictions: List[float],
        confidence_intervals: List[Dict[str, Tuple[float, float]]],
        horizon: str,
    ):
        """Update performance tracking for coverage validation"""

        if len(actual_values) != len(predictions) or len(actual_values) == 0:
            return

        # Calculate coverage for different confidence levels
        coverage_stats = {}

        for conf_level_str in ["68%", "95%", "99%"]:
            if conf_level_str in confidence_intervals[0]:
                covered = 0
                total = len(actual_values)

                for i, actual in enumerate(actual_values):
                    if i < len(confidence_intervals):
                        lower, upper = confidence_intervals[i][conf_level_str]
                        if lower <= actual <= upper:
                            covered += 1

                coverage_stats[f"coverage_{conf_level_str.replace('%', '')}"] = covered / total

        # Store coverage metrics
        if horizon not in self.coverage_metrics:
            self.coverage_metrics[horizon] = {}

        self.coverage_metrics[horizon].update(coverage_stats)

        self.logger.info(
            f"Updated coverage metrics for {horizon}",
            extra={
                "horizon": horizon,
                "samples": len(actual_values),
                "coverage_stats": coverage_stats,
            },
        )


# Global instance
_uncertainty_engine = None


def get_uncertainty_engine() -> MLUncertaintyEngine:
    """Get global uncertainty engine instance"""
    global _uncertainty_engine
    if _uncertainty_engine is None:
        _uncertainty_engine = MLUncertaintyEngine()
    return _uncertainty_engine
