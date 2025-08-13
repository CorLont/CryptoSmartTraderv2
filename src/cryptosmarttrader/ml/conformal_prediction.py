#!/usr/bin/env python3
"""
Conformal Prediction System
Formal uncertainty quantification with adaptive confidence intervals for better risk gating
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
import warnings

warnings.filterwarnings("ignore")


@dataclass
class ConformalInterval:
    """Conformal prediction interval"""

    point_prediction: float
    lower_bound: float
    upper_bound: float
    interval_width: float
    confidence_level: float
    coverage_probability: float
    nonconformity_score: float


@dataclass
class CoverageResult:
    """Coverage validation result"""

    nominal_coverage: float
    empirical_coverage: float
    coverage_gap: float
    interval_efficiency: float
    is_well_calibrated: bool


class ConformalPredictor:
    """Conformal prediction wrapper for any scikit-learn compatible model"""

    def __init__(self, base_model: BaseEstimator, confidence_level: float = 0.8):
        self.base_model = base_model
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.calibration_scores = None
        self.is_fitted = False
        self.logger = logging.getLogger(__name__)

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_calib: pd.DataFrame, y_calib: pd.Series
    ) -> Dict[str, Any]:
        """Fit conformal predictor with proper train/calibration split"""

        # Fit base model on training data
        self.base_model.fit(X_train, y_train)

        # Calculate nonconformity scores on calibration set
        calib_predictions = self.base_model.predict(X_calib)
        self.calibration_scores = np.abs(y_calib - calib_predictions)

        # Calculate quantile for confidence intervals
        n_calib = len(self.calibration_scores)
        q_level = np.ceil((n_calib + 1) * (1 - self.alpha)) / n_calib
        q_level = min(q_level, 1.0)  # Cap at 1.0

        self.quantile = np.quantile(self.calibration_scores, q_level)
        self.is_fitted = True

        # Calculate training statistics
        train_mae = np.mean(np.abs(y_train - self.base_model.predict(X_train)))
        calib_mae = np.mean(np.abs(y_calib - calib_predictions))

        result = {
            "success": True,
            "train_samples": len(X_train),
            "calib_samples": len(X_calib),
            "quantile_level": q_level,
            "interval_quantile": self.quantile,
            "train_mae": train_mae,
            "calib_mae": calib_mae,
            "confidence_level": self.confidence_level,
        }

        self.logger.info(f"Conformal predictor fitted with {len(X_calib)} calibration samples")
        return result

    def predict_with_intervals(self, X: pd.DataFrame) -> List[ConformalInterval]:
        """Generate predictions with conformal intervals"""

        if not self.is_fitted:
            raise ValueError("Conformal predictor not fitted yet")

        # Get point predictions
        point_predictions = self.base_model.predict(X)

        intervals = []

        for pred in point_predictions:
            # Calculate conformal interval
            lower_bound = pred - self.quantile
            upper_bound = pred + self.quantile
            interval_width = upper_bound - lower_bound

            # Estimate coverage probability (heuristic based on interval width)
            coverage_prob = min(
                0.99,
                self.confidence_level
                + (1 - self.confidence_level)
                * np.exp(-interval_width / np.std(self.calibration_scores)),
            )

            # Calculate nonconformity score (prediction uncertainty)
            nonconformity = self.quantile / (np.std(self.calibration_scores) + 1e-6)

            interval = ConformalInterval(
                point_prediction=float(pred),
                lower_bound=float(lower_bound),
                upper_bound=float(upper_bound),
                interval_width=float(interval_width),
                confidence_level=self.confidence_level,
                coverage_probability=float(coverage_prob),
                nonconformity_score=float(nonconformity),
            )

            intervals.append(interval)

        return intervals

    def validate_coverage(self, X_test: pd.DataFrame, y_test: pd.Series) -> CoverageResult:
        """Validate coverage on test set"""

        intervals = self.predict_with_intervals(X_test)

        # Check empirical coverage
        covered = []
        interval_widths = []

        for i, interval in enumerate(intervals):
            true_value = y_test.iloc[i]
            is_covered = interval.lower_bound <= true_value <= interval.upper_bound
            covered.append(is_covered)
            interval_widths.append(interval.interval_width)

        empirical_coverage = np.mean(covered)
        coverage_gap = abs(empirical_coverage - self.confidence_level)
        interval_efficiency = 1 / (np.mean(interval_widths) + 1e-6)  # Prefer narrower intervals
        is_well_calibrated = coverage_gap < 0.05  # Within 5% tolerance

        result = CoverageResult(
            nominal_coverage=self.confidence_level,
            empirical_coverage=empirical_coverage,
            coverage_gap=coverage_gap,
            interval_efficiency=interval_efficiency,
            is_well_calibrated=is_well_calibrated,
        )

        return result


class AdaptiveConformalPredictor:
    """Adaptive conformal predictor that adjusts to changing distributions"""

    def __init__(
        self, base_model: BaseEstimator, confidence_level: float = 0.8, adaptation_window: int = 100
    ):
        self.base_model = base_model
        self.confidence_level = confidence_level
        self.adaptation_window = adaptation_window
        self.alpha = 1 - confidence_level

        # Rolling calibration data
        self.rolling_errors = []
        self.rolling_predictions = []
        self.rolling_targets = []

        self.is_fitted = False
        self.logger = logging.getLogger(__name__)

    def fit(
        self, X_train: pd.DataFrame, y_train: pd.Series, X_calib: pd.DataFrame, y_calib: pd.Series
    ) -> Dict[str, Any]:
        """Fit adaptive conformal predictor"""

        # Fit base model
        self.base_model.fit(X_train, y_train)

        # Initialize rolling calibration data
        calib_predictions = self.base_model.predict(X_calib)
        calib_errors = np.abs(y_calib - calib_predictions)

        # Initialize rolling buffers
        self.rolling_errors = list(calib_errors[-self.adaptation_window :])
        self.rolling_predictions = list(calib_predictions[-self.adaptation_window :])
        self.rolling_targets = list(y_calib.iloc[-self.adaptation_window :])

        self.is_fitted = True

        result = {
            "success": True,
            "initial_quantile": self._calculate_current_quantile(),
            "adaptation_window": self.adaptation_window,
            "calibration_samples": len(calib_errors),
        }

        return result

    def predict_with_adaptive_intervals(
        self, X: pd.DataFrame, update_with_feedback: bool = False, y_true: pd.Series = None
    ) -> List[ConformalInterval]:
        """Generate predictions with adaptive intervals"""

        if not self.is_fitted:
            raise ValueError("Adaptive conformal predictor not fitted yet")

        point_predictions = self.base_model.predict(X)
        intervals = []

        for i, pred in enumerate(point_predictions):
            # Calculate current adaptive quantile
            current_quantile = self._calculate_current_quantile()

            # Generate interval
            lower_bound = pred - current_quantile
            upper_bound = pred + current_quantile
            interval_width = upper_bound - lower_bound

            # Estimate coverage probability based on recent performance
            recent_coverage = self._estimate_recent_coverage()

            interval = ConformalInterval(
                point_prediction=float(pred),
                lower_bound=float(lower_bound),
                upper_bound=float(upper_bound),
                interval_width=float(interval_width),
                confidence_level=self.confidence_level,
                coverage_probability=float(recent_coverage),
                nonconformity_score=float(current_quantile),
            )

            intervals.append(interval)

            # Update rolling data if feedback provided
            if update_with_feedback and y_true is not None and i < len(y_true):
                true_value = y_true.iloc[i]
                error = abs(true_value - pred)

                self._update_rolling_data(pred, true_value, error)

        return intervals

    def _calculate_current_quantile(self) -> float:
        """Calculate current quantile based on rolling calibration data"""

        if len(self.rolling_errors) == 0:
            return 1.0

        n = len(self.rolling_errors)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)

        quantile = np.quantile(self.rolling_errors, q_level)
        return quantile

    def _estimate_recent_coverage(self) -> float:
        """Estimate coverage based on recent predictions"""

        if len(self.rolling_errors) < 10:
            return self.confidence_level

        # REMOVED: Mock data pattern not allowed in production
        recent_quantile = self._calculate_current_quantile()
        recent_intervals_covered = []

        for i in range(min(50, len(self.rolling_predictions))):
            pred = self.rolling_predictions[-(i + 1)]
            true_val = self.rolling_targets[-(i + 1)]

            lower = pred - recent_quantile
            upper = pred + recent_quantile
            covered = lower <= true_val <= upper
            recent_intervals_covered.append(covered)

        return (
            np.mean(recent_intervals_covered) if recent_intervals_covered else self.confidence_level
        )

    def _update_rolling_data(self, prediction: float, true_value: float, error: float):
        """Update rolling calibration data"""

        self.rolling_predictions.append(prediction)
        self.rolling_targets.append(true_value)
        self.rolling_errors.append(error)

        # Maintain window size
        if len(self.rolling_errors) > self.adaptation_window:
            self.rolling_errors.pop(0)
            self.rolling_predictions.pop(0)
            self.rolling_targets.pop(0)


class ConformalRiskGate:
    """Risk gate using conformal prediction intervals for better confidence gating"""

    def __init__(self, min_confidence: float = 0.8, max_interval_width: float = 0.2):
        self.min_confidence = min_confidence
        self.max_interval_width = max_interval_width
        self.logger = logging.getLogger(__name__)

    def apply_conformal_gate(
        self, intervals: List[ConformalInterval], additional_features: pd.DataFrame = None
    ) -> Dict[str, Any]:
        """Apply risk gate using conformal intervals"""

        passed_indices = []
        gate_scores = []
        rejection_reasons = []

        for i, interval in enumerate(intervals):
            rejection_reason = None

            # Check 1: Minimum coverage probability
            if interval.coverage_probability < self.min_confidence:
                rejection_reason = (
                    f"Low coverage: {interval.coverage_probability:.3f} < {self.min_confidence}"
                )

            # Check 2: Maximum interval width (uncertainty)
            elif interval.interval_width > self.max_interval_width:
                rejection_reason = f"High uncertainty: width {interval.interval_width:.3f} > {self.max_interval_width}"

            # Check 3: Reasonable prediction range
            elif abs(interval.point_prediction) > 1.0:  # More than 100% return
                rejection_reason = f"Unrealistic prediction: {interval.point_prediction:.3f}"

            # Calculate gate score (higher is better)
            gate_score = (
                interval.coverage_probability * 0.4
                + (1 / (interval.interval_width + 0.01)) * 0.3
                + (1 / (interval.nonconformity_score + 0.01)) * 0.3
            )

            gate_scores.append(gate_score)

            if rejection_reason is None:
                passed_indices.append(i)
                rejection_reasons.append(None)
            else:
                rejection_reasons.append(rejection_reason)

        # Calculate statistics
        pass_rate = len(passed_indices) / len(intervals) if intervals else 0
        avg_coverage = np.mean([i.coverage_probability for i in intervals]) if intervals else 0
        avg_width = np.mean([i.interval_width for i in intervals]) if intervals else 0

        result = {
            "passed_indices": passed_indices,
            "gate_scores": gate_scores,
            "rejection_reasons": rejection_reasons,
            "pass_rate": pass_rate,
            "total_predictions": len(intervals),
            "passed_predictions": len(passed_indices),
            "avg_coverage_probability": avg_coverage,
            "avg_interval_width": avg_width,
            "gate_effectiveness": pass_rate * avg_coverage,  # Combined metric
        }

        return result

    def create_conformal_features(self, intervals: List[ConformalInterval]) -> pd.DataFrame:
        """Create ML features from conformal intervals"""

        if not intervals:
            return pd.DataFrame()

        features_data = []

        for interval in intervals:
            features = {
                "conformal_prediction": interval.point_prediction,
                "conformal_lower": interval.lower_bound,
                "conformal_upper": interval.upper_bound,
                "conformal_width": interval.interval_width,
                "conformal_coverage": interval.coverage_probability,
                "conformal_uncertainty": interval.nonconformity_score,
                "conformal_asymmetry": abs(
                    interval.point_prediction - ((interval.lower_bound + interval.upper_bound) / 2)
                ),
                "conformal_normalized_width": interval.interval_width
                / (abs(interval.point_prediction) + 0.01),
                "conformal_confidence_score": interval.coverage_probability
                / (interval.interval_width + 0.01),
            }

            features_data.append(features)

        return pd.DataFrame(features_data)


class ConformalPredictionSystem:
    """Complete conformal prediction system for crypto trading"""

    def __init__(
        self, base_models: List[BaseEstimator], confidence_levels: List[float] = [0.8, 0.9]
    ):
        self.base_models = base_models
        self.confidence_levels = confidence_levels
        self.conformal_predictors = {}
        self.adaptive_predictors = {}
        self.risk_gates = {}
        self.logger = logging.getLogger(__name__)

        # Initialize predictors for each confidence level
        for conf_level in confidence_levels:
            self.risk_gates[conf_level] = ConformalRiskGate(
                min_confidence=conf_level,
                max_interval_width=0.3 - (conf_level - 0.8) * 0.5,  # Tighter for higher confidence
            )

    def train_conformal_system(
        self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.3
    ) -> Dict[str, Any]:
        """Train complete conformal prediction system"""

        # Split data: train/calib/test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )

        X_train, X_calib, y_train, y_calib = train_test_split(
            X_temp, y_temp, test_size=0.4, random_state=42, shuffle=False
        )

        training_results = {}

        # Train conformal predictors for each model and confidence level
        for model_idx, base_model in enumerate(self.base_models):
            model_name = f"model_{model_idx}_{base_model.__class__.__name__}"

            for conf_level in self.confidence_levels:
                predictor_key = f"{model_name}_{conf_level}"

                # Standard conformal predictor
                conformal_pred = ConformalPredictor(base_model, conf_level)
                fit_result = conformal_pred.fit(X_train, y_train, X_calib, y_calib)

                if fit_result["success"]:
                    self.conformal_predictors[predictor_key] = conformal_pred

                    # Validate coverage
                    coverage_result = conformal_pred.validate_coverage(X_test, y_test)

                    # Adaptive conformal predictor
                    adaptive_pred = AdaptiveConformalPredictor(base_model, conf_level)
                    adaptive_fit = adaptive_pred.fit(X_train, y_train, X_calib, y_calib)

                    if adaptive_fit["success"]:
                        self.adaptive_predictors[predictor_key] = adaptive_pred

                    training_results[predictor_key] = {
                        "fit_result": fit_result,
                        "coverage_validation": coverage_result,
                        "adaptive_fit": adaptive_fit,
                    }

        # Overall system statistics
        system_stats = {
            "total_predictors": len(self.conformal_predictors),
            "total_adaptive": len(self.adaptive_predictors),
            "training_samples": len(X_train),
            "calibration_samples": len(X_calib),
            "test_samples": len(X_test),
            "avg_coverage_gap": np.mean(
                [
                    result["coverage_validation"].coverage_gap
                    for result in training_results.values()
                    if "coverage_validation" in result
                ]
            )
            if training_results
            else 0,
        }

        result = {
            "success": True,
            "system_stats": system_stats,
            "predictor_results": training_results,
        }

        self.logger.info(
            f"Conformal system trained with {len(self.conformal_predictors)} predictors"
        )
        return result

    def predict_with_conformal_ensemble(
        self, X: pd.DataFrame, confidence_level: float = 0.8
    ) -> Tuple[List[ConformalInterval], Dict[str, Any]]:
        """Generate ensemble conformal predictions"""

        # Get all predictors for this confidence level
        relevant_predictors = {
            key: predictor
            for key, predictor in self.conformal_predictors.items()
            if f"_{confidence_level}" in key
        }

        if not relevant_predictors:
            raise ValueError(f"No predictors available for confidence level {confidence_level}")

        # Collect predictions from all models
        all_intervals = []

        for predictor_name, predictor in relevant_predictors.items():
            intervals = predictor.predict_with_intervals(X)
            all_intervals.append(intervals)

        # Ensemble intervals (average predictions, union of intervals)
        ensemble_intervals = []

        for i in range(len(X)):
            # Average point predictions
            point_preds = [intervals[i].point_prediction for intervals in all_intervals]
            avg_point_pred = np.mean(point_preds)

            # Union of intervals (widest coverage)
            lower_bounds = [intervals[i].lower_bound for intervals in all_intervals]
            upper_bounds = [intervals[i].upper_bound for intervals in all_intervals]

            ensemble_lower = min(lower_bounds)
            ensemble_upper = max(upper_bounds)
            ensemble_width = ensemble_upper - ensemble_lower

            # Average coverage probability
            coverage_probs = [intervals[i].coverage_probability for intervals in all_intervals]
            avg_coverage = np.mean(coverage_probs)

            # Ensemble nonconformity (average)
            nonconformities = [intervals[i].nonconformity_score for intervals in all_intervals]
            avg_nonconformity = np.mean(nonconformities)

            ensemble_interval = ConformalInterval(
                point_prediction=avg_point_pred,
                lower_bound=ensemble_lower,
                upper_bound=ensemble_upper,
                interval_width=ensemble_width,
                confidence_level=confidence_level,
                coverage_probability=avg_coverage,
                nonconformity_score=avg_nonconformity,
            )

            ensemble_intervals.append(ensemble_interval)

        # Apply risk gate
        risk_gate = self.risk_gates[confidence_level]
        gate_result = risk_gate.apply_conformal_gate(ensemble_intervals)

        ensemble_stats = {
            "models_used": len(relevant_predictors),
            "avg_interval_width": np.mean([i.interval_width for i in ensemble_intervals]),
            "avg_coverage": np.mean([i.coverage_probability for i in ensemble_intervals]),
            "gate_result": gate_result,
        }

        return ensemble_intervals, ensemble_stats


def create_conformal_prediction_system(
    base_models: List[BaseEstimator],
) -> ConformalPredictionSystem:
    """Create conformal prediction system with multiple models"""
    return ConformalPredictionSystem(base_models)
