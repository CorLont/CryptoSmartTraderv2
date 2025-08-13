# ml/uncertainty_quantification.py - Advanced uncertainty with conformal prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class BayesianEnsemble(BaseEstimator, RegressorMixin):
    """Bayesian ensemble with Monte Carlo Dropout for uncertainty estimation"""

    def __init__(self, base_models: List, dropout_rate: float = 0.1, n_samples: int = 100):
        self.base_models = base_models
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples
        self.is_fitted = False

    def fit(self, X, y):
        """Fit all base models"""
        for model in self.base_models:
            model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_with_uncertainty(self, X) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation

        Returns:
            means: Point predictions
            epistemic_uncertainty: Model uncertainty (what we don't know)
            aleatoric_uncertainty: Data uncertainty (inherent randomness)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        predictions = []

        # Get predictions from all models
        for model in self.base_models:
            model_preds = []

            # Monte Carlo sampling for each model
            for _ in range(self.n_samples):
                # REMOVED: Mock data pattern not allowed in production
                X_dropped = self._apply_dropout(X)
                pred = model.predict(X_dropped)
                model_preds.append(pred)

            predictions.append(np.array(model_preds))

        # Convert to numpy array: (n_models, n_samples, n_observations)
        predictions = np.array(predictions)

        # Calculate epistemic uncertainty (across models)
        model_means = np.mean(predictions, axis=1)  # Mean across MC samples for each model
        epistemic_uncertainty = np.std(model_means, axis=0)  # Std across models

        # Calculate aleatoric uncertainty (within models)
        aleatoric_uncertainty = np.mean(
            np.std(predictions, axis=1), axis=0
        )  # Mean of std within models

        # Final predictions (mean across all)
        means = np.mean(model_means, axis=0)

        return means, epistemic_uncertainty, aleatoric_uncertainty

    def _apply_dropout(self, X: np.ndarray) -> np.ndarray:
        """Apply dropout to input features"""
        dropout_mask = np.random.random(X.shape) > self.dropout_rate
        return X * dropout_mask


class ConformalPredictor:
    """Conformal prediction for calibrated uncertainty intervals"""

    def __init__(self, base_model, alpha: float = 0.2):
        """
        Args:
            base_model: Base prediction model
            alpha: Miscoverage level (1-alpha = coverage probability)
        """
        self.base_model = base_model
        self.alpha = alpha
        self.calibration_scores = None
        self.is_calibrated = False

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """Calibrate the predictor using calibration set"""

        # Get predictions for calibration set
        if hasattr(self.base_model, "predict_with_uncertainty"):
            predictions, _, _ = self.base_model.predict_with_uncertainty(X_cal)
        else:
            predictions = self.base_model.predict(X_cal)

        # Calculate non-conformity scores (absolute residuals)
        self.calibration_scores = np.abs(y_cal - predictions)

        self.is_calibrated = True
        logger.info(f"Calibrated conformal predictor with {len(self.calibration_scores)} samples")

    def predict_with_intervals(self, X: np.ndarray) -> Dict:
        """
        Generate predictions with conformal prediction intervals

        Returns:
            Dictionary with predictions, lower/upper bounds, and interval width
        """
        if not self.is_calibrated:
            raise ValueError("Predictor not calibrated. Call calibrate() first.")

        # Get base predictions
        if hasattr(self.base_model, "predict_with_uncertainty"):
            predictions, epistemic_unc, aleatoric_unc = self.base_model.predict_with_uncertainty(X)
            total_uncertainty = np.sqrt(epistemic_unc**2 + aleatoric_unc**2)
        else:
            predictions = self.base_model.predict(X)
            epistemic_unc = np.zeros_like(predictions)
            aleatoric_unc = np.zeros_like(predictions)
            total_uncertainty = np.zeros_like(predictions)

        # Calculate conformal quantile
        n = len(self.calibration_scores)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        quantile = np.quantile(self.calibration_scores, quantile_level)

        # Create prediction intervals
        lower_bound = predictions - quantile
        upper_bound = predictions + quantile
        interval_width = 2 * quantile

        return {
            "predictions": predictions,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "interval_width": interval_width,
            "epistemic_uncertainty": epistemic_unc,
            "aleatoric_uncertainty": aleatoric_unc,
            "total_uncertainty": total_uncertainty,
            "coverage_probability": 1 - self.alpha,
        }


class EnhancedUncertaintyQuantifier:
    """Enhanced uncertainty quantification combining multiple methods"""

    def __init__(self, models: List, calibration_split: float = 0.2):
        self.models = models
        self.calibration_split = calibration_split
        self.bayesian_ensemble = None
        self.conformal_predictor = None
        self.is_trained = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit uncertainty quantification models"""

        # Split data for calibration
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=self.calibration_split, random_state=42
        )

        # Create Bayesian ensemble
        self.bayesian_ensemble = BayesianEnsemble(self.models)
        self.bayesian_ensemble.fit(X_train, y_train)

        # Create and calibrate conformal predictor
        self.conformal_predictor = ConformalPredictor(self.bayesian_ensemble)
        self.conformal_predictor.calibrate(X_cal, y_cal)

        self.is_trained = True
        logger.info("Enhanced uncertainty quantification model trained")

    def predict_with_full_uncertainty(self, X: np.ndarray) -> Dict:
        """Generate predictions with comprehensive uncertainty estimates"""

        if not self.is_trained:
            raise ValueError("Model not trained. Call fit() first.")

        # Get conformal predictions (includes Bayesian uncertainty)
        conformal_results = self.conformal_predictor.predict_with_intervals(X)

        # Calculate confidence scores based on uncertainty
        total_unc = conformal_results["total_uncertainty"]
        interval_widths = conformal_results["interval_width"]

        # Normalize uncertainties to get confidence (higher uncertainty = lower confidence)
        max_unc = np.percentile(total_unc, 95)  # Use 95th percentile for normalization
        max_width = np.percentile(interval_widths, 95)

        # Combine multiple uncertainty measures for confidence
        unc_confidence = 1 - np.clip(total_unc / max_unc, 0, 1)
        interval_confidence = 1 - np.clip(interval_widths / max_width, 0, 1)

        # Geometric mean of confidences
        combined_confidence = np.sqrt(unc_confidence * interval_confidence)

        # Calibrate confidence to be more conservative
        calibrated_confidence = self._calibrate_confidence(combined_confidence)

        return {
            "predictions": conformal_results["predictions"],
            "confidence": calibrated_confidence,
            "epistemic_uncertainty": conformal_results["epistemic_uncertainty"],
            "aleatoric_uncertainty": conformal_results["aleatoric_uncertainty"],
            "prediction_intervals": {
                "lower": conformal_results["lower_bound"],
                "upper": conformal_results["upper_bound"],
                "width": conformal_results["interval_width"],
            },
            "coverage_probability": conformal_results["coverage_probability"],
        }

    def _calibrate_confidence(self, raw_confidence: np.ndarray) -> np.ndarray:
        """Calibrate confidence scores to be more realistic"""

        # Apply sigmoid-like transformation to push confidence towards middle values
        # This makes the 80% gate more meaningful
        calibrated = 1 / (1 + np.exp(-5 * (raw_confidence - 0.6)))

        # Ensure minimum/maximum bounds
        calibrated = np.clip(calibrated, 0.1, 0.95)

        return calibrated

    def get_uncertainty_summary(self, X: np.ndarray) -> Dict:
        """Get summary statistics of uncertainty estimates"""

        results = self.predict_with_full_uncertainty(X)

        return {
            "mean_confidence": np.mean(results["confidence"]),
            "confidence_std": np.std(results["confidence"]),
            "high_confidence_fraction": np.mean(results["confidence"] > 0.8),
            "mean_interval_width": np.mean(results["prediction_intervals"]["width"]),
            "mean_epistemic_uncertainty": np.mean(results["epistemic_uncertainty"]),
            "mean_aleatoric_uncertainty": np.mean(results["aleatoric_uncertainty"]),
            "coverage_probability": results["coverage_probability"],
        }


def create_uncertainty_aware_predictions(
    features: pd.DataFrame, targets: pd.DataFrame, models: List
) -> pd.DataFrame:
    """
    Create uncertainty-aware predictions using advanced quantification methods

    Args:
        features: Input features
        targets: Target values
        models: List of base models to use in ensemble

    Returns:
        DataFrame with predictions and uncertainty estimates
    """

    quantifier = EnhancedUncertaintyQuantifier(models)

    # Fit uncertainty quantification
    X = features.values
    y = targets.values.ravel() if len(targets.shape) > 1 else targets.values

    quantifier.fit(X, y)

    # Generate predictions with uncertainty
    uncertainty_results = quantifier.predict_with_full_uncertainty(X)

    # Create results DataFrame
    results_df = pd.DataFrame(
        {
            "predictions": uncertainty_results["predictions"],
            "confidence": uncertainty_results["confidence"],
            "epistemic_uncertainty": uncertainty_results["epistemic_uncertainty"],
            "aleatoric_uncertainty": uncertainty_results["aleatoric_uncertainty"],
            "interval_lower": uncertainty_results["prediction_intervals"]["lower"],
            "interval_upper": uncertainty_results["prediction_intervals"]["upper"],
            "interval_width": uncertainty_results["prediction_intervals"]["width"],
        }
    )

    # Add feature information if available
    if hasattr(features, "index"):
        results_df.index = features.index

    # Get uncertainty summary
    uncertainty_summary = quantifier.get_uncertainty_summary(X)
    logger.info(f"Uncertainty summary: {uncertainty_summary}")

    return results_df
