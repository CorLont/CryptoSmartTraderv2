#!/usr/bin/env python3
"""
Conformal Prediction for Uncertainty Quantification
Provides theoretical guarantees for prediction intervals and confidence calibration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

from core.structured_logger import get_structured_logger


class ConformalPredictor:
    """Conformal prediction for uncertainty quantification with coverage guarantees"""

    def __init__(self, alpha: float = 0.2, calibration_split: float = 0.2):
        """
        Initialize conformal predictor

        Args:
            alpha: Miscoverage level (e.g., 0.2 for 80% coverage)
            calibration_split: Fraction of data for calibration
        """
        self.logger = get_structured_logger("ConformalPredictor")
        self.alpha = alpha
        self.coverage_level = 1 - alpha
        self.calibration_split = calibration_split

        # Calibration data
        self.calibration_scores = None
        self.quantile_threshold = None
        self.is_calibrated = False

    def calibrate(
        self, X_cal: np.ndarray, y_cal: np.ndarray, base_model, base_predictions: np.ndarray = None
    ) -> None:
        """Calibrate conformal predictor using calibration set"""

        self.logger.info(f"Calibrating conformal predictor with {len(X_cal)} samples")

        try:
            # Get predictions on calibration set if not provided
            if base_predictions is None:
                if hasattr(base_model, "predict"):
                    base_predictions = base_model.predict(X_cal)
                else:
                    # For PyTorch models
                    base_model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X_cal)
                        base_predictions = base_model(X_tensor).numpy().flatten()

            # Calculate nonconformity scores (absolute residuals)
            self.calibration_scores = np.abs(y_cal - base_predictions)

            # Calculate quantile threshold for desired coverage
            n_cal = len(self.calibration_scores)
            quantile_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
            quantile_level = min(quantile_level, 1.0)  # Ensure <= 1

            self.quantile_threshold = np.quantile(self.calibration_scores, quantile_level)

            self.is_calibrated = True

            # Calculate actual coverage on calibration set
            actual_coverage = self._calculate_coverage(base_predictions, y_cal)

            self.logger.info(f"Conformal calibration complete")
            self.logger.info(f"Target coverage: {self.coverage_level:.1%}")
            self.logger.info(f"Actual coverage: {actual_coverage:.1%}")
            self.logger.info(f"Quantile threshold: {self.quantile_threshold:.4f}")

        except Exception as e:
            self.logger.error(f"Conformal calibration failed: {e}")
            raise

    def predict_with_intervals(
        self, X_test: np.ndarray, base_model, base_predictions: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """Generate predictions with conformal prediction intervals"""

        if not self.is_calibrated:
            raise ValueError("Conformal predictor must be calibrated first")

        try:
            # Get base predictions if not provided
            if base_predictions is None:
                if hasattr(base_model, "predict"):
                    base_predictions = base_model.predict(X_test)
                else:
                    # For PyTorch models
                    base_model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X_test)
                        base_predictions = base_model(X_tensor).numpy().flatten()

            # Calculate prediction intervals
            lower_bounds = base_predictions - self.quantile_threshold
            upper_bounds = base_predictions + self.quantile_threshold

            # Calculate interval widths (uncertainty measure)
            interval_widths = upper_bounds - lower_bounds

            # Calculate efficiency (smaller intervals are better)
            efficiency_scores = 1.0 / (1.0 + interval_widths)

            self.logger.info(f"Generated conformal intervals for {len(X_test)} predictions")
            self.logger.info(f"Mean interval width: {np.mean(interval_widths):.4f}")

            return {
                "predictions": base_predictions,
                "lower_bounds": lower_bounds,
                "upper_bounds": upper_bounds,
                "interval_widths": interval_widths,
                "efficiency_scores": efficiency_scores,
                "coverage_level": self.coverage_level,
            }

        except Exception as e:
            self.logger.error(f"Conformal prediction failed: {e}")
            raise

    def _calculate_coverage(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate actual coverage on given data"""

        if not self.is_calibrated:
            return 0.0

        lower_bounds = predictions - self.quantile_threshold
        upper_bounds = predictions + self.quantile_threshold

        covered = (actuals >= lower_bounds) & (actuals <= upper_bounds)
        coverage = np.mean(covered)

        return coverage

    def adaptive_calibration(
        self, X_cal: np.ndarray, y_cal: np.ndarray, base_model, difficulty_scores: np.ndarray = None
    ) -> None:
        """Adaptive conformal prediction that adjusts intervals based on difficulty"""

        self.logger.info("Performing adaptive conformal calibration")

        try:
            # Get base predictions
            if hasattr(base_model, "predict"):
                base_predictions = base_model.predict(X_cal)
            else:
                base_model.eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(X_cal)
                    base_predictions = base_model(X_tensor).numpy().flatten()

            # Calculate nonconformity scores
            nonconformity_scores = np.abs(y_cal - base_predictions)

            if difficulty_scores is None:
                # Use prediction variance as difficulty score
                difficulty_scores = np.abs(base_predictions - np.mean(base_predictions))
                difficulty_scores = difficulty_scores / (np.std(difficulty_scores) + 1e-8)

            # Normalize difficulty scores
            difficulty_scores = difficulty_scores - np.min(difficulty_scores)
            difficulty_scores = difficulty_scores / (np.max(difficulty_scores) + 1e-8)

            # Weight nonconformity scores by difficulty
            weighted_scores = nonconformity_scores * (1 + difficulty_scores)

            # Calculate adaptive threshold
            n_cal = len(weighted_scores)
            quantile_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
            quantile_level = min(quantile_level, 1.0)

            self.quantile_threshold = np.quantile(weighted_scores, quantile_level)
            self.is_calibrated = True

            # Store difficulty adjustment for future predictions
            self.difficulty_adjustment = True
            self.difficulty_calibration_data = {
                "mean_difficulty": np.mean(difficulty_scores),
                "std_difficulty": np.std(difficulty_scores),
            }

            self.logger.info(f"Adaptive conformal calibration complete")
            self.logger.info(f"Adaptive threshold: {self.quantile_threshold:.4f}")

        except Exception as e:
            self.logger.error(f"Adaptive conformal calibration failed: {e}")
            raise


class MonteCarloDropoutUncertainty:
    """Monte Carlo Dropout for epistemic uncertainty estimation"""

    def __init__(self, n_samples: int = 100, dropout_rate: float = 0.1):
        self.logger = get_structured_logger("MCDropoutUncertainty")
        self.n_samples = n_samples
        self.dropout_rate = dropout_rate

    def enable_dropout_inference(self, model: nn.Module) -> None:
        """Enable dropout during inference for uncertainty estimation"""

        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()  # Keep dropout active during inference

    def predict_with_uncertainty(self, model: nn.Module, X: torch.Tensor) -> Dict[str, np.ndarray]:
        """Generate predictions with MC Dropout uncertainty"""

        self.logger.info(f"Generating MC Dropout predictions with {self.n_samples} samples")

        try:
            model.eval()  # Set to eval mode first
            self.enable_dropout_inference(model)  # Then enable dropout

            predictions = []

            with torch.no_grad():
                for _ in range(self.n_samples):
                    pred = model(X)
                    predictions.append(pred.cpu().numpy())

            predictions = np.array(predictions)  # Shape: (n_samples, batch_size, output_dim)

            # Calculate statistics
            mean_predictions = np.mean(predictions, axis=0).flatten()
            epistemic_uncertainty = np.std(predictions, axis=0).flatten()

            # Calculate confidence based on uncertainty
            # Lower uncertainty = higher confidence
            max_uncertainty = np.max(epistemic_uncertainty) + 1e-8
            confidence_scores = 1.0 - (epistemic_uncertainty / max_uncertainty)

            # Calculate prediction intervals (approximation)
            std_multiplier = 1.96  # 95% confidence interval
            lower_bounds = mean_predictions - std_multiplier * epistemic_uncertainty
            upper_bounds = mean_predictions + std_multiplier * epistemic_uncertainty

            self.logger.info(f"MC Dropout uncertainty estimation complete")
            self.logger.info(f"Mean epistemic uncertainty: {np.mean(epistemic_uncertainty):.4f}")

            return {
                "predictions": mean_predictions,
                "epistemic_uncertainty": epistemic_uncertainty,
                "confidence_scores": confidence_scores,
                "lower_bounds": lower_bounds,
                "upper_bounds": upper_bounds,
                "all_samples": predictions,
            }

        except Exception as e:
            self.logger.error(f"MC Dropout prediction failed: {e}")
            raise


class EnsembleUncertainty:
    """Uncertainty quantification using model ensemble"""

    def __init__(self, models: List[Any]):
        self.logger = get_structured_logger("EnsembleUncertainty")
        self.models = models

    def predict_with_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Generate ensemble predictions with uncertainty"""

        self.logger.info(f"Generating ensemble predictions with {len(self.models)} models")

        try:
            predictions = []

            for model in self.models:
                if hasattr(model, "predict"):
                    pred = model.predict(X)
                else:
                    # PyTorch model
                    model.eval()
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X)
                        pred = model(X_tensor).numpy()

                predictions.append(pred.flatten())

            predictions = np.array(predictions)  # Shape: (n_models, n_samples)

            # Calculate statistics
            mean_predictions = np.mean(predictions, axis=0)
            model_disagreement = np.std(predictions, axis=0)

            # Calculate confidence (inverse of disagreement)
            max_disagreement = np.max(model_disagreement) + 1e-8
            confidence_scores = 1.0 - (model_disagreement / max_disagreement)

            # Prediction intervals
            std_multiplier = 1.96
            lower_bounds = mean_predictions - std_multiplier * model_disagreement
            upper_bounds = mean_predictions + std_multiplier * model_disagreement

            self.logger.info(f"Ensemble uncertainty estimation complete")
            self.logger.info(f"Mean model disagreement: {np.mean(model_disagreement):.4f}")

            return {
                "predictions": mean_predictions,
                "model_disagreement": model_disagreement,
                "confidence_scores": confidence_scores,
                "lower_bounds": lower_bounds,
                "upper_bounds": upper_bounds,
                "individual_predictions": predictions,
            }

        except Exception as e:
            self.logger.error(f"Ensemble prediction failed: {e}")
            raise


def create_calibrated_confidence_gate(
    conformal_predictor: ConformalPredictor, confidence_threshold: float = 0.8
) -> callable:
    """Create a calibrated confidence gate using conformal prediction"""

    def calibrated_gate(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply calibrated confidence gate to predictions"""

        filtered_predictions = []

        for pred in predictions:
            # Use conformal prediction intervals to assess confidence
            interval_width = pred.get("interval_width", 0.1)
            efficiency_score = pred.get("efficiency_score", 0.5)

            # Calibrated confidence combines efficiency and original confidence
            original_confidence = pred.get("confidence", 0.5)
            calibrated_confidence = 0.6 * original_confidence + 0.4 * efficiency_score

            if calibrated_confidence >= confidence_threshold:
                pred["calibrated_confidence"] = calibrated_confidence
                filtered_predictions.append(pred)

        return {
            "filtered_predictions": filtered_predictions,
            "original_count": len(predictions),
            "filtered_count": len(filtered_predictions),
            "gate_passed": len(filtered_predictions) > 0,
            "calibrated_pass_rate": len(filtered_predictions) / len(predictions)
            if predictions
            else 0,
        }

    return calibrated_gate
