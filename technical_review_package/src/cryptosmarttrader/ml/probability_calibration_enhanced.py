#!/usr/bin/env python3
"""
Enhanced Probability Calibration
Calibrate probabilities and implement uncertainty for meaningful 80% gates
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings("ignore")


class EnhancedProbabilityCalibrator:
    """
    Enhanced probability calibration system for meaningful confidence gates
    """

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrator = None
        self.uncertainty_model = None
        self.is_calibrated = False
        self.calibration_metrics = {}

    def fit_calibration(
        self, probabilities: np.ndarray, true_labels: np.ndarray, validation_split: float = 0.3
    ) -> Dict[str, float]:
        """Fit calibration model with validation"""

        # Split for calibration and validation
        prob_cal, prob_val, labels_cal, labels_val = train_test_split(
            probabilities, true_labels, test_size=validation_split, random_state=42
        )

        # Fit calibrator
        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
        elif self.method == "platt":
            from sklearn.linear_model import LogisticRegression

            self.calibrator = LogisticRegression()
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        # Fit on calibration set
        if self.method == "platt":
            # Platt scaling requires reshaping
            self.calibrator.fit(prob_cal.reshape(-1, 1), labels_cal)
        else:
            self.calibrator.fit(prob_cal, labels_cal)

        self.is_calibrated = True

        # Evaluate on validation set
        if self.method == "platt":
            calibrated_probs = self.calibrator.predict_proba(prob_val.reshape(-1, 1))[:, 1]
        else:
            calibrated_probs = self.calibrator.transform(prob_val)

        # Calculate calibration metrics
        self.calibration_metrics = self._calculate_calibration_metrics(
            prob_val, calibrated_probs, labels_val
        )

        return self.calibration_metrics

    def calibrate_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities"""

        if not self.is_calibrated:
            raise ValueError("Calibrator must be fitted first")

        if self.method == "platt":
            return self.calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
        else:
            return self.calibrator.transform(probabilities)

    def _calculate_calibration_metrics(
        self, original_probs: np.ndarray, calibrated_probs: np.ndarray, true_labels: np.ndarray
    ) -> Dict[str, float]:
        """Calculate comprehensive calibration metrics"""

        metrics = {}

        # Brier Score (lower is better)
        brier_original = np.mean((original_probs - true_labels) ** 2)
        brier_calibrated = np.mean((calibrated_probs - true_labels) ** 2)

        metrics["brier_score_original"] = brier_original
        metrics["brier_score_calibrated"] = brier_calibrated
        metrics["brier_improvement"] = brier_original - brier_calibrated

        # Expected Calibration Error (ECE)
        ece_original = self._calculate_ece(original_probs, true_labels)
        ece_calibrated = self._calculate_ece(calibrated_probs, true_labels)

        metrics["ece_original"] = ece_original
        metrics["ece_calibrated"] = ece_calibrated
        metrics["ece_improvement"] = ece_original - ece_calibrated

        # Reliability at different confidence levels
        for conf_level in [0.8, 0.9, 0.95]:
            reliability = self._calculate_reliability_at_confidence(
                calibrated_probs, true_labels, conf_level
            )
            metrics[f"reliability_at_{conf_level}"] = reliability

        return metrics

    def _calculate_ece(
        self, probabilities: np.ndarray, true_labels: np.ndarray, n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error"""

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = true_labels[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def _calculate_reliability_at_confidence(
        self, probabilities: np.ndarray, true_labels: np.ndarray, confidence_level: float
    ) -> float:
        """Calculate reliability at specific confidence level"""

        high_conf_mask = probabilities >= confidence_level

        if high_conf_mask.sum() == 0:
            return 0.0

        actual_accuracy = true_labels[high_conf_mask].mean()
        return actual_accuracy


class UncertaintyQuantifier:
    """
    Quantify model uncertainty using multiple methods
    """

    def __init__(self, base_model: nn.Module):
        self.base_model = base_model
        self.uncertainty_methods = ["mc_dropout", "ensemble_variance", "prediction_variance"]

    def calculate_mc_dropout_uncertainty(
        self, x: torch.Tensor, n_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate uncertainty using Monte Carlo Dropout"""

        self.base_model.train()  # Enable dropout

        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = torch.sigmoid(self.base_model(x))
                predictions.append(pred)

        predictions = torch.stack(predictions)

        # Mean prediction and uncertainty (standard deviation)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)

        self.base_model.eval()  # Disable dropout

        return mean_pred, uncertainty

    def calculate_prediction_variance(
        self, predictions: torch.Tensor, targets: torch.Tensor, window_size: int = 50
    ) -> torch.Tensor:
        """Calculate uncertainty based on prediction variance"""

        # Calculate rolling prediction errors
        errors = torch.abs(predictions - targets)

        # Calculate rolling variance of errors
        uncertainties = []
        for i in range(len(errors)):
            start_idx = max(0, i - window_size)
            window_errors = errors[start_idx : i + 1]
            uncertainty = torch.var(window_errors) if len(window_errors) > 1 else torch.tensor(0.1)
            uncertainties.append(uncertainty)

        return torch.stack(uncertainties)


class EnhancedConfidenceGate:
    """
    Enhanced confidence gate with calibrated probabilities and uncertainty
    """

    def __init__(
        self,
        confidence_threshold: float = 0.8,
        uncertainty_threshold: float = 0.1,
        min_samples: int = 5,
    ):
        self.confidence_threshold = confidence_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.min_samples = min_samples
        self.calibrator = EnhancedProbabilityCalibrator()
        self.uncertainty_quantifier = None

    def calibrate_system(
        self, model_predictions: np.ndarray, true_outcomes: np.ndarray
    ) -> Dict[str, float]:
        """Calibrate the confidence gate system"""

        # Fit probability calibration
        calibration_results = self.calibrator.fit_calibration(model_predictions, true_outcomes)

        return calibration_results

    def apply_gate(
        self, predictions: np.ndarray, uncertainties: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Apply enhanced confidence gate"""

        if not self.calibrator.is_calibrated:
            return {
                "passed_gate": [],
                "failed_gate": list(range(len(predictions))),
                "reason": "calibration_required",
                "gate_stats": {"pass_rate": 0.0, "total_candidates": len(predictions)},
            }

        # Calibrate probabilities
        calibrated_probs = self.calibrator.calibrate_probabilities(predictions)

        # Apply confidence threshold
        confidence_mask = calibrated_probs >= self.confidence_threshold

        # Apply uncertainty threshold if provided
        if uncertainties is not None:
            uncertainty_mask = uncertainties <= self.uncertainty_threshold
            final_mask = confidence_mask & uncertainty_mask
        else:
            final_mask = confidence_mask

        # Apply minimum samples requirement
        passed_indices = np.where(final_mask)[0]
        failed_indices = np.where(~final_mask)[0]

        if len(passed_indices) < self.min_samples:
            # If not enough samples pass, fail all
            failed_indices = np.arange(len(predictions))
            passed_indices = np.array([])

        gate_stats = {
            "pass_rate": len(passed_indices) / len(predictions),
            "total_candidates": len(predictions),
            "confidence_mean": calibrated_probs.mean(),
            "confidence_passed_mean": calibrated_probs[passed_indices].mean()
            if len(passed_indices) > 0
            else 0,
            "uncertainty_mean": uncertainties.mean() if uncertainties is not None else 0,
        }

        return {
            "passed_gate": passed_indices.tolist(),
            "failed_gate": failed_indices.tolist(),
            "calibrated_probabilities": calibrated_probs,
            "gate_stats": gate_stats,
            "reason": "enhanced_gate_applied",
        }


def create_calibrated_confidence_system(
    model: nn.Module, training_data: Dict[str, torch.Tensor]
) -> EnhancedConfidenceGate:
    """Create complete calibrated confidence system"""

    # Initialize confidence gate
    confidence_gate = EnhancedConfidenceGate()

    # Get model predictions for calibration
    model.eval()
    with torch.no_grad():
        predictions = torch.sigmoid(model(training_data["features"])).cpu().numpy()
        targets = training_data["targets"].cpu().numpy()

    # Calibrate system
    calibration_results = confidence_gate.calibrate_system(predictions, targets)

    print("Calibration Results:")
    for key, value in calibration_results.items():
        print(f"   {key}: {value:.4f}")

    return confidence_gate


if __name__ == "__main__":
    print("🎯 TESTING ENHANCED PROBABILITY CALIBRATION")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    # Generate poorly calibrated predictions (overconfident)
    true_probs = np.random.normal(0, 1)  # True underlying probabilities
    predicted_probs = np.clip(true_probs * 1.5 + 0.1, 0, 1)  # Overconfident predictions
    true_labels = np.random.binomial(1, true_probs)

    print(f"Generated {n_samples} samples for calibration testing")

    # Test calibration
    calibrator = EnhancedProbabilityCalibrator()
    results = calibrator.fit_calibration(predicted_probs, true_labels)

    print("Calibration Metrics:")
    for key, value in results.items():
        print(f"   {key}: {value:.4f}")

    # Test calibrated predictions
    test_probs = np.random.normal(0, 1)
    calibrated = calibrator.calibrate_probabilities(test_probs)

    print(f"\nCalibration example:")
    print(f"   Original: {test_probs[:5]}")
    print(f"   Calibrated: {calibrated[:5]}")

    # Test confidence gate
    confidence_gate = EnhancedConfidenceGate()
    calibration_results = confidence_gate.calibrate_system(predicted_probs, true_labels)

    # Apply gate to test predictions
    gate_results = confidence_gate.apply_gate(test_probs)

    print(f"\nConfidence Gate Results:")
    print(f"   Pass rate: {gate_results['gate_stats']['pass_rate']:.2%}")
    print(
        f"   Passed: {len(gate_results['passed_gate'])}/{gate_results['gate_stats']['total_candidates']}"
    )

    print("✅ Enhanced probability calibration testing completed")
