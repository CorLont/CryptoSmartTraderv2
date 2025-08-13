#!/usr/bin/env python3
"""
Probability Calibrator
Advanced calibration methods for reliable confidence scores
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import matplotlib.pyplot as plt
import warnings
from dataclasses import dataclass
from enum import Enum
import pickle
import json
from pathlib import Path

warnings.filterwarnings("ignore")


class CalibrationMethod(Enum):
    """Available calibration methods"""

    PLATT_SCALING = "platt_scaling"
    ISOTONIC_REGRESSION = "isotonic_regression"
    TEMPERATURE_SCALING = "temperature_scaling"
    BETA_CALIBRATION = "beta_calibration"
    HISTOGRAM_BINNING = "histogram_binning"


@dataclass
class CalibrationResult:
    """Result of probability calibration"""

    method: CalibrationMethod
    calibrated_probabilities: np.ndarray
    calibration_error: float
    reliability_diagram_data: Dict[str, np.ndarray]
    brier_score: float
    log_loss_score: float
    auc_score: float
    is_well_calibrated: bool
    calibration_curve_x: np.ndarray
    calibration_curve_y: np.ndarray


@dataclass
class CalibrationMetrics:
    """Comprehensive calibration quality metrics"""

    expected_calibration_error: float
    maximum_calibration_error: float
    average_calibration_error: float
    overconfidence_error: float
    underconfidence_error: float
    sharpness: float
    reliability: float


class TemperatureScaling(nn.Module):
    """Temperature scaling for neural network calibration"""

    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        """Apply temperature scaling to logits"""
        return torch.div(logits, self.temperature)

    def fit(self, logits, labels, lr=0.01, max_iter=50):
        """Fit temperature parameter using validation set"""

        logits = torch.tensor(logits, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        optimizer = optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def eval():
            loss = nn.CrossEntropyLoss()(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval)

        return self.temperature.item()


class BetaCalibration:
    """Beta calibration for improved reliability"""

    def __init__(self):
        self.a = None
        self.b = None
        self.fitted = False

    def fit(self, probabilities, labels):
        """Fit beta calibration parameters"""

        # Use method of moments to estimate beta parameters
        p_mean = np.mean(probabilities)
        p_var = np.var(probabilities)

        if p_var > 0 and p_mean > 0 and p_mean < 1:
            self.a = p_mean * ((p_mean * (1 - p_mean)) / p_var - 1)
            self.b = (1 - p_mean) * ((p_mean * (1 - p_mean)) / p_var - 1)
            self.fitted = True
        else:
            # Fallback to uniform
            self.a = 1.0
            self.b = 1.0
            self.fitted = True

    def predict(self, probabilities):
        """Apply beta calibration"""

        if not self.fitted:
            return probabilities

        from scipy.stats import beta

        # Transform probabilities using beta distribution
        calibrated = beta.cdf(probabilities, self.a, self.b)
        return calibrated


class HistogramBinning:
    """Histogram binning calibration"""

    def __init__(self, n_bins=10):
        self.n_bins = n_bins
        self.bin_boundaries = None
        self.bin_lowers = None
        self.bin_uppers = None
        self.bin_calibrated = None
        self.fitted = False

    def fit(self, probabilities, labels):
        """Fit histogram binning calibration"""

        # Create bin boundaries
        self.bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        self.bin_lowers = self.bin_boundaries[:-1]
        self.bin_uppers = self.bin_boundaries[1:]

        # Calculate calibrated probability for each bin
        self.bin_calibrated = []

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Find samples in this bin
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)

            if np.sum(in_bin) > 0:
                # Calculate empirical probability
                bin_calibrated_prob = np.mean(labels[in_bin])
            else:
                # No samples in bin, use bin center
                bin_calibrated_prob = (bin_lower + bin_upper) / 2

            self.bin_calibrated.append(bin_calibrated_prob)

        self.bin_calibrated = np.array(self.bin_calibrated)
        self.fitted = True

    def predict(self, probabilities):
        """Apply histogram binning calibration"""

        if not self.fitted:
            return probabilities

        calibrated = np.zeros_like(probabilities)

        for i, (bin_lower, bin_upper) in enumerate(zip(self.bin_lowers, self.bin_uppers)):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            calibrated[in_bin] = self.bin_calibrated[i]

        return calibrated


class ProbabilityCalibrator:
    """Advanced probability calibration system"""

    def __init__(self, methods: List[CalibrationMethod] = None):
        self.methods = methods or [
            CalibrationMethod.PLATT_SCALING,
            CalibrationMethod.ISOTONIC_REGRESSION,
            CalibrationMethod.TEMPERATURE_SCALING,
        ]

        self.calibrators = {}
        self.best_method = None
        self.calibration_results = {}
        self.fitted = False

        self.logger = logging.getLogger(__name__)

    def fit(
        self, probabilities: np.ndarray, true_labels: np.ndarray, validation_split: float = 0.3
    ) -> Dict[CalibrationMethod, CalibrationResult]:
        """Fit multiple calibration methods and select best one"""

        # Validate inputs
        probabilities = np.asarray(probabilities)
        true_labels = np.asarray(true_labels)

        if len(probabilities) != len(true_labels):
            raise ValueError("Probabilities and labels must have same length")

        # Split into train/validation for calibration
        n_samples = len(probabilities)
        n_val = int(n_samples * validation_split)

        # Shuffle indices
        indices = np.random.permutation(n_samples)
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        train_probs = probabilities[train_idx]
        train_labels = true_labels[train_idx]
        val_probs = probabilities[val_idx]
        val_labels = true_labels[val_idx]

        results = {}

        # Fit each calibration method
        for method in self.methods:
            try:
                result = self._fit_single_method(
                    method, train_probs, train_labels, val_probs, val_labels
                )
                results[method] = result
                self.calibration_results[method] = result

            except Exception as e:
                self.logger.error(f"Failed to fit {method.value}: {e}")
                continue

        # Select best method based on calibration error
        if results:
            best_method = min(results.keys(), key=lambda m: results[m].calibration_error)
            self.best_method = best_method
            self.fitted = True

            self.logger.info(f"Best calibration method: {best_method.value}")
        else:
            raise ValueError("No calibration methods succeeded")

        return results

    def _fit_single_method(
        self,
        method: CalibrationMethod,
        train_probs: np.ndarray,
        train_labels: np.ndarray,
        val_probs: np.ndarray,
        val_labels: np.ndarray,
    ) -> CalibrationResult:
        """Fit single calibration method"""

        if method == CalibrationMethod.PLATT_SCALING:
            calibrator = self._fit_platt_scaling(train_probs, train_labels)
            calibrated_probs = calibrator.predict_proba(val_probs.reshape(-1, 1))[:, 1]

        elif method == CalibrationMethod.ISOTONIC_REGRESSION:
            calibrator = self._fit_isotonic_regression(train_probs, train_labels)
            calibrated_probs = calibrator.predict(val_probs)

        elif method == CalibrationMethod.TEMPERATURE_SCALING:
            calibrator = self._fit_temperature_scaling(train_probs, train_labels)
            # Convert probabilities to logits for temperature scaling
            logits = np.log(train_probs / (1 - train_probs + 1e-8))
            val_logits = np.log(val_probs / (1 - val_probs + 1e-8))
            calibrated_logits = val_logits / calibrator.temperature.item()
            calibrated_probs = 1 / (1 + np.exp(-calibrated_logits))

        elif method == CalibrationMethod.BETA_CALIBRATION:
            calibrator = BetaCalibration()
            calibrator.fit(train_probs, train_labels)
            calibrated_probs = calibrator.predict(val_probs)

        elif method == CalibrationMethod.HISTOGRAM_BINNING:
            calibrator = HistogramBinning(n_bins=10)
            calibrator.fit(train_probs, train_labels)
            calibrated_probs = calibrator.predict(val_probs)

        else:
            raise ValueError(f"Unknown calibration method: {method}")

        # Store calibrator
        self.calibrators[method] = calibrator

        # Calculate calibration metrics
        calibration_error = self._calculate_calibration_error(calibrated_probs, val_labels)
        reliability_data = self._calculate_reliability_diagram(calibrated_probs, val_labels)

        # Calculate performance metrics
        brier_score = brier_score_loss(val_labels, calibrated_probs)

        try:
            log_loss_score = log_loss(val_labels, calibrated_probs)
        except Exception:
            log_loss_score = float("inf")

        try:
            auc_score = roc_auc_score(val_labels, calibrated_probs)
        except Exception:
            auc_score = 0.5

        # Check if well calibrated (ECE < 0.1)
        is_well_calibrated = calibration_error < 0.1

        # Calculate calibration curve
        cal_x, cal_y = self._calculate_calibration_curve(calibrated_probs, val_labels)

        return CalibrationResult(
            method=method,
            calibrated_probabilities=calibrated_probs,
            calibration_error=calibration_error,
            reliability_diagram_data=reliability_data,
            brier_score=brier_score,
            log_loss_score=log_loss_score,
            auc_score=auc_score,
            is_well_calibrated=is_well_calibrated,
            calibration_curve_x=cal_x,
            calibration_curve_y=cal_y,
        )

    def _fit_platt_scaling(self, probabilities: np.ndarray, labels: np.ndarray):
        """Fit Platt scaling (logistic regression)"""

        # Convert probabilities to scores (logits)
        scores = np.log(probabilities / (1 - probabilities + 1e-8))

        # Fit logistic regression
        calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        calibrator.fit(scores.reshape(-1, 1), labels)

        return calibrator

    def _fit_isotonic_regression(self, probabilities: np.ndarray, labels: np.ndarray):
        """Fit isotonic regression"""

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(probabilities, labels)

        return calibrator

    def _fit_temperature_scaling(self, probabilities: np.ndarray, labels: np.ndarray):
        """Fit temperature scaling"""

        # Convert probabilities to logits
        logits = np.log(probabilities / (1 - probabilities + 1e-8))

        calibrator = TemperatureScaling()
        calibrator.fit(logits.reshape(-1, 1), labels)

        return calibrator

    def predict(
        self, probabilities: np.ndarray, method: Optional[CalibrationMethod] = None
    ) -> np.ndarray:
        """Apply calibration to new probabilities"""

        if not self.fitted:
            raise ValueError("Calibrator must be fitted before prediction")

        method = method or self.best_method

        if method not in self.calibrators:
            raise ValueError(f"Method {method.value} not fitted")

        calibrator = self.calibrators[method]

        if method == CalibrationMethod.PLATT_SCALING:
            scores = np.log(probabilities / (1 - probabilities + 1e-8))
            return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]

        elif method == CalibrationMethod.ISOTONIC_REGRESSION:
            return calibrator.predict(probabilities)

        elif method == CalibrationMethod.TEMPERATURE_SCALING:
            logits = np.log(probabilities / (1 - probabilities + 1e-8))
            calibrated_logits = logits / calibrator.temperature.item()
            return 1 / (1 + np.exp(-calibrated_logits))

        elif method == CalibrationMethod.BETA_CALIBRATION:
            return calibrator.predict(probabilities)

        elif method == CalibrationMethod.HISTOGRAM_BINNING:
            return calibrator.predict(probabilities)

        else:
            raise ValueError(f"Unknown method: {method}")

    def _calculate_calibration_error(
        self, probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
    ) -> float:
        """Calculate Expected Calibration Error (ECE)"""

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0.0
        total_samples = len(probabilities)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()

                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def _calculate_reliability_diagram(
        self, probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
    ) -> Dict[str, np.ndarray]:
        """Calculate reliability diagram data"""

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_centers = []
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)

            if np.sum(in_bin) > 0:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(labels[in_bin].mean())
                bin_confidences.append(probabilities[in_bin].mean())
                bin_counts.append(np.sum(in_bin))
            else:
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(0.0)
                bin_confidences.append((bin_lower + bin_upper) / 2)
                bin_counts.append(0)

        return {
            "bin_centers": np.array(bin_centers),
            "bin_accuracies": np.array(bin_accuracies),
            "bin_confidences": np.array(bin_confidences),
            "bin_counts": np.array(bin_counts),
        }

    def _calculate_calibration_curve(
        self, probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate calibration curve for plotting"""

        # Sort by probabilities
        sorted_indices = np.argsort(probabilities)
        sorted_probs = probabilities[sorted_indices]
        sorted_labels = labels[sorted_indices]

        # Create bins
        bin_size = len(probabilities) // n_bins
        bin_centers = []
        bin_accuracies = []

        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(probabilities)

            bin_probs = sorted_probs[start_idx:end_idx]
            bin_labels = sorted_labels[start_idx:end_idx]

            if len(bin_probs) > 0:
                bin_centers.append(np.mean(bin_probs))
                bin_accuracies.append(np.mean(bin_labels))

        return np.array(bin_centers), np.array(bin_accuracies)

    def get_calibration_metrics(
        self, probabilities: np.ndarray, labels: np.ndarray
    ) -> CalibrationMetrics:
        """Get comprehensive calibration metrics"""

        # Apply best calibration
        if self.fitted and self.best_method:
            calibrated_probs = self.predict(probabilities)
        else:
            calibrated_probs = probabilities

        # Expected Calibration Error
        ece = self._calculate_calibration_error(calibrated_probs, labels, n_bins=10)

        # Maximum Calibration Error
        bin_boundaries = np.linspace(0, 1, 11)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        max_ce = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (calibrated_probs > bin_lower) & (calibrated_probs <= bin_upper)
            if np.sum(in_bin) > 0:
                accuracy = labels[in_bin].mean()
                confidence = calibrated_probs[in_bin].mean()
                max_ce = max(max_ce, abs(confidence - accuracy))

        # Average Calibration Error
        ace = ece  # Same as ECE for uniform binning

        # Overconfidence and Underconfidence
        overconf = 0.0
        underconf = 0.0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (calibrated_probs > bin_lower) & (calibrated_probs <= bin_upper)
            if np.sum(in_bin) > 0:
                accuracy = labels[in_bin].mean()
                confidence = calibrated_probs[in_bin].mean()
                prop_in_bin = in_bin.mean()

                if confidence > accuracy:
                    overconf += (confidence - accuracy) * prop_in_bin
                else:
                    underconf += (accuracy - confidence) * prop_in_bin

        # Sharpness (average confidence)
        sharpness = np.mean(calibrated_probs)

        # Reliability (negative Brier score decomposition)
        reliability = 1.0 - brier_score_loss(labels, calibrated_probs)

        return CalibrationMetrics(
            expected_calibration_error=ece,
            maximum_calibration_error=max_ce,
            average_calibration_error=ace,
            overconfidence_error=overconf,
            underconfidence_error=underconf,
            sharpness=sharpness,
            reliability=reliability,
        )

    def save_calibrator(self, filepath: str):
        """Save calibrator to file"""

        save_data = {
            "methods": [m.value for m in self.methods],
            "best_method": self.best_method.value if self.best_method else None,
            "fitted": self.fitted,
            "calibrators": {},
        }

        # Save each calibrator
        for method, calibrator in self.calibrators.items():
            if method in [CalibrationMethod.PLATT_SCALING, CalibrationMethod.ISOTONIC_REGRESSION]:
                # Scikit-learn objects
                calibrator_path = f"{filepath}_{method.value}_calibrator.pkl"
                with open(calibrator_path, "wb") as f:
                    pickle.dump(calibrator, f)
                save_data["calibrators"][method.value] = calibrator_path

            elif method == CalibrationMethod.TEMPERATURE_SCALING:
                # PyTorch model
                calibrator_path = f"{filepath}_{method.value}_calibrator.pt"
                torch.save(calibrator.state_dict(), calibrator_path)
                save_data["calibrators"][method.value] = calibrator_path

            else:
                # Custom calibrators
                calibrator_path = f"{filepath}_{method.value}_calibrator.pkl"
                with open(calibrator_path, "wb") as f:
                    pickle.dump(calibrator, f)
                save_data["calibrators"][method.value] = calibrator_path

        # Save main config
        with open(f"{filepath}_config.json", "w") as f:
            json.dump(save_data, f)

    def load_calibrator(self, filepath: str):
        """Load calibrator from file"""

        # Load main config
        with open(f"{filepath}_config.json", "r") as f:
            save_data = json.load(f)

        self.methods = [CalibrationMethod(m) for m in save_data["methods"]]
        self.best_method = (
            CalibrationMethod(save_data["best_method"]) if save_data["best_method"] else None
        )
        self.fitted = save_data["fitted"]

        # Load each calibrator
        self.calibrators = {}
        for method_str, calibrator_path in save_data["calibrators"].items():
            method = CalibrationMethod(method_str)

            if method in [CalibrationMethod.PLATT_SCALING, CalibrationMethod.ISOTONIC_REGRESSION]:
                with open(calibrator_path, "rb") as f:
                    self.calibrators[method] = pickle.load(f)

            elif method == CalibrationMethod.TEMPERATURE_SCALING:
                calibrator = TemperatureScaling()
                calibrator.load_state_dict(torch.load(calibrator_path))
                self.calibrators[method] = calibrator

            else:
                with open(calibrator_path, "rb") as f:
                    self.calibrators[method] = pickle.load(f)


def create_probability_calibrator(methods: List[str] = None) -> ProbabilityCalibrator:
    """Create probability calibrator with specified methods"""

    if methods is None:
        methods = ["platt_scaling", "isotonic_regression", "temperature_scaling"]

    calibration_methods = [CalibrationMethod(m) for m in methods]
    return ProbabilityCalibrator(methods=calibration_methods)


def calibrate_predictions(
    probabilities: np.ndarray, true_labels: np.ndarray, method: str = "auto"
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """High-level function to calibrate predictions"""

    if method == "auto":
        calibrator = create_probability_calibrator()
        results = calibrator.fit(probabilities, true_labels)
        calibrated = calibrator.predict(probabilities)

        # Get best method info
        best_method = calibrator.best_method
        best_result = results[best_method]

        return calibrated, {
            "best_method": best_method.value,
            "calibration_error": best_result.calibration_error,
            "is_well_calibrated": best_result.is_well_calibrated,
            "brier_score": best_result.brier_score,
        }

    else:
        calibrator = create_probability_calibrator([method])
        results = calibrator.fit(probabilities, true_labels)
        calibrated = calibrator.predict(probabilities)

        result = list(results.values())[0]

        return calibrated, {
            "method": method,
            "calibration_error": result.calibration_error,
            "is_well_calibrated": result.is_well_calibrated,
            "brier_score": result.brier_score,
        }
