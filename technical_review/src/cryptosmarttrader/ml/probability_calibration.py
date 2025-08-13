#!/usr/bin/env python3
"""
Probability Calibration - Ensure confidence gates work properly
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import numpy as np
import pandas as pd
from typing import Tuple, Optional

class ConfidenceCalibrator:
    """Calibrate model confidence scores"""

    def __init__(self, method='isotonic'):
        self.method = method
        self.calibrator = None
        self.is_fitted = False

    def fit(self, probabilities: np.ndarray, true_labels: np.ndarray) -> 'ConfidenceCalibrator':
        """Fit calibration model"""

        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.calibrator.fit(probabilities, true_labels)
        self.is_fitted = True

        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities"""

        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted first")

        return self.calibrator.transform(probabilities)

    def reliability_plot_data(self, probabilities: np.ndarray, true_labels: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data for reliability plot"""

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_centers = []
        bin_accuracies = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = true_labels[in_bin].mean()
                bin_centers.append((bin_lower + bin_upper) / 2)
                bin_accuracies.append(accuracy_in_bin)

        return np.array(bin_centers), np.array(bin_accuracies)

def validate_calibration(probabilities: np.ndarray, true_labels: np.ndarray, confidence_threshold: float = 0.8) -> bool:
    """Validate that confidence threshold is meaningful"""

    calibrator = ConfidenceCalibrator()
    bin_centers, bin_accuracies = calibrator.reliability_plot_data(probabilities, true_labels)

    # Find bins around confidence threshold
    threshold_bins = bin_centers[(bin_centers >= confidence_threshold - 0.1) & (bin_centers <= confidence_threshold + 0.1)]
    threshold_accuracies = bin_accuracies[(bin_centers >= confidence_threshold - 0.1) & (bin_centers <= confidence_threshold + 0.1)]

    if len(threshold_accuracies) == 0:
        return False

    # Accuracy should be close to confidence for calibrated model
    mean_accuracy = threshold_accuracies.mean()
    calibration_error = abs(mean_accuracy - confidence_threshold)

    return calibration_error < 0.2  # Allow 20% calibration error
