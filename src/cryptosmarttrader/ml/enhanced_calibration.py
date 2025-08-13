#!/usr/bin/env python3
"""
Enhanced Probability Calibration
Ensures ML confidence scores are properly calibrated
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from typing import Tuple, Dict, Any

class EnhancedCalibratorV2:
    """Advanced probability calibration with validation"""

    def __init__(self):
        self.calibrator = None
        self.calibration_curve = None
        self.is_fitted = False

    def fit_and_validate(self, probabilities: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
        """Fit calibrator and validate performance"""

        # Fit isotonic regression calibrator
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(probabilities, true_labels)
        self.is_fitted = True

        # Calculate calibration metrics
        calibrated_probs = self.calibrator.transform(probabilities)

        # Brier score (lower is better)
        brier_original = brier_score_loss(true_labels, probabilities)
        brier_calibrated = brier_score_loss(true_labels, calibrated_probs)

        # Expected Calibration Error
        ece_original = self._calculate_ece(probabilities, true_labels)
        ece_calibrated = self._calculate_ece(calibrated_probs, true_labels)

        return {
            "brier_original": brier_original,
            "brier_calibrated": brier_calibrated,
            "brier_improvement": brier_original - brier_calibrated,
            "ece_original": ece_original,
            "ece_calibrated": ece_calibrated,
            "ece_improvement": ece_original - ece_calibrated
        }

    def calibrate_probabilities(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply calibration to new probabilities"""

        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before use")

        return self.calibrator.transform(probabilities)

    def _calculate_ece(self, probabilities: np.ndarray, true_labels: np.ndarray, n_bins: int = 10) -> float:
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

def create_confidence_gate_with_calibration(threshold: float = 0.8) -> callable:
    """Create calibrated confidence gate function"""

    def calibrated_confidence_gate(predictions_df: pd.DataFrame,
                                  calibrator: EnhancedCalibratorV2 = None) -> pd.DataFrame:
        """Apply calibrated confidence gate"""

        if calibrator and calibrator.is_fitted:
            # Apply calibration to confidence scores
            for col in predictions_df.columns:
                if col.startswith('conf_'):
                    predictions_df[col] = calibrator.calibrate_probabilities(predictions_df[col].values)

        # Apply threshold filter
        confidence_mask = True
        for col in predictions_df.columns:
            if col.startswith('conf_'):
                confidence_mask &= (predictions_df[col] >= threshold)

        return predictions_df[confidence_mask]

    return calibrated_confidence_gate
