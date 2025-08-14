#!/usr/bin/env python3
"""
Probability Calibrator - Solid uncertainty quantification
Ensures 80% confidence gate is meaningful through proper calibration
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import joblib
from pathlib import Path
import json

from core.structured_logger import get_structured_logger


class ProbabilityCalibrator:
    """Advanced probability calibration for meaningful confidence scores"""

    def __init__(self):
        self.logger = get_structured_logger("ProbabilityCalibrator")
        self.calibrators = {}  # One per horizon
        self.calibration_stats = {}
        self.is_fitted = False

    def fit_calibration(
        self, predictions: pd.DataFrame, actual_outcomes: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Fit calibration models for each horizon

        Args:
            predictions: DataFrame with pred_{horizon}, conf_{horizon} columns
            actual_outcomes: DataFrame with actual_{horizon} columns (binary: 1 if pred was correct)
        """

        horizons = ["1h", "24h", "168h", "720h"]
        calibration_results = {}

        for horizon in horizons:
            pred_col = f"pred_{horizon}"
            conf_col = f"conf_{horizon}"
            actual_col = f"actual_{horizon}"

            if pred_col not in predictions.columns or conf_col not in predictions.columns:
                self.logger.warning(f"Missing prediction columns for {horizon}")
                continue

            if actual_col not in actual_outcomes.columns:
                self.logger.warning(f"Missing actual outcomes for {horizon}")
                continue

            # Get data for this horizon
            pred_data = predictions[[pred_col, conf_col]].copy()
            actual_data = actual_outcomes[actual_col].copy()

            # Remove NaN values
            valid_mask = pred_data.notna().all(axis=1) & actual_data.notna()
            pred_data = pred_data[valid_mask]
            actual_data = actual_data[valid_mask]

            if len(pred_data) < 100:
                self.logger.warning(
                    f"Insufficient data for {horizon} calibration: {len(pred_data)} samples"
                )
                continue

            # Fit calibration
            horizon_results = self._fit_horizon_calibration(
                pred_data[conf_col].values, actual_data.values, horizon
            )

            calibration_results[horizon] = horizon_results

        self.is_fitted = len(calibration_results) > 0

        if self.is_fitted:
            self.logger.info(f"Calibration fitted for {len(calibration_results)} horizons")
        else:
            self.logger.error("Calibration fitting failed - no horizons calibrated")

        return {
            "success": self.is_fitted,
            "calibrated_horizons": list(calibration_results.keys()),
            "calibration_stats": calibration_results,
        }

    def _fit_horizon_calibration(
        self, confidence_scores: np.ndarray, binary_outcomes: np.ndarray, horizon: str
    ) -> Dict[str, Any]:
        """Fit calibration for a single horizon"""

        # Method 1: Isotonic Regression (non-parametric, monotonic)
        isotonic_cal = IsotonicRegression(out_of_bounds="clip")
        isotonic_cal.fit(confidence_scores, binary_outcomes)

        # Method 2: Platt scaling (logistic regression)
        # Reshape for sklearn
        X = confidence_scores.reshape(-1, 1)
        platt_cal = LogisticRegression()
        platt_cal.fit(X, binary_outcomes)

        # Evaluate calibration quality
        calibration_stats = self._evaluate_calibration(confidence_scores, binary_outcomes)

        # Store calibrators
        self.calibrators[horizon] = {
            "isotonic": isotonic_cal,
            "platt": platt_cal,
            "method": "isotonic" if calibration_stats["isotonic_better"] else "platt",
        }

        self.calibration_stats[horizon] = calibration_stats

        return {
            "calibration_error": calibration_stats["calibration_error"],
            "brier_score": calibration_stats["brier_score"],
            "reliability": calibration_stats["reliability"],
            "resolution": calibration_stats["resolution"],
            "sharpness": calibration_stats["sharpness"],
            "method_used": self.calibrators[horizon]["method"],
            "samples": len(confidence_scores),
        }

    def _evaluate_calibration(
        self, confidence_scores: np.ndarray, binary_outcomes: np.ndarray
    ) -> Dict[str, Any]:
        """Evaluate calibration quality with multiple metrics"""

        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = binary_outcomes[in_bin].mean()
                avg_confidence_in_bin = confidence_scores[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)

        # Brier Score
        brier_score = np.mean((confidence_scores - binary_outcomes) ** 2)

        # Reliability, Resolution, Sharpness decomposition
        overall_accuracy = binary_outcomes.mean()

        reliability = 0
        resolution = 0

        for i, (bin_lower, bin_upper) in enumerate(zip(bin_lowers, bin_uppers)):
            in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)

            if in_bin.sum() > 0:
                prop_in_bin = in_bin.mean()
                accuracy_in_bin = binary_outcomes[in_bin].mean()
                avg_confidence_in_bin = confidence_scores[in_bin].mean()

                # Reliability: how far predicted probabilities are from actual frequencies
                reliability += prop_in_bin * (avg_confidence_in_bin - accuracy_in_bin) ** 2

                # Resolution: how much predictions can distinguish between different outcomes
                resolution += prop_in_bin * (accuracy_in_bin - overall_accuracy) ** 2

        sharpness = np.var(confidence_scores)  # Variance of predictions

        # Compare isotonic vs platt for this data
        isotonic_cal = IsotonicRegression(out_of_bounds="clip")
        platt_cal = LogisticRegression()

        # Simple cross-validation comparison
        n_samples = len(confidence_scores)
        split_idx = n_samples // 2

        # Train on first half, test on second
        isotonic_cal.fit(confidence_scores[:split_idx], binary_outcomes[:split_idx])
        platt_cal.fit(confidence_scores[:split_idx].reshape(-1, 1), binary_outcomes[:split_idx])

        isotonic_pred = isotonic_cal.predict(confidence_scores[split_idx:])
        platt_pred = platt_cal.predict_proba(confidence_scores[split_idx:].reshape(-1, 1))[:, 1]

        isotonic_brier = np.mean((isotonic_pred - binary_outcomes[split_idx:]) ** 2)
        platt_brier = np.mean((platt_pred - binary_outcomes[split_idx:]) ** 2)

        return {
            "calibration_error": ece,
            "brier_score": brier_score,
            "reliability": reliability,
            "resolution": resolution,
            "sharpness": sharpness,
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
            "isotonic_brier": isotonic_brier,
            "platt_brier": platt_brier,
            "isotonic_better": isotonic_brier < platt_brier,
        }

    def calibrate_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Apply calibration to confidence scores"""

        if not self.is_fitted:
            self.logger.warning("Calibrator not fitted - returning original predictions")
            return predictions

        calibrated_predictions = predictions.copy()

        for horizon in ["1h", "24h", "168h", "720h"]:
            conf_col = f"conf_{horizon}"
            calibrated_col = f"conf_{horizon}_calibrated"

            if conf_col not in predictions.columns:
                continue

            if horizon not in self.calibrators:
                self.logger.warning(f"No calibrator for {horizon} - copying original")
                calibrated_predictions[calibrated_col] = predictions[conf_col]
                continue

            # Apply calibration
            original_conf = predictions[conf_col].values
            calibrator_info = self.calibrators[horizon]
            method = calibrator_info["method"]

            if method == "isotonic":
                calibrated_conf = calibrator_info["isotonic"].predict(original_conf)
            else:  # platt
                calibrated_conf = calibrator_info["platt"].predict_proba(
                    original_conf.reshape(-1, 1)
                )[:, 1]

            # Ensure values stay in [0, 1]
            calibrated_conf = np.clip(calibrated_conf, 0, 1)

            calibrated_predictions[calibrated_col] = calibrated_conf

            # Log calibration effect
            original_mean = original_conf.mean()
            calibrated_mean = calibrated_conf.mean()
            self.logger.info(f"{horizon} calibration: {original_mean:.3f} → {calibrated_mean:.3f}")

        return calibrated_predictions

    def save_calibration(self, filepath: str):
        """Save fitted calibrators"""

        if not self.is_fitted:
            raise ValueError("Calibrator not fitted - cannot save")

        calibration_data = {
            "calibrators": self.calibrators,
            "calibration_stats": self.calibration_stats,
            "is_fitted": self.is_fitted,
        }

        joblib.dump(calibration_data, filepath)
        self.logger.info(f"Calibration saved to {filepath}")

    def load_calibration(self, filepath: str):
        """Load fitted calibrators"""

        if not Path(filepath).exists():
            raise FileNotFoundError(f"Calibration file not found: {filepath}")

        calibration_data = joblib.load(filepath)

        self.calibrators = calibration_data["calibrators"]
        self.calibration_stats = calibration_data["calibration_stats"]
        self.is_fitted = calibration_data["is_fitted"]

        self.logger.info(f"Calibration loaded from {filepath}")

    def get_calibration_report(self) -> Dict[str, Any]:
        """Get comprehensive calibration report"""

        if not self.is_fitted:
            return {"error": "Calibrator not fitted"}

        report = {
            "fitted": True,
            "calibrated_horizons": list(self.calibrators.keys()),
            "calibration_quality": {},
        }

        for horizon, stats in self.calibration_stats.items():
            report["calibration_quality"][horizon] = {
                "calibration_error": stats["calibration_error"],
                "brier_score": stats["brier_score"],
                "reliability": stats["reliability"],
                "resolution": stats["resolution"],
                "method_used": self.calibrators[horizon]["method"],
                "quality_grade": self._grade_calibration_quality(stats["calibration_error"]),
            }

        return report

    def _grade_calibration_quality(self, calibration_error: float) -> str:
        """Grade calibration quality based on ECE"""

        if calibration_error < 0.05:
            return "EXCELLENT"
        elif calibration_error < 0.10:
            return "GOOD"
        elif calibration_error < 0.15:
            return "FAIR"
        else:
            return "POOR"


def create_synthetic_calibration_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create synthetic data for calibration testing"""

    np.random.seed(42)
    n_samples = 2000  # Increased for better calibration

    # Generate predictions with realistic confidence patterns
    predictions_data = []
    outcomes_data = []

    for i in range(n_samples):
        coin = f"COIN_{i % 50:03d}"

        # Generate correlated prediction and confidence
        true_signal = np.random.normal(0, 0.02)
        noise = np.random.normal(0, 0.01)

        for horizon in ["1h", "24h", "168h", "720h"]:
            # Prediction with noise
            prediction = true_signal + noise * np.random.normal()

            # Confidence that correlates with actual accuracy
            base_confidence = 0.7 + 0.2 * np.abs(true_signal) / 0.02
            confidence = np.clip(base_confidence + np.random.normal(0, 0.1), 0.1, 0.95)

            # Actual outcome (binary: was prediction correct?)
            # Higher confidence should correlate with higher accuracy
            accuracy_prob = 0.3 + 0.4 * confidence  # Base accuracy + confidence boost
            actual_correct = np.random.binomial(1, accuracy_prob)

            predictions_data.append(
                {"coin": coin, f"pred_{horizon}": prediction, f"conf_{horizon}": confidence}
            )

            outcomes_data.append({"coin": coin, f"actual_{horizon}": actual_correct})

    predictions_df = pd.DataFrame(predictions_data)
    outcomes_df = pd.DataFrame(outcomes_data)

    # Aggregate by coin (take mean)
    predictions_df = predictions_df.groupby("coin").mean().reset_index()
    outcomes_df = outcomes_df.groupby("coin").mean().reset_index()

    # Round outcomes to binary
    for horizon in ["1h", "24h", "168h", "720h"]:
        outcomes_df[f"actual_{horizon}"] = (outcomes_df[f"actual_{horizon}"] > 0.5).astype(int)

    return predictions_df, outcomes_df


if __name__ == "__main__":
    print("Testing Probability Calibrator...")

    # Create synthetic data
    predictions_df, outcomes_df = create_synthetic_calibration_data()
    print(f"Created synthetic data: {len(predictions_df)} predictions, {len(outcomes_df)} outcomes")

    # Test calibration
    calibrator = ProbabilityCalibrator()

    # Fit calibration
    fit_results = calibrator.fit_calibration(predictions_df, outcomes_df)
    print(f"Calibration fit results: {fit_results}")

    # Test calibration
    calibrated_predictions = calibrator.calibrate_predictions(predictions_df)
    print(f"Calibrated predictions shape: {calibrated_predictions.shape}")

    # Show calibration effect
    for horizon in ["1h", "24h", "168h", "720h"]:
        original_col = f"conf_{horizon}"
        calibrated_col = f"conf_{horizon}_calibrated"

        if calibrated_col in calibrated_predictions.columns:
            original_mean = predictions_df[original_col].mean()
            calibrated_mean = calibrated_predictions[calibrated_col].mean()
            print(f"{horizon}: {original_mean:.3f} → {calibrated_mean:.3f}")

    # Get calibration report
    report = calibrator.get_calibration_report()
    print(f"\nCalibration Report:")
    for horizon, quality in report["calibration_quality"].items():
        print(f"  {horizon}: {quality['quality_grade']} (ECE: {quality['calibration_error']:.3f})")

    print("\n✅ Probability Calibrator test completed!")
