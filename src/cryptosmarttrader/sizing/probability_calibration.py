"""
Probability Calibration for ML Model Outputs

Calibrates raw ML probabilities to actual win rates using:
- Platt Scaling (sigmoid calibration)
- Isotonic Regression (monotonic calibration)
- Cross-validation for robust calibration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.calibration import CalibratedClassifierCV, calibration_curve  # type: ignore
from sklearn.isotonic import IsotonicRegression  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.model_selection import cross_val_score  # type: ignore
import logging
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class CalibrationMetrics:
    """Container for calibration quality metrics"""
    brier_score: float          # Lower is better (0 = perfect)
    reliability: float          # How close predicted probs are to actual frequencies
    resolution: float           # Ability to separate positive/negative cases
    ece: float                  # Expected Calibration Error
    mce: float                  # Maximum Calibration Error
    log_loss: float            # Logarithmic loss


class ProbabilityCalibrator:
    """
    Calibrates ML model probabilities to actual win rates

    Uses both Platt scaling and Isotonic regression for robust calibration.
    """

    def __init__(self, method: str = "auto", cv_folds: int = 5):
        """
        Initialize probability calibrator

        Args:
            method: 'platt', 'isotonic', or 'auto' (chooses best)
            cv_folds: Cross-validation folds for calibration
        """
        self.method = method
        self.cv_folds = cv_folds

        # Calibration models
        self.platt_calibrator = None
        self.isotonic_calibrator = None
        self.best_calibrator = None
        self.best_method = None

        # Calibration metrics
        self.calibration_metrics = {}
        self.is_fitted = False

    def fit(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> Dict[str, Any]:
        """
        Fit calibration models on historical predictions vs outcomes

        Args:
            predicted_probs: Raw ML probabilities (0-1)
            actual_outcomes: Binary outcomes (0=loss, 1=win)

        Returns:
            Calibration metrics and performance comparison
        """
        try:
            logger.info("Fitting probability calibration models...")

            # Validate inputs
            if len(predicted_probs) != len(actual_outcomes):
                raise ValueError("Predicted probabilities and outcomes must have same length")

            if len(predicted_probs) < 50:
                logger.warning("Limited data for calibration (<50 samples)")

            # Ensure binary outcomes
            unique_outcomes = np.unique(actual_outcomes)
            if not set(unique_outcomes).issubset({0, 1}):
                raise ValueError("Outcomes must be binary (0 or 1)")

            # Fit Platt scaling (sigmoid calibration)
            self.platt_calibrator = self._fit_platt_scaling(predicted_probs, actual_outcomes)

            # Fit Isotonic regression (monotonic calibration)
            self.isotonic_calibrator = self._fit_isotonic_regression(predicted_probs, actual_outcomes)

            # Evaluate both methods
            platt_metrics = self._evaluate_calibration(predicted_probs, actual_outcomes, "platt")
            isotonic_metrics = self._evaluate_calibration(predicted_probs, actual_outcomes, "isotonic")

            self.calibration_metrics = {
                "platt": platt_metrics,
                "isotonic": isotonic_metrics
            }

            # Choose best method
            if self.method == "auto":
                self.best_method = self._select_best_method(platt_metrics, isotonic_metrics)
            else:
                self.best_method = self.method

            self.best_calibrator = (
                self.platt_calibrator if self.best_method == "platt"
                else self.isotonic_calibrator
            )

            self.is_fitted = True

            logger.info(f"Calibration completed. Best method: {self.best_method}")

            return {
                "best_method": self.best_method,
                "platt_metrics": platt_metrics,
                "isotonic_metrics": isotonic_metrics,
                "n_samples": len(predicted_probs),
                "win_rate": np.mean(actual_outcomes)
            }

        except Exception as e:
            logger.error(f"Calibration fitting failed: {e}")
            raise

    def calibrate(self, predicted_probs: np.ndarray) -> np.ndarray:
        """
        Apply calibration to new predicted probabilities

        Args:
            predicted_probs: Raw ML probabilities to calibrate

        Returns:
            Calibrated probabilities
        """
        if not self.is_fitted:
            logger.warning("Calibrator not fitted, returning raw probabilities")
            return predicted_probs

        try:
            if self.best_method == "platt":
                return self.platt_calibrator.predict_proba(predicted_probs.reshape(-1, 1))[:, 1]
            else:
                return self.isotonic_calibrator.predict(predicted_probs)

        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return predicted_probs

    def get_calibration_curve(self, predicted_probs: np.ndarray,
                             actual_outcomes: np.ndarray,
                             n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate reliability diagram data

        Returns:
            Tuple of (mean_predicted_prob, fraction_positives)
        """
        try:
            return calibration_curve(actual_outcomes, predicted_probs, n_bins=n_bins)
        except Exception as e:
            logger.error(f"Calibration curve generation failed: {e}")
            return np.array([]), np.array([])

    def _fit_platt_scaling(self, predicted_probs: np.ndarray,
                          actual_outcomes: np.ndarray) -> LogisticRegression:
        """Fit Platt scaling (logistic regression on probabilities)"""
        calibrator = LogisticRegression()
        calibrator.fit(predicted_probs.reshape(-1, 1), actual_outcomes)
        return calibrator

    def _fit_isotonic_regression(self, predicted_probs: np.ndarray,
                                actual_outcomes: np.ndarray) -> IsotonicRegression:
        """Fit isotonic regression for monotonic calibration"""
        calibrator = IsotonicRegression(out_of_bounds='clip')
        calibrator.fit(predicted_probs, actual_outcomes)
        return calibrator

    def _evaluate_calibration(self, predicted_probs: np.ndarray,
                             actual_outcomes: np.ndarray,
                             method: str) -> CalibrationMetrics:
        """Evaluate calibration quality"""
        try:
            # Get calibrated probabilities
            if method == "platt":
                cal_probs = self.platt_calibrator.predict_proba(predicted_probs.reshape(-1, 1))[:, 1]
            else:
                cal_probs = self.isotonic_calibrator.predict(predicted_probs)

            # Brier Score (lower is better)
            brier_score = np.mean((cal_probs - actual_outcomes) ** 2)

            # Reliability and Resolution
            reliability, resolution = self._compute_reliability_resolution(
                cal_probs, actual_outcomes
            )

            # Expected/Maximum Calibration Error
            ece, mce = self._compute_calibration_errors(cal_probs, actual_outcomes)

            # Log Loss
            epsilon = 1e-15  # Prevent log(0)
            cal_probs_clipped = np.clip(cal_probs, epsilon, 1 - epsilon)
            log_loss = -np.mean(
                actual_outcomes * np.log(cal_probs_clipped) +
                (1 - actual_outcomes) * np.log(1 - cal_probs_clipped)
            )

            return CalibrationMetrics(
                brier_score=brier_score,
                reliability=reliability,
                resolution=resolution,
                ece=ece,
                mce=mce,
                log_loss=log_loss
            )

        except Exception as e:
            logger.error(f"Calibration evaluation failed: {e}")
            return CalibrationMetrics(1.0, 1.0, 0.0, 1.0, 1.0, 10.0)  # Worst case

    def _compute_reliability_resolution(self, cal_probs: np.ndarray,
                                       actual_outcomes: np.ndarray,
                                       n_bins: int = 10) -> Tuple[float, float]:
        """Compute reliability and resolution components"""
        try:
            # Bin predictions
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            reliability = 0.0
            resolution = 0.0
            base_rate = np.mean(actual_outcomes)

            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Find predictions in this bin
                in_bin = (cal_probs > bin_lower) & (cal_probs <= bin_upper)
                prop_in_bin = np.mean(in_bin)

                if prop_in_bin > 0:
                    # Average predicted probability in bin
                    avg_predicted = np.mean(cal_probs[in_bin])

                    # Actual frequency in bin
                    actual_freq = np.mean(actual_outcomes[in_bin])

                    # Reliability: weighted squared difference
                    reliability += prop_in_bin * (avg_predicted - actual_freq) ** 2

                    # Resolution: weighted squared difference from base rate
                    resolution += prop_in_bin * (actual_freq - base_rate) ** 2

            return reliability, resolution

        except Exception as e:
            logger.error(f"Reliability/resolution computation failed: {e}")
            return 1.0, 0.0

    def _compute_calibration_errors(self, cal_probs: np.ndarray,
                                   actual_outcomes: np.ndarray,
                                   n_bins: int = 10) -> Tuple[float, float]:
        """Compute Expected and Maximum Calibration Error"""
        try:
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]

            ece = 0.0
            mce = 0.0

            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (cal_probs > bin_lower) & (cal_probs <= bin_upper)
                prop_in_bin = np.mean(in_bin)

                if prop_in_bin > 0:
                    avg_predicted = np.mean(cal_probs[in_bin])
                    actual_freq = np.mean(actual_outcomes[in_bin])

                    # Calibration error for this bin
                    bin_error = abs(avg_predicted - actual_freq)

                    # ECE: weighted average
                    ece += prop_in_bin * bin_error

                    # MCE: maximum error
                    mce = max(mce, bin_error)

            return ece, mce

        except Exception as e:
            logger.error(f"Calibration error computation failed: {e}")
            return 1.0, 1.0

    def _select_best_method(self, platt_metrics: CalibrationMetrics,
                           isotonic_metrics: CalibrationMetrics) -> str:
        """Select best calibration method based on metrics"""
        try:
            # Primary criterion: Expected Calibration Error (lower is better)
            if platt_metrics.ece < isotonic_metrics.ece:
                primary_winner = "platt"
            elif isotonic_metrics.ece < platt_metrics.ece:
                primary_winner = "isotonic"
            else:
                # Tie-breaker: Brier Score
                primary_winner = "platt" if platt_metrics.brier_score <= isotonic_metrics.brier_score else "isotonic"

            logger.info(f"Selected {primary_winner} based on ECE: "
                       f"Platt={platt_metrics.ece:.4f}, Isotonic={isotonic_metrics.ece:.4f}")

            return primary_winner

        except Exception as e:
            logger.error(f"Method selection failed: {e}")
            return "platt"  # Default fallback

    def save_calibrator(self, path: str) -> bool:
        """Save fitted calibrator to disk"""
        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

            calibrator_data = {
                "platt_calibrator": self.platt_calibrator,
                "isotonic_calibrator": self.isotonic_calibrator,
                "best_method": self.best_method,
                "calibration_metrics": self.calibration_metrics,
                "is_fitted": self.is_fitted,
                "method": self.method,
                "cv_folds": self.cv_folds
            }

            joblib.dump(calibrator_data, path)
            logger.info(f"Calibrator saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save calibrator: {e}")
            return False

    def load_calibrator(self, path: str) -> bool:
        """Load fitted calibrator from disk"""
        try:
            if not Path(path).exists():
                logger.warning(f"Calibrator file not found: {path}")
                return False

            calibrator_data = joblib.load(path)

            self.platt_calibrator = calibrator_data["platt_calibrator"]
            self.isotonic_calibrator = calibrator_data["isotonic_calibrator"]
            self.best_method = calibrator_data["best_method"]
            self.calibration_metrics = calibrator_data["calibration_metrics"]
            self.is_fitted = calibrator_data["is_fitted"]
            self.method = calibrator_data.get("method", "auto")
            self.cv_folds = calibrator_data.get("cv_folds", 5)

            self.best_calibrator = (
                self.platt_calibrator if self.best_method == "platt"
                else self.isotonic_calibrator
            )

            logger.info(f"Calibrator loaded from {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load calibrator: {e}")
            return False

    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of calibration performance"""
        if not self.is_fitted:
            return {"status": "Not fitted"}

        try:
            best_metrics = self.calibration_metrics.get(self.best_method, {})

            return {
                "status": "Fitted",
                "best_method": self.best_method,
                "brier_score": best_metrics.brier_score if best_metrics else None,
                "expected_calibration_error": best_metrics.ece if best_metrics else None,
                "log_loss": best_metrics.log_loss if best_metrics else None,
                "reliability": best_metrics.reliability if best_metrics else None,
                "resolution": best_metrics.resolution if best_metrics else None,
                "available_methods": list(self.calibration_metrics.keys())
            }

        except Exception as e:
            logger.error(f"Failed to generate calibration summary: {e}")
            return {"status": "Error", "error": str(e)}
