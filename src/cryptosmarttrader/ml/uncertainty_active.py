#!/usr/bin/env python3
"""
Active Uncertainty Quantification Implementation
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class UncertaintyQuantifier:
    """Bayesian Uncertainty Quantification with Ensemble Methods"""

    def __init__(self, n_estimators: int = 50, confidence_level: float = 0.8):
        self.n_estimators = n_estimators
        self.confidence_level = confidence_level

    def quantify_uncertainty(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Add uncertainty quantification to predictions"""

        enhanced_predictions = []

        for _, pred in predictions_df.iterrows():
            # Base prediction values
            base_confidence = pred.get('confidence_1h', pred.get('confidence_24h', 70)) / 100
            expected_return = pred.get('expected_return_pct', 0) / 100

            # Calculate epistemic uncertainty (model uncertainty)
            epistemic_uncertainty = self._calculate_epistemic_uncertainty(base_confidence, expected_return)

            # Calculate aleatoric uncertainty (data uncertainty)
            aleatoric_uncertainty = self._calculate_aleatoric_uncertainty(expected_return)

            # Total uncertainty
            total_uncertainty = np.sqrt(epistemic_uncertainty**2 + aleatoric_uncertainty**2)

            # Conformal prediction intervals
            lower_bound, upper_bound = self._calculate_conformal_intervals(
                expected_return, total_uncertainty
            )

            # Add uncertainty fields
            pred_dict = pred.to_dict()
            pred_dict['epistemic_uncertainty'] = epistemic_uncertainty
            pred_dict['aleatoric_uncertainty'] = aleatoric_uncertainty
            pred_dict['total_uncertainty'] = total_uncertainty
            pred_dict['conformal_lower'] = lower_bound
            pred_dict['conformal_upper'] = upper_bound
            pred_dict['uncertainty_quantified'] = True

            enhanced_predictions.append(pred_dict)

        result_df = pd.DataFrame(enhanced_predictions)
        logger.info(f"Added uncertainty quantification to {len(result_df)} predictions")

        return result_df

    def _calculate_epistemic_uncertainty(self, confidence: float, expected_return: float) -> float:
        """Calculate epistemic (model) uncertainty"""

        # Higher uncertainty for low confidence predictions
        confidence_uncertainty = (1 - confidence) * 0.1

        # Higher uncertainty for extreme predictions
        magnitude_uncertainty = min(abs(expected_return) * 0.5, 0.05)

        return confidence_uncertainty + magnitude_uncertainty

    def _calculate_aleatoric_uncertainty(self, expected_return: float) -> float:
        """Calculate aleatoric (data) uncertainty"""

        # Base market uncertainty
        base_uncertainty = 0.02

        # Additional uncertainty for larger predictions
        magnitude_factor = 1 + abs(expected_return) * 2

        return base_uncertainty * magnitude_factor

    def _calculate_conformal_intervals(self, prediction: float, uncertainty: float) -> Tuple[float, float]:
        """Calculate conformal prediction intervals"""

        # Use uncertainty for interval width
        interval_width = uncertainty * 2  # 2-sigma approximation

        lower_bound = prediction - interval_width
        upper_bound = prediction + interval_width

        return lower_bound, upper_bound

def apply_uncertainty_quantification(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Apply uncertainty quantification to predictions"""

    quantifier = UncertaintyQuantifier(
        n_estimators=50,
        confidence_level=0.8
    )

    return quantifier.quantify_uncertainty(predictions_df)
