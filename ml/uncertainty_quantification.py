#!/usr/bin/env python3
"""
Uncertainty Quantification for ML Models
Implements Monte Carlo Dropout and Ensemble Uncertainty
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

class MonteCarloDropoutModel(nn.Module):
    """Neural network with Monte Carlo Dropout for uncertainty"""
    
    def __init__(self, input_size: int, hidden_size: int = 64, dropout_rate: float = 0.1):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with uncertainty using MC Dropout"""
        
        self.train()  # Keep dropout active
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        
        # Calculate mean and uncertainty
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred.flatten(), uncertainty.flatten()

class EnsembleUncertaintyEstimator:
    """Ensemble-based uncertainty estimation"""
    
    def __init__(self, n_estimators: int = 10):
        self.n_estimators = n_estimators
        self.models = []
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Fit ensemble of models"""
        
        self.models = []
        
        for i in range(self.n_estimators):
            # Create bootstrapped dataset
            n_samples = len(X)
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Train model
            model = RandomForestRegressor(n_estimators=50, random_state=i)
            model.fit(X_boot, y_boot)
            self.models.append(model)
        
        self.is_fitted = True
        
        # Validate ensemble
        ensemble_predictions = self.predict_with_uncertainty(X)
        
        return {
            "success": True,
            "models_trained": len(self.models),
            "ensemble_variance": np.mean(ensemble_predictions[1]),
            "prediction_range": {
                "min": np.min(ensemble_predictions[0]),
                "max": np.max(ensemble_predictions[0])
            }
        }
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict with ensemble uncertainty"""
        
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted first")
        
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Calculate ensemble statistics
        mean_pred = np.mean(predictions, axis=0)
        uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty

class ConfidenceIntervalEstimator:
    """Confidence interval estimation for predictions"""
    
    def __init__(self, method: str = "bootstrap"):
        self.method = method
        self.percentiles = [5, 25, 50, 75, 95]
    
    def estimate_intervals(self, 
                          predictions: np.ndarray, 
                          uncertainties: np.ndarray,
                          confidence_levels: List[float] = [0.8, 0.9, 0.95]) -> Dict[str, np.ndarray]:
        """Estimate confidence intervals"""
        
        intervals = {}
        
        for conf_level in confidence_levels:
            # Calculate z-score for confidence level
            from scipy.stats import norm
            z_score = norm.ppf((1 + conf_level) / 2)
            
            # Calculate intervals
            lower_bound = predictions - z_score * uncertainties
            upper_bound = predictions + z_score * uncertainties
            
            intervals[f"CI_{int(conf_level*100)}"] = {
                "lower": lower_bound,
                "upper": upper_bound,
                "width": upper_bound - lower_bound
            }
        
        return intervals
    
    def validate_calibration(self, 
                           predictions: np.ndarray,
                           uncertainties: np.ndarray, 
                           actual_values: np.ndarray) -> Dict[str, float]:
        """Validate uncertainty calibration"""
        
        calibration_metrics = {}
        
        # Calculate prediction intervals
        intervals = self.estimate_intervals(predictions, uncertainties)
        
        for interval_name, interval_data in intervals.items():
            # Check coverage
            within_interval = (
                (actual_values >= interval_data["lower"]) & 
                (actual_values <= interval_data["upper"])
            )
            
            coverage = within_interval.mean()
            expected_coverage = float(interval_name.split("_")[1]) / 100
            
            calibration_metrics[f"{interval_name}_coverage"] = coverage
            calibration_metrics[f"{interval_name}_calibration_error"] = abs(coverage - expected_coverage)
        
        return calibration_metrics

class UncertaintyAwarePredictionSystem:
    """Complete uncertainty-aware prediction system"""
    
    def __init__(self):
        self.ensemble = EnsembleUncertaintyEstimator()
        self.confidence_estimator = ConfidenceIntervalEstimator()
        self.uncertainty_threshold = 0.1  # Maximum acceptable uncertainty
    
    def train_uncertainty_model(self, 
                               features: pd.DataFrame, 
                               targets: pd.Series) -> Dict[str, Any]:
        """Train uncertainty-aware model"""
        
        # Prepare data
        X = features.select_dtypes(include=[np.number]).fillna(0).values
        y = targets.fillna(0).values
        
        # Train ensemble
        ensemble_result = self.ensemble.fit(X, y)
        
        if not ensemble_result.get("success", False):
            return ensemble_result
        
        # Test uncertainty estimation
        test_predictions, test_uncertainties = self.ensemble.predict_with_uncertainty(X)
        
        # Calculate confidence intervals
        intervals = self.confidence_estimator.estimate_intervals(test_predictions, test_uncertainties)
        
        return {
            "success": True,
            "ensemble_result": ensemble_result,
            "uncertainty_stats": {
                "mean_uncertainty": np.mean(test_uncertainties),
                "uncertainty_range": {
                    "min": np.min(test_uncertainties),
                    "max": np.max(test_uncertainties)
                }
            },
            "confidence_intervals": {name: {
                "mean_width": np.mean(data["width"])
            } for name, data in intervals.items()}
        }
    
    def predict_with_confidence_gate(self, 
                                   features: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Make predictions with uncertainty-based confidence gate"""
        
        X = features.select_dtypes(include=[np.number]).fillna(0).values
        
        # Get predictions with uncertainty
        predictions, uncertainties = self.ensemble.predict_with_uncertainty(X)
        
        # Calculate confidence scores (inverse of uncertainty)
        max_uncertainty = np.max(uncertainties) if len(uncertainties) > 0 else 1.0
        confidence_scores = 1 - (uncertainties / max_uncertainty)
        
        # Apply uncertainty gate
        high_confidence_mask = uncertainties <= self.uncertainty_threshold
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'prediction': predictions,
            'uncertainty': uncertainties,
            'confidence': confidence_scores,
            'high_confidence': high_confidence_mask
        })
        
        gate_report = {
            "total_predictions": len(predictions),
            "high_confidence_count": high_confidence_mask.sum(),
            "high_confidence_rate": high_confidence_mask.mean(),
            "mean_uncertainty": np.mean(uncertainties),
            "uncertainty_threshold": self.uncertainty_threshold
        }
        
        return results_df, gate_report
