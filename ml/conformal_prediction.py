#!/usr/bin/env python3
"""
Conformal Prediction for Model Uncertainty
Provides formal, data-driven uncertainty intervals
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class ConformalPredictor:
    """
    Conformal prediction for regression with uncertainty intervals
    """
    
    def __init__(self, 
                 base_model: nn.Module,
                 alpha: float = 0.1):  # 1-alpha confidence level (90% for alpha=0.1)
        self.base_model = base_model
        self.alpha = alpha
        self.calibration_scores = None
        self.quantile_score = None
        self.is_calibrated = False
        
    def calibrate(self, 
                 cal_features: torch.Tensor,
                 cal_targets: torch.Tensor) -> Dict[str, float]:
        """Calibrate conformal predictor on calibration set"""
        
        self.base_model.eval()
        
        with torch.no_grad():
            # Get predictions on calibration set
            cal_predictions = self.base_model(cal_features)
            
            # Calculate conformity scores (absolute residuals)
            conformity_scores = torch.abs(cal_targets - cal_predictions).squeeze()
            
            # Sort scores and find quantile
            sorted_scores = torch.sort(conformity_scores)[0]
            n_cal = len(sorted_scores)
            
            # Calculate quantile index for (1-alpha) coverage
            quantile_index = int(np.ceil((n_cal + 1) * (1 - self.alpha))) - 1
            quantile_index = min(quantile_index, n_cal - 1)  # Ensure within bounds
            
            self.quantile_score = sorted_scores[quantile_index].item()
            self.calibration_scores = conformity_scores
            self.is_calibrated = True
        
        # Calculate empirical coverage on calibration set
        coverage = self._calculate_coverage(cal_features, cal_targets)
        
        return {
            'quantile_score': self.quantile_score,
            'calibration_size': n_cal,
            'empirical_coverage': coverage,
            'target_coverage': 1 - self.alpha
        }
    
    def predict_with_intervals(self, 
                             features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Make predictions with conformal intervals"""
        
        if not self.is_calibrated:
            raise ValueError("Model must be calibrated before making interval predictions")
        
        self.base_model.eval()
        
        with torch.no_grad():
            # Base predictions
            point_predictions = self.base_model(features)
            
            # Conformal intervals
            lower_bound = point_predictions - self.quantile_score
            upper_bound = point_predictions + self.quantile_score
            
            # Interval width
            interval_width = upper_bound - lower_bound
        
        return {
            'predictions': point_predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': interval_width,
            'confidence_level': 1 - self.alpha
        }
    
    def _calculate_coverage(self, 
                          features: torch.Tensor,
                          targets: torch.Tensor) -> float:
        """Calculate empirical coverage on given data"""
        
        intervals = self.predict_with_intervals(features)
        
        # Check if targets fall within intervals
        in_interval = (
            (targets >= intervals['lower_bound']) & 
            (targets <= intervals['upper_bound'])
        )
        
        coverage = in_interval.float().mean().item()
        return coverage

class AdaptiveConformalPredictor:
    """
    Adaptive conformal predictor that adjusts to changing distributions
    """
    
    def __init__(self, 
                 base_model: nn.Module,
                 alpha: float = 0.1,
                 window_size: int = 100,
                 update_frequency: int = 10):
        self.base_model = base_model
        self.alpha = alpha
        self.window_size = window_size
        self.update_frequency = update_frequency
        
        self.prediction_history = []
        self.target_history = []
        self.error_history = []
        self.quantile_history = []
        
        self.current_quantile = None
        self.update_counter = 0
        
    def update_and_predict(self, 
                          features: torch.Tensor,
                          targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Update calibration and make predictions"""
        
        self.base_model.eval()
        
        with torch.no_grad():
            predictions = self.base_model(features)
            
            # Update history if targets are available
            if targets is not None:
                errors = torch.abs(targets - predictions).squeeze()
                
                self.prediction_history.extend(predictions.cpu().numpy().flatten())
                self.target_history.extend(targets.cpu().numpy().flatten())
                self.error_history.extend(errors.cpu().numpy().flatten())
                
                # Keep only recent history
                if len(self.error_history) > self.window_size:
                    self.error_history = self.error_history[-self.window_size:]
                    self.prediction_history = self.prediction_history[-self.window_size:]
                    self.target_history = self.target_history[-self.window_size:]
                
                # Update quantile periodically
                self.update_counter += 1
                if self.update_counter >= self.update_frequency:
                    self._update_quantile()
                    self.update_counter = 0
            
            # Make interval predictions
            if self.current_quantile is not None:
                lower_bound = predictions - self.current_quantile
                upper_bound = predictions + self.current_quantile
                interval_width = upper_bound - lower_bound
            else:
                # Fallback to simple prediction if not calibrated
                lower_bound = predictions
                upper_bound = predictions
                interval_width = torch.zeros_like(predictions)
        
        return {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'interval_width': interval_width,
            'current_quantile': self.current_quantile or 0.0,
            'calibration_size': len(self.error_history)
        }
    
    def _update_quantile(self):
        """Update quantile based on recent errors"""
        
        if len(self.error_history) < 10:  # Need minimum samples
            return
        
        # Calculate new quantile
        errors = np.array(self.error_history)
        n = len(errors)
        quantile_index = int(np.ceil(n * (1 - self.alpha))) - 1
        quantile_index = min(quantile_index, n - 1)
        
        sorted_errors = np.sort(errors)
        new_quantile = sorted_errors[quantile_index]
        
        # Smooth update
        if self.current_quantile is None:
            self.current_quantile = new_quantile
        else:
            # Exponential smoothing
            smoothing_factor = 0.1
            self.current_quantile = (
                smoothing_factor * new_quantile + 
                (1 - smoothing_factor) * self.current_quantile
            )
        
        self.quantile_history.append(self.current_quantile)

class EnhancedConformalSystem:
    """
    Enhanced conformal prediction system with multiple confidence levels
    """
    
    def __init__(self, 
                 base_model: nn.Module,
                 confidence_levels: List[float] = [0.8, 0.9, 0.95]):
        self.base_model = base_model
        self.confidence_levels = confidence_levels
        self.conformal_predictors = {}
        
        # Create predictor for each confidence level
        for level in confidence_levels:
            alpha = 1 - level
            self.conformal_predictors[level] = ConformalPredictor(
                base_model=base_model,
                alpha=alpha
            )
    
    def calibrate_all(self, 
                     cal_features: torch.Tensor,
                     cal_targets: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """Calibrate all confidence levels"""
        
        calibration_results = {}
        
        for level, predictor in self.conformal_predictors.items():
            results = predictor.calibrate(cal_features, cal_targets)
            calibration_results[f'confidence_{level}'] = results
        
        return calibration_results
    
    def predict_multi_level(self, 
                           features: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """Predict with multiple confidence levels"""
        
        multi_predictions = {}
        
        for level, predictor in self.conformal_predictors.items():
            if predictor.is_calibrated:
                predictions = predictor.predict_with_intervals(features)
                multi_predictions[f'confidence_{level}'] = predictions
        
        return multi_predictions
    
    def get_adaptive_confidence(self, 
                               features: torch.Tensor,
                               desired_width: float) -> Dict[str, Any]:
        """Get confidence level that provides desired interval width"""
        
        if not any(p.is_calibrated for p in self.conformal_predictors.values()):
            return {'error': 'No calibrated predictors available'}
        
        # Try different confidence levels
        width_results = {}
        
        for level, predictor in self.conformal_predictors.items():
            if predictor.is_calibrated:
                predictions = predictor.predict_with_intervals(features)
                avg_width = predictions['interval_width'].mean().item()
                width_results[level] = avg_width
        
        # Find closest to desired width
        best_level = min(width_results.keys(), 
                        key=lambda x: abs(width_results[x] - desired_width))
        
        return {
            'best_confidence_level': best_level,
            'achieved_width': width_results[best_level],
            'desired_width': desired_width,
            'all_widths': width_results,
            'predictions': self.conformal_predictors[best_level].predict_with_intervals(features)
        }

def create_conformal_pipeline(model: nn.Module,
                             X: torch.Tensor,
                             y: torch.Tensor,
                             cal_ratio: float = 0.2) -> Tuple[EnhancedConformalSystem, Dict[str, Any]]:
    """Create complete conformal prediction pipeline"""
    
    # Split data for calibration
    X_train, X_cal, y_train, y_cal = train_test_split(
        X.cpu().numpy(), y.cpu().numpy(), 
        test_size=cal_ratio, random_state=42
    )
    
    X_cal = torch.FloatTensor(X_cal)
    y_cal = torch.FloatTensor(y_cal)
    
    # Create enhanced conformal system
    conformal_system = EnhancedConformalSystem(model)
    
    # Calibrate all levels
    calibration_results = conformal_system.calibrate_all(X_cal, y_cal)
    
    return conformal_system, calibration_results

if __name__ == "__main__":
    print("🎯 TESTING CONFORMAL PREDICTION SYSTEM")
    print("=" * 45)
    
    # Create sample model and data
    input_size = 10
    hidden_size = 64
    output_size = 1
    n_samples = 1000
    
    # Simple neural network
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Linear(hidden_size // 2, output_size)
    )
    
    # Generate sample data
    torch.manual_seed(42)
    X = torch.randn(n_samples, input_size)
    true_function = lambda x: torch.sum(x[:, :3], dim=1, keepdim=True) + 0.1 * torch.randn(x.shape[0], 1)
    y = true_function(X)
    
    # Train model briefly
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()
    
    print(f"Model trained, final loss: {loss.item():.4f}")
    
    # Create conformal pipeline
    conformal_system, calibration_results = create_conformal_pipeline(
        model, X, y, cal_ratio=0.2
    )
    
    print(f"\nConformal Calibration Results:")
    for level, results in calibration_results.items():
        print(f"   {level}:")
        print(f"      Target coverage: {results['target_coverage']:.3f}")
        print(f"      Empirical coverage: {results['empirical_coverage']:.3f}")
        print(f"      Quantile score: {results['quantile_score']:.4f}")
    
    # Test predictions
    test_X = torch.randn(100, input_size)
    multi_predictions = conformal_system.predict_multi_level(test_X)
    
    print(f"\nInterval Predictions:")
    for level, predictions in multi_predictions.items():
        avg_width = predictions['interval_width'].mean().item()
        print(f"   {level}: Average interval width = {avg_width:.4f}")
    
    # Test adaptive confidence
    adaptive_result = conformal_system.get_adaptive_confidence(
        test_X[:10], desired_width=0.5
    )
    
    if 'best_confidence_level' in adaptive_result:
        print(f"\nAdaptive Confidence:")
        print(f"   Desired width: {adaptive_result['desired_width']}")
        print(f"   Best level: {adaptive_result['best_confidence_level']}")
        print(f"   Achieved width: {adaptive_result['achieved_width']:.4f}")
    
    print("✅ Conformal prediction system testing completed")