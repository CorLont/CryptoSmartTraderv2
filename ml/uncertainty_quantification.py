#!/usr/bin/env python3
"""
Uncertainty Quantification - MC Dropout and Ensemble Methods
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional

class MCDropoutModel(nn.Module):
    """Model with Monte Carlo Dropout for uncertainty"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_rate: float = 0.2):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)
    
    def predict_with_uncertainty(self, x: torch.Tensor, n_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with uncertainty using MC Dropout"""
        
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Calculate mean and std
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        self.eval()  # Disable dropout
        
        return mean_pred, std_pred

class EnsembleUncertainty:
    """Ensemble-based uncertainty quantification"""
    
    def __init__(self, models: List[nn.Module]):
        self.models = models
    
    def predict_with_uncertainty(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with ensemble uncertainty"""
        
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred

def uncertainty_filter(predictions: np.ndarray, uncertainties: np.ndarray, confidence_threshold: float = 0.8) -> np.ndarray:
    """Filter predictions based on uncertainty"""
    
    # Convert uncertainty to confidence (inverse relationship)
    max_uncertainty = uncertainties.max()
    confidences = 1.0 - (uncertainties / max_uncertainty)
    
    # Apply confidence threshold
    high_confidence_mask = confidences >= confidence_threshold
    
    return high_confidence_mask
