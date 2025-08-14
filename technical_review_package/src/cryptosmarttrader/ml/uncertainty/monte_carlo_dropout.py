#!/usr/bin/env python3
"""
Monte Carlo Dropout Implementation
Lightweight MC Dropout for fast uncertainty estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")


@dataclass
class MCDropoutResult:
    """Result from MC Dropout inference"""

    mean_prediction: float
    uncertainty: float
    all_samples: List[float]
    confidence_interval: Tuple[float, float]
    mc_samples: int


class MCDropoutInference:
    """Fast Monte Carlo Dropout inference for uncertainty estimation"""

    def __init__(self, model: nn.Module, dropout_rate: float = 0.1):
        self.model = model
        self.dropout_rate = dropout_rate
        self.logger = logging.getLogger(__name__)

        # Enable dropout layers for inference
        self._enable_dropout()

    def _enable_dropout(self):
        """Enable dropout layers during inference"""

        def enable_dropout_layer(module):
            if isinstance(module, nn.Dropout):
                module.train()  # Keep dropout active

        self.model.apply(enable_dropout_layer)

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_samples: int = 100, confidence_level: float = 0.95
    ) -> MCDropoutResult:
        """Perform MC Dropout inference"""

        # Ensure model is in eval mode except for dropout layers
        self.model.eval()
        self._enable_dropout()

        samples = []

        # Collect MC samples
        with torch.no_grad():
            for _ in range(n_samples):
                output = self.model(x)

                # Handle different output types
                if isinstance(output, torch.Tensor):
                    if output.dim() > 1:
                        output = output.squeeze()
                    samples.append(output.item() if output.dim() == 0 else output.cpu().numpy())
                else:
                    samples.append(float(output))

        # Convert to numpy array for easier processing
        samples = np.array(samples)

        # Calculate statistics
        mean_pred = np.mean(samples)
        uncertainty = np.std(samples)

        # Calculate confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.quantile(samples, alpha / 2)
        ci_upper = np.quantile(samples, 1 - alpha / 2)

        return MCDropoutResult(
            mean_prediction=mean_pred,
            uncertainty=uncertainty,
            all_samples=samples.tolist(),
            confidence_interval=(ci_lower, ci_upper),
            mc_samples=n_samples,
        )

    def batch_predict_with_uncertainty(
        self, X: torch.Tensor, n_samples: int = 100, confidence_level: float = 0.95
    ) -> List[MCDropoutResult]:
        """Batch MC Dropout inference"""

        results = []

        for i in range(X.shape[0]):
            x_single = X[i : i + 1]  # Keep batch dimension
            result = self.predict_with_uncertainty(x_single, n_samples, confidence_level)
            results.append(result)

        return results


class FastMCDropout:
    """Lightweight MC Dropout wrapper for existing models"""

    @staticmethod
    def add_dropout_to_model(model: nn.Module, dropout_rate: float = 0.1) -> nn.Module:
        """Add dropout layers to existing model"""

        def add_dropout_after_linear(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Linear):
                    # Add dropout after linear layer
                    setattr(
                        module,
                        name + "_with_dropout",
                        nn.Sequential(child, nn.Dropout(dropout_rate)),
                    )
                else:
                    add_dropout_after_linear(child)

        model_copy = type(model)(**model.__dict__)  # Copy model
        add_dropout_after_linear(model_copy)

        return model_copy

    @staticmethod
    def estimate_uncertainty(
        model: nn.Module, x: torch.Tensor, n_samples: int = 50, dropout_rate: float = 0.1
    ) -> Tuple[float, float]:
        """Quick uncertainty estimation for any model"""

        # Enable dropout temporarily
        def enable_dropout(module):
            if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                module.train()

        model.eval()
        model.apply(enable_dropout)

        predictions = []

        with torch.no_grad():
            for _ in range(n_samples):
                # Apply manual dropout if no dropout layers present
                output = model(x)

                if not any(isinstance(m, nn.Dropout) for m in model.modules()):
                    # Manual dropout simulation
                    output = F.dropout(output, p=dropout_rate, training=True)

                predictions.append(output.item() if output.dim() == 0 else output.cpu().numpy())

        predictions = np.array(predictions)
        mean_pred = np.mean(predictions)
        uncertainty = np.std(predictions)

        return mean_pred, uncertainty


class UncertaintyAwarePredictor:
    """Wrapper to add uncertainty to any prediction model"""

    def __init__(self, base_model: Any, uncertainty_method: str = "mc_dropout"):
        self.base_model = base_model
        self.uncertainty_method = uncertainty_method
        self.mc_samples = 100

    def predict_with_confidence(
        self, X: np.ndarray, return_uncertainty: bool = True
    ) -> Dict[str, Any]:
        """Predict with uncertainty estimation"""

        # Get base prediction
        if hasattr(self.base_model, "predict"):
            base_pred = self.base_model.predict(X)
        elif hasattr(self.base_model, "__call__"):
            base_pred = self.base_model(X)
        else:
            raise ValueError("Model must have predict method or be callable")

        if not return_uncertainty:
            return {"prediction": base_pred, "uncertainty": 0.0}

        # Estimate uncertainty based on method
        if self.uncertainty_method == "mc_dropout":
            uncertainty = self._estimate_mc_dropout_uncertainty(X)
        elif self.uncertainty_method == "bootstrap":
            uncertainty = self._estimate_bootstrap_uncertainty(X)
        elif self.uncertainty_method == "variance":
            uncertainty = self._estimate_model_variance(X)
        else:
            uncertainty = 0.1  # Default uncertainty

        return {
            "prediction": base_pred,
            "uncertainty": uncertainty,
            "confidence_interval": self._calculate_confidence_interval(base_pred, uncertainty),
        }

    def _estimate_mc_dropout_uncertainty(self, X: np.ndarray) -> float:
        """Estimate uncertainty using MC Dropout approximation"""

        if isinstance(self.base_model, torch.nn.Module):
            # PyTorch model - use proper MC Dropout
            X_tensor = torch.FloatTensor(X)
            mc_inference = MCDropoutInference(self.base_model)
            result = mc_inference.predict_with_uncertainty(X_tensor, self.mc_samples)
            return result.uncertainty

        else:
            # Non-PyTorch model - simulate with multiple predictions
            predictions = []

            for _ in range(min(self.mc_samples, 20)):  # Limit for non-PyTorch models
                try:
                    # Add small noise to simulate dropout effect
                    noise_scale = 0.01
                    if hasattr(X, "shape") and len(X.shape) > 1:
                        X_noisy = X + np.random.normal(0, 1)
                    else:
                        X_noisy = X + np.random.normal(0, 1)

                    pred = self.base_model.predict(X_noisy)
                    predictions.append(pred)

                except Exception:
                    # If adding noise fails, just repeat prediction
                    pred = self.base_model.predict(X)
                    predictions.append(pred)

            if predictions:
                return np.std(predictions)
            else:
                return 0.1  # Default uncertainty

    def _estimate_bootstrap_uncertainty(self, X: np.ndarray) -> float:
        """Estimate uncertainty using bootstrap sampling"""

        # This would require retraining on bootstrap samples
        # For now, return a heuristic uncertainty
        if hasattr(X, "shape") and len(X.shape) > 1:
            feature_variance = np.var(X, axis=1).mean()
        else:
            feature_variance = np.var(X) if hasattr(X, "__len__") else 0.1

        # Scale feature variance to prediction uncertainty
        return np.sqrt(feature_variance) * 0.1

    def _estimate_model_variance(self, X: np.ndarray) -> float:
        """Estimate uncertainty from model internals if available"""

        # Check if model has uncertainty estimation
        if hasattr(self.base_model, "predict_proba"):
            # For classifiers, use prediction probability as confidence proxy
            probas = self.base_model.predict_proba(X)
            max_proba = np.max(probas, axis=1)
            # Convert to uncertainty (higher max proba = lower uncertainty)
            uncertainty = 1.0 - max_proba
            return np.mean(uncertainty)

        elif hasattr(self.base_model, "decision_function"):
            # For SVMs, use decision function distance
            decision_scores = self.base_model.decision_function(X)
            # Lower absolute scores = higher uncertainty
            uncertainty = 1.0 / (1.0 + np.abs(decision_scores))
            return np.mean(uncertainty)

        else:
            # Default heuristic uncertainty
            return 0.1

    def _calculate_confidence_interval(
        self, prediction: float, uncertainty: float, confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval from prediction and uncertainty"""

        # Assume normal distribution for confidence interval
        from scipy import stats

        # Z-score for confidence level
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha / 2)

        margin = z_score * uncertainty

        return (prediction - margin, prediction + margin)


def add_uncertainty_to_model(
    model: Any, uncertainty_method: str = "mc_dropout", mc_samples: int = 100
) -> UncertaintyAwarePredictor:
    """Add uncertainty estimation to any model"""

    wrapper = UncertaintyAwarePredictor(model, uncertainty_method)
    wrapper.mc_samples = mc_samples

    return wrapper


def quick_uncertainty_estimate(model: Any, X: np.ndarray, n_samples: int = 50) -> Dict[str, float]:
    """Quick uncertainty estimation for any model"""

    # Get base prediction
    if hasattr(model, "predict"):
        base_prediction = model.predict(X)
    else:
        raise ValueError("Model must have predict method")

    # Estimate uncertainty with simple bootstrap-like method
    predictions = []

    for _ in range(n_samples):
        try:
            # Add small perturbation to input
            if hasattr(X, "shape") and len(X.shape) > 1:
                X_perturbed = X + np.random.normal(0, 1)
            else:
                X_perturbed = X + np.random.normal(0, 1)

            pred = model.predict(X_perturbed)
            predictions.append(pred)

        except Exception:
            # If perturbation fails, use original prediction
            predictions.append(base_prediction)

    predictions = np.array(predictions)

    return {
        "prediction": float(np.mean(predictions)),
        "uncertainty": float(np.std(predictions)),
        "confidence_interval": (
            float(np.quantile(predictions, 0.025)),
            float(np.quantile(predictions, 0.975)),
        ),
    }
