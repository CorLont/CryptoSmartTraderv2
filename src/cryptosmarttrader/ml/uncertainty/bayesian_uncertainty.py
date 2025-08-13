#!/usr/bin/env python3
"""
Bayesian Uncertainty Quantification
Monte Carlo Dropout, ensemble modeling, and epistemic/aleatoric uncertainty
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings
from scipy import stats
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


@dataclass
class UncertaintyEstimate:
    """Comprehensive uncertainty estimate"""

    prediction: float
    epistemic_uncertainty: float  # Model uncertainty (reducible with more data)
    aleatoric_uncertainty: float  # Data uncertainty (irreducible noise)
    total_uncertainty: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    prediction_variance: float
    ensemble_predictions: List[float] = field(default_factory=list)
    mc_samples: int = 100
    prediction_method: str = "unknown"


@dataclass
class UncertaintyMetrics:
    """Model uncertainty quality metrics"""

    prediction_interval_coverage: float  # % of true values in predicted intervals
    interval_sharpness: float  # Average interval width
    calibration_error: float  # Difference between predicted and actual coverage
    uncertainty_correlation: float  # Correlation between uncertainty and error
    reliability_score: float  # Overall reliability of uncertainty estimates


class MCDropoutLayer(nn.Module):
    """Monte Carlo Dropout layer for uncertainty estimation"""

    def __init__(self, p: float = 0.5):
        super(MCDropoutLayer, self).__init__()
        self.p = p

    def forward(self, x):
        # Always apply dropout during inference for MC sampling
        return F.dropout(x, p=self.p, training=True)


class BayesianLinear(nn.Module):
    """Bayesian linear layer with weight uncertainty"""

    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        # Weight mean and log variance parameters
        self.weight_mean = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 5)

        # Bias mean and log variance parameters
        self.bias_mean = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_logvar = nn.Parameter(torch.randn(out_features) * 0.1 - 5)

    def forward(self, x):
        # Sample weights and biases from posterior
        weight_std = torch.exp(0.5 * self.weight_logvar)
        bias_std = torch.exp(0.5 * self.bias_logvar)

        weight_eps = torch.randn_like(self.weight_mean)
        bias_eps = torch.randn_like(self.bias_mean)

        weight = self.weight_mean + weight_std * weight_eps
        bias = self.bias_mean + bias_std * bias_eps

        return F.linear(x, weight, bias)

    def kl_divergence(self):
        """Calculate KL divergence for variational inference"""
        # KL divergence between posterior and prior
        weight_var = torch.exp(self.weight_logvar)
        bias_var = torch.exp(self.bias_logvar)

        weight_kl = 0.5 * torch.sum(
            (self.weight_mean**2 + weight_var) / (self.prior_std**2)
            - 1
            - torch.log(weight_var)
            + 2 * torch.log(torch.tensor(self.prior_std))

        bias_kl = 0.5 * torch.sum(
            (self.bias_mean**2 + bias_var) / (self.prior_std**2)
            - 1
            - torch.log(bias_var)
            + 2 * torch.log(torch.tensor(self.prior_std))

        return weight_kl + bias_kl


class MCDropoutModel(nn.Module):
    """Neural network with MC Dropout for uncertainty estimation"""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout_rate: float = 0.1,
        use_bayesian: bool = False,
    ):
        super(MCDropoutModel, self).__init__()
        self.use_bayesian = use_bayesian

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            if use_bayesian:
                layers.append(BayesianLinear(prev_size, hidden_size))
            else:
                layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(MCDropoutLayer(dropout_rate))
            prev_size = hidden_size

        # Output layer
        if use_bayesian:
            layers.append(BayesianLinear(prev_size, output_size))
        else:
            layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def kl_loss(self):
        """Calculate total KL divergence for Bayesian layers"""
        if not self.use_bayesian:
            return torch.tensor(0.0)

        kl_sum = 0.0
        for layer in self.network:
            if isinstance(layer, BayesianLinear):
                kl_sum += layer.kl_divergence()

        return kl_sum


class EnsembleModel:
    """Ensemble of models for uncertainty estimation"""

    def __init__(self, model_configs: List[Dict[str, Any]], device: str = "cpu"):
        self.models = []
        self.device = device
        self.trained = False

        # Create ensemble of models
        for i, config in enumerate(model_configs):
            model = MCDropoutModel(**config)
            model.to(device)
            self.models.append(model)

        self.logger = logging.getLogger(__name__)

    def train_ensemble(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 64,
    ):
        """Train ensemble of models with different initializations"""

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        if X_val is not None:
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)

        for i, model in enumerate(self.models):
            self.logger.info(f"Training ensemble model {i + 1}/{len(self.models)}")

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            # Train with different bootstrap samples
            n_samples = len(X_train)
            bootstrap_indices = torch.randint(0, n_samples, (n_samples,))
            X_bootstrap = X_train[bootstrap_indices]
            y_bootstrap = y_train[bootstrap_indices]

            model.train()
            for epoch in range(epochs):
                # Mini-batch training
                for batch_start in range(0, len(X_bootstrap), batch_size):
                    batch_end = min(batch_start + batch_size, len(X_bootstrap))
                    X_batch = X_bootstrap[batch_start:batch_end]
                    y_batch = y_bootstrap[batch_start:batch_end]

                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch.unsqueeze(1))

                    # Add KL divergence for Bayesian models
                    if model.use_bayesian:
                        kl_loss = model.kl_loss()
                        loss += kl_loss / len(X_batch)  # Scale by batch size

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                # Validation
                if X_val is not None and epoch % 20 == 0:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val)
                        val_loss = criterion(val_outputs, y_val.unsqueeze(1))
                    model.train()

                    if epoch % 50 == 0:
                        self.logger.debug(
                            f"Model {i + 1}, Epoch {epoch}: Val Loss = {val_loss:.4f}"
                        )

        self.trained = True

    def predict_with_uncertainty(
        self, X: torch.Tensor, mc_samples: int = 100, confidence_level: float = 0.95
    ) -> UncertaintyEstimate:
        """Predict with comprehensive uncertainty estimation"""

        if not self.trained:
            raise ValueError("Ensemble must be trained before prediction")

        X = X.to(self.device)

        # Collect predictions from all ensemble models and MC samples
        all_predictions = []

        for model in self.models:
            model.eval()  # But MC dropout still active

            # Multiple MC samples per model
            model_predictions = []
            for _ in range(mc_samples):
                with torch.no_grad():
                    pred = model(X).cpu().numpy().flatten()
                    model_predictions.append(pred)

            all_predictions.extend(model_predictions)

        # Convert to numpy array
        all_predictions = np.array(all_predictions)  # Shape: (ensemble_size * mc_samples, n_points)

        # Calculate statistics for each input point
        predictions = []
        for i in range(X.shape[0]):
            point_predictions = all_predictions[:, i]

            # Basic statistics
            mean_pred = np.mean(point_predictions)
            prediction_variance = np.var(point_predictions)
            total_uncertainty = np.std(point_predictions)

            # Decompose uncertainty
            epistemic_uncertainty, aleatoric_uncertainty = self._decompose_uncertainty(
                point_predictions, mc_samples
            )

            # Confidence intervals
            alpha = 1 - confidence_level
            lower_quantile = alpha / 2
            upper_quantile = 1 - alpha / 2

            ci_lower = np.quantile(point_predictions, lower_quantile)
            ci_upper = np.quantile(point_predictions, upper_quantile)

            uncertainty_est = UncertaintyEstimate(
                prediction=mean_pred,
                epistemic_uncertainty=epistemic_uncertainty,
                aleatoric_uncertainty=aleatoric_uncertainty,
                total_uncertainty=total_uncertainty,
                confidence_interval_lower=ci_lower,
                confidence_interval_upper=ci_upper,
                prediction_variance=prediction_variance,
                ensemble_predictions=point_predictions.tolist(),
                mc_samples=mc_samples * len(self.models),
                prediction_method="ensemble_mc_dropout",
            )

            predictions.append(uncertainty_est)

        return predictions if len(predictions) > 1 else predictions[0]

    def _decompose_uncertainty(
        self, predictions: np.ndarray, mc_samples: int
    ) -> Tuple[float, float]:
        """Decompose uncertainty into epistemic and aleatoric components"""

        n_models = len(self.models)

        if len(predictions) != n_models * mc_samples:
            # Fallback: treat all as epistemic uncertainty
            return np.std(predictions), 0.0

        # Reshape to separate ensemble and MC dimensions
        reshaped = predictions.reshape(n_models, mc_samples)

        # Epistemic uncertainty: variance across ensemble models
        ensemble_means = np.mean(reshaped, axis=1)
        epistemic_uncertainty = np.std(ensemble_means)

        # Aleatoric uncertainty: average within-model variance
        within_model_vars = np.var(reshaped, axis=1)
        aleatoric_uncertainty = np.sqrt(np.mean(within_model_vars))

        return epistemic_uncertainty, aleatoric_uncertainty


class UncertaintyQuantifier:
    """Complete uncertainty quantification system"""

    def __init__(
        self,
        model_type: str = "ensemble_mc_dropout",
        ensemble_size: int = 5,
        mc_samples: int = 100,
        device: str = "cpu",
    ):
        self.model_type = model_type
        self.ensemble_size = ensemble_size
        self.mc_samples = mc_samples
        self.device = device

        self.model = None
        self.is_trained = False
        self.uncertainty_calibration = None

        self.logger = logging.getLogger(__name__)

    def build_model(
        self,
        input_size: int,
        hidden_sizes: List[int] = None,
        output_size: int = 1,
        dropout_rate: float = 0.1,
    ):
        """Build uncertainty-aware model"""

        if hidden_sizes is None:
            hidden_sizes = [128, 64, 32]

        if self.model_type == "ensemble_mc_dropout":
            # Create ensemble of MC Dropout models
            model_configs = []
            for i in range(self.ensemble_size):
                config = {
                    "input_size": input_size,
                    "hidden_sizes": hidden_sizes,
                    "output_size": output_size,
                    "dropout_rate": dropout_rate + i * 0.02,  # Vary dropout rates
                    "use_bayesian": i % 2 == 0,  # Mix Bayesian and regular layers
                }
                model_configs.append(config)

            self.model = EnsembleModel(model_configs, self.device)

        elif self.model_type == "single_mc_dropout":
            # Single MC Dropout model
            self.model = MCDropoutModel(input_size, hidden_sizes, output_size, dropout_rate).to(
                self.device
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 64,
    ):
        """Train uncertainty-aware model"""

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train)

        X_val_tensor = None
        y_val_tensor = None
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.FloatTensor(y_val)

        if isinstance(self.model, EnsembleModel):
            self.model.train_ensemble(
                X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, epochs, lr, batch_size
            )
        else:
            # Train single model
            self._train_single_model(
                X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, epochs, lr, batch_size
            )

        self.is_trained = True

    def _train_single_model(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor],
        y_val: Optional[torch.Tensor],
        epochs: int,
        lr: float,
        batch_size: int,
    ):
        """Train single MC Dropout model"""

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        if X_val is not None:
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)

        self.model.train()
        for epoch in range(epochs):
            for batch_start in range(0, len(X_train), batch_size):
                batch_end = min(batch_start + batch_size, len(X_train))
                X_batch = X_train[batch_start:batch_end]
                y_batch = y_train[batch_start:batch_end]

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch.unsqueeze(1))

                # Add KL divergence for Bayesian models
                if hasattr(self.model, "kl_loss"):
                    kl_loss = self.model.kl_loss()
                    loss += kl_loss / len(X_batch)

                loss.backward()
                optimizer.step()

            # Validation
            if X_val is not None and epoch % 20 == 0:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val)
                    val_loss = criterion(val_outputs, y_val.unsqueeze(1))
                self.model.train()

                if epoch % 50 == 0:
                    self.logger.debug(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")

    def predict_with_uncertainty(
        self, X: np.ndarray, confidence_level: float = 0.95
    ) -> Union[UncertaintyEstimate, List[UncertaintyEstimate]]:
        """Predict with uncertainty quantification"""

        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        X_tensor = torch.FloatTensor(X)

        if isinstance(self.model, EnsembleModel):
            return self.model.predict_with_uncertainty(X_tensor, self.mc_samples, confidence_level)
        else:
            return self._predict_single_mc_dropout(X_tensor, confidence_level)

    def _predict_single_mc_dropout(
        self, X: torch.Tensor, confidence_level: float
    ) -> Union[UncertaintyEstimate, List[UncertaintyEstimate]]:
        """Predict with single MC Dropout model"""

        X = X.to(self.device)

        # Collect MC samples
        predictions = []
        self.model.eval()  # But dropout still active in MCDropoutLayer

        for _ in range(self.mc_samples):
            with torch.no_grad():
                pred = self.model(X).cpu().numpy().flatten()
                predictions.append(pred)

        predictions = np.array(predictions)  # Shape: (mc_samples, n_points)

        # Calculate uncertainty for each point
        results = []
        for i in range(X.shape[0]):
            point_predictions = predictions[:, i]

            mean_pred = np.mean(point_predictions)
            total_uncertainty = np.std(point_predictions)
            prediction_variance = np.var(point_predictions)

            # For single model, treat all uncertainty as epistemic
            epistemic_uncertainty = total_uncertainty
            aleatoric_uncertainty = 0.0

            # Confidence intervals
            alpha = 1 - confidence_level
            ci_lower = np.quantile(point_predictions, alpha / 2)
            ci_upper = np.quantile(point_predictions, 1 - alpha / 2)

            uncertainty_est = UncertaintyEstimate(
                prediction=mean_pred,
                epistemic_uncertainty=epistemic_uncertainty,
                aleatoric_uncertainty=aleatoric_uncertainty,
                total_uncertainty=total_uncertainty,
                confidence_interval_lower=ci_lower,
                confidence_interval_upper=ci_upper,
                prediction_variance=prediction_variance,
                ensemble_predictions=point_predictions.tolist(),
                mc_samples=self.mc_samples,
                prediction_method="single_mc_dropout",
            )

            results.append(uncertainty_est)

        return results if len(results) > 1 else results[0]

    def calibrate_uncertainty(
        self, X_cal: np.ndarray, y_cal: np.ndarray, confidence_levels: List[float] = None
    ) -> Dict[str, float]:
        """Calibrate uncertainty estimates using calibration data"""

        if confidence_levels is None:
            confidence_levels = [0.5, 0.68, 0.95, 0.99]

        # Get predictions with uncertainty
        uncertainty_predictions = self.predict_with_uncertainty(X_cal)

        if not isinstance(uncertainty_predictions, list):
            uncertainty_predictions = [uncertainty_predictions]

        # Calculate calibration metrics
        calibration_results = {}

        for conf_level in confidence_levels:
            # Count how many true values fall within predicted intervals
            coverage_count = 0
            interval_widths = []

            for i, uncertainty_pred in enumerate(uncertainty_predictions):
                true_value = y_cal[i]

                # Calculate interval for this confidence level
                alpha = 1 - conf_level
                predictions_array = np.array(uncertainty_pred.ensemble_predictions)
                ci_lower = np.quantile(predictions_array, alpha / 2)
                ci_upper = np.quantile(predictions_array, 1 - alpha / 2)

                # Check coverage
                if ci_lower <= true_value <= ci_upper:
                    coverage_count += 1

                # Record interval width
                interval_widths.append(ci_upper - ci_lower)

            # Calculate metrics
            actual_coverage = coverage_count / len(uncertainty_predictions)
            avg_interval_width = np.mean(interval_widths)
            calibration_error = abs(actual_coverage - conf_level)

            calibration_results[f"coverage_{conf_level}"] = actual_coverage
            calibration_results[f"width_{conf_level}"] = avg_interval_width
            calibration_results[f"error_{conf_level}"] = calibration_error

        # Store calibration info
        self.uncertainty_calibration = calibration_results

        return calibration_results

    def get_uncertainty_metrics(self, X_test: np.ndarray, y_test: np.ndarray) -> UncertaintyMetrics:
        """Calculate comprehensive uncertainty quality metrics"""

        uncertainty_predictions = self.predict_with_uncertainty(X_test)

        if not isinstance(uncertainty_predictions, list):
            uncertainty_predictions = [uncertainty_predictions]

        # Extract predictions and uncertainties
        predictions = np.array([up.prediction for up in uncertainty_predictions])
        uncertainties = np.array([up.total_uncertainty for up in uncertainty_predictions])

        # Calculate prediction errors
        errors = np.abs(predictions - y_test)

        # Prediction interval coverage (95% level)
        coverage_count = 0
        interval_widths = []

        for i, uncertainty_pred in enumerate(uncertainty_predictions):
            true_value = y_test[i]
            ci_lower = uncertainty_pred.confidence_interval_lower
            ci_upper = uncertainty_pred.confidence_interval_upper

            if ci_lower <= true_value <= ci_upper:
                coverage_count += 1

            interval_widths.append(ci_upper - ci_lower)

        coverage = coverage_count / len(uncertainty_predictions)
        sharpness = np.mean(interval_widths)

        # Calibration error (should be close to 0.05 for 95% intervals)
        calibration_error = abs(coverage - 0.95)

        # Correlation between uncertainty and error
        uncertainty_correlation = np.corrcoef(uncertainties, errors)[0, 1]
        if np.isnan(uncertainty_correlation):
            uncertainty_correlation = 0.0

        # Overall reliability score
        reliability_score = (
            coverage * 0.4  # Good coverage
            + (1 - calibration_error) * 0.3  # Well calibrated
            + max(0, uncertainty_correlation) * 0.3  # Uncertainty correlates with error
        )

        return UncertaintyMetrics(
            prediction_interval_coverage=coverage,
            interval_sharpness=sharpness,
            calibration_error=calibration_error,
            uncertainty_correlation=uncertainty_correlation,
            reliability_score=reliability_score,
        )


def create_uncertainty_quantifier(
    model_type: str = "ensemble_mc_dropout", ensemble_size: int = 5, mc_samples: int = 100
) -> UncertaintyQuantifier:
    """Create uncertainty quantifier with specified configuration"""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    return UncertaintyQuantifier(
        model_type=model_type, ensemble_size=ensemble_size, mc_samples=mc_samples, device=device
    )


def quantify_prediction_uncertainty(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    model_config: Dict[str, Any] = None,
) -> List[UncertaintyEstimate]:
    """High-level function for uncertainty quantification"""

    if model_config is None:
        model_config = {"model_type": "ensemble_mc_dropout", "ensemble_size": 3, "mc_samples": 50}

    # Create and train model
    quantifier = create_uncertainty_quantifier(**model_config)
    quantifier.build_model(input_size=X_train.shape[1])
    quantifier.train(X_train, y_train, epochs=50)

    # Get predictions with uncertainty
    return quantifier.predict_with_uncertainty(X_test)
