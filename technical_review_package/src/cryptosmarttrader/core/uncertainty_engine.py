#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Uncertainty Modeling Engine
Bayesian neural networks, quantile regression, ensemble spread for prediction confidence
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import warnings

warnings.filterwarnings("ignore")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal, MultivariateNormal
    import torch.optim as optim

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from scipy import stats
    from scipy.optimize import minimize

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class UncertaintyMethod(Enum):
    BAYESIAN_NN = "bayesian_nn"
    QUANTILE_REGRESSION = "quantile_regression"
    ENSEMBLE_SPREAD = "ensemble_spread"
    GAUSSIAN_PROCESS = "gaussian_process"
    MONTE_CARLO_DROPOUT = "mc_dropout"
    DEEP_ENSEMBLE = "deep_ensemble"


class ConfidenceLevel(Enum):
    VERY_LOW = 0.1  # < 10% confidence
    LOW = 0.3  # 10-30% confidence
    MEDIUM = 0.5  # 30-50% confidence
    HIGH = 0.7  # 50-70% confidence
    VERY_HIGH = 0.9  # > 70% confidence


@dataclass
class UncertaintyConfig:
    """Configuration for uncertainty modeling"""

    # Bayesian settings
    prior_sigma: float = 1.0
    posterior_samples: int = 100
    variational_samples: int = 10

    # Quantile regression
    quantiles: List[float] = field(default_factory=lambda: [0.05, 0.25, 0.5, 0.75, 0.95])
    pinball_loss_alpha: float = 0.1

    # Ensemble settings
    num_ensemble_models: int = 10
    ensemble_diversity_weight: float = 0.1

    # Monte Carlo dropout
    mc_dropout_samples: int = 100
    dropout_rate: float = 0.2

    # Confidence thresholds
    min_confidence_threshold: float = 0.7
    uncertainty_decay_factor: float = 0.95
    max_prediction_interval: float = 0.3  # 30% interval width

    # Trading decision thresholds
    trade_confidence_threshold: float = 0.8
    uncertainty_penalty_factor: float = 2.0


@dataclass
class UncertaintyPrediction:
    """Prediction with uncertainty quantification"""

    mean_prediction: float
    confidence: float
    uncertainty: float
    prediction_interval: Tuple[float, float]
    quantile_predictions: Dict[float, float]
    method_used: UncertaintyMethod
    timestamp: datetime
    model_confidence: float = 0.0
    ensemble_agreement: float = 0.0
    aleatoric_uncertainty: float = 0.0  # Data noise
    epistemic_uncertainty: float = 0.0  # Model uncertainty


class BayesianLinear(nn.Module):
    """Bayesian linear layer with variational inference"""

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Weight parameters (mean and log variance)
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_logvar = nn.Parameter(torch.randn(out_features, in_features) * 0.1 - 3.0)

        # Bias parameters
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_logvar = nn.Parameter(torch.randn(out_features) * 0.1 - 3.0)

        # Prior distributions
        self.weight_prior = Normal(0, prior_sigma)
        self.bias_prior = Normal(0, prior_sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sample weights and biases
        weight_sigma = torch.exp(0.5 * self.weight_logvar)
        bias_sigma = torch.exp(0.5 * self.bias_logvar)

        weight_eps = torch.randn_like(weight_sigma)
        bias_eps = torch.randn_like(bias_sigma)

        weight = self.weight_mu + weight_sigma * weight_eps
        bias = self.bias_mu + bias_sigma * bias_eps

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        """Calculate KL divergence between posterior and prior"""
        # Weight KL
        weight_var = torch.exp(self.weight_logvar)
        weight_kl = 0.5 * torch.sum(
            self.weight_logvar
            - torch.log(torch.tensor(self.prior_sigma**2))
            + (weight_var + self.weight_mu**2) / self.prior_sigma**2
            - 1
        )

        # Bias KL
        bias_var = torch.exp(self.bias_logvar)
        bias_kl = 0.5 * torch.sum(
            self.bias_logvar
            - torch.log(torch.tensor(self.prior_sigma**2))
            + (bias_var + self.bias_mu**2) / self.prior_sigma**2
            - 1
        )

        return weight_kl + bias_kl


class BayesianNeuralNetwork(nn.Module):
    """Bayesian neural network for uncertainty quantification"""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        prior_sigma: float = 1.0,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        # Input layer
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(BayesianLinear(current_dim, hidden_dim, prior_sigma))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        # Output layer
        self.layers.append(BayesianLinear(current_dim, output_dim, prior_sigma))

        # Activation function
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, (layer, dropout) in enumerate(zip(self.layers[:-1], self.dropout_layers)):
            x = dropout(self.activation(layer(x)))

        # Output layer (no activation for regression)
        x = self.layers[-1](x)
        return x

    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence for all layers"""
        kl = 0
        for layer in self.layers:
            if isinstance(layer, BayesianLinear):
                kl += layer.kl_divergence()
        return kl


class QuantileRegressionNetwork(nn.Module):
    """Neural network for quantile regression"""

    def __init__(self, input_dim: int, hidden_dims: List[int], quantiles: List[float]):
        super().__init__()

        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        # Shared layers
        self.shared_layers = nn.ModuleList()
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            self.shared_layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim

        # Quantile-specific heads
        self.quantile_heads = nn.ModuleList([nn.Linear(current_dim, 1) for _ in quantiles])

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Shared representation
        for layer in self.shared_layers:
            x = self.dropout(self.activation(layer(x)))

        # Quantile predictions
        outputs = []
        for head in self.quantile_heads:
            outputs.append(head(x))

        return torch.cat(outputs, dim=-1)

    def quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Pinball loss for quantile regression"""
        targets = targets.unsqueeze(-1).expand_as(predictions)

        losses = []
        for i, tau in enumerate(self.quantiles):
            pred = predictions[:, i : i + 1]
            error = targets[:, i : i + 1] - pred
            loss = torch.max(tau * error, (tau - 1) * error)
            losses.append(loss)

        return torch.cat(losses, dim=-1).mean()


class MCDropoutModel(nn.Module):
    """Monte Carlo Dropout model for uncertainty estimation"""

    def __init__(
        self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout_rate: float = 0.2
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        current_dim = input_dim
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(current_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer, dropout in zip(self.layers, self.dropout_layers):
            x = dropout(self.activation(layer(x)))

        return self.output_layer(x)

    def predict_with_uncertainty(
        self, x: torch.Tensor, num_samples: int = 100
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict with uncertainty using Monte Carlo sampling"""
        self.train()  # Enable dropout

        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(x)
                predictions.append(pred)

        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)

        self.eval()  # Disable dropout for normal inference
        return mean_pred, uncertainty


class UncertaintyEngine:
    """Main uncertainty modeling and quantification engine"""

    def __init__(self, config: Optional[UncertaintyConfig] = None):
        self.config = config or UncertaintyConfig()
        self.logger = logging.getLogger(f"{__name__}.UncertaintyEngine")

        if not HAS_TORCH:
            self.logger.error("PyTorch not available - uncertainty modeling disabled")
            return

        # Model registry
        self.uncertainty_models: Dict[str, Dict] = {}
        self.ensemble_models: Dict[str, List[nn.Module]] = {}

        # Uncertainty history
        self.prediction_history: List[UncertaintyPrediction] = []
        self.confidence_history: List[float] = []

        # Calibration data
        self.calibration_data: Dict[str, List] = {
            "predictions": [],
            "uncertainties": [],
            "actuals": [],
            "errors": [],
        }

        # Performance tracking
        self.uncertainty_metrics: Dict[str, float] = {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._lock = threading.RLock()

        self.logger.info(f"Uncertainty Engine initialized on {self.device}")

    def create_bayesian_model(
        self, model_id: str, input_dim: int, hidden_dims: List[int], output_dim: int = 1
    ) -> bool:
        """Create Bayesian neural network model"""
        with self._lock:
            try:
                if not HAS_TORCH:
                    return False

                model = BayesianNeuralNetwork(
                    input_dim,
                    hidden_dims,
                    output_dim,
                    self.config.prior_sigma,
                    self.config.dropout_rate,
                ).to(self.device)

                self.uncertainty_models[model_id] = {
                    "model": model,
                    "type": UncertaintyMethod.BAYESIAN_NN,
                    "trained": False,
                    "calibrated": False,
                }

                self.logger.info(f"Created Bayesian NN model: {model_id}")
                return True

            except Exception as e:
                self.logger.error(f"Failed to create Bayesian model {model_id}: {e}")
                return False

    def create_quantile_model(self, model_id: str, input_dim: int, hidden_dims: List[int]) -> bool:
        """Create quantile regression model"""
        with self._lock:
            try:
                if not HAS_TORCH:
                    return False

                model = QuantileRegressionNetwork(input_dim, hidden_dims, self.config.quantiles).to(
                    self.device
                )

                self.uncertainty_models[model_id] = {
                    "model": model,
                    "type": UncertaintyMethod.QUANTILE_REGRESSION,
                    "trained": False,
                    "calibrated": False,
                }

                self.logger.info(f"Created Quantile Regression model: {model_id}")
                return True

            except Exception as e:
                self.logger.error(f"Failed to create quantile model {model_id}: {e}")
                return False

    def create_mc_dropout_model(
        self, model_id: str, input_dim: int, hidden_dims: List[int], output_dim: int = 1
    ) -> bool:
        """Create Monte Carlo Dropout model"""
        with self._lock:
            try:
                if not HAS_TORCH:
                    return False

                model = MCDropoutModel(
                    input_dim, hidden_dims, output_dim, self.config.dropout_rate
                ).to(self.device)

                self.uncertainty_models[model_id] = {
                    "model": model,
                    "type": UncertaintyMethod.MONTE_CARLO_DROPOUT,
                    "trained": False,
                    "calibrated": False,
                }

                self.logger.info(f"Created MC Dropout model: {model_id}")
                return True

            except Exception as e:
                self.logger.error(f"Failed to create MC dropout model {model_id}: {e}")
                return False

    def create_ensemble(self, ensemble_id: str, base_models: List[nn.Module]) -> bool:
        """Create model ensemble for uncertainty estimation"""
        with self._lock:
            try:
                if not base_models:
                    return False

                # Move models to device
                ensemble = [model.to(self.device) for model in base_models]

                self.ensemble_models[ensemble_id] = ensemble

                self.uncertainty_models[ensemble_id] = {
                    "model": None,  # Ensemble doesn't have single model
                    "type": UncertaintyMethod.ENSEMBLE_SPREAD,
                    "trained": True,  # Assume base models are trained
                    "calibrated": False,
                }

                self.logger.info(f"Created ensemble: {ensemble_id} with {len(ensemble)} models")
                return True

            except Exception as e:
                self.logger.error(f"Failed to create ensemble {ensemble_id}: {e}")
                return False

    def train_uncertainty_model(
        self,
        model_id: str,
        train_data: torch.Tensor,
        train_targets: torch.Tensor,
        val_data: Optional[torch.Tensor] = None,
        val_targets: Optional[torch.Tensor] = None,
        epochs: int = 100,
    ) -> bool:
        """Train uncertainty model with appropriate loss function"""
        with self._lock:
            try:
                if model_id not in self.uncertainty_models:
                    return False

                model_info = self.uncertainty_models[model_id]
                model = model_info["model"]
                model_type = model_info["type"]

                if model is None:  # Ensemble
                    return True  # Assume base models are trained

                model.train()
                optimizer = optim.Adam(model.parameters(), lr=1e-3)

                # Training loop
                best_loss = float("inf")
                patience = 0

                for epoch in range(epochs):
                    optimizer.zero_grad()

                    if model_type == UncertaintyMethod.BAYESIAN_NN:
                        # Bayesian training with ELBO loss
                        predictions = model(train_data)
                        mse_loss = F.mse_loss(predictions, train_targets)
                        kl_loss = model.kl_divergence() / len(train_data)
                        loss = mse_loss + kl_loss

                    elif model_type == UncertaintyMethod.QUANTILE_REGRESSION:
                        # Quantile loss
                        predictions = model(train_data)
                        loss = model.quantile_loss(predictions, train_targets)

                    else:  # Standard MSE for MC Dropout
                        predictions = model(train_data)
                        loss = F.mse_loss(predictions, train_targets)

                    loss.backward()
                    optimizer.step()

                    # Validation
                    if val_data is not None and val_targets is not None:
                        model.eval()
                        with torch.no_grad():
                            val_pred = model(val_data)
                            if model_type == UncertaintyMethod.QUANTILE_REGRESSION:
                                val_loss = model.quantile_loss(val_pred, val_targets)
                            else:
                                val_loss = F.mse_loss(val_pred, val_targets)
                        model.train()

                        if val_loss < best_loss:
                            best_loss = val_loss
                            patience = 0
                        else:
                            patience += 1

                        if patience >= 10:  # Early stopping
                            break

                model.eval()
                model_info["trained"] = True

                self.logger.info(f"Training completed for uncertainty model: {model_id}")
                return True

            except Exception as e:
                self.logger.error(f"Training failed for uncertainty model {model_id}: {e}")
                return False

    def predict_with_uncertainty(
        self, model_id: str, input_data: torch.Tensor
    ) -> Optional[UncertaintyPrediction]:
        """Make prediction with comprehensive uncertainty quantification"""
        try:
            if model_id not in self.uncertainty_models:
                return None

            model_info = self.uncertainty_models[model_id]
            model_type = model_info["type"]

            if model_type == UncertaintyMethod.BAYESIAN_NN:
                return self._bayesian_prediction(model_id, input_data)

            elif model_type == UncertaintyMethod.QUANTILE_REGRESSION:
                return self._quantile_prediction(model_id, input_data)

            elif model_type == UncertaintyMethod.MONTE_CARLO_DROPOUT:
                return self._mc_dropout_prediction(model_id, input_data)

            elif model_type == UncertaintyMethod.ENSEMBLE_SPREAD:
                return self._ensemble_prediction(model_id, input_data)

            return None

        except Exception as e:
            self.logger.error(f"Prediction failed for {model_id}: {e}")
            return None

    def _bayesian_prediction(
        self, model_id: str, input_data: torch.Tensor
    ) -> UncertaintyPrediction:
        """Bayesian neural network prediction with epistemic uncertainty"""
        model = self.uncertainty_models[model_id]["model"]
        model.eval()

        # Multiple forward passes for sampling
        predictions = []
        for _ in range(self.config.posterior_samples):
            with torch.no_grad():
                pred = model(input_data)
                predictions.append(pred.cpu().numpy())

        predictions = np.array(predictions)

        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0).item()
        epistemic_uncertainty = np.std(predictions, axis=0).item()

        # Prediction interval (95% by default)
        lower_bound = np.percentile(predictions, 2.5, axis=0).item()
        upper_bound = np.percentile(predictions, 97.5, axis=0).item()

        # Calculate confidence
        interval_width = upper_bound - lower_bound
        confidence = max(0.1, min(0.9, 1.0 - interval_width / self.config.max_prediction_interval))

        return UncertaintyPrediction(
            mean_prediction=mean_pred,
            confidence=confidence,
            uncertainty=epistemic_uncertainty,
            prediction_interval=(lower_bound, upper_bound),
            quantile_predictions={0.025: lower_bound, 0.5: mean_pred, 0.975: upper_bound},
            method_used=UncertaintyMethod.BAYESIAN_NN,
            timestamp=datetime.now(),
            epistemic_uncertainty=epistemic_uncertainty,
            aleatoric_uncertainty=0.0,  # Not estimated in this method
        )

    def _quantile_prediction(
        self, model_id: str, input_data: torch.Tensor
    ) -> UncertaintyPrediction:
        """Quantile regression prediction with prediction intervals"""
        model = self.uncertainty_models[model_id]["model"]
        model.eval()

        with torch.no_grad():
            quantile_preds = model(input_data).cpu().numpy().flatten()

        # Map quantile predictions
        quantile_dict = {}
        for i, quantile in enumerate(self.config.quantiles):
            quantile_dict[quantile] = quantile_preds[i]

        # Median as point prediction
        median_idx = len(self.config.quantiles) // 2
        mean_pred = quantile_preds[median_idx]

        # Calculate uncertainty from quantile spread
        if 0.05 in quantile_dict and 0.95 in quantile_dict:
            lower_bound = quantile_dict[0.05]
            upper_bound = quantile_dict[0.95]
            interval_width = upper_bound - lower_bound
            uncertainty = interval_width / 3.92  # Approximate std from 90% interval
        else:
            lower_bound = quantile_dict[min(quantile_dict.keys())]
            upper_bound = quantile_dict[max(quantile_dict.keys())]
            uncertainty = (upper_bound - lower_bound) / 4.0

        # Calculate confidence
        confidence = max(0.1, min(0.9, 1.0 - interval_width / self.config.max_prediction_interval))

        return UncertaintyPrediction(
            mean_prediction=mean_pred,
            confidence=confidence,
            uncertainty=uncertainty,
            prediction_interval=(lower_bound, upper_bound),
            quantile_predictions=quantile_dict,
            method_used=UncertaintyMethod.QUANTILE_REGRESSION,
            timestamp=datetime.now(),
            aleatoric_uncertainty=uncertainty,  # Quantile regression captures aleatoric uncertainty
        )

    def _mc_dropout_prediction(
        self, model_id: str, input_data: torch.Tensor
    ) -> UncertaintyPrediction:
        """Monte Carlo dropout prediction"""
        model = self.uncertainty_models[model_id]["model"]

        mean_pred, uncertainty = model.predict_with_uncertainty(
            input_data, self.config.mc_dropout_samples
        )

        mean_val = mean_pred.cpu().numpy().item()
        uncertainty_val = uncertainty.cpu().numpy().item()

        # Approximate prediction interval
        lower_bound = mean_val - 1.96 * uncertainty_val
        upper_bound = mean_val + 1.96 * uncertainty_val

        # Calculate confidence
        interval_width = upper_bound - lower_bound
        confidence = max(0.1, min(0.9, 1.0 - interval_width / self.config.max_prediction_interval))

        return UncertaintyPrediction(
            mean_prediction=mean_val,
            confidence=confidence,
            uncertainty=uncertainty_val,
            prediction_interval=(lower_bound, upper_bound),
            quantile_predictions={0.025: lower_bound, 0.5: mean_val, 0.975: upper_bound},
            method_used=UncertaintyMethod.MONTE_CARLO_DROPOUT,
            timestamp=datetime.now(),
            epistemic_uncertainty=uncertainty_val,
        )

    def _ensemble_prediction(
        self, model_id: str, input_data: torch.Tensor
    ) -> UncertaintyPrediction:
        """Ensemble prediction with disagreement-based uncertainty"""
        ensemble = self.ensemble_models[model_id]

        predictions = []
        for model in ensemble:
            model.eval()
            with torch.no_grad():
                pred = model(input_data)
                predictions.append(pred.cpu().numpy().item())

        predictions = np.array(predictions)

        # Calculate statistics
        mean_pred = np.mean(predictions)
        uncertainty = np.std(predictions)

        # Agreement measure
        agreement = 1.0 - (uncertainty / (abs(mean_pred) + 1e-8))
        agreement = max(0.0, min(1.0, agreement))

        # Prediction interval
        lower_bound = np.percentile(predictions, 2.5)
        upper_bound = np.percentile(predictions, 97.5)

        # Calculate confidence based on ensemble agreement
        interval_width = upper_bound - lower_bound
        base_confidence = 1.0 - interval_width / self.config.max_prediction_interval
        confidence = max(0.1, min(0.9, base_confidence * agreement))

        return UncertaintyPrediction(
            mean_prediction=mean_pred,
            confidence=confidence,
            uncertainty=uncertainty,
            prediction_interval=(lower_bound, upper_bound),
            quantile_predictions={0.025: lower_bound, 0.5: mean_pred, 0.975: upper_bound},
            method_used=UncertaintyMethod.ENSEMBLE_SPREAD,
            timestamp=datetime.now(),
            ensemble_agreement=agreement,
        )

    def should_trade(self, prediction: UncertaintyPrediction) -> Tuple[bool, str]:
        """
        Determine if trade should be executed based on uncertainty

        Returns:
            (should_trade, reason)
        """
        # Check confidence threshold
        if prediction.confidence < self.config.trade_confidence_threshold:
            return (
                False,
                f"Low confidence: {prediction.confidence:.2f} < {self.config.trade_confidence_threshold:.2f}",
            )

        # Check uncertainty level
        if prediction.uncertainty > self.config.max_prediction_interval:
            return (
                False,
                f"High uncertainty: {prediction.uncertainty:.2f} > {self.config.max_prediction_interval:.2f}",
            )

        # Check prediction interval width
        interval_width = prediction.prediction_interval[1] - prediction.prediction_interval[0]
        if interval_width > self.config.max_prediction_interval:
            return (
                False,
                f"Wide prediction interval: {interval_width:.2f} > {self.config.max_prediction_interval:.2f}",
            )

        # All checks passed
        return (
            True,
            f"High confidence trade: {prediction.confidence:.2f} confidence, {prediction.uncertainty:.2f} uncertainty",
        )

    def calculate_position_size_with_uncertainty(
        self, prediction: UncertaintyPrediction, base_position_size: float
    ) -> float:
        """Adjust position size based on prediction uncertainty"""
        # Confidence-based scaling
        confidence_multiplier = prediction.confidence

        # Uncertainty penalty
        uncertainty_penalty = 1.0 / (
            1.0 + self.config.uncertainty_penalty_factor * prediction.uncertainty
        )

        # Combined scaling
        size_multiplier = confidence_multiplier * uncertainty_penalty

        return base_position_size * size_multiplier

    def update_calibration_data(self, prediction: UncertaintyPrediction, actual_value: float):
        """Update calibration data with new prediction-outcome pair"""
        with self._lock:
            error = abs(prediction.mean_prediction - actual_value)

            self.calibration_data["predictions"].append(prediction.mean_prediction)
            self.calibration_data["uncertainties"].append(prediction.uncertainty)
            self.calibration_data["actuals"].append(actual_value)
            self.calibration_data["errors"].append(error)

            # Limit history size
            max_history = 1000
            for key in self.calibration_data:
                if len(self.calibration_data[key]) > max_history:
                    self.calibration_data[key] = self.calibration_data[key][-max_history:]

            # Update uncertainty metrics
            self._update_uncertainty_metrics()

    def _update_uncertainty_metrics(self):
        """Update uncertainty calibration metrics"""
        if len(self.calibration_data["predictions"]) < 10:
            return

        errors = np.array(self.calibration_data["errors"])
        uncertainties = np.array(self.calibration_data["uncertainties"])
        predictions = np.array(self.calibration_data["predictions"])
        actuals = np.array(self.calibration_data["actuals"])

        # Coverage probability for prediction intervals
        in_interval = 0
        for i, pred in enumerate(self.prediction_history[-len(errors) :]):
            if pred.prediction_interval[0] <= actuals[i] <= pred.prediction_interval[1]:
                in_interval += 1

        coverage = in_interval / len(errors) if len(errors) > 0 else 0.0

        # Uncertainty calibration (correlation between uncertainty and error)
        if len(uncertainties) > 1 and np.std(uncertainties) > 0:
            correlation = np.corrcoef(uncertainties, errors)[0, 1]
        else:
            correlation = 0.0

        # Sharpness (average uncertainty)
        sharpness = np.mean(uncertainties)

        # Reliability (uncertainty should predict error magnitude)
        reliability = max(0.0, correlation)

        self.uncertainty_metrics = {
            "coverage_probability": coverage,
            "uncertainty_correlation": correlation,
            "average_uncertainty": sharpness,
            "reliability_score": reliability,
            "calibration_quality": coverage * reliability,
        }

    def get_uncertainty_summary(self) -> Dict[str, Any]:
        """Get comprehensive uncertainty modeling summary"""
        with self._lock:
            return {
                "models_created": len(self.uncertainty_models),
                "ensemble_models": len(self.ensemble_models),
                "predictions_made": len(self.prediction_history),
                "calibration_samples": len(self.calibration_data["predictions"]),
                "uncertainty_metrics": self.uncertainty_metrics,
                "config": {
                    "trade_confidence_threshold": self.config.trade_confidence_threshold,
                    "max_prediction_interval": self.config.max_prediction_interval,
                    "uncertainty_penalty_factor": self.config.uncertainty_penalty_factor,
                },
                "recent_confidence": np.mean(self.confidence_history[-50:])
                if self.confidence_history
                else 0.0,
            }


# Singleton uncertainty engine
_uncertainty_engine = None
_uncertainty_lock = threading.Lock()


def get_uncertainty_engine(config: Optional[UncertaintyConfig] = None) -> UncertaintyEngine:
    """Get the singleton uncertainty engine"""
    global _uncertainty_engine

    with _uncertainty_lock:
        if _uncertainty_engine is None:
            _uncertainty_engine = UncertaintyEngine(config)
        return _uncertainty_engine


def predict_with_confidence(
    model_id: str, input_data: torch.Tensor
) -> Optional[UncertaintyPrediction]:
    """Convenient function for uncertainty prediction"""
    engine = get_uncertainty_engine()
    return engine.predict_with_uncertainty(model_id, input_data)
