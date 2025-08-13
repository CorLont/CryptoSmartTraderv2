"""
Model Inference Port - Interface for ML model predictions

Defines the contract for all ML model implementations enabling
swappable models and inference engines without breaking prediction logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
from enum import Enum
from dataclasses import dataclass

class ModelType(Enum):
    """Types of ML models"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    TIME_SERIES = "time_series"
    ENSEMBLE = "ensemble"
    DEEP_LEARNING = "deep_learning"
    REINFORCEMENT = "reinforcement_learning"

class PredictionHorizon(Enum):
    """Prediction time horizons"""
    SHORT_TERM = "1h"      # 1 hour
    MEDIUM_TERM = "24h"    # 24 hours
    LONG_TERM = "7d"       # 7 days
    EXTENDED = "30d"       # 30 days

class ModelStatus(Enum):
    """Model operational status"""
    READY = "ready"
    TRAINING = "training"
    ERROR = "error"
    OUTDATED = "outdated"
    CALIBRATING = "calibrating"

class ConfidenceLevel(Enum):
    """Confidence levels for predictions"""
    VERY_LOW = "very_low"      # < 50%
    LOW = "low"                # 50-65%
    MEDIUM = "medium"          # 65-80%
    HIGH = "high"              # 80-90%
    VERY_HIGH = "very_high"    # > 90%

@dataclass
class PredictionRequest:
    """Request object for model predictions"""
    symbol: str
    feature_data: pd.DataFrame
    horizon: PredictionHorizon
    confidence_threshold: float = 0.8
    metadata: Optional[Dict] = None

@dataclass
class UncertaintyQuantification:
    """Uncertainty quantification for predictions"""
    epistemic_uncertainty: float      # Model uncertainty
    aleatoric_uncertainty: float      # Data uncertainty
    total_uncertainty: float          # Combined uncertainty
    confidence_interval: Tuple[float, float]  # Lower, upper bounds
    prediction_interval: Tuple[float, float]  # Prediction bounds

@dataclass
class PredictionResult:
    """Result object containing prediction and metadata"""
    symbol: str
    prediction: Union[float, np.ndarray]
    confidence: float
    horizon: PredictionHorizon
    timestamp: datetime
    model_name: str
    model_version: str
    uncertainty: Optional[UncertaintyQuantification] = None
    feature_importance: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None
    metadata: Optional[Dict] = None

@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    r2_score: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    hit_rate: Optional[float] = None
    calibration_score: Optional[float] = None

class ModelInferencePort(ABC):
    """
    Abstract interface for ML model inference

    This port defines the contract that all model implementations must follow,
    ensuring consistent behavior across different ML frameworks and model types.
    """

    @abstractmethod
    def predict(self, request: PredictionRequest) -> PredictionResult:
        """
        Generate prediction for given features

        Args:
            request: PredictionRequest with symbol, features, and parameters

        Returns:
            PredictionResult with prediction and confidence metrics

        Raises:
            ModelInferenceError: When prediction cannot be generated
        """
        pass

    @abstractmethod
    def predict_batch(self, requests: List[PredictionRequest]) -> List[PredictionResult]:
        """
        Generate predictions for multiple requests efficiently

        Args:
            requests: List of PredictionRequest objects

        Returns:
            List of PredictionResult objects in same order
        """
        pass

    @abstractmethod
    def get_feature_importance(self, symbol: str) -> Dict[str, float]:
        """
        Get feature importance scores for a symbol

        Args:
            symbol: Trading symbol

        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass

    @abstractmethod
    def explain_prediction(self, request: PredictionRequest) -> str:
        """
        Generate human-readable explanation for prediction

        Args:
            request: PredictionRequest to explain

        Returns:
            Text explanation of the prediction logic
        """
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata and configuration

        Returns:
            Dictionary with model information (name, version, type, etc.)
        """
        pass

    @abstractmethod
    def get_model_status(self) -> ModelStatus:
        """
        Get current model operational status

        Returns:
            ModelStatus enum indicating current state
        """
        pass

    @abstractmethod
    def validate_features(self, features: pd.DataFrame) -> bool:
        """
        Validate that features match model expectations

        Args:
            features: Feature DataFrame to validate

        Returns:
            True if features are valid, False otherwise
        """
        pass

    @abstractmethod
    def get_required_features(self) -> List[str]:
        """
        Get list of features required by this model

        Returns:
            List of required feature names
        """
        pass

class TrainableModelPort(ModelInferencePort):
    """Extended interface for models that support training"""

    @abstractmethod
    def train(self, training_data: pd.DataFrame, validation_data: Optional[pd.DataFrame] = None,
              target_column: str = "target") -> Dict[str, Any]:
        """
        Train the model on provided data

        Args:
            training_data: DataFrame with features and targets
            validation_data: Optional validation DataFrame
            target_column: Name of the target column

        Returns:
            Dictionary with training metrics and metadata
        """
        pass

    @abstractmethod
    def evaluate(self, test_data: pd.DataFrame, target_column: str = "target") -> ModelMetrics:
        """
        Evaluate model performance on test data

        Args:
            test_data: DataFrame with features and targets
            target_column: Name of the target column

        Returns:
            ModelMetrics with performance scores
        """
        pass

    @abstractmethod
    def save_model(self, path: str) -> bool:
        """
        Save model to disk

        Args:
            path: File path to save model

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def load_model(self, path: str) -> bool:
        """
        Load model from disk

        Args:
            path: File path to load model from

        Returns:
            True if successful, False otherwise
        """
        pass

class EnsembleModelPort(ModelInferencePort):
    """Interface for ensemble model implementations"""

    @abstractmethod
    def add_model(self, name: str, model: ModelInferencePort, weight: float = 1.0):
        """Add a model to the ensemble"""
        pass

    @abstractmethod
    def remove_model(self, name: str) -> bool:
        """Remove a model from the ensemble"""
        pass

    @abstractmethod
    def get_model_weights(self) -> Dict[str, float]:
        """Get current model weights in ensemble"""
        pass

    @abstractmethod
    def update_weights(self, weights: Dict[str, float]):
        """Update model weights based on performance"""
        pass

class UncertaintyAwareModelPort(ModelInferencePort):
    """Interface for models with uncertainty quantification"""

    @abstractmethod
    def predict_with_uncertainty(self, request: PredictionRequest) -> PredictionResult:
        """
        Generate prediction with uncertainty quantification

        Args:
            request: PredictionRequest

        Returns:
            PredictionResult with uncertainty information
        """
        pass

    @abstractmethod
    def calibrate_uncertainty(self, calibration_data: pd.DataFrame):
        """
        Calibrate uncertainty estimates using calibration data

        Args:
            calibration_data: Data for uncertainty calibration
        """
        pass

    @abstractmethod
    def get_prediction_intervals(self, request: PredictionRequest,
                               confidence_levels: List[float]) -> Dict[float, Tuple[float, float]]:
        """
        Get prediction intervals for different confidence levels

        Args:
            request: PredictionRequest
            confidence_levels: List of confidence levels (e.g., [0.8, 0.9, 0.95])

        Returns:
            Dictionary mapping confidence levels to (lower, upper) bounds
        """
        pass

class ModelInferenceError(Exception):
    """Exception raised by model inference implementations"""

    def __init__(self, message: str, error_code: Optional[str] = None,
                 model_name: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.model_name = model_name

class ModelRegistry:
    """Registry for managing multiple model implementations"""

    def __init__(self):
        self._models: Dict[str, ModelInferencePort] = {}
        self._model_metadata: Dict[str, Dict] = {}
        self._default_models: Dict[PredictionHorizon, str] = {}

    def register_model(self, name: str, model: ModelInferencePort,
                      metadata: Optional[Dict] = None,
                      default_for_horizon: Optional[PredictionHorizon] = None):
        """Register a model implementation"""
        self._models[name] = model
        self._model_metadata[name] = metadata or {}

        if default_for_horizon:
            self._default_models[default_for_horizon] = name

    def get_model(self, name: Optional[str] = None,
                 horizon: Optional[PredictionHorizon] = None) -> ModelInferencePort:
        """Get a specific model or default for horizon"""
        if name:
            if name not in self._models:
                raise ModelInferenceError(f"Model '{name}' not found")
            return self._models[name]

        if horizon and horizon in self._default_models:
            return self._models[self._default_models[horizon]]

        raise ModelInferenceError("No model specified and no default found")

    def list_models(self) -> List[str]:
        """Get list of registered model names"""
        return list(self._models.keys())

    def get_models_by_type(self, model_type: ModelType) -> List[str]:
        """Get models filtered by type"""
        matching_models = []
        for name, model in self._models.items():
            model_info = model.get_model_info()
            if model_info.get('type') == model_type.value:
                matching_models.append(name)
        return matching_models

    def get_healthy_models(self) -> List[str]:
        """Get list of models with ready status"""
        healthy = []
        for name, model in self._models.items():
            try:
                if model.get_model_status() == ModelStatus.READY:
                    healthy.append(name)
            except Exception:
                continue
        return healthy

# Global registry instance
model_registry = ModelRegistry()

# Utility functions for model operations
def validate_prediction_confidence(confidence: float, threshold: float = 0.8) -> bool:
    """Validate prediction meets confidence threshold"""
    return confidence >= threshold

def aggregate_ensemble_predictions(predictions: List[PredictionResult],
                                 weights: Optional[List[float]] = None) -> PredictionResult:
    """Aggregate multiple predictions into ensemble result"""
    if not predictions:
        raise ModelInferenceError("No predictions to aggregate")

    if weights is None:
        weights = [1.0] * len(predictions)

    if len(weights) != len(predictions):
        raise ModelInferenceError("Weights length must match predictions length")

    # Weighted average of predictions
    total_weight = sum(weights)
    weighted_prediction = sum(p.prediction * w for p, w in zip(predictions, weights)) / total_weight
    weighted_confidence = sum(p.confidence * w for p, w in zip(predictions, weights)) / total_weight

    # Use first prediction as template
    template = predictions[0]

    return PredictionResult(
        symbol=template.symbol,
        prediction=weighted_prediction,
        confidence=weighted_confidence,
        horizon=template.horizon,
        timestamp=datetime.utcnow(),
        model_name="ensemble",
        model_version="1.0",
        metadata={
            'component_models': [p.model_name for p in predictions],
            'weights': weights,
            'aggregation_method': 'weighted_average'
        }
    )
