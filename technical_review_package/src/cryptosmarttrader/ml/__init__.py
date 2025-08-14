"""
Machine Learning Module for CryptoSmartTrader
Enterprise ML infrastructure with model registry, walk-forward training, and drift detection.
"""

from .model_registry import (
    ModelRegistry,
    ModelVersion,
    ModelMetrics,
    ModelType,
    ModelStatus,
    DatasetInfo,
    TrainingConfig,
    create_model_registry,
)

from .walk_forward_trainer import (
    WalkForwardTrainer,
    WalkForwardResult,
    RetrainingConfig,
    RetrainingTrigger,
    ValidationMethod,
    create_walk_forward_trainer,
)

from .drift_detector import (
    DriftDetector,
    DriftAlert,
    DriftType,
    DriftSeverity,
    DriftConfig,
    create_drift_detector,
)

__all__ = [
    # Model Registry
    "ModelRegistry",
    "ModelVersion",
    "ModelMetrics",
    "ModelType",
    "ModelStatus",
    "DatasetInfo",
    "TrainingConfig",
    "create_model_registry",
    # Walk-Forward Training
    "WalkForwardTrainer",
    "WalkForwardResult",
    "RetrainingConfig",
    "RetrainingTrigger",
    "ValidationMethod",
    "create_walk_forward_trainer",
    # Drift Detection
    "DriftDetector",
    "DriftAlert",
    "DriftType",
    "DriftSeverity",
    "DriftConfig",
    "create_drift_detector",
]
