"""
Machine Learning Module for CryptoSmartTrader
Enterprise ML infrastructure with model registry, walk-forward training, and drift detection.
"""

from .model_registry import (
    ModelRegistry,
    ModelMetadata,
    DatasetVersion,
    ModelStatus,
    DriftStatus,
    get_model_registry,
)

from .canary_deployment import (
    CanaryDeploymentOrchestrator,
    CanaryConfig,
    CanaryMetrics,
    CanaryPhase,
    get_canary_orchestrator,
)

__all__ = [
    # Model Registry
    "ModelRegistry",
    "ModelMetadata",
    "DatasetVersion",
    "ModelStatus",
    "DriftStatus",
    "get_model_registry",
    # Canary Deployment
    "CanaryDeploymentOrchestrator",
    "CanaryConfig",
    "CanaryMetrics",
    "CanaryPhase",
    "get_canary_orchestrator",
]
