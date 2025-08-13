"""
Optimization Module

Advanced hyperparameter tuning and out-of-sample validation system
to prevent overfitting and ensure robust model performance.
"""

from .hyperparameter_optimizer import HyperparameterOptimizer, OptimizationResult, OptimizationMetrics
from .walk_forward_validator import WalkForwardValidator, ValidationResult, TimeSeriesCV
from .regime_aware_cv import RegimeAwareCV, RegimeSplit, RegimeValidationResult
from .bayesian_optimizer import BayesianOptimizer, ObjectiveFunction, SearchSpace
from .performance_evaluator import PerformanceEvaluator, OOSMetrics, StabilityMetrics

__all__ = [
    "HyperparameterOptimizer",
    "OptimizationResult",
    "OptimizationMetrics",
    "WalkForwardValidator",
    "ValidationResult",
    "TimeSeriesCV",
    "RegimeAwareCV",
    "RegimeSplit",
    "RegimeValidationResult",
    "BayesianOptimizer",
    "ObjectiveFunction",
    "SearchSpace",
    "PerformanceEvaluator",
    "OOSMetrics",
    "StabilityMetrics"
]
