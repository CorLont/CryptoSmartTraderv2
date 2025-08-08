"""
ML Package - Modern Machine Learning Infrastructure
Advanced ML/AI with uncertainty quantification, regime detection, and ensembles
"""

from .features import FeatureEngineering, AutoFeatures
from .models import ModelFactory, BaseModel
from .ensembles import EnsembleManager, UncertaintyEnsemble
from .regime import RegimeDetector, RegimeRouter
from .continual import ContinualLearner, MetaLearner

__all__ = [
    'FeatureEngineering',
    'AutoFeatures',
    'ModelFactory',
    'BaseModel',
    'EnsembleManager',
    'UncertaintyEnsemble',
    'RegimeDetector',
    'RegimeRouter',
    'ContinualLearner',
    'MetaLearner'
]