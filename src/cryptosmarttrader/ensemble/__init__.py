"""
Ensemble & Meta-Learning Module

Implements alpha-stacking by combining orthogonal information sources:
- Technical Analysis signals
- Sentiment Analysis signals  
- Regime Classification signals
- On-chain Analysis signals

Uses meta-learners to optimally blend base models for superior performance.
"""

from .base_models import BaseModelInterface, TechnicalAnalysisModel, SentimentModel, RegimeModel
from .meta_learner import MetaLearner, EnsembleConfig
from .alpha_blender import AlphaBlender, BlendingStrategy
from .signal_decay import SignalDecayManager, DecayConfig

__all__ = [
    "BaseModelInterface",
    "TechnicalAnalysisModel", 
    "SentimentModel",
    "RegimeModel",
    "MetaLearner",
    "EnsembleConfig",
    "AlphaBlender",
    "BlendingStrategy", 
    "SignalDecayManager",
    "DecayConfig"
]