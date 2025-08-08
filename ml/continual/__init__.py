"""
Continual Learning Module
Online learning and meta-learning for continuous adaptation
"""

from .continual_learner import ContinualLearner
from .meta_learner import MetaLearner
from .drift_detector import DriftDetector
from .online_updater import OnlineUpdater

__all__ = [
    'ContinualLearner',
    'MetaLearner',
    'DriftDetector',
    'OnlineUpdater'
]