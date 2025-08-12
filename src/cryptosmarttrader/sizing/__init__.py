"""
Confidence-Weighted Position Sizing Module

Implements fractional Kelly criterion with probability calibration
for optimal position sizing based on signal confidence.
"""

from .kelly_sizing import KellySizer, KellyParameters, KellyMode
from .probability_calibration import ProbabilityCalibrator
from .confidence_weigher import ConfidenceWeighter

__all__ = [
    "KellySizer",
    "KellyParameters",
    "KellyMode",
    "ProbabilityCalibrator",
    "ConfidenceWeighter"
]