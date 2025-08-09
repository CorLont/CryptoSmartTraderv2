"""
ML Package - Modern Machine Learning Infrastructure
Advanced ML/AI with uncertainty quantification, regime detection, and ensembles
"""

try:
    from .models import ModelFactory, BaseModel
except ImportError:
    ModelFactory = None
    BaseModel = None

__all__ = [
    'ModelFactory',
    'BaseModel'
]