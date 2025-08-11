"""
CryptoSmartTrader V2 - Enterprise Multi-Agent Cryptocurrency Trading Intelligence System
"""

__version__ = "2.0.0"
__author__ = "CryptoSmartTrader Team"
__description__ = "Enterprise-grade cryptocurrency trading intelligence system with multi-agent architecture"

# Core exports - conditional imports to handle missing dependencies
try:
    from .core.config_manager import ConfigManager
except ImportError:
    ConfigManager = None

try:
    from .core.data_manager import DataManager  
except ImportError:
    DataManager = None

try:
    from .core.health_monitor import HealthMonitor
except ImportError:
    HealthMonitor = None

# Domain interfaces
from .interfaces.data_provider_port import DataProviderPort
from .interfaces.storage_port import StoragePort
from .interfaces.model_inference_port import ModelInferencePort

# Agent exports - conditional imports
try:
    from .agents.technical_agent import TechnicalAgent
except ImportError:
    TechnicalAgent = None

try:
    from .agents.sentiment_agent import SentimentAgent
except ImportError:
    SentimentAgent = None

try:
    from .agents.ml_predictor_agent import MLPredictorAgent
except ImportError:
    MLPredictorAgent = None

__all__ = [
    "DataProviderPort",
    "StoragePort", 
    "ModelInferencePort",
    "ConfigManager",
    "DataManager",
    "HealthMonitor", 
    "TechnicalAgent",
    "SentimentAgent",
    "MLPredictorAgent"
]