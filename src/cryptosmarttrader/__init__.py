"""
CryptoSmartTrader V2 - Enterprise-grade cryptocurrency trading intelligence system.
"""

__version__ = "2.0.0"
__author__ = "CryptoSmartTrader Team"
__email__ = "team@cryptosmarttrader.com"

from .core.config_manager import ConfigManager
from .core.structured_logger import StructuredLogger

__all__ = [
    "ConfigManager",
    "StructuredLogger",
]