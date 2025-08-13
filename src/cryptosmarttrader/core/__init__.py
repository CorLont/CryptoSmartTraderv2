"""Core module for CryptoSmartTrader V2."""

from .config_manager import ConfigManager
from .structured_logger import StructuredLogger

__all__ = [
    "ConfigManager", 
    "StructuredLogger",
]