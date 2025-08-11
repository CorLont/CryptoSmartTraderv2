"""
Core module - Essential system components

Contains the fundamental building blocks of the CryptoSmartTrader system
including configuration, data management, logging, and monitoring.
"""

# Conditional imports to handle missing dependencies
try:
    from .config_manager import ConfigManager
except ImportError:
    ConfigManager = None

try:
    from .data_manager import DataManager
except ImportError:
    DataManager = None

try:
    from .consolidated_logging_manager import get_consolidated_logger
except ImportError:
    get_consolidated_logger = None

try:
    from .temporal_integrity_validator import TemporalIntegrityValidator
except ImportError:
    TemporalIntegrityValidator = None

__all__ = [
    "ConfigManager",
    "DataManager", 
    "get_consolidated_logger",
    "TemporalIntegrityValidator"
]