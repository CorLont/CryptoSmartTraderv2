"""CryptoSmartTrader V2 - Enterprise Multi-Agent Trading Intelligence System"""

__version__ = "2.0.0"
__author__ = "CryptoSmartTrader Team"
__description__ = "Advanced cryptocurrency trading intelligence with institutional-grade analysis"

from .config import Settings
from .logging import setup_logging

__all__ = ["Settings", "setup_logging"]