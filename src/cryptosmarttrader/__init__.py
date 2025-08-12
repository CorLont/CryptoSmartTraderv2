"""
CryptoSmartTrader V2 - Enterprise Cryptocurrency Trading Intelligence Platform

A sophisticated multi-agent cryptocurrency trading intelligence system designed for 
institutional-grade analysis and automated trading strategies.

Key Features:
- Multi-agent architecture with specialized trading agents
- Real-time market data analysis from multiple exchanges
- Advanced ML predictions with uncertainty quantification
- Risk management and portfolio optimization
- Enterprise-grade monitoring and observability

Modules:
- api: FastAPI web services and health monitoring
- agents: Specialized trading and analysis agents
- ml: Machine learning models and prediction pipelines
- core: Core business logic and domain models
- utils: Shared utilities and helper functions
"""

__version__ = "2.0.0"
__author__ = "CryptoSmartTrader Team"
__license__ = "Proprietary"

# Core imports for public API
from . import api, agents, ml, core, utils

__all__ = [
    "__version__",
    "__author__", 
    "__license__",
    "api",
    "agents", 
    "ml",
    "core",
    "utils",
]