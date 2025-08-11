"""CryptoSmartTrader V2 - Enterprise API Layer"""

__version__ = "2.0.0"

from .app import create_app
from .models.health import HealthOut, HealthDetailOut
from .models.market import MarketDataOut, PriceData
from .models.trading import SignalOut, PortfolioOut
from .models.agents import AgentStatus, AgentMetrics
from .dependencies import get_settings, get_orchestrator

__all__ = ["create_app", "get_settings", "get_orchestrator"]