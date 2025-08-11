"""FastAPI Dependencies - Dependency Injection for API Endpoints"""

from functools import lru_cache
from typing import Optional

from ..config import get_settings, Settings


@lru_cache()
def get_settings_cached() -> Settings:
    """Get cached settings instance for API endpoints"""
    return get_settings()


def get_settings() -> Settings:
    """Dependency to inject validated settings"""
    return get_settings_cached()


# Placeholder for orchestrator dependency
# This would be replaced with actual orchestrator implementation
class MockOrchestrator:
    """Mock orchestrator for development/testing"""
    
    async def get_market_data(self, limit: int = 100, sort_by: str = "market_cap"):
        """Mock market data"""
        return {
            "coins": [],
            "total_count": 0,
            "last_updated": "2025-01-11T12:00:00Z",
            "data_source": "kraken",
            "confidence_score": 0.95
        }
    
    async def get_trading_signals(self, limit: int = 50, min_confidence: float = 0.7, symbol: Optional[str] = None):
        """Mock trading signals"""
        return []
    
    async def get_portfolio_summary(self):
        """Mock portfolio summary"""
        return {
            "total_value": 10000.0,
            "available_balance": 5000.0,
            "invested_amount": 5000.0,
            "unrealized_pnl": 0.0,
            "unrealized_pnl_percent": 0.0,
            "positions": [],
            "last_updated": "2025-01-11T12:00:00Z"
        }
    
    async def get_all_agent_status(self):
        """Mock agent status"""
        return []


@lru_cache()
def get_orchestrator_cached():
    """Get cached orchestrator instance"""
    # This would return the actual orchestrator implementation
    return MockOrchestrator()


def get_orchestrator():
    """Dependency to inject orchestrator instance"""
    return get_orchestrator_cached()