"""
Data Provider Port - Interface for market data sources

Defines the contract for all data providers (Kraken, Binance, etc.)
enabling swappable adapters without breaking core business logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from datetime import datetime
from enum import Enum


class DataSourceType(Enum):
    """Types of data sources available"""

    PRICE_DATA = "price_data"
    ORDERBOOK = "orderbook"
    SENTIMENT = "sentiment"
    NEWS = "news"
    SOCIAL_MEDIA = "social_media"
    ON_CHAIN = "on_chain"


class DataQuality(Enum):
    """Data quality levels"""

    AUTHENTIC = "authentic"
    CACHED = "cached"
    INTERPOLATED = "interpolated"
    UNAVAILABLE = "unavailable"


class MarketDataRequest:
    """Request object for market data"""

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.start_time = start_time
        self.end_time = end_time
        self.limit = limit


class MarketDataResponse:
    """Response object containing market data and metadata"""

    def __init__(
        self,
        data: pd.DataFrame,
        quality: DataQuality,
        source: str,
        timestamp: datetime,
        metadata: Optional[Dict] = None,
    ):
        self.data = data
        self.quality = quality
        self.source = source
        self.timestamp = timestamp
        self.metadata = metadata or {}


class DataProviderPort(ABC):
    """
    Abstract interface for market data providers

    This port defines the contract that all data provider adapters must implement,
    ensuring consistent behavior across different exchanges and data sources.
    """

    @abstractmethod
    def get_price_data(self, request: MarketDataRequest) -> MarketDataResponse:
        """
        Retrieve price/OHLCV data for a trading pair

        Args:
            request: MarketDataRequest with symbol, timeframe, and optional time range

        Returns:
            MarketDataResponse with OHLCV data and quality metadata

        Raises:
            DataProviderError: When data cannot be retrieved
        """
        pass

    @abstractmethod
    def get_orderbook_data(self, symbol: str, depth: int = 100) -> MarketDataResponse:
        """
        Retrieve level-2 orderbook data

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSD')
            depth: Number of price levels to retrieve

        Returns:
            MarketDataResponse with orderbook data
        """
        pass

    @abstractmethod
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols

        Returns:
            List of symbol strings (e.g., ['BTCUSD', 'ETHUSD'])
        """
        pass

    @abstractmethod
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed information about a trading symbol

        Args:
            symbol: Trading pair symbol

        Returns:
            Dictionary with symbol metadata (decimals, min_size, etc.)
        """
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """
        Validate connection to data provider

        Returns:
            True if connection is healthy, False otherwise
        """
        pass

    @abstractmethod
    def get_rate_limits(self) -> Dict[str, Any]:
        """
        Get current rate limiting information

        Returns:
            Dictionary with rate limit details
        """
        pass

    @abstractmethod
    def get_data_quality_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about data quality and availability

        Returns:
            Dictionary with quality metrics (uptime, latency, etc.)
        """
        pass


class SentimentDataProviderPort(ABC):
    """Interface for sentiment data providers"""

    @abstractmethod
    def get_sentiment_analysis(self, symbol: str, lookback_hours: int = 24) -> MarketDataResponse:
        """Get sentiment analysis for a symbol"""
        pass

    @abstractmethod
    def get_news_sentiment(self, symbol: str, limit: int = 100) -> MarketDataResponse:
        """Get news-based sentiment analysis"""
        pass

    @abstractmethod
    def get_social_sentiment(self, symbol: str, platforms: List[str]) -> MarketDataResponse:
        """Get social media sentiment"""
        pass


class DataProviderError(Exception):
    """Exception raised by data provider implementations"""

    def __init__(
        self, message: str, error_code: Optional[str] = None, source: Optional[str] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.source = source


class DataProviderRegistry:
    """Registry for managing multiple data provider implementations"""

    def __init__(self):
        self._providers: Dict[str, DataProviderPort] = {}
        self._primary_provider: Optional[str] = None

    def register_provider(self, name: str, provider: DataProviderPort, is_primary: bool = False):
        """Register a data provider implementation"""
        self._providers[name] = provider
        if is_primary or self._primary_provider is None:
            self._primary_provider = name

    def get_provider(self, name: Optional[str] = None) -> DataProviderPort:
        """Get a specific provider or the primary one"""
        provider_name = name or self._primary_provider
        if provider_name not in self._providers:
            raise DataProviderError(f"Provider '{provider_name}' not found")
        return self._providers[provider_name]

    def list_providers(self) -> List[str]:
        """Get list of registered provider names"""
        return list(self._providers.keys())

    def get_healthy_providers(self) -> List[str]:
        """Get list of providers with healthy connections"""
        healthy = []
        for name, provider in self._providers.items():
            try:
                if provider.validate_connection():
                    healthy.append(name)
            except Exception:
                continue
        return healthy


# Global registry instance
data_provider_registry = DataProviderRegistry()
