"""
Kraken Data Adapter - Concrete implementation of DataProviderPort for Kraken exchange

Provides authentic market data from Kraken API following the DataProviderPort contract.
"""

import ccxt
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import time
import logging

from ..interfaces.data_provider_port import (
    DataProviderPort,
    MarketDataRequest,
    MarketDataResponse,
    DataQuality,
    DataProviderError,
)


class KrakenDataAdapter(DataProviderPort):
    """Kraken exchange data provider implementation"""

    def __init__(
        self, api_key: Optional[str] = None, secret: Optional[str] = None, rate_limit: bool = True
    ):
        """
        Initialize Kraken data adapter

        Args:
            api_key: Kraken API key (optional for public data)
            secret: Kraken secret key (optional for public data)
            rate_limit: Enable rate limiting
        """
        self.exchange = ccxt.kraken(
            {
                "apiKey": api_key,
                "secret": secret,
                "rateLimit": rate_limit,
                "enableRateLimit": rate_limit,
            }
        )

        self.logger = logging.getLogger(__name__)
        self._markets_cache: Optional[Dict] = None
        self._cache_timestamp: Optional[datetime] = None

        # Rate limiting tracking
        self._last_request_time = 0
        self._request_count = 0

    def get_price_data(self, request: MarketDataRequest) -> MarketDataResponse:
        """Retrieve OHLCV data from Kraken"""

        try:
            self._enforce_rate_limit()

            # Convert symbol format for Kraken
            kraken_symbol = self._convert_symbol_to_kraken(request.symbol)

            # Prepare parameters
            params = {}
            if request.start_time:
                params["since"] = int(request.start_time.timestamp() * 1000)

            # Fetch OHLCV data
            ohlcv_data = self.exchange.fetch_ohlcv(
                symbol=kraken_symbol,
                timeframe=request.timeframe,
                since=params.get("since"),
                limit=request.limit,
            )

            if not ohlcv_data:
                raise DataProviderError(f"No data available for {request.symbol}")

            # Convert to DataFrame
            df = pd.DataFrame(
                ohlcv_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

            # Apply time filtering if specified
            if request.start_time:
                df = df[df["timestamp"] >= request.start_time]
            if request.end_time:
                df = df[df["timestamp"] <= request.end_time]

            return MarketDataResponse(
                data=df,
                quality=DataQuality.AUTHENTIC,
                source="kraken",
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "symbol": request.symbol,
                    "timeframe": request.timeframe,
                    "records_count": len(df),
                    "data_range": {
                        "start": df["timestamp"].min().isoformat() if len(df) > 0 else None,
                        "end": df["timestamp"].max().isoformat() if len(df) > 0 else None,
                    },
                },
            )

        except ccxt.BaseError as e:
            self.logger.error(f"Kraken API error for {request.symbol}: {e}")
            raise DataProviderError(f"Kraken API error: {str(e)}", source="kraken")
        except Exception as e:
            self.logger.error(f"Unexpected error fetching data for {request.symbol}: {e}")
            raise DataProviderError(f"Data fetch failed: {str(e)}", source="kraken")

    def get_orderbook_data(self, symbol: str, depth: int = 100) -> MarketDataResponse:
        """Retrieve orderbook data from Kraken"""

        try:
            self._enforce_rate_limit()

            kraken_symbol = self._convert_symbol_to_kraken(symbol)

            # Fetch orderbook
            orderbook = self.exchange.fetch_order_book(kraken_symbol, limit=depth)

            # Convert to DataFrame format
            bids_df = pd.DataFrame(orderbook["bids"], columns=["price", "amount"])
            asks_df = pd.DataFrame(orderbook["asks"], columns=["price", "amount"])

            bids_df["side"] = "bid"
            asks_df["side"] = "ask"

            df = pd.concat([bids_df, asks_df], ignore_index=True)
            df["timestamp"] = pd.Timestamp.utcnow()

            return MarketDataResponse(
                data=df,
                quality=DataQuality.AUTHENTIC,
                source="kraken",
                timestamp=datetime.now(timezone.utc),
                metadata={
                    "symbol": symbol,
                    "depth": depth,
                    "bids_count": len(bids_df),
                    "asks_count": len(asks_df),
                    "spread": asks_df["price"].min() - bids_df["price"].max()
                    if len(bids_df) > 0 and len(asks_df) > 0
                    else None,
                },
            )

        except ccxt.BaseError as e:
            self.logger.error(f"Kraken orderbook error for {symbol}: {e}")
            raise DataProviderError(f"Orderbook fetch failed: {str(e)}", source="kraken")

    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols"""

        try:
            if not self._markets_cache or self._is_cache_expired():
                self._refresh_markets_cache()

            # Filter for USD pairs only
            usd_symbols = [
                symbol for symbol in self._markets_cache.keys() if symbol.endswith("/USD")
            ]

            return sorted(usd_symbols)

        except ccxt.BaseError as e:
            self.logger.error(f"Error fetching available symbols: {e}")
            raise DataProviderError(f"Symbols fetch failed: {str(e)}", source="kraken")

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Get detailed information about a trading symbol"""

        try:
            if not self._markets_cache or self._is_cache_expired():
                self._refresh_markets_cache()

            kraken_symbol = self._convert_symbol_to_kraken(symbol)

            if kraken_symbol not in self._markets_cache:
                raise DataProviderError(f"Symbol {symbol} not found", source="kraken")

            market_info = self._markets_cache[kraken_symbol]

            return {
                "symbol": symbol,
                "base": market_info.get("base"),
                "quote": market_info.get("quote"),
                "active": market_info.get("active", False),
                "precision": market_info.get("precision", {}),
                "limits": market_info.get("limits", {}),
                "fees": market_info.get("fees", {}),
                "info": market_info.get("info", {}),
            }

        except ccxt.BaseError as e:
            self.logger.error(f"Error fetching symbol info for {symbol}: {e}")
            raise DataProviderError(f"Symbol info fetch failed: {str(e)}", source="kraken")

    def validate_connection(self) -> bool:
        """Validate connection to Kraken"""

        try:
            # Test with a simple API call
            self.exchange.fetch_status()
            return True
        except Exception as e:
            self.logger.warning(f"Kraken connection validation failed: {e}")
            return False

    def get_rate_limits(self) -> Dict[str, Any]:
        """Get current rate limiting information"""

        return {
            "requests_per_minute": 60,  # Kraken typical limit
            "current_request_count": self._request_count,
            "rate_limit_enabled": self.exchange.rateLimit,
            "last_request_time": self._last_request_time,
        }

    def get_data_quality_metrics(self) -> Dict[str, Any]:
        """Get data quality metrics"""

        # This would be enhanced with actual quality tracking
        return {
            "uptime_percentage": 99.5,  # Placeholder - would track actual uptime
            "average_latency_ms": 150,  # Placeholder - would track actual latency
            "data_freshness_seconds": 1,  # Real-time data
            "error_rate_percentage": 0.1,  # Placeholder - would track actual errors
            "last_quality_check": datetime.now(timezone.utc).isoformat(),
        }

    def _convert_symbol_to_kraken(self, symbol: str) -> str:
        """Convert standard symbol format to Kraken format"""
        # Standard format: BTCUSD -> Kraken format: BTC/USD
        if "/" not in symbol and len(symbol) >= 6:
            # Assume format like BTCUSD
            base = symbol[:-3]  # Everything except last 3 chars
            quote = symbol[-3:]  # Last 3 chars
            return f"{base}/{quote}"
        return symbol

    def _refresh_markets_cache(self):
        """Refresh the markets cache"""
        self._enforce_rate_limit()
        self._markets_cache = self.exchange.load_markets()
        self._cache_timestamp = datetime.now(timezone.utc)

    def _is_cache_expired(self, max_age_minutes: int = 60) -> bool:
        """Check if markets cache is expired"""
        if not self._cache_timestamp:
            return True

        age = datetime.now(timezone.utc) - self._cache_timestamp
        return age.total_seconds() > (max_age_minutes * 60)

    def _enforce_rate_limit(self):
        """Enforce rate limiting to respect API limits"""
        current_time = time.time()

        # Reset request count every minute
        if current_time - self._last_request_time > 60:
            self._request_count = 0

        # Basic rate limiting - adjust based on Kraken's actual limits
        if self._request_count >= 60:  # 60 requests per minute
            sleep_time = 60 - (current_time - self._last_request_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
                self._request_count = 0

        self._request_count += 1
        self._last_request_time = current_time
