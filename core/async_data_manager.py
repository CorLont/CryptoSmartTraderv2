#!/usr/bin/env python3
"""
Async Data Manager with Global Rate Limiting and Robust Error Handling
Replaces blocking I/O with full async/await implementation
"""

import asyncio
import aiohttp
import aiofiles
import json
import time
import os
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from pathlib import Path
import logging
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
from asyncio_throttle.throttler import Throttler
import ccxt.async_support as ccxt_async
from dataclasses import dataclass
from core.logging_manager import get_logger
from core.data_quality_manager import get_data_quality_manager
from core.data_integrity_validator import get_data_integrity_validator
from core.secrets_manager import get_secrets_manager, secure_function, SecretRedactor


@dataclass
class RateLimitConfig:
    requests_per_second: float = 10.0
    burst_size: int = 50
    cool_down_period: int = 60


class AsyncDataManager:
    def __init__(self, rate_limit_config: Optional[RateLimitConfig] = None):
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self.throttler = Throttler(rate_limit=int(self.rate_limit_config.requests_per_second))
        self.session: Optional[aiohttp.ClientSession] = None
        self.exchanges: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        self.structured_logger = get_logger()

        # Initialize data quality components
        self.data_quality_manager = get_data_quality_manager()
        self.integrity_validator = get_data_integrity_validator()

        # Global rate limiter for all API calls
        self.api_semaphore = asyncio.Semaphore(self.rate_limit_config.burst_size)
        self.last_api_calls: List[float] = []

    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()

    async def initialize(self):
        """Initialize async session and exchanges"""
        # Create aiohttp session with optimized settings
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        connector = aiohttp.TCPConnector(
            limit=100,  # Connection pool size
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
        )

        self.session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
            headers={
                "User-Agent": "CryptoSmartTrader-AsyncClient/2.0",
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
            },
        )

        # Initialize async exchanges
        await self.setup_async_exchanges()

        self.logger.info("Async Data Manager initialized")

    @secure_function(redact_kwargs=["apiKey", "secret"])
    async def setup_async_exchanges(self):
        """Setup async exchange connections with secure credential handling"""
        secrets_manager = get_secrets_manager()

        try:
            # Kraken async with secure credentials
            kraken_key = secrets_manager.get_secret("KRAKEN_API_KEY")
            kraken_secret = secrets_manager.get_secret("KRAKEN_SECRET")

            if kraken_key and kraken_secret:
                self.exchanges["kraken"] = ccxt_async.kraken(
                    {
                        "apiKey": kraken_key,
                        "secret": kraken_secret,
                        "enableRateLimit": True,
                        "timeout": 30000,
                        "session": self.session,
                    }
                )
                self.structured_logger.info("Kraken exchange initialized with authenticated API")
            else:
                self.exchanges["kraken"] = ccxt_async.kraken(
                    {"enableRateLimit": True, "timeout": 30000, "session": self.session}
                )
                self.structured_logger.warning(
                    "Kraken exchange initialized in public mode - no API credentials"
                )

            # Binance with secure credentials
            binance_key = secrets_manager.get_secret("BINANCE_API_KEY")
            binance_secret = secrets_manager.get_secret("BINANCE_SECRET")

            if binance_key and binance_secret:
                self.exchanges["binance"] = ccxt_async.binance(
                    {
                        "apiKey": binance_key,
                        "secret": binance_secret,
                        "enableRateLimit": True,
                        "timeout": 30000,
                        "session": self.session,
                    }
                )
                self.structured_logger.info("Binance exchange initialized with authenticated API")
            else:
                self.exchanges["binance"] = ccxt_async.binance(
                    {"enableRateLimit": True, "timeout": 30000, "session": self.session}
                )
                self.structured_logger.warning(
                    "Binance exchange initialized in public mode - no API credentials"
                )

            self.structured_logger.info(
                f"Async exchange initialization completed",
                extra={
                    "total_exchanges": len(self.exchanges),
                    "authenticated_exchanges": sum(
                        1
                        for name in self.exchanges.keys()
                        if secrets_manager.get_secret(f"{name.upper()}_API_KEY"),
                },
            )

        except Exception as e:
            # Use SecretRedactor to ensure no credentials leak in error logs
            sanitized_error = SecretRedactor.sanitize_exception(e)
            self.structured_logger.error(
                f"Failed to setup async exchanges: {sanitized_error}",
                extra={"error_type": type(e).__name__},
            )

    async def global_rate_limit(self):
        """Global rate limiter with burst control"""
        async with self.api_semaphore:
            current_time = time.time()

            # Clean old timestamps (older than 1 second)
            self.last_api_calls = [
                call_time for call_time in self.last_api_calls if current_time - call_time < 1.0
            ]

            # Check if we're within rate limits
            if len(self.last_api_calls) >= self.rate_limit_config.requests_per_second:
                sleep_time = 1.0 - (current_time - self.last_api_calls[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            self.last_api_calls.append(current_time)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1, max=60, jitter=2),
        retry=retry_if_exception_type(
            (aiohttp.ClientError, asyncio.TimeoutError, ccxt_async.BaseError),
    )
    async def fetch_market_data_async(self, exchange_name: str) -> Dict[str, Any]:
        """Fetch market data with async retries and rate limiting"""
        await self.global_rate_limit()

        if exchange_name not in self.exchanges:
            raise ValueError(f"Exchange {exchange_name} not configured")

        exchange = self.exchanges[exchange_name]

        try:
            # Concurrent fetch of multiple data types
            tasks = [
                self.fetch_tickers_async(exchange),
                self.fetch_ohlcv_batch_async(exchange),
                self.fetch_order_book_async(exchange),
            ]

            tickers, ohlcv_data, order_books = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle partial failures gracefully
            result = {
                "exchange": exchange_name,
                "timestamp": datetime.now().isoformat(),
                "tickers": tickers if not isinstance(tickers, Exception) else None,
                "ohlcv": ohlcv_data if not isinstance(ohlcv_data, Exception) else None,
                "order_books": order_books if not isinstance(order_books, Exception) else None,
                "errors": [],
            }

            # Collect any errors
            for data, name in [
                (tickers, "tickers"),
                (ohlcv_data, "ohlcv"),
                (order_books, "order_books"),
            ]:
                if isinstance(data, Exception):
                    result["errors"].append(f"{name}: {str(data)}")

            return result

        except Exception as e:
            self.logger.error(f"Failed to fetch market data from {exchange_name}: {e}")
            raise

    async def fetch_tickers_async(self, exchange) -> Dict[str, Any]:
        """Fetch tickers with async throttling"""
        async with self.throttler:
            tickers = await exchange.fetch_tickers()

            # Filter for USD pairs and add enhanced data
            usd_pairs = {}
            for symbol, ticker in tickers.items():
                if symbol.endswith("/USD"):
                    usd_pairs[symbol] = {
                        "price": ticker["last"],
                        "bid": ticker["bid"],
                        "ask": ticker["ask"],
                        "spread": ((ticker["ask"] - ticker["bid"]) / ticker["bid"] * 100)
                        if ticker["bid"]
                        else 0,
                        "volume": ticker["quoteVolume"],
                        "change": ticker["percentage"],
                        "high": ticker["high"],
                        "low": ticker["low"],
                        "vwap": ticker.get("vwap"),
                        "timestamp": ticker["timestamp"],
                    }

            return usd_pairs

    async def fetch_ohlcv_batch_async(
        self, exchange, timeframes: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Fetch OHLCV data for multiple timeframes concurrently"""
        timeframes = timeframes or ["1h", "1d"]
        symbols = ["BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "SOL/USD"]

        tasks = []
        for symbol in symbols:
            for timeframe in timeframes:
                task = self.fetch_single_ohlcv_async(exchange, symbol, timeframe)
                tasks.append((symbol, timeframe, task))

        # Execute all OHLCV fetches concurrently
        results = await asyncio.gather(*[task for _, _, task in tasks], return_exceptions=True)

        # Organize results
        ohlcv_data = {}
        for (symbol, timeframe, _), result in zip(tasks, results):
            if not isinstance(result, Exception):
                if symbol not in ohlcv_data:
                    ohlcv_data[symbol] = {}
                ohlcv_data[symbol][timeframe] = result

        return ohlcv_data

    async def fetch_single_ohlcv_async(self, exchange, symbol: str, timeframe: str) -> List[List]:
        """Fetch single OHLCV with rate limiting and metrics"""
        start_time = time.time()

        async with self.throttler:
            try:
                result = await exchange.fetch_ohlcv(symbol, timeframe, limit=100)

                # Log successful API request
                duration = time.time() - start_time
                self.structured_logger.log_api_request(
                    exchange=exchange.name if hasattr(exchange, "name") else "unknown",
                    endpoint=f"ohlcv/{symbol}/{timeframe}",
                    duration=duration,
                    status="success",
                )

                return result

            except Exception as e:
                duration = time.time() - start_time

                # Log failed API request
                self.structured_logger.log_api_request(
                    exchange=exchange.name if hasattr(exchange, "name") else "unknown",
                    endpoint=f"ohlcv/{symbol}/{timeframe}",
                    duration=duration,
                    status="error",
                )

                self.structured_logger.warning(
                    f"Failed to fetch OHLCV data",
                    extra={
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "error": str(e),
                        "duration": duration,
                    },
                )
                return []

    async def fetch_order_book_async(self, exchange) -> Dict[str, Any]:
        """Fetch order books for major pairs"""
        major_pairs = ["BTC/USD", "ETH/USD", "ADA/USD"]

        tasks = [self.fetch_single_order_book_async(exchange, symbol) for symbol in major_pairs]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        order_books = {}
        for symbol, result in zip(major_pairs, results):
            if not isinstance(result, Exception):
                order_books[symbol] = result

        return order_books

    async def fetch_single_order_book_async(self, exchange, symbol: str) -> Dict[str, Any]:
        """Fetch single order book with throttling"""
        async with self.throttler:
            try:
                book = await exchange.fetch_order_book(symbol, limit=20)
                return {
                    "bids": book["bids"][:10],  # Top 10 bids
                    "asks": book["asks"][:10],  # Top 10 asks
                    "timestamp": book["timestamp"],
                }
            except Exception as e:
                self.logger.warning(f"Failed to fetch order book for {symbol}: {e}")
                return {}

    async def store_data_async(self, data: Dict[str, Any], file_path: Path):
        """Idempotent async file writes with atomic operations"""
        try:
            # Create directory if needed
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write to temporary file first (atomic operation)
            temp_path = file_path.with_suffix(file_path.suffix + ".tmp")

            async with aiofiles.open(temp_path, "w") as f:
                await f.write(json.dumps(data, indent=2, default=str))

            # Atomic rename (idempotent)
            temp_path.replace(file_path)

            self.logger.debug(f"Data stored to {file_path}")

        except Exception as e:
            self.logger.error(f"Failed to store data to {file_path}: {e}")
            raise

    async def batch_collect_all_exchanges_with_integrity_filter(self) -> Dict[str, Any]:
        """Collect data from all exchanges with HARD integrity filter - incomplete coins BLOCKED"""
        tasks = [
            self.fetch_market_data_async(exchange_name) for exchange_name in self.exchanges.keys()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        collected_data = {
            "timestamp": datetime.now().isoformat(),
            "exchanges": {},
            "summary": {
                "total_exchanges": len(self.exchanges),
                "successful": 0,
                "failed": 0,
                "coins_blocked_incomplete": 0,
                "coins_passed_filter": 0,
            },
        }

        for exchange_name, result in zip(self.exchanges.keys(), results):
            if isinstance(result, Exception):
                collected_data["exchanges"][exchange_name] = {
                    "status": "error",
                    "error": str(result),
                }
                collected_data["summary"]["failed"] += 1
            else:
                # Apply HARD data integrity filter
                filtered_result = await self._apply_hard_integrity_filter(result, exchange_name)
                collected_data["exchanges"][exchange_name] = filtered_result
                collected_data["summary"]["successful"] += 1
                collected_data["summary"]["coins_blocked_incomplete"] += filtered_result.get(
                    "blocked_coins_count", 0
                )
                collected_data["summary"]["coins_passed_filter"] += filtered_result.get(
                    "passed_coins_count", 0
                )

        return collected_data

    async def batch_collect_all_exchanges(self) -> Dict[str, Any]:
        """Backward compatibility wrapper - now uses hard integrity filter by default"""
        return await self.batch_collect_all_exchanges_with_integrity_filter()

    async def _apply_hard_integrity_filter(
        self, raw_data: Dict[str, Any], exchange_name: str
    ) -> Dict[str, Any]:
        """Apply HARD integrity filter - incomplete coins are BLOCKED, not passed through"""

        filtered_data = {
            "exchange": raw_data.get("exchange", exchange_name),
            "timestamp": raw_data.get("timestamp"),
            "tickers": {},
            "ohlcv": {},
            "order_books": {},
            "errors": raw_data.get("errors", []),
            "integrity_filter_results": {
                "total_coins_scanned": 0,
                "coins_passed_filter": 0,
                "coins_blocked_incomplete": 0,
                "blocked_reasons": {},
                "completeness_threshold": 0.8,  # 80% minimum completeness required
            },
            "blocked_coins_count": 0,
            "passed_coins_count": 0,
        }

        # Process tickers with hard filter
        if raw_data.get("tickers"):
            for symbol, ticker_data in raw_data["tickers"].items():
                filtered_data["integrity_filter_results"]["total_coins_scanned"] += 1

                # Check data completeness for this coin
                completeness_result = await self._check_coin_completeness(
                    symbol, ticker_data, raw_data
                )

                if completeness_result["is_complete"]:
                    # Coin passes filter - include in output
                    filtered_data["tickers"][symbol] = ticker_data
                    filtered_data["integrity_filter_results"]["coins_passed_filter"] += 1
                    filtered_data["passed_coins_count"] += 1

                    self.structured_logger.debug(
                        f"Coin passed integrity filter: {symbol}",
                        extra={
                            "symbol": symbol,
                            "completeness_score": completeness_result["completeness_score"],
                            "exchange": exchange_name,
                        },
                    )
                else:
                    # Coin BLOCKED - incomplete data
                    filtered_data["integrity_filter_results"]["coins_blocked_incomplete"] += 1
                    filtered_data["blocked_coins_count"] += 1

                    # Track reason for blocking
                    reason = (
                        completeness_result["missing_components"][0]
                        if completeness_result["missing_components"]
                        else "unknown"
                    )
                    if reason not in filtered_data["integrity_filter_results"]["blocked_reasons"]:
                        filtered_data["integrity_filter_results"]["blocked_reasons"][reason] = 0
                    filtered_data["integrity_filter_results"]["blocked_reasons"][reason] += 1

                    self.structured_logger.warning(
                        f"COIN BLOCKED - Incomplete data: {symbol}",
                        extra={
                            "symbol": symbol,
                            "completeness_score": completeness_result["completeness_score"],
                            "missing_components": completeness_result["missing_components"],
                            "exchange": exchange_name,
                            "action": "BLOCKED",
                        },
                    )

        # Process OHLCV data with same hard filter
        if raw_data.get("ohlcv"):
            for symbol, ohlcv_data in raw_data["ohlcv"].items():
                # Only include OHLCV if ticker passed filter
                if symbol in filtered_data["tickers"]:
                    filtered_data["ohlcv"][symbol] = ohlcv_data

        # Process order books with same hard filter
        if raw_data.get("order_books"):
            for symbol, order_book_data in raw_data["order_books"].items():
                # Only include order book if ticker passed filter
                if symbol in filtered_data["tickers"]:
                    filtered_data["order_books"][symbol] = order_book_data

        # Log filtering results
        self.structured_logger.info(
            f"Hard integrity filter applied for {exchange_name}",
            extra={
                "exchange": exchange_name,
                "total_scanned": filtered_data["integrity_filter_results"]["total_coins_scanned"],
                "passed_filter": filtered_data["integrity_filter_results"]["coins_passed_filter"],
                "blocked_incomplete": filtered_data["integrity_filter_results"][
                    "coins_blocked_incomplete"
                ],
                "blocked_reasons": filtered_data["integrity_filter_results"]["blocked_reasons"],
            },
        )

        return filtered_data

    async def _check_coin_completeness(
        self, symbol: str, ticker_data: Dict[str, Any], raw_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if coin has complete data across all required components"""

        required_components = {
            "price_data": 0.3,  # 30% weight
            "volume_data": 0.2,  # 20% weight
            "ohlcv_data": 0.2,  # 20% weight
            "order_book_data": 0.1,  # 10% weight
            "sentiment_data": 0.1,  # 10% weight
            "technical_data": 0.1,  # 10% weight
        }

        component_scores = {}
        missing_components = []

        # 1. Price data completeness
        price_score = 0.0
        if ticker_data.get("price") and ticker_data.get("bid") and ticker_data.get("ask"):
            if ticker_data["price"] > 0 and ticker_data["bid"] > 0 and ticker_data["ask"] > 0:
                price_score = 1.0

        if price_score < 0.8:
            missing_components.append("price_data")
        component_scores["price_data"] = price_score

        # 2. Volume data completeness
        volume_score = 0.0
        if ticker_data.get("volume") and ticker_data["volume"] > 0:
            volume_score = 1.0

        if volume_score < 0.8:
            missing_components.append("volume_data")
        component_scores["volume_data"] = volume_score

        # 3. OHLCV data completeness
        ohlcv_score = 0.0
        if raw_data.get("ohlcv", {}).get(symbol):
            ohlcv_data = raw_data["ohlcv"][symbol]
            # Check if we have data for multiple timeframes
            valid_timeframes = 0
            for timeframe, data in ohlcv_data.items():
                if data and len(data) > 10:  # At least 10 candles
                    valid_timeframes += 1

            if valid_timeframes >= 1:  # At least 1 valid timeframe
                ohlcv_score = min(valid_timeframes / 2.0, 1.0)  # Max score for 2+ timeframes

        if ohlcv_score < 0.5:
            missing_components.append("ohlcv_data")
        component_scores["ohlcv_data"] = ohlcv_score

        # 4. Order book data completeness
        order_book_score = 0.0
        if raw_data.get("order_books", {}).get(symbol):
            order_book = raw_data["order_books"][symbol]
            if order_book.get("bids") and order_book.get("asks"):
                if len(order_book["bids"]) >= 5 and len(order_book["asks"]) >= 5:
                    order_book_score = 1.0

        if order_book_score < 0.3:
            missing_components.append("order_book_data")
        component_scores["order_book_data"] = order_book_score

        # 5. Sentiment data completeness (simulated check - would integrate with real sentiment API)
        sentiment_score = 0.0
        # In real implementation, check if sentiment data exists
        # For now, simulate based on volume (high volume = likely to have sentiment data)
        if ticker_data.get("volume", 0) > 1000000:  # $1M+ volume
            sentiment_score = 0.8
        elif ticker_data.get("volume", 0) > 100000:  # $100k+ volume
            sentiment_score = 0.5

        if sentiment_score < 0.3:
            missing_components.append("sentiment_data")
        component_scores["sentiment_data"] = sentiment_score

        # 6. Technical data completeness
        technical_score = 0.0
        # Check if we have high/low/change data
        if (
            ticker_data.get("high")
            and ticker_data.get("low")
            and ticker_data.get("change") is not None
        ):
            technical_score = 1.0

        if technical_score < 0.5:
            missing_components.append("technical_data")
        component_scores["technical_data"] = technical_score

        # Calculate weighted completeness score
        total_score = sum(
            component_scores[component] * weight
            for component, weight in required_components.items()

        # Hard threshold: 80% completeness required
        is_complete = total_score >= 0.8 and len(missing_components) <= 2

        return {
            "symbol": symbol,
            "is_complete": is_complete,
            "completeness_score": total_score,
            "component_scores": component_scores,
            "missing_components": missing_components,
            "threshold": 0.8,
        }

    async def cleanup(self):
        """Cleanup async resources"""
        # Close all exchange connections
        for exchange in self.exchanges.values():
            try:
                await exchange.close()
            except Exception as e:
                self.logger.error(f"Error closing exchange: {e}")

        # Close aiohttp session
        if self.session:
            await self.session.close()

        self.logger.info("Async Data Manager cleaned up")


# Global async data manager instance
async_data_manager = None


async def get_async_data_manager() -> AsyncDataManager:
    """Get or create global async data manager"""
    global async_data_manager

    if async_data_manager is None:
        async_data_manager = AsyncDataManager()
        await async_data_manager.initialize()

    return async_data_manager
