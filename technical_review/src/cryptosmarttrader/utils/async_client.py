# utils/async_client.py
import aiohttp
import asyncio
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Dict, Any, Optional, List, Union
import logging
import time
import json
from config.settings import config
from utils.metrics import metrics_server


logger = logging.getLogger(__name__)


class AsyncHTTPClient:
    """
    Asynchronous HTTP client with retry logic and error handling.
    Implements Dutch requirements for async I/O and retry/backoff.
    """

    def __init__(self, timeout: int = 30, max_retries: int = 5):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def fetch_json(self, url: str, headers: Optional[Dict[str, str]] = None,
                        params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fetch JSON data from URL with retry logic.

        Args:
            url: Target URL
            headers: Optional HTTP headers
            params: Optional query parameters

        Returns:
            JSON response data

        Raises:
            aiohttp.ClientError: On HTTP errors
            asyncio.TimeoutError: On timeout
        """
        start_time = time.time()

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((
                aiohttp.ClientError,
                asyncio.TimeoutError,
                aiohttp.ServerTimeoutError
            )),
            reraise=True,
        ):
            with attempt:
                if not self.session:
                    raise RuntimeError("HTTP session not initialized. Use async context manager.")

                logger.debug(f"Fetching JSON from {url} (attempt {attempt.retry_state.attempt_number})")

                async with self.session.get(url, headers=headers, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

                    # Record metrics
                    latency = time.time() - start_time
                    metrics_server.record_exchange_latency("http_client", latency)

                    logger.debug(f"Successfully fetched data from {url}")
                    return data

    async def post_json(self, url: str, data: Dict[str, Any],
                       headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        POST JSON data to URL with retry logic.

        Args:
            url: Target URL
            data: JSON data to send
            headers: Optional HTTP headers

        Returns:
            JSON response data
        """
        start_time = time.time()

        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((
                aiohttp.ClientError,
                asyncio.TimeoutError
            )),
            reraise=True,
        ):
            with attempt:
                if not self.session:
                    raise RuntimeError("HTTP session not initialized. Use async context manager.")

                logger.debug(f"POSTing to {url} (attempt {attempt.retry_state.attempt_number})")

                async with self.session.post(
                    url,
                    json=data,
                    headers=headers
                ) as response:
                    response.raise_for_status()
                    result = await response.json()

                    # Record metrics
                    latency = time.time() - start_time
                    metrics_server.record_exchange_latency("http_client", latency)

                    logger.debug(f"Successfully posted to {url}")
                    return result

    async def fetch_multiple(self, urls: List[str],
                           headers: Optional[Dict[str, str]] = None,
                           max_concurrent: int = 10) -> List[Dict[str, Any]]:
        """
        Fetch multiple URLs concurrently with rate limiting.

        Args:
            urls: List of URLs to fetch
            headers: Optional HTTP headers
            max_concurrent: Maximum concurrent requests

        Returns:
            List of JSON responses in same order as URLs
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_with_semaphore(url: str) -> Dict[str, Any]:
            async with semaphore:
                try:
                    return await self.fetch_json(url, headers)
                except Exception as e:
                    logger.error(f"Failed to fetch {url}: {e}")
                    return {"error": str(e), "url": url}

        tasks = [fetch_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks)

        return results


class AsyncMarketDataClient:
    """
    Specialized async client for market data APIs.
    """

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.client = AsyncHTTPClient()

    async def __aenter__(self):
        await self.client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.__aexit__(exc_type, exc_val, exc_tb)

    def _get_headers(self) -> Dict[str, str]:
        """Get standard headers for API requests"""
        headers = {
            "User-Agent": "CryptoSmartTrader/2.0",
            "Accept": "application/json"
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def get_market_data(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Fetch market data for specified symbols.

        Args:
            symbols: List of cryptocurrency symbols

        Returns:
            Market data dictionary
        """
        url = f"{self.base_url}/tickers"
        params = {"symbols": ",".join(symbols)}

        try:
            return await self.client.fetch_json(
                url,
                headers=self._get_headers(),
                params=params
            )
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            metrics_server.record_error("market_data_fetch", "async_client")
            raise

    async def get_historical_data(self, symbol: str, timeframe: str = "1h",
                                 limit: int = 100) -> Dict[str, Any]:
        """
        Fetch historical price data.

        Args:
            symbol: Cryptocurrency symbol
            timeframe: Data timeframe (1h, 4h, 1d, etc.)
            limit: Number of data points

        Returns:
            Historical data dictionary
        """
        url = f"{self.base_url}/ohlcv/{symbol}"
        params = {
            "timeframe": timeframe,
            "limit": limit
        }

        try:
            return await self.client.fetch_json(
                url,
                headers=self._get_headers(),
                params=params
            )
        except Exception as e:
            logger.error(f"Failed to fetch historical data for {symbol}: {e}")
            metrics_server.record_error("historical_data_fetch", "async_client")
            raise


# Utility functions for common async operations
async def fetch_sentiment_data(sources: List[str]) -> List[Dict[str, Any]]:
    """
    Fetch sentiment data from multiple sources concurrently.

    Args:
        sources: List of sentiment data source URLs

    Returns:
        List of sentiment data responses
    """
    async with AsyncHTTPClient() as client:
        return await client.fetch_multiple(sources, max_concurrent=5)


async def test_exchange_connectivity(exchange_urls: List[str]) -> Dict[str, bool]:
    """
    Test connectivity to multiple exchanges concurrently.

    Args:
        exchange_urls: List of exchange API URLs

    Returns:
        Dictionary mapping exchange URLs to connectivity status
    """
    async with AsyncHTTPClient() as client:
        results = await client.fetch_multiple(exchange_urls, max_concurrent=10)

        connectivity = {}
        for i, url in enumerate(exchange_urls):
            connectivity[url] = "error" not in results[i]

        return connectivity
