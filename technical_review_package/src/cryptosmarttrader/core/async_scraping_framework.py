#!/usr/bin/env python3
"""
Async Scraping Framework
High-performance concurrent scraping with intelligent rate limiting and retry logic
"""

import asyncio
import aiohttp
import aiofiles
from aiohttp import ClientSession, ClientTimeout, TCPConnector
from asyncio import Semaphore, Queue
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Callable, Awaitable
import logging
import json
import time
import random
from dataclasses import dataclass, field
from enum import Enum
import warnings
import hashlib
import traceback
from pathlib import Path

warnings.filterwarnings("ignore")


class ScrapeStatus(Enum):
    """Scraping operation status"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    RETRY = "retry"


@dataclass
class ScrapeRequest:
    """Individual scrape request configuration"""

    url: str
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    data: Optional[Dict[str, Any]] = None
    timeout: float = 30.0
    priority: int = 1  # Lower = higher priority
    retry_count: int = 0
    max_retries: int = 3
    source_name: str = "unknown"
    request_id: str = field(
        default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
    )


@dataclass
class ScrapeResult:
    """Scraping operation result"""

    request: ScrapeRequest
    status: ScrapeStatus
    data: Optional[Any] = None
    response_time: float = 0.0
    status_code: Optional[int] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())
    response_headers: Dict[str, str] = field(default_factory=dict)
    response_size: int = 0


@dataclass
class RateLimitConfig:
    """Rate limiting configuration per source"""

    requests_per_second: float = 10.0
    burst_limit: int = 50
    cooldown_seconds: float = 60.0
    adaptive: bool = True
    backoff_multiplier: float = 2.0


class IntelligentRateLimiter:
    """Adaptive rate limiter with burst protection and cooldown"""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_times = []
        self.burst_count = 0
        self.last_request = 0.0
        self.cooldown_until = 0.0
        self.current_rate = config.requests_per_second
        self.success_streak = 0
        self.failure_streak = 0

    async def acquire(self) -> float:
        """Acquire permission to make request, returns delay in seconds"""

        current_time = time.time()

        # Check if in cooldown period
        if current_time < self.cooldown_until:
            delay = self.cooldown_until - current_time
            await asyncio.sleep(delay)
            current_time = time.time()

        # Clean old request times (older than 1 second)
        self.request_times = [t for t in self.request_times if current_time - t < 1.0]

        # Check burst limit
        if len(self.request_times) >= self.config.burst_limit:
            delay = 1.0 - (current_time - min(self.request_times))
            await asyncio.sleep(delay)
            current_time = time.time()
            self.request_times = [t for t in self.request_times if current_time - t < 1.0]

        # Check rate limit
        if len(self.request_times) >= self.current_rate:
            delay = 1.0 / self.current_rate
            await asyncio.sleep(delay)
            current_time = time.time()

        # Record request time
        self.request_times.append(current_time)
        self.last_request = current_time

        return 0.0

    def report_success(self):
        """Report successful request for adaptive rate adjustment"""
        self.success_streak += 1
        self.failure_streak = 0

        if self.config.adaptive and self.success_streak >= 10:
            # Gradually increase rate on success
            self.current_rate = min(self.config.requests_per_second * 1.5, self.current_rate * 1.1)
            self.success_streak = 0

    def report_failure(self, is_rate_limit: bool = False):
        """Report failed request for adaptive rate adjustment"""
        self.failure_streak += 1
        self.success_streak = 0

        if is_rate_limit:
            # Immediate cooldown for rate limiting
            self.cooldown_until = time.time() + self.config.cooldown_seconds
            self.current_rate *= 0.5  # Halve the rate
        elif self.config.adaptive and self.failure_streak >= 3:
            # Reduce rate on consecutive failures
            self.current_rate = max(self.config.requests_per_second * 0.1, self.current_rate * 0.8)


class AsyncScrapingFramework:
    """High-performance async scraping framework with concurrency control"""

    def __init__(
        self,
        max_concurrent: int = 100,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        enable_caching: bool = True,
        cache_ttl: int = 3600,
    ):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl

        # Concurrency control
        self.semaphore = Semaphore(max_concurrent)
        self.rate_limiters: Dict[str, IntelligentRateLimiter] = {}

        # Session management
        self.session: Optional[ClientSession] = None
        self.session_timeout = ClientTimeout(total=timeout)

        # Caching
        self.cache: Dict[str, Tuple[Any, float]] = {}

        # Statistics
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "rate_limited": 0,
            "timeouts": 0,
            "retries": 0,
            "total_response_time": 0.0,
            "requests_per_source": {},
            "start_time": time.time(),
        }

        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close_session()

    async def start_session(self):
        """Start aiohttp session with optimized settings"""

        connector = TCPConnector(
            limit=self.max_concurrent * 2,
            limit_per_host=50,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
        )

        self.session = ClientSession(
            connector=connector,
            timeout=self.session_timeout,
            headers={
                "User-Agent": "CryptoSmartTrader/2.0 (High-Performance Async Scraper)",
                "Accept": "application/json, text/plain, */*",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            },
        )

    async def close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None

    def get_rate_limiter(
        self, source_name: str, config: Optional[RateLimitConfig] = None
    ) -> IntelligentRateLimiter:
        """Get or create rate limiter for source"""

        if source_name not in self.rate_limiters:
            rate_config = config or RateLimitConfig()
            self.rate_limiters[source_name] = IntelligentRateLimiter(rate_config)

        return self.rate_limiters[source_name]

    async def scrape_single(self, request: ScrapeRequest) -> ScrapeResult:
        """Scrape single URL with rate limiting and retry logic"""

        # Check cache first
        if self.enable_caching:
            cache_key = self._get_cache_key(request)
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                self.stats["cache_hits"] += 1
                return ScrapeResult(
                    request=request,
                    status=ScrapeStatus.SUCCESS,
                    data=cached_result,
                    response_time=0.0,
                )

        # Acquire semaphore for concurrency control
        async with self.semaphore:
            # Apply rate limiting
            rate_limiter = self.get_rate_limiter(request.source_name)
            await rate_limiter.acquire()

            # Execute request with retries
            return await self._execute_with_retries(request, rate_limiter)

    async def _execute_with_retries(
        self, request: ScrapeRequest, rate_limiter: IntelligentRateLimiter
    ) -> ScrapeResult:
        """Execute request with intelligent retry logic"""

        last_error = None

        for attempt in range(request.max_retries + 1):
            if attempt > 0:
                # Exponential backoff with jitter
                delay = self.retry_delay * (2 ** (attempt - 1)) + random.choice
                await asyncio.sleep(delay)
                self.stats["retries"] += 1

            try:
                result = await self._execute_request(request)

                # Update statistics
                self.stats["total_requests"] += 1
                self.stats["total_response_time"] += result.response_time
                self.stats["requests_per_source"][request.source_name] = (
                    self.stats["requests_per_source"].get(request.source_name, 0) + 1
                )

                if result.status == ScrapeStatus.SUCCESS:
                    self.stats["successful_requests"] += 1
                    rate_limiter.report_success()

                    # Cache successful results
                    if self.enable_caching and result.data is not None:
                        cache_key = self._get_cache_key(request)
                        self._store_in_cache(cache_key, result.data)

                    return result

                elif result.status == ScrapeStatus.RATE_LIMITED:
                    self.stats["rate_limited"] += 1
                    rate_limiter.report_failure(is_rate_limit=True)
                    last_error = result.error
                    continue

                elif result.status == ScrapeStatus.TIMEOUT:
                    self.stats["timeouts"] += 1
                    rate_limiter.report_failure()
                    last_error = result.error
                    continue

                else:
                    self.stats["failed_requests"] += 1
                    rate_limiter.report_failure()
                    last_error = result.error
                    continue

            except Exception as e:
                self.stats["failed_requests"] += 1
                rate_limiter.report_failure()
                last_error = str(e)
                self.logger.error(f"Unexpected error in request {request.request_id}: {e}")
                continue

        # All retries exhausted
        return ScrapeResult(
            request=request,
            status=ScrapeStatus.FAILED,
            error=f"Max retries exhausted. Last error: {last_error}",
            timestamp=datetime.utcnow(),
        )

    async def _execute_request(self, request: ScrapeRequest) -> ScrapeResult:
        """Execute single HTTP request"""

        if not self.session:
            await self.start_session()

        start_time = time.time()

        try:
            async with self.session.request(
                method=request.method,
                url=request.url,
                headers=request.headers,
                params=request.params,
                json=request.data if request.method != "GET" else None,
                timeout=ClientTimeout(total=request.timeout),
            ) as response:
                response_time = time.time() - start_time

                # Check for rate limiting
                if response.status == 429:
                    return ScrapeResult(
                        request=request,
                        status=ScrapeStatus.RATE_LIMITED,
                        status_code=response.status,
                        response_time=response_time,
                        error="Rate limited by server",
                        response_headers=dict(response.headers),
                    )

                # Check for other errors
                if response.status >= 400:
                    return ScrapeResult(
                        request=request,
                        status=ScrapeStatus.FAILED,
                        status_code=response.status,
                        response_time=response_time,
                        error=f"HTTP {response.status}",
                        response_headers=dict(response.headers),
                    )

                # Parse response data
                content_type = response.headers.get("content-type", "").lower()

                if "application/json" in content_type:
                    data = await response.json()
                else:
                    data = await response.text()

                return ScrapeResult(
                    request=request,
                    status=ScrapeStatus.SUCCESS,
                    data=data,
                    response_time=response_time,
                    status_code=response.status,
                    response_headers=dict(response.headers),
                    response_size=len(str(data)),
                )

        except asyncio.TimeoutError:
            return ScrapeResult(
                request=request,
                status=ScrapeStatus.TIMEOUT,
                response_time=time.time() - start_time,
                error=f"Request timeout after {request.timeout}s",
            )

        except Exception as e:
            return ScrapeResult(
                request=request,
                status=ScrapeStatus.FAILED,
                response_time=time.time() - start_time,
                error=str(e),
            )

    async def scrape_batch(
        self,
        requests: List[ScrapeRequest],
        progress_callback: Optional[Callable[[int, int], Awaitable[None]]] = None,
    ) -> List[ScrapeResult]:
        """Scrape multiple URLs concurrently"""

        if not self.session:
            await self.start_session()

        # Sort by priority (lower number = higher priority)
        sorted_requests = sorted(requests, key=lambda r: r.priority)

        # Create tasks
        tasks = []
        for i, request in enumerate(sorted_requests):
            task = asyncio.create_task(
                self.scrape_single(request), name=f"scrape_{request.request_id}"
            )
            tasks.append(task)

        # Execute with progress tracking
        results = []
        completed = 0

        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
                completed += 1

                # Progress callback
                if progress_callback:
                    await progress_callback(completed, len(tasks))

            except Exception as e:
                self.logger.error(f"Task failed: {e}")
                # Create failed result
                failed_result = ScrapeResult(
                    request=ScrapeRequest(url="unknown"), status=ScrapeStatus.FAILED, error=str(e)
                )
                results.append(failed_result)
                completed += 1

        # Sort results back to original request order
        request_id_to_index = {req.request_id: i for i, req in enumerate(sorted_requests)}
        sorted_results = [None] * len(results)

        for result in results:
            original_index = request_id_to_index.get(result.request.request_id, 0)
            sorted_results[original_index] = result

        return [r for r in sorted_results if r is not None]

    def _get_cache_key(self, request: ScrapeRequest) -> str:
        """Generate cache key for request"""

        key_data = {
            "url": request.url,
            "method": request.method,
            "params": request.params,
            "data": request.data,
        }

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if not expired"""

        if cache_key in self.cache:
            data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return data
            else:
                # Remove expired entry
                del self.cache[cache_key]

        return None

    def _store_in_cache(self, cache_key: str, data: Any):
        """Store data in cache with timestamp"""

        self.cache[cache_key] = (data, time.time())

        # Simple cache cleanup (remove oldest 10% when cache gets large)
        if len(self.cache) > 1000:
            oldest_keys = sorted(self.cache.keys(), key=lambda k: self.cache[k][1])[:100]

            for key in oldest_keys:
                del self.cache[key]

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scraping statistics"""

        runtime = time.time() - self.stats["start_time"]

        return {
            "runtime_seconds": runtime,
            "total_requests": self.stats["total_requests"],
            "successful_requests": self.stats["successful_requests"],
            "failed_requests": self.stats["failed_requests"],
            "success_rate": self.stats["successful_requests"]
            / max(self.stats["total_requests"], 1),
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate": self.stats["cache_hits"] / max(self.stats["total_requests"], 1),
            "rate_limited": self.stats["rate_limited"],
            "timeouts": self.stats["timeouts"],
            "retries": self.stats["retries"],
            "avg_response_time": self.stats["total_response_time"]
            / max(self.stats["successful_requests"], 1),
            "requests_per_second": self.stats["total_requests"] / max(runtime, 1),
            "requests_per_source": self.stats["requests_per_source"],
            "rate_limiter_status": {
                source: {
                    "current_rate": limiter.current_rate,
                    "success_streak": limiter.success_streak,
                    "failure_streak": limiter.failure_streak,
                }
                for source, limiter in self.rate_limiters.items()
            },
        }

    async def scrape_with_queue(
        self,
        request_generator: Callable[[], List[ScrapeRequest]],
        batch_size: int = 100,
        max_batches: Optional[int] = None,
    ) -> List[ScrapeResult]:
        """Scrape using producer-consumer pattern with queue"""

        results = []
        batch_count = 0

        while max_batches is None or batch_count < max_batches:
            # Generate next batch of requests
            requests = request_generator()

            if not requests:
                break

            # Limit batch size
            batch_requests = requests[:batch_size]

            # Process batch
            batch_results = await self.scrape_batch(batch_requests)
            results.extend(batch_results)

            batch_count += 1

            # Log progress
            self.logger.info(f"Completed batch {batch_count}, total results: {len(results)}")

        return results


# Convenience functions for common use cases


async def scrape_crypto_exchanges(
    exchange_configs: Dict[str, Dict[str, Any]], symbols: List[str] = None, max_concurrent: int = 50
) -> Dict[str, List[ScrapeResult]]:
    """Scrape multiple crypto exchanges concurrently"""

    if symbols is None:
        symbols = ["BTC/USD", "ETH/USD", "ADA/USD", "DOT/USD", "LINK/USD"]

    async with AsyncScrapingFramework(max_concurrent=max_concurrent) as scraper:
        all_requests = []

        for exchange_name, config in exchange_configs.items():
            base_url = config["base_url"]
            headers = config.get("headers", {})
            rate_limit = RateLimitConfig(
                requests_per_second=config.get("rate_limit", 10),
                burst_limit=config.get("burst_limit", 50),
            )

            # Configure rate limiter
            scraper.get_rate_limiter(exchange_name, rate_limit)

            # Create requests for each symbol
            for symbol in symbols:
                url = f"{base_url}/ticker/{symbol.replace('/', '')}"
                request = ScrapeRequest(
                    url=url, headers=headers, source_name=exchange_name, priority=1
                )
                all_requests.append(request)

        # Execute all requests
        results = await scraper.scrape_batch(all_requests)

        # Group results by exchange
        exchange_results = {}
        for result in results:
            exchange_name = result.request.source_name
            if exchange_name not in exchange_results:
                exchange_results[exchange_name] = []
            exchange_results[exchange_name].append(result)

        return exchange_results


async def scrape_news_sources(
    news_configs: Dict[str, Dict[str, Any]], keywords: List[str] = None, max_concurrent: int = 30
) -> List[ScrapeResult]:
    """Scrape multiple news sources for crypto-related content"""

    if keywords is None:
        keywords = ["bitcoin", "ethereum", "cryptocurrency", "blockchain"]

    async with AsyncScrapingFramework(max_concurrent=max_concurrent) as scraper:
        requests = []

        for source_name, config in news_configs.items():
            base_url = config["base_url"]
            headers = config.get("headers", {})

            for keyword in keywords:
                url = f"{base_url}/search?q={keyword}"
                request = ScrapeRequest(
                    url=url, headers=headers, source_name=source_name, priority=2
                )
                requests.append(request)

        return await scraper.scrape_batch(requests)


def create_scraping_framework(
    max_concurrent: int = 100,
    timeout: float = 30.0,
    max_retries: int = 3,
    enable_caching: bool = True,
) -> AsyncScrapingFramework:
    """Create configured async scraping framework"""

    return AsyncScrapingFramework(
        max_concurrent=max_concurrent,
        timeout=timeout,
        max_retries=max_retries,
        enable_caching=enable_caching,
    )
