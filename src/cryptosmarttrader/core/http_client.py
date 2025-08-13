#!/usr/bin/env python3
"""
Enterprise HTTP Client - Centralized async client with resilience patterns
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential_jitter


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class CircuitBreaker:
    """Circuit breaker for external service resilience"""

    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3

    # State tracking
    failure_count: int = field(default=0)
    last_failure_time: Optional[datetime] = field(default=None)
    state: CircuitState = field(default=CircuitState.CLOSED)
    successful_calls: int = field(default=0)

    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if (
                self.last_failure_time
                and (datetime.utcnow() - self.last_failure_time).seconds >= self.recovery_timeout
            ):
                self.state = CircuitState.HALF_OPEN
                self.successful_calls = 0
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return self.successful_calls < self.half_open_max_calls

        return False

    def record_success(self):
        """Record successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self.successful_calls += 1
            if self.successful_calls >= self.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0

    def record_failure(self):
        """Record failed call"""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()

        if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN


@dataclass
class CacheEntry:
    """Cache entry with TTL and metadata"""

    data: Any
    timestamp: datetime
    ttl_seconds: int
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = field(default=0)

    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return (datetime.utcnow() - self.timestamp).total_seconds() > self.ttl_seconds

    def is_stale(self, stale_threshold: float = 0.8) -> bool:
        """Check if cache entry is stale (approaching expiry)"""
        elapsed = (datetime.utcnow() - self.timestamp).total_seconds()
        return elapsed > (self.ttl_seconds * stale_threshold)

    def access(self):
        """Record cache access"""
        self.access_count += 1
        self.last_accessed = datetime.utcnow()


class EnterpriseHTTPClient:
    """
    Enterprise HTTP client with resilience patterns

    Features:
    - Circuit breakers per service
    - Exponential backoff with jitter
    - Response caching with TTL
    - Rate limiting per endpoint
    - Comprehensive error handling
    - Request/response logging
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # HTTP clients per service for isolation
        self.clients: Dict[str, httpx.AsyncClient] = {}

        # Circuit breakers per service
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            "kraken": CircuitBreaker(failure_threshold=5, recovery_timeout=60),
            "binance": CircuitBreaker(failure_threshold=5, recovery_timeout=60),
            "coinmarketcap": CircuitBreaker(failure_threshold=3, recovery_timeout=120),
            "coingecko": CircuitBreaker(failure_threshold=3, recovery_timeout=90),
            "reddit": CircuitBreaker(failure_threshold=2, recovery_timeout=300),
            "twitter": CircuitBreaker(failure_threshold=2, recovery_timeout=300),
            "news": CircuitBreaker(failure_threshold=3, recovery_timeout=180),
        }

        # Cache storage
        self.cache: Dict[str, CacheEntry] = {}

        # Rate limiting tracking
        self.rate_limits: Dict[str, Dict[str, Any]] = {}

        # Default cache TTLs per data type
        self.default_ttls = {
            "price": 30,  # Price data: 30 seconds
            "orderbook": 10,  # Order book: 10 seconds
            "market": 60,  # Market data: 1 minute
            "sentiment": 300,  # Sentiment: 5 minutes
            "news": 600,  # News: 10 minutes
            "social": 180,  # Social data: 3 minutes
            "metadata": 3600,  # Metadata: 1 hour
        }

    async def _get_client(self, service: str) -> httpx.AsyncClient:
        """Get or create HTTP client for service"""
        if service not in self.clients:
            # Service-specific configurations
            timeout_configs = {
                "kraken": 30.0,
                "binance": 30.0,
                "coinmarketcap": 45.0,
                "coingecko": 45.0,
                "reddit": 60.0,
                "twitter": 60.0,
                "news": 90.0,
            }

            headers = {
                "User-Agent": "CryptoSmartTrader/2.0 (Enterprise Trading Bot; +https://your-domain.com/contact)",
                "Accept": "application/json",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }

            # Service-specific headers
            if service in ["reddit", "twitter"]:
                headers.update(
                    {
                        "Accept-Language": "en-US,en;q=0.9",
                        "Cache-Control": "no-cache",
                    }
                )

            self.clients[service] = httpx.AsyncClient(
                timeout=timeout_configs.get(service, 30.0),
                headers=headers,
                limits=httpx.Limits(
                    max_keepalive_connections=5, max_connections=10, keepalive_expiry=30.0
                ),
                follow_redirects=True,
                verify=True,
            )

        return self.clients[service]

    def _generate_cache_key(
        self, service: str, endpoint: str, params: Optional[Dict] = None
    ) -> str:
        """Generate cache key for request"""
        key_parts = [service, endpoint]
        if params:
            # Sort params for consistent keys
            param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
            key_parts.append(param_str)
        return ":".join(key_parts)

    def _get_cached_response(self, cache_key: str) -> Optional[Any]:
        """Get cached response if valid"""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            if not entry.is_expired():
                entry.access()
                self.logger.debug(f"Cache hit for {cache_key}")
                return entry.data
            else:
                # Remove expired entry
                del self.cache[cache_key]
                self.logger.debug(f"Cache expired for {cache_key}")

        return None

    def _cache_response(
        self, cache_key: str, data: Any, data_type: str = "default", ttl: Optional[int] = None
    ):
        """Cache response with TTL"""
        if ttl is None:
            ttl = self.default_ttls.get(data_type, 300)

        self.cache[cache_key] = CacheEntry(data=data, timestamp=datetime.utcnow(), ttl_seconds=ttl)

        self.logger.debug(f"Cached response for {cache_key} (TTL: {ttl}s)")

        # Clean up old entries periodically
        if len(self.cache) > 1000:
            self._cleanup_cache()

    def _cleanup_cache(self):
        """Remove expired and least accessed cache entries"""
        now = datetime.utcnow()

        # Remove expired entries
        expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]

        for key in expired_keys:
            del self.cache[key]

        # If still too many entries, remove least accessed
        if len(self.cache) > 500:
            sorted_entries = sorted(
                self.cache.items(), key=lambda x: (x[1].access_count, x[1].last_accessed)

            # Remove bottom 25%
            remove_count = len(sorted_entries) // 4
            for key, _ in sorted_entries[:remove_count]:
                del self.cache[key]

        self.logger.info(f"Cache cleanup complete. Entries: {len(self.cache)}")

    def _should_use_stale_while_revalidate(self, cache_key: str) -> bool:
        """Check if should use stale-while-revalidate pattern"""
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            return entry.is_stale() and not entry.is_expired()
        return False

    async def _check_rate_limit(self, service: str, endpoint: str) -> bool:
        """Check if request is within rate limits"""
        rate_key = f"{service}:{endpoint}"

        if rate_key not in self.rate_limits:
            self.rate_limits[rate_key] = {
                "requests": [],
                "window": 60,  # 1 minute window
            }

        rate_info = self.rate_limits[rate_key]
        now = time.time()

        # Remove old requests outside window
        rate_info["requests"] = [
            req_time for req_time in rate_info["requests"] if now - req_time < rate_info["window"]
        ]

        # Service-specific rate limits (requests per minute)
        limits = {
            "kraken": 20,
            "binance": 60,
            "coinmarketcap": 30,
            "coingecko": 50,
            "reddit": 10,
            "twitter": 5,
            "news": 30,
        }

        limit = limits.get(service, 30)

        if len(rate_info["requests"]) >= limit:
            self.logger.warning(f"Rate limit exceeded for {service}:{endpoint}")
            return False

        rate_info["requests"].append(now)
        return True

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=60, jitter=2),
        reraise=True,
    )
    async def request(
        self,
        service: str,
        method: str,
        url: str,
        data_type: str = "default",
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        use_cache: bool = True,
        cache_ttl: Optional[int] = None,
        **kwargs,
    ) -> Union[Dict, List, str]:
        """
        Make HTTP request with resilience patterns

        Args:
            service: Service name for circuit breaker
            method: HTTP method
            url: Request URL
            data_type: Data type for cache TTL
            params: Query parameters
            headers: Additional headers
            json_data: JSON request body
            use_cache: Whether to use caching
            cache_ttl: Custom cache TTL
            **kwargs: Additional httpx parameters

        Returns:
            Response data

        Raises:
            httpx.HTTPError: On HTTP errors
            RuntimeError: On circuit breaker open
        """

        # Check circuit breaker
        if service in self.circuit_breakers:
            circuit = self.circuit_breakers[service]
            if not circuit.can_execute():
                raise RuntimeError(f"Circuit breaker open for {service}")

        # Check rate limits
        endpoint = url.split("/")[-1] if "/" in url else url
        if not await self._check_rate_limit(service, endpoint):
            # Wait before retry
            await asyncio.sleep(random.choice)
            if not await self._check_rate_limit(service, endpoint):
                raise RuntimeError(f"Rate limit exceeded for {service}")

        # Generate cache key
        cache_key = self._generate_cache_key(service, endpoint, params)

        # Check cache for GET requests
        if method.upper() == "GET" and use_cache:
            cached_data = self._get_cached_response(cache_key)
            if cached_data is not None:
                return cached_data

            # Stale-while-revalidate: return stale data and revalidate async
            if self._should_use_stale_while_revalidate(cache_key):
                stale_data = self.cache[cache_key].data
                # Schedule background revalidation
                asyncio.create_task(
                    self._background_revalidate(
                        service,
                        method,
                        url,
                        data_type,
                        params,
                        headers,
                        json_data,
                        cache_key,
                        cache_ttl,
                        **kwargs,
                    )
                return stale_data

        # Get client
        client = await self._get_client(service)

        # Merge headers
        request_headers = client.headers.copy()
        if headers:
            request_headers.update(headers)

        # Log request
        self.logger.info(f"HTTP {method.upper()} {service}: {url}")

        try:
            # Make request
            response = await client.request(
                method=method,
                url=url,
                params=params,
                headers=request_headers,
                json=json_data,
                **kwargs,
            )

            response.raise_for_status()

            # Record success for circuit breaker
            if service in self.circuit_breakers:
                self.circuit_breakers[service].record_success()

            # Parse response
            if response.headers.get("content-type", "").startswith("application/json"):
                data = response.json()
            else:
                data = response.text

            # Cache successful GET responses
            if method.upper() == "GET" and use_cache and response.status_code == 200:
                self._cache_response(cache_key, data, data_type, cache_ttl)

            self.logger.debug(f"HTTP {response.status_code} {service}: {url}")
            return data

        except httpx.HTTPError as e:
            # Record failure for circuit breaker
            if service in self.circuit_breakers:
                self.circuit_breakers[service].record_failure()

            self.logger.error(f"HTTP error {service}: {url} - {e}")
            raise

        except Exception as e:
            # Record failure for circuit breaker
            if service in self.circuit_breakers:
                self.circuit_breakers[service].record_failure()

            self.logger.error(f"Request error {service}: {url} - {e}")
            raise

    async def _background_revalidate(
        self,
        service: str,
        method: str,
        url: str,
        data_type: str,
        params: Optional[Dict],
        headers: Optional[Dict],
        json_data: Optional[Dict],
        cache_key: str,
        cache_ttl: Optional[int],
        **kwargs,
    ):
        """Background cache revalidation"""
        try:
            await self.request(
                service=service,
                method=method,
                url=url,
                data_type=data_type,
                params=params,
                headers=headers,
                json_data=json_data,
                use_cache=False,  # Don't use cache for revalidation
                cache_ttl=cache_ttl,
                **kwargs,
            )
            self.logger.debug(f"Background revalidation complete for {cache_key}")
        except Exception as e:
            self.logger.warning(f"Background revalidation failed for {cache_key}: {e}")

    async def get(self, service: str, url: str, **kwargs) -> Union[Dict, List, str]:
        """GET request wrapper"""
        return await self.request(service, "GET", url, **kwargs)

    async def post(self, service: str, url: str, **kwargs) -> Union[Dict, List, str]:
        """POST request wrapper"""
        return await self.request(service, "POST", url, **kwargs)

    async def close(self):
        """Close all HTTP clients"""
        for client in self.clients.values():
            await client.aclose()
        self.clients.clear()
        self.logger.info("HTTP clients closed")

    def get_circuit_breaker_status(self) -> Dict[str, Dict]:
        """Get circuit breaker status for monitoring"""
        return {
            service: {
                "state": breaker.state.value,
                "failure_count": breaker.failure_count,
                "last_failure": breaker.last_failure_time.isoformat()
                if breaker.last_failure_time
                else None,
                "successful_calls": breaker.successful_calls,
            }
            for service, breaker in self.circuit_breakers.items()
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring"""
        total_entries = len(self.cache)
        expired_entries = sum(1 for entry in self.cache.values() if entry.is_expired())

        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "memory_usage_mb": sum(len(str(entry.data)) for entry in self.cache.values())
            / (1024 * 1024),
            "hit_rate": sum(entry.access_count for entry in self.cache.values())
            / max(total_entries, 1),
        }


# Global HTTP client instance
http_client = EnterpriseHTTPClient()


# Convenience functions for backward compatibility
async def get_market_data(exchange: str, symbol: str, **kwargs) -> Dict:
    """Get market data from exchange"""
    return await http_client.get(
        service=exchange.lower(), url=f"/api/v1/ticker/{symbol}", data_type="price", **kwargs
    )


async def get_sentiment_data(source: str, query: str, **kwargs) -> Dict:
    """Get sentiment data from social media"""
    return await http_client.get(
        service=source.lower(),
        url=f"/api/search",
        params={"q": query},
        data_type="sentiment",
        **kwargs,
    )


async def get_news_data(source: str, category: str = "crypto", **kwargs) -> List:
    """Get news data"""
    return await http_client.get(
        service=source.lower(), url=f"/api/news/{category}", data_type="news", **kwargs
    )
