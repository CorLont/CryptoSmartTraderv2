#!/usr/bin/env python3
"""
Hardened HTTP Client
Enterprise-grade HTTP client met timeout, exponential backoff, circuit breaker per source
"""

import asyncio
import aiohttp
import time
import random
import logging
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import hashlib

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, blocking requests  
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class HTTPConfig:
    """HTTP client configuration"""
    base_timeout: float = 10.0
    max_timeout: float = 30.0
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter_factor: float = 0.1
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    rate_limit_per_minute: int = 60
    connection_pool_size: int = 100


@dataclass
class RequestMetrics:
    """Request metrics tracking"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    last_request_time: Optional[datetime] = None
    last_error: Optional[str] = None
    consecutive_failures: int = 0


class CircuitBreaker:
    """Circuit breaker per data source"""
    
    def __init__(self, source: str, config: HTTPConfig):
        self.source = source
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.next_attempt_time: Optional[datetime] = None
        
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
            
        if self.state == CircuitBreakerState.OPEN:
            if self.next_attempt_time and datetime.now() >= self.next_attempt_time:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info(f"Circuit breaker for {self.source} moving to HALF_OPEN")
                return True
            return False
            
        if self.state == CircuitBreakerState.HALF_OPEN:
            return True
            
        return False
    
    def record_success(self):
        """Record successful request"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info(f"Circuit breaker for {self.source} CLOSED - service recovered")
        
        self.failure_count = 0
        self.last_failure_time = None
    
    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.config.circuit_breaker_threshold:
            self.state = CircuitBreakerState.OPEN
            self.next_attempt_time = datetime.now() + timedelta(seconds=self.config.circuit_breaker_timeout)
            logger.warning(f"Circuit breaker for {self.source} OPEN - too many failures ({self.failure_count})")


class RateLimiter:
    """Rate limiter per source"""
    
    def __init__(self, source: str, limit_per_minute: int):
        self.source = source
        self.limit_per_minute = limit_per_minute
        self.request_times: List[datetime] = []
        
    async def acquire(self) -> bool:
        """Acquire rate limit token"""
        now = datetime.now()
        
        # Remove old requests (older than 1 minute)
        cutoff = now - timedelta(minutes=1)
        self.request_times = [t for t in self.request_times if t > cutoff]
        
        if len(self.request_times) >= self.limit_per_minute:
            # Calculate wait time until next slot
            oldest_request = min(self.request_times)
            wait_time = (oldest_request + timedelta(minutes=1) - now).total_seconds()
            
            if wait_time > 0:
                logger.debug(f"Rate limit reached for {self.source}, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
                return await self.acquire()
        
        self.request_times.append(now)
        return True


class HardenedHTTPClient:
    """Enterprise-grade HTTP client met alle hardening features"""
    
    def __init__(self, config: Optional[HTTPConfig] = None):
        self.config = config or HTTPConfig()
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.metrics: Dict[str, RequestMetrics] = {}
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def start(self):
        """Start HTTP client session"""
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=self.config.max_timeout)
            connector = aiohttp.TCPConnector(
                limit=self.config.connection_pool_size,
                limit_per_host=20,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'User-Agent': 'CryptoSmartTrader-Enterprise/2.0'}
            )
    
    async def close(self):
        """Close HTTP client session"""
        if self.session:
            await self.session.close()
            self.session = None
    
    def get_circuit_breaker(self, source: str) -> CircuitBreaker:
        """Get or create circuit breaker for source"""
        if source not in self.circuit_breakers:
            self.circuit_breakers[source] = CircuitBreaker(source, self.config)
        return self.circuit_breakers[source]
    
    def get_rate_limiter(self, source: str) -> RateLimiter:
        """Get or create rate limiter for source"""
        if source not in self.rate_limiters:
            self.rate_limiters[source] = RateLimiter(source, self.config.rate_limit_per_minute)
        return self.rate_limiters[source]
    
    def get_metrics(self, source: str) -> RequestMetrics:
        """Get or create metrics for source"""
        if source not in self.metrics:
            self.metrics[source] = RequestMetrics()
        return self.metrics[source]
    
    def calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter"""
        delay = min(self.config.base_delay * (2 ** attempt), self.config.max_delay)
        jitter = delay * self.config.jitter_factor * random.random()
        return delay + jitter
    
    async def request(
        self,
        method: str,
        url: str,
        source: str,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute hardened HTTP request with all protections"""
        
        # Get protection components
        circuit_breaker = self.get_circuit_breaker(source)
        rate_limiter = self.get_rate_limiter(source)
        metrics = self.get_metrics(source)
        
        # Check circuit breaker
        if not circuit_breaker.can_execute():
            raise Exception(f"Circuit breaker OPEN for {source}")
        
        # Apply rate limiting
        await rate_limiter.acquire()
        
        # Use configured timeout or default
        request_timeout = timeout or self.config.base_timeout
        
        # Retry loop with exponential backoff
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            start_time = time.time()
            
            try:
                # Ensure session is started
                if not self.session:
                    await self.start()
                
                # Execute request
                if not self.session:
                    raise Exception("HTTP session not initialized")
                
                async with self.session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json_data,
                    timeout=aiohttp.ClientTimeout(total=request_timeout),
                    **kwargs
                ) as response:
                    
                    # Calculate latency
                    latency_ms = (time.time() - start_time) * 1000
                    
                    # Update metrics
                    metrics.total_requests += 1
                    metrics.last_request_time = datetime.now()
                    
                    # Update average latency
                    if metrics.avg_latency_ms == 0:
                        metrics.avg_latency_ms = latency_ms
                    else:
                        metrics.avg_latency_ms = (metrics.avg_latency_ms * 0.9) + (latency_ms * 0.1)
                    
                    # Check response status
                    if response.status >= 400:
                        error_text = await response.text()
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"HTTP {response.status}: {error_text}"
                        )
                    
                    # Parse response
                    try:
                        result_data = await response.json()
                    except:
                        result_data = await response.text()
                    
                    # Record success
                    metrics.successful_requests += 1
                    metrics.consecutive_failures = 0
                    circuit_breaker.record_success()
                    
                    return {
                        'status': 'success',
                        'data': result_data,
                        'status_code': response.status,
                        'latency_ms': latency_ms,
                        'source': source,
                        'timestamp': time.time()
                    }
            
            except Exception as e:
                last_exception = e
                latency_ms = (time.time() - start_time) * 1000
                
                # Update metrics
                metrics.total_requests += 1
                metrics.failed_requests += 1
                metrics.consecutive_failures += 1
                metrics.last_error = str(e)
                metrics.last_request_time = datetime.now()
                
                # Record failure in circuit breaker
                circuit_breaker.record_failure()
                
                logger.warning(f"Request failed for {source} (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}")
                
                # Don't retry on client errors (4xx)
                if isinstance(e, aiohttp.ClientResponseError) and 400 <= e.status < 500:
                    break
                
                # Apply exponential backoff before retry
                if attempt < self.config.max_retries:
                    delay = self.calculate_backoff_delay(attempt)
                    logger.debug(f"Backing off {delay:.2f}s before retry")
                    await asyncio.sleep(delay)
        
        # All retries failed
        raise Exception(f"All retries failed for {source}: {last_exception}")
    
    async def get(self, url: str, source: str, **kwargs) -> Dict[str, Any]:
        """HTTP GET with hardening"""
        return await self.request('GET', url, source, **kwargs)
    
    async def post(self, url: str, source: str, **kwargs) -> Dict[str, Any]:
        """HTTP POST with hardening"""
        return await self.request('POST', url, source, **kwargs)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all sources"""
        status = {}
        
        for source in set(list(self.circuit_breakers.keys()) + list(self.metrics.keys())):
            circuit_breaker = self.circuit_breakers.get(source)
            metrics = self.metrics.get(source)
            
            source_status = {
                'circuit_breaker_state': circuit_breaker.state.value if circuit_breaker else 'unknown',
                'consecutive_failures': circuit_breaker.failure_count if circuit_breaker else 0,
                'total_requests': metrics.total_requests if metrics else 0,
                'success_rate': (
                    metrics.successful_requests / metrics.total_requests 
                    if metrics and metrics.total_requests > 0 else 0
                ),
                'avg_latency_ms': metrics.avg_latency_ms if metrics else 0,
                'last_error': metrics.last_error if metrics else None
            }
            
            status[source] = source_status
        
        return status


# Factory function voor gemakkelijk gebruik
async def create_hardened_client(config: Optional[HTTPConfig] = None) -> HardenedHTTPClient:
    """Create and start hardened HTTP client"""
    client = HardenedHTTPClient(config)
    await client.start()
    return client