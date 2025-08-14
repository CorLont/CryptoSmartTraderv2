#!/usr/bin/env python3
"""
Centralized Throttling & Rate Limiting Infrastructure
Enterprise-grade throttling voor alle scrapers, LLM-agents, en external API calls
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock, RLock
from typing import Any, Callable, Dict, List, Optional, Union
import json
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger(__name__)


class ServiceType(Enum):
    """Service categorization voor verschillende throttling policies"""
    LLM_API = "llm_api"           # OpenAI, Anthropic, etc.
    SOCIAL_MEDIA = "social_media"  # Reddit, Twitter, Telegram, Discord
    EXCHANGE_API = "exchange_api"  # Kraken, Binance, etc.
    WEB_SCRAPER = "web_scraper"   # General web scraping
    INTERNAL_API = "internal_api"  # Internal service calls


@dataclass
class ThrottleConfig:
    """Configuration voor rate limiting per service type"""
    requests_per_second: float = 1.0
    requests_per_minute: int = 60
    requests_per_hour: int = 3600
    burst_capacity: int = 5
    backoff_base: float = 1.0
    backoff_max: float = 300.0
    backoff_exponential: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    retry_attempts: int = 3
    jitter_enabled: bool = True


# Enterprise throttling configurations per service type
ENTERPRISE_THROTTLE_CONFIGS = {
    ServiceType.LLM_API: ThrottleConfig(
        requests_per_second=0.5,      # Conservative voor LLM APIs
        requests_per_minute=20,
        requests_per_hour=1000,
        burst_capacity=3,
        backoff_base=2.0,
        backoff_max=120.0,
        circuit_breaker_threshold=3,
        circuit_breaker_timeout=300.0,
        retry_attempts=2
    ),
    ServiceType.SOCIAL_MEDIA: ThrottleConfig(
        requests_per_second=0.2,      # Conservative voor TOS compliance
        requests_per_minute=10,
        requests_per_hour=600,
        burst_capacity=2,
        backoff_base=5.0,
        backoff_max=900.0,
        circuit_breaker_threshold=2,
        circuit_breaker_timeout=600.0,
        retry_attempts=1
    ),
    ServiceType.EXCHANGE_API: ThrottleConfig(
        requests_per_second=2.0,      # Higher voor trading data
        requests_per_minute=100,
        requests_per_hour=5000,
        burst_capacity=10,
        backoff_base=1.0,
        backoff_max=60.0,
        circuit_breaker_threshold=5,
        circuit_breaker_timeout=120.0,
        retry_attempts=3
    ),
    ServiceType.WEB_SCRAPER: ThrottleConfig(
        requests_per_second=0.5,
        requests_per_minute=25,
        requests_per_hour=1500,
        burst_capacity=3,
        backoff_base=2.0,
        backoff_max=180.0,
        circuit_breaker_threshold=3,
        circuit_breaker_timeout=240.0,
        retry_attempts=2
    ),
    ServiceType.INTERNAL_API: ThrottleConfig(
        requests_per_second=10.0,     # Higher voor internal services
        requests_per_minute=500,
        requests_per_hour=10000,
        burst_capacity=20,
        backoff_base=0.5,
        backoff_max=30.0,
        circuit_breaker_threshold=10,
        circuit_breaker_timeout=60.0,
        retry_attempts=3
    )
}


@dataclass
class RequestMetrics:
    """Metrics tracking voor requests"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    throttled_requests: int = 0
    circuit_breaker_trips: int = 0
    last_request_time: float = 0.0
    last_success_time: float = 0.0
    last_failure_time: float = 0.0
    consecutive_failures: int = 0
    average_response_time: float = 0.0
    
    def update_success(self, response_time: float) -> None:
        """Update metrics op successful request"""
        self.total_requests += 1
        self.successful_requests += 1
        self.consecutive_failures = 0
        self.last_request_time = time.time()
        self.last_success_time = time.time()
        
        # Update average response time
        if self.average_response_time == 0.0:
            self.average_response_time = response_time
        else:
            self.average_response_time = (self.average_response_time * 0.9) + (response_time * 0.1)
    
    def update_failure(self) -> None:
        """Update metrics op failed request"""
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.last_request_time = time.time()
        self.last_failure_time = time.time()
    
    def update_throttled(self) -> None:
        """Update metrics op throttled request"""
        self.throttled_requests += 1
    
    def trip_circuit_breaker(self) -> None:
        """Trip circuit breaker"""
        self.circuit_breaker_trips += 1


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"     # Normal operation
    OPEN = "open"         # Circuit tripped, rejecting requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker implementation"""
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    next_attempt_time: float = 0.0
    threshold: int = 5
    timeout: float = 60.0
    
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        current_time = time.time()
        
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if current_time >= self.next_attempt_time:
                self.state = CircuitBreakerState.HALF_OPEN
                return True
            return False
        elif self.state == CircuitBreakerState.HALF_OPEN:
            return True
        
        return False
    
    def record_success(self) -> None:
        """Record successful request"""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def record_failure(self) -> None:
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.threshold:
            self.state = CircuitBreakerState.OPEN
            self.next_attempt_time = time.time() + self.timeout


class TokenBucket:
    """Token bucket implementation voor rate limiting"""
    
    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()
        self._lock = Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from bucket"""
        with self._lock:
            current_time = time.time()
            
            # Add tokens based on time elapsed
            time_passed = current_time - self.last_refill
            self.tokens = min(
                self.capacity, 
                self.tokens + (time_passed * self.refill_rate)
            )
            self.last_refill = current_time
            
            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def tokens_available(self) -> int:
        """Get current token count"""
        with self._lock:
            current_time = time.time()
            time_passed = current_time - self.last_refill
            self.tokens = min(
                self.capacity,
                self.tokens + (time_passed * self.refill_rate)
            )
            return int(self.tokens)


class CentralizedThrottleManager:
    """
    Centralized throttling manager voor ALL external API calls
    KRITIEK: Single point of control voor rate limiting
    """
    
    _instance: Optional['CentralizedThrottleManager'] = None
    _lock: RLock = RLock()
    
    def __new__(cls) -> 'CentralizedThrottleManager':
        """Singleton pattern enforcement"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize throttle manager"""
        if getattr(self, '_initialized', False):
            return
            
        self._service_buckets: Dict[str, TokenBucket] = {}
        self._service_configs: Dict[str, ThrottleConfig] = {}
        self._service_metrics: Dict[str, RequestMetrics] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._service_locks: Dict[str, Lock] = {}
        self._global_lock = RLock()
        
        # Initialize default configurations
        for service_type, config in ENTERPRISE_THROTTLE_CONFIGS.items():
            self._register_service(service_type.value, config)
        
        self._initialized = True
        logger.info("🚦 CentralizedThrottleManager initialized with enterprise configs")
    
    def _register_service(self, service_name: str, config: ThrottleConfig) -> None:
        """Register a service with throttling configuration"""
        with self._global_lock:
            self._service_configs[service_name] = config
            self._service_buckets[service_name] = TokenBucket(
                capacity=config.burst_capacity,
                refill_rate=config.requests_per_second
            )
            self._service_metrics[service_name] = RequestMetrics()
            self._circuit_breakers[service_name] = CircuitBreaker(
                threshold=config.circuit_breaker_threshold,
                timeout=config.circuit_breaker_timeout
            )
            self._service_locks[service_name] = Lock()
    
    def register_custom_service(self, service_name: str, config: ThrottleConfig) -> None:
        """Register custom service configuration"""
        logger.info(f"🚦 Registering custom service: {service_name}")
        self._register_service(service_name, config)
    
    def _get_service_key(self, service_type: ServiceType, endpoint: Optional[str] = None) -> str:
        """Generate service key voor tracking"""
        base_key = service_type.value
        if endpoint:
            # Create hash van endpoint voor privacy
            endpoint_hash = hashlib.md5(endpoint.encode()).hexdigest()[:8]
            return f"{base_key}_{endpoint_hash}"
        return base_key
    
    async def throttled_request(
        self,
        service_type: ServiceType,
        request_func: Callable,
        *args,
        endpoint: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Execute throttled request with automatic backoff and circuit breaking
        
        Args:
            service_type: Type of service (LLM_API, SOCIAL_MEDIA, etc.)
            request_func: Function to execute (can be sync or async)
            endpoint: Optional endpoint identifier voor granular tracking
            *args, **kwargs: Arguments for request_func
            
        Returns:
            Result from request_func
            
        Raises:
            Exception: If all retry attempts fail or circuit breaker is open
        """
        service_key = self._get_service_key(service_type, endpoint)
        
        # Ensure service is registered
        if service_key not in self._service_configs:
            self._register_service(service_key, ENTERPRISE_THROTTLE_CONFIGS[service_type])
        
        config = self._service_configs[service_key]
        bucket = self._service_buckets[service_key]
        metrics = self._service_metrics[service_key]
        circuit_breaker = self._circuit_breakers[service_key]
        
        for attempt in range(config.retry_attempts + 1):
            try:
                # Check circuit breaker
                if not circuit_breaker.can_execute():
                    logger.warning(f"🚦 Circuit breaker OPEN voor {service_key}")
                    metrics.update_throttled()
                    raise Exception(f"Circuit breaker open voor {service_key}")
                
                # Wait for token availability
                while not bucket.consume():
                    wait_time = 1.0 / config.requests_per_second
                    if config.jitter_enabled:
                        import random
                        wait_time *= (0.5 + random.random())
                    
                    logger.debug(f"🚦 Rate limiting {service_key}, waiting {wait_time:.2f}s")
                    metrics.update_throttled()
                    await asyncio.sleep(wait_time)
                
                # Execute request
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(request_func):
                    result = await request_func(*args, **kwargs)
                else:
                    result = request_func(*args, **kwargs)
                
                response_time = time.time() - start_time
                
                # Record success
                metrics.update_success(response_time)
                circuit_breaker.record_success()
                
                logger.debug(f"🚦 Request successful: {service_key} ({response_time:.3f}s)")
                return result
                
            except Exception as e:
                # Record failure
                metrics.update_failure()
                circuit_breaker.record_failure()
                
                if attempt == config.retry_attempts:
                    logger.error(f"🚦 All retry attempts failed voor {service_key}: {e}")
                    raise
                
                # Calculate backoff time
                if config.backoff_exponential:
                    backoff_time = min(
                        config.backoff_max,
                        config.backoff_base * (2 ** attempt)
                    )
                else:
                    backoff_time = config.backoff_base
                
                if config.jitter_enabled:
                    import random
                    backoff_time *= (0.5 + random.random())
                
                logger.warning(
                    f"🚦 Request failed voor {service_key} (attempt {attempt + 1}), "
                    f"retrying in {backoff_time:.2f}s: {e}"
                )
                
                await asyncio.sleep(backoff_time)
    
    def get_service_metrics(self, service_type: ServiceType, endpoint: Optional[str] = None) -> RequestMetrics:
        """Get metrics voor specific service"""
        service_key = self._get_service_key(service_type, endpoint)
        
        if service_key not in self._service_metrics:
            return RequestMetrics()
        
        return self._service_metrics[service_key]
    
    def get_all_metrics(self) -> Dict[str, RequestMetrics]:
        """Get all service metrics"""
        return self._service_metrics.copy()
    
    def reset_service_metrics(self, service_type: ServiceType, endpoint: Optional[str] = None) -> None:
        """Reset metrics voor specific service"""
        service_key = self._get_service_key(service_type, endpoint)
        
        if service_key in self._service_metrics:
            self._service_metrics[service_key] = RequestMetrics()
            logger.info(f"🚦 Metrics reset voor {service_key}")
    
    def reset_circuit_breaker(self, service_type: ServiceType, endpoint: Optional[str] = None) -> None:
        """Manually reset circuit breaker"""
        service_key = self._get_service_key(service_type, endpoint)
        
        if service_key in self._circuit_breakers:
            self._circuit_breakers[service_key] = CircuitBreaker(
                threshold=self._service_configs[service_key].circuit_breaker_threshold,
                timeout=self._service_configs[service_key].circuit_breaker_timeout
            )
            logger.info(f"🚦 Circuit breaker reset voor {service_key}")
    
    def get_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive status report"""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_services": len(self._service_configs),
            "services": {}
        }
        
        for service_name, metrics in self._service_metrics.items():
            circuit_breaker = self._circuit_breakers[service_name]
            bucket = self._service_buckets[service_name]
            
            report["services"][service_name] = {
                "metrics": {
                    "total_requests": metrics.total_requests,
                    "successful_requests": metrics.successful_requests,
                    "failed_requests": metrics.failed_requests,
                    "throttled_requests": metrics.throttled_requests,
                    "success_rate": (
                        metrics.successful_requests / max(1, metrics.total_requests) * 100
                    ),
                    "consecutive_failures": metrics.consecutive_failures,
                    "average_response_time": metrics.average_response_time,
                    "circuit_breaker_trips": metrics.circuit_breaker_trips
                },
                "circuit_breaker": {
                    "state": circuit_breaker.state.value,
                    "failure_count": circuit_breaker.failure_count,
                    "can_execute": circuit_breaker.can_execute()
                },
                "rate_limit": {
                    "tokens_available": bucket.tokens_available(),
                    "capacity": bucket.capacity,
                    "refill_rate": bucket.refill_rate
                }
            }
        
        return report


# Global singleton instance
throttle_manager = CentralizedThrottleManager()


def throttled(service_type: ServiceType, endpoint: Optional[str] = None):
    """
    Decorator voor automatic throttling van functions
    
    Usage:
        @throttled(ServiceType.LLM_API)
        async def call_openai():
            return openai.chat.completions.create(...)
        
        @throttled(ServiceType.SOCIAL_MEDIA, "reddit_api")
        def get_reddit_posts():
            return reddit.subreddit("cryptocurrency").hot()
    """
    def decorator(func: Callable) -> Callable:
        async def async_wrapper(*args, **kwargs):
            return await throttle_manager.throttled_request(
                service_type, func, *args, endpoint=endpoint, **kwargs
            )
        
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(
                throttle_manager.throttled_request(
                    service_type, func, *args, endpoint=endpoint, **kwargs
                )
            )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Export public interface
__all__ = [
    'CentralizedThrottleManager',
    'ServiceType',
    'ThrottleConfig',
    'RequestMetrics',
    'throttled',
    'throttle_manager',
    'ENTERPRISE_THROTTLE_CONFIGS'
]