#!/usr/bin/env python3
"""
Enterprise Data Ingestion System
Robust data collection met timeout/retry/exponential backoff
"""

import asyncio
import aiohttp
import time
import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import json
import hashlib
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class DataSourceStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    OFFLINE = "offline"


@dataclass
class DataRequest:
    """Enterprise data request structure"""
    source: str
    endpoint: str
    params: Dict[str, Any]
    priority: DataPriority = DataPriority.MEDIUM
    timeout: float = 30.0
    retry_attempts: int = 3
    cache_ttl: int = 60
    
    def cache_key(self) -> str:
        """Generate cache key voor request"""
        key_components = [
            self.source,
            self.endpoint,
            json.dumps(self.params, sort_keys=True)
        ]
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()


@dataclass
class DataResponse:
    """Enterprise data response structure"""
    status: str
    data: Optional[Dict[str, Any]]
    timestamp: float
    latency_ms: float
    source: str
    cached: bool = False
    error: Optional[str] = None


class RobustExchangeConnector:
    """Robust exchange connector met error handling"""
    
    def __init__(self, exchange_id: str):
        self.id = exchange_id
        self.status = DataSourceStatus.HEALTHY
        self.last_error = None
        self.error_count = 0
        self.last_success = time.time()
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.total_latency = 0.0
        
    async def ensure_connection(self) -> bool:
        """Ensure connection is healthy"""
        if self.status == DataSourceStatus.OFFLINE:
            return await self.reconnect()
        return True
    
    async def reconnect(self) -> bool:
        """Attempt to reconnect to exchange"""
        try:
            # Mock reconnection logic
            await asyncio.sleep(0.1)
            self.status = DataSourceStatus.HEALTHY
            self.error_count = 0
            logger.info(f"Reconnected to {self.id}")
            return True
        except Exception as e:
            logger.error(f"Reconnection failed for {self.id}: {e}")
            return False
    
    def classify_error(self, error: Exception) -> str:
        """Classify error type voor appropriate handling"""
        error_str = str(error).lower()
        
        if isinstance(error, asyncio.TimeoutError):
            return "timeout"
        elif isinstance(error, aiohttp.ClientTimeout):
            return "timeout"
        elif isinstance(error, aiohttp.ClientConnectorError):
            return "connection_error"
        elif isinstance(error, aiohttp.ClientResponseError):
            if error.status == 429:
                return "rate_limit"
            elif error.status >= 500:
                return "server_error"
            else:
                return "client_error"
        else:
            return "unknown_error"
    
    def record_request_metrics(self, endpoint: str, latency: float, success: bool):
        """Record request performance metrics"""
        self.request_count += 1
        self.total_latency += latency
        
        if success:
            self.success_count += 1
            self.last_success = time.time()
        else:
            self.error_count += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get connector performance metrics"""
        return {
            'total_requests': self.request_count,
            'success_rate': self.success_count / max(self.request_count, 1),
            'avg_latency_ms': (self.total_latency / max(self.request_count, 1)) * 1000,
            'error_count': self.error_count,
            'status': self.status.value,
            'last_success': self.last_success
        }


class EnterpriseDataManager:
    """Enterprise data manager met comprehensive robustness features"""
    
    def __init__(self):
        # Configuration
        self.connection_pool_size = 50
        self.max_retries = 3
        self.base_backoff_delay = 1.0
        self.max_backoff_delay = 30.0
        
        # Data sources
        self.sources: Dict[str, RobustExchangeConnector] = {}
        
        # Rate limiting
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}
        
        # Caching
        self.cache: Dict[str, Dict[str, Any]] = {}
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Request queue (priority-based)
        self.request_queue = asyncio.PriorityQueue()
        
        # Performance metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        logger.info("EnterpriseDataManager initialized")
    
    def register_source(self, source_id: str, config: Dict[str, Any]):
        """Register data source"""
        self.sources[source_id] = RobustExchangeConnector(source_id)
        
        # Initialize rate limiter
        self.rate_limiters[source_id] = {
            'max_calls_per_second': config.get('rate_limit', 10),
            'burst_allowance': config.get('burst_allowance', 20),
            'last_reset': time.time(),
            'current_calls': 0
        }
        
        # Initialize circuit breaker
        self.circuit_breakers[source_id] = {
            'failure_threshold': 5,
            'success_threshold': 3,
            'timeout_seconds': 60,
            'failure_count': 0,
            'last_failure': 0,
            'state': 'closed'  # closed, open, half_open
        }
        
        logger.info(f"Registered data source: {source_id}")
    
    async def fetch_data(self, request: DataRequest) -> DataResponse:
        """Fetch data met comprehensive robustness"""
        start_time = time.time()
        self.total_requests += 1
        
        try:
            # Check cache first
            cached_response = self._check_cache(request)
            if cached_response:
                return cached_response
            
            # Check circuit breaker
            if self._is_circuit_open(request.source):
                return DataResponse(
                    status="circuit_open",
                    data=None,
                    timestamp=time.time(),
                    latency_ms=0,
                    source=request.source,
                    error="Circuit breaker open"
                )
            
            # Check rate limits
            if not self._check_rate_limit(request.source):
                return DataResponse(
                    status="rate_limited",
                    data=None,
                    timestamp=time.time(),
                    latency_ms=0,
                    source=request.source,
                    error="Rate limit exceeded"
                )
            
            # Execute request met retry logic
            response = await self._execute_with_retry(request)
            
            # Cache successful response
            if response.status == "success":
                self._cache_response(request, response)
                self.successful_requests += 1
                self._record_circuit_success(request.source)
            else:
                self.failed_requests += 1
                self._record_circuit_failure(request.source)
            
            return response
            
        except Exception as e:
            self.failed_requests += 1
            self._record_circuit_failure(request.source)
            
            return DataResponse(
                status="error",
                data=None,
                timestamp=time.time(),
                latency_ms=(time.time() - start_time) * 1000,
                source=request.source,
                error=str(e)
            )
    
    async def _execute_with_retry(self, request: DataRequest) -> DataResponse:
        """Execute request met exponential backoff retry"""
        last_error = None
        
        for attempt in range(request.retry_attempts + 1):
            try:
                # Calculate backoff delay
                if attempt > 0:
                    delay = min(
                        self.base_backoff_delay * (2 ** (attempt - 1)),
                        self.max_backoff_delay
                    )
                    # Add jitter
                    delay *= (0.5 + 0.5 * (time.time() % 1))
                    await asyncio.sleep(delay)
                
                # Execute actual request
                response = await self._execute_request(request)
                return response
                
            except asyncio.TimeoutError:
                last_error = "Request timeout"
                logger.warning(f"Timeout on attempt {attempt + 1} for {request.source}/{request.endpoint}")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Error on attempt {attempt + 1} for {request.source}/{request.endpoint}: {e}")
        
        # All retries failed
        return DataResponse(
            status="timeout" if "timeout" in last_error.lower() else "error",
            data=None,
            timestamp=time.time(),
            latency_ms=0,
            source=request.source,
            error=last_error
        )
    
    async def _execute_request(self, request: DataRequest) -> DataResponse:
        """Execute actual API request"""
        start_time = time.time()
        
        try:
            # Simulate API call
            await asyncio.wait_for(
                self._mock_api_call(request),
                timeout=request.timeout
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Mock successful response
            response_data = {
                "symbol": request.params.get("symbol", "UNKNOWN"),
                "price": 50000.0 + (time.time() % 1000),
                "volume": 1000000.0,
                "timestamp": time.time()
            }
            
            return DataResponse(
                status="success",
                data=response_data,
                timestamp=time.time(),
                latency_ms=latency_ms,
                source=request.source
            )
            
        except asyncio.TimeoutError:
            raise asyncio.TimeoutError(f"Request timeout after {request.timeout}s")
    
    async def _mock_api_call(self, request: DataRequest):
        """Mock API call voor testing"""
        # Simulate variable latency
        delay = 0.05 + (time.time() % 0.1)  # 50-150ms
        await asyncio.sleep(delay)
    
    def _check_cache(self, request: DataRequest) -> Optional[DataResponse]:
        """Check if cached response is available"""
        cache_key = request.cache_key()
        
        if cache_key in self.cache:
            cached_item = self.cache[cache_key]
            
            # Check if cache is still valid
            if time.time() - cached_item['timestamp'] < request.cache_ttl:
                cached_response = cached_item['response']
                cached_response.cached = True
                return cached_response
            else:
                # Remove expired cache
                del self.cache[cache_key]
        
        return None
    
    def _cache_response(self, request: DataRequest, response: DataResponse):
        """Cache successful response"""
        cache_key = request.cache_key()
        
        self.cache[cache_key] = {
            'timestamp': time.time(),
            'response': response
        }
        
        # Limit cache size
        if len(self.cache) > 10000:
            # Remove oldest 20% of cache entries
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1]['timestamp']
            )
            
            to_remove = int(len(sorted_items) * 0.2)
            for cache_key, _ in sorted_items[:to_remove]:
                del self.cache[cache_key]
    
    def _check_rate_limit(self, source: str) -> bool:
        """Check if request is within rate limits"""
        if source not in self.rate_limiters:
            return True
        
        limiter = self.rate_limiters[source]
        current_time = time.time()
        
        # Reset counter if window passed
        if current_time - limiter['last_reset'] >= 1.0:
            limiter['current_calls'] = 0
            limiter['last_reset'] = current_time
        
        # Check if within limits
        if limiter['current_calls'] < limiter['max_calls_per_second']:
            limiter['current_calls'] += 1
            return True
        
        return False
    
    def _is_circuit_open(self, source: str) -> bool:
        """Check if circuit breaker is open"""
        if source not in self.circuit_breakers:
            return False
        
        breaker = self.circuit_breakers[source]
        current_time = time.time()
        
        if breaker['state'] == 'open':
            # Check if timeout period has passed
            if current_time - breaker['last_failure'] > breaker['timeout_seconds']:
                breaker['state'] = 'half_open'
                return False
            return True
        
        return False
    
    def _record_circuit_failure(self, source: str):
        """Record circuit breaker failure"""
        if source not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[source]
        breaker['failure_count'] += 1
        breaker['last_failure'] = time.time()
        
        # Open circuit if threshold reached
        if breaker['failure_count'] >= breaker['failure_threshold']:
            breaker['state'] = 'open'
            logger.warning(f"Circuit breaker opened for {source}")
    
    def _record_circuit_success(self, source: str):
        """Record circuit breaker success"""
        if source not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[source]
        
        if breaker['state'] == 'half_open':
            breaker['failure_count'] = 0
            breaker['state'] = 'closed'
            logger.info(f"Circuit breaker closed for {source}")
    
    def validate_data_quality(self, data: Dict[str, Any]) -> float:
        """Validate data quality and return score (0-1)"""
        score = 0.0
        total_checks = 6
        
        # Check 1: Completeness
        required_fields = ['symbol', 'price', 'timestamp']
        present_fields = sum(1 for field in required_fields if field in data and data[field] is not None)
        score += present_fields / len(required_fields)
        
        # Check 2: Data freshness
        if 'timestamp' in data:
            age_seconds = time.time() - data['timestamp']
            if age_seconds < 60:  # Less than 1 minute old
                score += 1.0
            elif age_seconds < 300:  # Less than 5 minutes old
                score += 0.5
        
        # Check 3: Price validity
        if 'price' in data and isinstance(data['price'], (int, float)) and data['price'] > 0:
            score += 1.0
        
        # Check 4: Volume validity
        if 'volume' in data and isinstance(data['volume'], (int, float)) and data['volume'] >= 0:
            score += 1.0
        
        # Check 5: Symbol format
        if 'symbol' in data and isinstance(data['symbol'], str) and len(data['symbol']) > 0:
            score += 1.0
        
        # Check 6: No null values in critical fields
        critical_nulls = sum(1 for field in ['price', 'symbol'] if data.get(field) is None)
        if critical_nulls == 0:
            score += 1.0
        
        return score / total_checks
    
    def get_source_status(self, source: str) -> DataSourceStatus:
        """Get current status of data source"""
        if source not in self.sources:
            return DataSourceStatus.OFFLINE
        
        return self.sources[source].status
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status"""
        source_statuses = {}
        
        for source_id, connector in self.sources.items():
            metrics = connector.get_performance_metrics()
            source_statuses[source_id] = metrics
        
        overall_health = {
            'total_requests': self.total_requests,
            'success_rate': self.successful_requests / max(self.total_requests, 1),
            'failed_requests': self.failed_requests,
            'cache_size': len(self.cache),
            'registered_sources': len(self.sources),
            'source_details': source_statuses
        }
        
        return overall_health


# Export main classes
__all__ = [
    'EnterpriseDataManager', 
    'DataRequest', 
    'DataResponse', 
    'DataPriority', 
    'DataSourceStatus',
    'RobustExchangeConnector'
]