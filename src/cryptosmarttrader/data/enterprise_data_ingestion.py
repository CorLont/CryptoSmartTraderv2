#!/usr/bin/env python3
"""
Enterprise Data Ingestion Framework
Robuuste data-inname met retry/backoff, rate limiting, caching en monitoring
"""

import asyncio
import time
import hashlib
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from collections import defaultdict, deque
import ccxt
import ccxt.async_support as ccxt_async
import redis
from tenacity import (
    retry, 
    stop_after_attempt, 
    wait_exponential, 
    retry_if_exception_type,
    before_sleep_log
)

# SECURITY: Import secure subprocess framework
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from core.secure_subprocess import secure_subprocess, SecureSubprocessError


class DataSourceStatus(Enum):
    """Data source status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILING = "failing"
    OFFLINE = "offline"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"


class DataPriority(Enum):
    """Data priority levels"""
    CRITICAL = 1  # Real-time trading data
    HIGH = 2      # Price feeds, order books
    MEDIUM = 3    # Market indicators
    LOW = 4       # Historical analytics


@dataclass
class DataRequest:
    """Structured data request with metadata"""
    source: str
    endpoint: str
    params: Dict[str, Any]
    priority: DataPriority
    timeout: float = 30.0
    retry_attempts: int = 3
    cache_ttl: Optional[int] = None
    callback: Optional[Callable] = None
    request_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    created_at: datetime = field(default_factory=datetime.now)


@dataclass 
class DataResponse:
    """Structured data response with metadata"""
    request_id: str
    data: Any
    status: str
    latency_ms: float
    source: str
    cached: bool = False
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None


class RateLimiter:
    """Advanced rate limiter with burst support and backoff"""
    
    def __init__(self, requests_per_second: float, burst_size: int = 5):
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_update = time.time()
        self._lock = threading.Lock()
        
    def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens from rate limiter"""
        with self._lock:
            now = time.time()
            elapsed = now - self.last_update
            
            # Add tokens based on elapsed time
            self.tokens = min(
                self.burst_size, 
                self.tokens + elapsed * self.requests_per_second
            )
            self.last_update = now
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def wait_time(self, tokens: int = 1) -> float:
        """Calculate wait time for tokens"""
        with self._lock:
            if self.tokens >= tokens:
                return 0.0
            needed_tokens = tokens - self.tokens
            return needed_tokens / self.requests_per_second


class DataCache:
    """High-performance data cache with Redis backend"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()
            self.redis_available = True
        except Exception:
            self.redis_available = False
            self.local_cache = {}
            self.cache_times = {}
        
        self.logger = logging.getLogger(__name__)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached data"""
        try:
            if self.redis_available:
                data = self.redis_client.get(key)
                return json.loads(data) if data else None
            else:
                # Fallback to local cache
                if key in self.local_cache:
                    # Check expiry
                    if key in self.cache_times:
                        if time.time() - self.cache_times[key] > 300:  # 5 min default
                            del self.local_cache[key]
                            del self.cache_times[key]
                            return None
                    return self.local_cache[key]
                return None
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        """Set cached data with TTL"""
        try:
            if self.redis_available:
                return self.redis_client.setex(
                    key, ttl, json.dumps(value, default=str)
                )
            else:
                # Fallback to local cache
                self.local_cache[key] = value
                self.cache_times[key] = time.time()
                return True
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete cached data"""
        try:
            if self.redis_available:
                return bool(self.redis_client.delete(key))
            else:
                if key in self.local_cache:
                    del self.local_cache[key]
                if key in self.cache_times:
                    del self.cache_times[key]
                return True
        except Exception as e:
            self.logger.error(f"Cache delete error: {e}")
            return False


class DataSourceMonitor:
    """Monitors data source health and performance"""
    
    def __init__(self):
        self.source_stats = defaultdict(lambda: {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'avg_latency_ms': 0.0,
            'last_success': None,
            'last_failure': None,
            'consecutive_failures': 0,
            'status': DataSourceStatus.HEALTHY,
            'rate_limit_hits': 0,
            'latency_history': deque(maxlen=100)
        })
        self._lock = threading.Lock()
    
    def record_request(self, source: str, success: bool, latency_ms: float, error: str = None):
        """Record request metrics"""
        with self._lock:
            stats = self.source_stats[source]
            stats['total_requests'] += 1
            stats['latency_history'].append(latency_ms)
            
            if success:
                stats['successful_requests'] += 1
                stats['last_success'] = datetime.now()
                stats['consecutive_failures'] = 0
            else:
                stats['failed_requests'] += 1
                stats['last_failure'] = datetime.now()
                stats['consecutive_failures'] += 1
                
                if error and 'rate limit' in error.lower():
                    stats['rate_limit_hits'] += 1
            
            # Update average latency
            if stats['latency_history']:
                stats['avg_latency_ms'] = sum(stats['latency_history']) / len(stats['latency_history'])
            
            # Update status
            stats['status'] = self._calculate_status(stats)
    
    def _calculate_status(self, stats: Dict) -> DataSourceStatus:
        """Calculate data source status"""
        if stats['consecutive_failures'] >= 5:
            return DataSourceStatus.OFFLINE
        elif stats['consecutive_failures'] >= 3:
            return DataSourceStatus.FAILING
        elif stats['rate_limit_hits'] > 0 and stats['last_failure']:
            if datetime.now() - stats['last_failure'] < timedelta(minutes=5):
                return DataSourceStatus.RATE_LIMITED
        elif stats['total_requests'] > 0:
            success_rate = stats['successful_requests'] / stats['total_requests']
            if success_rate < 0.5:
                return DataSourceStatus.FAILING
            elif success_rate < 0.8:
                return DataSourceStatus.DEGRADED
        
        return DataSourceStatus.HEALTHY
    
    def get_source_status(self, source: str) -> Dict[str, Any]:
        """Get comprehensive source status"""
        with self._lock:
            return dict(self.source_stats[source])


class EnterpriseDataIngestion:
    """Enterprise-grade data ingestion with robust error handling"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.cache = DataCache(config.get('redis_url', 'redis://localhost:6379/0'))
        self.monitor = DataSourceMonitor()
        
        # Rate limiters per exchange
        self.rate_limiters = {}
        self._init_rate_limiters()
        
        # Connection pools
        self.sync_exchanges = {}
        self.async_exchanges = {}
        self._init_exchanges()
        
        # Request queues by priority
        self.request_queues = {
            DataPriority.CRITICAL: asyncio.Queue(maxsize=100),
            DataPriority.HIGH: asyncio.Queue(maxsize=200),
            DataPriority.MEDIUM: asyncio.Queue(maxsize=500),
            DataPriority.LOW: asyncio.Queue(maxsize=1000)
        }
        
        # Worker pools
        self.executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="DataIngestion")
        self.workers_active = False
        
        # Metrics
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'rate_limit_delays': 0,
            'timeouts': 0,
            'start_time': datetime.now()
        }
    
    def _init_rate_limiters(self):
        """Initialize rate limiters for each exchange"""
        exchange_limits = self.config.get('rate_limits', {
            'kraken': 1.0,      # 1 request per second
            'binance': 10.0,    # 10 requests per second
            'kucoin': 3.0,      # 3 requests per second
            'huobi': 5.0        # 5 requests per second
        })
        
        for exchange, limit in exchange_limits.items():
            self.rate_limiters[exchange] = RateLimiter(
                requests_per_second=limit,
                burst_size=int(limit * 5)  # 5 second burst
            )
    
    def _init_exchanges(self):
        """Initialize exchange connections with connection pooling"""
        exchanges_config = self.config.get('exchanges', {})
        
        for exchange_name, exchange_config in exchanges_config.items():
            try:
                # Sync exchange
                if hasattr(ccxt, exchange_name):
                    exchange_class = getattr(ccxt, exchange_name)
                    self.sync_exchanges[exchange_name] = exchange_class({
                        'apiKey': exchange_config.get('api_key', ''),
                        'secret': exchange_config.get('secret', ''),
                        'timeout': 30000,  # 30 second timeout
                        'rateLimit': 1000,  # Managed by our rate limiter
                        'enableRateLimit': False,  # We handle rate limiting
                        'sandbox': exchange_config.get('sandbox', False),
                        'options': {
                            'adjustForTimeDifference': True,
                            'recvWindow': 10000,
                        }
                    })
                
                # Async exchange
                if hasattr(ccxt_async, exchange_name):
                    async_exchange_class = getattr(ccxt_async, exchange_name)
                    self.async_exchanges[exchange_name] = async_exchange_class({
                        'apiKey': exchange_config.get('api_key', ''),
                        'secret': exchange_config.get('secret', ''),
                        'timeout': 30000,
                        'rateLimit': 1000,
                        'enableRateLimit': False,
                        'sandbox': exchange_config.get('sandbox', False),
                        'options': {
                            'adjustForTimeDifference': True,
                            'recvWindow': 10000,
                        }
                    })
                
                self.logger.info(f"Initialized {exchange_name} exchange connections")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize {exchange_name}: {e}")
    
    async def start_workers(self):
        """Start background worker tasks"""
        if self.workers_active:
            return
        
        self.workers_active = True
        self.worker_tasks = []
        
        # Start priority-based workers
        for priority in DataPriority:
            for i in range(2):  # 2 workers per priority level
                task = asyncio.create_task(self._worker(priority))
                self.worker_tasks.append(task)
        
        # Start monitoring task
        monitor_task = asyncio.create_task(self._monitoring_loop())
        self.worker_tasks.append(monitor_task)
        
        self.logger.info("Started data ingestion workers")
    
    async def stop_workers(self):
        """Stop background worker tasks"""
        if not self.workers_active:
            return
        
        self.workers_active = False
        
        # Cancel all tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Close async exchanges
        for exchange in self.async_exchanges.values():
            await exchange.close()
        
        self.logger.info("Stopped data ingestion workers")
    
    async def request_data(self, request: DataRequest) -> DataResponse:
        """Submit data request for processing"""
        self.metrics['total_requests'] += 1
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        cached_data = self.cache.get(cache_key)
        
        if cached_data and request.cache_ttl:
            self.metrics['cache_hits'] += 1
            return DataResponse(
                request_id=request.request_id,
                data=cached_data,
                status="success",
                latency_ms=0.0,
                source=request.source,
                cached=True
            )
        
        self.metrics['cache_misses'] += 1
        
        # Add to appropriate priority queue
        future = asyncio.Future()
        request.callback = lambda response: future.set_result(response)
        
        try:
            await self.request_queues[request.priority].put((request, future))
            response = await asyncio.wait_for(future, timeout=request.timeout + 10)
            return response
        except asyncio.TimeoutError:
            self.metrics['timeouts'] += 1
            return DataResponse(
                request_id=request.request_id,
                data=None,
                status="timeout",
                latency_ms=request.timeout * 1000,
                source=request.source,
                error="Request timeout"
            )
    
    async def _worker(self, priority: DataPriority):
        """Background worker for processing requests"""
        queue = self.request_queues[priority]
        
        while self.workers_active:
            try:
                # Get request from queue
                request, future = await asyncio.wait_for(queue.get(), timeout=1.0)
                
                # Process request
                response = await self._process_request(request)
                
                # Cache successful responses
                if response.status == "success" and request.cache_ttl:
                    cache_key = self._generate_cache_key(request)
                    self.cache.set(cache_key, response.data, request.cache_ttl)
                
                # Return response
                if request.callback:
                    request.callback(response)
                
                queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Worker error: {e}")
                continue
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ccxt.NetworkError, ccxt.RequestTimeout)),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING)
    )
    async def _process_request(self, request: DataRequest) -> DataResponse:
        """Process individual data request with retry logic"""
        start_time = time.time()
        
        try:
            # Rate limiting
            rate_limiter = self.rate_limiters.get(request.source)
            if rate_limiter:
                while not rate_limiter.acquire():
                    wait_time = rate_limiter.wait_time()
                    await asyncio.sleep(wait_time)
                    self.metrics['rate_limit_delays'] += 1
            
            # Get exchange
            exchange = self.async_exchanges.get(request.source)
            if not exchange:
                raise ValueError(f"Exchange {request.source} not available")
            
            # Execute request
            data = await self._execute_request(exchange, request)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Record metrics
            self.monitor.record_request(request.source, True, latency_ms)
            
            return DataResponse(
                request_id=request.request_id,
                data=data,
                status="success",
                latency_ms=latency_ms,
                source=request.source
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            
            # Record metrics
            self.monitor.record_request(request.source, False, latency_ms, error_msg)
            
            return DataResponse(
                request_id=request.request_id,
                data=None,
                status="error",
                latency_ms=latency_ms,
                source=request.source,
                error=error_msg
            )
    
    async def _execute_request(self, exchange, request: DataRequest) -> Any:
        """Execute specific request on exchange"""
        endpoint = request.endpoint
        params = request.params
        
        # Route to appropriate exchange method
        if endpoint == "fetch_ticker":
            return await exchange.fetch_ticker(params.get('symbol'))
        elif endpoint == "fetch_tickers":
            return await exchange.fetch_tickers(params.get('symbols'))
        elif endpoint == "fetch_ohlcv":
            return await exchange.fetch_ohlcv(
                params.get('symbol'),
                params.get('timeframe', '1m'),
                params.get('since'),
                params.get('limit', 500)
            )
        elif endpoint == "fetch_order_book":
            return await exchange.fetch_order_book(
                params.get('symbol'),
                params.get('limit', 20)
            )
        elif endpoint == "fetch_trades":
            return await exchange.fetch_trades(
                params.get('symbol'),
                params.get('since'),
                params.get('limit', 50)
            )
        else:
            raise ValueError(f"Unsupported endpoint: {endpoint}")
    
    def _generate_cache_key(self, request: DataRequest) -> str:
        """Generate cache key for request"""
        key_data = f"{request.source}:{request.endpoint}:{json.dumps(request.params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _monitoring_loop(self):
        """Background monitoring and health checks"""
        while self.workers_active:
            try:
                # Log system metrics every 60 seconds
                await asyncio.sleep(60)
                await self._log_system_metrics()
                
                # Health check exchanges
                await self._health_check_exchanges()
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
    
    async def _log_system_metrics(self):
        """Log comprehensive system metrics"""
        uptime = datetime.now() - self.metrics['start_time']
        
        metrics_summary = {
            'uptime_seconds': uptime.total_seconds(),
            'total_requests': self.metrics['total_requests'],
            'cache_hit_rate': self.metrics['cache_hits'] / max(1, self.metrics['cache_hits'] + self.metrics['cache_misses']),
            'rate_limit_delays': self.metrics['rate_limit_delays'],
            'timeouts': self.metrics['timeouts'],
            'queue_sizes': {
                priority.name: queue.qsize() 
                for priority, queue in self.request_queues.items()
            },
            'source_statuses': {
                source: self.monitor.get_source_status(source)['status'].value
                for source in self.sync_exchanges.keys()
            }
        }
        
        self.logger.info(f"Data ingestion metrics: {json.dumps(metrics_summary, indent=2)}")
    
    async def _health_check_exchanges(self):
        """Perform health checks on all exchanges"""
        for exchange_name, exchange in self.async_exchanges.items():
            try:
                # Simple health check - fetch server time
                await asyncio.wait_for(exchange.fetch_time(), timeout=10.0)
                self.monitor.record_request(exchange_name, True, 10.0)
            except Exception as e:
                self.monitor.record_request(exchange_name, False, 10000.0, str(e))
                self.logger.warning(f"Health check failed for {exchange_name}: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        uptime = datetime.now() - self.metrics['start_time']
        
        return {
            'status': 'healthy' if self.workers_active else 'stopped',
            'uptime_seconds': uptime.total_seconds(),
            'metrics': self.metrics.copy(),
            'queue_sizes': {
                priority.name: queue.qsize() 
                for priority, queue in self.request_queues.items()
            },
            'sources': {
                source: self.monitor.get_source_status(source)
                for source in self.sync_exchanges.keys()
            },
            'cache_available': self.cache.redis_available
        }


# Convenience functions for easy usage
async def create_data_ingestion(config: Dict[str, Any]) -> EnterpriseDataIngestion:
    """Create and start data ingestion system"""
    system = EnterpriseDataIngestion(config)
    await system.start_workers()
    return system


def create_market_data_request(
    exchange: str,
    symbol: str,
    priority: DataPriority = DataPriority.HIGH,
    cache_ttl: int = 30
) -> DataRequest:
    """Create market data request"""
    return DataRequest(
        source=exchange,
        endpoint="fetch_ticker",
        params={'symbol': symbol},
        priority=priority,
        cache_ttl=cache_ttl
    )


def create_orderbook_request(
    exchange: str,
    symbol: str,
    limit: int = 20,
    priority: DataPriority = DataPriority.CRITICAL
) -> DataRequest:
    """Create order book request"""
    return DataRequest(
        source=exchange,
        endpoint="fetch_order_book",
        params={'symbol': symbol, 'limit': limit},
        priority=priority,
        cache_ttl=5  # 5 second cache for order books
    )


def create_ohlcv_request(
    exchange: str,
    symbol: str,
    timeframe: str = "1m",
    limit: int = 500,
    priority: DataPriority = DataPriority.MEDIUM,
    cache_ttl: int = 60
) -> DataRequest:
    """Create OHLCV data request"""
    return DataRequest(
        source=exchange,
        endpoint="fetch_ohlcv",
        params={
            'symbol': symbol,
            'timeframe': timeframe,
            'limit': limit
        },
        priority=priority,
        cache_ttl=cache_ttl
    )