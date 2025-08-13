#!/usr/bin/env python3
"""
Async HTTP Client - Enterprise Scraping Framework
High-performance async client with rate limiting, retries, and proxy support
"""

import asyncio
import aiohttp
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

from tenacity import (
    retry, stop_after_attempt, wait_exponential,
    retry_if_exception_type, before_sleep_log
)

# Import structured logger
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ..core.structured_logger import get_logger

@dataclass
class ProxyConfig:
    """Proxy configuration"""
    enabled: bool = False
    proxies: List[str] = field(default_factory=list)
    rotation_interval: int = 300  # 5 minutes
    current_proxy_index: int = 0

@dataclass
class RateLimit:
    """Rate limiting configuration per source"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_limit: int = 10

    # Internal tracking
    minute_requests: List[float] = field(default_factory=list)
    hour_requests: List[float] = field(default_factory=list)
    last_request_time: float = 0

@dataclass
class SourceMetrics:
    """Metrics tracking per source"""
    source_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    last_success_timestamp: Optional[datetime] = None
    last_error_timestamp: Optional[datetime] = None
    last_error_message: str = ""
    completeness_percentage: float = 0.0
    average_response_time: float = 0.0
    response_times: List[float] = field(default_factory=list)

class AsyncScrapeClient:
    """Enterprise async HTTP client with comprehensive rate limiting and monitoring"""

    def __init__(self,
                 max_connections: int = 100,
                 max_connections_per_host: int = 10,
                 timeout: int = 30,
                 enable_metrics: bool = True):

        self.logger = get_logger("ScrapingClient")
        self.max_connections = max_connections
        self.max_connections_per_host = max_connections_per_host
        self.timeout = timeout
        self.enable_metrics = enable_metrics

        # Rate limiting per source
        self.rate_limits: Dict[str, RateLimit] = {}
        self.source_metrics: Dict[str, SourceMetrics] = {}

        # Proxy configuration
        self.proxy_config = ProxyConfig()

        # Connection semaphores
        self.connection_semaphore = asyncio.Semaphore(max_connections)
        self.host_semaphores: Dict[str, asyncio.Semaphore] = {}

        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None

        # User agents rotation
        self.user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:121.0) Gecko/20100101 Firefox/121.0"
        ]

    async def __aenter__(self):
        """Async context manager entry"""
        await self._create_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._close_session()

    async def _create_session(self):
        """Create aiohttp session with optimized settings"""

        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_connections_per_host,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )

        timeout = aiohttp.ClientTimeout(total=self.timeout)

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": random.choice}
        )

        self.logger.info("HTTP session created",
                        max_connections=self.max_connections,
                        timeout=self.timeout)

    async def _close_session(self):
        """Close HTTP session"""
        if self.session:
            await self.session.close()
            self.logger.info("HTTP session closed")

    def configure_rate_limit(self, source: str,
                           requests_per_minute: int = 60,
                           requests_per_hour: int = 1000,
                           burst_limit: int = 10):
        """Configure rate limiting for a specific source"""

        self.rate_limits[source] = RateLimit(
            requests_per_minute=requests_per_minute,
            requests_per_hour=requests_per_hour,
            burst_limit=burst_limit
        )

        if source not in self.source_metrics:
            self.source_metrics[source] = SourceMetrics(source_name=source)

        self.logger.info(f"Rate limit configured for {source}",
                        rpm=requests_per_minute,
                        rph=requests_per_hour,
                        burst=burst_limit)

    def configure_proxy(self, proxies: List[str], rotation_interval: int = 300):
        """Configure rotating proxy support"""

        self.proxy_config = ProxyConfig(
            enabled=bool(proxies),
            proxies=proxies,
            rotation_interval=rotation_interval
        )

        self.logger.info(f"Proxy configuration updated",
                        proxy_count=len(proxies),
                        rotation_interval=rotation_interval)

    async def _check_rate_limit(self, source: str) -> bool:
        """Check if request is within rate limits"""

        if source not in self.rate_limits:
            return True

        rate_limit = self.rate_limits[source]
        current_time = time.time()

        # Clean old requests (older than 1 hour)
        rate_limit.hour_requests = [
            req_time for req_time in rate_limit.hour_requests
            if current_time - req_time < 3600
        ]

        # Clean old requests (older than 1 minute)
        rate_limit.minute_requests = [
            req_time for req_time in rate_limit.minute_requests
            if current_time - req_time < 60
        ]

        # Check hourly limit
        if len(rate_limit.hour_requests) >= rate_limit.requests_per_hour:
            return False

        # Check minute limit
        if len(rate_limit.minute_requests) >= rate_limit.requests_per_minute:
            return False

        # Check burst limit (requests in last 10 seconds)
        recent_requests = [
            req_time for req_time in rate_limit.minute_requests
            if current_time - req_time < 10
        ]

        if len(recent_requests) >= rate_limit.burst_limit:
            return False

        return True

    async def _wait_for_rate_limit(self, source: str):
        """Wait until rate limit allows next request"""

        max_wait_time = 60  # Maximum 1 minute wait
        wait_time = 1

        while not await self._check_rate_limit(source) and wait_time <= max_wait_time:
            await asyncio.sleep(wait_time)
            wait_time = min(wait_time * 1.1, max_wait_time)  # Exponential backoff

        if wait_time > max_wait_time:
            raise Exception(f"Rate limit exceeded for {source}, max wait time reached")

    def _record_request(self, source: str):
        """Record request for rate limiting"""

        if source in self.rate_limits:
            current_time = time.time()
            rate_limit = self.rate_limits[source]

            rate_limit.minute_requests.append(current_time)
            rate_limit.hour_requests.append(current_time)
            rate_limit.last_request_time = current_time

    def _get_current_proxy(self) -> Optional[str]:
        """Get current proxy for rotation"""

        if not self.proxy_config.enabled or not self.proxy_config.proxies:
            return None

        # Rotate proxy based on time interval
        current_time = time.time()
        rotation_index = int(current_time // self.proxy_config.rotation_interval) % len(self.proxy_config.proxies)

        return self.proxy_config.proxies[rotation_index]

    def _update_metrics(self, source: str, success: bool, response_time: float, error_msg: str = ""):
        """Update source metrics"""

        if not self.enable_metrics:
            return

        if source not in self.source_metrics:
            self.source_metrics[source] = SourceMetrics(source_name=source)

        metrics = self.source_metrics[source]
        metrics.total_requests += 1

        if success:
            metrics.successful_requests += 1
            metrics.last_success_timestamp = datetime.now()
        else:
            metrics.failed_requests += 1
            metrics.last_error_timestamp = datetime.now()
            metrics.last_error_message = error_msg

        # Update response time tracking
        metrics.response_times.append(response_time)
        # Keep only last 100 response times
        if len(metrics.response_times) > 100:
            metrics.response_times = metrics.response_times[-100:]

        metrics.average_response_time = sum(metrics.response_times) / len(metrics.response_times)

        # Calculate completeness percentage
        if metrics.total_requests > 0:
            metrics.completeness_percentage = (metrics.successful_requests / metrics.total_requests) * 100

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        before_sleep=before_sleep_log(logging.getLogger(), logging.WARNING)
    )
    async def fetch_json(self,
                        url: str,
                        source: str = "unknown",
                        headers: Optional[Dict[str, str]] = None,
                        params: Optional[Dict[str, Any]] = None,
                        timeout: Optional[int] = None) -> Dict[str, Any]:
        """Fetch JSON data with retries and rate limiting"""

        start_time = time.time()

        try:
            # Wait for rate limit
            await self._wait_for_rate_limit(source)

            # Record request
            self._record_request(source)

            # Prepare headers
            request_headers = {"User-Agent": random.choice}
            if headers:
                request_headers.update(headers)

            # Get proxy
            proxy = self._get_current_proxy()

            # Use connection semaphore
            async with self.connection_semaphore:
                # Get host semaphore
                host = url.split('/')[2]
                if host not in self.host_semaphores:
                    self.host_semaphores[host] = asyncio.Semaphore(self.max_connections_per_host)

                async with self.host_semaphores[host]:
                    # Add jitter to avoid thundering herd
                    await asyncio.sleep(random.choice)

                    # Make request
                    async with self.session.get(
                        url,
                        headers=request_headers,
                        params=params,
                        proxy=proxy,
                        timeout=aiohttp.ClientTimeout(total=timeout or self.timeout)
                    ) as response:

                        response_time = time.time() - start_time

                        if response.status == 200:
                            data = await response.json()
                            self._update_metrics(source, True, response_time)

                            self.logger.debug(f"JSON fetched successfully from {source}",
                                            url=url,
                                            response_time=response_time,
                                            status=response.status)

                            return data
                        else:
                            error_msg = f"HTTP {response.status}"
                            self._update_metrics(source, False, response_time, error_msg)
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status
                            )

        except Exception as e:
            response_time = time.time() - start_time
            error_msg = str(e)
            self._update_metrics(source, False, response_time, error_msg)

            self.logger.error(f"Failed to fetch JSON from {source}",
                            url=url,
                            error=error_msg,
                            response_time=response_time)
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        before_sleep=before_sleep_log(logging.getLogger(), logging.WARNING)
    )
    async def fetch_html(self,
                        url: str,
                        source: str = "unknown",
                        headers: Optional[Dict[str, str]] = None,
                        timeout: Optional[int] = None) -> str:
        """Fetch HTML content with retries and rate limiting"""

        start_time = time.time()

        try:
            # Wait for rate limit
            await self._wait_for_rate_limit(source)

            # Record request
            self._record_request(source)

            # Prepare headers
            request_headers = {"User-Agent": random.choice}
            if headers:
                request_headers.update(headers)

            # Get proxy
            proxy = self._get_current_proxy()

            # Use connection semaphore
            async with self.connection_semaphore:
                # Get host semaphore
                host = url.split('/')[2]
                if host not in self.host_semaphores:
                    self.host_semaphores[host] = asyncio.Semaphore(self.max_connections_per_host)

                async with self.host_semaphores[host]:
                    # Add jitter
                    await asyncio.sleep(random.choice)

                    # Make request
                    async with self.session.get(
                        url,
                        headers=request_headers,
                        proxy=proxy,
                        timeout=aiohttp.ClientTimeout(total=timeout or self.timeout)
                    ) as response:

                        response_time = time.time() - start_time

                        if response.status == 200:
                            content = await response.text()
                            self._update_metrics(source, True, response_time)

                            self.logger.debug(f"HTML fetched successfully from {source}",
                                            url=url,
                                            response_time=response_time,
                                            content_length=len(content))

                            return content
                        else:
                            error_msg = f"HTTP {response.status}"
                            self._update_metrics(source, False, response_time, error_msg)
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status
                            )

        except Exception as e:
            response_time = time.time() - start_time
            error_msg = str(e)
            self._update_metrics(source, False, response_time, error_msg)

            self.logger.error(f"Failed to fetch HTML from {source}",
                            url=url,
                            error=error_msg,
                            response_time=response_time)
            raise

    async def batch_fetch_json(self,
                              requests: List[Dict[str, Any]],
                              max_concurrent: int = 10) -> List[Optional[Dict[str, Any]]]:
        """Batch fetch multiple JSON endpoints concurrently"""

        semaphore = asyncio.Semaphore(max_concurrent)

        async def fetch_single(request_config):
            async with semaphore:
                try:
                    return await self.fetch_json(**request_config)
                except Exception as e:
                    self.logger.error(f"Batch fetch failed for {request_config.get('url', 'unknown')}: {e}")
                    return None

        self.logger.info(f"Starting batch fetch of {len(requests)} URLs",
                        max_concurrent=max_concurrent)

        start_time = time.time()
        results = await asyncio.gather(*[fetch_single(req) for req in requests], return_exceptions=False)
        execution_time = time.time() - start_time

        successful_requests = len([r for r in results if r is not None])

        self.logger.info(f"Batch fetch completed",
                        total_requests=len(requests),
                        successful=successful_requests,
                        failed=len(requests) - successful_requests,
                        execution_time=execution_time)

        return results

    def get_source_metrics(self, source: str) -> Optional[SourceMetrics]:
        """Get metrics for a specific source"""
        return self.source_metrics.get(source)

    def get_all_metrics(self) -> Dict[str, SourceMetrics]:
        """Get metrics for all sources"""
        return self.source_metrics.copy()

    async def generate_scraping_report(self) -> Dict[str, Any]:
        """Generate comprehensive scraping report"""

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_sources": len(self.source_metrics),
            "proxy_config": {
                "enabled": self.proxy_config.enabled,
                "proxy_count": len(self.proxy_config.proxies),
                "rotation_interval": self.proxy_config.rotation_interval
            },
            "sources": {}
        }

        total_requests = 0
        total_successful = 0
        total_failed = 0

        for source, metrics in self.source_metrics.items():
            total_requests += metrics.total_requests
            total_successful += metrics.successful_requests
            total_failed += metrics.failed_requests

            report["sources"][source] = {
                "total_requests": metrics.total_requests,
                "successful_requests": metrics.successful_requests,
                "failed_requests": metrics.failed_requests,
                "success_rate": metrics.completeness_percentage,
                "last_success": metrics.last_success_timestamp.isoformat() if metrics.last_success_timestamp else None,
                "last_error": metrics.last_error_timestamp.isoformat() if metrics.last_error_timestamp else None,
                "last_error_message": metrics.last_error_message,
                "average_response_time": metrics.average_response_time,
                "status": "healthy" if metrics.completeness_percentage >= 80 else "degraded" if metrics.completeness_percentage >= 50 else "unhealthy"
            }

        # Overall statistics
        overall_success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0

        report["summary"] = {
            "total_requests": total_requests,
            "total_successful": total_successful,
            "total_failed": total_failed,
            "overall_success_rate": overall_success_rate,
            "healthy_sources": len([s for s in report["sources"].values() if s["status"] == "healthy"]),
            "degraded_sources": len([s for s in report["sources"].values() if s["status"] == "degraded"]),
            "unhealthy_sources": len([s for s in report["sources"].values() if s["status"] == "unhealthy"])
        }

        return report

    async def save_daily_scraping_report(self) -> Path:
        """Save daily scraping report to logs"""

        # Create daily log directory
        today_str = datetime.now().strftime("%Y%m%d")
        daily_log_dir = Path("logs/daily") / today_str
        daily_log_dir.mkdir(parents=True, exist_ok=True)

        # Generate report
        report = await self.generate_scraping_report()

        # Save report
        timestamp_str = datetime.now().strftime("%H%M%S")
        report_file = daily_log_dir / f"scraping_report_{timestamp_str}.json"

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Also save as latest
        latest_file = daily_log_dir / "scraping_report.json"
        with open(latest_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Scraping report saved: {report_file}")

        return report_file


# Global client instance
_async_client: Optional[AsyncScrapeClient] = None

async def get_async_client(**kwargs) -> AsyncScrapeClient:
    """Get global async client instance"""
    global _async_client

    if _async_client is None:
        _async_client = AsyncScrapeClient(**kwargs)
        await _async_client._create_session()

    return _async_client

async def close_async_client():
    """Close global async client"""
    global _async_client

    if _async_client is not None:
        await _async_client._close_session()
        _async_client = None
