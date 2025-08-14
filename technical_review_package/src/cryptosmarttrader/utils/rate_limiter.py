# utils/rate_limiter.py
import time
import threading
from typing import Dict, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Rate limit configuration"""

    requests: int  # Number of requests
    window: float  # Time window in seconds
    burst_allowance: int = 0  # Additional burst requests allowed


class RateLimiter:
    """Advanced rate limiting system with burst handling and priorities"""

    def __init__(self):
        self.limits: Dict[str, RateLimit] = {}
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque())
        self.priority_weights: Dict[str, float] = {}
        self._locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)

        # Default limits for different endpoints/services
        self._setup_default_limits()

    def _setup_default_limits(self):
        """Setup default rate limits for common services"""
        # Cryptocurrency exchange API limits
        self.set_rate_limit("kraken_api", RateLimit(requests=60, window=60))  # 1 req/sec
        self.set_rate_limit("binance_api", RateLimit(requests=1200, window=60))  # 20 req/sec
        self.set_rate_limit("kucoin_api", RateLimit(requests=100, window=10))  # 10 req/sec

        # AI/ML service limits
        self.set_rate_limit("openai_api", RateLimit(requests=50, window=60, burst_allowance=10))

        # Internal system limits
        self.set_rate_limit("data_processing", RateLimit(requests=1000, window=60))
        self.set_rate_limit("ml_training", RateLimit(requests=10, window=300))  # 10 every 5 min

        # Priority weights (higher = more priority)
        self.priority_weights = {"critical": 1.0, "high": 0.8, "normal": 0.6, "low": 0.4}

    def set_rate_limit(self, key: str, limit: RateLimit):
        """Set rate limit for a specific key"""
        self.limits[key] = limit
        logger.info(f"Rate limit set for '{key}': {limit.requests} requests per {limit.window}s")

    def can_proceed(self, key: str, priority: str = "normal") -> bool:
        """
        Check if request can proceed without blocking

        Args:
            key: Rate limit key (e.g., 'kraken_api')
            priority: Request priority ('critical', 'high', 'normal', 'low')

        Returns:
            bool: True if request can proceed immediately
        """
        if key not in self.limits:
            return True

        with self._locks[key]:
            current_time = time.time()
            limit = self.limits[key]
            history = self.request_history[key]

            # Clean old requests outside the window
            cutoff_time = current_time - limit.window
            while history and history[0]["timestamp"] < cutoff_time:
                history.popleft()

            # Calculate available quota with priority consideration
            priority_weight = self.priority_weights.get(priority, 0.6)
            effective_limit = int(limit.requests * priority_weight) + limit.burst_allowance

            return len(history) < effective_limit

    def wait_if_needed(
        self, key: str, priority: str = "normal", timeout: Optional[float] = None
    ) -> bool:
        """
        Wait if rate limit would be exceeded, with timeout

        Args:
            key: Rate limit key
            priority: Request priority
            timeout: Maximum time to wait (None for no timeout)

        Returns:
            bool: True if can proceed, False if timeout exceeded
        """
        if key not in self.limits:
            return True

        start_time = time.time()

        while not self.can_proceed(key, priority):
            if timeout and (time.time() - start_time) > timeout:
                return False

            # Calculate wait time based on oldest request
            with self._locks[key]:
                history = self.request_history[key]
                if history:
                    oldest_request = history[0]
                    limit = self.limits[key]
                    wait_time = min(1.0, oldest_request["timestamp"] + limit.window - time.time())
                    if wait_time > 0:
                        time.sleep(wait_time)
                else:
                    time.sleep(0.1)

        return True

    def record_request(self, key: str, metadata: Optional[Dict[str, Any]] = None):
        """Record a request for rate limiting purposes"""
        if key not in self.limits:
            return

        with self._locks[key]:
            self.request_history[key].append({"timestamp": time.time(), "metadata": metadata or {}})

    def get_rate_limit_status(self, key: str) -> Dict[str, Any]:
        """Get current rate limit status for a key"""
        if key not in self.limits:
            return {"error": f"No rate limit configured for key: {key}"}

        with self._locks[key]:
            current_time = time.time()
            limit = self.limits[key]
            history = self.request_history[key]

            # Clean old requests
            cutoff_time = current_time - limit.window
            while history and history[0]["timestamp"] < cutoff_time:
                history.popleft()

            requests_in_window = len(history)
            remaining = max(0, limit.requests - requests_in_window)

            reset_time = None
            if history:
                reset_time = history[0]["timestamp"] + limit.window

            return {
                "key": key,
                "limit": limit.requests,
                "window": limit.window,
                "requests_in_window": requests_in_window,
                "remaining": remaining,
                "reset_time": reset_time,
                "burst_allowance": limit.burst_allowance,
            }

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get rate limit status for all configured keys"""
        return {key: self.get_rate_limit_status(key) for key in self.limits.keys()}

    def reset_history(self, key: str):
        """Reset request history for a specific key"""
        if key in self.request_history:
            with self._locks[key]:
                self.request_history[key].clear()
            logger.info(f"Rate limit history reset for '{key}'")

    def adjust_limit(self, key: str, new_requests: int, new_window: Optional[float] = None):
        """Dynamically adjust rate limits"""
        if key in self.limits:
            limit = self.limits[key]
            limit.requests = new_requests
            if new_window:
                limit.window = new_window
            logger.info(
                f"Rate limit adjusted for '{key}': {new_requests} requests per {limit.window}s"
            )


# Global rate limiter instance
rate_limiter = RateLimiter()


def rate_limited(key: str, priority: str = "normal", timeout: Optional[float] = 30):
    """
    Decorator for rate-limited functions

    Args:
        key: Rate limit key
        priority: Request priority
        timeout: Maximum wait time
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if not rate_limiter.wait_if_needed(key, priority, timeout):
                raise TimeoutError(f"Rate limit timeout exceeded for '{key}'")

            try:
                result = func(*args, **kwargs)
                rate_limiter.record_request(key, {"function": func.__name__, "success": True})
                return result
            except Exception as e:
                rate_limiter.record_request(
                    key, {"function": func.__name__, "success": False, "error": str(e)}
                )
                raise

        return wrapper

    return decorator
