import threading
import time
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import sys
import pickle
import pandas as pd
import numpy as np


class CacheManager:
    """Intelligent cache manager with memory calculation and TTL management"""

    def __init__(self, max_memory_mb: int = 1000):
        self.max_memory_mb = max_memory_mb
        self._cache: Dict[str, Dict] = {}  # Add _cache attribute for compatibility
        self.cache: Dict[str, Dict] = self._cache  # Keep both for compatibility
        self._lock = threading.Lock()
        self.cleanup_active = True

        # Start cleanup thread
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()

    def _calculate_memory_usage(self, obj: Any) -> float:
        """Calculate memory usage of object in MB"""
        try:
            if isinstance(obj, pd.DataFrame):
                return obj.memory_usage(deep=True).sum() / (1024 * 1024)
            elif isinstance(obj, np.ndarray):
                return obj.nbytes / (1024 * 1024)
            else:
                # General Python object
                return sys.getsizeof(obj) / (1024 * 1024)
        except Exception:
            return 0.1  # Default small size

    def _cleanup_loop(self):
        """Background cleanup of expired cache entries"""
        while self.cleanup_active:
            try:
                self._cleanup_expired()
                self._enforce_memory_limit()
                time.sleep(60)  # Cleanup every minute
            except Exception as e:
                print(f"Cache cleanup error: {e}")
                time.sleep(60)

    def _cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = datetime.now()
        expired_keys = []

        with self._lock:
            for key, cache_item in self._cache.items():
                if current_time > cache_item["expires"]:
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]

    def _enforce_memory_limit(self):
        """Enforce memory limit by removing oldest entries"""
        total_memory = self.get_total_memory_usage()

        if total_memory > self.max_memory_mb:
            # Sort by access time, remove oldest
            with self._lock:
                sorted_items = sorted(self._cache.items(), key=lambda x: x[1]["last_accessed"])

                while total_memory > self.max_memory_mb * 0.8 and sorted_items:
                    key, _ = sorted_items.pop(0)
                    if key in self._cache:
                        total_memory -= self._cache[key]["memory_mb"]
                        del self._cache[key]

    def set(self, key: str, value: Any, ttl_minutes: int = 60):
        """Set cache value with TTL"""
        memory_usage = self._calculate_memory_usage(value)
        expires = datetime.now() + timedelta(minutes=ttl_minutes)

        cache_item = {
            "value": value,
            "expires": expires,
            "memory_mb": memory_usage,
            "created": datetime.now(),
            "last_accessed": datetime.now(),
            "access_count": 0,
        }

        with self._lock:
            self._cache[key] = cache_item

    def get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        with self._lock:
            if key in self._cache:
                cache_item = self._cache[key]

                # Check if expired
                if datetime.now() > cache_item["expires"]:
                    del self._cache[key]
                    return None

                # Update access info
                cache_item["last_accessed"] = datetime.now()
                cache_item["access_count"] += 1

                return cache_item["value"]

            return None

    def delete(self, key: str) -> bool:
        """Delete cache entry"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self._cache.clear()

    def get_total_memory_usage(self) -> float:
        """Get total memory usage in MB"""
        with self._lock:
            return sum(item["memory_mb"] for item in self._cache.values())

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_items = len(self._cache)
            total_memory = self.get_total_memory_usage()

            if total_items > 0:
                avg_memory = total_memory / total_items
                total_accesses = sum(item["access_count"] for item in self._cache.values())
                avg_accesses = total_accesses / total_items
            else:
                avg_memory = 0
                avg_accesses = 0

            return {
                "total_items": total_items,
                "total_memory_mb": total_memory,
                "max_memory_mb": self.max_memory_mb,
                "memory_usage_percent": (total_memory / self.max_memory_mb) * 100,
                "avg_memory_per_item_mb": avg_memory,
                "avg_accesses_per_item": avg_accesses,
            }

    def get_cache_info(self) -> Dict[str, Dict]:
        """Get detailed info about cache entries"""
        with self._lock:
            info = {}
            for key, cache_item in self.cache.items():
                info[key] = {
                    "memory_mb": cache_item["memory_mb"],
                    "expires": cache_item["expires"].isoformat(),
                    "created": cache_item["created"].isoformat(),
                    "last_accessed": cache_item["last_accessed"].isoformat(),
                    "access_count": cache_item["access_count"],
                    "ttl_remaining_minutes": (
                        cache_item["expires"] - datetime.now()
                    ).total_seconds()
                    / 60,
                }
            return info

    def extend_ttl(self, key: str, additional_minutes: int) -> bool:
        """Extend TTL for a cache entry"""
        with self._lock:
            if key in self.cache:
                self.cache[key]["expires"] += timedelta(minutes=additional_minutes)
                return True
            return False

    def get_keys_by_pattern(self, pattern: str) -> list:
        """Get cache keys matching a pattern"""
        with self._lock:
            import fnmatch

            return [key for key in self.cache.keys() if fnmatch.fnmatch(key, pattern)]

    def stop_cleanup(self):
        """Stop the cleanup thread"""
        self.cleanup_active = False
