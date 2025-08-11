"""
Storage Port - Interface for data persistence and caching

Defines the contract for all storage implementations (cache, database, file system)
enabling swappable storage backends without affecting business logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Tuple
import pandas as pd
from datetime import datetime, timedelta
from enum import Enum
import json

class StorageType(Enum):
    """Types of storage backends"""
    MEMORY_CACHE = "memory_cache"
    REDIS_CACHE = "redis_cache"
    FILE_SYSTEM = "file_system"
    DATABASE = "database"
    OBJECT_STORAGE = "object_storage"

class DataFormat(Enum):
    """Supported data formats"""
    JSON = "json"
    PARQUET = "parquet"
    CSV = "csv"
    PICKLE = "pickle"
    ARROW = "arrow"

class CachePolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    TTL = "ttl"  # Time To Live
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out

class StorageRequest:
    """Request object for storage operations"""
    
    def __init__(self, key: str, data: Any = None, ttl: Optional[int] = None,
                 metadata: Optional[Dict] = None):
        self.key = key
        self.data = data
        self.ttl = ttl  # Time to live in seconds
        self.metadata = metadata or {}

class StorageResponse:
    """Response object for storage operations"""
    
    def __init__(self, success: bool, data: Any = None, metadata: Optional[Dict] = None,
                 error: Optional[str] = None):
        self.success = success
        self.data = data
        self.metadata = metadata or {}
        self.error = error

class StoragePort(ABC):
    """
    Abstract interface for storage operations
    
    This port defines the contract for all storage implementations,
    providing a unified interface for caching, persistence, and retrieval.
    """
    
    @abstractmethod
    def store(self, key: str, data: Any, ttl: Optional[int] = None,
              metadata: Optional[Dict] = None) -> StorageResponse:
        """
        Store data with optional TTL and metadata
        
        Args:
            key: Unique identifier for the data
            data: Data to store (DataFrame, dict, etc.)
            ttl: Time to live in seconds (None = no expiration)
            metadata: Additional metadata to store with data
            
        Returns:
            StorageResponse indicating success/failure
        """
        pass
    
    @abstractmethod
    def retrieve(self, key: str) -> StorageResponse:
        """
        Retrieve data by key
        
        Args:
            key: Unique identifier for the data
            
        Returns:
            StorageResponse with data if found, error if not
        """
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if key exists in storage
        
        Args:
            key: Key to check
            
        Returns:
            True if key exists, False otherwise
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> StorageResponse:
        """
        Delete data by key
        
        Args:
            key: Key to delete
            
        Returns:
            StorageResponse indicating success/failure
        """
        pass
    
    @abstractmethod
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        List all keys, optionally filtered by pattern
        
        Args:
            pattern: Optional pattern to filter keys (e.g., 'market_data:*')
            
        Returns:
            List of matching keys
        """
        pass
    
    @abstractmethod
    def clear(self, pattern: Optional[str] = None) -> int:
        """
        Clear storage, optionally filtered by pattern
        
        Args:
            pattern: Optional pattern to filter keys to clear
            
        Returns:
            Number of keys cleared
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics
        
        Returns:
            Dictionary with storage metrics (size, hit rate, etc.)
        """
        pass

class CachePort(StoragePort):
    """Extended interface specifically for cache implementations"""
    
    @abstractmethod
    def get_or_compute(self, key: str, compute_func: callable, 
                      ttl: Optional[int] = None) -> StorageResponse:
        """
        Get cached value or compute and cache it
        
        Args:
            key: Cache key
            compute_func: Function to compute value if not cached
            ttl: Time to live for computed value
            
        Returns:
            StorageResponse with cached or computed data
        """
        pass
    
    @abstractmethod
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern
        
        Args:
            pattern: Pattern to match (e.g., 'market_data:BTC*')
            
        Returns:
            Number of keys invalidated
        """
        pass
    
    @abstractmethod
    def set_ttl(self, key: str, ttl: int) -> bool:
        """
        Update TTL for existing key
        
        Args:
            key: Key to update
            ttl: New TTL in seconds
            
        Returns:
            True if TTL was set, False if key doesn't exist
        """
        pass
    
    @abstractmethod
    def get_ttl(self, key: str) -> Optional[int]:
        """
        Get remaining TTL for key
        
        Args:
            key: Key to check
            
        Returns:
            Remaining TTL in seconds, None if no TTL or key doesn't exist
        """
        pass

class DatabasePort(StoragePort):
    """Extended interface specifically for database implementations"""
    
    @abstractmethod
    def store_dataframe(self, table_name: str, df: pd.DataFrame, 
                       if_exists: str = 'replace') -> StorageResponse:
        """
        Store DataFrame in database table
        
        Args:
            table_name: Name of the table
            df: DataFrame to store
            if_exists: What to do if table exists ('replace', 'append', 'fail')
            
        Returns:
            StorageResponse indicating success/failure
        """
        pass
    
    @abstractmethod
    def query_dataframe(self, query: str, params: Optional[Dict] = None) -> StorageResponse:
        """
        Execute SQL query and return DataFrame
        
        Args:
            query: SQL query string
            params: Optional query parameters
            
        Returns:
            StorageResponse with DataFrame result
        """
        pass
    
    @abstractmethod
    def create_index(self, table_name: str, columns: List[str]) -> StorageResponse:
        """
        Create database index for performance
        
        Args:
            table_name: Name of the table
            columns: Columns to index
            
        Returns:
            StorageResponse indicating success/failure
        """
        pass

class TimeSeriesStoragePort(StoragePort):
    """Interface for time-series optimized storage"""
    
    @abstractmethod
    def store_timeseries(self, symbol: str, timeframe: str, df: pd.DataFrame,
                        start_time: Optional[datetime] = None) -> StorageResponse:
        """
        Store time-series data optimized for temporal queries
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSD')
            timeframe: Data timeframe (e.g., '1h', '1d')
            df: Time-series DataFrame
            start_time: Optional start time for the data
            
        Returns:
            StorageResponse indicating success/failure
        """
        pass
    
    @abstractmethod
    def retrieve_timeseries(self, symbol: str, timeframe: str,
                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> StorageResponse:
        """
        Retrieve time-series data with temporal filtering
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            StorageResponse with filtered time-series data
        """
        pass

class StorageError(Exception):
    """Exception raised by storage implementations"""
    
    def __init__(self, message: str, error_code: Optional[str] = None,
                 operation: Optional[str] = None):
        super().__init__(message)
        self.error_code = error_code
        self.operation = operation

class StorageRegistry:
    """Registry for managing multiple storage implementations"""
    
    def __init__(self):
        self._storages: Dict[str, StoragePort] = {}
        self._default_storage: Optional[str] = None
    
    def register_storage(self, name: str, storage: StoragePort, is_default: bool = False):
        """Register a storage implementation"""
        self._storages[name] = storage
        if is_default or self._default_storage is None:
            self._default_storage = name
    
    def get_storage(self, name: Optional[str] = None) -> StoragePort:
        """Get a specific storage or the default one"""
        storage_name = name or self._default_storage
        if storage_name not in self._storages:
            raise StorageError(f"Storage '{storage_name}' not found")
        return self._storages[storage_name]
    
    def list_storages(self) -> List[str]:
        """Get list of registered storage names"""
        return list(self._storages.keys())

# Global registry instance
storage_registry = StorageRegistry()

# Utility functions for common storage patterns
def create_cache_key(prefix: str, *args, **kwargs) -> str:
    """Create standardized cache key"""
    key_parts = [prefix] + [str(arg) for arg in args]
    if kwargs:
        sorted_kwargs = sorted(kwargs.items())
        key_parts.extend([f"{k}={v}" for k, v in sorted_kwargs])
    return ":".join(key_parts)

def get_ttl_for_timeframe(timeframe: str) -> int:
    """Get appropriate TTL based on data timeframe"""
    ttl_mapping = {
        '1m': 60,        # 1 minute
        '5m': 300,       # 5 minutes  
        '15m': 900,      # 15 minutes
        '1h': 3600,      # 1 hour
        '4h': 14400,     # 4 hours
        '1d': 86400,     # 1 day
        '1w': 604800,    # 1 week
    }
    return ttl_mapping.get(timeframe, 3600)  # Default 1 hour