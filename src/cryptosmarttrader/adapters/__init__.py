"""
Adapters - Concrete implementations of domain interfaces

This module contains the concrete implementations (adapters) that fulfill
the contracts defined by the domain interfaces (ports).
"""

from .kraken_data_adapter import KrakenDataAdapter
from .redis_cache_adapter import RedisCacheAdapter
from .file_storage_adapter import FileStorageAdapter
from .sklearn_model_adapter import SklearnModelAdapter

__all__ = ["KrakenDataAdapter", "RedisCacheAdapter", "FileStorageAdapter", "SklearnModelAdapter"]
