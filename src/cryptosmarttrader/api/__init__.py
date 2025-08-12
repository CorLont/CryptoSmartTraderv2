"""
API Module - FastAPI web services and health monitoring

This module provides the REST API interface for CryptoSmartTrader V2,
including health endpoints, market data access, and prediction services.
"""

from .main import app
from .health_endpoint import health_router

__all__ = ["app", "health_router"]