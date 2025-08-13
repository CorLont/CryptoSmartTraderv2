"""
CryptoSmartTrader V2 - API Module

FastAPI-based REST API for system integration and monitoring:
- Health endpoints for service monitoring
- Trading data and analytics endpoints
- Risk management status and controls
- ML model predictions and confidence scores
"""

# API router imports
from .main import app
from .health_endpoint import health_router

__all__ = ["app", "health_router"]
