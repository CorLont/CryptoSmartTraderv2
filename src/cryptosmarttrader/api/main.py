#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Enterprise FastAPI Application
Contract-first API with hardening, rate limiting, and comprehensive monitoring
"""

import sys
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from typing import List

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

# Add src to path for configuration
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cryptosmarttrader.config import settings, log_startup_config
from api.routers import health, data, predictions, signals
from api.middleware import security_headers_middleware, logging_middleware


# Rate limiter configuration
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logging.info("Starting CryptoSmartTrader V2 API...")
    log_startup_config(settings)

    # Initialize Prometheus metrics
    if settings.ENABLE_PROMETHEUS:
        instrumentator = Instrumentator()
        instrumentator.instrument(app).expose(app)
        logging.info(f"Prometheus metrics enabled on /metrics")

    logging.info(f"API server ready on {settings.API_HOST}:{settings.API_PORT}")

    yield

    # Shutdown
    logging.info("Shutting down CryptoSmartTrader V2 API...")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""

    # Validate configuration
    missing_secrets = settings.validate_required_secrets()
    if missing_secrets and settings.is_production():
        raise RuntimeError(f"Missing required secrets: {missing_secrets}")

    # Create FastAPI app
    app = FastAPI(
        title="CryptoSmartTrader V2 API",
        description="Enterprise cryptocurrency trading intelligence API",
        version="2.0.0",
        docs_url="/docs" if not settings.is_production() else None,
        redoc_url="/redoc" if not settings.is_production() else None,
        openapi_url="/openapi.json" if not settings.is_production() else None,
        lifespan=lifespan,
        # Security headers
        swagger_ui_parameters={
            "displayRequestDuration": True,
            "tryItOutEnabled": True,
            "requestSnippetsEnabled": True,
        }
    )

    # Add rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    # CORS Configuration - Enterprise hardening
    if settings.ENABLE_CORS:
        allowed_origins = [
            f"http://localhost:{settings.DASHBOARD_PORT}",
            f"http://127.0.0.1:{settings.DASHBOARD_PORT}",
            f"http://0.0.0.0:{settings.DASHBOARD_PORT}",
        ]

        if settings.is_production():
            # Production: Restrict to specific domains
            allowed_origins = [
                "https://your-production-domain.com",  # Replace with actual domain
            ]

        app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
            expose_headers=["X-Rate-Limit-*", "X-Process-Time"],
        )

    # Trusted host middleware (security)
    if settings.is_production():
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["your-production-domain.com", "*.your-domain.com"]
        )

    # Custom middleware
    app.middleware("http")(security_headers_middleware)
    app.middleware("http")(logging_middleware)

    # Include routers with tags
    app.include_router(
        health.router,
        prefix="/health",
        tags=["Health & Monitoring"],
    )

    app.include_router(
        data.router,
        prefix="/api/v1/data",
        tags=["Market Data"],
    )

    app.include_router(
        predictions.router,
        prefix="/api/v1/predictions",
        tags=["ML Predictions"],
    )

    app.include_router(
        signals.router,
        prefix="/api/v1/signals",
        tags=["Trading Signals"],
    )

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logging.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred" if settings.is_production() else str(exc),
                "request_id": getattr(request.state, "request_id", None)
            }
        )

    return app


# Create application instance
app = create_app()


def main():
    """Run the application"""
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Server configuration with limits
    config = uvicorn.Config(
        app=app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
        # Performance & Security limits
        limit_concurrency=1000,
        limit_max_requests=10000,
        timeout_keep_alive=5,
        timeout_graceful_shutdown=10,
        # Request size limits
        h11_max_incomplete_event_size=16 * 1024,  # 16KB
    )

    logging.info(f"Starting server on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(config)


if __name__ == "__main__":
    main()
