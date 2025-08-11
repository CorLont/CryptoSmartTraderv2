"""FastAPI Application Factory - Enterprise API Setup"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
from typing import Dict, Any

from .routers import health_router, market_router, trading_router, agents_router
from ..config import get_settings
from ..logging import get_logger, log_performance_metric

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting CryptoSmartTrader API server")
    settings = get_settings()
    logger.info(f"API server configuration: {settings.get_summary()}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down CryptoSmartTrader API server")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    settings = get_settings()
    
    # Create FastAPI instance
    app = FastAPI(
        title="CryptoSmartTrader V2 API",
        description="Enterprise Trading Intelligence API for Cryptocurrency Analysis",
        version="2.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan
    )
    
    # Add security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["localhost", "127.0.0.1", "0.0.0.0", "*.replit.app", "*.replit.dev"]
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.DEBUG_MODE else [
            f"http://localhost:{settings.DASHBOARD_PORT}",
            f"https://localhost:{settings.DASHBOARD_PORT}"
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"]
    )
    
    # Add request timing middleware
    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        # Log performance metric
        log_performance_metric(
            metric_name="api_request_duration",
            value=process_time,
            unit="seconds",
            tags={
                "method": request.method,
                "endpoint": str(request.url.path),
                "status_code": str(response.status_code)
            }
        )
        
        return response
    
    # Add exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            f"Unhandled exception in API: {exc}",
            exc_info=True,
            extra={
                "method": request.method,
                "url": str(request.url),
                "client_ip": request.client.host if request.client else None
            }
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred. Please try again later.",
                "timestamp": time.time()
            }
        )
    
    # Include API routers
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(market_router, prefix="/api/v1")
    app.include_router(trading_router, prefix="/api/v1")
    app.include_router(agents_router, prefix="/api/v1")
    
    # Root endpoint
    @app.get("/", tags=["root"])
    async def root() -> Dict[str, Any]:
        """API root endpoint with service information"""
        settings = get_settings()
        return {
            "service": "CryptoSmartTrader V2 API",
            "version": "2.0.0",
            "status": "operational",
            "docs_url": "/api/docs",
            "health_check": "/api/v1/health",
            "environment": "development" if settings.DEBUG_MODE else "production",
            "timestamp": time.time()
        }
    
    # Health check endpoint (simple)
    @app.get("/health", tags=["health"])
    async def simple_health():
        """Simple health check endpoint"""
        return {"status": "ok", "timestamp": time.time()}
    
    return app


# Create application instance
app = create_app()