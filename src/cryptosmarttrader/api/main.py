"""
FastAPI main application for CryptoSmartTrader V2
Enterprise-grade API with comprehensive documentation and monitoring.
"""

from contextlib import asynccontextmanager
from typing import Dict, Any, List
import logging
import time
import uuid
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
import uvicorn

# Application imports
from cryptosmarttrader.core.config import get_settings
from cryptosmarttrader.core.logging_config import setup_logging
from cryptosmarttrader.api.routers import health, trading, agents, market, portfolio, security


# Pydantic models for API documentation
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service health status")
    timestamp: datetime = Field(..., description="Response timestamp")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    components: Dict[str, str] = Field(..., description="Component health status")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Dict[str, Any] = Field(default={}, description="Additional error details")
    timestamp: datetime = Field(..., description="Error timestamp")
    request_id: str = Field(..., description="Request ID for tracking")


class VersionResponse(BaseModel):
    """Version information response model."""
    version: str = Field(..., description="Application version")
    build_date: str = Field(..., description="Build timestamp")
    git_commit: str = Field(..., description="Git commit hash")
    environment: str = Field(..., description="Environment name")
    python_version: str = Field(..., description="Python version")


# Global variables for tracking
start_time = time.time()
request_count = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger = logging.getLogger(__name__)
    logger.info("CryptoSmartTrader API starting up...")
    
    # Initialize services
    settings = get_settings()
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    yield
    
    # Shutdown
    logger.info("CryptoSmartTrader API shutting down...")


# Create FastAPI application
app = FastAPI(
    title="CryptoSmartTrader V2 API",
    description="""
    **Enterprise-grade cryptocurrency trading intelligence system**
    
    This API provides comprehensive access to:
    - Multi-agent trading intelligence
    - Real-time market analysis and predictions
    - Portfolio management and optimization
    - Risk management and monitoring
    - ML model registry and management
    - Security and compliance features
    
    ## Features
    
    - **Multi-Agent Architecture**: 8+ specialized agents for market analysis
    - **ML Intelligence**: Advanced machine learning with drift detection
    - **Risk Management**: Progressive escalation with kill-switch capabilities
    - **Real-time Analytics**: Prometheus monitoring with 24/7 observability
    - **Enterprise Security**: Comprehensive secrets management and audit logging
    - **Exchange Integration**: Kraken, Binance, KuCoin support with rate limiting
    
    ## Authentication
    
    API key authentication required for trading endpoints:
    ```
    Authorization: Bearer <your-api-key>
    ```
    
    ## Rate Limiting
    
    - Public endpoints: 100 requests/minute
    - Authenticated endpoints: 1000 requests/minute
    - Trading endpoints: 100 requests/minute
    
    ## Error Handling
    
    All errors return standardized error responses with:
    - Error type and message
    - Request ID for tracking
    - Additional details where applicable
    
    ## Support
    
    - Documentation: [README_OPERATIONS.md](../README_OPERATIONS.md)
    - Health Status: `/health`
    - Metrics: `:8000/metrics`
    """,
    version="2.5.0",
    docs_url=None,  # We'll serve custom docs
    redoc_url=None,  # We'll serve custom redoc
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request middleware for logging and metrics
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Request middleware for logging and metrics."""
    global request_count
    request_count += 1
    
    # Generate request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Start timing
    start_time = time.time()
    
    # Log request
    logger = logging.getLogger(__name__)
    logger.info(
        f"Request {request_id}: {request.method} {request.url}",
        extra={
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host
        }
    )
    
    # Process request
    try:
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Log response
        logger.info(
            f"Response {request_id}: {response.status_code} ({duration:.3f}s)",
            extra={
                "request_id": request_id,
                "status_code": response.status_code,
                "duration": duration
            }
        )
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}s"
        
        return response
        
    except Exception as e:
        # Log error
        logger.error(
            f"Request {request_id} failed: {str(e)}",
            extra={
                "request_id": request_id,
                "error": str(e)
            }
        )
        
        # Return error response
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "message": "An internal server error occurred",
                "details": {"error_type": type(e).__name__},
                "timestamp": datetime.utcnow().isoformat(),
                "request_id": request_id
            }
        )


# Custom documentation endpoints
@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    """Custom Swagger UI with enhanced styling."""
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - Swagger UI",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@4/swagger-ui.css",
    )


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    """Custom ReDoc documentation."""
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=f"{app.title} - ReDoc",
        redoc_js_url="https://cdn.jsdelivr.net/npm/redoc@next/bundles/redoc.standalone.js",
    )


# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema with enhanced metadata."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom metadata
    openapi_schema["info"]["contact"] = {
        "name": "CryptoSmartTrader Support",
        "email": "support@cryptosmarttrader.com",
        "url": "https://github.com/clont1/cryptosmarttrader"
    }
    
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT"
    }
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    # Add global security
    openapi_schema["security"] = [
        {"BearerAuth": []},
        {"ApiKeyAuth": []}
    ]
    
    # Add tags
    openapi_schema["tags"] = [
        {"name": "health", "description": "Health check and system status"},
        {"name": "trading", "description": "Trading operations and order management"},
        {"name": "portfolio", "description": "Portfolio management and analytics"},
        {"name": "market", "description": "Market data and analysis"},
        {"name": "agents", "description": "Multi-agent system management"},
        {"name": "security", "description": "Security and compliance features"},
        {"name": "ml", "description": "Machine learning and model management"},
    ]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# Root endpoint
@app.get("/", 
         response_model=Dict[str, Any],
         tags=["root"],
         summary="API Root Information")
async def root():
    """Get API root information and available endpoints."""
    return {
        "name": "CryptoSmartTrader V2 API",
        "version": app.version,
        "description": "Enterprise cryptocurrency trading intelligence system",
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
            "openapi": "/api/v1/openapi.json"
        },
        "endpoints": {
            "health": "/health",
            "trading": "/api/v1/trading",
            "portfolio": "/api/v1/portfolio", 
            "market": "/api/v1/market",
            "agents": "/api/v1/agents",
            "security": "/api/v1/security"
        },
        "monitoring": {
            "metrics": ":8000/metrics",
            "dashboard": ":5000"
        },
        "support": {
            "documentation": "README_OPERATIONS.md",
            "security": "SECURITY.md",
            "changelog": "CHANGELOG.md"
        }
    }


# Version endpoint
@app.get("/version",
         response_model=VersionResponse,
         tags=["root"],
         summary="Get version information")
async def get_version():
    """Get detailed version information."""
    import sys
    import os
    
    return VersionResponse(
        version=app.version,
        build_date=datetime.utcnow().isoformat(),
        git_commit=os.getenv("GIT_COMMIT", "unknown"),
        environment=os.getenv("ENVIRONMENT", "development"),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )


# Include routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(trading.router, prefix="/api/v1/trading", tags=["trading"])
app.include_router(portfolio.router, prefix="/api/v1/portfolio", tags=["portfolio"])
app.include_router(market.router, prefix="/api/v1/market", tags=["market"])
app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
app.include_router(security.router, prefix="/api/v1/security", tags=["security"])


# Global exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with standardized error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "message": exc.detail,
            "details": {},
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions with standardized error responses."""
    logger = logging.getLogger(__name__)
    logger.exception(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An internal server error occurred",
            "details": {"error_type": type(exc).__name__},
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )


# Application factory
def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    setup_logging()
    return app


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )