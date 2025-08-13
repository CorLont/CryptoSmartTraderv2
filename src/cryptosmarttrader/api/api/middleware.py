#!/usr/bin/env python3
"""
FastAPI Middleware - Security headers, logging, and request processing
"""

import time
import uuid
import logging
from typing import Callable
from fastapi import Request, Response


async def security_headers_middleware(request: Request, call_next: Callable) -> Response:
    """Add security headers to all responses"""
    response = await call_next(request)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    return response


async def logging_middleware(request: Request, call_next: Callable) -> Response:
    """Log requests and add correlation ID"""
    # Generate request ID
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Start timer
    start_time = time.time()
    
    # Log request
    logger = logging.getLogger("api.requests")
    logger.info(
        f"Request started",
        extra={
            "request_id": request_id,
            "method": request.method,
            "url": str(request.url),
            "client_ip": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }
    )
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Add headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(process_time)
    
    # Log response
    logger.info(
        f"Request completed",
        extra={
            "request_id": request_id,
            "status_code": response.status_code,
            "process_time": process_time,
        }
    )
    
    return response