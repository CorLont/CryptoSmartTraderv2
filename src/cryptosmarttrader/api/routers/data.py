#!/usr/bin/env python3
"""
Data Router - Market data endpoints with rate limiting and validation
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from slowapi import Limiter
from slowapi.util import get_remote_address
from typing import List, Optional
from datetime import datetime

from api.schemas.common import BaseResponse, PaginatedResponse

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

router = APIRouter()

@router.get("/coins", summary="Get available cryptocurrencies")
@limiter.limit("100/minute")
async def get_coins(request, limit: int = Query(100, le=500)) -> BaseResponse:
    """Get list of available cryptocurrencies"""
    return BaseResponse()
