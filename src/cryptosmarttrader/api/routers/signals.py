#!/usr/bin/env python3
"""
Signals Router - Trading signal endpoints
"""

from fastapi import APIRouter
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.schemas.common import BaseResponse

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

@router.get("/active", summary="Get active trading signals")
@limiter.limit("30/minute")
async def get_signals(request) -> BaseResponse:
    """Get active trading signals"""
    return BaseResponse()