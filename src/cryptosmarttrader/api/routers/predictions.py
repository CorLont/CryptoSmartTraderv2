#!/usr/bin/env python3
"""
Predictions Router - ML prediction endpoints
"""

from fastapi import APIRouter
from slowapi import Limiter
from slowapi.util import get_remote_address

from ..api.schemas.common import BaseResponse

limiter = Limiter(key_func=get_remote_address)
router = APIRouter()


@router.get("/latest", summary="Get latest predictions")
@limiter.limit("50/minute")
async def get_predictions(request) -> BaseResponse:
    """Get latest ML predictions"""
    return BaseResponse()
