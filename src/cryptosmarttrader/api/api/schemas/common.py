#!/usr/bin/env python3
"""
Common API Schemas - Base response models and shared DTOs
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Generic, TypeVar
from pydantic import BaseModel, Field, ConfigDict

T = TypeVar("T")


class BaseResponse(BaseModel):
    """Base response model with common fields"""

    success: bool = Field(default=True, description="Request success status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(default=None, description="Request correlation ID")

    model_config = ConfigDict(json_encoders={datetime: lambda v: v.isoformat() + "Z"})


class ErrorResponse(BaseResponse):
    """Error response model"""

    success: bool = Field(default=False)
    error: str = Field(description="Error type")
    detail: str = Field(description="Detailed error message")
    code: Optional[str] = Field(default=None, description="Error code")


class PaginatedResponse(BaseResponse, Generic[T]):
    """Paginated response model"""

    data: List[T] = Field(description="Response data items")
    pagination: Dict[str, Any] = Field(description="Pagination metadata")

    @classmethod
    def create(
        cls, data: List[T], page: int = 1, page_size: int = 100, total_items: Optional[int] = None
    ) -> "PaginatedResponse[T]":
        """Create paginated response"""
        total_items = total_items or len(data)
        total_pages = (total_items + page_size - 1) // page_size

        return cls(
            data=data,
            pagination={
                "page": page,
                "page_size": page_size,
                "total_items": total_items,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1,
            },
        )


class ValidationErrorDetail(BaseModel):
    """Validation error detail"""

    field: str = Field(description="Field name that failed validation")
    message: str = Field(description="Validation error message")
    value: Any = Field(description="Invalid value")


class MetricsResponse(BaseResponse):
    """Metrics response model"""

    metrics: Dict[str, Any] = Field(description="System metrics")
    period: str = Field(description="Metrics time period")


class StatusInfo(BaseModel):
    """Status information model"""

    status: str = Field(description="Component status")
    message: Optional[str] = Field(default=None, description="Status message")
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    details: Optional[Dict[str, Any]] = Field(default=None)
