#!/usr/bin/env python3
"""
Predictions Schemas - DTOs for ML predictions
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

from .common import BaseResponse


class PredictionRequest(BaseModel):
    """Prediction request"""
    symbol: str = Field(description="Trading symbol")
    horizon: str = Field(description="Prediction horizon", example="24h")


class MLMetrics(BaseModel):
    """ML model metrics"""
    accuracy: float = Field(description="Model accuracy", ge=0, le=1)
    confidence: float = Field(description="Prediction confidence", ge=0, le=1)
    last_training: datetime = Field(description="Last training timestamp")


class PredictionResponse(BaseResponse):
    """Prediction response"""
    symbol: str = Field(description="Trading symbol")
    predicted_price: float = Field(description="Predicted price", gt=0)
    direction: str = Field(description="Price direction", example="up")
    confidence: float = Field(description="Prediction confidence", ge=0, le=1)
    horizon: str = Field(description="Prediction horizon")
    metrics: MLMetrics = Field(description="Model metrics")
