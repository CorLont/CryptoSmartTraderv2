# models/validation_models.py
from pydantic import BaseModel, Field, validator, root_validator
from typing import List, Dict, Optional, Union, Any
from decimal import Decimal
from datetime import datetime
from enum import Enum


class TimeFrame(str, Enum):
    """Valid timeframes for data requests"""
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    TWELVE_HOURS = "12h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"


class ExchangeName(str, Enum):
    """Supported cryptocurrency exchanges"""
    KRAKEN = "kraken"
    BINANCE = "binance"
    KUCOIN = "kucoin"
    HUOBI = "huobi"


class AgentType(str, Enum):
    """Available agent types"""
    SENTIMENT = "sentiment"
    TECHNICAL = "technical"
    ML_PREDICTOR = "ml_predictor"
    BACKTEST = "backtest"
    TRADE_EXECUTOR = "trade_executor"
    WHALE_DETECTOR = "whale_detector"


class CoinSymbolRequest(BaseModel):
    """Validates cryptocurrency symbol requests"""
    symbols: List[str] = Field(..., min_items=1, max_items=100)
    
    @validator("symbols", each_item=True)
    def validate_symbol(cls, v):
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Symbol must be a non-empty string")
        
        symbol = v.strip().upper()
        if not symbol.replace('-', '').replace('_', '').isalnum():
            raise ValueError("Symbol contains invalid characters")
        
        if len(symbol) < 2 or len(symbol) > 10:
            raise ValueError("Symbol must be between 2 and 10 characters")
        
        return symbol


class MarketDataRequest(BaseModel):
    """Validates market data requests"""
    symbols: List[str] = Field(..., min_items=1, max_items=50)
    timeframe: TimeFrame = Field(default=TimeFrame.ONE_HOUR)
    limit: int = Field(default=100, ge=1, le=1000)
    exchange: Optional[ExchangeName] = None
    
    @validator("symbols", each_item=True)
    def validate_symbols(cls, v):
        return v.strip().upper()


class PredictionRequest(BaseModel):
    """Validates ML prediction requests"""
    symbol: str = Field(..., min_length=2, max_length=10)
    timeframe: TimeFrame = Field(default=TimeFrame.ONE_HOUR)
    prediction_horizons: List[str] = Field(default=["1h", "4h", "24h"])
    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    
    @validator("symbol")
    def validate_symbol(cls, v):
        symbol = v.strip().upper()
        if not symbol.replace('-', '').replace('_', '').isalnum():
            raise ValueError("Invalid symbol format")
        return symbol
    
    @validator("prediction_horizons", each_item=True)
    def validate_horizons(cls, v):
        valid_horizons = ["1h", "4h", "12h", "24h", "7d", "30d"]
        if v not in valid_horizons:
            raise ValueError(f"Invalid prediction horizon: {v}. Must be one of {valid_horizons}")
        return v


class TradingSignalRequest(BaseModel):
    """Validates trading signal requests"""
    symbol: str = Field(..., min_length=2, max_length=10)
    position_size: float = Field(..., gt=0, le=1.0)
    stop_loss: Optional[float] = Field(None, gt=0, le=0.5)
    take_profit: Optional[float] = Field(None, gt=0, le=2.0)
    risk_tolerance: str = Field(default="medium", regex="^(low|medium|high)$")
    
    @validator("symbol")
    def validate_symbol(cls, v):
        return v.strip().upper()
    
    @root_validator
    def validate_risk_parameters(cls, values):
        stop_loss = values.get("stop_loss")
        take_profit = values.get("take_profit")
        
        if stop_loss and take_profit and stop_loss >= take_profit:
            raise ValueError("Stop loss must be less than take profit")
        
        return values


class BacktestRequest(BaseModel):
    """Validates backtesting requests"""
    symbols: List[str] = Field(..., min_items=1, max_items=20)
    start_date: datetime
    end_date: datetime
    initial_capital: float = Field(default=10000.0, gt=0)
    strategy_name: str = Field(..., min_length=3, max_length=50)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    @validator("symbols", each_item=True)
    def validate_symbols(cls, v):
        return v.strip().upper()
    
    @root_validator
    def validate_dates(cls, values):
        start_date = values.get("start_date")
        end_date = values.get("end_date")
        
        if start_date and end_date:
            if start_date >= end_date:
                raise ValueError("Start date must be before end date")
            
            if (end_date - start_date).days < 1:
                raise ValueError("Backtest period must be at least 1 day")
        
        return values


class AgentConfigRequest(BaseModel):
    """Validates agent configuration requests"""
    agent_type: AgentType
    enabled: bool = Field(default=True)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=1, ge=1, le=10)
    
    @validator("parameters")
    def validate_parameters(cls, v, values):
        agent_type = values.get("agent_type")
        
        # Validate specific parameters for each agent type
        if agent_type == AgentType.SENTIMENT:
            allowed_params = {"source_weight", "sentiment_threshold", "update_frequency"}
        elif agent_type == AgentType.TECHNICAL:
            allowed_params = {"indicators", "timeframes", "signal_threshold"}
        elif agent_type == AgentType.ML_PREDICTOR:
            allowed_params = {"model_type", "retrain_frequency", "confidence_threshold"}
        else:
            allowed_params = set()
        
        if allowed_params:
            invalid_params = set(v.keys()) - allowed_params
            if invalid_params:
                raise ValueError(f"Invalid parameters for {agent_type}: {invalid_params}")
        
        return v


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., regex="^(healthy|degraded|unhealthy)$")
    timestamp: datetime = Field(default_factory=datetime.now)
    components: Dict[str, str]
    overall_grade: str
    overall_grade_numeric: float = Field(..., ge=0.0, le=1.0)
    details: Optional[Dict[str, Any]] = None


class MetricsResponse(BaseModel):
    """System metrics response model"""
    timestamp: datetime = Field(default_factory=datetime.now)
    request_count: int = Field(..., ge=0)
    error_rate: float = Field(..., ge=0.0, le=1.0)
    avg_response_time: float = Field(..., ge=0.0)
    health_score: float = Field(..., ge=0.0, le=1.0)
    active_agents: int = Field(..., ge=0)
    cache_hit_ratio: float = Field(..., ge=0.0, le=1.0)


class ErrorResponse(BaseModel):
    """Standard error response model"""
    error: str
    error_code: str
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Input sanitization utilities
def sanitize_string(value: str, max_length: int = 255, allow_special: bool = False) -> str:
    """
    Sanitize string input to prevent injection attacks.
    
    Args:
        value: Input string
        max_length: Maximum allowed length
        allow_special: Whether to allow special characters
        
    Returns:
        Sanitized string
    """
    if not isinstance(value, str):
        raise ValueError("Input must be a string")
    
    # Remove null bytes and control characters
    sanitized = value.replace('\x00', '').strip()
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    # Remove dangerous characters if not allowed
    if not allow_special:
        dangerous_chars = ['<', '>', '"', "'", '&', '`', '|', ';']
        for char in dangerous_chars:
            sanitized = sanitized.replace(char, '')
    
    return sanitized


def validate_numeric_range(value: Union[int, float], min_val: float = None, 
                         max_val: float = None) -> Union[int, float]:
    """
    Validate numeric values are within acceptable ranges.
    
    Args:
        value: Numeric value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Validated numeric value
    """
    if not isinstance(value, (int, float)):
        raise ValueError("Value must be numeric")
    
    if min_val is not None and value < min_val:
        raise ValueError(f"Value {value} is below minimum {min_val}")
    
    if max_val is not None and value > max_val:
        raise ValueError(f"Value {value} is above maximum {max_val}")
    
    return value