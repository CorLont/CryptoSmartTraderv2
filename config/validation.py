# config/validation.py
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional
from enum import Enum


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ExchangeType(str, Enum):
    KRAKEN = "kraken"
    BINANCE = "binance"
    KUCOIN = "kucoin"
    HUOBI = "huobi"
    COINBASE = "coinbase"
    BITFINEX = "bitfinex"


class MLModel(str, Enum):
    XGBOOST = "xgboost"
    SKLEARN = "sklearn"
    LIGHTGBM = "lightgbm"


class PredictionHorizon(str, Enum):
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    SEVEN_DAYS = "7d"
    THIRTY_DAYS = "30d"


class AgentConfig(BaseModel):
    """Configuration for individual agents"""

    enabled: bool = True
    update_interval: int = Field(ge=30, le=3600)  # 30 seconds to 1 hour
    use_openai: Optional[bool] = False
    openai_model: Optional[str] = "gpt-4o"


class SecurityConfig(BaseModel):
    """Security configuration"""

    enable_rate_limiting: bool = True
    max_failed_attempts: int = Field(ge=3, le=10)
    lockout_duration_minutes: int = Field(ge=5, le=60)
    audit_logging: bool = True
    input_validation: bool = True


class PerformanceConfig(BaseModel):
    """Performance optimization configuration"""

    parallel_workers: int = Field(ge=1, le=16)
    memory_limit_gb: int = Field(ge=2, le=64)
    enable_gpu: bool = False
    cache_ttl_minutes: int = Field(ge=5, le=1440)  # 5 minutes to 24 hours
    auto_optimization: bool = True


class SystemConfiguration(BaseModel):
    """Complete system configuration with validation"""

    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")

    # Exchange configuration
    exchanges: List[ExchangeType] = [ExchangeType.KRAKEN]
    api_rate_limit: int = Field(ge=10, le=1000)
    timeout_seconds: int = Field(ge=5, le=120)

    # Agent configurations
    agents: Dict[str, AgentConfig] = Field(default_factory=dict)

    # ML configuration
    prediction_horizons: List[PredictionHorizon] = [
        PredictionHorizon.ONE_HOUR,
        PredictionHorizon.FOUR_HOURS,
        PredictionHorizon.ONE_DAY,
        PredictionHorizon.SEVEN_DAYS,
        PredictionHorizon.THIRTY_DAYS,
    ]
    ml_models: List[MLModel] = [MLModel.XGBOOST, MLModel.SKLEARN]
    ensemble_weights: Dict[str, float] = Field(default_factory=dict)

    # System limits
    max_coins: int = Field(ge=10, le=1000)
    data_retention_days: int = Field(ge=30, le=3650)  # 30 days to 10 years

    # Performance settings
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    # Security settings
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    # Monitoring
    health_check_interval: int = Field(ge=1, le=60)  # 1 to 60 minutes
    alert_threshold: int = Field(ge=50, le=95)  # 50% to 95%
    log_level: LogLevel = LogLevel.INFO

    # API keys (will be empty in config, loaded from environment)
    api_keys: Dict[str, str] = Field(default_factory=dict)

    @validator("ensemble_weights")
    def validate_ensemble_weights(cls, v, values):
        """Validate that ensemble weights sum to 1.0"""
        if v and abs(sum(v.values()) - 1.0) > 0.01:
            raise ValueError("Ensemble weights must sum to 1.0")
        return v

    @validator("agents")
    def validate_agent_configs(cls, v):
        """Validate agent configurations"""
        required_agents = [
            "sentiment",
            "technical",
            "ml_predictor",
            "backtest",
            "trade_executor",
            "whale_detector",
        ]

        for agent in required_agents:
            if agent not in v:
                v[agent] = AgentConfig()

        return v

    @validator("prediction_horizons")
    def validate_prediction_horizons(cls, v):
        """Ensure at least one prediction horizon is specified"""
        if not v:
            raise ValueError("At least one prediction horizon must be specified")
        return v

    @validator("ml_models")
    def validate_ml_models(cls, v):
        """Ensure at least one ML model is specified"""
        if not v:
            raise ValueError("At least one ML model must be specified")
        return v

    class Config:
        use_enum_values = True
        validate_assignment = True


class TradingConfiguration(BaseModel):
    """Trading-specific configuration with strict validation"""

    # Risk management
    max_position_size: float = Field(ge=0.01, le=1.0)  # 1% to 100% of portfolio
    stop_loss_percentage: float = Field(ge=0.5, le=20.0)  # 0.5% to 20%
    take_profit_percentage: float = Field(ge=1.0, le=50.0)  # 1% to 50%
    max_daily_trades: int = Field(ge=1, le=100)

    # Portfolio settings
    base_currency: str = Field(pattern=r"^[A-Z]{3,4}$")  # USD, EUR, BTC, etc.
    initial_balance: float = Field(ge=100.0)  # Minimum $100
    max_positions: int = Field(ge=1, le=50)

    # Strategy settings
    enable_auto_trading: bool = False
    require_manual_approval: bool = True
    min_confidence_threshold: float = Field(ge=0.5, le=0.95)

    @validator("base_currency")
    def validate_base_currency(cls, v):
        """Validate base currency format"""
        allowed_currencies = ["USD", "EUR", "BTC", "ETH", "USDT", "USDC"]
        if v not in allowed_currencies:
            raise ValueError(f"Base currency must be one of: {allowed_currencies}")
        return v


class APIConfiguration(BaseModel):
    """API-specific configuration"""

    # FastAPI settings
    enable_api_server: bool = False
    api_port: int = Field(ge=1000, le=65535)
    enable_cors: bool = True
    api_rate_limit: int = Field(ge=10, le=1000)

    # Authentication
    require_api_key: bool = True
    api_key_length: int = Field(ge=32, le=128)

    # Documentation
    enable_docs: bool = True
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"


def validate_configuration(config_dict: Dict[str, Any]) -> SystemConfiguration:
    """Validate complete system configuration"""
    try:
        return SystemConfiguration(**config_dict)
    except Exception as e:
        raise ValueError(f"Configuration validation failed: {str(e)}")


def get_default_configuration() -> Dict[str, Any]:
    """Get default configuration dictionary"""
    return {
        "version": "2.0.0",
        "exchanges": ["kraken"],
        "api_rate_limit": 100,
        "timeout_seconds": 30,
        "agents": {
            "sentiment": {
                "enabled": True,
                "update_interval": 300,
                "use_openai": True,
                "openai_model": "gpt-4o",
            },
            "technical": {"enabled": True, "update_interval": 60},
            "ml_predictor": {"enabled": True, "update_interval": 900},
            "backtest": {"enabled": True, "update_interval": 3600},
            "trade_executor": {"enabled": True, "update_interval": 120},
            "whale_detector": {"enabled": True, "update_interval": 180},
        },
        "prediction_horizons": ["1h", "4h", "1d", "7d", "30d"],
        "ml_models": ["xgboost", "sklearn"],
        "ensemble_weights": {"xgboost": 0.6, "sklearn": 0.4},
        "max_coins": 453,
        "data_retention_days": 365,
        "performance": {
            "parallel_workers": 4,
            "memory_limit_gb": 8,
            "enable_gpu": False,
            "cache_ttl_minutes": 60,
            "auto_optimization": True,
        },
        "security": {
            "enable_rate_limiting": True,
            "max_failed_attempts": 5,
            "lockout_duration_minutes": 15,
            "audit_logging": True,
            "input_validation": True,
        },
        "health_check_interval": 5,
        "alert_threshold": 80,
        "log_level": "INFO",
        "api_keys": {},
    }
