#!/usr/bin/env python3
"""
System Settings - Pydantic BaseSettings Configuration
Centralized configuration management with environment variable support
"""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, validator
except ImportError:
    from pydantic import BaseSettings, Field, validator
from enum import Enum

class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class LogFormat(str, Enum):
    """Log formats"""
    JSON = "json"
    TEXT = "text"

class Environment(str, Enum):
    """Runtime environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class SystemSettings(BaseSettings):
    """System-wide configuration settings"""
    
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=True)
    
    # Logging Configuration
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_format: LogFormat = Field(default=LogFormat.JSON)
    log_dir: Path = Field(default=Path("logs"))
    
    # Performance Settings
    max_workers: int = Field(default=4, ge=1, le=16)
    rate_limit_requests: int = Field(default=100, ge=1)
    rate_limit_window: int = Field(default=3600, ge=60)
    
    # Monitoring
    prometheus_port: int = Field(default=8090, ge=1024, le=65535)
    health_check_interval: int = Field(default=60, ge=10)
    
    # ML/AI Configuration
    model_path: Path = Field(default=Path("models"))
    checkpoint_path: Path = Field(default=Path("checkpoints"))
    cache_ttl: int = Field(default=3600, ge=300)
    
    # GPU Configuration
    cuda_visible_devices: Optional[str] = Field(default="0")
    torch_device: str = Field(default="cuda")
    
    # Data Processing
    batch_size: int = Field(default=1000, ge=1)
    max_retries: int = Field(default=3, ge=1)
    timeout_seconds: int = Field(default=300, ge=30)
    
    # Security
    secret_key: Optional[str] = Field(default=None)
    jwt_secret: Optional[str] = Field(default=None)
    
    @validator('log_dir', 'model_path', 'checkpoint_path')
    def create_directories(cls, v):
        """Ensure directories exist"""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('environment')
    def validate_environment(cls, v):
        """Validate environment setting"""
        if v == Environment.PRODUCTION:
            # Additional production validations can be added here
            pass
        return v
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

class ExchangeSettings(BaseSettings):
    """Exchange API configuration"""
    
    # Kraken
    kraken_api_key: Optional[str] = Field(default=None)
    kraken_secret: Optional[str] = Field(default=None)
    kraken_sandbox: bool = Field(default=True)
    
    # Binance
    binance_api_key: Optional[str] = Field(default=None)
    binance_secret: Optional[str] = Field(default=None)
    binance_sandbox: bool = Field(default=True)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

class MLSettings(BaseSettings):
    """Machine Learning configuration"""
    
    # Model Parameters
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    prediction_horizons: List[str] = Field(default=["1h", "4h", "24h", "7d", "30d"])
    
    # Training Parameters
    train_test_split: float = Field(default=0.8, ge=0.5, le=0.95)
    validation_split: float = Field(default=0.2, ge=0.05, le=0.5)
    
    # Feature Engineering
    feature_selection_threshold: float = Field(default=0.01, ge=0.001, le=0.1)
    max_features: Optional[int] = Field(default=None)
    
    # Ensemble Parameters
    ensemble_size: int = Field(default=5, ge=3, le=20)
    uncertainty_method: str = Field(default="bayesian")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

class DataSettings(BaseSettings):
    """Data processing and storage configuration"""
    
    # Data Sources
    coinmarketcap_api_key: Optional[str] = Field(default=None)
    coingecko_api_key: Optional[str] = Field(default=None)
    
    # Social Media APIs
    twitter_bearer_token: Optional[str] = Field(default=None)
    reddit_client_id: Optional[str] = Field(default=None)
    reddit_client_secret: Optional[str] = Field(default=None)
    
    # Database
    database_url: Optional[str] = Field(default=None)
    redis_url: Optional[str] = Field(default="redis://localhost:6379/0")
    
    # Data Quality
    min_data_completeness: float = Field(default=0.8, ge=0.5, le=1.0)
    max_missing_values: float = Field(default=0.2, ge=0.0, le=0.5)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

class NotificationSettings(BaseSettings):
    """Notification and alerting configuration"""
    
    # Email Configuration
    smtp_host: str = Field(default="smtp.gmail.com")
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_user: Optional[str] = Field(default=None)
    smtp_password: Optional[str] = Field(default=None)
    notification_email: Optional[str] = Field(default=None)
    
    # Alert Thresholds
    error_rate_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    latency_threshold_ms: int = Field(default=5000, ge=100)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

class APISettings(BaseSettings):
    """External API configuration"""
    
    # OpenAI
    openai_api_key: Optional[str] = Field(default=None)
    openai_model: str = Field(default="gpt-4o")
    openai_max_tokens: int = Field(default=4000, ge=100, le=8000)
    
    # Vault (Optional)
    vault_url: Optional[str] = Field(default=None)
    vault_token: Optional[str] = Field(default=None)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

# Global settings instances
def get_system_settings() -> SystemSettings:
    """Get system settings instance"""
    return SystemSettings()

def get_exchange_settings() -> ExchangeSettings:
    """Get exchange settings instance"""
    return ExchangeSettings()

def get_ml_settings() -> MLSettings:
    """Get ML settings instance"""
    return MLSettings()

def get_data_settings() -> DataSettings:
    """Get data settings instance"""
    return DataSettings()

def get_notification_settings() -> NotificationSettings:
    """Get notification settings instance"""
    return NotificationSettings()

def get_api_settings() -> APISettings:
    """Get API settings instance"""
    return APISettings()

def get_all_settings() -> Dict[str, Any]:
    """Get all settings as dictionary"""
    return {
        "system": get_system_settings().dict(),
        "exchange": get_exchange_settings().dict(),
        "ml": get_ml_settings().dict(),
        "data": get_data_settings().dict(),
        "notification": get_notification_settings().dict(),
        "api": get_api_settings().dict()
    }