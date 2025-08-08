#!/usr/bin/env python3
"""
Pydantic Configuration Management with Environment Variables
Centralized, type-safe configuration for the entire system
"""

from pydantic import BaseSettings, Field, validator
from typing import Optional, List, Dict, Any
from pathlib import Path
import os
from enum import Enum

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ExchangeSettings(BaseSettings):
    """Exchange-specific configuration"""
    kraken_api_key: Optional[str] = Field(None, env="KRAKEN_API_KEY")
    kraken_secret: Optional[str] = Field(None, env="KRAKEN_SECRET")
    binance_api_key: Optional[str] = Field(None, env="BINANCE_API_KEY")
    binance_secret: Optional[str] = Field(None, env="BINANCE_SECRET")
    
    # Rate limiting
    requests_per_second: float = Field(10.0, ge=1.0, le=100.0)
    burst_size: int = Field(50, ge=10, le=1000)
    timeout_seconds: int = Field(30, ge=5, le=120)
    
    @validator('kraken_api_key', 'kraken_secret')
    def validate_kraken_credentials(cls, v, values, field):
        """Ensure both Kraken credentials are provided together"""
        if field.name == 'kraken_secret' and values.get('kraken_api_key') and not v:
            raise ValueError("Kraken secret required when API key provided")
        return v

class DatabaseSettings(BaseSettings):
    """Database configuration"""
    database_url: Optional[str] = Field(None, env="DATABASE_URL")
    connection_pool_size: int = Field(10, ge=1, le=100)
    connection_timeout: int = Field(30, ge=5, le=120)
    query_timeout: int = Field(60, ge=10, le=300)

class MLSettings(BaseSettings):
    """Machine Learning configuration"""
    model_cache_dir: Path = Field(Path("models"), env="MODEL_CACHE_DIR")
    training_data_days: int = Field(90, ge=30, le=365)
    prediction_confidence_threshold: float = Field(0.7, ge=0.0, le=1.0)
    batch_size: int = Field(32, ge=1, le=1024)
    learning_rate: float = Field(0.001, ge=0.0001, le=0.1)
    
    # Model types
    enable_lstm: bool = Field(True, env="ENABLE_LSTM")
    enable_transformer: bool = Field(True, env="ENABLE_TRANSFORMER")
    enable_ensemble: bool = Field(True, env="ENABLE_ENSEMBLE")
    
    @validator('model_cache_dir')
    def create_model_dir(cls, v):
        """Ensure model directory exists"""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

class AgentSettings(BaseSettings):
    """Agent process configuration"""
    health_check_interval: int = Field(30, ge=5, le=300)
    restart_limit: int = Field(3, ge=1, le=10)
    circuit_breaker_threshold: int = Field(5, ge=2, le=20)
    backoff_multiplier: float = Field(2.0, ge=1.1, le=5.0)
    max_backoff_seconds: int = Field(60, ge=10, le=600)
    
    # Resource limits
    max_memory_mb: int = Field(1024, ge=256, le=8192)
    max_cpu_percent: int = Field(90, ge=50, le=100)

class LoggingSettings(BaseSettings):
    """Logging configuration"""
    log_level: LogLevel = Field(LogLevel.INFO, env="LOG_LEVEL")
    log_dir: Path = Field(Path("logs"), env="LOG_DIR")
    log_rotation_mb: int = Field(100, ge=10, le=1000)
    log_retention_days: int = Field(30, ge=1, le=365)
    enable_json_logging: bool = Field(True, env="ENABLE_JSON_LOGGING")
    
    @validator('log_dir')
    def create_log_dir(cls, v):
        """Ensure log directory exists"""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

class DataSettings(BaseSettings):
    """Data storage and processing configuration"""
    data_dir: Path = Field(Path("data"), env="DATA_DIR")
    cache_ttl_seconds: int = Field(300, ge=60, le=3600)
    cleanup_interval_hours: int = Field(24, ge=1, le=168)
    max_file_age_days: int = Field(7, ge=1, le=90)
    compression_enabled: bool = Field(True, env="ENABLE_COMPRESSION")
    
    @validator('data_dir')
    def create_data_dir(cls, v):
        """Ensure data directory exists"""
        for subdir in ['raw', 'processed', 'market_data', 'predictions']:
            (Path(v) / subdir).mkdir(parents=True, exist_ok=True)
        return v

class SecuritySettings(BaseSettings):
    """Security and encryption configuration"""
    encryption_key: Optional[str] = Field(None, env="ENCRYPTION_KEY")
    api_rate_limit_per_hour: int = Field(1000, ge=100, le=10000)
    max_concurrent_connections: int = Field(100, ge=10, le=1000)
    ssl_verify: bool = Field(True, env="SSL_VERIFY")

class AppSettings(BaseSettings):
    """Main application configuration aggregating all settings"""
    
    # Environment
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(False, env="DEBUG")
    
    # External services
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    
    # Component settings
    exchange: ExchangeSettings = Field(default_factory=ExchangeSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    ml: MLSettings = Field(default_factory=MLSettings)
    agents: AgentSettings = Field(default_factory=AgentSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        """Initialize settings with validation"""
        super().__init__(**kwargs)
        self._validate_critical_settings()
    
    def _validate_critical_settings(self):
        """Validate critical configuration combinations"""
        # Check if any exchange credentials are available
        has_kraken = self.exchange.kraken_api_key and self.exchange.kraken_secret
        has_binance = self.exchange.binance_api_key and self.exchange.binance_secret
        
        if not (has_kraken or has_binance):
            import warnings
            warnings.warn(
                "No exchange credentials configured. System will use public APIs only.",
                UserWarning
            )
    
    def get_exchange_config(self, exchange_name: str) -> Dict[str, Any]:
        """Get configuration for specific exchange"""
        base_config = {
            'enableRateLimit': True,
            'timeout': self.exchange.timeout_seconds * 1000,
            'rateLimit': int(1000 / self.exchange.requests_per_second)
        }
        
        if exchange_name.lower() == 'kraken':
            if self.exchange.kraken_api_key and self.exchange.kraken_secret:
                base_config.update({
                    'apiKey': self.exchange.kraken_api_key,
                    'secret': self.exchange.kraken_secret,
                    'sandbox': self.environment == 'development'
                })
        elif exchange_name.lower() == 'binance':
            if self.exchange.binance_api_key and self.exchange.binance_secret:
                base_config.update({
                    'apiKey': self.exchange.binance_api_key,
                    'secret': self.exchange.binance_secret,
                    'sandbox': self.environment == 'development'
                })
        
        return base_config
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == 'production'
    
    def export_config(self) -> Dict[str, Any]:
        """Export configuration for debugging (excludes secrets)"""
        config = self.dict()
        
        # Remove sensitive data
        sensitive_keys = ['api_key', 'secret', 'password', 'token', 'encryption_key']
        
        def remove_sensitive(obj, keys):
            if isinstance(obj, dict):
                return {
                    k: "***REDACTED***" if any(sk in k.lower() for sk in keys) else remove_sensitive(v, keys)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [remove_sensitive(item, keys) for item in obj]
            return obj
        
        return remove_sensitive(config, sensitive_keys)

# Global settings instance with lazy loading
_settings: Optional[AppSettings] = None

def get_settings() -> AppSettings:
    """Get global settings instance (singleton pattern with DI support)"""
    global _settings
    if _settings is None:
        _settings = AppSettings()
    return _settings

def set_settings(settings: AppSettings) -> None:
    """Set global settings instance (for testing/DI)"""
    global _settings
    _settings = settings

def reset_settings() -> None:
    """Reset global settings (for testing)"""
    global _settings
    _settings = None