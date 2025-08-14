#!/usr/bin/env python3
"""
FASE E - Production Configuration Management
Pydantic Settings with environment variable support and validation
"""

import os
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from enum import Enum

from pydantic import BaseSettings, Field, validator, SecretStr
from pydantic.env_settings import SettingsSourceCallable


class Environment(str, Enum):
    """Application environment enumeration"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseSettings(BaseSettings):
    """Database configuration settings"""
    
    # PostgreSQL settings
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_user: str = Field(default="cryptotrader", description="PostgreSQL username")
    postgres_password: SecretStr = Field(default="", description="PostgreSQL password")
    postgres_database: str = Field(default="cryptosmarttrader", description="PostgreSQL database name")
    postgres_ssl_mode: str = Field(default="prefer", description="PostgreSQL SSL mode")
    
    # Redis settings
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: Optional[SecretStr] = Field(default=None, description="Redis password")
    redis_database: int = Field(default=0, description="Redis database number")
    
    @property
    def postgres_url(self) -> str:
        """Generate PostgreSQL connection URL"""
        password_part = f":{self.postgres_password.get_secret_value()}" if self.postgres_password.get_secret_value() else ""
        return (
            f"postgresql://{self.postgres_user}{password_part}@"
            f"{self.postgres_host}:{self.postgres_port}/{self.postgres_database}"
            f"?sslmode={self.postgres_ssl_mode}"
        )
    
    @property
    def redis_url(self) -> str:
        """Generate Redis connection URL"""
        password_part = f":{self.redis_password.get_secret_value()}@" if self.redis_password else ""
        return f"redis://{password_part}{self.redis_host}:{self.redis_port}/{self.redis_database}"
    
    class Config:
        env_prefix = "CRYPTOSMARTTRADER_DB_"


class ExchangeSettings(BaseSettings):
    """Exchange API configuration settings"""
    
    # Kraken settings
    kraken_api_key: SecretStr = Field(default="", description="Kraken API key")
    kraken_secret: SecretStr = Field(default="", description="Kraken API secret")
    kraken_sandbox: bool = Field(default=True, description="Use Kraken sandbox")
    
    # Rate limiting
    kraken_rate_limit: int = Field(default=10, description="Kraken API rate limit (requests/second)")
    
    # Connection settings
    exchange_timeout: int = Field(default=30, description="Exchange API timeout (seconds)")
    exchange_retries: int = Field(default=3, description="Exchange API retry attempts")
    
    class Config:
        env_prefix = "CRYPTOSMARTTRADER_EXCHANGE_"


class AISettings(BaseSettings):
    """AI/ML service configuration settings"""
    
    # OpenAI settings
    openai_api_key: SecretStr = Field(default="", description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o", description="Default OpenAI model")
    openai_timeout: int = Field(default=60, description="OpenAI API timeout (seconds)")
    
    # Model settings
    model_cache_ttl: int = Field(default=3600, description="Model cache TTL (seconds)")
    ml_batch_size: int = Field(default=32, description="ML batch size")
    feature_cache_size: int = Field(default=10000, description="Feature cache size")
    
    class Config:
        env_prefix = "CRYPTOSMARTTRADER_AI_"


class SecuritySettings(BaseSettings):
    """Security configuration settings"""
    
    # Authentication
    secret_key: SecretStr = Field(default="", description="Application secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration: int = Field(default=3600, description="JWT expiration (seconds)")
    
    # API security
    api_key_header: str = Field(default="X-API-Key", description="API key header name")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per window")
    rate_limit_window: int = Field(default=60, description="Rate limit window (seconds)")
    
    # Encryption
    encryption_algorithm: str = Field(default="AES-256-GCM", description="Encryption algorithm")
    
    class Config:
        env_prefix = "CRYPTOSMARTTRADER_SECURITY_"


class MonitoringSettings(BaseSettings):
    """Monitoring and observability settings"""
    
    # Prometheus
    prometheus_port: int = Field(default=8000, description="Prometheus metrics port")
    prometheus_host: str = Field(default="0.0.0.0", description="Prometheus metrics host")
    
    # Health checks
    health_check_port: int = Field(default=8001, description="Health check API port")
    health_check_interval: int = Field(default=30, description="Health check interval (seconds)")
    
    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    log_format: str = Field(default="json", description="Log format (json|text)")
    log_file: Optional[str] = Field(default=None, description="Log file path")
    
    # Alerts
    alert_webhook_url: Optional[str] = Field(default=None, description="Alert webhook URL")
    alert_email_enabled: bool = Field(default=False, description="Enable email alerts")
    
    class Config:
        env_prefix = "CRYPTOSMARTTRADER_MONITORING_"


class TradingSettings(BaseSettings):
    """Trading configuration settings"""
    
    # Risk management
    max_daily_loss_usd: float = Field(default=5000.0, description="Maximum daily loss (USD)")
    max_daily_loss_percent: float = Field(default=5.0, description="Maximum daily loss (%)")
    max_drawdown_percent: float = Field(default=10.0, description="Maximum drawdown (%)")
    max_position_count: int = Field(default=10, description="Maximum position count")
    
    # Execution settings
    default_order_timeout: int = Field(default=300, description="Default order timeout (seconds)")
    max_slippage_bps: int = Field(default=50, description="Maximum slippage (basis points)")
    min_order_size_usd: float = Field(default=10.0, description="Minimum order size (USD)")
    
    # Signal settings
    signal_timeout_minutes: int = Field(default=30, description="Signal timeout (minutes)")
    min_signal_confidence: float = Field(default=0.7, description="Minimum signal confidence")
    
    class Config:
        env_prefix = "CRYPTOSMARTTRADER_TRADING_"


class CryptoSmartTraderSettings(BaseSettings):
    """Main application settings"""
    
    # Application metadata
    app_name: str = Field(default="CryptoSmartTrader V2", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Application environment")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=5000, description="Server port")
    workers: int = Field(default=1, description="Number of workers")
    
    # Data directories
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")
    models_dir: Path = Field(default=Path("models"), description="Models directory")
    exports_dir: Path = Field(default=Path("exports"), description="Exports directory")
    
    # Nested settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    exchanges: ExchangeSettings = Field(default_factory=ExchangeSettings)
    ai: AISettings = Field(default_factory=AISettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    trading: TradingSettings = Field(default_factory=TradingSettings)
    
    @validator('environment', pre=True)
    def validate_environment(cls, v):
        """Validate environment setting"""
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator('data_dir', 'logs_dir', 'models_dir', 'exports_dir', pre=True)
    def validate_directories(cls, v):
        """Validate and create directories"""
        path = Path(v)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == Environment.DEVELOPMENT
    
    def setup_logging(self) -> None:
        """Setup application logging"""
        log_config = {
            'level': getattr(logging, self.monitoring.log_level.value),
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
        
        if self.monitoring.log_file:
            log_config['filename'] = self.monitoring.log_file
        
        logging.basicConfig(**log_config)
        
        # Set third-party loggers to WARNING in production
        if self.is_production:
            for logger_name in ['urllib3', 'requests', 'ccxt', 'streamlit']:
                logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    class Config:
        env_prefix = "CRYPTOSMARTTRADER_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True


# Global settings instance
_settings: Optional[CryptoSmartTraderSettings] = None


def get_settings() -> CryptoSmartTraderSettings:
    """Get application settings singleton"""
    global _settings
    if _settings is None:
        _settings = CryptoSmartTraderSettings()
        _settings.setup_logging()
    return _settings


def reset_settings() -> None:
    """Reset settings singleton (for testing)"""
    global _settings
    _settings = None


# Development helper
if __name__ == "__main__":
    settings = get_settings()
    print(f"Environment: {settings.environment}")
    print(f"Debug: {settings.debug}")
    print(f"Database URL: {settings.database.postgres_url}")
    print(f"Redis URL: {settings.database.redis_url}")
    print(f"Data directory: {settings.data_dir}")