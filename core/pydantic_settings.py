#!/usr/bin/env python3
"""
Pydantic Settings - P1 Config & Secrets validation
Enterprise environment configuration with Pydantic validation instead of direct os.environ access
"""

import os
from typing import Optional, List, Literal
from pathlib import Path

try:
    # Pydantic v2
    from pydantic import BaseSettings, Field, field_validator
    from pydantic_settings import BaseSettings as V2BaseSettings
    from pydantic import SecretStr

    PYDANTIC_V2 = True
except ImportError:
    try:
        # Pydantic v1
        from pydantic import BaseSettings, Field, validator, SecretStr

        PYDANTIC_V2 = False
    except ImportError:
        # Fallback for minimal environment
        class BaseSettings:
            pass

        class Field:
            def __init__(self, *args, **kwargs):
                pass

        class SecretStr(str):
            def get_secret_value(self):
                return str(self)

        PYDANTIC_V2 = False


class CryptoTraderSettings(V2BaseSettings if PYDANTIC_V2 else BaseSettings):
    """
    P1 Enterprise Configuration with Pydantic validation

    Replaces direct os.environ[...] access with validated settings.
    All environment variables have defaults and validation.
    """

    # === REQUIRED API CREDENTIALS ===
    kraken_api_key: SecretStr = Field(
        default="", env="KRAKEN_API_KEY", description="Kraken exchange API key"
    )
    kraken_secret: SecretStr = Field(
        default="", env="KRAKEN_SECRET", description="Kraken exchange secret key"
    )
    openai_api_key: SecretStr = Field(
        default="", env="OPENAI_API_KEY", description="OpenAI API key for AI analysis"
    )

    # === OPTIONAL EXCHANGE APIS ===
    binance_api_key: Optional[SecretStr] = Field(
        default=None, env="BINANCE_API_KEY", description="Binance exchange API key"
    )
    binance_secret: Optional[SecretStr] = Field(
        default=None, env="BINANCE_SECRET", description="Binance exchange secret"
    )

    # === APPLICATION CONFIGURATION ===
    environment: Literal["development", "staging", "production"] = Field(
        default="development", env="ENVIRONMENT", description="Application environment"
    )
    debug: bool = Field(default=True, env="DEBUG", description="Enable debug mode")
    secret_key: SecretStr = Field(
        default="dev-secret-key-change-in-production",
        env="SECRET_KEY",
        description="Application secret key",
    )

    # === MULTI-SERVICE PORTS ===
    dashboard_port: int = Field(
        default=5000,
        env="DASHBOARD_PORT",
        ge=1024,
        le=65535,
        description="Streamlit dashboard port",
    )
    api_port: int = Field(
        default=8001, env="API_PORT", ge=1024, le=65535, description="FastAPI service port"
    )
    metrics_port: int = Field(
        default=8000, env="METRICS_PORT", ge=1024, le=65535, description="Prometheus metrics port"
    )

    # === DATABASE CONFIGURATION ===
    database_url: str = Field(
        default="sqlite:///cryptotrader.db",
        env="DATABASE_URL",
        description="Database connection URL",
    )

    # === MONITORING & LOGGING ===
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", env="LOG_LEVEL", description="Logging level"
    )
    log_format: Literal["json", "text"] = Field(
        default="json", env="LOG_FORMAT", description="Log output format"
    )
    prometheus_enabled: bool = Field(
        default=True, env="PROMETHEUS_ENABLED", description="Enable Prometheus metrics"
    )

    # === TRADING CONFIGURATION ===
    paper_trading: bool = Field(
        default=True, env="PAPER_TRADING", description="Use paper trading (no real money)"
    )
    trading_enabled: bool = Field(
        default=False, env="TRADING_ENABLED", description="Enable actual trading"
    )
    max_position_size: int = Field(
        default=1000, env="MAX_POSITION_SIZE", gt=0, description="Maximum position size in USD"
    )
    risk_percentage: float = Field(
        default=0.02, env="RISK_PERCENTAGE", gt=0.0, le=1.0, description="Risk percentage per trade"
    )

    # === ML CONFIGURATION ===
    model_training_enabled: bool = Field(
        default=True, env="MODEL_TRAINING_ENABLED", description="Enable ML model training"
    )
    auto_retrain_enabled: bool = Field(
        default=True, env="AUTO_RETRAIN_ENABLED", description="Enable automatic model retraining"
    )
    retrain_interval_hours: int = Field(
        default=24,
        env="RETRAIN_INTERVAL_HOURS",
        gt=0,
        description="Model retraining interval in hours",
    )
    torch_device: str = Field(
        default="auto", env="TORCH_DEVICE", description="PyTorch device (auto, cpu, cuda)"
    )

    # === SECURITY CONFIGURATION ===
    encryption_enabled: bool = Field(
        default=True, env="ENCRYPTION_ENABLED", description="Enable encryption for sensitive data"
    )
    jwt_secret_key: SecretStr = Field(
        default="jwt-secret-change-in-production",
        env="JWT_SECRET_KEY",
        description="JWT token secret key",
    )
    token_expire_hours: int = Field(
        default=24, env="TOKEN_EXPIRE_HOURS", gt=0, description="JWT token expiration in hours"
    )

    # === NOTIFICATIONS ===
    email_enabled: bool = Field(
        default=False, env="EMAIL_ENABLED", description="Enable email notifications"
    )
    smtp_host: Optional[str] = Field(default=None, env="SMTP_HOST", description="SMTP server host")
    smtp_port: int = Field(
        default=587, env="SMTP_PORT", ge=1, le=65535, description="SMTP server port"
    )
    smtp_username: Optional[str] = Field(
        default=None, env="SMTP_USERNAME", description="SMTP username"
    )
    smtp_password: Optional[SecretStr] = Field(
        default=None, env="SMTP_PASSWORD", description="SMTP password"
    )

    # === RATE LIMITING ===
    rate_limit_enabled: bool = Field(
        default=True, env="RATE_LIMIT_ENABLED", description="Enable API rate limiting"
    )
    rate_limit_requests_per_minute: int = Field(
        default=100,
        env="RATE_LIMIT_REQUESTS_PER_MINUTE",
        gt=0,
        description="Rate limit requests per minute",
    )

    # === DATA CONFIGURATION ===
    data_retention_days: int = Field(
        default=30, env="DATA_RETENTION_DAYS", gt=0, description="Data retention period in days"
    )
    backup_enabled: bool = Field(
        default=True, env="BACKUP_ENABLED", description="Enable automatic backups"
    )

    # === DEVELOPMENT SETTINGS ===
    mock_exchanges: bool = Field(
        default=False, env="MOCK_EXCHANGES", description="Use mock exchange data for development"
    )
    demo_mode: bool = Field(default=False, env="DEMO_MODE", description="Run in demonstration mode")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        validate_assignment = True
        arbitrary_types_allowed = True

        # Custom validation messages
        fields = {
            "kraken_api_key": {"description": "Kraken API key is required for production"},
            "openai_api_key": {"description": "OpenAI API key is required for AI features"},
        }

    if PYDANTIC_V2:

        @field_validator("environment")
        @classmethod
        def validate_environment(cls, v):
            """Validate environment setting"""
            if v not in ["development", "staging", "production"]:
                raise ValueError("Environment must be development, staging, or production")
            return v

        @field_validator("database_url")
        @classmethod
        def validate_database_url(cls, v):
            """Validate database URL format"""
            if not v.startswith(("sqlite://", "postgresql://", "mysql://")):
                raise ValueError(
                    "Database URL must start with sqlite://, postgresql://, or mysql://"
                )
            return v

        @field_validator("torch_device")
        @classmethod
        def validate_torch_device(cls, v):
            """Validate PyTorch device setting"""
            valid_devices = ["auto", "cpu", "cuda", "mps"]
            if v not in valid_devices and not v.startswith("cuda:"):
                raise ValueError(f"Invalid torch device. Must be one of {valid_devices} or cuda:N")
            return v
    else:

        @validator("environment")
        def validate_environment(cls, v):
            """Validate environment setting"""
            if v not in ["development", "staging", "production"]:
                raise ValueError("Environment must be development, staging, or production")
            return v

        @validator("database_url")
        def validate_database_url(cls, v):
            """Validate database URL format"""
            if not v.startswith(("sqlite://", "postgresql://", "mysql://")):
                raise ValueError(
                    "Database URL must start with sqlite://, postgresql://, or mysql://"
                )
            return v

        @validator("torch_device")
        def validate_torch_device(cls, v):
            """Validate PyTorch device setting"""
            valid_devices = ["auto", "cpu", "cuda", "mps"]
            if v not in valid_devices and not v.startswith("cuda:"):
                raise ValueError(f"Invalid torch device. Must be one of {valid_devices} or cuda:N")
            return v

        @validator("trading_enabled")
        def validate_trading_safety(cls, v, values):
            """Ensure trading safety - never enable trading in development"""
            if v and values.get("environment") == "development":
                raise ValueError("Trading cannot be enabled in development environment")
            return v

    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == "development"

    def get_secret_value(self, secret: SecretStr) -> str:
        """Safely get secret value"""
        return secret.get_secret_value() if secret else ""

    def validate_required_secrets(self) -> List[str]:
        """Validate that required secrets are provided"""
        missing_secrets = []

        if not self.get_secret_value(self.kraken_api_key):
            missing_secrets.append("KRAKEN_API_KEY")

        if not self.get_secret_value(self.kraken_secret):
            missing_secrets.append("KRAKEN_SECRET")

        if not self.get_secret_value(self.openai_api_key):
            missing_secrets.append("OPENAI_API_KEY")

        return missing_secrets

    def to_dict(self, include_secrets: bool = False) -> dict:
        """Convert settings to dictionary"""
        data = self.dict()

        if not include_secrets:
            # Mask secret values
            for key, value in data.items():
                if isinstance(value, SecretStr):
                    data[key] = "***MASKED***"

        return data


# Global settings instance
settings = CryptoTraderSettings()


# Utility functions for backward compatibility
def get_env(key: str, default: str = "") -> str:
    """
    Replacement for os.environ.get() with validation

    Args:
        key: Environment variable key
        default: Default value if not found

    Returns:
        Environment variable value
    """
    return os.getenv(key, default)


def require_env(key: str) -> str:
    """
    Require environment variable with clear error message

    Args:
        key: Environment variable key

    Returns:
        Environment variable value

    Raises:
        ValueError: If environment variable is not set
    """
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Required environment variable {key} is not set")
    return value


# Validation on import
if __name__ != "__main__":
    missing_secrets = settings.validate_required_secrets()
    if missing_secrets and settings.is_production():
        raise ValueError(f"Missing required secrets in production: {missing_secrets}")
