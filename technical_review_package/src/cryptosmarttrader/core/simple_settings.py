#!/usr/bin/env python3
"""
Simple Settings - P1 Config & Secrets validation
Enterprise environment configuration replacing direct os.environ access
"""

import os
from typing import Optional, Dict, Any


class CryptoTraderSettings:
    """
    P1 Enterprise Configuration with validation

    Replaces direct os.environ[...] access with validated settings.
    All environment variables have defaults and validation.
    """

    def __init__(self):
        """Initialize settings from environment variables"""

        # === REQUIRED API CREDENTIALS ===
        self.kraken_api_key = os.getenv("KRAKEN_API_KEY", "")
        self.kraken_secret = os.getenv("KRAKEN_SECRET", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")

        # === OPTIONAL EXCHANGE APIS ===
        self.binance_api_key = os.getenv("BINANCE_API_KEY")
        self.binance_secret = os.getenv("BINANCE_SECRET")

        # === APPLICATION CONFIGURATION ===
        self.environment = os.getenv("ENVIRONMENT", "development")
        self.debug = os.getenv("DEBUG", "true").lower() == "true"
        self.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

        # === MULTI-SERVICE PORTS ===
        self.dashboard_port = int(os.getenv("DASHBOARD_PORT", "5000"))
        self.api_port = int(os.getenv("API_PORT", "8001"))
        self.metrics_port = int(os.getenv("METRICS_PORT", "8000"))

        # === DATABASE CONFIGURATION ===
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///cryptotrader.db")

        # === MONITORING & LOGGING ===
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        self.log_format = os.getenv("LOG_FORMAT", "json")
        self.prometheus_enabled = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"

        # === TRADING CONFIGURATION ===
        self.paper_trading = os.getenv("PAPER_TRADING", "true").lower() == "true"
        self.trading_enabled = os.getenv("TRADING_ENABLED", "false").lower() == "true"
        self.max_position_size = int(os.getenv("MAX_POSITION_SIZE", "1000"))
        self.risk_percentage = float(os.getenv("RISK_PERCENTAGE", "0.02"))

        # === ML CONFIGURATION ===
        self.model_training_enabled = os.getenv("MODEL_TRAINING_ENABLED", "true").lower() == "true"
        self.auto_retrain_enabled = os.getenv("AUTO_RETRAIN_ENABLED", "true").lower() == "true"
        self.retrain_interval_hours = int(os.getenv("RETRAIN_INTERVAL_HOURS", "24"))
        self.torch_device = os.getenv("TORCH_DEVICE", "auto")

        # === SECURITY CONFIGURATION ===
        self.encryption_enabled = os.getenv("ENCRYPTION_ENABLED", "true").lower() == "true"
        self.jwt_secret_key = os.getenv("JWT_SECRET_KEY", "jwt-secret-change-in-production")
        self.token_expire_hours = int(os.getenv("TOKEN_EXPIRE_HOURS", "24"))

        # Validate critical settings
        self._validate_settings()

    def _validate_settings(self):
        """Validate critical settings"""

        # Validate environment
        if self.environment not in ["development", "staging", "production"]:
            raise ValueError("Environment must be development, staging, or production")

        # Validate database URL
        if not self.database_url.startswith(("sqlite://", "postgresql://", "mysql://")):
            raise ValueError("Database URL must start with sqlite://, postgresql://, or mysql://")

        # Validate ports
        for port_name, port_value in [
            ("dashboard_port", self.dashboard_port),
            ("api_port", self.api_port),
            ("metrics_port", self.metrics_port),
        ]:
            if not (1024 <= port_value <= 65535):
                raise ValueError(f"{port_name} must be between 1024 and 65535")

        # Validate torch device
        valid_devices = ["auto", "cpu", "cuda", "mps"]
        if self.torch_device not in valid_devices and not self.torch_device.startswith("cuda:"):
            raise ValueError(f"Invalid torch device. Must be one of {valid_devices} or cuda:N")

        # Trading safety validation
        if self.trading_enabled and self.environment == "development":
            raise ValueError("Trading cannot be enabled in development environment")

        # Risk percentage validation
        if not (0.0 < self.risk_percentage <= 1.0):
            raise ValueError("Risk percentage must be between 0.0 and 1.0")

    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == "development"

    def validate_required_secrets(self) -> list:
        """Validate that required secrets are provided"""
        missing_secrets = []

        if not self.kraken_api_key:
            missing_secrets.append("KRAKEN_API_KEY")

        if not self.kraken_secret:
            missing_secrets.append("KRAKEN_SECRET")

        if not self.openai_api_key:
            missing_secrets.append("OPENAI_API_KEY")

        return missing_secrets

    def to_dict(self, include_secrets: bool = False) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        data = {}

        for key, value in self.__dict__.items():
            if not include_secrets and any(
                secret in key.lower() for secret in ["key", "secret", "password"]
            ):
                data[key] = "***MASKED***"
            else:
                data[key] = value

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
