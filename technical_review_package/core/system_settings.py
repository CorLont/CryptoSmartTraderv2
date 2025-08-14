#!/usr/bin/env python3
"""
System Settings - Enterprise Pydantic configuration management

Comprehensive system configuration using Pydantic v2 with cross-version compatibility,
consistent horizon notation, and robust device detection for production environments.
"""

import os
import platform
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Pydantic v2 compatibility with fallback handling
try:
    from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
    from pydantic_settings import BaseSettings, SettingsConfigDict

    PYDANTIC_V2 = True
except ImportError:
    try:
        # Fallback to Pydantic v1
        from pydantic import BaseModel, BaseSettings, Field, validator, root_validator

        PYDANTIC_V2 = False
    except ImportError:
        raise ImportError("Neither Pydantic v1 nor v2 is available")

# Device detection utilities
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from core.consolidated_logging_manager import get_consolidated_logger
except ImportError:
    import logging

    def get_consolidated_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)


logger = get_consolidated_logger("SystemSettings")


def detect_optimal_device() -> str:
    """
    Detect optimal compute device with robust GPU validation

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'
    """

    if not TORCH_AVAILABLE:
        logger.info("PyTorch not available, defaulting to CPU")
        return "cpu"

    # Check CUDA availability
    if torch.cuda.is_available():
        try:
            # Test actual CUDA functionality
            test_tensor = torch.tensor([1.0], device="cuda")
            test_result = test_tensor + 1
            logger.info(f"CUDA available with {torch.cuda.device_count()} GPU(s)")
            return "cuda"
        except Exception as e:
            logger.warning(f"CUDA advertised but not functional: {e}")

    # Check MPS (Apple Silicon) availability
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            # Test MPS functionality
            test_tensor = torch.tensor([1.0], device="mps")
            test_result = test_tensor + 1
            logger.info("MPS (Apple Silicon GPU) available")
            return "mps"
        except Exception as e:
            logger.warning(f"MPS advertised but not functional: {e}")

    logger.info("No GPU available, using CPU")
    return "cpu"


def normalize_horizon_notation(horizons: List[str]) -> List[str]:
    """
    Normalize horizon notation to consistent format

    Args:
        horizons: List of horizon strings in various formats

    Returns:
        Normalized horizon list in 'XXh' format
    """

    normalized = []

    for horizon in horizons:
        horizon = horizon.strip().lower()

        # Convert different notation formats to hours
        if horizon.endswith("h"):
            # Already in hour format (1h, 24h, etc.)
            normalized.append(horizon)
        elif horizon.endswith("d"):
            # Convert days to hours (7d -> 168h)
            days = int(horizon[:-1])
            hours = days * 24
            normalized.append(f"{hours}h")
        elif horizon.endswith("m"):
            # Convert minutes to hours (30m -> 0.5h, but keep as minutes for sub-hour)
            minutes = int(horizon[:-1])
            if minutes >= 60:
                hours = minutes // 60
                normalized.append(f"{hours}h")
            else:
                normalized.append(horizon)  # Keep minutes for sub-hour intervals
        elif horizon.endswith("w"):
            # Convert weeks to hours (1w -> 168h)
            weeks = int(horizon[:-1])
            hours = weeks * 24 * 7
            normalized.append(f"{hours}h")
        elif horizon.isdigit():
            # Assume hours if no unit specified
            normalized.append(f"{horizon}h")
        else:
            # Keep as-is if unrecognized format
            logger.warning(f"Unrecognized horizon format: {horizon}")
            normalized.append(horizon)

    return normalized


# Base settings class with version compatibility
if PYDANTIC_V2:

    class BaseSystemSettings(BaseSettings):
        """Base settings class for Pydantic v2"""

        model_config = SettingsConfigDict(
            env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
        )

        def to_dict(self) -> Dict[str, Any]:
            """Version-compatible dictionary conversion"""
            return self.model_dump()
else:

    class BaseSystemSettings(BaseSettings):
        """Base settings class for Pydantic v1"""

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False
            extra = "ignore"

        def to_dict(self) -> Dict[str, Any]:
            """Version-compatible dictionary conversion"""
            return self.dict()


class SystemSettings(BaseSystemSettings):
    """Core system configuration settings"""

    # Environment and deployment
    environment: str = Field(
        default="development", description="Environment: development, staging, production"
    )
    debug_mode: bool = Field(default=True, description="Enable debug logging and features")
    log_level: str = Field(default="INFO", description="Logging level")

    # System resources
    max_workers: int = Field(default=4, description="Maximum worker threads")
    memory_limit_gb: float = Field(default=8.0, description="Memory usage limit in GB")
    cpu_limit_percent: float = Field(default=80.0, description="CPU usage limit percentage")

    # Security and secrets
    secret_key: str = Field(default="", description="Application secret key")
    encryption_enabled: bool = Field(default=True, description="Enable data encryption")

    # Monitoring and health
    health_check_interval: int = Field(default=300, description="Health check interval in seconds")
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")
    prometheus_port: int = Field(default=8000, description="Prometheus metrics port")

    # Cross-version validation
    if PYDANTIC_V2:

        @field_validator("environment")
        @classmethod
        def validate_environment(cls, v):
            allowed = ["development", "staging", "production"]
            if v not in allowed:
                raise ValueError(f"Environment must be one of {allowed}")
            return v

        @field_validator("log_level")
        @classmethod
        def validate_log_level(cls, v):
            allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if v.upper() not in allowed:
                raise ValueError(f"Log level must be one of {allowed}")
            return v.upper()
    else:

        @validator("environment")
        def validate_environment(cls, v):
            allowed = ["development", "staging", "production"]
            if v not in allowed:
                raise ValueError(f"Environment must be one of {allowed}")
            return v

        @validator("log_level")
        def validate_log_level(cls, v):
            allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if v.upper() not in allowed:
                raise ValueError(f"Log level must be one of {allowed}")
            return v.upper()


class ExchangeSettings(BaseSystemSettings):
    """Exchange and trading configuration"""

    # Primary exchange settings
    primary_exchange: str = Field(default="kraken", description="Primary exchange for trading")
    exchange_timeout: int = Field(default=30, description="Exchange API timeout in seconds")
    rate_limit_requests: int = Field(default=100, description="Rate limit requests per minute")

    # Trading configuration
    trading_enabled: bool = Field(default=False, description="Enable live trading")
    paper_trading: bool = Field(default=True, description="Use paper trading mode")
    max_position_size: float = Field(default=1000.0, description="Maximum position size in USD")

    # API credentials (loaded from environment)
    kraken_api_key: str = Field(default="", description="Kraken API key")
    kraken_secret: str = Field(default="", description="Kraken API secret")
    binance_api_key: str = Field(default="", description="Binance API key")
    binance_secret: str = Field(default="", description="Binance API secret")


class MLSettings(BaseSystemSettings):
    """Machine learning configuration with consistent horizon notation"""

    # Model training
    model_training_enabled: bool = Field(default=True, description="Enable model training")
    auto_retrain_enabled: bool = Field(default=True, description="Enable automatic retraining")
    retrain_interval_hours: int = Field(default=24, description="Retraining interval in hours")

    # Prediction horizons - CONSISTENT NOTATION
    prediction_horizons: List[str] = Field(
        default=["1h", "24h", "168h", "720h"],  # 1h, 1d, 7d, 30d in consistent format
        description="Prediction horizons in consistent notation",
    )

    # Model configuration
    model_types: List[str] = Field(
        default=["xgboost", "neural", "ensemble"], description="Enabled model types"
    )

    # Device and compute - ROBUST GPU DETECTION
    torch_device: str = Field(
        default_factory=detect_optimal_device,
        description="PyTorch device: auto-detected based on availability",
    )

    batch_size: int = Field(default=32, description="Training batch size")
    max_epochs: int = Field(default=100, description="Maximum training epochs")
    early_stopping_patience: int = Field(default=10, description="Early stopping patience")

    # Model persistence
    model_save_interval: int = Field(default=3600, description="Model save interval in seconds")
    model_history_keep: int = Field(default=5, description="Number of model versions to keep")

    # Cross-version validation for horizons
    if PYDANTIC_V2:

        @field_validator("prediction_horizons")
        @classmethod
        def normalize_horizons(cls, v):
            return normalize_horizon_notation(v)

        @field_validator("torch_device")
        @classmethod
        def validate_device(cls, v):
            if v not in ["cpu", "cuda", "mps", "auto"]:
                logger.warning(f"Unusual device specified: {v}")
            return v
    else:

        @validator("prediction_horizons")
        def normalize_horizons(cls, v):
            return normalize_horizon_notation(v)

        @validator("torch_device")
        def validate_device(cls, v):
            if v not in ["cpu", "cuda", "mps", "auto"]:
                logger.warning(f"Unusual device specified: {v}")
            return v


class DataSettings(BaseSystemSettings):
    """Data management configuration"""

    # Data sources
    data_sources: List[str] = Field(
        default=["kraken", "binance", "coinbase"], description="Enabled data sources"
    )

    # Data retention
    raw_data_retention_days: int = Field(default=30, description="Raw data retention in days")
    processed_data_retention_days: int = Field(
        default=90, description="Processed data retention in days"
    )
    backup_retention_days: int = Field(default=365, description="Backup retention in days")

    # Data quality
    quality_check_enabled: bool = Field(default=True, description="Enable data quality checks")
    missing_data_threshold: float = Field(default=0.1, description="Missing data threshold (10%)")
    outlier_detection_enabled: bool = Field(default=True, description="Enable outlier detection")

    # Data directories
    data_root_dir: str = Field(default="data", description="Root data directory")
    cache_dir: str = Field(default="cache", description="Cache directory")
    backup_dir: str = Field(default="backups", description="Backup directory")

    # Performance
    data_workers: int = Field(default=2, description="Data processing workers")
    cache_size_mb: int = Field(default=1000, description="Cache size in MB")


class NotificationSettings(BaseSystemSettings):
    """Notification and alerting configuration"""

    # Notification channels
    email_enabled: bool = Field(default=False, description="Enable email notifications")
    slack_enabled: bool = Field(default=False, description="Enable Slack notifications")
    webhook_enabled: bool = Field(default=False, description="Enable webhook notifications")

    # Email configuration
    smtp_host: str = Field(default="", description="SMTP host")
    smtp_port: int = Field(default=587, description="SMTP port")
    smtp_username: str = Field(default="", description="SMTP username")
    smtp_password: str = Field(default="", description="SMTP password")

    # Alert thresholds
    error_threshold: int = Field(default=5, description="Error count threshold for alerts")
    latency_threshold_ms: int = Field(default=1000, description="Latency threshold in ms")
    memory_threshold_percent: float = Field(default=90.0, description="Memory usage threshold")


class APISettings(BaseSystemSettings):
    """API server configuration"""

    # Server configuration
    host: str = Field(default="0.0.0.0", description="API server host")
    port: int = Field(default=8000, description="API server port")
    workers: int = Field(default=1, description="API server workers")

    # Security
    cors_enabled: bool = Field(default=True, description="Enable CORS")
    cors_origins: List[str] = Field(default=["*"], description="CORS allowed origins")
    rate_limiting_enabled: bool = Field(default=True, description="Enable rate limiting")

    # API features
    docs_enabled: bool = Field(default=True, description="Enable API documentation")
    openapi_url: str = Field(default="/openapi.json", description="OpenAPI spec URL")

    # Authentication
    auth_enabled: bool = Field(default=False, description="Enable API authentication")
    jwt_secret: str = Field(default="", description="JWT secret key")
    token_expire_hours: int = Field(default=24, description="Token expiration in hours")


# Configuration factory functions
def get_system_settings() -> SystemSettings:
    """Get system configuration settings"""
    return SystemSettings()


def get_exchange_settings() -> ExchangeSettings:
    """Get exchange configuration settings"""
    return ExchangeSettings()


def get_ml_settings() -> MLSettings:
    """Get ML configuration settings"""
    return MLSettings()


def get_data_settings() -> DataSettings:
    """Get data configuration settings"""
    return DataSettings()


def get_notification_settings() -> NotificationSettings:
    """Get notification configuration settings"""
    return NotificationSettings()


def get_api_settings() -> APISettings:
    """Get API configuration settings"""
    return APISettings()


def get_all_settings() -> Dict[str, Any]:
    """
    Get all configuration settings with version-compatible serialization

    Returns:
        Dictionary with all settings categories
    """

    # Get all settings instances
    system = get_system_settings()
    exchange = get_exchange_settings()
    ml = get_ml_settings()
    data = get_data_settings()
    notification = get_notification_settings()
    api = get_api_settings()

    # Version-agnostic dictionary conversion
    def to_dict_safe(obj) -> Dict[str, Any]:
        """Safe dictionary conversion for both Pydantic v1 and v2"""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()  # Pydantic v2
        elif hasattr(obj, "dict"):
            return obj.dict()  # Pydantic v1
        else:
            # Fallback for unexpected types
            return vars(obj)

    return {
        "system": to_dict_safe(system),
        "exchange": to_dict_safe(exchange),
        "ml": to_dict_safe(ml),
        "data": to_dict_safe(data),
        "notification": to_dict_safe(notification),
        "api": to_dict_safe(api),
        "metadata": {
            "pydantic_version": "v2" if PYDANTIC_V2 else "v1",
            "torch_available": TORCH_AVAILABLE,
            "detected_device": detect_optimal_device(),
            "platform": platform.system(),
            "python_version": platform.python_version(),
        },
    }


def validate_all_settings() -> Dict[str, Any]:
    """
    Validate all settings and return validation report

    Returns:
        Validation report with errors and warnings
    """

    errors = []
    warnings = []

    try:
        # Test all settings instantiation
        system = get_system_settings()
        exchange = get_exchange_settings()
        ml = get_ml_settings()
        data = get_data_settings()
        notification = get_notification_settings()
        api = get_api_settings()

        # Validate ML settings
        if ml.torch_device == "cuda" and not TORCH_AVAILABLE:
            warnings.append("CUDA device specified but PyTorch not available")

        if ml.torch_device == "cuda" and TORCH_AVAILABLE and not torch.cuda.is_available():
            warnings.append("CUDA device specified but CUDA not available")

        # Validate horizon notation
        normalized_horizons = normalize_horizon_notation(ml.prediction_horizons)
        if normalized_horizons != ml.prediction_horizons:
            warnings.append(
                f"Horizons normalized: {ml.prediction_horizons} -> {normalized_horizons}"
            )

        # Validate directories
        for directory in [data.data_root_dir, data.cache_dir, data.backup_dir]:
            if not Path(directory).exists():
                warnings.append(f"Directory does not exist: {directory}")

        # Validate API credentials
        if exchange.trading_enabled and not exchange.kraken_api_key:
            errors.append("Trading enabled but no API credentials provided")

    except Exception as e:
        errors.append(f"Settings validation failed: {str(e)}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "timestamp": str(datetime.now()),
    }


def print_settings_summary():
    """Print a summary of current settings"""

    print("System Settings Summary")
    print("=" * 50)

    try:
        settings = get_all_settings()
        metadata = settings.get("metadata", {})

        print(f"Pydantic Version: {metadata.get('pydantic_version', 'unknown')}")
        print(f"PyTorch Available: {metadata.get('torch_available', False)}")
        print(f"Detected Device: {metadata.get('detected_device', 'unknown')}")
        print(f"Platform: {metadata.get('platform', 'unknown')}")

        # ML Settings
        ml_settings = settings.get("ml", {})
        print(f"\nML Configuration:")
        print(f"  Device: {ml_settings.get('torch_device', 'unknown')}")
        print(f"  Horizons: {ml_settings.get('prediction_horizons', [])}")
        print(f"  Models: {ml_settings.get('model_types', [])}")

        # System Settings
        system_settings = settings.get("system", {})
        print(f"\nSystem Configuration:")
        print(f"  Environment: {system_settings.get('environment', 'unknown')}")
        print(f"  Debug: {system_settings.get('debug_mode', False)}")
        print(f"  Workers: {system_settings.get('max_workers', 0)}")

        # Validation
        validation = validate_all_settings()
        print(f"\nValidation:")
        print(f"  Valid: {validation['valid']}")
        if validation["errors"]:
            print(f"  Errors: {len(validation['errors'])}")
        if validation["warnings"]:
            print(f"  Warnings: {len(validation['warnings'])}")

    except Exception as e:
        print(f"Error getting settings summary: {e}")


if __name__ == "__main__":
    # Test settings system
    print("Testing System Settings")

    # Show settings summary
    print_settings_summary()

    # Test device detection
    print(f"\nDevice Detection:")
    print(f"Optimal device: {detect_optimal_device()}")

    # Test horizon normalization
    test_horizons = ["1h", "4h", "1d", "7d", "30d", "168h"]
    normalized = normalize_horizon_notation(test_horizons)
    print(f"\nHorizon Normalization:")
    print(f"Input: {test_horizons}")
    print(f"Normalized: {normalized}")

    # Validation report
    validation = validate_all_settings()
    print(f"\nValidation Report:")
    print(f"Valid: {validation['valid']}")
    if validation["errors"]:
        for error in validation["errors"]:
            print(f"  ERROR: {error}")
    if validation["warnings"]:
        for warning in validation["warnings"]:
            print(f"  WARNING: {warning}")

    print("\n✅ SYSTEM SETTINGS TEST COMPLETE")
