#!/usr/bin/env python3
"""
Enterprise Configuration Management - Pydantic Settings
Centralized configuration with type validation, defaults, and startup logging
"""

import os
import logging
from typing import Optional, List, Literal
from pathlib import Path

try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, SecretStr, validator
    PYDANTIC_V2 = True
except ImportError:
    try:
        from pydantic import BaseSettings, Field, SecretStr, validator
        PYDANTIC_V2 = False
    except ImportError:
        raise ImportError("Pydantic is required for configuration management")


class Settings(BaseSettings):
    """
    Enterprise Configuration Settings
    Single source of truth for all environment variables with type validation
    """
    
    # === CORE APPLICATION ===
    APP_NAME: str = "CryptoSmartTrader V2"
    APP_VERSION: str = "2.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = Field(default="development")
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # === API CONFIGURATION ===
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001
    DASHBOARD_PORT: int = 5000
    METRICS_PORT: int = 8000
    
    # === EXCHANGE API KEYS ===
    KRAKEN_API_KEY: Optional[SecretStr] = None
    KRAKEN_API_SECRET: Optional[SecretStr] = None
    BINANCE_API_KEY: Optional[SecretStr] = None
    BINANCE_API_SECRET: Optional[SecretStr] = None
    
    # === AI/ML SERVICES ===
    OPENAI_API_KEY: Optional[SecretStr] = None
    
    # === DATABASE ===
    DATABASE_URL: str = "sqlite:///cryptotrader.db"
    REDIS_URL: Optional[str] = None
    
    # === MONITORING & METRICS ===
    ENABLE_PROMETHEUS: bool = True
    PROMETHEUS_PORT: int = 8000
    PROMETHEUS_HOST: str = "0.0.0.0"
    
    # === LOGGING ===
    LOG_FORMAT: Literal["json", "text"] = "json"
    LOG_FILE: Optional[str] = None
    ENABLE_FILE_LOGGING: bool = False
    
    # === TRADING ===
    PAPER_TRADING: bool = True
    TRADING_ENABLED: bool = False
    MAX_POSITION_SIZE: int = 1000
    RISK_PERCENTAGE: float = 0.02
    
    # === ML/AI ===
    MODEL_PATH: str = "./models"
    TORCH_DEVICE: str = "auto"
    ENABLE_GPU: bool = True
    MODEL_TRAINING_ENABLED: bool = True
    AUTO_RETRAIN_ENABLED: bool = True
    RETRAIN_INTERVAL_HOURS: int = 24
    
    # === RATE LIMITING ===
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 100
    
    # === SECURITY ===
    SECRET_KEY: SecretStr = Field(default="dev-secret-key-change-in-production")
    JWT_SECRET_KEY: Optional[SecretStr] = None
    ENCRYPTION_ENABLED: bool = True
    
    # === EMAIL NOTIFICATIONS ===
    EMAIL_ENABLED: bool = False
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USERNAME: Optional[str] = None
    SMTP_PASSWORD: Optional[SecretStr] = None
    
    # === DEVELOPMENT ===
    MOCK_EXCHANGES: bool = False
    DEMO_MODE: bool = False
    ENABLE_CORS: bool = True
    
    # === DATA RETENTION ===
    DATA_RETENTION_DAYS: int = 30
    BACKUP_ENABLED: bool = True
    BACKUP_RETENTION_DAYS: int = 365
    
    # === FEATURE FLAGS ===
    ENABLE_SENTIMENT_ANALYSIS: bool = True
    ENABLE_WHALE_DETECTION: bool = True
    ENABLE_TECHNICAL_ANALYSIS: bool = True
    ENABLE_NEWS_ANALYSIS: bool = True
    
    if PYDANTIC_V2:
        model_config = {
            "env_file": ".env",
            "env_file_encoding": "utf-8",
            "case_sensitive": True,
            "extra": "ignore",
            "validate_assignment": True
        }
    else:
        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = True
            extra = "ignore"
            validate_assignment = True
    
    @validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Validate environment setting"""
        if v not in ["development", "staging", "production"]:
            raise ValueError("ENVIRONMENT must be development, staging, or production")
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    @validator("API_PORT", "DASHBOARD_PORT", "METRICS_PORT", "PROMETHEUS_PORT")
    def validate_ports(cls, v):
        """Validate port numbers"""
        if not (1024 <= v <= 65535):
            raise ValueError("Port must be between 1024 and 65535")
        return v
    
    @validator("RISK_PERCENTAGE")
    def validate_risk_percentage(cls, v):
        """Validate risk percentage"""
        if not (0.0 < v <= 1.0):
            raise ValueError("RISK_PERCENTAGE must be between 0.0 and 1.0")
        return v
    
    @validator("TRADING_ENABLED")
    def validate_trading_safety(cls, v, values):
        """Ensure trading safety in development"""
        if v and values.get("ENVIRONMENT") == "development":
            raise ValueError("TRADING_ENABLED cannot be True in development environment")
        return v
    
    @validator("DATABASE_URL")
    def validate_database_url(cls, v):
        """Validate database URL format"""
        if not v.startswith(("sqlite://", "postgresql://", "mysql://", "redis://")):
            raise ValueError("DATABASE_URL must start with sqlite://, postgresql://, mysql://, or redis://")
        return v
    
    def get_secret_value(self, secret: Optional[SecretStr]) -> Optional[str]:
        """Safely get secret value"""
        return secret.get_secret_value() if secret else None
    
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.ENVIRONMENT == "production"
    
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.ENVIRONMENT == "development"
    
    def validate_required_secrets(self) -> List[str]:
        """Validate required secrets for the current environment"""
        missing_secrets = []
        
        # Always required
        if not self.get_secret_value(self.OPENAI_API_KEY):
            missing_secrets.append("OPENAI_API_KEY")
        
        # Required for production
        if self.is_production():
            if not self.get_secret_value(self.KRAKEN_API_KEY):
                missing_secrets.append("KRAKEN_API_KEY")
            if not self.get_secret_value(self.KRAKEN_API_SECRET):
                missing_secrets.append("KRAKEN_API_SECRET")
            if not self.get_secret_value(self.JWT_SECRET_KEY):
                missing_secrets.append("JWT_SECRET_KEY")
        
        return missing_secrets
    
    def get_startup_config_summary(self) -> dict:
        """Get non-sensitive configuration summary for startup logging"""
        return {
            "app": {
                "name": self.APP_NAME,
                "version": self.APP_VERSION,
                "environment": self.ENVIRONMENT,
                "debug": self.DEBUG
            },
            "services": {
                "api_host": self.API_HOST,
                "api_port": self.API_PORT,
                "dashboard_port": self.DASHBOARD_PORT,
                "metrics_port": self.METRICS_PORT
            },
            "features": {
                "prometheus": self.ENABLE_PROMETHEUS,
                "paper_trading": self.PAPER_TRADING,
                "trading_enabled": self.TRADING_ENABLED,
                "gpu_enabled": self.ENABLE_GPU,
                "cors": self.ENABLE_CORS
            },
            "ml_features": {
                "sentiment_analysis": self.ENABLE_SENTIMENT_ANALYSIS,
                "whale_detection": self.ENABLE_WHALE_DETECTION,
                "technical_analysis": self.ENABLE_TECHNICAL_ANALYSIS,
                "news_analysis": self.ENABLE_NEWS_ANALYSIS
            },
            "integrations": {
                "kraken_configured": bool(self.KRAKEN_API_KEY),
                "binance_configured": bool(self.BINANCE_API_KEY),
                "openai_configured": bool(self.OPENAI_API_KEY),
                "redis_configured": bool(self.REDIS_URL),
                "email_configured": self.EMAIL_ENABLED
            }
        }


# Global settings instance with startup validation
def create_settings() -> Settings:
    """Create and validate settings instance"""
    try:
        settings = Settings()
        
        # Validate required secrets
        missing_secrets = settings.validate_required_secrets()
        if missing_secrets:
            if settings.is_production():
                raise ValueError(f"Missing required secrets in production: {missing_secrets}")
            else:
                logging.warning(f"Missing optional secrets in {settings.ENVIRONMENT}: {missing_secrets}")
        
        return settings
    except Exception as e:
        logging.error(f"Configuration validation failed: {e}")
        raise


def log_startup_config(settings: Settings):
    """Log startup configuration (non-sensitive data only)"""
    config_summary = settings.get_startup_config_summary()
    
    logger = logging.getLogger(__name__)
    logger.info("=== CryptoSmartTrader V2 Configuration ===")
    logger.info(f"Environment: {config_summary['app']['environment']}")
    logger.info(f"Debug Mode: {config_summary['app']['debug']}")
    logger.info(f"API Server: {config_summary['services']['api_host']}:{config_summary['services']['api_port']}")
    logger.info(f"Dashboard: Port {config_summary['services']['dashboard_port']}")
    logger.info(f"Metrics: Port {config_summary['services']['metrics_port']}")
    
    logger.info("=== Features Enabled ===")
    for feature, enabled in config_summary['features'].items():
        status = "✓" if enabled else "✗"
        logger.info(f"{status} {feature.replace('_', ' ').title()}: {enabled}")
    
    logger.info("=== ML Features ===")
    for feature, enabled in config_summary['ml_features'].items():
        status = "✓" if enabled else "✗"
        logger.info(f"{status} {feature.replace('_', ' ').title()}: {enabled}")
    
    logger.info("=== Integrations ===")
    for integration, configured in config_summary['integrations'].items():
        status = "✓" if configured else "✗"
        logger.info(f"{status} {integration.replace('_', ' ').title()}: {configured}")
    
    logger.info("=== Configuration Complete ===")


# Global settings instance
settings = create_settings()

# Backward compatibility functions
def get_env(key: str, default: str = "") -> str:
    """Backward compatibility for os.environ.get()"""
    return getattr(settings, key, default)

def require_env(key: str) -> str:
    """Backward compatibility for required env vars"""
    value = getattr(settings, key, None)
    if value is None:
        raise ValueError(f"Required environment variable {key} is not set")
    return value if not isinstance(value, SecretStr) else value.get_secret_value()


if __name__ == "__main__":
    # Test configuration
    import json
    
    print("=== Testing Configuration ===")
    test_settings = create_settings()
    
    print(f"Environment: {test_settings.ENVIRONMENT}")
    print(f"Debug: {test_settings.DEBUG}")
    print(f"API Port: {test_settings.API_PORT}")
    print(f"Features enabled: {sum(1 for f in [test_settings.ENABLE_PROMETHEUS, test_settings.ENABLE_SENTIMENT_ANALYSIS, test_settings.ENABLE_WHALE_DETECTION])}")
    
    missing = test_settings.validate_required_secrets()
    if missing:
        print(f"Missing secrets: {missing}")
    else:
        print("All required secrets configured")
    
    print("\n=== Startup Config Summary ===")
    summary = test_settings.get_startup_config_summary()
    print(json.dumps(summary, indent=2))