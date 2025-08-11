"""
Enterprise Configuration Management with Fail-Fast Validation
Centralized Pydantic Settings with type validation and startup checks
"""

import os
import sys
from typing import Optional, List, Dict, Any
from pathlib import Path
from pydantic import BaseSettings, Field, validator, ValidationError
from pydantic.env_settings import SettingsSourceCallable
import logging


class Settings(BaseSettings):
    """Enterprise Configuration with Fail-Fast Validation"""
    
    # === Core System Settings ===
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    DEBUG_MODE: bool = Field(default=False, description="Enable debug mode")
    PERFORMANCE_MODE: bool = Field(default=True, description="Enable performance optimizations")
    
    # === Service Configuration ===
    DASHBOARD_PORT: int = Field(default=5000, description="Streamlit dashboard port")
    API_PORT: int = Field(default=8001, description="FastAPI service port") 
    METRICS_PORT: int = Field(default=8000, description="Prometheus metrics port")
    
    # === API Keys (Required for Production) ===
    KRAKEN_API_KEY: Optional[str] = Field(default=None, description="Kraken exchange API key")
    KRAKEN_SECRET: Optional[str] = Field(default=None, description="Kraken exchange secret")
    OPENAI_API_KEY: Optional[str] = Field(default=None, description="OpenAI API key for sentiment analysis")
    
    # === Data & Storage ===
    DATA_DIR: Path = Field(default=Path("data"), description="Data storage directory")
    CACHE_DIR: Path = Field(default=Path("cache"), description="Cache storage directory")
    MODELS_DIR: Path = Field(default=Path("models"), description="ML models directory")
    LOGS_DIR: Path = Field(default=Path("logs"), description="Logs directory")
    
    # === Trading Configuration ===
    CONFIDENCE_THRESHOLD: float = Field(default=0.8, description="Minimum confidence threshold (80%)")
    MAX_POSITIONS: int = Field(default=10, description="Maximum concurrent positions")
    RISK_LIMIT_PERCENT: float = Field(default=2.0, description="Maximum risk per trade (%)")
    
    # === Performance Settings ===
    CACHE_SIZE_MB: int = Field(default=500, description="Cache size limit in MB")
    UPDATE_INTERVAL_SECONDS: int = Field(default=5, description="Data update interval")
    MAX_WORKERS: int = Field(default=4, description="Maximum worker threads")
    
    # === Feature Toggles ===
    ENABLE_PROMETHEUS: bool = Field(default=True, description="Enable Prometheus metrics")
    ENABLE_SENTIMENT: bool = Field(default=True, description="Enable sentiment analysis")
    ENABLE_WHALE_DETECTION: bool = Field(default=True, description="Enable whale detection")
    ENABLE_ML_PREDICTIONS: bool = Field(default=True, description="Enable ML predictions")
    ENABLE_PAPER_TRADING: bool = Field(default=True, description="Enable paper trading mode")
    
    # === Security & Monitoring ===
    RATE_LIMIT_REQUESTS: int = Field(default=100, description="API rate limit per minute")
    HEALTH_CHECK_INTERVAL: int = Field(default=30, description="Health check interval (seconds)")
    LOG_ROTATION_DAYS: int = Field(default=30, description="Log retention days")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        validate_assignment = True
        
        @classmethod
        def customise_sources(
            cls,
            init_settings: SettingsSourceCallable,
            env_settings: SettingsSourceCallable,
            file_secret_settings: SettingsSourceCallable,
        ) -> tuple[SettingsSourceCallable, ...]:
            """Customize settings loading order: env vars > .env file > defaults"""
            return env_settings, init_settings
    
    # === Validators ===
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate logging level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()
    
    @validator("DASHBOARD_PORT", "API_PORT", "METRICS_PORT")
    def validate_ports(cls, v):
        """Validate port numbers"""
        if not 1024 <= v <= 65535:
            raise ValueError("Port must be between 1024 and 65535")
        return v
    
    @validator("CONFIDENCE_THRESHOLD")
    def validate_confidence_threshold(cls, v):
        """Validate confidence threshold"""
        if not 0.5 <= v <= 1.0:
            raise ValueError("CONFIDENCE_THRESHOLD must be between 0.5 and 1.0")
        return v
    
    @validator("RISK_LIMIT_PERCENT")
    def validate_risk_limit(cls, v):
        """Validate risk limit percentage"""
        if not 0.1 <= v <= 10.0:
            raise ValueError("RISK_LIMIT_PERCENT must be between 0.1 and 10.0")
        return v
    
    @validator("DATA_DIR", "CACHE_DIR", "MODELS_DIR", "LOGS_DIR")
    def validate_directories(cls, v):
        """Validate and create directories if needed"""
        path = Path(v)
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except Exception as e:
            raise ValueError(f"Cannot create directory {path}: {e}")
    
    def validate_startup_requirements(self) -> Dict[str, Any]:
        """Validate critical startup requirements"""
        issues = []
        warnings = []
        
        # Check API keys for production features
        if self.ENABLE_SENTIMENT and not self.OPENAI_API_KEY:
            warnings.append("OPENAI_API_KEY missing - sentiment analysis will be limited")
        
        if not self.KRAKEN_API_KEY or not self.KRAKEN_SECRET:
            issues.append("KRAKEN_API_KEY and KRAKEN_SECRET required for live trading")
        
        # Check directory permissions
        for dir_name, dir_path in [
            ("DATA_DIR", self.DATA_DIR),
            ("CACHE_DIR", self.CACHE_DIR), 
            ("MODELS_DIR", self.MODELS_DIR),
            ("LOGS_DIR", self.LOGS_DIR)
        ]:
            if not dir_path.exists():
                issues.append(f"{dir_name} does not exist: {dir_path}")
            elif not os.access(dir_path, os.W_OK):
                issues.append(f"{dir_name} is not writable: {dir_path}")
        
        # Check port availability
        import socket
        for port_name, port in [
            ("DASHBOARD_PORT", self.DASHBOARD_PORT),
            ("API_PORT", self.API_PORT),
            ("METRICS_PORT", self.METRICS_PORT)
        ]:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                if sock.connect_ex(('localhost', port)) == 0:
                    warnings.append(f"{port_name} {port} is already in use")
        
        # Check system resources
        import psutil
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 4:
            warnings.append(f"Low system memory: {memory_gb:.1f}GB (recommended: 8GB+)")
        
        disk_gb = psutil.disk_usage('.').free / (1024**3)
        if disk_gb < 10:
            warnings.append(f"Low disk space: {disk_gb:.1f}GB (recommended: 50GB+)")
        
        return {
            "critical_issues": issues,
            "warnings": warnings,
            "startup_ready": len(issues) == 0
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging"""
        return {
            "version": "2.0.0",
            "environment": "development" if self.DEBUG_MODE else "production",
            "services": {
                "dashboard": self.DASHBOARD_PORT,
                "api": self.API_PORT, 
                "metrics": self.METRICS_PORT
            },
            "features": {
                "prometheus": self.ENABLE_PROMETHEUS,
                "sentiment": self.ENABLE_SENTIMENT,
                "whale_detection": self.ENABLE_WHALE_DETECTION,
                "ml_predictions": self.ENABLE_ML_PREDICTIONS,
                "paper_trading": self.ENABLE_PAPER_TRADING
            },
            "thresholds": {
                "confidence": self.CONFIDENCE_THRESHOLD,
                "risk_limit": self.RISK_LIMIT_PERCENT,
                "max_positions": self.MAX_POSITIONS
            }
        }


def load_and_validate_settings() -> Settings:
    """Load and validate settings with fail-fast behavior"""
    try:
        # Load settings with Pydantic validation
        settings = Settings()
        
        # Perform startup validation
        validation_result = settings.validate_startup_requirements()
        
        # Log configuration summary
        logger = logging.getLogger(__name__)
        logger.info("Configuration loaded successfully")
        logger.info(f"Settings summary: {settings.get_summary()}")
        
        # Handle validation results
        if validation_result["warnings"]:
            for warning in validation_result["warnings"]:
                logger.warning(f"Configuration warning: {warning}")
        
        if validation_result["critical_issues"]:
            for issue in validation_result["critical_issues"]:
                logger.error(f"Configuration error: {issue}")
            
            logger.critical("Critical configuration issues detected - system cannot start")
            sys.exit(1)
        
        logger.info("✅ Configuration validation passed - system ready to start")
        return settings
        
    except ValidationError as e:
        print(f"❌ Configuration validation failed:")
        for error in e.errors():
            field = error.get("loc", ["unknown"])[0]
            message = error.get("msg", "validation error")
            print(f"  - {field}: {message}")
        
        print("\n💡 Check your .env file and ensure all required settings are properly configured")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)


# Global settings instance (lazy loaded)
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get validated settings instance (singleton pattern)"""
    global _settings
    if _settings is None:
        _settings = load_and_validate_settings()
    return _settings