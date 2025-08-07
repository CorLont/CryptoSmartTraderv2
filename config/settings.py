# config/settings.py
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Dict, List, Optional
import os
from pathlib import Path


class CryptoConfig(BaseSettings):
    """
    Enhanced configuration using Pydantic BaseSettings with environment variable support.
    Follows Dutch implementation requirements for env-var overrides and secure config management.
    """
    
    # Application settings
    app_name: str = Field("CryptoSmartTrader V2", env="APP_NAME")
    debug_mode: bool = Field(False, env="DEBUG_MODE")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # Server configuration
    server_host: str = Field("0.0.0.0", env="SERVER_HOST")
    server_port: int = Field(5000, env="SERVER_PORT")
    
    # External API keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    
    # Exchange settings
    primary_exchange: str = Field("kraken", env="PRIMARY_EXCHANGE")
    supported_exchanges: List[str] = Field(
        default=["kraken", "binance", "kucoin", "huobi"],
        env="SUPPORTED_EXCHANGES"
    )
    
    # Data management
    data_cache_size: int = Field(1000, env="DATA_CACHE_SIZE")
    cache_ttl_minutes: int = Field(15, env="CACHE_TTL_MINUTES")
    coin_refresh_interval: int = Field(3600, env="COIN_REFRESH_INTERVAL")
    
    # Agent configuration
    agent_worker_threads: int = Field(6, env="AGENT_WORKER_THREADS")
    sentiment_analysis_enabled: bool = Field(True, env="SENTIMENT_ANALYSIS_ENABLED")
    ml_prediction_enabled: bool = Field(True, env="ML_PREDICTION_ENABLED")
    
    # Risk management
    max_position_size: float = Field(0.1, env="MAX_POSITION_SIZE")
    stop_loss_percentage: float = Field(0.02, env="STOP_LOSS_PERCENTAGE")
    take_profit_percentage: float = Field(0.05, env="TAKE_PROFIT_PERCENTAGE")
    
    # ML settings
    model_retrain_interval: int = Field(86400, env="MODEL_RETRAIN_INTERVAL")  # 24 hours
    prediction_horizons: List[str] = Field(
        default=["1h", "4h", "12h", "24h", "7d", "30d"],
        env="PREDICTION_HORIZONS"
    )
    
    # Health monitoring
    health_check_interval: int = Field(60, env="HEALTH_CHECK_INTERVAL")
    alert_threshold_score: float = Field(0.7, env="ALERT_THRESHOLD_SCORE")
    
    # Metrics and monitoring
    metrics_port: int = Field(8000, env="METRICS_PORT")
    enable_prometheus: bool = Field(True, env="ENABLE_PROMETHEUS")
    
    # Vault integration (optional)
    vault_addr: Optional[str] = Field(None, env="VAULT_ADDR")
    vault_token: Optional[str] = Field(None, env="VAULT_TOKEN")
    vault_enabled: bool = Field(False, env="VAULT_ENABLED")
    
    # Database settings (for future PostgreSQL integration)
    database_url: Optional[str] = Field(None, env="DATABASE_URL")
    
    # File paths
    data_directory: str = Field("data", env="DATA_DIRECTORY")
    models_directory: str = Field("models", env="MODELS_DIRECTORY")
    logs_directory: str = Field("logs", env="LOGS_DIRECTORY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.data_directory,
            self.models_directory,
            self.logs_directory,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @property
    def has_openai_key(self) -> bool:
        """Check if OpenAI API key is available"""
        return self.openai_api_key is not None and len(self.openai_api_key.strip()) > 0
    
    @property
    def vault_configured(self) -> bool:
        """Check if Vault is properly configured"""
        return (self.vault_enabled and 
                self.vault_addr is not None and 
                self.vault_token is not None)


# Global configuration instance
config = CryptoConfig()