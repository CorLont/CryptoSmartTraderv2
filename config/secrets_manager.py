"""
Enterprise Secrets Management voor CryptoSmartTrader V2
Centraal beheer van API keys, environment variabelen en veilige configuratie
"""

import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
import warnings
from .security import security_manager

logger = logging.getLogger(__name__)


@dataclass
class SecretsConfig:
    """Centralized secrets configuration"""
    
    # Exchange API credentials
    kraken_api_key: Optional[str] = None
    kraken_secret: Optional[str] = None
    binance_api_key: Optional[str] = None
    binance_secret: Optional[str] = None
    
    # AI/ML API keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    
    # Notification services
    slack_bot_token: Optional[str] = None
    slack_channel_id: Optional[str] = None
    sendgrid_api_key: Optional[str] = None
    
    # Database credentials
    database_url: Optional[str] = None
    
    # Application security
    jwt_secret_key: Optional[str] = None
    api_secret_key: Optional[str] = None
    
    # Environment configuration
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    
    # Trading configuration
    trading_mode: str = "paper"
    initial_portfolio_value: float = 100000.0
    max_daily_loss_percent: float = 5.0
    max_drawdown_percent: float = 10.0
    
    # System ports
    dashboard_port: int = 5000
    metrics_port: int = 8000
    health_port: int = 8001
    
    # Required secrets for specific modes
    _required_secrets: Dict[str, list] = field(default_factory=lambda: {
        "production": [
            "kraken_api_key", "kraken_secret", "openai_api_key", 
            "jwt_secret_key", "api_secret_key"
        ],
        "development": [
            "kraken_api_key", "kraken_secret", "openai_api_key"
        ],
        "testing": []
    })


class SecretsManager:
    """Enterprise secrets management met security validatie"""
    
    def __init__(self):
        self.config = SecretsConfig()
        self._load_from_environment()
        self._validate_configuration()
    
    def _load_from_environment(self) -> None:
        """Load alle secrets uit environment variabelen"""
        
        # Exchange credentials
        self.config.kraken_api_key = os.getenv("KRAKEN_API_KEY")
        self.config.kraken_secret = os.getenv("KRAKEN_SECRET")
        self.config.binance_api_key = os.getenv("BINANCE_API_KEY") 
        self.config.binance_secret = os.getenv("BINANCE_SECRET")
        
        # AI API keys
        self.config.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.config.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.config.gemini_api_key = os.getenv("GEMINI_API_KEY")
        
        # Notification services
        self.config.slack_bot_token = os.getenv("SLACK_BOT_TOKEN")
        self.config.slack_channel_id = os.getenv("SLACK_CHANNEL_ID")
        self.config.sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
        
        # Database
        self.config.database_url = os.getenv("DATABASE_URL")
        
        # Security
        self.config.jwt_secret_key = os.getenv("JWT_SECRET_KEY")
        self.config.api_secret_key = os.getenv("API_SECRET_KEY")
        
        # Environment
        self.config.environment = os.getenv("ENVIRONMENT", "development")
        self.config.debug = os.getenv("DEBUG", "true").lower() == "true"
        self.config.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Trading
        self.config.trading_mode = os.getenv("TRADING_MODE", "paper")
        self.config.initial_portfolio_value = float(os.getenv("INITIAL_PORTFOLIO_VALUE", "100000.0"))
        self.config.max_daily_loss_percent = float(os.getenv("MAX_DAILY_LOSS_PERCENT", "5.0"))
        self.config.max_drawdown_percent = float(os.getenv("MAX_DRAWDOWN_PERCENT", "10.0"))
        
        # Ports
        self.config.dashboard_port = int(os.getenv("DASHBOARD_PORT", "5000"))
        self.config.metrics_port = int(os.getenv("PROMETHEUS_PORT", "8000"))
        self.config.health_port = int(os.getenv("HEALTH_CHECK_PORT", "8001"))
    
    def _validate_configuration(self) -> None:
        """Valideer secrets en log security events"""
        
        environment = self.config.environment
        required_secrets = self.config._required_secrets.get(environment, [])
        
        missing_secrets = []
        
        for secret_name in required_secrets:
            secret_value = getattr(self.config, secret_name, None)
            
            if not secret_value:
                missing_secrets.append(secret_name)
            else:
                # Validate API key format
                if "api_key" in secret_name or "secret" in secret_name:
                    if not security_manager.validate_input(secret_value, "api_key"):
                        logger.warning(f"Invalid format for {secret_name}")
                        missing_secrets.append(secret_name)
        
        if missing_secrets:
            error_msg = f"Missing required secrets for {environment} environment: {missing_secrets}"
            logger.error(error_msg)
            
            security_manager.log_security_event(
                "MISSING_SECRETS",
                {
                    "environment": environment,
                    "missing_count": len(missing_secrets),
                    "secrets": [security_manager.hash_sensitive_data(s) for s in missing_secrets]
                },
                "ERROR"
            )
            
            if environment == "production":
                raise ValueError(error_msg)
            else:
                warnings.warn(f"Development mode: {error_msg}")
    
    def get_exchange_credentials(self, exchange: str) -> Dict[str, str]:
        """Haal exchange credentials op met security validatie"""
        
        exchange = exchange.lower()
        
        if exchange == "kraken":
            api_key = self.config.kraken_api_key
            secret = self.config.kraken_secret
        elif exchange == "binance":
            api_key = self.config.binance_api_key
            secret = self.config.binance_secret
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")
        
        if not api_key or not secret:
            raise ValueError(f"Missing {exchange} credentials")
        
        # Log access (gehashed voor security)
        security_manager.log_security_event(
            "EXCHANGE_CREDENTIALS_ACCESS",
            {
                "exchange": exchange,
                "key_hash": security_manager.hash_sensitive_data(api_key)
            },
            "INFO"
        )
        
        return {
            "api_key": api_key,
            "secret": secret
        }
    
    def get_ai_api_key(self, provider: str) -> str:
        """Haal AI API key op"""
        
        provider = provider.lower()
        
        if provider == "openai":
            key = self.config.openai_api_key
        elif provider == "anthropic":
            key = self.config.anthropic_api_key
        elif provider == "gemini":
            key = self.config.gemini_api_key
        else:
            raise ValueError(f"Unsupported AI provider: {provider}")
        
        if not key:
            raise ValueError(f"Missing {provider} API key")
        
        return key
    
    def get_security_config(self) -> Dict[str, str]:
        """Haal security configuratie op"""
        
        if not self.config.jwt_secret_key:
            raise ValueError("Missing JWT secret key")
        
        return {
            "jwt_secret": self.config.jwt_secret_key,
            "api_secret": self.config.api_secret_key or self.config.jwt_secret_key
        }
    
    def get_notification_config(self) -> Dict[str, Optional[str]]:
        """Haal notification configuratie op"""
        
        return {
            "slack_token": self.config.slack_bot_token,
            "slack_channel": self.config.slack_channel_id,
            "sendgrid_key": self.config.sendgrid_api_key
        }
    
    def get_system_config(self) -> Dict[str, Any]:
        """Haal system configuratie op"""
        
        return {
            "environment": self.config.environment,
            "debug": self.config.debug,
            "log_level": self.config.log_level,
            "trading_mode": self.config.trading_mode,
            "initial_portfolio": self.config.initial_portfolio_value,
            "max_daily_loss": self.config.max_daily_loss_percent,
            "max_drawdown": self.config.max_drawdown_percent,
            "dashboard_port": self.config.dashboard_port,
            "metrics_port": self.config.metrics_port,
            "health_port": self.config.health_port
        }
    
    def validate_trading_mode(self) -> bool:
        """Valideer trading mode configuratie"""
        
        valid_modes = ["paper", "live"]
        
        if self.config.trading_mode not in valid_modes:
            logger.error(f"Invalid trading mode: {self.config.trading_mode}")
            return False
        
        # In live mode, extra validatie required
        if self.config.trading_mode == "live":
            if self.config.environment != "production":
                logger.error("Live trading only allowed in production environment")
                return False
            
            # Valideer dat alle exchange credentials aanwezig zijn
            try:
                self.get_exchange_credentials("kraken")
                logger.info("Live trading mode validated successfully")
            except ValueError as e:
                logger.error(f"Live trading validation failed: {e}")
                return False
        
        return True
    
    def get_health_status(self) -> Dict[str, Any]:
        """Haal secrets health status op"""
        
        try:
            environment = self.config.environment
            required_secrets = self.config._required_secrets.get(environment, [])
            
            secrets_status = {}
            
            for secret_name in required_secrets:
                secret_value = getattr(self.config, secret_name, None)
                secrets_status[secret_name] = {
                    "present": bool(secret_value),
                    "valid_format": bool(secret_value and 
                                       security_manager.validate_input(secret_value, "api_key"))
                }
            
            all_present = all(status["present"] for status in secrets_status.values())
            all_valid = all(status["valid_format"] for status in secrets_status.values())
            
            return {
                "status": "healthy" if all_present and all_valid else "degraded",
                "environment": environment,
                "secrets_count": len(required_secrets),
                "all_present": all_present,
                "all_valid": all_valid,
                "details": secrets_status
            }
            
        except Exception as e:
            logger.error(f"Failed to get secrets health status: {e}")
            return {"status": "error", "error": str(e)}


# Global secrets manager instance
secrets_manager = SecretsManager()


def get_secrets_manager() -> SecretsManager:
    """Haal global secrets manager instance op"""
    return secrets_manager