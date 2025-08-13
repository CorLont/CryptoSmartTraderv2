"""Enterprise-grade configuration management with strict validation."""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


logger = logging.getLogger(__name__)


class DatabaseConfig(BaseSettings):
    """Database configuration with validation."""

    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="cryptotrader", description="Database name")
    user: str = Field(default="trader", description="Database user")
    password: str = Field(description="Database password")

    model_config = SettingsConfigDict(env_prefix="DB_")


class ApiConfig(BaseSettings):
    """API configuration with validation."""

    kraken_api_key: Optional[str] = Field(default=None, description="Kraken API key")
    kraken_secret: Optional[str] = Field(default=None, description="Kraken secret")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")

    model_config = SettingsConfigDict(env_prefix="API_")


class TradingConfig(BaseSettings):
    """Trading configuration with validation."""

    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    max_position_size: float = Field(default=0.02, ge=0.001, le=0.1)
    risk_free_rate: float = Field(default=0.05, ge=0.0, le=0.2)

    model_config = SettingsConfigDict(env_prefix="TRADING_")


class ConfigManager:
    """Enterprise configuration manager with fail-fast validation."""

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize configuration manager with optional config file."""
        self.config_path = Path(config_path) if config_path else Path("config.json")
        self._config: Dict[str, Any] = {}
        self._validated = False

        # Load and validate configuration
        try:
            self._load_configuration()
            self._validate_configuration()
            self._validated = True
            logger.info("Configuration loaded and validated successfully")
        except Exception as exc:
            logger.error(f"Configuration validation failed: {exc}")
            raise RuntimeError(f"Failed to initialize configuration: {exc}") from exc

    def _load_configuration(self) -> None:
        """Load configuration from environment and files."""
        try:
            # Load from environment variables first
            self._config.update({
                "database": DatabaseConfig().model_dump(),
                "api": ApiConfig().model_dump(),
                "trading": TradingConfig().model_dump(),
            })

            # Load from file if exists
            if self.config_path.exists():
                import json
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                    self._merge_configs(file_config)

        except (IOError, json.JSONDecodeError) as exc:
            logger.error(f"Failed to load configuration file {self.config_path}: {exc}")
            raise

    def _merge_configs(self, file_config: Dict[str, Any]) -> None:
        """Merge file configuration with environment configuration."""
        for section, values in file_config.items():
            if section in self._config and isinstance(values, dict):
                self._config[section].update(values)
            else:
                self._config[section] = values

    def _validate_configuration(self) -> None:
        """Validate critical configuration sections."""
        required_sections = ["database", "api", "trading"]

        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate API keys in production
        if os.getenv("ENVIRONMENT") == "production":
            api_config = self._config.get("api", {})
            if not api_config.get("kraken_api_key"):
                raise ValueError("Kraken API key required in production")
            if not api_config.get("openai_api_key"):
                raise ValueError("OpenAI API key required in production")

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support."""
        if not self._validated:
            raise RuntimeError("Configuration not validated - cannot retrieve values")

        try:
            keys = key.split(".")
            value = self._config

            for k in keys:
                value = value[k]

            return value

        except (KeyError, TypeError):
            logger.warning(f"Configuration key not found: {key}, using default: {default}")
            return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section."""
        if not self._validated:
            raise RuntimeError("Configuration not validated - cannot retrieve sections")

        return self._config.get(section, {})

    def set(self, key: str, value: Any) -> None:
        """Set configuration value with dot notation support."""
        if not self._validated:
            raise RuntimeError("Configuration not validated - cannot set values")

        keys = key.split(".")
        config = self._config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value
        logger.info(f"Configuration updated: {key} = {value}")

    def save(self) -> None:
        """Save current configuration to file."""
        if not self._validated:
            raise RuntimeError("Configuration not validated - cannot save")

        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, default=str)
            logger.info(f"Configuration saved to {self.config_path}")

        except IOError as exc:
            logger.error(f"Failed to save configuration: {exc}")
            raise

    def validate_startup_requirements(self) -> Dict[str, bool]:
        """Validate all startup requirements and return status."""
        validation_results = {
            "config_loaded": self._validated,
            "api_keys_present": False,
            "directories_exist": False,
            "ports_available": False,
        }

        try:
            # Check API keys
            api_config = self.get_section("api")
            validation_results["api_keys_present"] = bool(
                api_config.get("kraken_api_key") and api_config.get("openai_api_key")
            )

            # Check critical directories
            required_dirs = ["data", "logs", "models"]
            validation_results["directories_exist"] = all(
                Path(d).exists() for d in required_dirs
            )

            # Check port availability (simplified)
            import socket
            ports_to_check = [5000, 8000, 8001]
            available_ports = []

            for port in ports_to_check:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    result = sock.connect_ex(('localhost', port))
                    available_ports.append(result != 0)  # Port is available if connection fails

            validation_results["ports_available"] = all(available_ports)

        except Exception as exc:
            logger.error(f"Startup validation error: {exc}")

        return validation_results

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"

    @property
    def is_validated(self) -> bool:
        """Check if configuration has been validated."""
        return self._validated
