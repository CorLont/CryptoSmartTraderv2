"""Tests for configuration management."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, mock_open

from src.cryptosmarttrader.core.config_manager import (
    ConfigManager,
    DatabaseConfig,
    ApiConfig,
    TradingConfig,
)


class TestConfigManager:
    """Test configuration manager functionality."""

    def test_config_manager_initialization(self):
        """Test configuration manager initialization."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config_data = {
                "database": {"host": "localhost", "port": 5432},
                "api": {"kraken_api_key": "test_key"},
                "trading": {"confidence_threshold": 0.8},
            }
            json.dump(config_data, f)
            config_path = f.name

        try:
            config_manager = ConfigManager(config_path=config_path)
            assert config_manager.is_validated
            assert config_manager.get("database.host") == "localhost"
        finally:
            Path(config_path).unlink()

    def test_missing_config_file(self):
        """Test handling of missing configuration file."""
        with patch.dict("os.environ", {}, clear=True):
            config_manager = ConfigManager(config_path="nonexistent.json")
            # Should still initialize with environment defaults
            assert config_manager.is_validated

    @pytest.mark.unit
    def test_validation_failure(self):
        """Test configuration validation failure."""
        with patch.dict("os.environ", {"ENVIRONMENT": "production"}, clear=True):
            with pytest.raises(RuntimeError, match="Failed to initialize configuration"):
                ConfigManager()

    def test_get_set_configuration(self):
        """Test getting and setting configuration values."""
        config_manager = ConfigManager()

        # Test setting and getting values
        config_manager.set("test.value", 42)
        assert config_manager.get("test.value") == 42

        # Test default values
        assert config_manager.get("nonexistent.key", "default") == "default"

    def test_startup_validation(self):
        """Test startup requirements validation."""
        config_manager = ConfigManager()
        validation_results = config_manager.validate_startup_requirements()

        assert isinstance(validation_results, dict)
        assert "config_loaded" in validation_results
        assert "api_keys_present" in validation_results
        assert "directories_exist" in validation_results
        assert "ports_available" in validation_results
