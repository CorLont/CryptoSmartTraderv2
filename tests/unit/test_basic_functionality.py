#!/usr/bin/env python3
"""
Basic functionality tests to verify core system components
"""

import pytest
import os
from pathlib import Path
import tempfile


@pytest.mark.unit
class TestBasicSystemComponents:
    """Test basic system functionality"""

    def test_imports_work(self):
        """Test that core imports work correctly"""
        # Test core imports
        from config.settings import AppSettings, get_settings
        from core.secrets_manager import SecretsManager

        assert AppSettings is not None
        assert get_settings is not None
        assert SecretsManager is not None

    def test_settings_creation(self):
        """Test that settings can be created"""
        from config.settings import AppSettings

        # Create settings with minimal config
        settings = AppSettings()

        assert settings is not None
        assert settings.environment == "development"
        assert settings.debug is False

    def test_secrets_manager_basic(self):
        """Test basic secrets manager functionality"""
        from core.secrets_manager import SecretsManager

        manager = SecretsManager()

        # Test getting a secret that doesn't exist
        secret = manager.get_secret("NON_EXISTENT_SECRET")
        assert secret is None

        # Test getting a secret with default
        secret = manager.get_secret("NON_EXISTENT_SECRET", "default_value")
        assert secret == "default_value"

    def test_environment_variable_handling(self):
        """Test environment variable handling"""
        # Set a test environment variable
        test_key = "TEST_SECRET_KEY"
        test_value = "test_secret_value"

        os.environ[test_key] = test_value

        try:
            from core.secrets_manager import SecretsManager

            manager = SecretsManager()

            # Should retrieve the environment variable
            retrieved = manager.get_secret(test_key)
            assert retrieved == test_value

        finally:
            # Clean up
            if test_key in os.environ:
                del os.environ[test_key]

    def test_directory_creation(self):
        """Test that required directories can be created"""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "test_data"

            # Should create directory
            test_path.mkdir(parents=True, exist_ok=True)
            assert test_path.exists()
            assert test_path.is_dir()


@pytest.mark.unit
class TestConfigurationSystem:
    """Test configuration system basics"""

    def test_default_configuration(self):
        """Test default configuration values"""
        from config.settings import AppSettings

        settings = AppSettings()

        # Test defaults
        assert settings.environment == "development"
        assert settings.debug is False
        assert settings.exchange.requests_per_second == 10.0
        assert settings.ml.training_data_days == 90

    def test_environment_override(self):
        """Test environment variable override"""
        # Set test environment variables
        test_env = {"ENVIRONMENT": "testing", "DEBUG": "true", "REQUESTS_PER_SECOND": "25.0"}

        # Apply environment
        original_env = {}
        for key, value in test_env.items():
            original_env[key] = os.environ.get(key)
            os.environ[key] = value

        try:
            from config.settings import AppSettings

            # Create new settings instance
            settings = AppSettings()

            # Should use environment values
            assert settings.environment == "testing"
            assert settings.debug is True
            assert settings.exchange.requests_per_second == 25.0

        finally:
            # Restore original environment
            for key, original_value in original_env.items():
                if original_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = original_value


@pytest.mark.unit
def test_basic_imports():
    """Test that all core modules can be imported"""

    # Test individual imports
    import config.settings
    import core.secrets_manager

    # Test that key classes exist
    assert hasattr(config.settings, "AppSettings")
    assert hasattr(core.secrets_manager, "SecretsManager")


@pytest.mark.unit
def test_pathlib_functionality():
    """Test pathlib functionality used throughout system"""

    # Test path operations
    test_path = Path("test") / "subdir" / "file.txt"

    assert str(test_path) in ["test/subdir/file.txt", "test\\subdir\\file.txt"]  # Handle Windows
    assert test_path.name == "file.txt"
    assert test_path.suffix == ".txt"
    assert test_path.parent.name == "subdir"
