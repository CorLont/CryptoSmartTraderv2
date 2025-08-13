"""Test configuration and fixtures for CryptoSmartTrader V2."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Mock external dependencies that aren't critical for core tests
@pytest.fixture(autouse=True)
def mock_external_dependencies():
    """Mock external dependencies that might not be available in CI."""
    with patch.dict('sys.modules', {
        'ccxt': MagicMock(),
        'openai': MagicMock(),
        'anthropic': MagicMock(),
        'streamlit': MagicMock(),
        'plotly': MagicMock(),
    }):
        yield

@pytest.fixture
def temp_config_file():
    """Create a temporary configuration file for testing."""
    config_data = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "test_db"
        },
        "api": {
            "kraken_api_key": "test_key"
        },
        "trading": {
            "confidence_threshold": 0.8
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name
    
    yield config_path
    
    # Cleanup
    Path(config_path).unlink()

@pytest.fixture
def mock_environment():
    """Mock environment variables for testing."""
    env_vars = {
        'ENVIRONMENT': 'test',
        'KRAKEN_API_KEY': 'test_key',
        'KRAKEN_SECRET': 'test_secret',
        'OPENAI_API_KEY': 'test_openai_key'
    }
    
    with patch.dict('os.environ', env_vars, clear=True):
        yield

@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    return {
        'BTC/USDT': {
            'symbol': 'BTC/USDT',
            'bid': 50000.0,
            'ask': 50100.0,
            'last': 50050.0,
            'timestamp': 1640995200000,
            'volume': 1000.0
        }
    }