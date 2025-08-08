#!/usr/bin/env python3
"""
Pytest configuration and fixtures for CryptoSmartTrader testing
"""

import pytest
import asyncio
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch
import json
from datetime import datetime, timedelta

# Import application components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import AppSettings, get_settings
from core.dependency_container import Container, wire_container
from core.secrets_manager import SecretsManager, get_secrets_manager
from core.logging_manager import get_logger, configure_logging
from core.async_data_manager import AsyncDataManager, RateLimitConfig

# Configure pytest-asyncio
pytest_plugins = ("pytest_asyncio",)

@pytest.fixture(scope="session")
def event_loop_policy():
    """Use asyncio event loop policy for tests"""
    return asyncio.get_event_loop_policy()

@pytest.fixture(scope="function")
async def event_loop():
    """Create a new event loop for each test"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test data"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture(scope="function")
def test_env_vars(temp_dir: Path) -> Generator[Dict[str, str], None, None]:
    """Set up test environment variables"""
    original_env = dict(os.environ)
    
    test_vars = {
        'LOG_LEVEL': 'DEBUG',
        'LOG_DIR': str(temp_dir / 'logs'),
        'DATA_DIR': str(temp_dir / 'data'),
        'CACHE_DIR': str(temp_dir / 'cache'),
        'ENABLE_JSON_LOGGING': 'true',
        'METRICS_PORT': '8091',  # Different port for tests
        'ENABLE_PROMETHEUS': 'false',  # Disable metrics server in tests
        'KRAKEN_API_KEY': 'test_kraken_key',
        'KRAKEN_SECRET': 'test_kraken_secret',
        'BINANCE_API_KEY': 'test_binance_key',
        'BINANCE_SECRET': 'test_binance_secret',
        'OPENAI_API_KEY': 'test_openai_key',
        'REQUESTS_PER_SECOND': '100.0',  # Higher for tests
        'BURST_SIZE': '100',
        'TIMEOUT_SECONDS': '5',
        'ENABLE_SECRET_REDACTION': 'true',
        'REQUIRE_VAULT_FOR_PRODUCTION': 'false',  # Disable for tests
    }
    
    # Apply test environment
    os.environ.update(test_vars)
    
    yield test_vars
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

@pytest.fixture(scope="function")
def test_settings(test_env_vars: Dict[str, str]) -> AppSettings:
    """Create test settings with isolated configuration"""
    # Reset settings cache to ensure fresh config
    if hasattr(get_settings, 'cache_clear'):
        get_settings.cache_clear()
    
    return get_settings()

@pytest.fixture(scope="function")
def test_secrets_manager(test_settings: AppSettings, temp_dir: Path) -> SecretsManager:
    """Create test secrets manager with isolated state"""
    # Reset global secrets manager
    import core.secrets_manager
    core.secrets_manager._secrets_manager = None
    
    config = {
        'secrets_metadata': {
            'KRAKEN_API_KEY': {
                'name': 'KRAKEN_API_KEY',
                'secret_type': 'api_key',
                'source': 'env'
            },
            'KRAKEN_SECRET': {
                'name': 'KRAKEN_SECRET',
                'secret_type': 'api_key',
                'source': 'env'
            }
        }
    }
    
    return SecretsManager(config)

@pytest.fixture(scope="function")
def test_logger(test_settings: AppSettings):
    """Create isolated logger for tests"""
    # Reset logging state
    # Note: reset_logging function needs to be implemented if needed
    
    # Configure test logging
    logger_config = {
        'log_dir': str(test_settings.logging.log_dir),
        'metrics_port': test_settings.logging.metrics_port
    }
    
    return configure_logging(logger_config)

@pytest.fixture(scope="function")
def test_container(test_settings: AppSettings, test_secrets_manager: SecretsManager) -> Container:
    """Create isolated DI container for tests"""
    container = Container()
    
    # Override providers with test instances
    container.config.override(test_settings)
    container.secrets_manager.override(test_secrets_manager)
    
    # Wire for testing
    wire_container([__name__])
    
    yield container
    
    # Unwire after test
    container.unwire()

@pytest.fixture(scope="function")
async def test_rate_limit_config() -> RateLimitConfig:
    """Create test rate limit configuration"""
    return RateLimitConfig(
        requests_per_second=100.0,  # Higher for tests
        burst_size=100,
        timeout_seconds=5,
        cool_down_period=1
    )

@pytest.fixture(scope="function")
async def mock_async_data_manager(test_rate_limit_config: RateLimitConfig) -> AsyncMock:
    """Create mocked async data manager"""
    mock_manager = AsyncMock(spec=AsyncDataManager)
    mock_manager.rate_limit_config = test_rate_limit_config
    mock_manager.exchanges = {}
    
    # Mock common methods
    mock_manager.initialize = AsyncMock()
    mock_manager.cleanup = AsyncMock()
    mock_manager.batch_collect_all_exchanges = AsyncMock()
    mock_manager.setup_async_exchanges = AsyncMock()
    
    return mock_manager

# Exchange API Mocks
@pytest.fixture(scope="function")
def mock_kraken_data() -> Dict[str, Any]:
    """Sample Kraken API response data"""
    return {
        "error": [],
        "result": {
            "XXBTZUSD": {
                "a": ["50000.00000", "1", "1.000"],
                "b": ["49999.00000", "2", "2.000"],
                "c": ["50000.50000", "0.01000000"],
                "v": ["1234.12345678", "2345.23456789"],
                "p": ["49500.00000", "49750.00000"],
                "t": [1000, 2000],
                "l": ["49000.00000", "49250.00000"],
                "h": ["51000.00000", "50750.00000"],
                "o": "49800.00000"
            }
        }
    }

@pytest.fixture(scope="function")
def mock_binance_data() -> Dict[str, Any]:
    """Sample Binance API response data"""
    return {
        "symbol": "BTCUSDT",
        "priceChange": "500.00000000",
        "priceChangePercent": "1.020",
        "weightedAvgPrice": "49750.00000000",
        "prevClosePrice": "49500.00000000",
        "lastPrice": "50000.00000000",
        "lastQty": "0.01000000",
        "bidPrice": "49999.00000000",
        "bidQty": "1.00000000",
        "askPrice": "50001.00000000",
        "askQty": "1.00000000",
        "openPrice": "49500.00000000",
        "highPrice": "51000.00000000",
        "lowPrice": "49000.00000000",
        "volume": "1234.12345678",
        "quoteVolume": "61234567.89000000",
        "openTime": 1640995200000,
        "closeTime": 1641081599999,
        "count": 1000
    }

@pytest.fixture(scope="function")
def mock_ohlcv_data() -> list:
    """Sample OHLCV data"""
    base_time = int(datetime.now().timestamp() * 1000)
    return [
        [base_time - 3600000, 49000, 50000, 48500, 49500, 100],  # 1 hour ago
        [base_time - 1800000, 49500, 51000, 49200, 50200, 150],  # 30 min ago
        [base_time, 50200, 50500, 49800, 50000, 120]  # Now
    ]

@pytest.fixture(scope="function")
def mock_exchange_factory():
    """Factory for creating mock exchange instances"""
    def create_mock_exchange(name: str, has_credentials: bool = True):
        mock_exchange = AsyncMock()
        mock_exchange.name = name
        mock_exchange.id = name
        mock_exchange.has = {
            'fetchTickers': True,
            'fetchOHLCV': True,
            'fetchOrderBook': True,
            'fetchTrades': True
        }
        
        # Mock API methods
        mock_exchange.fetch_tickers = AsyncMock()
        mock_exchange.fetch_ohlcv = AsyncMock()
        mock_exchange.fetch_order_book = AsyncMock()
        mock_exchange.fetch_trades = AsyncMock()
        mock_exchange.load_markets = AsyncMock()
        
        # Mock credentials
        if has_credentials:
            mock_exchange.apiKey = f"test_{name}_key"
            mock_exchange.secret = f"test_{name}_secret"
        else:
            mock_exchange.apiKey = None
            mock_exchange.secret = None
        
        return mock_exchange
    
    return create_mock_exchange

@pytest.fixture(scope="function")
def mock_sentiment_data() -> Dict[str, Any]:
    """Sample sentiment analysis data"""
    return {
        "reddit": {
            "posts": [
                {
                    "title": "Bitcoin to the moon! 🚀",
                    "score": 156,
                    "num_comments": 45,
                    "created_utc": datetime.now().timestamp(),
                    "sentiment": {"compound": 0.8, "pos": 0.7, "neu": 0.2, "neg": 0.1}
                },
                {
                    "title": "Market looks bearish today",
                    "score": 23,
                    "num_comments": 12,
                    "created_utc": datetime.now().timestamp(),
                    "sentiment": {"compound": -0.6, "pos": 0.1, "neu": 0.3, "neg": 0.6}
                }
            ],
            "overall_sentiment": {"compound": 0.2, "pos": 0.4, "neu": 0.4, "neg": 0.2}
        },
        "twitter": {
            "tweets": [
                {
                    "text": "Just bought more #Bitcoin! HODL strong! 💪",
                    "retweet_count": 45,
                    "favorite_count": 120,
                    "created_at": datetime.now().isoformat(),
                    "sentiment": {"compound": 0.7, "pos": 0.6, "neu": 0.3, "neg": 0.1}
                }
            ],
            "overall_sentiment": {"compound": 0.5, "pos": 0.5, "neu": 0.3, "neg": 0.2}
        }
    }

@pytest.fixture(scope="function")
def mock_openai_response() -> Dict[str, Any]:
    """Sample OpenAI API response"""
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "sentiment": "bullish",
                        "confidence": 0.85,
                        "key_factors": [
                            "Positive technical indicators",
                            "Strong social media sentiment",
                            "Institutional adoption"
                        ],
                        "risk_level": "medium"
                    })
                }
            }
        ],
        "usage": {
            "prompt_tokens": 150,
            "completion_tokens": 100,
            "total_tokens": 250
        }
    }

# Async context managers for testing
@pytest.fixture(scope="function")
async def async_test_context(
    test_container: Container,
    mock_async_data_manager: AsyncMock
) -> AsyncGenerator[Dict[str, Any], None]:
    """Async context for integration tests"""
    
    # Override async data manager in container
    test_container.async_data_manager.override(mock_async_data_manager)
    
    context = {
        'container': test_container,
        'data_manager': mock_async_data_manager,
        'start_time': datetime.now()
    }
    
    try:
        yield context
    finally:
        # Cleanup
        if hasattr(mock_async_data_manager, 'cleanup'):
            await mock_async_data_manager.cleanup()

# Test data factories
class TestDataFactory:
    """Factory for generating test data"""
    
    @staticmethod
    def create_market_data(
        exchange: str = "kraken",
        symbol: str = "BTC/USD",
        timestamp: datetime = None
    ) -> Dict[str, Any]:
        """Create sample market data"""
        if timestamp is None:
            timestamp = datetime.now()
        
        return {
            "timestamp": timestamp.isoformat(),
            "exchange": exchange,
            "symbol": symbol,
            "bid": 49999.0,
            "ask": 50001.0,
            "last": 50000.0,
            "volume": 1234.56,
            "high": 51000.0,
            "low": 49000.0,
            "change": 500.0,
            "change_percent": 1.02
        }
    
    @staticmethod
    def create_prediction_data(
        symbol: str = "BTC/USD",
        timeframe: str = "1h",
        confidence: float = 0.85
    ) -> Dict[str, Any]:
        """Create sample ML prediction data"""
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "prediction": {
                "direction": "up",
                "price_target": 52000.0,
                "confidence": confidence,
                "horizon": "24h"
            },
            "features": {
                "technical_score": 0.7,
                "sentiment_score": 0.6,
                "volume_score": 0.8
            },
            "model_version": "v2.1.0",
            "created_at": datetime.now().isoformat()
        }

@pytest.fixture(scope="function")
def test_data_factory() -> TestDataFactory:
    """Test data factory instance"""
    return TestDataFactory()

# Performance testing utilities
@pytest.fixture(scope="function")
def performance_monitor():
    """Monitor performance during tests"""
    import psutil
    import time
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.process = psutil.Process()
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss
        
        def stop(self):
            end_time = time.time()
            end_memory = self.process.memory_info().rss
            
            return {
                'duration': end_time - self.start_time,
                'memory_delta': end_memory - self.start_memory,
                'peak_memory': self.process.memory_info().rss
            }
    
    return PerformanceMonitor()

# Integration test markers
def pytest_configure(config):
    """Configure custom pytest markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "contract: mark test as contract test (external APIs)"
    )
    config.addinivalue_line(
        "markers", "smoke: mark test as smoke test (dashboard/UI)"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )