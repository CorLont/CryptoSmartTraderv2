#!/usr/bin/env python3
"""
API Test Fixtures - Mock data and API responses for testing
"""

import pytest
import json
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime


class MockAPIResponse:
    """Mock API response with common attributes"""

    def __init__(self, data: Dict[str, Any], status_code: int = 200, headers: Dict = None):
        self.data = data
        self.status_code = status_code
        self.headers = headers or {}
        self.text = json.dumps(data)

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


@pytest.fixture
def mock_kraken_response():
    """Mock Kraken API response"""
    return {
        "error": [],
        "result": {
            "XBTUSD": {
                "a": ["50150.0", "1", "1.000"],
                "b": ["50140.0", "2", "2.000"],
                "c": ["50145.0", "0.12345678"],
                "v": ["1234.567", "2345.678"],
                "p": ["50100.0", "50050.0"],
                "t": [123, 234],
                "l": ["49950.0", "49900.0"],
                "h": ["50200.0", "50250.0"],
                "o": "50000.0",
            }
        },
    }


@pytest.fixture
def mock_binance_response():
    """Mock Binance API response"""
    return {
        "symbol": "BTCUSDT",
        "priceChange": "150.00000000",
        "priceChangePercent": "0.300",
        "weightedAvgPrice": "50075.00000000",
        "prevClosePrice": "50000.00000000",
        "lastPrice": "50150.00000000",
        "lastQty": "0.12345678",
        "bidPrice": "50140.00000000",
        "bidQty": "2.00000000",
        "askPrice": "50150.00000000",
        "askQty": "1.00000000",
        "openPrice": "50000.00000000",
        "highPrice": "50200.00000000",
        "lowPrice": "49950.00000000",
        "volume": "1234.56700000",
        "quoteVolume": "61827890.00000000",
        "openTime": 1640995200000,
        "closeTime": 1641081599999,
        "firstId": 123456,
        "lastId": 123457,
        "count": 2,
    }


@pytest.fixture
def mock_portfolio_data():
    """Mock portfolio data for testing"""
    return {
        "account_id": "test_account",
        "total_value": 100000.0,
        "cash_balance": 20000.0,
        "positions": [
            {
                "symbol": "BTC/USD",
                "quantity": 2.0,
                "avg_price": 40000.0,
                "current_price": 50000.0,
                "unrealized_pnl": 20000.0,
                "realized_pnl": 5000.0,
            },
            {
                "symbol": "ETH/USD",
                "quantity": 10.0,
                "avg_price": 2500.0,
                "current_price": 3000.0,
                "unrealized_pnl": 5000.0,
                "realized_pnl": 1000.0,
            },
        ],
        "last_update": "2024-01-15T12:00:00Z",
    }


@pytest.fixture
def mock_trading_signals():
    """Mock trading signals for testing"""
    return [
        {
            "timestamp": "2024-01-15T12:00:00Z",
            "symbol": "BTC/USD",
            "signal_type": "buy",
            "confidence": 0.85,
            "price": 50000.0,
            "agent": "Technical",
            "reason": "Golden cross detected",
        },
        {
            "timestamp": "2024-01-15T12:30:00Z",
            "symbol": "ETH/USD",
            "signal_type": "sell",
            "confidence": 0.75,
            "price": 3000.0,
            "agent": "Sentiment",
            "reason": "Negative sentiment detected",
        },
    ]


@pytest.fixture
def mock_health_data():
    """Mock system health data"""
    return {
        "overall_score": 85.5,
        "grade": "B",
        "components": {
            "data_quality": 90.0,
            "system_performance": 85.0,
            "model_performance": 80.0,
            "api_health": 88.0,
            "trading_system": 85.0,
            "security": 95.0,
        },
        "trading_enabled": True,
        "last_update": "2024-01-15T12:00:00Z",
    }


@pytest.fixture
def mock_http_client():
    """Mock HTTP client for testing"""
    mock_client = AsyncMock()

    # Configure common responses
    mock_client.get.return_value = MockAPIResponse({"status": "success"})
    mock_client.post.return_value = MockAPIResponse({"status": "success"})

    return mock_client


@pytest.fixture
def mock_exchange_apis():
    """Mock all exchange APIs"""
    with patch("ccxt.kraken") as mock_kraken, patch("ccxt.binance") as mock_binance:
        # Configure Kraken mock
        mock_kraken_instance = Mock()
        mock_kraken_instance.fetch_ticker.return_value = {
            "symbol": "BTC/USD",
            "last": 50000.0,
            "bid": 49995.0,
            "ask": 50005.0,
            "high": 50200.0,
            "low": 49800.0,
            "volume": 1234.5,
        }
        mock_kraken.return_value = mock_kraken_instance

        # Configure Binance mock
        mock_binance_instance = Mock()
        mock_binance_instance.fetch_ticker.return_value = {
            "symbol": "BTC/USDT",
            "last": 50000.0,
            "bid": 49995.0,
            "ask": 50005.0,
            "high": 50200.0,
            "low": 49800.0,
            "volume": 2345.6,
        }
        mock_binance.return_value = mock_binance_instance

        yield {"kraken": mock_kraken_instance, "binance": mock_binance_instance}


@pytest.fixture
def api_key_required():
    """Marker for tests that require API keys"""
    pytest.importorskip("os")
    import os

    if not os.getenv("KRAKEN_API_KEY") or not os.getenv("BINANCE_API_KEY"):
        pytest.skip("API keys not available for this test")


@pytest.fixture
def mock_ml_model():
    """Mock ML model for testing"""
    mock_model = Mock()

    # Configure model responses
    mock_model.predict.return_value = [0.85, 0.75, 0.65]  # Confidence scores
    mock_model.predict_proba.return_value = [[0.15, 0.85], [0.25, 0.75], [0.35, 0.65]]
    mock_model.score.return_value = 0.82  # Accuracy score

    return mock_model


@pytest.fixture
def mock_database():
    """Mock database connection for testing"""
    mock_db = Mock()

    # Configure database responses
    mock_db.execute.return_value = Mock()
    mock_db.fetchall.return_value = []
    mock_db.fetchone.return_value = None
    mock_db.commit.return_value = None

    return mock_db


class MockStreamlitSession:
    """Mock Streamlit session state for testing"""

    def __init__(self):
        self._state = {}

    def __getitem__(self, key):
        return self._state[key]

    def __setitem__(self, key, value):
        self._state[key] = value

    def __contains__(self, key):
        return key in self._state

    def get(self, key, default=None):
        return self._state.get(key, default)

    def clear(self):
        self._state.clear()


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit for dashboard testing"""
    mock_session = MockStreamlitSession()

    with (
        patch("streamlit.session_state", mock_session),
        patch("streamlit.cache_data"),
        patch("streamlit.rerun"),
    ):
        yield mock_session


@pytest.fixture(scope="session")
def test_config():
    """Test configuration settings"""
    return {
        "database_url": "sqlite:///:memory:",
        "redis_url": "redis://localhost:6379/0",
        "log_level": "DEBUG",
        "environment": "test",
        "api_timeout": 5,
        "max_retries": 2,
    }


class APITestHelper:
    """Helper class for API testing"""

    @staticmethod
    def create_mock_response(data: Dict[str, Any], status: int = 200) -> MockAPIResponse:
        """Create a mock API response"""
        return MockAPIResponse(data, status)

    @staticmethod
    def create_error_response(message: str, status: int = 400) -> MockAPIResponse:
        """Create a mock error response"""
        return MockAPIResponse({"error": message}, status)

    @staticmethod
    def assert_api_call(mock_client, method: str, url: str, **kwargs):
        """Assert that an API call was made with correct parameters"""
        getattr(mock_client, method.lower()).assert_called_with(url, **kwargs)


@pytest.fixture
def api_helper():
    """API testing helper fixture"""
    return APITestHelper()
