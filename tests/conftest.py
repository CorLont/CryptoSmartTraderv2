#!/usr/bin/env python3
"""
Test configuration for CryptoSmartTrader V2
Controls test discovery and excludes experimental/WIP modules
"""

import sys
import os
import pytest
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Exclude patterns for experimental/WIP modules
EXCLUDED_MODULES = [
    "demo_*",
    "experimental*",
    "*_demo*",
    "*experimental*",
    "*_wip*",
    "wip_*",
    "*temp*",
    "*backup*",
    "*old*",
    "*legacy*",
]


def pytest_configure(config):
    """Configure pytest with custom settings"""

    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "api_key: marks tests that require API keys")
    config.addinivalue_line("markers", "experimental: marks experimental/WIP tests to exclude")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to exclude experimental modules"""

    # Skip experimental/WIP modules
    for item in items[:]:  # Use slice copy to avoid modification issues
        file_path = str(item.fspath)

        # Check if file matches exclusion patterns
        should_exclude = False
        for pattern in EXCLUDED_MODULES:
            if pattern.replace("*", "") in file_path.lower():
                should_exclude = True
                break

        # Also exclude based on markers
        if item.get_closest_marker("experimental"):
            should_exclude = True

        if should_exclude:
            items.remove(item)
            print(f"Excluding experimental module: {file_path}")


def pytest_sessionstart(session):
    """Called after the Session object has been created"""
    print("\n🧪 Starting CryptoSmartTrader V2 Test Suite")
    print("=" * 50)
    print("✅ Experimental modules excluded from test run")
    print("✅ Coverage target: ≥70%")
    print("✅ Test markers configured")


@pytest.fixture(scope="session")
def project_root():
    """Provide project root path"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def src_path(project_root):
    """Provide src path"""
    return project_root / "src"


@pytest.fixture(scope="session")
def test_data_path(project_root):
    """Provide test data path"""
    return project_root / "tests" / "data"


@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    return {
        "kraken_api_key": "test_key",
        "kraken_secret": "test_secret",
        "openai_api_key": "test_openai_key",
        "log_level": "INFO",
        "environment": "test",
    }


@pytest.fixture
def sample_market_data():
    """Sample market data for tests"""
    return {
        "symbol": "BTC/USD",
        "price": 45000.0,
        "volume": 1000.0,
        "timestamp": "2024-01-01T00:00:00Z",
    }


@pytest.fixture
def sample_signal():
    """Sample trading signal for tests"""
    return {
        "symbol": "BTC/USD",
        "signal": "buy",
        "confidence": 0.85,
        "timestamp": "2024-01-01T00:00:00Z",
        "agent": "test_agent",
    }


# Additional configuration for clean test runs
def pytest_ignore_collect(path, config):
    """Ignore experimental/demo files during collection"""

    path_str = str(path).lower()

    # Skip files matching exclusion patterns
    for pattern in EXCLUDED_MODULES:
        if pattern.replace("*", "") in path_str:
            return True

    return False
