"""API Test Configuration"""

import pytest
from httpx import AsyncClient

from src.cryptosmarttrader.api.app import get_app


@pytest.fixture
async def client():
    """Test client fixture for API tests"""
    app = get_app()
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac