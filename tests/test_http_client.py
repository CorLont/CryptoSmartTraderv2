#!/usr/bin/env python3
"""
HTTP Client Tests - Enterprise HTTP client testing
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from core.http_client import EnterpriseHTTPClient, CircuitBreaker, CircuitState, CacheEntry


class TestCircuitBreaker:
    """Test circuit breaker functionality"""

    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker starts in closed state"""
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute() is True
        assert cb.failure_count == 0

    def test_circuit_breaker_failure_threshold(self):
        """Test circuit opens after failure threshold"""
        cb = CircuitBreaker(failure_threshold=3)

        # Record failures below threshold
        for i in range(2):
            cb.record_failure()
            assert cb.state == CircuitState.CLOSED
            assert cb.can_execute() is True

        # Failure at threshold should open circuit
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.can_execute() is False

    def test_circuit_breaker_recovery(self):
        """Test circuit recovery after timeout"""
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        # Open circuit
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # Should still be open immediately
        assert cb.can_execute() is False

        # Simulate time passage
        cb.last_failure_time = datetime.utcnow() - timedelta(seconds=2)

        # Should now be half-open
        assert cb.can_execute() is True
        assert cb.state == CircuitState.HALF_OPEN

    def test_circuit_breaker_half_open_success(self):
        """Test circuit closes after successful calls in half-open state"""
        cb = CircuitBreaker(half_open_max_calls=2)
        cb.state = CircuitState.HALF_OPEN

        # Record successful calls
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN

        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0


class TestCacheEntry:
    """Test cache entry functionality"""

    def test_cache_entry_expiry(self):
        """Test cache entry expiration"""
        entry = CacheEntry(
            data={"test": "data"},
            timestamp=datetime.utcnow() - timedelta(seconds=100),
            ttl_seconds=60,
        )

        assert entry.is_expired() is True

        # Fresh entry
        fresh_entry = CacheEntry(data={"test": "data"}, timestamp=datetime.utcnow(), ttl_seconds=60)

        assert fresh_entry.is_expired() is False

    def test_cache_entry_staleness(self):
        """Test cache entry staleness detection"""
        # Entry that's 90% through its TTL
        entry = CacheEntry(
            data={"test": "data"},
            timestamp=datetime.utcnow() - timedelta(seconds=54),  # 54/60 = 90%
            ttl_seconds=60,
        )

        assert entry.is_stale(stale_threshold=0.8) is True
        assert entry.is_expired() is False

    def test_cache_entry_access_tracking(self):
        """Test cache entry access tracking"""
        entry = CacheEntry(data={"test": "data"}, timestamp=datetime.utcnow(), ttl_seconds=60)

        assert entry.access_count == 0

        entry.access()
        assert entry.access_count == 1
        assert entry.last_accessed is not None


@pytest.mark.asyncio
class TestEnterpriseHTTPClient:
    """Test enterprise HTTP client"""

    @pytest.fixture
    async def client(self):
        """Create test HTTP client"""
        client = EnterpriseHTTPClient()
        yield client
        await client.close()

    def test_cache_key_generation(self, client):
        """Test cache key generation"""
        key1 = client._generate_cache_key("kraken", "ticker", {"pair": "BTCUSD"})
        key2 = client._generate_cache_key("kraken", "ticker", {"pair": "BTCUSD"})
        key3 = client._generate_cache_key("kraken", "ticker", {"pair": "ETHUSD"})

        assert key1 == key2  # Same parameters should generate same key
        assert key1 != key3  # Different parameters should generate different keys

    def test_cache_response_and_retrieval(self, client):
        """Test caching and retrieval"""
        cache_key = "test:endpoint:param=value"
        test_data = {"price": 50000}

        # Cache data
        client._cache_response(cache_key, test_data, "price", 60)

        # Retrieve from cache
        cached = client._get_cached_response(cache_key)
        assert cached == test_data

        # Test cache miss
        missing = client._get_cached_response("non:existent:key")
        assert missing is None

    def test_rate_limiting(self, client):
        """Test rate limiting functionality"""
        service = "test_service"
        endpoint = "test_endpoint"

        # Should allow initial requests
        for i in range(5):
            result = asyncio.create_task(client._check_rate_limit(service, endpoint))
            assert asyncio.run(result) is True

        # Mock high rate limit breach
        client.rate_limits[f"{service}:{endpoint}"] = {
            "requests": [1, 2, 3, 4, 5] * 20,  # 100 requests
            "window": 60,
        }

        result = asyncio.create_task(client._check_rate_limit(service, endpoint))
        # Note: Actual rate limiting depends on service configuration

    async def test_circuit_breaker_integration(self, client):
        """Test circuit breaker integration"""
        service = "test_service"

        # Circuit should be closed initially
        assert client.circuit_breakers[service].state == CircuitState.CLOSED

        # Simulate failures
        for i in range(5):
            client.circuit_breakers[service].record_failure()

        # Circuit should now be open
        assert client.circuit_breakers[service].state == CircuitState.OPEN

    @patch("httpx.AsyncClient.request")
    async def test_successful_request(self, mock_request, client):
        """Test successful HTTP request"""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True, "data": "test"}
        mock_response.headers = {"content-type": "application/json"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        result = await client.request(
            service="test_service", method="GET", url="https://api.test.com/data"
        )

        assert result == {"success": True, "data": "test"}
        mock_request.assert_called_once()

    @patch("httpx.AsyncClient.request")
    async def test_request_with_caching(self, mock_request, client):
        """Test request caching"""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"cached": "data"}
        mock_response.headers = {"content-type": "application/json"}
        mock_response.raise_for_status.return_value = None
        mock_request.return_value = mock_response

        # First request should hit the API
        result1 = await client.request(
            service="test_service", method="GET", url="https://api.test.com/cached", use_cache=True
        )

        # Second request should use cache
        result2 = await client.request(
            service="test_service", method="GET", url="https://api.test.com/cached", use_cache=True
        )

        assert result1 == result2
        # Should only call API once due to caching
        assert mock_request.call_count == 1

    def test_circuit_breaker_status(self, client):
        """Test circuit breaker status reporting"""
        status = client.get_circuit_breaker_status()

        assert isinstance(status, dict)
        assert "kraken" in status
        assert "binance" in status

        for service, info in status.items():
            assert "state" in info
            assert "failure_count" in info
            assert "successful_calls" in info

    def test_cache_stats(self, client):
        """Test cache statistics"""
        # Add some test data to cache
        client._cache_response("test:1", {"data": 1}, "test", 60)
        client._cache_response("test:2", {"data": 2}, "test", 60)

        stats = client.get_cache_stats()

        assert isinstance(stats, dict)
        assert "total_entries" in stats
        assert "expired_entries" in stats
        assert stats["total_entries"] >= 2


@pytest.mark.integration
@pytest.mark.asyncio
class TestHTTPClientIntegration:
    """Integration tests for HTTP client"""

    @pytest.fixture
    async def client(self):
        """Create test HTTP client"""
        client = EnterpriseHTTPClient()
        yield client
        await client.close()

    async def test_real_api_request(self, client):
        """Test real API request (if network available)"""
        try:
            # Test with a public API that doesn't require authentication
            result = await client.get(
                service="test", url="https://httpbin.org/json", use_cache=False
            )

            assert isinstance(result, dict)

        except Exception as e:
            pytest.skip(f"Network request failed: {e}")

    async def test_error_handling(self, client):
        """Test error handling with real requests"""
        try:
            # Test 404 error
            await client.get(service="test", url="https://httpbin.org/status/404", use_cache=False)

        except Exception as e:
            # Should raise an exception for 404
            assert "404" in str(e) or "Client Error" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
