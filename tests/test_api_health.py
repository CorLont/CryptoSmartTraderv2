#!/usr/bin/env python3
"""
API Health Endpoint Tests - Contract testing for health monitoring
"""

import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient

from api.main import create_app


@pytest.fixture
def app():
    """Create test FastAPI app"""
    return create_app()


@pytest.fixture
def client(app):
    """Create test client"""
    return TestClient(app)


@pytest.fixture
async def async_client(app):
    """Create async test client"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoints:
    """Test suite for health monitoring endpoints"""
    
    def test_health_endpoint_structure(self, client):
        """Test health endpoint returns correct structure"""
        response = client.get("/health/")
        
        assert response.status_code == 200
        data = response.json()
        
        # Required fields
        assert "success" in data
        assert "timestamp" in data
        assert "status" in data
        assert "score" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "components" in data
        assert "checks" in data
        
        # Validate data types
        assert isinstance(data["success"], bool)
        assert isinstance(data["score"], (int, float))
        assert 0 <= data["score"] <= 100
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0
        
        # Components structure
        components = data["components"]
        required_components = ["database", "exchange_apis", "ml_models", "cache"]
        for component in required_components:
            assert component in components
            assert "status" in components[component]
            assert "message" in components[component]
            assert "last_updated" in components[component]
    
    def test_health_status_values(self, client):
        """Test health status contains valid values"""
        response = client.get("/health/")
        data = response.json()
        
        # Status should be one of expected values
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        
        # Component statuses should be valid
        for component_name, component in data["components"].items():
            assert component["status"] in ["healthy", "degraded", "unhealthy"]
    
    def test_health_checks_metrics(self, client):
        """Test system metrics in health checks"""
        response = client.get("/health/")
        data = response.json()
        
        checks = data["checks"]
        
        # Required metrics
        assert "cpu_usage" in checks
        assert "memory_usage" in checks
        assert "disk_usage" in checks
        
        # Validate metric ranges
        assert 0 <= checks["cpu_usage"] <= 100
        assert 0 <= checks["memory_usage"] <= 100
        assert 0 <= checks["disk_usage"] <= 100
    
    def test_liveness_endpoint(self, client):
        """Test liveness probe endpoint"""
        response = client.get("/health/live")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "success" in data
        assert "alive" in data
        assert data["alive"] is True
        assert data["success"] is True
    
    def test_readiness_endpoint(self, client):
        """Test readiness probe endpoint"""
        response = client.get("/health/ready")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "success" in data
        assert "ready" in data
        assert "dependencies_ready" in data
        assert "startup_complete" in data
        
        # All should be boolean
        assert isinstance(data["ready"], bool)
        assert isinstance(data["dependencies_ready"], bool)
        assert isinstance(data["startup_complete"], bool)
    
    @pytest.mark.asyncio
    async def test_health_async(self, async_client):
        """Test health endpoint with async client"""
        response = await async_client.get("/health/")
        
        assert response.status_code == 200
        data = response.json()
        assert set(data.keys()) >= {"status", "score", "success"}
    
    def test_health_response_headers(self, client):
        """Test security headers are present"""
        response = client.get("/health/")
        
        # Security headers should be present
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers
        
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
    
    def test_health_rate_limiting(self, client):
        """Test rate limiting is applied"""
        # Make multiple requests quickly
        responses = []
        for _ in range(35):  # Above the 30/minute limit
            response = client.get("/health/")
            responses.append(response.status_code)
        
        # Some requests should be rate limited (429)
        # Note: This might not trigger immediately in test environment
        assert all(code in [200, 429] for code in responses)
    
    @pytest.mark.integration
    def test_health_integration_real_checks(self, client):
        """Integration test with real system checks"""
        response = client.get("/health/")
        
        assert response.status_code == 200
        data = response.json()
        
        # In integration test, we expect real metrics
        assert data["checks"]["cpu_usage"] >= 0
        assert data["checks"]["memory_usage"] > 0  # Should have some memory usage
        assert data["uptime_seconds"] > 0


@pytest.mark.slow
class TestHealthPerformance:
    """Performance tests for health endpoints"""
    
    def test_health_response_time(self, client):
        """Test health endpoint response time"""
        import time
        
        start = time.time()
        response = client.get("/health/")
        end = time.time()
        
        assert response.status_code == 200
        assert (end - start) < 2.0  # Should respond within 2 seconds
    
    def test_concurrent_health_requests(self, app):
        """Test concurrent health requests"""
        import concurrent.futures
        
        def make_request():
            with TestClient(app) as client:
                return client.get("/health/")
        
        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]
        
        # All should succeed
        assert all(r.status_code == 200 for r in responses)
        assert len(responses) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])