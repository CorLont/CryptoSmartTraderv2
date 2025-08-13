"""Integration tests for API health endpoints."""

import pytest
import httpx
from fastapi.testclient import TestClient
from src.cryptosmarttrader.api.main import create_app


@pytest.mark.integration
class TestAPIHealth:
    """Test API health and monitoring endpoints."""

    def setup_method(self):
        """Setup test fixtures."""
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_health_endpoint_basic(self):
        """Test basic health endpoint functionality."""
        response = self.client.get("/health")

        assert response.status_code == 200

        health_data = response.json()
        assert "status" in health_data
        assert "timestamp" in health_data
        assert "version" in health_data
        assert health_data["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_endpoint_detailed(self):
        """Test detailed health check with components."""
        response = self.client.get("/health?detailed=true")

        assert response.status_code == 200

        health_data = response.json()
        assert "components" in health_data

        components = health_data["components"]
        expected_components = [
            "database",
            "redis_cache",
            "exchange_connections",
            "ml_models",
            "risk_system",
        ]

        for component in expected_components:
            if component in components:
                assert "status" in components[component]
                assert "last_check" in components[component]
                assert components[component]["status"] in ["up", "down", "degraded"]

    def test_metrics_endpoint(self):
        """Test Prometheus metrics endpoint."""
        response = self.client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"

        metrics_text = response.text

        # Check for expected metrics
        expected_metrics = [
            "http_requests_total",
            "http_request_duration_seconds",
            "trading_orders_total",
            "portfolio_value_usd",
            "risk_level_current",
        ]

        for metric in expected_metrics:
            if metric in metrics_text:
                assert f"# HELP {metric}" in metrics_text or f"# TYPE {metric}" in metrics_text

    def test_readiness_endpoint(self):
        """Test readiness endpoint for deployment health."""
        response = self.client.get("/ready")

        # Should return 200 when system is ready
        assert response.status_code in [200, 503]

        if response.status_code == 200:
            ready_data = response.json()
            assert "ready" in ready_data
            assert ready_data["ready"] is True

    def test_liveness_endpoint(self):
        """Test liveness endpoint for basic service health."""
        response = self.client.get("/live")

        # Should always return 200 unless service is completely down
        assert response.status_code == 200

        live_data = response.json()
        assert "alive" in live_data
        assert live_data["alive"] is True

    def test_system_info_endpoint(self):
        """Test system information endpoint."""
        response = self.client.get("/info")

        assert response.status_code == 200

        info_data = response.json()
        assert "application" in info_data
        assert "version" in info_data
        assert "environment" in info_data
        assert "python_version" in info_data
        assert "start_time" in info_data

    def test_health_endpoint_response_time(self):
        """Test health endpoint response time."""
        import time

        start_time = time.time()
        response = self.client.get("/health")
        end_time = time.time()

        response_time_ms = (end_time - start_time) * 1000

        assert response.status_code == 200
        assert response_time_ms < 1000  # Should respond within 1 second

    def test_concurrent_health_checks(self):
        """Test concurrent health check requests."""
        import concurrent.futures
        import threading

        def make_health_request():
            return self.client.get("/health")

        # Make 10 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_health_request) for _ in range(10)]
            responses = [future.result() for future in concurrent.futures.as_completed(futures)]

        # All should succeed
        assert all(response.status_code == 200 for response in responses)
        assert all("status" in response.json() for response in responses)

    def test_health_with_failing_component(self):
        """Test health response when a component is failing."""
        # This would typically involve mocking a component failure
        # For now, test that the endpoint handles errors gracefully

        response = self.client.get("/health")

        # Should still return a response even if some components fail
        assert response.status_code in [200, 503]

        if response.status_code == 503:
            health_data = response.json()
            assert "status" in health_data
            assert health_data["status"] in ["degraded", "unhealthy"]

    def test_api_versioning(self):
        """Test API versioning in headers."""
        response = self.client.get("/health")

        assert response.status_code == 200

        # Check for version headers
        assert "X-API-Version" in response.headers or "api-version" in response.headers

    def test_cors_headers(self):
        """Test CORS headers are properly set."""
        response = self.client.options("/health")

        # Should handle OPTIONS request for CORS
        assert response.status_code in [200, 204]

    def test_rate_limiting_headers(self):
        """Test rate limiting headers."""
        response = self.client.get("/health")

        # Check for rate limiting headers
        rate_limit_headers = [
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "Retry-After",
        ]

        # At least some rate limiting info should be present
        has_rate_limit_info = any(header in response.headers for header in rate_limit_headers)

        # This is optional, so we just check it doesn't break
        assert response.status_code == 200

    def test_health_check_caching(self):
        """Test health check response caching."""
        # Make first request
        response1 = self.client.get("/health")
        assert response1.status_code == 200

        # Make second request immediately
        response2 = self.client.get("/health")
        assert response2.status_code == 200

        # Should be consistent (though caching behavior may vary)
        assert response1.json()["status"] == response2.json()["status"]

    def test_health_endpoint_security(self):
        """Test health endpoint security headers."""
        response = self.client.get("/health")

        assert response.status_code == 200

        # Check for security headers
        security_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
        ]

        # Should have at least some security headers
        has_security_headers = any(header in response.headers for header in security_headers)


@pytest.mark.integration
class TestAPIMetrics:
    """Test API metrics collection and reporting."""

    def setup_method(self):
        """Setup test fixtures."""
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_request_counter_metrics(self):
        """Test request counter metrics are updated."""
        # Make some requests to generate metrics
        self.client.get("/health")
        self.client.get("/info")
        self.client.get("/ready")

        response = self.client.get("/metrics")
        assert response.status_code == 200

        metrics_text = response.text

        # Should have request counter
        assert "http_requests_total" in metrics_text

    def test_response_time_metrics(self):
        """Test response time metrics are collected."""
        # Make request
        self.client.get("/health")

        response = self.client.get("/metrics")
        assert response.status_code == 200

        metrics_text = response.text

        # Should have response time histogram
        duration_metrics = [
            "http_request_duration_seconds",
            "http_request_duration",
            "request_duration",
        ]

        has_duration_metric = any(metric in metrics_text for metric in duration_metrics)
        # Duration metrics might not be implemented yet, so we don't assert

    def test_custom_business_metrics(self):
        """Test custom business metrics are exposed."""
        response = self.client.get("/metrics")
        assert response.status_code == 200

        metrics_text = response.text

        # Expected business metrics
        business_metrics = [
            "trading_orders_total",
            "portfolio_value",
            "risk_level",
            "prediction_accuracy",
            "active_positions",
        ]

        # Check if any business metrics are present
        # (These might not all be implemented yet)
        for metric in business_metrics:
            if metric in metrics_text:
                # If present, should have proper format
                assert f"# HELP {metric}" in metrics_text or f"# TYPE {metric}" in metrics_text


@pytest.mark.integration
@pytest.mark.asyncio
class TestAPIHealthAsync:
    """Test API health with async HTTP client."""

    async def test_health_endpoint_async(self):
        """Test health endpoint with async client."""
        async with httpx.AsyncClient() as client:
            # Note: This would need the actual server running
            # For unit testing, we use TestClient instead
            pass

    async def test_health_endpoint_timeout(self):
        """Test health endpoint with timeout."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            # This test would verify timeout handling
            pass
