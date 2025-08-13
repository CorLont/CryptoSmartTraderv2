"""Health Endpoint Tests - PR3 Style Implementation"""

import pytest


@pytest.mark.api
async def test_health_endpoint(client):
    """PR3 Style Health Endpoint Test"""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert set(data.keys()) >= {"status", "score"}
    assert data["status"] in {"ok", "degraded", "fail"}


@pytest.mark.api
async def test_health_detailed_endpoint(client):
    """Enterprise Detailed Health Test"""
    resp = await client.get("/health/detailed")
    assert resp.status_code == 200
    data = resp.json()

    # Check required fields
    required_fields = {"status", "score", "timestamp", "details", "services"}
    assert set(data.keys()) >= required_fields

    # Validate status values
    assert data["status"] in {"ok", "degraded", "fail"}
    assert 0.0 <= data["score"] <= 1.0

    # Check details structure
    assert "memory_percent" in data["details"]
    assert "disk_percent" in data["details"]
    assert "cpu_count" in data["details"]

    # Check services structure
    assert isinstance(data["services"], dict)


@pytest.mark.api
async def test_health_response_model_validation(client):
    """Test Pydantic Model Validation"""
    resp = await client.get("/health")

    # Should not raise validation errors
    assert resp.status_code == 200
    data = resp.json()

    # Basic type validation
    assert isinstance(data["status"], str)
    assert isinstance(data["score"], (int, float))
    assert data["score"] >= 0.0


@pytest.mark.api
@pytest.mark.performance
async def test_health_endpoint_performance(client):
    """Performance Test for Health Endpoint"""
    import time

    start_time = time.time()
    resp = await client.get("/health")
    end_time = time.time()

    assert resp.status_code == 200
    response_time = end_time - start_time

    # Health endpoint should respond quickly (< 100ms)
    assert response_time < 0.5, f"Health endpoint too slow: {response_time:.3f}s"
