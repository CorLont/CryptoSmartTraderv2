#!/usr/bin/env python3
"""
Critical Tests for Data Ingestion Robustness
Ensures timeout/retry/exponential backoff voor stabiele datacollectie
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch, AsyncMock
import aiohttp
from datetime import datetime, timedelta

import sys
sys.path.append('.')

from src.cryptosmarttrader.data.enterprise_data_ingestion import (
    EnterpriseDataManager,
    DataRequest,
    DataPriority,
    DataSourceStatus,
    RobustExchangeConnector
)


class TestEnterpriseDataManager:
    """Critical tests voor enterprise data manager robustness"""
    
    @pytest.fixture
    def data_manager(self):
        """Fresh data manager instance"""
        return EnterpriseDataManager()
    
    @pytest.fixture  
    def mock_exchange(self):
        """Mock exchange connector"""
        mock = Mock(spec=RobustExchangeConnector)
        mock.id = "test_exchange"
        mock.status = DataSourceStatus.HEALTHY
        return mock
    
    @pytest.mark.asyncio
    async def test_timeout_enforcement(self, data_manager):
        """CRITICAL: Timeout enforcement voor hanging requests"""
        
        # Mock a slow response that exceeds timeout
        async def slow_request(*args, **kwargs):
            await asyncio.sleep(2.0)  # 2 second delay
            return {"data": "slow_response"}
        
        request = DataRequest(
            source="test_exchange",
            endpoint="ticker",
            params={"symbol": "ETH/USD"},
            priority=DataPriority.CRITICAL,
            timeout=0.5  # 500ms timeout
        )
        
        with patch.object(data_manager, '_execute_request', slow_request):
            start_time = time.time()
            
            response = await data_manager.fetch_data(request)
            
            elapsed = time.time() - start_time
            
            # Should timeout within reasonable time
            assert elapsed < 1.0  # Should not wait full 2 seconds
            assert response.status == "timeout"
    
    @pytest.mark.asyncio
    async def test_exponential_backoff_retry(self, data_manager):
        """CRITICAL: Exponential backoff retry logic"""
        call_times = []
        
        async def failing_request(*args, **kwargs):
            call_times.append(time.time())
            if len(call_times) < 3:
                raise aiohttp.ClientError("Connection failed")
            return {"data": "success_after_retries"}
        
        request = DataRequest(
            source="test_exchange",
            endpoint="ticker", 
            params={"symbol": "ETH/USD"},
            priority=DataPriority.CRITICAL,
            retry_attempts=3,
            timeout=1.0
        )
        
        with patch.object(data_manager, '_execute_request', failing_request):
            response = await data_manager.fetch_data(request)
            
            # Should succeed after retries
            assert response.status == "success"
            assert len(call_times) == 3
            
            # Verify exponential backoff timing
            if len(call_times) >= 2:
                first_retry_delay = call_times[1] - call_times[0]
                second_retry_delay = call_times[2] - call_times[1]
                
                # Second delay should be longer (exponential backoff)
                assert second_retry_delay > first_retry_delay
    
    @pytest.mark.asyncio
    async def test_rate_limiting_enforcement(self, data_manager):
        """CRITICAL: Rate limiting enforcement"""
        # Configure tight rate limits
        data_manager.rate_limiters["test_exchange"] = {
            'max_calls_per_second': 2,
            'burst_allowance': 3,
            'last_reset': time.time(),
            'current_calls': 0
        }
        
        request = DataRequest(
            source="test_exchange",
            endpoint="ticker",
            params={"symbol": "ETH/USD"},
            priority=DataPriority.CRITICAL
        )
        
        # Mock successful request
        async def mock_request(*args, **kwargs):
            return {"data": "rate_limited_response"}
        
        with patch.object(data_manager, '_execute_request', mock_request):
            start_time = time.time()
            
            # Fire multiple requests rapidly
            tasks = []
            for i in range(5):
                tasks.append(data_manager.fetch_data(request))
            
            responses = await asyncio.gather(*tasks)
            
            elapsed = time.time() - start_time
            
            # Should be rate limited (not all complete immediately)
            assert elapsed > 1.0  # Should take at least 1 second due to rate limiting
            
            # Some requests should be rate limited
            rate_limited_count = sum(1 for r in responses if r.status == "rate_limited")
            assert rate_limited_count > 0
    
    @pytest.mark.asyncio  
    async def test_caching_effectiveness(self, data_manager):
        """CRITICAL: Caching effectiveness voor duplicate requests"""
        call_count = 0
        
        async def counted_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {"data": f"response_{call_count}", "timestamp": time.time()}
        
        request = DataRequest(
            source="test_exchange",
            endpoint="ticker",
            params={"symbol": "ETH/USD"},
            priority=DataPriority.HIGH,
            cache_ttl=5  # 5 second cache
        )
        
        with patch.object(data_manager, '_execute_request', counted_request):
            # First request
            response1 = await data_manager.fetch_data(request)
            
            # Second identical request (should be cached)
            response2 = await data_manager.fetch_data(request)
            
            # Third request after cache expiry
            await asyncio.sleep(6)  # Wait for cache to expire
            response3 = await data_manager.fetch_data(request)
            
            # Verify caching behavior
            assert call_count == 2  # Should only call API twice (once + after cache expiry)
            assert response1.data == response2.data  # Second should be cached
            assert response1.data != response3.data  # Third should be fresh
    
    @pytest.mark.asyncio
    async def test_priority_queue_ordering(self, data_manager):
        """CRITICAL: Priority queue ordering voor data requests"""
        execution_order = []
        
        async def tracking_request(request, *args, **kwargs):
            execution_order.append(request.priority.value)
            await asyncio.sleep(0.1)  # Simulate processing time
            return {"data": f"priority_{request.priority.value}"}
        
        # Create requests with different priorities
        requests = [
            DataRequest("test", "endpoint", {}, DataPriority.LOW),
            DataRequest("test", "endpoint", {}, DataPriority.CRITICAL),
            DataRequest("test", "endpoint", {}, DataPriority.HIGH),
            DataRequest("test", "endpoint", {}, DataPriority.MEDIUM)
        ]
        
        with patch.object(data_manager, '_execute_request', tracking_request):
            # Submit all requests simultaneously
            tasks = [data_manager.fetch_data(req) for req in requests]
            await asyncio.gather(*tasks)
            
            # Verify execution order respects priority
            # CRITICAL (1) should execute before HIGH (2), etc.
            assert execution_order[0] == 1  # CRITICAL first
            assert execution_order[-1] == 4  # LOW last
    
    @pytest.mark.asyncio
    async def test_connection_pool_management(self, data_manager):
        """Test connection pool management voor resource efficiency"""
        # Verify connection pool limits
        assert data_manager.connection_pool_size > 0
        assert data_manager.connection_pool_size <= 100  # Reasonable limit
        
        # Test concurrent connection handling
        async def mock_request(*args, **kwargs):
            await asyncio.sleep(0.1)
            return {"data": "pooled_response"}
        
        requests = [
            DataRequest("test", "endpoint", {"id": i}, DataPriority.HIGH)
            for i in range(20)  # More than typical pool size
        ]
        
        with patch.object(data_manager, '_execute_request', mock_request):
            start_time = time.time()
            
            # Execute many concurrent requests
            responses = await asyncio.gather(*[data_manager.fetch_data(req) for req in requests])
            
            elapsed = time.time() - start_time
            
            # Should complete without errors
            assert len(responses) == 20
            assert all(r.status == "success" for r in responses)
            
            # Should not take too long (connection pooling should help)
            assert elapsed < 5.0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, data_manager):
        """CRITICAL: Circuit breaker functionality voor failing sources"""
        failure_count = 0
        
        async def failing_request(*args, **kwargs):
            nonlocal failure_count
            failure_count += 1
            raise aiohttp.ClientError("Service unavailable")
        
        request = DataRequest(
            source="failing_exchange",
            endpoint="ticker",
            params={"symbol": "ETH/USD"},
            priority=DataPriority.CRITICAL,
            retry_attempts=1  # Quick failure
        )
        
        with patch.object(data_manager, '_execute_request', failing_request):
            # Execute multiple requests to trigger circuit breaker
            for i in range(10):
                response = await data_manager.fetch_data(request)
                
                if i > 5:  # After several failures
                    # Circuit breaker should prevent further attempts
                    assert response.status == "circuit_open"
                    break
        
        # Verify circuit breaker activated
        assert data_manager.get_source_status("failing_exchange") == DataSourceStatus.FAILING
    
    def test_data_quality_validation(self, data_manager):
        """CRITICAL: Data quality validation"""
        # Test complete data
        complete_data = {
            "symbol": "ETH/USD",
            "price": 2000.0,
            "volume": 100000.0,
            "timestamp": time.time(),
            "bid": 1999.0,
            "ask": 2001.0
        }
        
        quality_score = data_manager.validate_data_quality(complete_data)
        assert quality_score >= 0.9  # Should be high quality
        
        # Test incomplete data
        incomplete_data = {
            "symbol": "ETH/USD",
            "price": 2000.0
            # Missing volume, timestamp, bid, ask
        }
        
        quality_score = data_manager.validate_data_quality(incomplete_data)
        assert quality_score < 0.7  # Should be lower quality
    
    def test_health_monitoring(self, data_manager):
        """Test health monitoring en status reporting"""
        # Add mock source
        data_manager.register_source("test_exchange", {"type": "ccxt", "id": "binance"})
        
        # Get health status
        health_status = data_manager.get_health_status()
        
        assert "test_exchange" in health_status
        assert "total_requests" in health_status["test_exchange"]
        assert "success_rate" in health_status["test_exchange"]
        assert "avg_latency_ms" in health_status["test_exchange"]
        assert "status" in health_status["test_exchange"]


class TestRobustExchangeConnector:
    """Test robust exchange connector implementation"""
    
    @pytest.fixture
    def exchange_connector(self):
        """Fresh exchange connector"""
        return RobustExchangeConnector("binance")
    
    @pytest.mark.asyncio
    async def test_connection_recovery(self, exchange_connector):
        """CRITICAL: Connection recovery na disconnection"""
        # Simulate connection failure
        exchange_connector.status = DataSourceStatus.OFFLINE
        
        # Mock successful reconnection
        async def mock_reconnect():
            exchange_connector.status = DataSourceStatus.HEALTHY
            return True
        
        with patch.object(exchange_connector, 'reconnect', mock_reconnect):
            result = await exchange_connector.ensure_connection()
            
            assert result is True
            assert exchange_connector.status == DataSourceStatus.HEALTHY
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, exchange_connector):
        """CRITICAL: API error handling en classification"""
        # Test different error types
        errors_to_test = [
            (aiohttp.ClientTimeout(), "timeout"),
            (aiohttp.ClientConnectorError(None, OSError()), "connection_error"),
            (aiohttp.ClientResponseError(None, None, status=429), "rate_limit"),
            (aiohttp.ClientResponseError(None, None, status=500), "server_error")
        ]
        
        for error, expected_classification in errors_to_test:
            classification = exchange_connector.classify_error(error)
            assert classification == expected_classification
    
    def test_performance_metrics_tracking(self, exchange_connector):
        """Test performance metrics tracking"""
        # Simulate some API calls
        exchange_connector.record_request_metrics("ticker", 0.150, True)  # 150ms success
        exchange_connector.record_request_metrics("orderbook", 0.350, False)  # 350ms failure
        exchange_connector.record_request_metrics("ticker", 0.200, True)  # 200ms success
        
        metrics = exchange_connector.get_performance_metrics()
        
        assert metrics["total_requests"] == 3
        assert metrics["success_rate"] == 2/3  # 2 out of 3 successful
        assert metrics["avg_latency_ms"] > 0
        assert "ticker" in metrics["endpoint_metrics"]
        assert "orderbook" in metrics["endpoint_metrics"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])