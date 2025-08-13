#!/usr/bin/env python3
"""
Integration tests for data collection workflow
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime
import json

from agents.data_collector import AsyncDataCollectorAgent
from core.async_data_manager import AsyncDataManager
from core.dependency_container import Container


@pytest.mark.integration
@pytest.mark.asyncio
class TestDataCollectionIntegration:
    """Integration tests for data collection workflow"""

    async def test_complete_data_collection_workflow(
        self, async_test_context, mock_kraken_data, mock_binance_data, test_data_factory
    ):
        """Test complete data collection workflow from agent to storage"""
        container = async_test_context["container"]
        mock_data_manager = async_test_context["data_manager"]

        # Mock data manager responses
        mock_market_data = {
            "timestamp": datetime.now().isoformat(),
            "exchanges": {
                "kraken": {"tickers": mock_kraken_data["result"]},
                "binance": {"tickers": {"BTCUSDT": mock_binance_data}},
            },
            "summary": {"successful": 2, "failed": 0},
        }

        mock_data_manager.batch_collect_all_exchanges.return_value = mock_market_data

        # Create agent with DI
        from dependency_injector.wiring import Provide

        @pytest.fixture
        async def create_agent():
            agent = AsyncDataCollectorAgent(
                config={"collection_interval": 1},  # Fast for testing
                settings=container.config(),
                data_manager=mock_data_manager,
                rate_limit_config=container.rate_limit_config(),
            )
            return agent

        # Test agent initialization
        agent = await create_agent()
        assert agent is not None
        assert agent.name == "async_data_collector"

        # Test data collection
        collected_data = await agent.collect_comprehensive_market_data()

        assert collected_data is not None
        assert "timestamp" in collected_data
        assert "exchanges" in collected_data
        assert collected_data["summary"]["successful"] == 2

        # Verify data manager was called
        mock_data_manager.batch_collect_all_exchanges.assert_called_once()

    async def test_agent_error_handling_and_recovery(self, async_test_context, test_data_factory):
        """Test agent error handling and recovery mechanisms"""
        container = async_test_context["container"]
        mock_data_manager = async_test_context["data_manager"]

        # First call fails, second succeeds
        mock_data_manager.batch_collect_all_exchanges.side_effect = [
            Exception("Network error"),
            test_data_factory.create_market_data(),
        ]

        agent = AsyncDataCollectorAgent(
            config={"collection_interval": 0.1},
            settings=container.config(),
            data_manager=mock_data_manager,
            rate_limit_config=container.rate_limit_config(),
        )

        # Test that error is handled gracefully
        result1 = await agent.collect_comprehensive_market_data()
        assert "error" in result1
        assert result1["summary"]["successful"] == 0

        # Test recovery on next call
        result2 = await agent.collect_comprehensive_market_data()
        assert "error" not in result2

    async def test_concurrent_agent_operations(self, async_test_context, mock_kraken_data):
        """Test concurrent operations between multiple agents"""
        container = async_test_context["container"]

        # Create multiple data managers for concurrent testing
        mock_managers = [AsyncMock() for _ in range(3)]

        for i, manager in enumerate(mock_managers):
            manager.batch_collect_all_exchanges.return_value = {
                "timestamp": datetime.now().isoformat(),
                "exchanges": {"test": f"data_{i}"},
                "summary": {"successful": 1, "failed": 0},
            }

        # Create multiple agents
        agents = []
        for i, manager in enumerate(mock_managers):
            agent = AsyncDataCollectorAgent(
                config={"collection_interval": 0.1},
                settings=container.config(),
                data_manager=manager,
                rate_limit_config=container.rate_limit_config(),
            )
            agents.append(agent)

        # Run concurrent data collection
        tasks = [agent.collect_comprehensive_market_data() for agent in agents]
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result["exchanges"]["test"] == f"data_{i}"

    async def test_data_completeness_validation(self, async_test_context, test_data_factory):
        """Test data completeness validation and metrics"""
        container = async_test_context["container"]
        mock_data_manager = async_test_context["data_manager"]

        # Mock partial data (some exchanges fail)
        partial_data = {
            "timestamp": datetime.now().isoformat(),
            "exchanges": {
                "kraken": {"tickers": {"BTC/USD": "data"}},
                "binance": {"error": "API Error"},
                "coinbase": {"tickers": {}},  # Empty data
            },
            "summary": {"successful": 1, "failed": 2},
        }

        mock_data_manager.batch_collect_all_exchanges.return_value = partial_data

        agent = AsyncDataCollectorAgent(
            config={"collection_interval": 0.1},
            settings=container.config(),
            data_manager=mock_data_manager,
            rate_limit_config=container.rate_limit_config(),
        )

        result = await agent.collect_comprehensive_market_data()

        # Should calculate completeness metrics
        assert "data_completeness" in result
        completeness = result["data_completeness"]

        # Should be less than 100% due to failures
        assert completeness["overall"] < 1.0
        assert "per_exchange" in completeness

    async def test_circuit_breaker_integration(self, async_test_context):
        """Test circuit breaker behavior during consecutive failures"""
        container = async_test_context["container"]
        mock_data_manager = async_test_context["data_manager"]

        # Configure to always fail
        mock_data_manager.batch_collect_all_exchanges.side_effect = Exception("Persistent error")

        agent = AsyncDataCollectorAgent(
            config={"collection_interval": 0.01},  # Very fast for testing
            settings=container.config(),
            data_manager=mock_data_manager,
            rate_limit_config=container.rate_limit_config(),
        )

        # Mock the main loop to stop after a few iterations
        original_running = agent.running
        call_count = 0

        def mock_running():
            nonlocal call_count
            call_count += 1
            return call_count < 8  # Stop after 8 calls

        agent.__class__.running = property(mock_running)

        # Mock sleep to speed up test
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # Run main loop (should trigger circuit breaker)
            try:
                await agent.main_loop()
            except Exception:
                pass  # Expected due to persistent failures

        # Should have attempted multiple retries and triggered circuit breaker
        assert mock_data_manager.batch_collect_all_exchanges.call_count >= 5

        # Should have applied exponential backoff
        mock_sleep.assert_called()
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list if call[0]]
        assert any(duration >= 60 for duration in sleep_calls)  # Circuit breaker duration

        # Restore original property
        agent.__class__.running = original_running


@pytest.mark.integration
@pytest.mark.asyncio
class TestAsyncDataManagerIntegration:
    """Integration tests for AsyncDataManager with real-like scenarios"""

    async def test_exchange_setup_integration(self, test_container, test_secrets_manager):
        """Test exchange setup with dependency injection"""
        # Configure test secrets
        test_secrets_manager.secrets_cache.update(
            {"KRAKEN_API_KEY": "test_key", "KRAKEN_SECRET": "test_secret"}
        )

        rate_config = test_container.rate_limit_config()
        manager = AsyncDataManager(rate_config)

        with (
            patch("ccxt.async_support.kraken") as mock_kraken,
            patch("ccxt.async_support.binance") as mock_binance,
        ):
            mock_kraken_instance = AsyncMock()
            mock_binance_instance = AsyncMock()
            mock_kraken.return_value = mock_kraken_instance
            mock_binance.return_value = mock_binance_instance

            # Initialize session
            manager.session = AsyncMock()
            manager.structured_logger = Mock()
            manager.structured_logger.info = Mock()
            manager.structured_logger.warning = Mock()

            await manager.setup_async_exchanges()

            # Verify proper exchange configuration
            assert "kraken" in manager.exchanges
            assert "binance" in manager.exchanges

            # Kraken should be configured with credentials
            kraken_call_args = mock_kraken.call_args[0][0]
            assert "apiKey" in kraken_call_args
            assert kraken_call_args["apiKey"] == "test_key"

    async def test_rate_limiting_integration(self, test_rate_limit_config):
        """Test rate limiting with multiple concurrent requests"""
        # Use strict rate limiting for testing
        strict_config = RateLimitConfig(requests_per_second=2.0, burst_size=3, timeout_seconds=5)

        manager = AsyncDataManager(strict_config)

        # Mock exchange for testing
        mock_exchange = AsyncMock()
        mock_exchange.name = "test"
        mock_exchange.fetch_ohlcv.return_value = [[1, 2, 3, 4, 5, 6]]

        manager.structured_logger = Mock()
        manager.structured_logger.log_api_request = Mock()

        # Make multiple concurrent requests
        start_time = asyncio.get_event_loop().time()

        tasks = [
            manager.fetch_single_ohlcv_async(mock_exchange, "BTC/USD", "1h")
            for _ in range(6)  # More than burst size
        ]

        results = await asyncio.gather(*tasks)

        end_time = asyncio.get_event_loop().time()
        duration = end_time - start_time

        # All requests should succeed
        assert all(result == [[1, 2, 3, 4, 5, 6]] for result in results)

        # Should take time due to rate limiting
        assert duration > 1.0  # Should be rate limited

        # Should have logged all API requests
        assert manager.structured_logger.log_api_request.call_count == 6

    async def test_error_handling_and_metrics_integration(self, test_rate_limit_config):
        """Test error handling integration with metrics and logging"""
        manager = AsyncDataManager(test_rate_limit_config)

        # Mock structured logger
        manager.structured_logger = Mock()
        manager.structured_logger.log_api_request = Mock()
        manager.structured_logger.warning = Mock()

        # Create exchange that fails
        failing_exchange = AsyncMock()
        failing_exchange.name = "failing_exchange"
        failing_exchange.fetch_ohlcv.side_effect = Exception("API Error")

        result = await manager.fetch_single_ohlcv_async(failing_exchange, "BTC/USD", "1h")

        # Should return empty list on failure
        assert result == []

        # Should log the error
        manager.structured_logger.warning.assert_called_once()

        # Should log API request with error status
        manager.structured_logger.log_api_request.assert_called_once()
        call_args = manager.structured_logger.log_api_request.call_args
        assert call_args[1]["status"] == "error"
