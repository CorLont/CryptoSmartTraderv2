#!/usr/bin/env python3
"""
Unit tests for StructuredLogger and logging functionality
"""

import pytest
import time
import uuid
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from core.logging_manager import get_logger, configure_logging


@pytest.mark.unit
class TestCorrelationContext:
    """Test correlation context functionality"""

    def test_correlation_context_creation(self):
        """Test correlation context creation with default values"""
        context = CorrelationContext(correlation_id="test-id", operation="test_operation")

        assert context.correlation_id == "test-id"
        assert context.operation == "test_operation"
        assert context.agent_name is None
        assert context.exchange is None
        assert context.start_time is not None

    def test_correlation_context_to_dict(self):
        """Test correlation context serialization"""
        context = CorrelationContext(
            correlation_id="test-id", operation="test_operation", agent_name="test_agent"
        )

        context_dict = context.to_dict()

        assert isinstance(context_dict, dict)
        assert context_dict["correlation_id"] == "test-id"
        assert context_dict["operation"] == "test_operation"
        assert context_dict["agent_name"] == "test_agent"
        assert "start_time" in context_dict


@pytest.mark.unit
class TestMetricsCollector:
    """Test Prometheus metrics collector"""

    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization"""
        collector = MetricsCollector()

        # Check that all required metrics are initialized
        assert collector.operation_duration is not None
        assert collector.operation_total is not None
        assert collector.data_collection_completeness is not None
        assert collector.api_requests_total is not None
        assert collector.system_memory_usage is not None
        assert collector.error_total is not None

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    def test_update_system_metrics(self, mock_cpu, mock_memory):
        """Test system metrics update"""
        # Mock system metrics
        mock_memory.return_value.used = 1024 * 1024 * 1024  # 1GB
        mock_cpu.return_value = 75.5

        collector = MetricsCollector()
        collector.update_system_metrics()

        # Verify that system metrics were set
        mock_memory.assert_called_once()
        mock_cpu.assert_called_once()


@pytest.mark.unit
class TestStructuredLogger:
    """Test structured logger functionality"""

    def test_structured_logger_singleton(self):
        """Test that StructuredLogger is a singleton"""
        logger1 = StructuredLogger()
        logger2 = StructuredLogger()

        assert logger1 is logger2

    @patch("core.logging_manager.start_http_server")
    def test_logger_initialization(self, mock_http_server, temp_dir):
        """Test logger initialization with configuration"""
        config = {"log_dir": str(temp_dir / "logs"), "metrics_port": 8091}

        logger = StructuredLogger(config)

        assert logger.log_dir == temp_dir / "logs"
        assert logger.metrics is not None
        mock_http_server.assert_called_once_with(8091, registry=logger.metrics.registry)

    def test_correlation_context_manager(self, temp_dir):
        """Test correlation context manager functionality"""
        reset_logging()

        config = {"log_dir": str(temp_dir / "logs"), "metrics_port": 8092}

        with patch("core.logging_manager.start_http_server"):
            logger = StructuredLogger(config)

        # Test context manager
        with logger.correlation_context("test_operation", "test_agent") as context:
            assert isinstance(context, CorrelationContext)
            assert context.operation == "test_operation"
            assert context.agent_name == "test_agent"
            assert context.correlation_id is not None

            # Test that context is stored in thread-local storage
            current_context = logger._get_log_context()
            assert current_context["correlation_id"] == context.correlation_id

    def test_correlation_context_exception_handling(self, temp_dir):
        """Test correlation context handling during exceptions"""
        reset_logging()

        config = {"log_dir": str(temp_dir / "logs"), "metrics_port": 8093}

        with patch("core.logging_manager.start_http_server"):
            logger = StructuredLogger(config)

        # Test exception handling in context
        with pytest.raises(ValueError):
            with logger.correlation_context("test_operation") as context:
                assert context is not None
                raise ValueError("Test exception")

        # Context should be properly cleaned up
        current_context = logger._get_log_context()
        assert current_context["correlation_id"] == "no-context"

    def test_logging_methods(self, temp_dir):
        """Test different logging methods"""
        reset_logging()

        config = {"log_dir": str(temp_dir / "logs"), "metrics_port": 8094}

        with (
            patch("core.logging_manager.start_http_server"),
            patch.object(StructuredLogger, "_log_with_context") as mock_log,
        ):
            logger = StructuredLogger(config)

            # Test different log levels
            logger.info("Test info message")
            logger.warning("Test warning message")
            logger.error("Test error message")
            logger.critical("Test critical message")
            logger.debug("Test debug message")

            # Verify all methods were called
            assert mock_log.call_count == 5

            # Check call arguments
            call_levels = [call[0][0] for call in mock_log.call_args_list]
            assert "INFO" in call_levels
            assert "WARNING" in call_levels
            assert "ERROR" in call_levels
            assert "CRITICAL" in call_levels
            assert "DEBUG" in call_levels

    def test_api_request_logging(self, temp_dir):
        """Test API request logging with metrics"""
        reset_logging()

        config = {"log_dir": str(temp_dir / "logs"), "metrics_port": 8095}

        with patch("core.logging_manager.start_http_server"):
            logger = StructuredLogger(config)

        # Mock metrics
        logger.metrics.api_request_duration = Mock()
        logger.metrics.api_requests_total = Mock()

        # Test API request logging
        logger.log_api_request("kraken", "tickers", 1.5, "success")

        # Verify metrics were updated
        logger.metrics.api_request_duration.labels.assert_called_with(
            exchange="kraken", endpoint="tickers"
        )
        logger.metrics.api_requests_total.labels.assert_called_with(
            exchange="kraken", endpoint="tickers", status="success"
        )

    def test_data_completeness_logging(self, temp_dir):
        """Test data completeness logging"""
        reset_logging()

        config = {"log_dir": str(temp_dir / "logs"), "metrics_port": 8096}

        with patch("core.logging_manager.start_http_server"):
            logger = StructuredLogger(config)

        # Mock metrics
        logger.metrics.data_collection_completeness = Mock()

        # Test data completeness logging
        logger.log_data_completeness("kraken", "tickers", 0.95)

        # Verify metrics were updated
        logger.metrics.data_collection_completeness.labels.assert_called_with(
            exchange="kraken", data_type="tickers"
        )

    def test_alert_system(self, temp_dir):
        """Test intelligent alerting system"""
        reset_logging()

        config = {"log_dir": str(temp_dir / "logs"), "metrics_port": 8097}

        with patch("core.logging_manager.start_http_server"):
            logger = StructuredLogger(config)

        # Mock app logger
        logger.app_logger = Mock()

        # Test alert triggering
        with patch.object(logger, "_trigger_alert") as mock_trigger:
            context = {"test": "context"}
            logger._check_alert_conditions("ERROR", context)

            # Should not trigger alert for single error
            mock_trigger.assert_not_called()

    def test_thread_safety(self, temp_dir):
        """Test thread safety of logging operations"""
        reset_logging()

        config = {"log_dir": str(temp_dir / "logs"), "metrics_port": 8098}

        with patch("core.logging_manager.start_http_server"):
            logger = StructuredLogger(config)

        results = []

        def logging_worker(worker_id):
            """Worker function for threading test"""
            with logger.correlation_context(f"operation_{worker_id}"):
                logger.info(f"Message from worker {worker_id}")
                context = logger._get_log_context()
                results.append(context["correlation_id"])

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=logging_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Each thread should have had its own correlation ID
        assert len(results) == 5
        assert len(set(results)) == 5  # All unique


@pytest.mark.unit
def test_get_logger_singleton():
    """Test get_logger returns singleton"""
    reset_logging()

    logger1 = get_logger()
    logger2 = get_logger()

    assert logger1 is logger2


@pytest.mark.unit
def test_configure_logging():
    """Test logging configuration"""
    reset_logging()

    config = {"log_dir": "/tmp/test_logs", "metrics_port": 8099}

    with patch("core.logging_manager.start_http_server"):
        logger = configure_logging(config)

    assert isinstance(logger, StructuredLogger)

    # Test reset
    reset_logging()

    # Should be able to get new logger after reset
    new_logger = get_logger()
    assert new_logger is not logger
