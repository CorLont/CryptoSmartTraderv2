"""Tests for structured logging functionality."""

import pytest
import json
import logging
from io import StringIO
from unittest.mock import patch

from src.cryptosmarttrader.core.structured_logger import (
    StructuredLogger,
    StructuredFormatter,
    SecurityFilter,
    get_logger,
)


class TestStructuredFormatter:
    """Test structured formatter functionality."""

    def test_format_log_record(self):
        """Test formatting of log records."""
        formatter = StructuredFormatter()

        # Create a test log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert parsed["level"] == "INFO"
        assert parsed["message"] == "Test message"
        assert parsed["module"] == "test"
        assert parsed["line"] == 42
        assert "timestamp" in parsed
        assert "correlation_id" in parsed


class TestSecurityFilter:
    """Test security filtering functionality."""

    def test_filter_sensitive_data(self):
        """Test filtering of sensitive information."""
        security_filter = SecurityFilter()

        # Create record with sensitive data
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="API key is secret_key_123",
            args=(),
            exc_info=None,
        )

        # Filter should modify the record
        result = security_filter.filter(record)

        assert result is True
        assert "REDACTED" in record.msg
        assert "secret_key_123" not in record.msg


class TestStructuredLogger:
    """Test structured logger functionality."""

    def test_logger_initialization(self):
        """Test logger initialization."""
        logger = StructuredLogger("test_logger")

        assert logger.name == "test_logger"
        assert logger.logger.level == logging.INFO
        assert len(logger.logger.handlers) > 0

    def test_correlation_id_management(self):
        """Test correlation ID functionality."""
        logger = StructuredLogger("test_logger")

        # Test setting correlation ID
        cid = logger.set_correlation_id("test-123")
        assert cid == "test-123"
        assert logger.get_correlation_id() == "test-123"

        # Test auto-generated correlation ID
        auto_cid = logger.set_correlation_id()
        assert auto_cid is not None
        assert len(auto_cid) == 36  # UUID format

        # Test clearing correlation ID
        logger.clear_correlation_id()
        assert logger.get_correlation_id() is None

    @pytest.mark.unit
    def test_performance_timer(self):
        """Test performance timing functionality."""
        logger = StructuredLogger("test_logger")

        with logger.performance_timer("test_operation") as timer:
            assert timer.operation == "test_operation"
            # Timer should complete without error

    def test_logging_methods(self):
        """Test various logging methods."""
        # Capture log output
        stream = StringIO()
        handler = logging.StreamHandler(stream)

        logger = StructuredLogger("test_logger", enable_console=False)
        logger.logger.addHandler(handler)

        # Test different log levels
        logger.info("Info message", extra_field="value")
        logger.warning("Warning message")
        logger.error("Error message")

        output = stream.getvalue()
        assert "Info message" in output
        assert "Warning message" in output
        assert "Error message" in output

    def test_audit_logging(self):
        """Test audit trail logging."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)

        logger = StructuredLogger("test_logger", enable_console=False)
        logger.logger.addHandler(handler)

        logger.log_user_action("login", user_id="user123", resource="dashboard")

        output = stream.getvalue()
        assert "User action: login" in output
        assert "user123" in output

    def test_security_event_logging(self):
        """Test security event logging."""
        stream = StringIO()
        handler = logging.StreamHandler(stream)

        logger = StructuredLogger("test_logger", enable_console=False)
        logger.logger.addHandler(handler)

        logger.log_security_event("failed_login", severity="warning", source_ip="192.168.1.1")

        output = stream.getvalue()
        assert "Security event: failed_login" in output
        assert "192.168.1.1" in output


def test_get_logger_factory():
    """Test logger factory function."""
    logger = get_logger("factory_test")

    assert isinstance(logger, StructuredLogger)
    assert logger.name == "factory_test"
