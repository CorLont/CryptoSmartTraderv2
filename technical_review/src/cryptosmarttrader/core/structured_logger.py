"""Enterprise structured logging with correlation IDs and security."""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Union
from pathlib import Path
import threading
from contextvars import ContextVar

# Context variable for correlation tracking
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "thread_id": threading.get_ident(),
            "correlation_id": correlation_id.get(),
        }

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields from the record
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info'):
                log_entry[key] = value

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class SecurityFilter(logging.Filter):
    """Filter to prevent logging of sensitive information."""

    SENSITIVE_KEYS = {
        'password', 'passwd', 'secret', 'token', 'key', 'api_key',
        'authorization', 'auth', 'credentials', 'private_key'
    }

    def filter(self, record: logging.LogRecord) -> bool:
        """Filter out sensitive information from log records."""
        message = record.getMessage().lower()

        # Check if any sensitive keywords are in the message
        for sensitive_key in self.SENSITIVE_KEYS:
            if sensitive_key in message:
                # Replace the entire message with a security notice
                record.msg = f"[REDACTED] Potential sensitive information in {record.funcName}"
                record.args = ()
                break

        return True


class PerformanceLogger:
    """Context manager for performance timing and logging."""

    def __init__(self, operation: str, logger: logging.Logger):
        """Initialize performance logger."""
        self.operation = operation
        self.logger = logger
        self.start_time: Optional[float] = None

    def __enter__(self) -> 'PerformanceLogger':
        """Start timing the operation."""
        self.start_time = time.perf_counter()
        self.logger.info(f"Starting operation: {self.operation}")
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Log operation completion with timing."""
        if self.start_time is not None:
            elapsed = time.perf_counter() - self.start_time

            if exc_type is None:
                self.logger.info(
                    f"Operation completed: {self.operation}",
                    extra={
                        "operation": self.operation,
                        "duration_ms": round(elapsed * 1000, 2),
                        "status": "success"
                    }
                )
            else:
                self.logger.error(
                    f"Operation failed: {self.operation}",
                    extra={
                        "operation": self.operation,
                        "duration_ms": round(elapsed * 1000, 2),
                        "status": "error",
                        "error_type": exc_type.__name__ if exc_type else None
                    }
                )


class StructuredLogger:
    """Enterprise structured logger with correlation IDs and security filtering."""

    def __init__(
        self,
        name: str,
        log_level: Union[str, int] = logging.INFO,
        log_file: Optional[Union[str, Path]] = None,
        enable_console: bool = True,
        enable_security_filter: bool = True
    ) -> None:
        """Initialize structured logger."""
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)

        # Clear existing handlers to avoid duplication
        self.logger.handlers.clear()

        # Configure structured formatter
        formatter = StructuredFormatter()

        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            if enable_security_filter:
                console_handler.addFilter(SecurityFilter())
            self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            if enable_security_filter:
                file_handler.addFilter(SecurityFilter())
            self.logger.addHandler(file_handler)

    def set_correlation_id(self, cid: Optional[str] = None) -> str:
        """Set correlation ID for request tracking."""
        if cid is None:
            cid = str(uuid.uuid4())
        correlation_id.set(cid)
        return cid

    def get_correlation_id(self) -> Optional[str]:
        """Get current correlation ID."""
        return correlation_id.get()

    def clear_correlation_id(self) -> None:
        """Clear correlation ID."""
        correlation_id.set(None)

    def performance_timer(self, operation: str) -> PerformanceLogger:
        """Create a performance timing context manager."""
        return PerformanceLogger(operation, self.logger)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with structured data."""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with structured data."""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with structured data."""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with structured data."""
        self.logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message with structured data."""
        self.logger.critical(message, extra=kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback and structured data."""
        self.logger.exception(message, extra=kwargs)

    def log_user_action(
        self,
        action: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Log user action for audit trail."""
        self.info(
            f"User action: {action}",
            action=action,
            user_id=user_id,
            resource=resource,
            audit=True,
            **kwargs
        )

    def log_system_event(
        self,
        event: str,
        component: str,
        status: str = "info",
        **kwargs: Any
    ) -> None:
        """Log system event for monitoring."""
        log_method = getattr(self.logger, status.lower(), self.logger.info)
        log_method(
            f"System event: {event}",
            event=event,
            component=component,
            system=True,
            **kwargs
        )

    def log_security_event(
        self,
        event: str,
        severity: str = "warning",
        source_ip: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Log security-related event."""
        log_method = getattr(self.logger, severity.lower(), self.logger.warning)
        log_method(
            f"Security event: {event}",
            event=event,
            severity=severity,
            source_ip=source_ip,
            security=True,
            **kwargs
        )

    def log_performance_metric(
        self,
        metric_name: str,
        value: Union[int, float],
        unit: str = "ms",
        **kwargs: Any
    ) -> None:
        """Log performance metric."""
        self.info(
            f"Performance metric: {metric_name}",
            metric_name=metric_name,
            value=value,
            unit=unit,
            metric=True,
            **kwargs
        )


def get_logger(
    name: str,
    log_level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None
) -> StructuredLogger:
    """Factory function to create structured logger instances."""
    return StructuredLogger(
        name=name,
        log_level=log_level,
        log_file=log_file
    )


def setup_root_logger(log_level: Union[str, int] = logging.INFO) -> None:
    """Setup root logger with structured formatting."""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Add structured console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(StructuredFormatter())
    console_handler.addFilter(SecurityFilter())
    root_logger.addHandler(console_handler)
