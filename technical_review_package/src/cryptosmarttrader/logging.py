"""
Enterprise Logging Configuration with Structured JSON Logging
Centralized logging setup with correlation IDs and log rotation
"""

import logging
import logging.handlers
import sys
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import uuid


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def __init__(self, service_name: str = "cryptosmarttrader"):
        self.service_name = service_name
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""

        # Base log structure
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "service": self.service_name,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation ID if available
        if hasattr(record, "correlation_id"):
            log_entry["correlation_id"] = getattr(record, "correlation_id", None)

        # Add exception info if present
        if record.exc_info and record.exc_info[0] is not None:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        # Add extra fields from record
        extra_fields = {}
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
                "correlation_id",
            ]:
                extra_fields[key] = value

        if extra_fields:
            log_entry["extra"] = extra_fields

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class CorrelationFilter(logging.Filter):
    """Add correlation ID to log records"""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID if not already present"""
        if not hasattr(record, "correlation_id"):
            record.correlation_id = str(uuid.uuid4())[:8]
        return True


def setup_simple_logging(level: str = "INFO") -> None:
    """Eenvoudige, consistente logging-setup (stdout, niveau en format) - PR2 Style"""
    root = logging.getLogger()
    # verwijder bestaande handlers
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    fmt = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
    datefmt = "%Y-%m-%dT%H:%M:%S"
    handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))

    root.addHandler(handler)
    root.setLevel(level.upper())


def setup_logging(
    level: str = "INFO",
    log_dir: Path = Path("logs"),
    service_name: str = "cryptosmarttrader",
    enable_console: bool = True,
    enable_file: bool = True,
    max_bytes: int = 50 * 1024 * 1024,  # 50MB
    backup_count: int = 10,
) -> logging.Logger:
    """
    Setup enterprise logging configuration with structured JSON output

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        service_name: Service name for log identification
        enable_console: Enable console logging
        enable_file: Enable file logging
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep

    Returns:
        Configured root logger
    """

    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)

    # Clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set log level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Create formatters
    json_formatter = StructuredFormatter(service_name)
    console_formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Add correlation filter
    correlation_filter = CorrelationFilter()

    # Console handler (human-readable for development)
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(console_formatter)
        console_handler.addFilter(correlation_filter)
        root_logger.addHandler(console_handler)

    # File handler (JSON for production)
    if enable_file:
        log_file = log_dir / "app.log"
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(json_formatter)
        file_handler.addFilter(correlation_filter)
        root_logger.addHandler(file_handler)

    # Error file handler (errors and criticals only)
    if enable_file:
        error_file = log_dir / "error.log"
        error_handler = logging.handlers.RotatingFileHandler(
            filename=error_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(json_formatter)
        error_handler.addFilter(correlation_filter)
        root_logger.addHandler(error_handler)

    # Configure third-party loggers
    configure_third_party_loggers(numeric_level)

    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured successfully",
        extra={
            "log_level": level,
            "log_dir": str(log_dir),
            "service": service_name,
            "console_enabled": enable_console,
            "file_enabled": enable_file,
        },
    )

    return root_logger


def configure_third_party_loggers(level: int):
    """Configure third-party library loggers"""

    # Reduce noise from third-party libraries
    noisy_loggers = [
        "urllib3.connectionpool",
        "requests.packages.urllib3",
        "ccxt.base.exchange",
        "matplotlib",
        "PIL",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(max(level, logging.WARNING))

    # Set specific levels for key libraries
    logging.getLogger("streamlit").setLevel(max(level, logging.WARNING))
    logging.getLogger("uvicorn").setLevel(max(level, logging.INFO))
    logging.getLogger("fastapi").setLevel(max(level, logging.INFO))


def get_logger(name: str) -> logging.Logger:
    """Get a logger with correlation support"""
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding correlation ID to logs"""

    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())[:8]
        self.old_factory = logging.getLogRecordFactory()

    def __enter__(self):
        def record_factory(*args, **kwargs):
            record = self.old_factory(*args, **kwargs)
            record.correlation_id = self.correlation_id
            return record

        logging.setLogRecordFactory(record_factory)
        return self.correlation_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


# Trading-specific logging helpers
def log_trading_event(
    event_type: str, symbol: str, data: Dict[str, Any], correlation_id: Optional[str] = None
):
    """Log structured trading events"""
    logger = get_logger("trading")
    logger.info(
        f"Trading event: {event_type}",
        extra={
            "event_type": event_type,
            "symbol": symbol,
            "trading_data": data,
            "correlation_id": correlation_id or str(uuid.uuid4())[:8],
        },
    )


def log_performance_metric(
    metric_name: str, value: float, unit: str = "", tags: Optional[Dict[str, str]] = None
):
    """Log performance metrics"""
    logger = get_logger("performance")
    logger.info(
        f"Performance metric: {metric_name}",
        extra={"metric_name": metric_name, "value": value, "unit": unit, "tags": tags or {}},
    )


def log_security_event(event_type: str, details: Dict[str, Any]):
    """Log security-related events"""
    logger = get_logger("security")
    logger.warning(
        f"Security event: {event_type}", extra={"security_event": event_type, "details": details}
    )
