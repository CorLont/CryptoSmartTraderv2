#!/usr/bin/env python3
"""
Enterprise Logging Configuration - Structured JSON logging with correlation IDs
"""

import json
import logging
import sys
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from contextvars import ContextVar
from pathlib import Path

from python_json_logger import jsonlogger


# Context variables for correlation tracking
correlation_id_ctx: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
user_id_ctx: ContextVar[Optional[str]] = ContextVar("user_id", default=None)
request_path_ctx: ContextVar[Optional[str]] = ContextVar("request_path", default=None)


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID and context to log records"""

    def filter(self, record):
        # Add correlation ID
        record.correlation_id = correlation_id_ctx.get()
        record.user_id = user_id_ctx.get()
        record.request_path = request_path_ctx.get()

        # Add timestamp in ISO format
        record.timestamp_iso = datetime.utcnow().isoformat() + "Z"

        # Add service name
        record.service = "cryptosmarttrader"

        return True


class TradingEventFilter(logging.Filter):
    """Special filter for trading-related events"""

    def filter(self, record):
        # Only pass trading-related logs
        trading_keywords = ["trade", "order", "signal", "position", "pnl", "slippage", "fee"]

        message = record.getMessage().lower()
        return any(keyword in message for keyword in trading_keywords)


class SecurityFilter(logging.Filter):
    """Filter to redact sensitive information"""

    SENSITIVE_FIELDS = ["api_key", "secret", "password", "token", "auth"]

    def filter(self, record):
        # Redact sensitive information from message
        message = record.getMessage()

        for field in self.SENSITIVE_FIELDS:
            if field in message.lower():
                # Replace with asterisks
                record.msg = "[REDACTED - SENSITIVE DATA]"
                break

        return True


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with enterprise fields"""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)

        # Add standard enterprise fields
        log_record["@timestamp"] = getattr(
            record, "timestamp_iso", datetime.utcnow().isoformat() + "Z"
        )
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["service"] = getattr(record, "service", "cryptosmarttrader")
        log_record["correlation_id"] = getattr(record, "correlation_id", None)
        log_record["user_id"] = getattr(record, "user_id", None)
        log_record["request_path"] = getattr(record, "request_path", None)

        # Add file and line info for debugging
        if hasattr(record, "pathname"):
            log_record["file"] = Path(record.pathname).name
            log_record["line"] = record.lineno
            log_record["function"] = record.funcName

        # Add thread info
        log_record["thread"] = record.thread
        log_record["thread_name"] = record.threadName

        # Add process info
        log_record["process"] = record.process

        # Clean up None values
        log_record = {k: v for k, v in log_record.items() if v is not None}


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    enable_json: bool = True,
    enable_trading_logs: bool = True,
):
    """
    Setup enterprise logging configuration

    Args:
        log_level: Logging level
        log_file: Optional log file path
        enable_json: Use JSON formatting
        enable_trading_logs: Enable separate trading event logs
    """

    # Clear existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set log level
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Create formatters
    if enable_json:
        formatter = CustomJsonFormatter(format="%(timestamp_iso)s %(level)s %(name)s %(message)s")
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(CorrelationIdFilter())
    console_handler.addFilter(SecurityFilter())
    root_logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.addFilter(CorrelationIdFilter())
        file_handler.addFilter(SecurityFilter())
        root_logger.addHandler(file_handler)

    # Trading events handler
    if enable_trading_logs:
        trading_handler = logging.FileHandler("logs/trading_events.jsonl")
        trading_handler.setFormatter(formatter)
        trading_handler.addFilter(CorrelationIdFilter())
        trading_handler.addFilter(TradingEventFilter())
        trading_handler.setLevel(logging.INFO)

        # Create trading logger
        trading_logger = logging.getLogger("trading")
        trading_logger.addHandler(trading_handler)
        trading_logger.setLevel(logging.INFO)

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    logging.info("Logging configuration initialized")


def set_correlation_id(correlation_id: Optional[str] = None) -> str:
    """Set correlation ID for current context"""
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())

    correlation_id_ctx.set(correlation_id)
    return correlation_id


def set_user_context(user_id: Optional[str] = None, request_path: Optional[str] = None):
    """Set user context for logging"""
    if user_id:
        user_id_ctx.set(user_id)
    if request_path:
        request_path_ctx.set(request_path)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID"""
    return correlation_id_ctx.get()


def log_trading_event(
    event_type: str, symbol: str, data: Dict[str, Any], logger: Optional[logging.Logger] = None
):
    """
    Log structured trading event

    Args:
        event_type: Type of event (trade, signal, order, etc.)
        symbol: Trading symbol
        data: Event data
        logger: Optional logger (defaults to trading logger)
    """

    if logger is None:
        logger = logging.getLogger("trading")

    event_data = {
        "event_type": event_type,
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        **data,
    }

    logger.info(f"Trading event: {event_type} for {symbol}", extra={"trading_event": event_data})


def log_signal_event(
    signal_type: str,
    symbol: str,
    confidence: float,
    direction: str,
    price: float,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Log trading signal event"""

    data = {
        "signal_type": signal_type,
        "confidence": confidence,
        "direction": direction,
        "price": price,
        "metadata": metadata or {},
    }

    log_trading_event("signal", symbol, data)


def log_order_event(
    order_id: str,
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    order_type: str,
    status: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Log order event"""

    data = {
        "order_id": order_id,
        "side": side,
        "quantity": quantity,
        "price": price,
        "order_type": order_type,
        "status": status,
        "metadata": metadata or {},
    }

    log_trading_event("order", symbol, data)


def log_trade_execution(
    trade_id: str,
    symbol: str,
    side: str,
    quantity: float,
    executed_price: float,
    slippage: float,
    fees: float,
    pnl: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Log trade execution event"""

    data = {
        "trade_id": trade_id,
        "side": side,
        "quantity": quantity,
        "executed_price": executed_price,
        "slippage": slippage,
        "fees": fees,
        "pnl": pnl,
        "metadata": metadata or {},
    }

    log_trading_event("trade_execution", symbol, data)


def log_position_update(
    symbol: str,
    position_size: float,
    avg_price: float,
    unrealized_pnl: float,
    realized_pnl: float,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Log position update event"""

    data = {
        "position_size": position_size,
        "avg_price": avg_price,
        "unrealized_pnl": unrealized_pnl,
        "realized_pnl": realized_pnl,
        "metadata": metadata or {},
    }

    log_trading_event("position_update", symbol, data)


def log_risk_event(
    event_type: str,
    severity: str,
    message: str,
    symbol: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Log risk management event"""

    data = {"severity": severity, "message": message, "metadata": metadata or {}}

    log_trading_event(f"risk_{event_type}", symbol or "SYSTEM", data)


# Convenience loggers
def get_logger(name: str) -> logging.Logger:
    """Get logger with correlation ID support"""
    logger = logging.getLogger(name)
    return logger


# Module-level convenience functions
def info(message: str, **kwargs):
    """Log info message with correlation"""
    logging.getLogger(__name__).info(message, extra=kwargs)


def warning(message: str, **kwargs):
    """Log warning message with correlation"""
    logging.getLogger(__name__).warning(message, extra=kwargs)


def error(message: str, **kwargs):
    """Log error message with correlation"""
    logging.getLogger(__name__).error(message, extra=kwargs)


def debug(message: str, **kwargs):
    """Log debug message with correlation"""
    logging.getLogger(__name__).debug(message, extra=kwargs)
