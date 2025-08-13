# config/structured_logging.py
import logging
import logging.handlers
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import traceback


class StructuredFormatter(logging.Formatter):
    """JSON structured logging formatter for enterprise systems"""

    def __init__(self, service_name: str = "cryptosmarttrader", version: str = "2.0.0"):
        self.service_name = service_name
        self.version = version
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""

        # Base log structure
        log_entry = {
            "@timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": self.service_name,
            "version": self.version,
            "thread": record.thread,
            "thread_name": record.threadName,
            "process": record.process,
            "filename": record.filename,
            "line_number": record.lineno,
            "function": record.funcName,
        }

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # Add extra fields from LogRecord
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
                "message",
                "exc_info",
                "exc_text",
                "stack_info",
                "getMessage",
            ]:
                extra_fields[key] = value

        if extra_fields:
            log_entry["extra"] = extra_fields

        return json.dumps(log_entry, default=str, ensure_ascii=False)


class AuditLogger:
    """Specialized logger for audit trails and compliance"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create audit logger
        self.logger = logging.getLogger("audit")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Setup audit log handler with rotation
        audit_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "audit.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
        )
        audit_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(audit_handler)

    def log_user_action(
        self,
        user_id: str,
        action: str,
        resource: str,
        result: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Log user actions for audit trail"""
        self.logger.info(
            f"User action: {action}",
            extra={
                "audit_type": "user_action",
                "user_id": user_id,
                "action": action,
                "resource": resource,
                "result": result,
                "details": details or {},
            },
        )

    def log_system_event(
        self,
        event_type: str,
        component: str,
        description: str,
        severity: str = "INFO",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log system events for monitoring"""
        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method(
            f"System event: {description}",
            extra={
                "audit_type": "system_event",
                "event_type": event_type,
                "component": component,
                "severity": severity,
                "metadata": metadata or {},
            },
        )

    def log_security_event(
        self,
        event_type: str,
        source_ip: str,
        description: str,
        risk_level: str = "LOW",
        additional_data: Optional[Dict[str, Any]] = None,
    ):
        """Log security events for threat monitoring"""
        self.logger.warning(
            f"Security event: {description}",
            extra={
                "audit_type": "security_event",
                "event_type": event_type,
                "source_ip": source_ip,
                "risk_level": risk_level,
                "description": description,
                "additional_data": additional_data or {},
            },
        )


class PerformanceLogger:
    """Logger for performance metrics and optimization"""

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Create performance logger
        self.logger = logging.getLogger("performance")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        # Setup performance log handler
        perf_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "performance.log",
            maxBytes=100 * 1024 * 1024,  # 100MB
            backupCount=5,
        )
        perf_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(perf_handler)

    def log_execution_time(
        self,
        component: str,
        operation: str,
        execution_time: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log execution times for performance analysis"""
        self.logger.info(
            f"Operation {operation} completed in {execution_time:.3f}s",
            extra={
                "metric_type": "execution_time",
                "component": component,
                "operation": operation,
                "execution_time": execution_time,
                "success": success,
                "metadata": metadata or {},
            },
        )

    def log_resource_usage(
        self,
        cpu_percent: float,
        memory_percent: float,
        disk_usage: float,
        active_threads: int,
        additional_metrics: Optional[Dict[str, float]] = None,
    ):
        """Log system resource usage"""
        self.logger.info(
            f"Resource usage - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%",
            extra={
                "metric_type": "resource_usage",
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "disk_usage": disk_usage,
                "active_threads": active_threads,
                "additional_metrics": additional_metrics or {},
            },
        )

    def log_ml_performance(
        self,
        model_name: str,
        accuracy: float,
        prediction_time: float,
        training_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Log ML model performance metrics"""
        self.logger.info(
            f"ML model {model_name} - Accuracy: {accuracy:.3f}, Prediction time: {prediction_time:.3f}s",
            extra={
                "metric_type": "ml_performance",
                "model_name": model_name,
                "accuracy": accuracy,
                "prediction_time": prediction_time,
                "training_time": training_time,
                "metadata": metadata or {},
            },
        )


def setup_structured_logging(
    service_name: str = "cryptosmarttrader",
    log_level: str = "INFO",
    log_dir: str = "logs",
    enable_console: bool = True,
    enable_file: bool = True,
) -> Dict[str, Any]:
    """Setup comprehensive structured logging system"""

    log_directory = Path(log_dir)
    log_directory.mkdir(exist_ok=True)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatters
    structured_formatter = StructuredFormatter(service_name)
    console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    handlers_created = []

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        handlers_created.append("console")

    # File handlers
    if enable_file:
        # Main application log
        file_handler = logging.handlers.RotatingFileHandler(
            log_directory / f"{service_name}.log",
            maxBytes=50 * 1024 * 1024,  # 50MB
            backupCount=10,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(structured_formatter)
        root_logger.addHandler(file_handler)
        handlers_created.append("file")

        # Error log (errors and above only)
        error_handler = logging.handlers.RotatingFileHandler(
            log_directory / f"{service_name}_errors.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(structured_formatter)
        root_logger.addHandler(error_handler)
        handlers_created.append("error_file")

    # Initialize specialized loggers
    audit_logger = AuditLogger(log_dir)
    performance_logger = PerformanceLogger(log_dir)

    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info(
        f"Structured logging initialized for {service_name}",
        extra={
            "component": "logging_system",
            "log_level": log_level,
            "log_directory": str(log_directory),
            "handlers": handlers_created,
            "audit_logging": True,
            "performance_logging": True,
        },
    )

    return {
        "service_name": service_name,
        "log_level": log_level,
        "log_directory": str(log_directory),
        "handlers": handlers_created,
        "audit_logger": audit_logger,
        "performance_logger": performance_logger,
    }


# Global instances
audit_logger = None
performance_logger = None


def get_audit_logger() -> AuditLogger:
    """Get global audit logger instance"""
    global audit_logger
    if audit_logger is None:
        audit_logger = AuditLogger()
    return audit_logger


def get_performance_logger() -> PerformanceLogger:
    """Get global performance logger instance"""
    global performance_logger
    if performance_logger is None:
        performance_logger = PerformanceLogger()
    return performance_logger
