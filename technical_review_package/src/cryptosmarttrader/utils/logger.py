import logging
import logging.handlers
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def setup_logger(
    name: str,
    log_level: str = "INFO",
    log_dir: str = "logs",
    max_bytes: int = 10485760,
    backup_count: int = 5,
) -> logging.Logger:
    """Setup enhanced logging with file rotation and console output"""

    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Set log level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    simple_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
    )

    # File handler with rotation
    log_file = log_path / f"{name}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)

    # Error file handler
    error_log_file = log_path / f"{name}_errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file, maxBytes=max_bytes, backupCount=backup_count
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(error_handler)

    return logger


class EnhancedLogger:
    """Enhanced logger with performance tracking and structured logging"""

    def __init__(self, name: str, config_manager=None):
        self.name = name
        self.config_manager = config_manager

        # Get log level from config or default to INFO
        log_level = "INFO"
        if config_manager:
            log_level = config_manager.get("log_level", "INFO")

        self.logger = setup_logger(name, log_level)

        # Performance tracking
        self.performance_logs = []
        self.error_count = 0
        self.warning_count = 0

    def debug(self, message: str, extra_data: Dict[str, Any] = None):
        """Log debug message with optional structured data"""
        self._log_with_extra(logging.DEBUG, message, extra_data)

    def info(self, message: str, extra_data: Dict[str, Any] = None):
        """Log info message with optional structured data"""
        self._log_with_extra(logging.INFO, message, extra_data)

    def warning(self, message: str, extra_data: Dict[str, Any] = None):
        """Log warning message with optional structured data"""
        self.warning_count += 1
        self._log_with_extra(logging.WARNING, message, extra_data)

    def error(self, message: str, extra_data: Dict[str, Any] = None):
        """Log error message with optional structured data"""
        self.error_count += 1
        self._log_with_extra(logging.ERROR, message, extra_data)

    def critical(self, message: str, extra_data: Dict[str, Any] = None):
        """Log critical message with optional structured data"""
        self.error_count += 1
        self._log_with_extra(logging.CRITICAL, message, extra_data)

    def _log_with_extra(self, level: int, message: str, extra_data: Dict[str, Any] = None):
        """Log message with extra structured data"""
        if extra_data:
            # Create structured log entry
            log_entry = {
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "logger": self.name,
                "extra_data": extra_data,
            }

            # Log both the message and structured data
            self.logger.log(level, message)
            self.logger.log(level, f"Extra data: {extra_data}")
        else:
            self.logger.log(level, message)

    def log_performance(
        self, operation: str, duration: float, additional_metrics: Dict[str, Any] = None
    ):
        """Log performance metrics"""
        perf_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "duration_seconds": duration,
            "logger": self.name,
        }

        if additional_metrics:
            perf_entry.update(additional_metrics)

        self.performance_logs.append(perf_entry)

        # Keep only last 1000 performance logs
        if len(self.performance_logs) > 1000:
            self.performance_logs = self.performance_logs[-1000:]

        # Log performance info
        self.info(f"Performance: {operation} completed in {duration:.4f}s", perf_entry)

    def log_api_call(
        self,
        endpoint: str,
        method: str,
        status_code: int,
        duration: float,
        response_size: int = None,
    ):
        """Log API call with details"""
        api_data = {
            "endpoint": endpoint,
            "method": method,
            "status_code": status_code,
            "duration_seconds": duration,
            "response_size_bytes": response_size,
        }

        if status_code >= 400:
            self.error(f"API call failed: {method} {endpoint} - {status_code}", api_data)
        else:
            self.debug(f"API call: {method} {endpoint} - {status_code}", api_data)

    def log_data_quality(
        self, dataset: str, quality_score: float, issues: list = None, record_count: int = None
    ):
        """Log data quality metrics"""
        quality_data = {
            "dataset": dataset,
            "quality_score": quality_score,
            "record_count": record_count,
            "issues": issues or [],
        }

        if quality_score < 0.7:
            self.warning(
                f"Data quality issue in {dataset}: score {quality_score:.2f}", quality_data
            )
        else:
            self.info(f"Data quality check: {dataset} score {quality_score:.2f}", quality_data)

    def log_ml_metrics(
        self, model_name: str, metrics: Dict[str, float], training_time: float = None
    ):
        """Log machine learning metrics"""
        ml_data = {
            "model_name": model_name,
            "metrics": metrics,
            "training_time_seconds": training_time,
        }

        self.info(f"ML metrics for {model_name}: {metrics}", ml_data)

        if training_time:
            self.log_performance(f"ML_training_{model_name}", training_time, ml_data)

    def log_trade_signal(
        self, symbol: str, signal: str, confidence: float, components: Dict[str, Any] = None
    ):
        """Log trading signal generation"""
        signal_data = {
            "symbol": symbol,
            "signal": signal,
            "confidence": confidence,
            "components": components or {},
        }

        self.info(
            f"Trade signal: {signal} for {symbol} (confidence: {confidence:.2f})", signal_data
        )

    def log_system_health(
        self, component: str, health_score: float, details: Dict[str, Any] = None
    ):
        """Log system health metrics"""
        health_data = {
            "component": component,
            "health_score": health_score,
            "details": details or {},
        }

        if health_score < 0.5:
            self.error(f"System health critical: {component} score {health_score:.2f}", health_data)
        elif health_score < 0.8:
            self.warning(
                f"System health degraded: {component} score {health_score:.2f}", health_data
            )
        else:
            self.debug(f"System health good: {component} score {health_score:.2f}", health_data)

    def get_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        recent_performance = [
            p
            for p in self.performance_logs
            if (datetime.now() - datetime.fromisoformat(p["timestamp"])).total_seconds() < 3600
        ]

        avg_duration = 0
        if recent_performance:
            avg_duration = sum(p["duration_seconds"] for p in recent_performance) / len(
                recent_performance
            )

        return {
            "logger_name": self.name,
            "total_performance_logs": len(self.performance_logs),
            "recent_performance_logs": len(recent_performance),
            "avg_operation_duration": avg_duration,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "log_level": self.logger.level,
        }

    def export_performance_logs(self, hours: int = 24) -> list:
        """Export performance logs for analysis"""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)

        return [
            log
            for log in self.performance_logs
            if datetime.fromisoformat(log["timestamp"]).timestamp() > cutoff_time
        ]

    def clear_old_logs(self, days: int = 30):
        """Clear old log files"""
        try:
            log_path = Path("logs")
            if not log_path.exists():
                return

            cutoff_time = datetime.now().timestamp() - (days * 24 * 3600)

            for log_file in log_path.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    self.info(f"Deleted old log file: {log_file}")

        except Exception as e:
            self.error(f"Error clearing old logs: {str(e)}")


# Convenience function for quick logger setup
def get_logger(name: str, config_manager=None) -> EnhancedLogger:
    """Get an enhanced logger instance"""
    return EnhancedLogger(name, config_manager)
