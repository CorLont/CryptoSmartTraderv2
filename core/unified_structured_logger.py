#!/usr/bin/env python3
"""
Unified Structured Logger - Single consistent logging system
Prevents double-JSON issues and provides enterprise-grade logging
"""

import logging
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import threading
import time

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging - prevents double-JSON encoding"""
    
    def format(self, record):
        """Format log record as clean JSON without double encoding"""
        
        # Create clean log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add correlation ID if available
        if hasattr(record, 'correlation_id'):
            log_entry['correlation_id'] = record.correlation_id
        
        # Add extra fields if present
        if hasattr(record, '__dict__'):
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                              'filename', 'module', 'lineno', 'funcName', 'created', 
                              'msecs', 'relativeCreated', 'thread', 'threadName', 
                              'processName', 'process', 'getMessage', 'exc_info', 
                              'exc_text', 'stack_info', 'correlation_id']:
                    if not key.startswith('_'):
                        log_entry[key] = value
        
        # Handle exception info
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Return clean JSON - never double-encode
        return json.dumps(log_entry, default=str, ensure_ascii=False)

class UnifiedStructuredLogger:
    """Unified structured logger preventing double-JSON issues"""
    
    _instances = {}
    _lock = threading.Lock()
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self.correlation_id = None
        
        # Configure logger only once
        if not self.logger.handlers:
            self._configure_logger()
    
    def _configure_logger(self):
        """Configure logger with JSON formatter"""
        
        self.logger.setLevel(logging.INFO)
        
        # Console handler with JSON formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(JSONFormatter())
        self.logger.addHandler(console_handler)
        
        # File handler for daily logs
        try:
            log_dir = Path("logs/daily") / datetime.now().strftime("%Y%m%d")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            log_file = log_dir / f"{self.name.lower()}.jsonl"
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(JSONFormatter())
            self.logger.addHandler(file_handler)
            
        except Exception as e:
            # If file logging fails, continue with console only
            self.logger.warning(f"File logging setup failed: {e}")
        
        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for request tracking"""
        self.correlation_id = correlation_id
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log message with context and correlation ID"""
        
        extra = kwargs.copy()
        if self.correlation_id:
            extra['correlation_id'] = self.correlation_id
        
        self.logger.log(level, message, extra=extra)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = "", **kwargs):
        """Log performance metric"""
        self.info(f"Performance metric: {metric_name}",
                 metric_name=metric_name,
                 metric_value=value,
                 metric_unit=unit,
                 metric_type="performance",
                 **kwargs)
    
    def log_system_check(self, check_name: str, passed: bool, details: str = "", **kwargs):
        """Log system check result"""
        self.info(f"System check: {check_name}",
                 check_name=check_name,
                 check_passed=passed,
                 check_details=details,
                 check_type="system_validation",
                 **kwargs)
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error with full context"""
        self.error(f"Error occurred: {str(error)}",
                  error_type=type(error).__name__,
                  error_message=str(error),
                  context=context,
                  log_type="error_with_context")

def get_unified_logger(name: str) -> UnifiedStructuredLogger:
    """Get or create unified structured logger instance"""
    
    with UnifiedStructuredLogger._lock:
        if name not in UnifiedStructuredLogger._instances:
            UnifiedStructuredLogger._instances[name] = UnifiedStructuredLogger(name)
        return UnifiedStructuredLogger._instances[name]

# Alias for compatibility
def get_structured_logger(name: str) -> UnifiedStructuredLogger:
    """Compatibility alias for get_unified_logger"""
    return get_unified_logger(name)

# Configure root logger to prevent interference
def configure_root_logger():
    """Configure root logger to prevent interference with structured logging"""
    
    root_logger = logging.getLogger()
    
    # Remove existing handlers to prevent duplication
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Set level to WARNING to reduce noise
    root_logger.setLevel(logging.WARNING)
    
    # Add single handler for critical issues only
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

# Auto-configure on import
configure_root_logger()