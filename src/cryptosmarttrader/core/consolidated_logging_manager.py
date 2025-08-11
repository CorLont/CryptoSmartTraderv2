#!/usr/bin/env python3
"""
Consolidated Logging Manager - Single Logging System
Replaces all duplicate logging systems to prevent inconsistent observability
"""

import logging
import json
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
import threading
from logging.handlers import RotatingFileHandler

class ConsolidatedLogger:
    """Single, consistent logger for entire system"""
    
    _instances = {}
    _lock = threading.Lock()
    _initialized = False
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        
        # Configure only once globally
        if not ConsolidatedLogger._initialized:
            self._setup_global_logging()
            ConsolidatedLogger._initialized = True
    
    def _setup_global_logging(self):
        """Setup global logging configuration once"""
        
        # Clear any existing handlers to prevent duplicates
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set root logger level
        root_logger.setLevel(logging.WARNING)
        
        # Create formatters
        json_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": %(message)s, "module": "%(module)s", "function": "%(funcName)s", "line": %(lineno)d}'
        )
        
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(console_formatter)
        
        # Daily log file handler
        try:
            log_dir = Path("logs/consolidated")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            today = datetime.now().strftime("%Y%m%d")
            log_file = log_dir / f"system_{today}.log"
            
            file_handler = RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
            )
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(json_formatter)
            
            # Add handlers to root logger
            root_logger.addHandler(console_handler)
            root_logger.addHandler(file_handler)
            
        except Exception as e:
            # If file logging fails, continue with console only
            print(f"Warning: File logging setup failed: {e}")
            root_logger.addHandler(console_handler)
        
        # Prevent double logging
        root_logger.propagate = False
    
    def info(self, message: str, **kwargs):
        """Log info message with consistent format"""
        self._log_with_context(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with consistent format"""
        self._log_with_context(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with consistent format"""
        self._log_with_context(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message with consistent format"""
        self._log_with_context(logging.CRITICAL, message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with consistent format"""
        self._log_with_context(logging.DEBUG, message, **kwargs)
    
    def _log_with_context(self, level: int, message: str, **kwargs):
        """Log with context data"""
        
        # Create structured message
        if kwargs:
            # Ensure message is JSON-safe
            context_data = {
                "msg": message,
                "context": kwargs
            }
            structured_message = json.dumps(context_data, default=str, ensure_ascii=False)
        else:
            structured_message = json.dumps(message, ensure_ascii=False)
        
        self.logger.log(level, structured_message)
    
    def log_system_check(self, check_name: str, passed: bool, details: str = "", **kwargs):
        """Log system check result consistently"""
        self.info(f"System check: {check_name}",
                 check_name=check_name,
                 check_passed=passed,
                 check_details=details,
                 check_type="system_validation",
                 **kwargs)
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = "", **kwargs):
        """Log performance metric consistently"""
        self.info(f"Performance metric: {metric_name}",
                 metric_name=metric_name,
                 metric_value=value,
                 metric_unit=unit,
                 metric_type="performance",
                 **kwargs)
    
    def log_confidence_gate(self, gate_id: str, input_count: int, output_count: int, 
                           threshold: float, **kwargs):
        """Log confidence gate application consistently"""
        self.info(f"Confidence gate applied: {gate_id}",
                 gate_id=gate_id,
                 input_count=input_count,
                 output_count=output_count,
                 pass_rate=output_count/max(input_count,1),
                 threshold=threshold,
                 log_type="confidence_gate",
                 **kwargs)
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error with full context consistently"""
        self.error(f"Error occurred: {str(error)}",
                  error_type=type(error).__name__,
                  error_message=str(error),
                  context=context,
                  log_type="error_with_context")

def get_consolidated_logger(name: str) -> ConsolidatedLogger:
    """Get or create consolidated logger instance"""
    
    with ConsolidatedLogger._lock:
        if name not in ConsolidatedLogger._instances:
            ConsolidatedLogger._instances[name] = ConsolidatedLogger(name)
        return ConsolidatedLogger._instances[name]

# Compatibility aliases to replace old logging systems
def get_logger(name: str = "CryptoSmartTrader") -> ConsolidatedLogger:
    """Compatibility alias for get_consolidated_logger"""
    return get_consolidated_logger(name)

def get_structured_logger(name: str) -> ConsolidatedLogger:
    """Compatibility alias for get_consolidated_logger"""
    return get_consolidated_logger(name)

def get_unified_logger(name: str) -> ConsolidatedLogger:
    """Compatibility alias for get_consolidated_logger"""
    return get_consolidated_logger(name)

# Initialize root logging on import
_root_logger = get_consolidated_logger("CryptoSmartTrader")
_root_logger.info("Consolidated logging system initialized")