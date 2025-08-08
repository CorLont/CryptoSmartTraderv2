#!/usr/bin/env python3
"""
Structured Logger - Enterprise JSON Logging
Advanced structured logging with daily rotation and metrics integration
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union
from pythonjsonlogger import jsonlogger
import threading

class StructuredJSONLogger:
    """Enterprise structured JSON logger with daily rotation"""
    
    def __init__(self, 
                 name: str = "CryptoSmartTrader",
                 log_level: str = "INFO",
                 daily_logs: bool = True):
        
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.daily_logs = daily_logs
        self._lock = threading.Lock()
        
        # Setup logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_console_handler()
        if daily_logs:
            self._setup_daily_file_handler()
    
    def _setup_console_handler(self):
        """Setup console handler with JSON formatting"""
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        
        # JSON formatter
        json_formatter = jsonlogger.JsonFormatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler.setFormatter(json_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_daily_file_handler(self):
        """Setup daily rotating file handler"""
        
        # Create daily log directory
        today_str = datetime.now().strftime("%Y%m%d")
        daily_log_dir = Path("logs/daily") / today_str
        daily_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file path
        log_file = daily_log_dir / f"{self.name.lower()}.jsonl"
        
        # File handler
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(self.log_level)
        
        # JSON formatter for file
        json_formatter = jsonlogger.JsonFormatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(funcName)s %(lineno)d %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(json_formatter)
        self.logger.addHandler(file_handler)
    
    def log(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log message with optional extra context"""
        
        with self._lock:
            log_method = getattr(self.logger, level.lower())
            
            if extra:
                # Add extra context to message
                log_method(message, extra=extra)
            else:
                log_method(message)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.log("info", message, kwargs if kwargs else None)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.log("warning", message, kwargs if kwargs else None)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.log("error", message, kwargs if kwargs else None)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.log("debug", message, kwargs if kwargs else None)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.log("critical", message, kwargs if kwargs else None)
    
    def structured_log(self, 
                      level: str,
                      event_type: str,
                      message: str,
                      context: Optional[Dict[str, Any]] = None,
                      metrics: Optional[Dict[str, Union[int, float]]] = None):
        """Log structured event with context and metrics"""
        
        log_data = {
            "event_type": event_type,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        if context:
            log_data["context"] = context
        
        if metrics:
            log_data["metrics"] = metrics
        
        self.log(level, json.dumps(log_data))

# Global logger instance
_structured_logger: Optional[StructuredJSONLogger] = None

def get_structured_logger(name: str = "CryptoSmartTrader") -> StructuredJSONLogger:
    """Get global structured logger instance"""
    global _structured_logger
    
    if _structured_logger is None:
        _structured_logger = StructuredJSONLogger(name)
    
    return _structured_logger