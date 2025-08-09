#!/usr/bin/env python3
"""
Structured Logger - Simple structured logging implementation
Fallback for when daily_logger is not available
"""

import logging
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional

class StructuredLogger:
    """Simple structured logger"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '{"timestamp": "%(asctime)s", "name": "%(name)s", "level": "%(levelname)s", "message": "%(message)s"}'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log info message"""
        log_data = {"message": message}
        if context:
            log_data.update(context)
        self.logger.info(json.dumps(log_data))
    
    def error(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log error message"""
        log_data = {"message": message}
        if context:
            log_data.update(context)
        self.logger.error(json.dumps(log_data))
    
    def warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        log_data = {"message": message}
        if context:
            log_data.update(context)
        self.logger.warning(json.dumps(log_data))

def get_structured_logger(name: str) -> StructuredLogger:
    """Get structured logger instance"""
    return StructuredLogger(name)