#!/usr/bin/env python3
"""
Structured Logger - Unified structured logging implementation
Consolidates logging approach to prevent double-JSON issues
"""

import logging
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Try to import the better structured formatter from config
try:
    from config.structured_logging import StructuredFormatter
    STRUCTURED_FORMATTER_AVAILABLE = True
except ImportError:
    STRUCTURED_FORMATTER_AVAILABLE = False

class StructuredLogger:
    """Unified structured logger - prevents double JSON encoding"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            
            if STRUCTURED_FORMATTER_AVAILABLE:
                # Use the better StructuredFormatter from config
                from config.structured_logging import StructuredFormatter
                formatter = StructuredFormatter()
                handler.setFormatter(formatter)
            else:
                # Fallback: plain formatter (no double JSON)
                formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
                handler.setFormatter(formatter)
            
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log info message - uses extra fields to prevent double JSON"""
        if STRUCTURED_FORMATTER_AVAILABLE and context:
            # Use extra fields for proper JSON structure
            self.logger.info(message, extra=context)
        elif context:
            # Build JSON manually only if no structured formatter
            payload = {"message": message}
            payload.update(context)
            self.logger.info(json.dumps(payload, ensure_ascii=False))
        else:
            self.logger.info(message)
    
    def error(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log error message - uses extra fields to prevent double JSON"""
        if STRUCTURED_FORMATTER_AVAILABLE and context:
            self.logger.error(message, extra=context)
        elif context:
            payload = {"message": message}
            payload.update(context)
            self.logger.error(json.dumps(payload, ensure_ascii=False))
        else:
            self.logger.error(message)
    
    def warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log warning message - uses extra fields to prevent double JSON"""
        if STRUCTURED_FORMATTER_AVAILABLE and context:
            self.logger.warning(message, extra=context)
        elif context:
            payload = {"message": message}
            payload.update(context)
            self.logger.warning(json.dumps(payload, ensure_ascii=False))
        else:
            self.logger.warning(message)

def get_structured_logger(name: str) -> StructuredLogger:
    """Get structured logger instance"""
    return StructuredLogger(name)