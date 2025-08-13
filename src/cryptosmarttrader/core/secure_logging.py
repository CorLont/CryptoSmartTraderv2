#!/usr/bin/env python3
"""
Secure Logging Manager
Prevents secrets leakage and implements correlation IDs
"""

import logging
import re
import uuid
import json
from typing import Any, Dict, Optional
from datetime import datetime

class SecureLogFilter(logging.Filter):
    """Filter to redact sensitive information from logs"""

    def __init__(self):
        super().__init__()
        # Patterns to redact
        self.secret_patterns = [
            re.compile(r'(api_key\s*[:=]\s*)([^\s]+)', re.IGNORECASE),
            re.compile(r'(secret\s*[:=]\s*)([^\s]+)', re.IGNORECASE),
            re.compile(r'(password\s*[:=]\s*)([^\s]+)', re.IGNORECASE),
            re.compile(r'(token\s*[:=]\s*)([^\s]+)', re.IGNORECASE),
            re.compile(r'Bearer\s+([A-Za-z0-9\-_=]+)', re.IGNORECASE)
        ]

    def filter(self, record):
        """Filter sensitive information from log record"""

        # Redact secrets from message
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            for pattern in self.secret_patterns:
                record.msg = pattern.sub(r'\1***REDACTED***', record.msg)

        # Redact from args
        if hasattr(record, 'args') and record.args:
            redacted_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    for pattern in self.secret_patterns:
                        arg = pattern.sub(r'\1***REDACTED***', arg)
                redacted_args.append(arg)
            record.args = tuple(redacted_args)

        return True

class CorrelatedLogger:
    """Logger with automatic correlation ID tracking"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.correlation_id = None

        # Add secure filter
        secure_filter = SecureLogFilter()
        self.logger.addFilter(secure_filter)

    def set_correlation_id(self, correlation_id: Optional[str] = None):
        """Set correlation ID for request tracking"""
        self.correlation_id = correlation_id or str(uuid.uuid4())[:8]

    def _add_correlation(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """Add correlation ID to log extra"""
        if extra is None:
            extra = {}

        if self.correlation_id:
            extra['correlation_id'] = self.correlation_id

        extra['timestamp'] = datetime.utcnow().isoformat()
        return extra

    def info(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        """Log info with correlation ID"""
        self.logger.info(msg, extra=self._add_correlation(extra))

    def warning(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning with correlation ID"""
        self.logger.warning(msg, extra=self._add_correlation(extra))

    def error(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        """Log error with correlation ID"""
        self.logger.error(msg, extra=self._add_correlation(extra))

    def debug(self, msg: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug with correlation ID"""
        self.logger.debug(msg, extra=self._add_correlation(extra))

def get_secure_logger(name: str) -> CorrelatedLogger:
    """Get secure logger instance"""
    return CorrelatedLogger(name)
