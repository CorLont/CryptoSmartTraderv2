#!/usr/bin/env python3
"""
Unified Error Handler Infrastructure
Centrale, consistente error handling voor alle scrapers, LLM-agents, en external API calls
"""

import asyncio
import logging
import traceback
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union
import json
from datetime import datetime
from functools import wraps

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories voor classification"""
    RATE_LIMIT = "rate_limit"
    AUTHENTICATION = "authentication"
    NETWORK = "network"
    TIMEOUT = "timeout"
    PERMISSION = "permission"
    DATA_VALIDATION = "data_validation"
    SERVICE_UNAVAILABLE = "service_unavailable"
    QUOTA_EXCEEDED = "quota_exceeded"
    UNKNOWN = "unknown"


@dataclass
class ErrorPattern:
    """Pattern definition voor error recognition"""
    exception_types: List[Type[Exception]] = field(default_factory=list)
    message_patterns: List[str] = field(default_factory=list)
    status_codes: List[int] = field(default_factory=list)
    category: ErrorCategory = ErrorCategory.UNKNOWN
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    retry_recommended: bool = True
    backoff_multiplier: float = 1.0
    custom_handler: Optional[Callable] = None


@dataclass
class ErrorEvent:
    """Error event voor logging en analysis"""
    timestamp: datetime
    service_type: str
    endpoint: Optional[str]
    category: ErrorCategory
    severity: ErrorSeverity
    exception_type: str
    message: str
    stack_trace: str
    retry_attempt: int
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary voor logging"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "service_type": self.service_type,
            "endpoint": self.endpoint,
            "category": self.category.value,
            "severity": self.severity.value,
            "exception_type": self.exception_type,
            "message": self.message,
            "stack_trace": self.stack_trace,
            "retry_attempt": self.retry_attempt,
            "context": self.context
        }


class UnifiedErrorHandler:
    """
    Unified error handling system voor ALL external service calls
    KRITIEK: Consistent error handling across all scrapers and agents
    """
    
    _instance: Optional['UnifiedErrorHandler'] = None
    
    def __new__(cls) -> 'UnifiedErrorHandler':
        """Singleton pattern enforcement"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize error handler"""
        if getattr(self, '_initialized', False):
            return
            
        self._error_patterns: Dict[str, List[ErrorPattern]] = {}
        self._error_history: List[ErrorEvent] = []
        self._error_counts: Dict[str, Dict[ErrorCategory, int]] = {}
        
        # Initialize standard error patterns
        self._initialize_standard_patterns()
        
        self._initialized = True
        logger.info("🔧 UnifiedErrorHandler initialized with standard patterns")
    
    def _initialize_standard_patterns(self) -> None:
        """Initialize standard error patterns voor common services"""
        
        # OpenAI/LLM API patterns
        self._register_error_patterns("llm_api", [
            ErrorPattern(
                exception_types=[Exception],
                message_patterns=["rate limit", "Rate limit", "429"],
                status_codes=[429],
                category=ErrorCategory.RATE_LIMIT,
                severity=ErrorSeverity.MEDIUM,
                retry_recommended=True,
                backoff_multiplier=2.0
            ),
            ErrorPattern(
                exception_types=[Exception],
                message_patterns=["quota", "usage limit", "billing"],
                category=ErrorCategory.QUOTA_EXCEEDED,
                severity=ErrorSeverity.HIGH,
                retry_recommended=False
            ),
            ErrorPattern(
                exception_types=[Exception],
                message_patterns=["authentication", "unauthorized", "invalid api key"],
                status_codes=[401, 403],
                category=ErrorCategory.AUTHENTICATION,
                severity=ErrorSeverity.CRITICAL,
                retry_recommended=False
            ),
            ErrorPattern(
                exception_types=[Exception],
                message_patterns=["timeout", "timed out"],
                category=ErrorCategory.TIMEOUT,
                severity=ErrorSeverity.MEDIUM,
                retry_recommended=True,
                backoff_multiplier=1.5
            )
        ])
        
        # Social Media API patterns
        self._register_error_patterns("social_media", [
            ErrorPattern(
                exception_types=[Exception],
                message_patterns=["rate limit", "too many requests"],
                status_codes=[429],
                category=ErrorCategory.RATE_LIMIT,
                severity=ErrorSeverity.HIGH,
                retry_recommended=True,
                backoff_multiplier=3.0
            ),
            ErrorPattern(
                exception_types=[Exception],
                message_patterns=["forbidden", "access denied", "suspended"],
                status_codes=[403],
                category=ErrorCategory.PERMISSION,
                severity=ErrorSeverity.CRITICAL,
                retry_recommended=False
            ),
            ErrorPattern(
                exception_types=[Exception],
                message_patterns=["not found", "404"],
                status_codes=[404],
                category=ErrorCategory.DATA_VALIDATION,
                severity=ErrorSeverity.LOW,
                retry_recommended=False
            )
        ])
        
        # Exchange API patterns
        self._register_error_patterns("exchange_api", [
            ErrorPattern(
                exception_types=[Exception],
                message_patterns=["rate limit", "request weight"],
                category=ErrorCategory.RATE_LIMIT,
                severity=ErrorSeverity.MEDIUM,
                retry_recommended=True,
                backoff_multiplier=1.5
            ),
            ErrorPattern(
                exception_types=[Exception],
                message_patterns=["maintenance", "system maintenance"],
                category=ErrorCategory.SERVICE_UNAVAILABLE,
                severity=ErrorSeverity.HIGH,
                retry_recommended=True,
                backoff_multiplier=5.0
            )
        ])
        
        # Web Scraper patterns
        self._register_error_patterns("web_scraper", [
            ErrorPattern(
                exception_types=[Exception],
                message_patterns=["blocked", "captcha", "bot detection"],
                category=ErrorCategory.PERMISSION,
                severity=ErrorSeverity.CRITICAL,
                retry_recommended=False
            ),
            ErrorPattern(
                exception_types=[Exception],
                message_patterns=["connection", "network", "dns"],
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.MEDIUM,
                retry_recommended=True,
                backoff_multiplier=2.0
            )
        ])
    
    def _register_error_patterns(self, service_type: str, patterns: List[ErrorPattern]) -> None:
        """Register error patterns voor service type"""
        if service_type not in self._error_patterns:
            self._error_patterns[service_type] = []
            self._error_counts[service_type] = {category: 0 for category in ErrorCategory}
        
        self._error_patterns[service_type].extend(patterns)
    
    def register_custom_patterns(self, service_type: str, patterns: List[ErrorPattern]) -> None:
        """Register custom error patterns"""
        logger.info(f"🔧 Registering custom error patterns voor {service_type}")
        self._register_error_patterns(service_type, patterns)
    
    def _classify_error(
        self,
        service_type: str,
        exception: Exception,
        status_code: Optional[int] = None
    ) -> ErrorPattern:
        """Classify error based on patterns"""
        patterns = self._error_patterns.get(service_type, [])
        
        for pattern in patterns:
            # Check exception type
            if pattern.exception_types and not any(
                isinstance(exception, exc_type) for exc_type in pattern.exception_types
            ):
                continue
            
            # Check status code
            if status_code and pattern.status_codes and status_code not in pattern.status_codes:
                continue
            
            # Check message patterns
            error_message = str(exception).lower()
            if pattern.message_patterns:
                if not any(
                    pattern_text.lower() in error_message 
                    for pattern_text in pattern.message_patterns
                ):
                    continue
            
            return pattern
        
        # Default pattern voor unrecognized errors
        return ErrorPattern(
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            retry_recommended=True
        )
    
    def handle_error(
        self,
        service_type: str,
        exception: Exception,
        endpoint: Optional[str] = None,
        retry_attempt: int = 0,
        context: Optional[Dict[str, Any]] = None,
        status_code: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Handle error with classification and logging
        
        Returns:
            Dict with error handling recommendations
        """
        # Classify error
        pattern = self._classify_error(service_type, exception, status_code)
        
        # Create error event
        error_event = ErrorEvent(
            timestamp=datetime.utcnow(),
            service_type=service_type,
            endpoint=endpoint,
            category=pattern.category,
            severity=pattern.severity,
            exception_type=type(exception).__name__,
            message=str(exception),
            stack_trace=traceback.format_exc(),
            retry_attempt=retry_attempt,
            context=context or {}
        )
        
        # Log error event
        self._log_error_event(error_event)
        
        # Update error counts
        self._error_counts[service_type][pattern.category] += 1
        
        # Store in history (keep last 1000 events)
        self._error_history.append(error_event)
        if len(self._error_history) > 1000:
            self._error_history.pop(0)
        
        # Execute custom handler if defined
        if pattern.custom_handler:
            try:
                pattern.custom_handler(error_event)
            except Exception as handler_error:
                logger.error(f"🔧 Custom error handler failed: {handler_error}")
        
        # Return handling recommendations
        return {
            "category": pattern.category,
            "severity": pattern.severity,
            "retry_recommended": pattern.retry_recommended,
            "backoff_multiplier": pattern.backoff_multiplier,
            "should_alert": pattern.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL],
            "should_circuit_break": (
                pattern.category in [ErrorCategory.AUTHENTICATION, ErrorCategory.PERMISSION] or
                pattern.severity == ErrorSeverity.CRITICAL
            )
        }
    
    def _log_error_event(self, event: ErrorEvent) -> None:
        """Log error event with appropriate level"""
        log_data = event.to_dict()
        
        if event.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"🔧 CRITICAL ERROR: {event.service_type} - {event.message}", extra=log_data)
        elif event.severity == ErrorSeverity.HIGH:
            logger.error(f"🔧 HIGH SEVERITY: {event.service_type} - {event.message}", extra=log_data)
        elif event.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"🔧 MEDIUM SEVERITY: {event.service_type} - {event.message}", extra=log_data)
        else:
            logger.info(f"🔧 LOW SEVERITY: {event.service_type} - {event.message}", extra=log_data)
    
    def get_error_statistics(self, service_type: Optional[str] = None) -> Dict[str, Any]:
        """Get error statistics"""
        if service_type:
            if service_type not in self._error_counts:
                return {"error": f"Service type {service_type} not found"}
            
            return {
                "service_type": service_type,
                "error_counts": {cat.value: count for cat, count in self._error_counts[service_type].items()},
                "total_errors": sum(self._error_counts[service_type].values())
            }
        
        # All services
        total_stats = {}
        for svc_type, counts in self._error_counts.items():
            total_stats[svc_type] = {
                "error_counts": {cat.value: count for cat, count in counts.items()},
                "total_errors": sum(counts.values())
            }
        
        return total_stats
    
    def get_recent_errors(self, limit: int = 50, service_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get recent error events"""
        filtered_events = self._error_history
        
        if service_type:
            filtered_events = [e for e in filtered_events if e.service_type == service_type]
        
        # Sort by timestamp descending and limit
        sorted_events = sorted(filtered_events, key=lambda x: x.timestamp, reverse=True)
        return [event.to_dict() for event in sorted_events[:limit]]
    
    def clear_error_history(self, service_type: Optional[str] = None) -> None:
        """Clear error history"""
        if service_type:
            self._error_history = [e for e in self._error_history if e.service_type != service_type]
            if service_type in self._error_counts:
                self._error_counts[service_type] = {category: 0 for category in ErrorCategory}
        else:
            self._error_history.clear()
            for svc_type in self._error_counts:
                self._error_counts[svc_type] = {category: 0 for category in ErrorCategory}
        
        logger.info(f"🔧 Error history cleared voor {service_type or 'all services'}")


# Global singleton instance
error_handler = UnifiedErrorHandler()


def unified_error_handling(service_type: str, endpoint: Optional[str] = None):
    """
    Decorator voor unified error handling
    
    Usage:
        @unified_error_handling("llm_api")
        async def call_openai():
            return openai.chat.completions.create(...)
        
        @unified_error_handling("social_media", "reddit_api")
        def get_reddit_posts():
            return reddit.subreddit("cryptocurrency").hot()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_info = error_handler.handle_error(
                    service_type=service_type,
                    exception=e,
                    endpoint=endpoint,
                    context={"function": func.__name__, "args_count": len(args), "kwargs_keys": list(kwargs.keys())}
                )
                
                # Re-raise with additional context
                raise type(e)(f"{str(e)} [Error Category: {error_info['category'].value}]") from e
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_info = error_handler.handle_error(
                    service_type=service_type,
                    exception=e,
                    endpoint=endpoint,
                    context={"function": func.__name__, "args_count": len(args), "kwargs_keys": list(kwargs.keys())}
                )
                
                # Re-raise with additional context
                raise type(e)(f"{str(e)} [Error Category: {error_info['category'].value}]") from e
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Export public interface
__all__ = [
    'UnifiedErrorHandler',
    'ErrorSeverity',
    'ErrorCategory',
    'ErrorPattern',
    'ErrorEvent',
    'unified_error_handling',
    'error_handler'
]