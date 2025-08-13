# utils/error_handler.py
import logging
import traceback
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type, Union
from contextlib import contextmanager
import threading
from collections import defaultdict, deque


logger = logging.getLogger(__name__)


class ErrorHandler:
    """Advanced error handling and recovery system"""

    def __init__(self, max_error_history: int = 1000):
        self.error_history = deque(maxlen=max_error_history)
        self.error_counts = defaultdict(int)
        self.recovery_strategies = {}
        self._lock = threading.Lock()

    def register_recovery_strategy(self, error_type: Type[Exception],
                                 strategy: Callable[[Exception], Any]):
        """Register a recovery strategy for specific error types"""
        self.recovery_strategies[error_type] = strategy
        logger.info(f"Registered recovery strategy for {error_type.__name__}")

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Handle an error with optional recovery

        Args:
            error: The exception that occurred
            context: Additional context information

        Returns:
            bool: True if error was handled/recovered, False otherwise
        """
        error_info = {
            'timestamp': time.time(),
            'type': type(error).__name__,
            'message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {}
        }

        with self._lock:
            self.error_history.append(error_info)
            self.error_counts[type(error).__name__] += 1

        # Log the error
        logger.error(f"Error handled: {type(error).__name__}: {str(error)}",
                    extra={
                        'error_type': type(error).__name__,
                        'error_context': context,
                        'traceback': error_info['traceback']
                    })

        # Try recovery strategies
        return self._attempt_recovery(error, context)

    def _attempt_recovery(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> bool:
        """Attempt to recover from the error using registered strategies"""
        error_type = type(error)

        # Check for exact type match first
        if error_type in self.recovery_strategies:
            try:
                self.recovery_strategies[error_type](error)
                logger.info(f"Successfully recovered from {error_type.__name__}")
                return True
            except Exception as recovery_error:
                logger.error(f"Recovery strategy failed: {recovery_error}")

        # Check for parent class matches
        for registered_type, strategy in self.recovery_strategies.items():
            if isinstance(error, registered_type):
                try:
                    strategy(error)
                    logger.info(f"Successfully recovered from {error_type.__name__} using {registered_type.__name__} strategy")
                    return True
                except Exception as recovery_error:
                    logger.error(f"Recovery strategy failed: {recovery_error}")

        return False

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and patterns"""
        with self._lock:
            total_errors = sum(self.error_counts.values())
            recent_errors = [e for e in self.error_history
                           if time.time() - e['timestamp'] < 3600]  # Last hour

            return {
                'total_errors': total_errors,
                'recent_errors_count': len(recent_errors),
                'error_types': dict(self.error_counts),
                'most_common_error': max(self.error_counts.items(),
                                       key=lambda x: x[1])[0] if self.error_counts else None,
                'error_rate_per_hour': len(recent_errors),
                'registered_recovery_strategies': list(self.recovery_strategies.keys())
            }

    def clear_error_history(self):
        """Clear error history and counts"""
        with self._lock:
            self.error_history.clear()
            self.error_counts.clear()
        logger.info("Error history cleared")


# Global error handler instance
error_handler = ErrorHandler()


def handle_errors(recovery_strategy: Optional[Callable[[Exception], Any]] = None,
                 log_errors: bool = True,
                 reraise: bool = False):
    """
    Decorator for automatic error handling

    Args:
        recovery_strategy: Optional recovery function
        log_errors: Whether to log errors
        reraise: Whether to reraise errors after handling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'function': func.__name__,
                    'args': str(args)[:200],  # Limit length
                    'kwargs': str(kwargs)[:200]
                }

                recovered = error_handler.handle_error(e, context)

                if recovery_strategy and not recovered:
                    try:
                        return recovery_strategy(e)
                    except Exception as recovery_error:
                        logger.error(f"Recovery function failed: {recovery_error}")

                if reraise:
                    raise

                return None

        return wrapper
    return decorator


@contextmanager
def error_context(context_name: str, **context_data):
    """Context manager for error handling with context"""
    try:
        yield
    except Exception as e:
        context = {'context_name': context_name, **context_data}
        error_handler.handle_error(e, context)
        raise


# Common recovery strategies
def api_timeout_recovery(error: Exception) -> None:
    """Recovery strategy for API timeout errors"""
    logger.info("Implementing API timeout recovery: reducing request frequency")
    time.sleep(5)  # Brief delay before retry


def memory_error_recovery(error: Exception) -> None:
    """Recovery strategy for memory errors"""
    import gc
    logger.info("Implementing memory error recovery: forcing garbage collection")
    gc.collect()


def connection_error_recovery(error: Exception) -> None:
    """Recovery strategy for connection errors"""
    logger.info("Implementing connection error recovery: waiting for reconnection")
    time.sleep(10)


# Register default recovery strategies
error_handler.register_recovery_strategy(TimeoutError, api_timeout_recovery)
error_handler.register_recovery_strategy(MemoryError, memory_error_recovery)
error_handler.register_recovery_strategy(ConnectionError, connection_error_recovery)
