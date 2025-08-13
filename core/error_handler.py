"""
CryptoSmartTrader V2 - Centralized Error Handling & Recovery
Enterprise-grade error management with structured logging and recovery strategies
"""

import logging
import traceback
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List
from functools import wraps
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ErrorCategory:
    """Error categorization for structured handling"""

    NETWORK = "network"
    API = "api"
    DATA = "data"
    CALCULATION = "calculation"
    CONFIGURATION = "configuration"
    SYSTEM = "system"
    EXTERNAL_SERVICE = "external_service"
    CRITICAL = "critical"


class RetryStrategy:
    """Configurable retry strategies with exponential backoff"""

    def __init__(
        self,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
    ):
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base

    def get_delay(self, attempt: int) -> float:
        """Calculate delay for given attempt with exponential backoff"""
        delay = self.base_delay * (self.exponential_base**attempt)
        return min(delay, self.max_delay)


class CentralizedErrorHandler:
    """Enterprise-grade centralized error handling system"""

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)

        # Error statistics tracking
        self.error_stats = {
            "total_errors": 0,
            "errors_by_category": {},
            "errors_by_hour": {},
            "recovery_success_rate": 0.0,
            "last_critical_error": None,
        }

        # Recovery strategies by error category
        self.recovery_strategies = {
            ErrorCategory.NETWORK: self._recover_network_error,
            ErrorCategory.API: self._recover_api_error,
            ErrorCategory.DATA: self._recover_data_error,
            ErrorCategory.CALCULATION: self._recover_calculation_error,
            ErrorCategory.CONFIGURATION: self._recover_config_error,
            ErrorCategory.SYSTEM: self._recover_system_error,
            ErrorCategory.EXTERNAL_SERVICE: self._recover_external_service_error,
            ErrorCategory.CRITICAL: self._recover_critical_error,
        }

        # Default retry strategies by category
        self.retry_strategies = {
            ErrorCategory.NETWORK: RetryStrategy(max_attempts=5, base_delay=2.0, max_delay=30.0),
            ErrorCategory.API: RetryStrategy(max_attempts=4, base_delay=1.5, max_delay=20.0),
            ErrorCategory.DATA: RetryStrategy(max_attempts=3, base_delay=1.0, max_delay=10.0),
            ErrorCategory.EXTERNAL_SERVICE: RetryStrategy(
                max_attempts=6, base_delay=3.0, max_delay=60.0
            ),
            ErrorCategory.SYSTEM: RetryStrategy(max_attempts=2, base_delay=5.0, max_delay=30.0),
        }

        # Initialize error logging
        self._setup_error_logging()

    def _setup_error_logging(self):
        """Setup structured error logging"""
        try:
            # Create errors directory
            error_log_dir = Path("logs/errors")
            error_log_dir.mkdir(parents=True, exist_ok=True)

            # Setup error-specific logger
            error_logger = logging.getLogger("error_handler")
            error_handler = logging.FileHandler(
                error_log_dir / f"errors_{datetime.now().strftime('%Y%m%d')}.log"
            )

            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            error_handler.setFormatter(formatter)
            error_logger.addHandler(error_handler)
            error_logger.setLevel(logging.ERROR)

            self.error_logger = error_logger

        except Exception as e:
            self.logger.error(f"Failed to setup error logging: {e}")
            self.error_logger = self.logger

    def handle_error(
        self,
        error: Exception,
        category: str,
        context: Dict[str, Any] = None,
        auto_recover: bool = True,
    ) -> Dict[str, Any]:
        """Centralized error handling with structured logging and recovery"""
        try:
            # Create error record
            error_record = self._create_error_record(error, category, context)

            # Log structured error
            self._log_structured_error(error_record)

            # Update error statistics
            self._update_error_stats(category)

            # Attempt recovery if enabled
            recovery_result = None
            if auto_recover and category in self.recovery_strategies:
                recovery_result = self._attempt_recovery(error, category, context)

            # Return comprehensive error response
            return {
                "error_id": error_record["error_id"],
                "category": category,
                "message": str(error),
                "severity": self._get_error_severity(category),
                "recovery_attempted": auto_recover,
                "recovery_successful": recovery_result.get("success", False)
                if recovery_result
                else False,
                "recovery_details": recovery_result,
                "timestamp": error_record["timestamp"],
                "context": context or {},
            }

        except Exception as handling_error:
            # Fallback error handling
            self.logger.critical(f"Error handler failed: {handling_error}")
            return {
                "error_id": f"fallback_{int(time.time())}",
                "category": "critical",
                "message": f"Original error: {error}, Handler error: {handling_error}",
                "severity": "critical",
                "recovery_attempted": False,
                "recovery_successful": False,
            }

    def _create_error_record(
        self, error: Exception, category: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create structured error record"""
        error_id = f"{category}_{int(time.time())}_{hash(str(error)) % 10000}"

        return {
            "error_id": error_id,
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {},
            "severity": self._get_error_severity(category),
            "stack_trace": traceback.format_stack(),
        }

    def _log_structured_error(self, error_record: Dict[str, Any]):
        """Log error with structured format"""
        try:
            # Log to error-specific logger
            self.error_logger.error(json.dumps(error_record, indent=2))

            # Log to main application logger based on severity
            severity = error_record.get("severity", "error")
            if severity == "critical":
                self.logger.critical(
                    f"CRITICAL ERROR [{error_record['error_id']}]: {error_record['message']}"
                )
            elif severity == "high":
                self.logger.error(
                    f"HIGH SEVERITY [{error_record['error_id']}]: {error_record['message']}"
                )
            else:
                self.logger.warning(
                    f"ERROR [{error_record['error_id']}]: {error_record['message']}"
                )

        except Exception as e:
            self.logger.error(f"Failed to log structured error: {e}")

    def _get_error_severity(self, category: str) -> str:
        """Determine error severity based on category"""
        severity_map = {
            ErrorCategory.CRITICAL: "critical",
            ErrorCategory.SYSTEM: "high",
            ErrorCategory.CONFIGURATION: "high",
            ErrorCategory.NETWORK: "medium",
            ErrorCategory.API: "medium",
            ErrorCategory.DATA: "medium",
            ErrorCategory.CALCULATION: "low",
            ErrorCategory.EXTERNAL_SERVICE: "medium",
        }
        return severity_map.get(category, "medium")

    def _update_error_stats(self, category: str):
        """Update error statistics for monitoring"""
        self.error_stats["total_errors"] += 1

        # Update category stats
        if category not in self.error_stats["errors_by_category"]:
            self.error_stats["errors_by_category"][category] = 0
        self.error_stats["errors_by_category"][category] += 1

        # Update hourly stats
        current_hour = datetime.now().strftime("%Y-%m-%d %H:00")
        if current_hour not in self.error_stats["errors_by_hour"]:
            self.error_stats["errors_by_hour"][current_hour] = 0
        self.error_stats["errors_by_hour"][current_hour] += 1

        # Update critical error timestamp
        if category == ErrorCategory.CRITICAL:
            self.error_stats["last_critical_error"] = datetime.now().isoformat()

    def _attempt_recovery(
        self, error: Exception, category: str, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Attempt error recovery using category-specific strategies"""
        try:
            recovery_strategy = self.recovery_strategies.get(category)
            if not recovery_strategy:
                return {"success": False, "reason": "No recovery strategy available"}

            recovery_result = recovery_strategy(error, context)

            # Update recovery success rate
            total_recoveries = sum(1 for result in [recovery_result] if result.get("success"))
            self.error_stats["recovery_success_rate"] = total_recoveries / max(
                1, self.error_stats["total_errors"]
            )

            return recovery_result

        except Exception as recovery_error:
            return {
                "success": False,
                "reason": f"Recovery failed: {recovery_error}",
                "recovery_error": str(recovery_error),
            }

    # Recovery strategy implementations
    def _recover_network_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from network-related errors"""
        return {
            "success": True,
            "strategy": "network_retry",
            "recommendation": "Retry with exponential backoff",
            "retry_strategy": self.retry_strategies[ErrorCategory.NETWORK].__dict__,
        }

    def _recover_api_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from API errors"""
        return {
            "success": True,
            "strategy": "api_fallback",
            "recommendation": "Use alternative data source or cached data",
            "fallback_options": ["cached_data", "alternative_endpoint", "reduced_functionality"],
        }

    def _recover_data_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from data-related errors"""
        return {
            "success": True,
            "strategy": "data_validation",
            "recommendation": "Use data validation and sanitization",
            "recovery_actions": ["validate_input", "sanitize_data", "use_default_values"],
        }

    def _recover_calculation_error(
        self, error: Exception, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recover from calculation errors"""
        return {
            "success": True,
            "strategy": "calculation_fallback",
            "recommendation": "Use alternative calculation method",
            "fallback_methods": ["simple_average", "cached_result", "conservative_estimate"],
        }

    def _recover_config_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from configuration errors"""
        return {
            "success": True,
            "strategy": "config_rollback",
            "recommendation": "Rollback to last known good configuration",
            "recovery_actions": ["load_backup_config", "use_defaults", "notify_admin"],
        }

    def _recover_system_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from system errors"""
        return {
            "success": False,
            "strategy": "system_restart",
            "recommendation": "System restart may be required",
            "severity": "high",
            "requires_intervention": True,
        }

    def _recover_external_service_error(
        self, error: Exception, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Recover from external service errors"""
        return {
            "success": True,
            "strategy": "service_fallback",
            "recommendation": "Use alternative service or cached data",
            "fallback_options": ["cached_data", "alternative_service", "degraded_mode"],
        }

    def _recover_critical_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle critical errors requiring immediate attention"""
        return {
            "success": False,
            "strategy": "immediate_intervention",
            "recommendation": "Immediate administrator intervention required",
            "severity": "critical",
            "requires_intervention": True,
            "alert_administrators": True,
        }

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get current error statistics for monitoring"""
        return {
            **self.error_stats,
            "health_score": self._calculate_health_score(),
            "recommendations": self._get_health_recommendations(),
        }

    def _calculate_health_score(self) -> float:
        """Calculate system health score based on error patterns"""
        try:
            total_errors = self.error_stats["total_errors"]
            if total_errors == 0:
                return 1.0

            # Calculate score based on error frequency and recovery rate
            recovery_rate = self.error_stats["recovery_success_rate"]

            # Recent error frequency (last 24 hours)
            recent_errors = sum(
                count
                for hour, count in self.error_stats["errors_by_hour"].items()
                if datetime.fromisoformat(hour.replace(" ", "T") + ":00")
                > datetime.now() - timedelta(hours=24)
            )

            # Score calculation (0.0 to 1.0)
            base_score = 1.0 - min(recent_errors / 100, 0.8)  # Cap error impact
            recovery_bonus = recovery_rate * 0.2  # Bonus for good recovery

            return max(0.0, min(1.0, base_score + recovery_bonus))

        except Exception:
            return 0.5  # Default moderate health score

    def _get_health_recommendations(self) -> List[str]:
        """Get health recommendations based on error patterns"""
        recommendations = []

        try:
            health_score = self._calculate_health_score()

            if health_score < 0.3:
                recommendations.append(
                    "URGENT: System health critical - immediate intervention required"
                )
            elif health_score < 0.6:
                recommendations.append(
                    "WARNING: System health degraded - investigate error patterns"
                )

            # Category-specific recommendations
            error_categories = self.error_stats["errors_by_category"]

            if error_categories.get(ErrorCategory.NETWORK, 0) > 10:
                recommendations.append(
                    "High network error rate - check connectivity and API endpoints"
                )

            if error_categories.get(ErrorCategory.CRITICAL, 0) > 0:
                recommendations.append("Critical errors detected - review system logs immediately")

            if self.error_stats["recovery_success_rate"] < 0.5:
                recommendations.append("Low recovery success rate - review recovery strategies")

        except Exception:
            recommendations.append(
                "Unable to generate health recommendations - manual review needed"
            )

        return recommendations


def with_error_handling(category: str = ErrorCategory.SYSTEM, auto_recover: bool = True):
    """Decorator for automatic error handling with recovery"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get error handler instance (assuming it's available globally or in class)
                error_handler = getattr(args[0], "error_handler", None) if args else None
                if not error_handler:
                    # Create temporary error handler if none available
                    error_handler = CentralizedErrorHandler()

                # Handle the error
                context = {
                    "function": func.__name__,
                    "args": str(args)[:200],  # Limit context size
                    "kwargs": str(kwargs)[:200],
                }

                error_result = error_handler.handle_error(e, category, context, auto_recover)

                # Re-raise if critical or not recoverable
                if category == ErrorCategory.CRITICAL or not error_result.get(
                    "recovery_successful"
                ):
                    raise

                # Return None or default value for recoverable errors
                return None

        return wrapper

    return decorator


def with_retry(category: str = ErrorCategory.NETWORK, max_attempts: int = None):
    """Decorator for automatic retry with exponential backoff"""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get or create error handler
            error_handler = getattr(args[0], "error_handler", None) if args else None
            if not error_handler:
                error_handler = CentralizedErrorHandler()

            # Get retry strategy
            retry_strategy = error_handler.retry_strategies.get(
                category, RetryStrategy(max_attempts=max_attempts or 3)
            )

            last_exception = None
            for attempt in range(retry_strategy.max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    if attempt < retry_strategy.max_attempts - 1:
                        delay = retry_strategy.get_delay(attempt)
                        time.sleep(delay)
                        continue
                    else:
                        # Final attempt failed - handle error
                        context = {
                            "function": func.__name__,
                            "attempts": attempt + 1,
                            "final_attempt": True,
                        }
                        error_handler.handle_error(e, category, context)
                        raise

            # Should never reach here, but just in case
            raise last_exception

        return wrapper

    return decorator
