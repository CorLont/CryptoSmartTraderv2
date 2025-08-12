"""
Circuit Breaker System

Advanced circuit breaker implementation for protecting against
cascading failures and system overload scenarios.
"""

import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import deque
import statistics

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Circuit is open, requests are blocked
    HALF_OPEN = "half_open"  # Testing if system has recovered


class BreakReason(Enum):
    """Reasons for circuit breaking"""
    HIGH_ERROR_RATE = "high_error_rate"
    HIGH_LATENCY = "high_latency"
    TIMEOUT_THRESHOLD = "timeout_threshold"
    DEPENDENCY_FAILURE = "dependency_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    MANUAL_BREAK = "manual_break"


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    name: str
    
    # Failure thresholds
    failure_threshold: int = 5          # Number of failures to open circuit
    timeout_threshold_ms: int = 5000    # Timeout threshold in milliseconds
    error_rate_threshold: float = 0.5   # Error rate threshold (0.0-1.0)
    
    # Time windows
    rolling_window_seconds: int = 60    # Rolling window for metrics
    recovery_timeout_seconds: int = 30  # Time to wait before trying half-open
    half_open_timeout_seconds: int = 10 # Timeout for half-open state
    
    # Half-open behavior
    half_open_max_calls: int = 3        # Max calls to allow in half-open state
    half_open_success_threshold: int = 2 # Successful calls needed to close circuit


@dataclass
class CallResult:
    """Result of a protected call"""
    success: bool
    duration_ms: float
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class CircuitBreakerSystem:
    """
    Enterprise circuit breaker system
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        
        # Metrics tracking
        self.call_history: deque = deque(maxlen=1000)  # Keep last 1000 calls
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.last_state_change = datetime.now()
        
        # Half-open state tracking
        self.half_open_calls = 0
        self.half_open_successes = 0
        
        # Callbacks
        self.state_change_callbacks: List[Callable] = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Monitoring
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_circuit, daemon=True)
        self._monitor_thread.start()
    
    def call(self, protected_function: Callable, *args, **kwargs) -> Any:
        """Execute a function with circuit breaker protection"""
        
        with self._lock:
            # Check if call is allowed
            if not self._is_call_allowed():
                raise CircuitBreakerOpenException(
                    f"Circuit breaker '{self.config.name}' is OPEN"
                )
            
            start_time = time.time()
            
            try:
                # Execute the protected function
                result = protected_function(*args, **kwargs)
                
                # Record successful call
                duration_ms = (time.time() - start_time) * 1000
                self._record_call(CallResult(
                    success=True,
                    duration_ms=duration_ms
                ))
                
                return result
                
            except Exception as e:
                # Record failed call
                duration_ms = (time.time() - start_time) * 1000
                self._record_call(CallResult(
                    success=False,
                    duration_ms=duration_ms,
                    error=str(e)
                ))
                
                # Re-raise the original exception
                raise
    
    def _is_call_allowed(self) -> bool:
        """Check if call is allowed based on circuit state"""
        
        if self.state == CircuitState.CLOSED:
            return True
        
        elif self.state == CircuitState.OPEN:
            # Check if recovery timeout has passed
            if self.last_state_change + timedelta(seconds=self.config.recovery_timeout_seconds) <= datetime.now():
                self._transition_to_half_open()
                return True
            return False
        
        elif self.state == CircuitState.HALF_OPEN:
            # Allow limited calls in half-open state
            if self.half_open_calls < self.config.half_open_max_calls:
                return True
            return False
        
        return False
    
    def _record_call(self, result: CallResult):
        """Record call result and update circuit state"""
        
        # Add to call history
        self.call_history.append(result)
        
        # Update metrics based on current state
        if self.state == CircuitState.CLOSED:
            self._handle_closed_state_call(result)
        elif self.state == CircuitState.HALF_OPEN:
            self._handle_half_open_state_call(result)
        
        # Clean old metrics
        self._cleanup_old_metrics()
    
    def _handle_closed_state_call(self, result: CallResult):
        """Handle call result in closed state"""
        
        if not result.success:
            self.failure_count += 1
            self.last_failure_time = result.timestamp
            
            # Check if we should open the circuit
            if self._should_open_circuit():
                self._transition_to_open(BreakReason.HIGH_ERROR_RATE)
        
        elif result.duration_ms > self.config.timeout_threshold_ms:
            # High latency can also trigger circuit break
            logger.warning(f"High latency detected: {result.duration_ms}ms")
            if self._should_open_circuit_due_to_latency():
                self._transition_to_open(BreakReason.HIGH_LATENCY)
        
        else:
            # Successful call - reset failure count
            if self.failure_count > 0:
                self.failure_count = max(0, self.failure_count - 1)
    
    def _handle_half_open_state_call(self, result: CallResult):
        """Handle call result in half-open state"""
        
        self.half_open_calls += 1
        
        if result.success:
            self.half_open_successes += 1
            
            # Check if we can close the circuit
            if self.half_open_successes >= self.config.half_open_success_threshold:
                self._transition_to_closed()
        
        else:
            # Failure in half-open state - go back to open
            self._transition_to_open(BreakReason.HIGH_ERROR_RATE)
    
    def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened"""
        
        # Check failure count threshold
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Check error rate over rolling window
        recent_calls = self._get_recent_calls()
        if len(recent_calls) >= 5:  # Need minimum calls for meaningful rate
            error_rate = sum(1 for call in recent_calls if not call.success) / len(recent_calls)
            if error_rate >= self.config.error_rate_threshold:
                return True
        
        return False
    
    def _should_open_circuit_due_to_latency(self) -> bool:
        """Check if circuit should be opened due to high latency"""
        
        recent_calls = self._get_recent_calls()
        if len(recent_calls) >= 5:
            # Calculate average latency
            latencies = [call.duration_ms for call in recent_calls]
            avg_latency = statistics.mean(latencies)
            
            if avg_latency > self.config.timeout_threshold_ms:
                return True
        
        return False
    
    def _get_recent_calls(self) -> List[CallResult]:
        """Get calls within the rolling window"""
        
        cutoff_time = datetime.now() - timedelta(seconds=self.config.rolling_window_seconds)
        return [call for call in self.call_history if call.timestamp >= cutoff_time]
    
    def _transition_to_open(self, reason: BreakReason):
        """Transition circuit to open state"""
        
        old_state = self.state
        self.state = CircuitState.OPEN
        self.last_state_change = datetime.now()
        
        logger.warning(f"Circuit breaker '{self.config.name}' opened due to {reason.value}")
        
        # Reset half-open counters
        self.half_open_calls = 0
        self.half_open_successes = 0
        
        self._notify_state_change(old_state, self.state, reason)
    
    def _transition_to_half_open(self):
        """Transition circuit to half-open state"""
        
        old_state = self.state
        self.state = CircuitState.HALF_OPEN
        self.last_state_change = datetime.now()
        
        # Reset half-open counters
        self.half_open_calls = 0
        self.half_open_successes = 0
        
        logger.info(f"Circuit breaker '{self.config.name}' entering half-open state")
        
        self._notify_state_change(old_state, self.state, None)
    
    def _transition_to_closed(self):
        """Transition circuit to closed state"""
        
        old_state = self.state
        self.state = CircuitState.CLOSED
        self.last_state_change = datetime.now()
        
        # Reset counters
        self.failure_count = 0
        self.half_open_calls = 0
        self.half_open_successes = 0
        
        logger.info(f"Circuit breaker '{self.config.name}' closed - normal operation resumed")
        
        self._notify_state_change(old_state, self.state, None)
    
    def _notify_state_change(self, old_state: CircuitState, new_state: CircuitState, reason: Optional[BreakReason]):
        """Notify callbacks of state change"""
        
        for callback in self.state_change_callbacks:
            try:
                callback(self.config.name, old_state, new_state, reason)
            except Exception as e:
                logger.error(f"Circuit breaker callback failed: {e}")
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics outside rolling window"""
        
        cutoff_time = datetime.now() - timedelta(seconds=self.config.rolling_window_seconds * 2)
        
        # Remove old call history
        while self.call_history and self.call_history[0].timestamp < cutoff_time:
            self.call_history.popleft()
    
    def _monitor_circuit(self):
        """Background monitoring thread"""
        
        while self._monitoring_active:
            try:
                with self._lock:
                    # Check for timeout in half-open state
                    if (self.state == CircuitState.HALF_OPEN and 
                        self.last_state_change + timedelta(seconds=self.config.half_open_timeout_seconds) <= datetime.now()):
                        
                        # Half-open timeout - go back to open
                        self._transition_to_open(BreakReason.TIMEOUT_THRESHOLD)
                
                # Sleep for monitoring interval
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Circuit breaker monitoring error: {e}")
                time.sleep(10)  # Longer sleep on error
    
    def manual_open(self, reason: str):
        """Manually open the circuit breaker"""
        
        with self._lock:
            if self.state != CircuitState.OPEN:
                logger.warning(f"Manually opening circuit breaker '{self.config.name}': {reason}")
                self._transition_to_open(BreakReason.MANUAL_BREAK)
    
    def manual_close(self):
        """Manually close the circuit breaker"""
        
        with self._lock:
            if self.state != CircuitState.CLOSED:
                logger.info(f"Manually closing circuit breaker '{self.config.name}'")
                self._transition_to_closed()
    
    def add_state_change_callback(self, callback: Callable):
        """Add callback for state changes"""
        self.state_change_callbacks.append(callback)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current circuit breaker status"""
        
        recent_calls = self._get_recent_calls()
        
        # Calculate metrics
        success_rate = 0.0
        avg_latency = 0.0
        
        if recent_calls:
            success_rate = sum(1 for call in recent_calls if call.success) / len(recent_calls)
            latencies = [call.duration_ms for call in recent_calls]
            avg_latency = statistics.mean(latencies)
        
        return {
            "name": self.config.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_rate": success_rate,
            "avg_latency_ms": avg_latency,
            "recent_calls": len(recent_calls),
            "last_state_change": self.last_state_change.isoformat(),
            "half_open_calls": self.half_open_calls if self.state == CircuitState.HALF_OPEN else None,
            "half_open_successes": self.half_open_successes if self.state == CircuitState.HALF_OPEN else None,
            "config": {
                "failure_threshold": self.config.failure_threshold,
                "timeout_threshold_ms": self.config.timeout_threshold_ms,
                "error_rate_threshold": self.config.error_rate_threshold,
                "recovery_timeout_seconds": self.config.recovery_timeout_seconds
            }
        }
    
    def shutdown(self):
        """Shutdown circuit breaker monitoring"""
        self._monitoring_active = False
        logger.info(f"Circuit breaker '{self.config.name}' monitoring shutdown")


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class CircuitBreakerManager:
    """
    Manager for multiple circuit breakers
    """
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreakerSystem] = {}
        self._lock = threading.RLock()
    
    def create_circuit_breaker(self, config: CircuitBreakerConfig) -> CircuitBreakerSystem:
        """Create and register a new circuit breaker"""
        
        with self._lock:
            circuit_breaker = CircuitBreakerSystem(config)
            self.circuit_breakers[config.name] = circuit_breaker
            
            logger.info(f"Created circuit breaker: {config.name}")
            return circuit_breaker
    
    def get_circuit_breaker(self, name: str) -> Optional[CircuitBreakerSystem]:
        """Get circuit breaker by name"""
        return self.circuit_breakers.get(name)
    
    def protect_call(self, circuit_name: str, protected_function: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        
        circuit_breaker = self.circuit_breakers.get(circuit_name)
        if not circuit_breaker:
            raise ValueError(f"Circuit breaker '{circuit_name}' not found")
        
        return circuit_breaker.call(protected_function, *args, **kwargs)
    
    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all circuit breakers"""
        
        with self._lock:
            return {
                name: cb.get_status()
                for name, cb in self.circuit_breakers.items()
            }
    
    def manual_open_all(self, reason: str):
        """Manually open all circuit breakers"""
        
        with self._lock:
            for circuit_breaker in self.circuit_breakers.values():
                circuit_breaker.manual_open(reason)
    
    def manual_close_all(self):
        """Manually close all circuit breakers"""
        
        with self._lock:
            for circuit_breaker in self.circuit_breakers.values():
                circuit_breaker.manual_close()
    
    def shutdown_all(self):
        """Shutdown all circuit breakers"""
        
        with self._lock:
            for circuit_breaker in self.circuit_breakers.values():
                circuit_breaker.shutdown()
            
            self.circuit_breakers.clear()
            logger.info("All circuit breakers shut down")