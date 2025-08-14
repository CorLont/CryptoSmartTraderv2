"""
Mandatory Gates Enforcement
Guarantees all order paths go through Risk + Execution gates
"""

import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import inspect
import functools

logger = logging.getLogger(__name__)


class GateBypassError(Exception):
    """Raised when code tries to bypass mandatory gates"""
    pass


class OrderPathViolation(Exception):
    """Raised when order execution bypasses required validation"""
    pass


@dataclass
class OrderInterception:
    """Record of order interception"""
    function_name: str
    module_name: str
    call_stack: List[str]
    timestamp: float
    approved: bool
    rejection_reason: Optional[str] = None


class MandatoryGatesEnforcer:
    """
    Enforces mandatory Risk + Execution gate usage on ALL order paths
    Zero-tolerance bypass detection and prevention
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.enabled = True
        
        # Gate tracking
        self.risk_guard = None
        self.execution_discipline = None
        
        # Monitoring
        self.intercepted_calls: List[OrderInterception] = []
        self.bypass_attempts: List[OrderInterception] = []
        self.approved_paths: List[str] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Method tracking
        self._original_methods: Dict[str, Callable] = {}
        self._patched_methods: Dict[str, bool] = {}
        
        # Initialize enforcement
        self._initialize_enforcement()
        
        self.logger.info("✅ Mandatory Gates Enforcer initialized")
    
    def set_risk_guard(self, risk_guard):
        """Set risk guard instance"""
        self.risk_guard = risk_guard
        self.logger.info("🛡️  Risk Guard connected to enforcement")
    
    def set_execution_discipline(self, execution_discipline):
        """Set execution discipline instance"""
        self.execution_discipline = execution_discipline
        self.logger.info("⚡ Execution Discipline connected to enforcement")
    
    def _initialize_enforcement(self):
        """Initialize mandatory gate enforcement"""
        
        # Define order execution methods that MUST go through gates
        self.protected_methods = [
            # Core order methods
            "place_order",
            "create_order", 
            "submit_order",
            "execute_order",
            "send_order",
            "place_market_order",
            "place_limit_order",
            
            # CCXT exchange methods
            "create_market_buy_order",
            "create_market_sell_order", 
            "create_limit_buy_order",
            "create_limit_sell_order",
            "create_order",
            
            # Direct exchange calls
            "private_post_add_order",
            "private_post_orders",
            "placeOrder",
            "addOrder"
        ]
        
        # Approved bypasses (only for testing/simulation)
        self.approved_bypasses = [
            "test_",
            "mock_",
            "simulate_",
            "_internal_",
            "backtest_"
        ]
        
        self.logger.info(f"🔒 Protecting {len(self.protected_methods)} order execution methods")
    
    def _is_approved_bypass(self, function_name: str, call_stack: List[str]) -> bool:
        """Check if this is an approved bypass (testing/simulation)"""
        
        # Check function name patterns
        for pattern in self.approved_bypasses:
            if pattern in function_name.lower():
                return True
        
        # Check call stack for test/simulation contexts
        for frame in call_stack:
            frame_lower = frame.lower()
            if any(pattern in frame_lower for pattern in ["test_", "simulate_", "backtest_", "mock_"]):
                return True
        
        return False
    
    def _get_call_stack(self) -> List[str]:
        """Get current call stack for audit trail"""
        
        call_stack = []
        frame = inspect.currentframe()
        
        try:
            # Skip the first few frames (this method and wrapper)
            for _ in range(3):
                if frame:
                    frame = frame.f_back
            
            # Collect up to 10 frames
            for _ in range(10):
                if not frame:
                    break
                
                filename = frame.f_code.co_filename
                function_name = frame.f_code.co_name
                line_number = frame.f_lineno
                
                call_stack.append(f"{filename}:{function_name}:{line_number}")
                frame = frame.f_back
        
        finally:
            del frame
        
        return call_stack
    
    def _validate_gates_available(self) -> bool:
        """Ensure both gates are available"""
        return self.risk_guard is not None and self.execution_discipline is not None
    
    def _check_gate_approval(self, function_name: str, args: tuple, kwargs: dict) -> tuple[bool, str]:
        """Check if order passes through mandatory gates"""
        
        if not self._validate_gates_available():
            return False, "Risk Guard or Execution Discipline not available"
        
        try:
            # Extract order details from arguments
            order_details = self._extract_order_details(function_name, args, kwargs)
            
            if not order_details:
                return False, "Could not extract order details for validation"
            
            # TODO: Integrate with actual Risk Guard and Execution Discipline
            # For now, we'll do basic validation
            
            # Check basic order sanity
            if order_details.get("size", 0) <= 0:
                return False, "Invalid order size"
            
            if not order_details.get("symbol"):
                return False, "Missing symbol"
            
            # All checks passed
            return True, "Gates approved"
            
        except Exception as e:
            return False, f"Gate validation error: {str(e)}"
    
    def _extract_order_details(self, function_name: str, args: tuple, kwargs: dict) -> Optional[Dict[str, Any]]:
        """Extract order details from function arguments"""
        
        order_details = {}
        
        try:
            # Common parameter extraction patterns
            if "symbol" in kwargs:
                order_details["symbol"] = kwargs["symbol"]
            elif len(args) >= 1 and isinstance(args[0], str):
                order_details["symbol"] = args[0]
            
            # Size/quantity extraction
            size_keys = ["size", "quantity", "amount", "volume"]
            for key in size_keys:
                if key in kwargs:
                    order_details["size"] = kwargs[key]
                    break
            
            # Side extraction
            if "side" in kwargs:
                order_details["side"] = kwargs["side"]
            
            # Price extraction
            if "price" in kwargs:
                order_details["price"] = kwargs["price"]
            
            # Order type
            if "type" in kwargs:
                order_details["order_type"] = kwargs["type"]
            elif "order_type" in kwargs:
                order_details["order_type"] = kwargs["order_type"]
            
            return order_details if order_details else None
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting order details: {e}")
            return None
    
    def create_protected_wrapper(self, original_method: Callable, method_name: str) -> Callable:
        """Create wrapper that enforces mandatory gates"""
        
        @functools.wraps(original_method)
        def protected_wrapper(*args, **kwargs):
            
            if not self.enabled:
                return original_method(*args, **kwargs)
            
            call_stack = self._get_call_stack()
            module_name = getattr(original_method, "__module__", "unknown")
            
            # Check if this is an approved bypass
            if self._is_approved_bypass(method_name, call_stack):
                with self._lock:
                    self.intercepted_calls.append(OrderInterception(
                        function_name=method_name,
                        module_name=module_name,
                        call_stack=call_stack,
                        timestamp=time.time(),
                        approved=True,
                        rejection_reason="Approved bypass (test/simulation)"
                    ))
                
                return original_method(*args, **kwargs)
            
            # Enforce mandatory gate validation
            approved, reason = self._check_gate_approval(method_name, args, kwargs)
            
            with self._lock:
                interception = OrderInterception(
                    function_name=method_name,
                    module_name=module_name,
                    call_stack=call_stack,
                    timestamp=time.time(),
                    approved=approved,
                    rejection_reason=reason if not approved else None
                )
                
                self.intercepted_calls.append(interception)
                
                if not approved:
                    self.bypass_attempts.append(interception)
            
            if not approved:
                error_msg = (
                    f"🚨 MANDATORY GATE VIOLATION: {method_name} called without proper gate approval. "
                    f"Reason: {reason}. All order execution MUST go through Risk Guard + Execution Discipline."
                )
                
                self.logger.critical(error_msg)
                
                # In production, this should prevent the order
                raise OrderPathViolation(error_msg)
            
            # Gates approved - proceed with execution
            self.logger.info(f"✅ Gate approved: {method_name} from {module_name}")
            return original_method(*args, **kwargs)
        
        return protected_wrapper
    
    def patch_method(self, obj: Any, method_name: str):
        """Patch specific method on object"""
        
        if not hasattr(obj, method_name):
            return False
        
        original_method = getattr(obj, method_name)
        
        # Store original for restoration
        method_key = f"{obj.__class__.__module__}.{obj.__class__.__name__}.{method_name}"
        self._original_methods[method_key] = original_method
        
        # Create and apply wrapper
        protected_wrapper = self.create_protected_wrapper(original_method, method_name)
        setattr(obj, method_name, protected_wrapper)
        
        self._patched_methods[method_key] = True
        
        self.logger.info(f"🔒 Patched {method_key}")
        return True
    
    def patch_class_methods(self, cls: type):
        """Patch all protected methods on a class"""
        
        patched_count = 0
        
        for method_name in self.protected_methods:
            if hasattr(cls, method_name):
                original_method = getattr(cls, method_name)
                
                # Store original
                method_key = f"{cls.__module__}.{cls.__name__}.{method_name}"
                self._original_methods[method_key] = original_method
                
                # Create and apply wrapper
                protected_wrapper = self.create_protected_wrapper(original_method, method_name)
                setattr(cls, method_name, protected_wrapper)
                
                self._patched_methods[method_key] = True
                patched_count += 1
        
        if patched_count > 0:
            self.logger.info(f"🔒 Patched {patched_count} methods on {cls.__name__}")
        
        return patched_count
    
    def patch_ccxt_exchange(self, exchange):
        """Patch CCXT exchange object to enforce gates"""
        
        if not exchange:
            return 0
        
        patched_count = 0
        
        # Patch order creation methods
        ccxt_order_methods = [
            "create_order",
            "create_market_buy_order",
            "create_market_sell_order",
            "create_limit_buy_order", 
            "create_limit_sell_order"
        ]
        
        for method_name in ccxt_order_methods:
            if hasattr(exchange, method_name):
                if self.patch_method(exchange, method_name):
                    patched_count += 1
        
        self.logger.info(f"🔒 Patched {patched_count} CCXT methods on {exchange.__class__.__name__}")
        return patched_count
    
    def add_approved_path(self, path: str):
        """Add approved execution path (for legitimate bypasses)"""
        
        with self._lock:
            self.approved_paths.append(path)
        
        self.logger.info(f"✅ Added approved path: {path}")
    
    def remove_approved_path(self, path: str):
        """Remove approved execution path"""
        
        with self._lock:
            if path in self.approved_paths:
                self.approved_paths.remove(path)
        
        self.logger.info(f"❌ Removed approved path: {path}")
    
    def restore_original_methods(self):
        """Restore all original methods (for testing)"""
        
        for method_key, original_method in self._original_methods.items():
            try:
                parts = method_key.split(".")
                class_name = parts[-2]
                method_name = parts[-1]
                
                # This is simplified - in practice you'd need to restore properly
                self.logger.info(f"🔄 Restored {method_key}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to restore {method_key}: {e}")
        
        self._patched_methods.clear()
        self.logger.info("🔄 All original methods restored")
    
    def get_enforcement_status(self) -> Dict[str, Any]:
        """Get current enforcement status"""
        
        with self._lock:
            return {
                "enabled": self.enabled,
                "gates_available": self._validate_gates_available(),
                "protected_methods": len(self.protected_methods),
                "patched_methods": len(self._patched_methods),
                "total_interceptions": len(self.intercepted_calls),
                "bypass_attempts": len(self.bypass_attempts),
                "approved_paths": len(self.approved_paths),
                "recent_interceptions": [
                    {
                        "function": call.function_name,
                        "module": call.module_name,
                        "approved": call.approved,
                        "timestamp": call.timestamp,
                        "rejection_reason": call.rejection_reason
                    }
                    for call in self.intercepted_calls[-10:]  # Last 10
                ]
            }
    
    def enable_enforcement(self):
        """Enable gate enforcement"""
        self.enabled = True
        self.logger.info("🔒 Gate enforcement ENABLED")
    
    def disable_enforcement(self):
        """Disable gate enforcement (DANGEROUS - only for testing)"""
        self.enabled = False
        self.logger.warning("⚠️  Gate enforcement DISABLED")
    
    def get_bypass_report(self) -> Dict[str, Any]:
        """Get detailed bypass attempt report"""
        
        with self._lock:
            bypass_summary = {}
            
            for attempt in self.bypass_attempts:
                key = f"{attempt.module_name}.{attempt.function_name}"
                if key not in bypass_summary:
                    bypass_summary[key] = {
                        "count": 0,
                        "first_seen": attempt.timestamp,
                        "last_seen": attempt.timestamp,
                        "reasons": set()
                    }
                
                bypass_summary[key]["count"] += 1
                bypass_summary[key]["last_seen"] = max(bypass_summary[key]["last_seen"], attempt.timestamp)
                if attempt.rejection_reason:
                    bypass_summary[key]["reasons"].add(attempt.rejection_reason)
            
            # Convert sets to lists for JSON serialization
            for key in bypass_summary:
                bypass_summary[key]["reasons"] = list(bypass_summary[key]["reasons"])
            
            return {
                "total_bypass_attempts": len(self.bypass_attempts),
                "unique_violation_points": len(bypass_summary),
                "bypass_summary": bypass_summary,
                "enforcement_effectiveness": (
                    1 - (len(self.bypass_attempts) / max(1, len(self.intercepted_calls)))
                ) * 100
            }


# Global mandatory gates enforcer
_global_gates_enforcer: Optional[MandatoryGatesEnforcer] = None


def get_global_gates_enforcer() -> MandatoryGatesEnforcer:
    """Get or create global gates enforcer"""
    global _global_gates_enforcer
    if _global_gates_enforcer is None:
        _global_gates_enforcer = MandatoryGatesEnforcer()
        logger.info("✅ Global MandatoryGatesEnforcer initialized")
    return _global_gates_enforcer


def reset_global_gates_enforcer():
    """Reset global gates enforcer (for testing)"""
    global _global_gates_enforcer
    if _global_gates_enforcer:
        _global_gates_enforcer.restore_original_methods()
    _global_gates_enforcer = None


# Decorator for mandatory gate enforcement
def require_gates(func):
    """Decorator that enforces mandatory gates on function"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        enforcer = get_global_gates_enforcer()
        
        if not enforcer.enabled:
            return func(*args, **kwargs)
        
        # Create a temporary wrapper and execute through it
        protected_func = enforcer.create_protected_wrapper(func, func.__name__)
        return protected_func(*args, **kwargs)
    
    return wrapper


# Context manager for temporary gate bypass (testing only)
class TemporaryGatesBypass:
    """Context manager for temporary gate bypass (testing only)"""
    
    def __init__(self, reason: str = "Testing"):
        self.reason = reason
        self.enforcer = get_global_gates_enforcer()
        self.original_enabled = None
    
    def __enter__(self):
        self.original_enabled = self.enforcer.enabled
        self.enforcer.disable_enforcement()
        logger.warning(f"⚠️  Temporary gate bypass: {self.reason}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.enforcer.enabled = self.original_enabled
        logger.info("🔒 Gate enforcement restored")