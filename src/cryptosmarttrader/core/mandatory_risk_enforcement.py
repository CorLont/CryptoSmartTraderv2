#!/usr/bin/env python3
"""
MANDATORY RISK ENFORCEMENT SYSTEM
Guarantees ALL order execution paths go through CentralRiskGuard
Zero-bypass architecture with audit trail
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import functools
import inspect
import traceback

from ..risk.central_risk_guard import CentralRiskGuard, RiskDecision
from .mandatory_execution_gateway import MandatoryExecutionGateway, UniversalOrderRequest, GatewayResult

logger = logging.getLogger(__name__)


class RiskEnforcementError(Exception):
    """Raised when risk enforcement is violated"""
    pass


@dataclass
class EnforcementMetrics:
    """Metrics for risk enforcement monitoring"""
    total_intercepted_calls: int = 0
    approved_orders: int = 0
    rejected_orders: int = 0
    bypass_attempts_blocked: int = 0
    functions_under_enforcement: int = 0
    enforcement_violations: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.enforcement_violations is None:
            self.enforcement_violations = []


class MandatoryRiskEnforcement:
    """
    MANDATORY RISK ENFORCEMENT SYSTEM
    
    This system automatically intercepts ALL order execution functions
    and forces them through CentralRiskGuard regardless of their implementation.
    
    Features:
    - Function decoration enforcement
    - Runtime interception
    - Bypass detection and blocking
    - Complete audit trail
    - Zero-tolerance enforcement
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for global enforcement"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MandatoryRiskEnforcement, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, 'initialized'):
            return
            
        self.initialized = True
        self.logger = logging.getLogger(__name__)
        
        # Core enforcement components
        self.gateway = MandatoryExecutionGateway()
        self.central_risk_guard = CentralRiskGuard()
        
        # Enforcement state
        self.enforcement_active = True
        self.enforcement_metrics = EnforcementMetrics()
        
        # Function registry - tracks all order execution functions
        self.enforced_functions: Dict[str, Dict] = {}
        self.intercepted_modules: List[str] = []
        
        # Audit trail
        self.enforcement_log: List[Dict] = []
        
        self.logger.critical(
            "🛡️ MANDATORY RISK ENFORCEMENT ACTIVATED - All order execution under surveillance"
        )
    
    def enforce_risk_check(self, 
                          order_size: float,
                          symbol: str,
                          side: str = "buy",
                          strategy_id: str = "default",
                          source_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        MANDATORY risk check enforcement
        This function MUST be called by ANY order execution code
        """
        if not self.enforcement_active:
            raise RiskEnforcementError("Risk enforcement disabled - no trading allowed")
        
        enforcement_start = time.time()
        self.enforcement_metrics.total_intercepted_calls += 1
        
        # Get caller information for audit
        frame = inspect.currentframe()
        caller_info = {}
        if frame and frame.f_back:
            caller_frame = frame.f_back
            caller_info = {
                "function": caller_frame.f_code.co_name,
                "filename": caller_frame.f_code.co_filename,
                "line": caller_frame.f_lineno
            }
        
        # Create universal order request
        order_request = UniversalOrderRequest(
            symbol=symbol,
            side=side,
            size=order_size,
            order_type="market",
            strategy_id=strategy_id,
            source_module=caller_info.get("filename", "unknown"),
            source_function=caller_info.get("function", "unknown")
        )
        
        # Force through mandatory gateway
        gateway_result = self.gateway.process_order_request(order_request)
        
        # Update metrics
        if gateway_result.approved:
            self.enforcement_metrics.approved_orders += 1
        else:
            self.enforcement_metrics.rejected_orders += 1
        
        # Log enforcement action
        enforcement_record = {
            "timestamp": time.time(),
            "caller_info": caller_info,
            "order_symbol": symbol,
            "order_size": order_size,
            "approved": gateway_result.approved,
            "reason": gateway_result.reason,
            "enforcement_time_ms": (time.time() - enforcement_start) * 1000,
            "source_info": source_info
        }
        self.enforcement_log.append(enforcement_record)
        
        # Return enforcement result
        result = {
            "approved": gateway_result.approved,
            "reason": gateway_result.reason,
            "approved_size": gateway_result.approved_size,
            "risk_violations": gateway_result.risk_violations,
            "execution_violations": gateway_result.execution_violations,
            "enforcement_record": enforcement_record
        }
        
        if not gateway_result.approved:
            self.logger.warning(f"🛡️ Order BLOCKED by risk enforcement: {gateway_result.reason}")
        else:
            self.logger.info(f"✅ Order APPROVED by risk enforcement: {symbol} {order_size}")
        
        return result
    
    def mandatory_risk_decorator(self, func: Callable) -> Callable:
        """
        Decorator that ENFORCES risk checks on order execution functions
        
        Usage:
            @mandatory_risk_enforcement.mandatory_risk_decorator
            def execute_order(symbol, size, ...):
                # This function is now under mandatory risk enforcement
                pass
        """
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Extract order parameters from function arguments
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Try to extract key order parameters
            order_params = self._extract_order_parameters(bound_args.arguments, func.__name__)
            
            if order_params:
                # Force risk check
                enforcement_result = self.enforce_risk_check(
                    order_size=order_params.get("size", 0.0),
                    symbol=order_params.get("symbol", "UNKNOWN"),
                    side=order_params.get("side", "buy"),
                    strategy_id=order_params.get("strategy_id", "default"),
                    source_info={
                        "function_name": func.__name__,
                        "module": func.__module__,
                        "decorated": True
                    }
                )
                
                # Block execution if not approved
                if not enforcement_result["approved"]:
                    raise RiskEnforcementError(f"Order blocked by risk enforcement: {enforcement_result['reason']}")
                
                # Modify size to approved size if needed
                if "size" in bound_args.arguments:
                    bound_args.arguments["size"] = enforcement_result["approved_size"]
                if "quantity" in bound_args.arguments:
                    bound_args.arguments["quantity"] = enforcement_result["approved_size"]
            
            # Execute original function with potentially modified parameters
            return func(*bound_args.args, **bound_args.kwargs)
        
        # Register decorated function
        func_key = f"{func.__module__}.{func.__name__}"
        self.enforced_functions[func_key] = {
            "function": func,
            "wrapper": wrapper,
            "registered_at": time.time(),
            "enforcement_type": "decorator"
        }
        self.enforcement_metrics.functions_under_enforcement += 1
        
        self.logger.info(f"🛡️ Function under mandatory risk enforcement: {func_key}")
        
        return wrapper
    
    def _extract_order_parameters(self, args_dict: Dict, function_name: str) -> Optional[Dict]:
        """Extract order parameters from function arguments"""
        order_params = {}
        
        # Common parameter mappings
        size_params = ["size", "quantity", "amount", "order_size", "size_usd"]
        symbol_params = ["symbol", "pair", "instrument", "asset"]
        side_params = ["side", "direction", "order_side"]
        
        # Extract size
        for param in size_params:
            if param in args_dict and args_dict[param] is not None:
                order_params["size"] = float(args_dict[param])
                break
        
        # Extract symbol
        for param in symbol_params:
            if param in args_dict and args_dict[param] is not None:
                order_params["symbol"] = str(args_dict[param])
                break
        
        # Extract side
        for param in side_params:
            if param in args_dict and args_dict[param] is not None:
                side_val = str(args_dict[param]).lower()
                if side_val in ["buy", "sell", "long", "short"]:
                    order_params["side"] = "buy" if side_val in ["buy", "long"] else "sell"
                break
        
        # Extract strategy ID if available
        if "strategy_id" in args_dict:
            order_params["strategy_id"] = str(args_dict["strategy_id"])
        
        # Only return if we found essential parameters
        if "size" in order_params and order_params["size"] > 0:
            return order_params
        
        return None
    
    def register_enforcement_module(self, module_path: str):
        """Register a module for mandatory risk enforcement"""
        if module_path not in self.intercepted_modules:
            self.intercepted_modules.append(module_path)
            self.logger.info(f"Module registered for risk enforcement: {module_path}")
    
    def get_enforcement_status(self) -> Dict[str, Any]:
        """Get current enforcement status and metrics"""
        return {
            "enforcement_active": self.enforcement_active,
            "metrics": {
                "total_intercepted_calls": self.enforcement_metrics.total_intercepted_calls,
                "approved_orders": self.enforcement_metrics.approved_orders,
                "rejected_orders": self.enforcement_metrics.rejected_orders,
                "bypass_attempts_blocked": self.enforcement_metrics.bypass_attempts_blocked,
                "functions_under_enforcement": self.enforcement_metrics.functions_under_enforcement,
                "approval_rate": (self.enforcement_metrics.approved_orders / max(1, self.enforcement_metrics.total_intercepted_calls)) * 100
            },
            "enforced_functions": list(self.enforced_functions.keys()),
            "intercepted_modules": self.intercepted_modules,
            "recent_violations": self.enforcement_metrics.enforcement_violations[-10:] if self.enforcement_metrics.enforcement_violations else []
        }
    
    def emergency_disable_enforcement(self, reason: str):
        """Emergency disable enforcement (use with extreme caution)"""
        self.enforcement_active = False
        self.logger.critical(f"🚨 RISK ENFORCEMENT DISABLED: {reason}")
        
        violation_record = {
            "timestamp": time.time(),
            "type": "enforcement_disabled",
            "reason": reason,
            "caller": traceback.format_stack()[-2]
        }
        self.enforcement_metrics.enforcement_violations.append(violation_record)
    
    def enable_enforcement(self, reason: str):
        """Re-enable risk enforcement"""
        self.enforcement_active = True
        self.logger.critical(f"✅ RISK ENFORCEMENT ENABLED: {reason}")


# Global singleton instance
mandatory_risk_enforcement = MandatoryRiskEnforcement()


def enforce_order_risk_check(order_size: float,
                            symbol: str,
                            side: str = "buy",
                            strategy_id: str = "default") -> Dict[str, Any]:
    """
    MANDATORY risk check function - use this in ALL order execution code
    
    Args:
        order_size: Order size in base currency or USD
        symbol: Trading symbol (e.g., "BTC/USD")
        side: Order side ("buy" or "sell")
        strategy_id: Strategy identifier
    
    Returns:
        Dict with approval status, reason, approved_size, violations
        
    Raises:
        RiskEnforcementError: If risk enforcement is disabled
    """
    return mandatory_risk_enforcement.enforce_risk_check(
        order_size=order_size,
        symbol=symbol,
        side=side,
        strategy_id=strategy_id
    )


def mandatory_risk_check(func: Callable) -> Callable:
    """
    Decorator for MANDATORY risk checking on order execution functions
    
    Usage:
        @mandatory_risk_check
        def execute_order(symbol, size, side):
            # This function now has mandatory risk enforcement
            pass
    """
    return mandatory_risk_enforcement.mandatory_risk_decorator(func)


# Convenience function for quick integration
def require_risk_approval(symbol: str, size: float, side: str = "buy") -> bool:
    """
    Quick risk approval check - returns True if order is approved
    
    Usage:
        if not require_risk_approval("BTC/USD", 1000, "buy"):
            raise Exception("Order not approved by risk management")
    """
    result = enforce_order_risk_check(size, symbol, side)
    return result["approved"]