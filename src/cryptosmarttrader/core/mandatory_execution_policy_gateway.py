#!/usr/bin/env python3
"""
MANDATORY EXECUTION POLICY GATEWAY
Enforces that ALL order execution paths go through ExecutionPolicy gates
Zero bypass architecture for execution discipline
"""

import logging
import time
import threading
from typing import Dict, Optional, Tuple, Any, Union, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ExecutionPolicyViolation(Exception):
    """Raised when ExecutionPolicy is bypassed"""
    pass


class GatewayEnforcement(Enum):
    """Enforcement levels for gateway"""
    STRICT = "strict"        # No bypass allowed
    WARNING = "warning"      # Log bypass but allow
    DISABLED = "disabled"    # Gateway disabled


@dataclass
class ExecutionPolicyResult:
    """Result from mandatory execution policy evaluation"""
    approved: bool
    client_order_id: str
    reason: str
    gate_results: Dict[str, bool]
    slippage_budget_used: float
    tif_validated: bool
    execution_time_ms: float
    policy_score: float = 0.0
    recommendations: List[str] = None


class MandatoryExecutionPolicyGateway:
    """
    Singleton gateway ensuring ALL order execution goes through ExecutionPolicy
    
    Features:
    - Mandatory ExecutionPolicy.decide() calls
    - Idempotent Client Order ID (COID) enforcement
    - Time-in-Force (TIF) validation
    - Slippage budget controls
    - Zero bypass architecture
    - Complete audit trail
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
            
        self.logger = logging.getLogger(__name__)
        self._initialized = True
        
        # Gateway enforcement
        self.enforcement_level = GatewayEnforcement.STRICT
        self.gateway_active = True
        
        # ExecutionPolicy integration
        self._execution_policy = None
        self._policy_lock = threading.Lock()
        
        # Enforcement tracking
        self.total_order_checks = 0
        self.approved_orders = 0
        self.rejected_orders = 0
        self.bypass_attempts = 0
        self.policy_violations = []
        
        # Performance tracking
        self.avg_evaluation_time_ms = 0.0
        self.max_evaluation_time_ms = 0.0
        
        self.logger.info("🛡️ MANDATORY EXECUTION POLICY GATEWAY ACTIVATED")
    
    def _get_execution_policy(self):
        """Get or create ExecutionPolicy instance"""
        if self._execution_policy is None:
            with self._policy_lock:
                if self._execution_policy is None:
                    try:
                        from ..execution.execution_discipline import ExecutionPolicy
                        self._execution_policy = ExecutionPolicy()
                        self.logger.info("ExecutionPolicy instance created for gateway")
                    except Exception as e:
                        self.logger.error(f"Failed to create ExecutionPolicy: {str(e)}")
                        raise ExecutionPolicyViolation(f"Cannot create ExecutionPolicy: {str(e)}")
        
        return self._execution_policy
    
    def enforce_execution_policy(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "limit",
        limit_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        max_slippage_bps: float = 10.0,
        time_in_force: str = "post_only",
        strategy_id: str = "default",
        market_conditions: Optional[Dict] = None
    ) -> ExecutionPolicyResult:
        """
        MANDATORY enforcement of ExecutionPolicy for all orders
        
        This function MUST be called for every order execution
        No bypass allowed when enforcement is STRICT
        """
        
        if not self.gateway_active:
            raise ExecutionPolicyViolation("ExecutionPolicy gateway is not active")
        
        start_time = time.time()
        self.total_order_checks += 1
        
        try:
            # Get ExecutionPolicy instance
            execution_policy = self._get_execution_policy()
            
            # Import required types
            from ..execution.execution_discipline import (
                OrderRequest, OrderSide, TimeInForce, MarketConditions
            )
            
            # Convert parameters to proper types
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            
            # Map TIF string to enum
            tif_mapping = {
                "gtc": TimeInForce.GTC,
                "ioc": TimeInForce.IOC, 
                "fok": TimeInForce.FOK,
                "post_only": TimeInForce.POST_ONLY
            }
            tif_enum = tif_mapping.get(time_in_force.lower(), TimeInForce.POST_ONLY)
            
            # Create OrderRequest
            order_request = OrderRequest(
                symbol=symbol,
                side=order_side,
                size=size,
                order_type=order_type,
                limit_price=limit_price,
                time_in_force=tif_enum,
                client_order_id=client_order_id,
                max_slippage_bps=max_slippage_bps,
                strategy_id=strategy_id
            )
            
            # Create or use provided MarketConditions
            if market_conditions:
                market_cond = MarketConditions(
                    spread_bps=market_conditions.get("spread_bps", 20.0),
                    bid_depth_usd=market_conditions.get("bid_depth_usd", 10000.0),
                    ask_depth_usd=market_conditions.get("ask_depth_usd", 10000.0),
                    volume_1m_usd=market_conditions.get("volume_1m_usd", 100000.0),
                    last_price=market_conditions.get("last_price", limit_price or 100.0),
                    bid_price=market_conditions.get("bid_price", (limit_price or 100.0) * 0.999),
                    ask_price=market_conditions.get("ask_price", (limit_price or 100.0) * 1.001),
                    timestamp=market_conditions.get("timestamp", time.time())
                )
            else:
                # Create default market conditions if none provided
                base_price = limit_price or 100.0
                market_cond = MarketConditions(
                    spread_bps=20.0,
                    bid_depth_usd=10000.0,
                    ask_depth_usd=10000.0,
                    volume_1m_usd=100000.0,
                    last_price=base_price,
                    bid_price=base_price * 0.999,
                    ask_price=base_price * 1.001,
                    timestamp=time.time()
                )
            
            # MANDATORY ExecutionPolicy.decide() call
            execution_result = execution_policy.decide(order_request, market_cond)
            
            # Calculate execution time
            execution_time_ms = (time.time() - start_time) * 1000
            self._update_performance_metrics(execution_time_ms)
            
            # Track results
            if execution_result.decision.value == "approve":
                self.approved_orders += 1
                
                # Mark order as executed in policy's idempotency tracker
                execution_policy.idempotency.mark_executed(order_request.client_order_id)
                
                policy_result = ExecutionPolicyResult(
                    approved=True,
                    client_order_id=order_request.client_order_id,
                    reason="All ExecutionPolicy gates passed",
                    gate_results=execution_result.gate_results,
                    slippage_budget_used=order_request.max_slippage_bps,
                    tif_validated=execution_result.gate_results.get("tif", False),
                    execution_time_ms=execution_time_ms,
                    policy_score=100.0 - execution_result.risk_score,
                    recommendations=[]
                )
                
                self.logger.info(f"✅ ExecutionPolicy APPROVED: {symbol} {side} {size}")
                return policy_result
                
            else:
                self.rejected_orders += 1
                
                # Mark order as failed in policy's idempotency tracker
                execution_policy.idempotency.mark_failed(order_request.client_order_id)
                
                policy_result = ExecutionPolicyResult(
                    approved=False,
                    client_order_id=order_request.client_order_id,
                    reason=execution_result.reason,
                    gate_results=execution_result.gate_results,
                    slippage_budget_used=order_request.max_slippage_bps,
                    tif_validated=execution_result.gate_results.get("tif", False),
                    execution_time_ms=execution_time_ms,
                    policy_score=execution_result.risk_score,
                    recommendations=[
                        "Check market conditions",
                        "Adjust order parameters",
                        "Consider different execution time"
                    ]
                )
                
                self.logger.warning(f"❌ ExecutionPolicy REJECTED: {symbol} {side} {size} - {execution_result.reason}")
                return policy_result
        
        except Exception as e:
            self.rejected_orders += 1
            execution_time_ms = (time.time() - start_time) * 1000
            
            error_result = ExecutionPolicyResult(
                approved=False,
                client_order_id=client_order_id or "unknown",
                reason=f"ExecutionPolicy enforcement error: {str(e)}",
                gate_results={},
                slippage_budget_used=max_slippage_bps,
                tif_validated=False,
                execution_time_ms=execution_time_ms
            )
            
            self.logger.error(f"❌ ExecutionPolicy enforcement failed: {str(e)}")
            
            if self.enforcement_level == GatewayEnforcement.STRICT:
                raise ExecutionPolicyViolation(f"ExecutionPolicy enforcement failed: {str(e)}")
            
            return error_result
    
    def _update_performance_metrics(self, execution_time_ms: float):
        """Update performance tracking metrics"""
        if self.total_order_checks == 1:
            self.avg_evaluation_time_ms = execution_time_ms
        else:
            # Running average
            self.avg_evaluation_time_ms = (
                (self.avg_evaluation_time_ms * (self.total_order_checks - 1) + execution_time_ms) / 
                self.total_order_checks
            )
        
        if execution_time_ms > self.max_evaluation_time_ms:
            self.max_evaluation_time_ms = execution_time_ms
    
    def detect_bypass_attempt(self, function_name: str, call_stack: List[str]):
        """Detect and log bypass attempts"""
        self.bypass_attempts += 1
        
        violation = {
            "timestamp": time.time(),
            "function": function_name,
            "call_stack": call_stack,
            "enforcement_level": self.enforcement_level.value
        }
        
        self.policy_violations.append(violation)
        
        self.logger.error(f"🚨 BYPASS ATTEMPT DETECTED: {function_name}")
        self.logger.error(f"Call stack: {' -> '.join(call_stack)}")
        
        if self.enforcement_level == GatewayEnforcement.STRICT:
            raise ExecutionPolicyViolation(
                f"ExecutionPolicy bypass attempted in {function_name}. "
                f"All order execution must go through enforce_execution_policy()"
            )
    
    def get_gateway_status(self) -> Dict[str, Any]:
        """Get current gateway status and metrics"""
        approval_rate = (self.approved_orders / max(1, self.total_order_checks)) * 100
        
        return {
            "gateway_active": self.gateway_active,
            "enforcement_level": self.enforcement_level.value,
            "total_order_checks": self.total_order_checks,
            "approved_orders": self.approved_orders,
            "rejected_orders": self.rejected_orders,
            "approval_rate_pct": approval_rate,
            "bypass_attempts": self.bypass_attempts,
            "policy_violations": len(self.policy_violations),
            "avg_evaluation_time_ms": self.avg_evaluation_time_ms,
            "max_evaluation_time_ms": self.max_evaluation_time_ms,
            "execution_policy_active": self._execution_policy is not None
        }
    
    def activate_gateway(self, enforcement_level: GatewayEnforcement = GatewayEnforcement.STRICT):
        """Activate the ExecutionPolicy gateway"""
        self.gateway_active = True
        self.enforcement_level = enforcement_level
        
        self.logger.info(f"🛡️ ExecutionPolicy Gateway ACTIVATED - Level: {enforcement_level.value}")
    
    def deactivate_gateway(self, reason: str = "Manual deactivation"):
        """Deactivate the ExecutionPolicy gateway"""
        self.gateway_active = False
        
        self.logger.warning(f"⚠️ ExecutionPolicy Gateway DEACTIVATED - Reason: {reason}")


# Global gateway instance
execution_policy_gateway = MandatoryExecutionPolicyGateway()


def enforce_execution_policy_check(
    symbol: str,
    side: str, 
    size: float,
    **kwargs
) -> ExecutionPolicyResult:
    """
    Convenient function to enforce ExecutionPolicy for all orders
    
    This is the primary interface that all execution modules should use
    """
    return execution_policy_gateway.enforce_execution_policy(
        symbol=symbol,
        side=side,
        size=size,
        **kwargs
    )


def require_execution_policy_approval(
    symbol: str,
    side: str,
    size: float,
    **kwargs
) -> bool:
    """
    Simple boolean check for ExecutionPolicy approval
    
    Returns True if order is approved, False otherwise
    """
    try:
        result = execution_policy_gateway.enforce_execution_policy(
            symbol=symbol,
            side=side, 
            size=size,
            **kwargs
        )
        return result.approved
    except Exception:
        return False


def get_execution_policy_status() -> Dict[str, Any]:
    """Get current ExecutionPolicy gateway status"""
    return execution_policy_gateway.get_gateway_status()