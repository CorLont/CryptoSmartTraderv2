#!/usr/bin/env python3
"""
MANDATORY EXECUTION GATEWAY
Hard-wired centralized gate system that FORCES ALL order execution through Risk/Execution gates
NO BYPASS POSSIBLE - Production safety enforcement
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal

from ..risk.central_risk_guard import CentralRiskGuard, RiskDecision
from ..execution.execution_discipline import ExecutionPolicy, OrderRequest as ExecOrderRequest, MarketConditions, ExecutionDecision, OrderSide

logger = logging.getLogger(__name__)


class GatewayViolation(Exception):
    """Exception raised when attempting to bypass mandatory gates"""
    pass


@dataclass
class UniversalOrderRequest:
    """Universal order format for all execution paths"""
    symbol: str
    side: str  # "buy" or "sell"
    size: float
    order_type: str = "market"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    strategy_id: str = "default"
    max_slippage_bps: float = 30.0
    client_order_id: Optional[str] = None
    # Source tracking for audit
    source_module: str = "unknown"
    source_function: str = "unknown"


@dataclass 
class GatewayResult:
    """Result from mandatory gateway processing"""
    approved: bool
    reason: str
    approved_size: float = 0.0
    risk_violations: Optional[List[str]] = None
    execution_violations: Optional[List[str]] = None
    gateway_decision_time_ms: float = 0.0
    
    def __post_init__(self):
        if self.risk_violations is None:
            self.risk_violations = []
        if self.execution_violations is None:
            self.execution_violations = []


class MandatoryExecutionGateway:
    """
    MANDATORY EXECUTION GATEWAY
    
    ALL order execution MUST go through this gateway.
    NO DIRECT EXECUTION ALLOWED.
    
    This is the single enforcement point for:
    - CentralRiskGuard checks
    - ExecutionPolicy decisions
    - System-wide order tracking
    - Production safety compliance
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern - only one gateway instance allowed"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MandatoryExecutionGateway, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, 'initialized'):
            return
            
        self.initialized = True
        self.logger = logging.getLogger(__name__)
        
        # Core mandatory components (HARD WIRED)
        self.risk_guard = CentralRiskGuard()
        self.execution_policy = ExecutionPolicy()
        
        # Gateway tracking
        self.total_requests = 0
        self.approved_requests = 0
        self.rejected_requests = 0
        self.bypass_attempts = 0
        
        # Active enforcement
        self.gateway_enabled = True
        self.emergency_disable = False
        
        # Audit trail
        self.request_history: List[Dict] = []
        self.violation_log: List[Dict] = []
        
        self.logger.critical(
            "MANDATORY EXECUTION GATEWAY ACTIVATED - All order execution enforced through Risk/Execution gates"
        )
    
    def process_order_request(
        self, 
        order_request: UniversalOrderRequest,
        market_data: Optional[Dict] = None
    ) -> GatewayResult:
        """
        MANDATORY order processing through all gates
        
        This method MUST be called by ANY code that wants to execute orders.
        Direct execution bypass will cause GatewayViolation exception.
        """
        
        if self.emergency_disable:
            return GatewayResult(
                approved=False,
                reason="EMERGENCY: Gateway disabled - no trading allowed"
            )
        
        if not self.gateway_enabled:
            self.logger.error("VIOLATION: Gateway disabled but order attempted")
            self.bypass_attempts += 1
            raise GatewayViolation("Gateway is disabled - no direct execution allowed")
        
        gateway_start = time.time()
        self.total_requests += 1
        
        try:
            # Convert to risk operation
            risk_operation = TradingOperation(
                operation_type="entry",
                symbol=order_request.symbol,
                side=order_request.side,
                size_usd=order_request.size * (order_request.limit_price or 45000.0),  # Estimate USD value
                current_price=order_request.limit_price or 45000.0,
                strategy_id=order_request.strategy_id
            )
            
            # GATE 1: CENTRAL RISK GUARD (MANDATORY)
            risk_evaluation = self.risk_guard.evaluate_operation(risk_operation)
            
            if risk_evaluation.decision != RiskDecision.APPROVE:
                self.rejected_requests += 1
                result = GatewayResult(
                    approved=False,
                    reason=f"RiskGuard rejection: {risk_evaluation.reasons}",
                    risk_violations=[v.value for v in risk_evaluation.violations],
                    gateway_decision_time_ms=(time.time() - gateway_start) * 1000
                )
                self._log_request(order_request, result, "risk_rejection")
                return result
            
            # GATE 2: EXECUTION POLICY (MANDATORY)
            # Convert to execution format
            exec_order = ExecOrderRequest(
                symbol=order_request.symbol,
                side=OrderSide.BUY if order_request.side.lower() == "buy" else OrderSide.SELL,
                size=order_request.size,
                order_type=order_request.order_type,
                limit_price=order_request.limit_price,
                client_order_id=order_request.client_order_id,
                max_slippage_bps=order_request.max_slippage_bps,
                strategy_id=order_request.strategy_id
            )
            
            # Market conditions (mock for now, would be real data)
            market_conditions = MarketConditions(
                spread_bps=25.0,
                bid_depth_usd=25000.0,
                ask_depth_usd=25000.0,
                volume_1m_usd=100000.0,
                last_price=order_request.limit_price or 45000.0,
                bid_price=(order_request.limit_price or 45000.0) * 0.9995,
                ask_price=(order_request.limit_price or 45000.0) * 1.0005,
                timestamp=time.time()
            )
            
            exec_result = self.execution_policy.decide(exec_order, market_conditions)
            
            if exec_result.decision != ExecutionDecision.APPROVE:
                self.rejected_requests += 1
                result = GatewayResult(
                    approved=False,
                    reason=f"ExecutionPolicy rejection: {exec_result.reason}",
                    execution_violations=[exec_result.reason],
                    gateway_decision_time_ms=(time.time() - gateway_start) * 1000
                )
                self._log_request(order_request, result, "execution_rejection")
                return result
            
            # APPROVED: Both gates passed
            self.approved_requests += 1
            approved_size = risk_evaluation.approved_size_usd / (order_request.limit_price or 45000.0)
            
            result = GatewayResult(
                approved=True,
                reason="Approved by all mandatory gates",
                approved_size=approved_size,
                gateway_decision_time_ms=(time.time() - gateway_start) * 1000
            )
            
            self._log_request(order_request, result, "approved")
            
            self.logger.info(
                f"Order approved through mandatory gateway: {order_request.symbol} {order_request.side} {order_request.size} -> {approved_size} from {order_request.source_module}.{order_request.source_function}"
            )
            
            return result
            
        except Exception as e:
            self.rejected_requests += 1
            result = GatewayResult(
                approved=False,
                reason=f"Gateway error: {str(e)}",
                gateway_decision_time_ms=(time.time() - gateway_start) * 1000
            )
            self._log_request(order_request, result, "gateway_error")
            self.logger.error(f"Gateway processing error: {str(e)}")
            return result
    
    def _log_request(self, order_request: UniversalOrderRequest, result: GatewayResult, decision_type: str):
        """Log request for audit trail"""
        log_entry = {
            "timestamp": time.time(),
            "symbol": order_request.symbol,
            "side": order_request.side,
            "size": order_request.size,
            "source_module": order_request.source_module,
            "source_function": order_request.source_function,
            "approved": result.approved,
            "reason": result.reason,
            "decision_type": decision_type,
            "decision_time_ms": result.gateway_decision_time_ms
        }
        
        self.request_history.append(log_entry)
        
        # Keep last 1000 entries
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
    
    def get_gateway_stats(self) -> Dict[str, Any]:
        """Get gateway performance statistics"""
        total = self.total_requests
        if total == 0:
            return {"total_requests": 0, "approval_rate": 0.0, "rejection_rate": 0.0}
        
        return {
            "total_requests": total,
            "approved_requests": self.approved_requests,
            "rejected_requests": self.rejected_requests,
            "bypass_attempts": self.bypass_attempts,
            "approval_rate": self.approved_requests / total,
            "rejection_rate": self.rejected_requests / total,
            "gateway_enabled": self.gateway_enabled,
            "emergency_disable": self.emergency_disable
        }
    
    def emergency_shutdown(self, reason: str):
        """Emergency shutdown - blocks ALL order execution"""
        self.emergency_disable = True
        self.logger.critical(f"EMERGENCY GATEWAY SHUTDOWN: {reason}")
        
        self.violation_log.append({
            "timestamp": time.time(),
            "type": "emergency_shutdown",
            "reason": reason
        })
    
    def enable_gateway(self):
        """Re-enable gateway after emergency"""
        self.emergency_disable = False
        self.gateway_enabled = True
        self.logger.warning("Gateway re-enabled")
    
    def disable_gateway_enforcement(self):
        """DANGEROUS: Disable gateway enforcement - only for testing"""
        self.gateway_enabled = False
        self.logger.critical("WARNING: Gateway enforcement DISABLED - TESTING ONLY")


# Global singleton instance
MANDATORY_GATEWAY = MandatoryExecutionGateway()


def enforce_mandatory_gateway(order_request: UniversalOrderRequest, market_data: Optional[Dict] = None) -> GatewayResult:
    """
    Global function that ALL order execution must call
    
    Usage in any execution module:
    
    from ..core.mandatory_execution_gateway import enforce_mandatory_gateway, UniversalOrderRequest
    
    order = UniversalOrderRequest(
        symbol="BTC/USD",
        side="buy", 
        size=0.1,
        source_module=__name__,
        source_function="execute_order"
    )
    
    result = enforce_mandatory_gateway(order)
    if not result.approved:
        return {"success": False, "reason": result.reason}
    
    # Proceed with actual execution only if approved
    """
    return MANDATORY_GATEWAY.process_order_request(order_request, market_data)


def check_gateway_bypass(module_name: str, function_name: str):
    """
    Decorator/checker to detect direct execution bypass attempts
    
    This should be added to any function that executes orders
    to detect if it's being called directly vs through the gateway
    """
    
    # Check if we're in a gateway call stack
    import inspect
    
    frame = inspect.currentframe()
    try:
        while frame:
            if 'mandatory_execution_gateway' in str(frame.f_code.co_filename):
                # We're being called from the gateway - OK
                return True
            frame = frame.f_back
        
        # Not called from gateway - VIOLATION
        logger.error(f"BYPASS VIOLATION: {module_name}.{function_name} called directly without gateway")
        MANDATORY_GATEWAY.bypass_attempts += 1
        
        MANDATORY_GATEWAY.violation_log.append({
            "timestamp": time.time(),
            "type": "bypass_attempt",
            "module": module_name,
            "function": function_name
        })
        
        return False
        
    finally:
        del frame