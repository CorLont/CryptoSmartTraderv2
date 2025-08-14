"""
Hard Execution Discipline System
Mandatory gates for all order execution with idempotency protection
"""

import uuid
import time
import hashlib
import threading
from typing import Dict, Optional, Tuple, List, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class TimeInForce(Enum):
    GTC = "gtc"  # Good Till Canceled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    POST_ONLY = "post_only"  # Post-only (maker only)


class ExecutionDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    DEFER = "defer"


@dataclass
class MarketConditions:
    """Current market conditions for decision making"""
    spread_bps: float
    bid_depth_usd: float
    ask_depth_usd: float
    volume_1m_usd: float
    last_price: float
    bid_price: float
    ask_price: float
    timestamp: float


@dataclass
class OrderRequest:
    """Order request with all required parameters"""
    symbol: str
    side: OrderSide
    size: float
    order_type: str = "limit"
    limit_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.POST_ONLY
    client_order_id: Optional[str] = None
    max_slippage_bps: float = 10.0
    strategy_id: str = "default"
    
    def __post_init__(self):
        if not self.client_order_id:
            self.client_order_id = self._generate_idempotent_id()
    
    def _generate_idempotent_id(self) -> str:
        """Generate idempotent client order ID"""
        # Create deterministic ID based on order parameters
        params_str = f"{self.symbol}_{self.side.value}_{self.size}_{self.limit_price}_{self.strategy_id}_{int(time.time() // 60)}"
        hash_obj = hashlib.sha256(params_str.encode())
        return f"CST_{hash_obj.hexdigest()[:16]}"


@dataclass
class ExecutionGates:
    """Execution gate thresholds"""
    max_spread_bps: float = 50.0
    min_depth_usd: float = 10000.0
    min_volume_1m_usd: float = 100000.0
    max_slippage_bps: float = 25.0
    require_post_only: bool = True


@dataclass
class ExecutionResult:
    """Result of execution decision"""
    decision: ExecutionDecision
    reason: str
    approved_order: Optional[OrderRequest] = None
    risk_score: float = 0.0
    gate_results: Dict[str, bool] = field(default_factory=dict)


class IdempotencyTracker:
    """Track orders to prevent double execution"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self._executed_orders: Set[str] = set()
        self._pending_orders: Set[str] = set()
        self._order_timestamps: Dict[str, float] = {}
        self._ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
    
    def is_duplicate(self, client_order_id: str) -> bool:
        """Check if order ID has already been processed"""
        with self._lock:
            self._cleanup_expired()
            return client_order_id in self._executed_orders or client_order_id in self._pending_orders
    
    def mark_pending(self, client_order_id: str) -> bool:
        """Mark order as pending execution"""
        with self._lock:
            if self.is_duplicate(client_order_id):
                return False
            
            self._pending_orders.add(client_order_id)
            self._order_timestamps[client_order_id] = time.time()
            return True
    
    def mark_executed(self, client_order_id: str):
        """Mark order as successfully executed"""
        with self._lock:
            self._pending_orders.discard(client_order_id)
            self._executed_orders.add(client_order_id)
            self._order_timestamps[client_order_id] = time.time()
    
    def mark_failed(self, client_order_id: str):
        """Mark order as failed (remove from pending)"""
        with self._lock:
            self._pending_orders.discard(client_order_id)
            # Remove from timestamps to allow retry
            self._order_timestamps.pop(client_order_id, None)
    
    def _cleanup_expired(self):
        """Remove expired order IDs"""
        current_time = time.time()
        expired_ids = [
            order_id for order_id, timestamp in self._order_timestamps.items()
            if current_time - timestamp > self._ttl_seconds
        ]
        
        for order_id in expired_ids:
            self._executed_orders.discard(order_id)
            self._pending_orders.discard(order_id)
            del self._order_timestamps[order_id]


class ExecutionPolicy:
    """
    Hard execution discipline system with mandatory gates
    ALL orders must pass through decide() method
    """
    
    def __init__(self, gates: Optional[ExecutionGates] = None):
        self.gates = gates or ExecutionGates()
        self.idempotency = IdempotencyTracker()
        self._execution_count = 0
        self._rejection_count = 0
        self._lock = threading.Lock()
    
    def decide(
        self, 
        order_request: OrderRequest, 
        market_conditions: MarketConditions
    ) -> ExecutionResult:
        """
        MANDATORY gate for ALL order execution
        
        Args:
            order_request: Order to evaluate
            market_conditions: Current market state
            
        Returns:
            ExecutionResult with decision and reasoning
        """
        
        logger.info(f"Evaluating order: {order_request.client_order_id} for {order_request.symbol}")
        
        # Gate 1: Idempotency check (CRITICAL)
        client_order_id = order_request.client_order_id or "unknown"
        if self.idempotency.is_duplicate(client_order_id):
            return ExecutionResult(
                decision=ExecutionDecision.REJECT,
                reason=f"Duplicate order ID: {client_order_id}",
                gate_results={"idempotency": False}
            )
        
        gate_results = {"idempotency": True}
        
        # Gate 2: Spread check
        spread_ok = market_conditions.spread_bps <= self.gates.max_spread_bps
        gate_results["spread"] = spread_ok
        
        if not spread_ok:
            return ExecutionResult(
                decision=ExecutionDecision.REJECT,
                reason=f"Spread too wide: {market_conditions.spread_bps:.1f} > {self.gates.max_spread_bps} bps",
                gate_results=gate_results
            )
        
        # Gate 3: Depth check
        required_depth = self.gates.min_depth_usd
        if order_request.side == OrderSide.BUY:
            available_depth = market_conditions.ask_depth_usd
        else:
            available_depth = market_conditions.bid_depth_usd
        
        depth_ok = available_depth >= required_depth
        gate_results["depth"] = depth_ok
        
        if not depth_ok:
            return ExecutionResult(
                decision=ExecutionDecision.REJECT,
                reason=f"Insufficient depth: ${available_depth:,.0f} < ${required_depth:,.0f}",
                gate_results=gate_results
            )
        
        # Gate 4: Volume check (1-minute volume)
        volume_ok = market_conditions.volume_1m_usd >= self.gates.min_volume_1m_usd
        gate_results["volume"] = volume_ok
        
        if not volume_ok:
            return ExecutionResult(
                decision=ExecutionDecision.REJECT,
                reason=f"Low volume: ${market_conditions.volume_1m_usd:,.0f} < ${self.gates.min_volume_1m_usd:,.0f}",
                gate_results=gate_results
            )
        
        # Gate 5: Slippage budget check
        slippage_ok = order_request.max_slippage_bps <= self.gates.max_slippage_bps
        gate_results["slippage"] = slippage_ok
        
        if not slippage_ok:
            return ExecutionResult(
                decision=ExecutionDecision.REJECT,
                reason=f"Slippage budget too high: {order_request.max_slippage_bps} > {self.gates.max_slippage_bps} bps",
                gate_results=gate_results
            )
        
        # Gate 6: Time-in-Force validation
        if self.gates.require_post_only and order_request.time_in_force != TimeInForce.POST_ONLY:
            gate_results["tif"] = False
            return ExecutionResult(
                decision=ExecutionDecision.REJECT,
                reason=f"Post-only required, got: {order_request.time_in_force.value}",
                gate_results=gate_results
            )
        
        gate_results["tif"] = True
        
        # Gate 7: Price validation for limit orders
        if order_request.order_type == "limit" and order_request.limit_price:
            if order_request.side == OrderSide.BUY:
                # Buy limit should be at or below current ask
                price_valid = order_request.limit_price <= market_conditions.ask_price
            else:
                # Sell limit should be at or above current bid
                price_valid = order_request.limit_price >= market_conditions.bid_price
            
            gate_results["price"] = price_valid
            
            if not price_valid:
                return ExecutionResult(
                    decision=ExecutionDecision.REJECT,
                    reason=f"Invalid limit price: {order_request.limit_price} for {order_request.side.value}",
                    gate_results=gate_results
                )
        else:
            gate_results["price"] = True
        
        # All gates passed - mark as pending and approve
        if not self.idempotency.mark_pending(client_order_id):
            return ExecutionResult(
                decision=ExecutionDecision.REJECT,
                reason="Failed to acquire execution lock",
                gate_results=gate_results
            )
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(order_request, market_conditions)
        
        with self._lock:
            self._execution_count += 1
        
        logger.info(f"Order approved: {order_request.client_order_id}")
        
        return ExecutionResult(
            decision=ExecutionDecision.APPROVE,
            reason="All execution gates passed",
            approved_order=order_request,
            risk_score=risk_score,
            gate_results=gate_results
        )
    
    def _calculate_risk_score(self, order: OrderRequest, market: MarketConditions) -> float:
        """Calculate risk score for approved orders"""
        
        # Factors: spread, depth ratio, volume ratio, slippage
        spread_factor = min(market.spread_bps / self.gates.max_spread_bps, 1.0)
        
        depth_factor = 1.0 - min(
            (market.bid_depth_usd + market.ask_depth_usd) / (2 * self.gates.min_depth_usd), 
            1.0
        )
        
        volume_factor = 1.0 - min(market.volume_1m_usd / self.gates.min_volume_1m_usd, 1.0)
        
        slippage_factor = order.max_slippage_bps / self.gates.max_slippage_bps
        
        # Weighted risk score (0-1)
        risk_score = (
            0.3 * spread_factor +
            0.25 * depth_factor +
            0.25 * volume_factor +
            0.2 * slippage_factor
        )
        
        return min(risk_score, 1.0)
    
    def mark_order_executed(self, client_order_id: str):
        """Mark order as successfully executed"""
        self.idempotency.mark_executed(client_order_id)
        logger.info(f"Order executed: {client_order_id}")
    
    def mark_order_failed(self, client_order_id: str, reason: str):
        """Mark order as failed"""
        self.idempotency.mark_failed(client_order_id)
        with self._lock:
            self._rejection_count += 1
        logger.warning(f"Order failed: {client_order_id} - {reason}")
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get execution statistics"""
        with self._lock:
            return {
                "total_evaluations": self._execution_count + self._rejection_count,
                "approvals": self._execution_count,
                "rejections": self._rejection_count,
                "approval_rate": self._execution_count / max(self._execution_count + self._rejection_count, 1)
            }


class OrderExecutor:
    """
    Order executor with mandatory execution policy
    Demonstrates proper usage of ExecutionPolicy.decide()
    """
    
    def __init__(self, execution_policy: ExecutionPolicy):
        self.policy = execution_policy
        self.executed_orders: List[OrderRequest] = []
        self._lock = threading.Lock()
    
    def execute_order(
        self, 
        order_request: OrderRequest, 
        market_conditions: MarketConditions
    ) -> Tuple[bool, str]:
        """
        Execute order through mandatory policy gates
        
        Returns:
            (success, message) tuple
        """
        
        # MANDATORY RISK ENFORCEMENT: All orders must pass CentralRiskGuard first
        try:
            from ..core.mandatory_risk_enforcement import enforce_order_risk_check
            
            risk_check_result = enforce_order_risk_check(
                order_size=order_request.size,
                symbol=order_request.symbol,
                side=order_request.side.value,
                strategy_id=order_request.strategy_id or "execution_discipline"
            )
            
            if not risk_check_result["approved"]:
                return False, f"Risk Guard rejection: {risk_check_result['reason']}"
            
            # Use approved size from risk check
            if risk_check_result["approved_size"] != order_request.size:
                order_request.size = risk_check_result["approved_size"]
                
        except Exception as e:
            return False, f"Risk enforcement error: {str(e)}"
        
        # MANDATORY: All orders must go through policy.decide()
        result = self.policy.decide(order_request, market_conditions)
        client_order_id = order_request.client_order_id or "unknown"
        
        if result.decision != ExecutionDecision.APPROVE:
            return False, f"Order rejected: {result.reason}"
        
        try:
            # Simulate order execution (replace with actual exchange API)
            approved_order = result.approved_order or order_request
            success = self._send_to_exchange(approved_order)
            
            if success:
                self.policy.mark_order_executed(client_order_id)
                with self._lock:
                    self.executed_orders.append(order_request)
                return True, f"Order executed successfully: {client_order_id}"
            else:
                self.policy.mark_order_failed(client_order_id, "Exchange rejected")
                return False, "Exchange execution failed"
                
        except Exception as e:
            self.policy.mark_order_failed(client_order_id, str(e))
            return False, f"Execution error: {str(e)}"
    
    def _send_to_exchange(self, order: OrderRequest) -> bool:
        """
        Simulate sending order to exchange
        Replace with actual exchange API integration
        """
        # Simulate network delay and potential failure
        import random
        time.sleep(0.1)  # Simulate network latency
        
        # 95% success rate for simulation
        return random.random() > 0.05


# Global execution policy instance
_execution_policy: Optional[ExecutionPolicy] = None
_policy_lock = threading.Lock()


def get_execution_policy() -> ExecutionPolicy:
    """Get global execution policy instance"""
    global _execution_policy
    
    if _execution_policy is None:
        with _policy_lock:
            if _execution_policy is None:
                _execution_policy = ExecutionPolicy()
    
    return _execution_policy


def reset_execution_policy():
    """Reset policy for testing"""
    global _execution_policy
    with _policy_lock:
        _execution_policy = None


if __name__ == "__main__":
    # Example usage demonstrating execution discipline
    
    policy = ExecutionPolicy()
    executor = OrderExecutor(policy)
    
    # Example market conditions
    market = MarketConditions(
        spread_bps=15.0,
        bid_depth_usd=50000.0,
        ask_depth_usd=45000.0,
        volume_1m_usd=200000.0,
        last_price=50000.0,
        bid_price=49992.5,
        ask_price=50007.5,
        timestamp=time.time()
    )
    
    # Example order request
    order = OrderRequest(
        symbol="BTC/USD",
        side=OrderSide.BUY,
        size=0.1,
        limit_price=50000.0,
        max_slippage_bps=10.0,
        strategy_id="momentum_v1"
    )
    
    print(f"Order ID: {order.client_order_id}")
    
    # Execute through mandatory gates
    success, message = executor.execute_order(order, market)
    print(f"Execution result: {success} - {message}")
    
    # Test duplicate prevention
    duplicate_order = OrderRequest(
        symbol="BTC/USD",
        side=OrderSide.BUY,
        size=0.1,
        limit_price=50000.0,
        max_slippage_bps=10.0,
        strategy_id="momentum_v1",
        client_order_id=order.client_order_id  # Same ID
    )
    
    success2, message2 = executor.execute_order(duplicate_order, market)
    print(f"Duplicate test: {success2} - {message2}")
    
    # Print stats
    stats = policy.get_stats()
    print(f"Execution stats: {stats}")
