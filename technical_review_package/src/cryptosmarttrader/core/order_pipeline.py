"""
Centralized Order Pipeline - HARD WIRE-UP
All orders MUST pass through ExecutionPolicy gates - NO BYPASS POSSIBLE
"""

import time
import hashlib
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from ..execution.execution_policy import ExecutionPolicy, OrderRequest, OrderResult, OrderStatus
from ..risk.risk_guard import RiskGuard
from ..core.structured_logger import get_logger


@dataclass
class OrderDecision:
    """Order pipeline decision result."""
    
    approved: bool
    order_request: Optional[OrderRequest] = None
    rejection_reasons: List[str] = field(default_factory=list)
    risk_level: str = "UNKNOWN"
    execution_policy_checks: Dict[str, bool] = field(default_factory=dict)
    client_order_id: Optional[str] = None


class CentralizedOrderPipeline:
    """
    ENTERPRISE ORDER PIPELINE - MANDATORY EXECUTION POLICY ENFORCEMENT
    
    Zero bypass policy - every order passes through:
    1. RiskGuard verification
    2. ExecutionPolicy gates (spread/depth/volume/slippage)
    3. Client Order ID (COID) generation with SHA256 idempotency
    4. Slippage budget enforcement
    """
    
    def __init__(self, execution_policy: ExecutionPolicy, risk_guard: RiskGuard):
        """Initialize with mandatory components."""
        self.execution_policy = execution_policy
        self.risk_guard = risk_guard
        self.logger = get_logger("order_pipeline")
        
        # Track order history for idempotency
        self.order_history: Dict[str, datetime] = {}
        self.deduplication_window_minutes = 60
        
        self.logger.info("Centralized Order Pipeline initialized with HARD enforcement")
    
    def generate_client_order_id(self, order_request: OrderRequest) -> str:
        """Generate idempotent Client Order ID using SHA256."""
        # Create deterministic hash from order characteristics
        order_signature = (
            f"{order_request.symbol}|"
            f"{order_request.side}|"
            f"{order_request.quantity}|"
            f"{order_request.order_type.value}|"
            f"{order_request.price or 'MARKET'}|"
            f"{int(time.time() // 60)}"  # 1-minute window for deduplication
        )
        
        # Generate SHA256 hash
        hash_object = hashlib.sha256(order_signature.encode())
        client_order_id = f"CST_{hash_object.hexdigest()[:16]}"
        
        return client_order_id
    
    def is_duplicate_order(self, client_order_id: str) -> bool:
        """Check for duplicate orders within deduplication window."""
        if client_order_id in self.order_history:
            order_time = self.order_history[client_order_id]
            age_minutes = (datetime.now() - order_time).total_seconds() / 60
            
            if age_minutes < self.deduplication_window_minutes:
                return True
                
        return False
    
    async def decide_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float, 
        order_type: str = "MARKET",
        price: Optional[float] = None,
        confidence_score: Optional[float] = None,
        strategy_id: Optional[str] = None
    ) -> OrderDecision:
        """
        MANDATORY ORDER DECISION PIPELINE
        Every trade decision MUST pass through this method - NO EXCEPTIONS
        """
        
        decision = OrderDecision(approved=False)
        
        try:
            # Step 1: Generate idempotent Client Order ID
            order_request = OrderRequest(
                client_order_id="",  # Will be generated
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                price=price,
                confidence_score=confidence_score,
                strategy_id=strategy_id
            )
            
            client_order_id = self.generate_client_order_id(order_request)
            order_request.client_order_id = client_order_id
            decision.client_order_id = client_order_id
            
            # Step 2: Deduplication check
            if self.is_duplicate_order(client_order_id):
                decision.rejection_reasons.append(
                    f"Duplicate order detected within {self.deduplication_window_minutes}min window"
                )
                self.logger.warning("Order rejected - duplicate COID", 
                                  client_order_id=client_order_id)
                return decision
            
            # Step 3: MANDATORY RiskGuard check
            portfolio_value = 1000000.0  # TODO: Get from portfolio manager - using fixed value for demo
            risk_check = self.risk_guard.run_risk_check(portfolio_value)
            decision.risk_level = risk_check.get("risk_level", "UNKNOWN")
            
            if risk_check.get("kill_switch_active", False):
                decision.rejection_reasons.append("Kill switch active - all trading halted")
                self.logger.error("Order rejected - kill switch active", 
                                client_order_id=client_order_id)
                return decision
            
            if risk_check.get("trading_mode") == "SHUTDOWN":
                decision.rejection_reasons.append("Trading mode: SHUTDOWN")
                self.logger.error("Order rejected - shutdown mode", 
                                client_order_id=client_order_id)
                return decision
            
            # Step 4: MANDATORY ExecutionPolicy gates
            policy_checks = {}
            
            # Tradability gate
            tradable, tradability_issues = self.execution_policy.assess_tradability(symbol)
            policy_checks["tradability"] = tradable
            if not tradable:
                decision.rejection_reasons.extend(tradability_issues)
            
            # Order validation
            valid_order, validation_issues = self.execution_policy.validate_order_request(order_request)
            policy_checks["validation"] = valid_order
            if not valid_order:
                decision.rejection_reasons.extend(validation_issues)
            
            # Slippage budget check
            estimated_slippage = self.execution_policy.estimate_slippage(order_request)
            max_slippage = self.execution_policy.slippage_budget.max_slippage_percent
            policy_checks["slippage_budget"] = estimated_slippage <= max_slippage
            
            if estimated_slippage > max_slippage:
                decision.rejection_reasons.append(
                    f"Estimated slippage {estimated_slippage:.3f}% exceeds budget {max_slippage:.3f}%"
                )
            
            decision.execution_policy_checks = policy_checks
            
            # Step 5: Final approval decision
            if not decision.rejection_reasons:
                decision.approved = True
                decision.order_request = order_request
                
                # Record order for deduplication
                self.order_history[client_order_id] = datetime.now()
                
                self.logger.info("Order approved through pipeline", 
                               client_order_id=client_order_id,
                               symbol=symbol,
                               side=side,
                               quantity=quantity,
                               estimated_slippage=estimated_slippage)
            else:
                self.logger.warning("Order rejected by pipeline", 
                                  client_order_id=client_order_id,
                                  reasons=decision.rejection_reasons)
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Order pipeline error: {e}", 
                            client_order_id=decision.client_order_id)
            decision.rejection_reasons.append(f"Pipeline error: {str(e)}")
            return decision
    
    async def execute_approved_order(self, decision: OrderDecision) -> OrderResult:
        """Execute pre-approved order through ExecutionPolicy."""
        if not decision.approved or not decision.order_request:
            raise ValueError("Cannot execute unapproved order")
        
        try:
            # Execute through ExecutionPolicy (mock implementation for now)
            from ..execution.execution_policy import OrderResult, OrderStatus
            result = OrderResult(
                client_order_id=decision.order_request.client_order_id,
                symbol=decision.order_request.symbol,
                status=OrderStatus.FILLED,
                filled_quantity=decision.order_request.quantity,
                filled_price=decision.order_request.price or 45000.0,
                slippage_percent=0.1,
                fees=decision.order_request.quantity * 0.001,
                timestamp=datetime.now()
            )
            
            self.logger.info("Order executed successfully", 
                           client_order_id=decision.client_order_id,
                           status=result.status.value,
                           filled_quantity=result.filled_quantity,
                           slippage=result.slippage_percent)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Order execution failed: {e}", 
                            client_order_id=decision.client_order_id)
            raise
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        return {
            "orders_processed": len(self.order_history),
            "deduplication_window_minutes": self.deduplication_window_minutes,
            "active_dedup_entries": len([
                coid for coid, timestamp in self.order_history.items()
                if (datetime.now() - timestamp).total_seconds() < self.deduplication_window_minutes * 60
            ]),
            "pipeline_status": "OPERATIONAL"
        }


def create_order_pipeline(execution_policy: ExecutionPolicy, risk_guard: RiskGuard) -> CentralizedOrderPipeline:
    """Factory function to create hardwired order pipeline."""
    return CentralizedOrderPipeline(execution_policy, risk_guard)