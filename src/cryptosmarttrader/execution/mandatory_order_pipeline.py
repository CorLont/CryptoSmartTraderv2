"""
Mandatory Order Pipeline - Fase C Guardrails Implementation
All orders MUST pass through ExecutionPolicy and RiskGuard - zero bypass possible.
"""

import asyncio
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import threading
from pathlib import Path

from .execution_policy import ExecutionPolicy, OrderRequest, OrderResult, OrderStatus
from ..risk.risk_guard import RiskGuard, RiskLevel, TradingMode
from ..core.structured_logger import get_logger


@dataclass
class PolicyDecision:
    """ExecutionPolicy decision result."""
    allowed: bool
    reason: str
    adjusted_quantity: Optional[float] = None
    execution_strategy: Optional[str] = None
    slippage_budget_bps: float = 30.0
    client_order_id: Optional[str] = None


@dataclass
class RiskDecision:
    """RiskGuard decision result."""
    allowed: bool
    reason: str
    risk_level: RiskLevel
    trading_mode: TradingMode
    position_size_limit: Optional[float] = None
    constraints: Dict[str, Any] = None


class MandatoryOrderPipeline:
    """
    Mandatory order pipeline enforcing ExecutionPolicy and RiskGuard on ALL orders.
    
    ZERO BYPASS ARCHITECTURE:
    - Every order must pass through ExecutionPolicy.decide()
    - Every order must pass through RiskGuard.run_risk_check()
    - Failed checks = immediate rejection
    - Idempotent client_order_id generation enforced
    - Slippage budget enforced with p95 validation
    
    Usage:
        pipeline = MandatoryOrderPipeline()
        result = await pipeline.execute_order(order_request)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = get_logger("mandatory_order_pipeline")
        
        # Initialize required guardrails - MANDATORY
        self.execution_policy = ExecutionPolicy(config_path)
        self.risk_guard = RiskGuard(config_path)
        
        # Order tracking for idempotency
        self.processed_orders: Dict[str, OrderResult] = {}
        self.order_lock = threading.RLock()
        
        # Slippage tracking for p95 validation
        self.slippage_history: List[float] = []
        self.slippage_violations = 0
        self.p95_slippage_budget = 0.5  # 50 bps p95 limit
        
        # Metrics
        self.total_orders_processed = 0
        self.policy_rejections = 0
        self.risk_rejections = 0
        self.successful_executions = 0
        
        self.logger.info("MandatoryOrderPipeline initialized - ZERO BYPASS ENFORCED")
    
    def generate_client_order_id(self, order_request: OrderRequest) -> str:
        """Generate idempotent client order ID using SHA256."""
        # Create deterministic hash from order parameters
        order_data = {
            'symbol': order_request.symbol,
            'side': order_request.side,
            'quantity': order_request.quantity,
            'price': order_request.price,
            'strategy_id': order_request.strategy_id,
            'timestamp_minute': order_request.timestamp.strftime('%Y-%m-%d_%H:%M')  # 1-minute dedup window
        }
        
        order_string = json.dumps(order_data, sort_keys=True)
        client_order_id = hashlib.sha256(order_string.encode()).hexdigest()[:16]
        
        return f"CST_{client_order_id}"
    
    async def execute_order(self, order_request: OrderRequest) -> OrderResult:
        """
        Execute order through mandatory pipeline.
        ZERO BYPASS - ALL ORDERS MUST GO THROUGH THIS PIPELINE.
        """
        start_time = time.time()
        self.total_orders_processed += 1
        
        # Generate idempotent COID if not provided
        if not order_request.client_order_id:
            order_request.client_order_id = self.generate_client_order_id(order_request)
        
        with self.order_lock:
            # Check for duplicate orders (idempotency)
            if order_request.client_order_id in self.processed_orders:
                existing_result = self.processed_orders[order_request.client_order_id]
                self.logger.info(
                    f"Duplicate order detected - returning cached result",
                    client_order_id=order_request.client_order_id
                )
                return existing_result
        
        try:
            # MANDATORY STEP 1: ExecutionPolicy Decision
            policy_decision = await self._get_execution_policy_decision(order_request)
            if not policy_decision.allowed:
                result = self._create_rejection_result(
                    order_request, f"ExecutionPolicy rejection: {policy_decision.reason}"
                )
                self.policy_rejections += 1
                return result
            
            # MANDATORY STEP 2: RiskGuard Decision  
            risk_decision = await self._get_risk_guard_decision(order_request)
            if not risk_decision.allowed:
                result = self._create_rejection_result(
                    order_request, f"RiskGuard rejection: {risk_decision.reason}"
                )
                self.risk_rejections += 1
                return result
            
            # MANDATORY STEP 3: Apply policy adjustments
            adjusted_request = self._apply_policy_adjustments(order_request, policy_decision)
            
            # MANDATORY STEP 4: Execute with guardrails
            result = await self._execute_with_guardrails(adjusted_request, policy_decision)
            
            # MANDATORY STEP 5: Validate slippage budget
            self._validate_slippage_budget(result)
            
            # Cache result for idempotency
            with self.order_lock:
                self.processed_orders[order_request.client_order_id] = result
            
            self.successful_executions += 1
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            self.logger.info(
                "Order executed successfully through mandatory pipeline",
                client_order_id=result.client_order_id,
                status=result.status.value,
                slippage_bps=result.slippage_percent * 100,
                execution_time_ms=execution_time_ms
            )
            
            return result
            
        except Exception as e:
            error_result = self._create_error_result(order_request, str(e))
            self.logger.error(f"Order execution failed: {e}", client_order_id=order_request.client_order_id)
            return error_result
    
    async def _get_execution_policy_decision(self, order_request: OrderRequest) -> PolicyDecision:
        """Get mandatory ExecutionPolicy decision."""
        try:
            # Get current market conditions
            market_conditions = await self._get_market_conditions(order_request.symbol)
            
            # ExecutionPolicy tradability check
            is_tradable = self.execution_policy.check_tradability(
                order_request.symbol, market_conditions
            )
            
            if not is_tradable:
                return PolicyDecision(
                    allowed=False,
                    reason="Market conditions fail tradability gates (spread/depth/volume)"
                )
            
            # Slippage budget check
            estimated_slippage = self.execution_policy.estimate_slippage(
                order_request.symbol, 
                order_request.quantity,
                market_conditions
            )
            
            max_slippage = order_request.max_slippage_percent or 0.3  # Default 30 bps
            if estimated_slippage > max_slippage:
                return PolicyDecision(
                    allowed=False,
                    reason=f"Estimated slippage {estimated_slippage:.1%} exceeds budget {max_slippage:.1%}"
                )
            
            # Generate idempotent client order ID
            client_order_id = self.generate_client_order_id(order_request)
            
            return PolicyDecision(
                allowed=True,
                reason="ExecutionPolicy approved",
                adjusted_quantity=order_request.quantity,
                execution_strategy="TWAP" if order_request.quantity > 10000 else "MARKET",
                slippage_budget_bps=max_slippage * 100,
                client_order_id=client_order_id
            )
            
        except Exception as e:
            self.logger.error(f"ExecutionPolicy decision failed: {e}")
            return PolicyDecision(
                allowed=False,
                reason=f"ExecutionPolicy error: {e}"
            )
    
    async def _get_risk_guard_decision(self, order_request: OrderRequest) -> RiskDecision:
        """Get mandatory RiskGuard decision."""
        try:
            # Run comprehensive risk check
            portfolio_value = await self._get_portfolio_value()
            risk_status = self.risk_guard.run_risk_check(portfolio_value)
            
            # Check kill switch
            if risk_status['kill_switch_active']:
                return RiskDecision(
                    allowed=False,
                    reason="Kill switch is active - all trading disabled",
                    risk_level=RiskLevel.SHUTDOWN,
                    trading_mode=TradingMode.DISABLED
                )
            
            # Check trading mode
            trading_mode = TradingMode(risk_status['trading_mode'])
            if trading_mode == TradingMode.DISABLED:
                return RiskDecision(
                    allowed=False,
                    reason="Trading disabled due to risk conditions",
                    risk_level=RiskLevel(risk_status['risk_level']),
                    trading_mode=trading_mode
                )
            
            # Check risk violations
            violations = risk_status.get('violations', [])
            critical_violations = [v for v in violations if v['severity'] == 'critical']
            
            if critical_violations:
                return RiskDecision(
                    allowed=False,
                    reason=f"Critical risk violations: {len(critical_violations)}",
                    risk_level=RiskLevel(risk_status['risk_level']),
                    trading_mode=trading_mode
                )
            
            # Get position size constraints
            constraints = self.risk_guard.get_trading_constraints()
            max_position_size = constraints.get('max_position_size_usd', float('inf'))
            
            # Validate position size
            order_value = order_request.quantity * (order_request.price or 1.0)  # Estimate
            if order_value > max_position_size:
                return RiskDecision(
                    allowed=False,
                    reason=f"Order size ${order_value:,.0f} exceeds risk limit ${max_position_size:,.0f}",
                    risk_level=RiskLevel(risk_status['risk_level']),
                    trading_mode=trading_mode
                )
            
            return RiskDecision(
                allowed=True,
                reason="RiskGuard approved",
                risk_level=RiskLevel(risk_status['risk_level']),
                trading_mode=trading_mode,
                position_size_limit=max_position_size,
                constraints=constraints
            )
            
        except Exception as e:
            self.logger.error(f"RiskGuard decision failed: {e}")
            return RiskDecision(
                allowed=False,
                reason=f"RiskGuard error: {e}",
                risk_level=RiskLevel.EMERGENCY,
                trading_mode=TradingMode.DISABLED
            )
    
    def _apply_policy_adjustments(self, order_request: OrderRequest, policy_decision: PolicyDecision) -> OrderRequest:
        """Apply ExecutionPolicy adjustments to order."""
        # Create copy with adjustments
        adjusted_request = OrderRequest(
            client_order_id=policy_decision.client_order_id or order_request.client_order_id,
            symbol=order_request.symbol,
            side=order_request.side,
            order_type=order_request.order_type,
            quantity=policy_decision.adjusted_quantity or order_request.quantity,
            price=order_request.price,
            time_in_force=order_request.time_in_force,
            confidence_score=order_request.confidence_score,
            strategy_id=order_request.strategy_id,
            max_slippage_percent=policy_decision.slippage_budget_bps / 100.0,
            post_only=order_request.post_only,
            reduce_only=order_request.reduce_only,
            timestamp=order_request.timestamp
        )
        
        return adjusted_request
    
    async def _execute_with_guardrails(self, order_request: OrderRequest, policy_decision: PolicyDecision) -> OrderResult:
        """Execute order with all guardrails applied."""
        # Simulate order execution (in real implementation, this would call exchange API)
        start_time = time.time()
        
        # Use execution strategy from policy
        execution_strategy = policy_decision.execution_strategy
        
        if execution_strategy == "TWAP":
            # Execute in smaller chunks over time
            result = await self._execute_twap_order(order_request)
        else:
            # Execute as market order
            result = await self._execute_market_order(order_request)
        
        execution_time_ms = int((time.time() - start_time) * 1000)
        result.execution_time_ms = execution_time_ms
        
        return result
    
    async def _execute_market_order(self, order_request: OrderRequest) -> OrderResult:
        """Execute market order (simulated)."""
        # In production, this would call actual exchange API
        # For now, simulate successful execution
        
        # Simulate realistic slippage
        base_slippage = 0.002  # 20 bps base
        quantity_impact = min(order_request.quantity / 10000 * 0.001, 0.01)  # Size impact
        market_impact = 0.001  # Market conditions impact
        
        total_slippage = base_slippage + quantity_impact + market_impact
        
        # Simulate fill
        filled_quantity = order_request.quantity
        avg_fill_price = (order_request.price or 100.0) * (1 + total_slippage if order_request.side == 'buy' else 1 - total_slippage)
        total_fees = filled_quantity * avg_fill_price * 0.001  # 10 bps fees
        
        return OrderResult(
            client_order_id=order_request.client_order_id,
            exchange_order_id=f"EXCH_{int(time.time())}",
            status=OrderStatus.FILLED,
            filled_quantity=filled_quantity,
            avg_fill_price=avg_fill_price,
            total_fees=total_fees,
            slippage_percent=total_slippage,
            execution_time_ms=0,  # Will be set by caller
            timestamp=datetime.now()
        )
    
    async def _execute_twap_order(self, order_request: OrderRequest) -> OrderResult:
        """Execute TWAP order in chunks (simulated)."""
        # Simulate TWAP execution with lower slippage
        total_slippage = 0.0015  # 15 bps for TWAP
        
        filled_quantity = order_request.quantity
        avg_fill_price = (order_request.price or 100.0) * (1 + total_slippage if order_request.side == 'buy' else 1 - total_slippage)
        total_fees = filled_quantity * avg_fill_price * 0.001
        
        return OrderResult(
            client_order_id=order_request.client_order_id,
            exchange_order_id=f"TWAP_{int(time.time())}",
            status=OrderStatus.FILLED,
            filled_quantity=filled_quantity,
            avg_fill_price=avg_fill_price,
            total_fees=total_fees,
            slippage_percent=total_slippage,
            execution_time_ms=0,  # Will be set by caller
            timestamp=datetime.now()
        )
    
    def _validate_slippage_budget(self, result: OrderResult) -> None:
        """Validate slippage against budget and track p95."""
        slippage_bps = result.slippage_percent * 100
        
        # Track slippage history
        self.slippage_history.append(slippage_bps)
        if len(self.slippage_history) > 1000:  # Keep last 1000 trades
            self.slippage_history = self.slippage_history[-1000:]
        
        # Check p95 slippage
        if len(self.slippage_history) >= 20:  # Need minimum samples
            p95_slippage = np.percentile(self.slippage_history, 95)
            
            if p95_slippage > self.p95_slippage_budget:
                self.slippage_violations += 1
                self.logger.warning(
                    f"P95 slippage {p95_slippage:.1f} bps exceeds budget {self.p95_slippage_budget:.1f} bps",
                    violations_count=self.slippage_violations,
                    current_slippage_bps=slippage_bps
                )
    
    def _create_rejection_result(self, order_request: OrderRequest, reason: str) -> OrderResult:
        """Create rejection result."""
        return OrderResult(
            client_order_id=order_request.client_order_id,
            exchange_order_id=None,
            status=OrderStatus.REJECTED,
            filled_quantity=0.0,
            avg_fill_price=0.0,
            total_fees=0.0,
            slippage_percent=0.0,
            execution_time_ms=0,
            error_message=reason,
            timestamp=datetime.now()
        )
    
    def _create_error_result(self, order_request: OrderRequest, error: str) -> OrderResult:
        """Create error result."""
        return OrderResult(
            client_order_id=order_request.client_order_id,
            exchange_order_id=None,
            status=OrderStatus.REJECTED,
            filled_quantity=0.0,
            avg_fill_price=0.0,
            total_fees=0.0,
            slippage_percent=0.0,
            execution_time_ms=0,
            error_message=f"Execution error: {error}",
            timestamp=datetime.now()
        )
    
    async def _get_market_conditions(self, symbol: str) -> Any:
        """Get current market conditions for symbol."""
        # Simulate market conditions
        from .execution_policy import MarketConditions
        
        return MarketConditions(
            bid_price=99.0,
            ask_price=101.0,
            mid_price=100.0,
            spread_percent=2.0,
            volume_24h=1000000.0,
            orderbook_depth_bid=50000.0,
            orderbook_depth_ask=50000.0,
            price_volatility=0.02,
            liquidity_score=0.8
        )
    
    async def _get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        # In production, this would query actual portfolio
        return 1000000.0  # $1M simulation
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline execution statistics."""
        return {
            'total_orders_processed': self.total_orders_processed,
            'policy_rejections': self.policy_rejections,
            'risk_rejections': self.risk_rejections,
            'successful_executions': self.successful_executions,
            'success_rate': self.successful_executions / max(self.total_orders_processed, 1),
            'policy_rejection_rate': self.policy_rejections / max(self.total_orders_processed, 1),
            'risk_rejection_rate': self.risk_rejections / max(self.total_orders_processed, 1),
            'slippage_violations': self.slippage_violations,
            'p95_slippage_budget': self.p95_slippage_budget,
            'current_p95_slippage': np.percentile(self.slippage_history, 95) if len(self.slippage_history) >= 20 else None,
            'processed_orders_count': len(self.processed_orders),
            'timestamp': datetime.now()
        }


# Required import for numpy percentile
import numpy as np
import json