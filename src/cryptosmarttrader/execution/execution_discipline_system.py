#!/usr/bin/env python3
"""
Enterprise Execution Discipline System
Hard execution policy enforcement voor ALLE orders
"""

import time
import hashlib
import threading
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class TimeInForce(Enum):
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill
    POST_ONLY = "POST_ONLY"  # Post Only (maker only)


class ExecutionDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    DEFER = "defer"
    REDUCE_SIZE = "reduce_size"


@dataclass
class OrderRequest:
    """Enterprise order request structure"""
    symbol: str
    side: OrderSide
    size: float
    limit_price: Optional[float] = None
    market_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.POST_ONLY
    max_slippage_bps: float = 20.0
    strategy_id: str = "default"
    client_order_id: Optional[str] = None
    
    def __post_init__(self):
        if self.client_order_id is None:
            self.client_order_id = self._generate_client_order_id()
    
    def _generate_client_order_id(self) -> str:
        """Generate deterministic client order ID voor idempotency"""
        # Create deterministic ID based on order parameters + minute timestamp
        minute_timestamp = int(time.time() // 60) * 60
        
        id_components = [
            self.symbol,
            self.side.value,
            str(self.size),
            str(self.limit_price or self.market_price or 0),
            self.strategy_id,
            str(minute_timestamp)
        ]
        
        id_string = "|".join(id_components)
        return hashlib.sha256(id_string.encode()).hexdigest()[:16]


@dataclass
class MarketConditions:
    """Real-time market conditions voor execution validation"""
    spread_bps: float
    bid_depth_usd: float
    ask_depth_usd: float
    volume_1m_usd: float
    last_price: float
    bid_price: float
    ask_price: float
    timestamp: float
    
    def is_stale(self, max_age_seconds: float = 30.0) -> bool:
        """Check if market data is stale"""
        return time.time() - self.timestamp > max_age_seconds


class ExecutionDisciplineSystem:
    """Enterprise execution discipline system met mandatory gates"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = self._get_default_config()
        if config:
            self.config.update(config)
        
        # Decision tracking
        self.decision_history: List[Dict[str, Any]] = []
        self.processed_orders: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.evaluation_count = 0
        self.total_evaluation_time = 0.0
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("ExecutionDisciplineSystem initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default execution policy configuration"""
        return {
            'max_spread_bps': 100.0,          # Reject spreads wider than 1%
            'min_depth_ratio': 3.0,           # Require 3x order size in depth
            'min_volume_ratio': 10.0,         # Require 10x order size in volume
            'max_slippage_bps': 50.0,         # Maximum allowed slippage
            'max_price_deviation': 0.05,      # 5% max price deviation from market
            'max_market_data_age_seconds': 30.0,  # Max age for market data
            'enable_idempotency_check': True,
            'enable_spread_gate': True,
            'enable_depth_gate': True,
            'enable_volume_gate': True,
            'enable_slippage_gate': True,
            'enable_price_validation': True
        }
    
    def evaluate_order(
        self, 
        order: OrderRequest, 
        market_conditions: Optional[MarketConditions] = None
    ) -> Tuple[ExecutionDecision, str, Optional[Dict[str, Any]]]:
        """
        MANDATORY: Evaluate order through all execution gates
        
        Returns:
            decision: ExecutionDecision enum
            reason: Human-readable reason for decision
            details: Additional evaluation details
        """
        start_time = time.time()
        
        with self._lock:
            self.evaluation_count += 1
            
            try:
                # Execute all mandatory gates
                decision, reason, details = self._execute_mandatory_gates(order, market_conditions)
                
                # Record decision
                evaluation_time_ms = (time.time() - start_time) * 1000
                self._record_decision(order, decision, reason, details, evaluation_time_ms)
                
                # Update performance metrics
                self.total_evaluation_time += evaluation_time_ms / 1000
                
                if details is None:
                    details = {}
                details['evaluation_time_ms'] = evaluation_time_ms
                
                return decision, reason, details
                
            except Exception as e:
                error_reason = f"EVALUATION_ERROR: {str(e)}"
                logger.error(f"Order evaluation failed: {e}")
                return ExecutionDecision.REJECT, error_reason, {'error': str(e)}
    
    def _execute_mandatory_gates(
        self, 
        order: OrderRequest, 
        market_conditions: Optional[MarketConditions]
    ) -> Tuple[ExecutionDecision, str, Optional[Dict[str, Any]]]:
        """Execute all mandatory execution gates"""
        
        # Gate 1: Idempotency protection
        if self.config['enable_idempotency_check']:
            if order.client_order_id in self.processed_orders:
                return ExecutionDecision.REJECT, "DUPLICATE_ORDER: Order already processed", None
        
        # Gate 2: Market data validation
        if market_conditions is None:
            return ExecutionDecision.DEFER, "NO_MARKET_DATA: Market conditions required", None
        
        if market_conditions.is_stale(self.config['max_market_data_age_seconds']):
            return ExecutionDecision.REJECT, "STALE_MARKET_DATA: Market data too old", None
        
        # Gate 3: Spread gate
        if self.config['enable_spread_gate']:
            if market_conditions.spread_bps > self.config['max_spread_bps']:
                return ExecutionDecision.REJECT, f"SPREAD_TOO_WIDE: {market_conditions.spread_bps:.1f} bps > {self.config['max_spread_bps']:.1f} bps", None
        
        # Gate 4: Depth gate
        if self.config['enable_depth_gate']:
            order_value_usd = order.size * (order.limit_price or market_conditions.last_price)
            required_depth = order_value_usd * self.config['min_depth_ratio']
            
            available_depth = market_conditions.bid_depth_usd if order.side == OrderSide.SELL else market_conditions.ask_depth_usd
            
            if available_depth < required_depth:
                return ExecutionDecision.REJECT, f"INSUFFICIENT_DEPTH: {available_depth:.0f} < {required_depth:.0f} USD required", None
        
        # Gate 5: Volume gate
        if self.config['enable_volume_gate']:
            order_value_usd = order.size * (order.limit_price or market_conditions.last_price)
            required_volume = order_value_usd * self.config['min_volume_ratio']
            
            if market_conditions.volume_1m_usd < required_volume:
                return ExecutionDecision.REJECT, f"INSUFFICIENT_VOLUME: {market_conditions.volume_1m_usd:.0f} < {required_volume:.0f} USD required", None
        
        # Gate 6: Slippage budget enforcement
        if self.config['enable_slippage_gate']:
            estimated_slippage = self._estimate_slippage(order, market_conditions)
            
            if estimated_slippage > order.max_slippage_bps:
                return ExecutionDecision.REJECT, f"SLIPPAGE_BUDGET_EXCEEDED: {estimated_slippage:.1f} bps > {order.max_slippage_bps:.1f} bps", None
        
        # Gate 7: Time-in-Force validation
        if order.time_in_force == TimeInForce.POST_ONLY:
            if self._would_cross_spread(order, market_conditions):
                return ExecutionDecision.REJECT, "POST_ONLY_WOULD_CROSS: Order would take liquidity", None
        
        # Gate 8: Price validation
        if self.config['enable_price_validation'] and order.limit_price:
            if self._is_price_too_aggressive(order, market_conditions):
                return ExecutionDecision.REJECT, f"PRICE_TOO_AGGRESSIVE: Price deviation > {self.config['max_price_deviation']:.1%}", None
        
        # All gates passed
        estimated_slippage = self._estimate_slippage(order, market_conditions)
        
        details = {
            'estimated_slippage_bps': estimated_slippage,
            'market_spread_bps': market_conditions.spread_bps,
            'available_depth_usd': market_conditions.bid_depth_usd if order.side == OrderSide.SELL else market_conditions.ask_depth_usd,
            'market_volume_1m_usd': market_conditions.volume_1m_usd
        }
        
        return ExecutionDecision.APPROVE, "APPROVED: All execution gates passed", details
    
    def _estimate_slippage(self, order: OrderRequest, market_conditions: MarketConditions) -> float:
        """Estimate execution slippage in basis points"""
        # Simple slippage estimation based on spread and market impact
        spread_component = market_conditions.spread_bps / 2  # Half spread
        
        # Market impact based on order size vs depth
        order_value = order.size * (order.limit_price or market_conditions.last_price)
        available_depth = market_conditions.bid_depth_usd if order.side == OrderSide.SELL else market_conditions.ask_depth_usd
        
        impact_ratio = order_value / max(available_depth, 1)
        impact_component = min(impact_ratio * 50, 100)  # Max 100 bps impact
        
        return spread_component + impact_component
    
    def _would_cross_spread(self, order: OrderRequest, market_conditions: MarketConditions) -> bool:
        """Check if POST_ONLY order would cross the spread"""
        if not order.limit_price:
            return True  # Market orders always cross
        
        if order.side == OrderSide.BUY:
            return order.limit_price >= market_conditions.ask_price
        else:
            return order.limit_price <= market_conditions.bid_price
    
    def _is_price_too_aggressive(self, order: OrderRequest, market_conditions: MarketConditions) -> bool:
        """Check if order price is too far from market"""
        if not order.limit_price:
            return False
        
        market_price = market_conditions.last_price
        price_deviation = abs(order.limit_price - market_price) / market_price
        
        return price_deviation > self.config['max_price_deviation']
    
    def _record_decision(
        self, 
        order: OrderRequest, 
        decision: ExecutionDecision, 
        reason: str, 
        details: Optional[Dict[str, Any]], 
        evaluation_time_ms: float
    ):
        """Record execution decision voor audit trail"""
        
        decision_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'client_order_id': order.client_order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'size': order.size,
            'decision': decision.value,
            'reason': reason,
            'evaluation_time_ms': evaluation_time_ms,
            'details': details or {}
        }
        
        self.decision_history.append(decision_record)
        
        # Mark order as processed voor idempotency
        if decision == ExecutionDecision.APPROVE:
            self.processed_orders[order.client_order_id] = decision_record
        
        # Keep history size manageable
        if len(self.decision_history) > 10000:
            self.decision_history = self.decision_history[-5000:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get execution system performance metrics"""
        with self._lock:
            avg_evaluation_time = self.total_evaluation_time / max(self.evaluation_count, 1)
            
            # Calculate approval/rejection rates
            recent_decisions = self.decision_history[-1000:]  # Last 1000 decisions
            total_recent = len(recent_decisions)
            
            if total_recent > 0:
                approved = sum(1 for d in recent_decisions if d['decision'] == 'approve')
                rejected = sum(1 for d in recent_decisions if d['decision'] == 'reject')
                
                approval_rate = approved / total_recent
                rejection_rate = rejected / total_recent
            else:
                approval_rate = rejection_rate = 0.0
            
            return {
                'total_evaluations': self.evaluation_count,
                'avg_evaluation_time_ms': avg_evaluation_time * 1000,
                'approval_rate': approval_rate,
                'rejection_rate': rejection_rate,
                'processed_orders_count': len(self.processed_orders),
                'decision_history_size': len(self.decision_history)
            }
    
    def get_recent_decisions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent execution decisions"""
        with self._lock:
            return self.decision_history[-limit:]
    
    def clear_processed_orders(self, older_than_hours: int = 24):
        """Clear old processed orders voor memory management"""
        cutoff_time = datetime.utcnow() - timedelta(hours=older_than_hours)
        
        with self._lock:
            # Remove orders older than cutoff
            to_remove = []
            for order_id, record in self.processed_orders.items():
                record_time = datetime.fromisoformat(record['timestamp'])
                if record_time < cutoff_time:
                    to_remove.append(order_id)
            
            for order_id in to_remove:
                del self.processed_orders[order_id]
            
            logger.info(f"Cleared {len(to_remove)} old processed orders")


# Export main class
__all__ = ['ExecutionDisciplineSystem', 'OrderRequest', 'OrderSide', 'TimeInForce', 'ExecutionDecision', 'MarketConditions']