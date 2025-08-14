"""
Advanced Execution Simulation System
Models realistic trading conditions: fees, partial fills, latency, queue effects
"""

import time
import json
import random
import math
from typing import Dict, List, Optional, Tuple, Deque
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    QUEUED = "queued"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class OrderType(Enum):
    """Order types for simulation"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class FillType(Enum):
    """Types of order fills"""
    MAKER = "maker"      # Provided liquidity
    TAKER = "taker"      # Removed liquidity
    AGGRESSIVE = "aggressive"  # Market order
    PASSIVE = "passive"   # Limit order filled


@dataclass
class MarketConditions:
    """Current market conditions for simulation"""
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    volume_1m: float
    volatility: float
    timestamp: float
    
    @property
    def spread_bps(self) -> float:
        """Spread in basis points"""
        if self.last_price > 0:
            return ((self.ask_price - self.bid_price) / self.last_price) * 10000
        return 0.0
    
    @property
    def mid_price(self) -> float:
        """Mid price"""
        return (self.bid_price + self.ask_price) / 2.0


@dataclass
class FeeStructure:
    """Trading fee structure"""
    maker_fee_bps: float = 5.0      # 5 bps maker fee
    taker_fee_bps: float = 10.0     # 10 bps taker fee
    market_fee_bps: float = 15.0    # 15 bps market order fee
    minimum_fee_usd: float = 0.01   # Minimum fee in USD
    
    def calculate_fee(self, fill_value: float, fill_type: FillType) -> float:
        """Calculate trading fee"""
        if fill_type == FillType.MAKER:
            fee_rate = self.maker_fee_bps / 10000
        elif fill_type == FillType.TAKER:
            fee_rate = self.taker_fee_bps / 10000
        elif fill_type == FillType.AGGRESSIVE:
            fee_rate = self.market_fee_bps / 10000
        else:  # PASSIVE
            fee_rate = self.maker_fee_bps / 10000
        
        fee = fill_value * fee_rate
        return max(fee, self.minimum_fee_usd)


@dataclass
class OrderFill:
    """Individual order fill"""
    fill_id: str
    order_id: str
    symbol: str
    side: str
    size: float
    price: float
    fill_type: FillType
    fee: float
    timestamp: float
    latency_ms: float
    
    @property
    def fill_value(self) -> float:
        """Total value of fill"""
        return self.size * self.price


@dataclass
class SimulatedOrder:
    """Simulated order with realistic execution"""
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    order_type: OrderType
    size: float
    limit_price: Optional[float]
    stop_price: Optional[float]
    
    # Execution tracking
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    remaining_size: float = 0.0
    average_fill_price: float = 0.0
    total_fees: float = 0.0
    fills: List[OrderFill] = field(default_factory=list)
    
    # Timing
    submit_time: float = field(default_factory=time.time)
    first_fill_time: Optional[float] = None
    complete_time: Optional[float] = None
    
    # Queue position and latency
    queue_position: int = 0
    submit_latency_ms: float = 0.0
    
    # Risk rejection tracking
    rejection_reason: Optional[str] = None
    
    def __post_init__(self):
        self.remaining_size = self.size
    
    @property
    def is_complete(self) -> bool:
        """Check if order is completely filled"""
        return self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]
    
    @property
    def fill_percentage(self) -> float:
        """Percentage of order filled"""
        if self.size > 0:
            return self.filled_size / self.size
        return 0.0


class LatencyModel:
    """Models realistic trading latency"""
    
    def __init__(self):
        # Base latencies in milliseconds
        self.order_submit_latency = (50, 200)    # 50-200ms submit latency
        self.market_data_latency = (5, 50)       # 5-50ms market data latency
        self.fill_notification_latency = (20, 100)  # 20-100ms fill notification
        
        # Network conditions
        self.network_stability = 0.95  # 95% stable network
        self.spike_multiplier = 5.0    # 5x latency during spikes
    
    def get_order_submit_latency(self) -> float:
        """Get realistic order submission latency"""
        base_latency = random.uniform(*self.order_submit_latency)
        
        # Add network instability
        if random.random() > self.network_stability:
            base_latency *= self.spike_multiplier
        
        return base_latency
    
    def get_market_data_latency(self) -> float:
        """Get market data latency"""
        return random.uniform(*self.market_data_latency)
    
    def get_fill_latency(self) -> float:
        """Get fill notification latency"""
        base_latency = random.uniform(*self.fill_notification_latency)
        
        # Add network instability
        if random.random() > self.network_stability:
            base_latency *= self.spike_multiplier
        
        return base_latency


class LiquidityModel:
    """Models market liquidity and queue effects"""
    
    def __init__(self):
        self.base_queue_size = 100     # Base number of orders in queue
        self.queue_volatility = 0.3    # Queue size volatility
        self.liquidity_factor = 1.0    # Overall liquidity multiplier
        
    def get_queue_position(self, market_conditions: MarketConditions, is_aggressive: bool) -> int:
        """Get position in order queue"""
        if is_aggressive:
            return 0  # Market orders go to front
        
        # Model queue based on market conditions
        base_queue = self.base_queue_size
        
        # Wider spreads = more orders in queue
        spread_factor = max(1.0, market_conditions.spread_bps / 10.0)
        
        # Higher volatility = more orders
        vol_factor = max(1.0, market_conditions.volatility * 100)
        
        # Lower volume = deeper queues
        volume_factor = max(0.5, min(2.0, market_conditions.volume_1m / 1000000))
        
        queue_size = base_queue * spread_factor * vol_factor / volume_factor
        
        # Add randomness
        queue_size *= random.uniform(1 - self.queue_volatility, 1 + self.queue_volatility)
        
        return max(1, int(queue_size))
    
    def get_partial_fill_probability(self, order: SimulatedOrder, market_conditions: MarketConditions) -> float:
        """Get probability of partial fill"""
        
        # Market orders rarely get partial fills
        if order.order_type == OrderType.MARKET:
            return 0.1
        
        # Limit orders have higher partial fill probability
        if order.order_type == OrderType.LIMIT:
            # Large orders more likely to be partially filled
            size_factor = min(1.0, order.remaining_size / 1000)  # Assume 1000 is "large"
            
            # Tight spreads = higher partial fill probability
            spread_factor = max(0.1, min(1.0, 20.0 / market_conditions.spread_bps))
            
            # Low liquidity = higher partial fill probability
            liquidity_factor = max(0.1, min(1.0, 100000 / market_conditions.volume_1m))
            
            return min(0.8, size_factor * spread_factor * liquidity_factor)
        
        return 0.3  # Default for other order types


class SlippageModel:
    """Models realistic price slippage"""
    
    def __init__(self):
        self.base_slippage_bps = 2.0   # Base slippage in bps
        self.impact_coefficient = 0.5  # Market impact coefficient
        
    def calculate_slippage(
        self, 
        order: SimulatedOrder, 
        market_conditions: MarketConditions,
        fill_size: float
    ) -> float:
        """Calculate realistic slippage for order fill"""
        
        if order.order_type == OrderType.MARKET:
            # Market orders have more slippage
            base_slippage = self.base_slippage_bps * 2.0
        else:
            # Limit orders have less slippage
            base_slippage = self.base_slippage_bps * 0.5
        
        # Market impact based on order size vs available liquidity
        if order.side == "buy":
            available_liquidity = market_conditions.ask_size
            reference_price = market_conditions.ask_price
        else:
            available_liquidity = market_conditions.bid_size
            reference_price = market_conditions.bid_price
        
        # Impact increases with order size relative to available liquidity
        if available_liquidity > 0:
            impact_ratio = fill_size / available_liquidity
            market_impact = impact_ratio * self.impact_coefficient * 100  # In bps
        else:
            market_impact = 10.0  # Default high impact if no liquidity data
        
        # Volatility increases slippage
        volatility_impact = market_conditions.volatility * 50  # In bps
        
        # Spread increases slippage
        spread_impact = market_conditions.spread_bps * 0.2
        
        total_slippage_bps = base_slippage + market_impact + volatility_impact + spread_impact
        
        # Convert to price impact
        slippage_factor = total_slippage_bps / 10000
        
        if order.side == "buy":
            return reference_price * slippage_factor  # Positive slippage for buys
        else:
            return -reference_price * slippage_factor  # Negative slippage for sells


class ExecutionSimulator:
    """
    Comprehensive execution simulator
    Models realistic trading conditions for backtest-live parity
    """
    
    def __init__(self, fee_structure: Optional[FeeStructure] = None):
        self.fee_structure = fee_structure or FeeStructure()
        self.latency_model = LatencyModel()
        self.liquidity_model = LiquidityModel()
        self.slippage_model = SlippageModel()
        
        # Order tracking
        self.active_orders: Dict[str, SimulatedOrder] = {}
        self.completed_orders: List[SimulatedOrder] = []
        self.order_fills: List[OrderFill] = []
        
        # Execution statistics
        self.total_orders = 0
        self.total_fills = 0
        self.total_fees = 0.0
        self.total_slippage = 0.0
        
        self._lock = threading.Lock()
    
    def submit_order(
        self, 
        order_id: str,
        symbol: str,
        side: str,
        order_type: OrderType,
        size: float,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        market_conditions: Optional[MarketConditions] = None
    ) -> SimulatedOrder:
        """Submit order for simulation"""
        
        # MANDATORY RISK ENFORCEMENT: All orders must pass CentralRiskGuard
        try:
            from ..core.mandatory_risk_enforcement import enforce_order_risk_check
            
            risk_check_result = enforce_order_risk_check(
                order_size=size,
                symbol=symbol,
                side=side,
                strategy_id="execution_simulator"
            )
            
            if not risk_check_result["approved"]:
                # Return rejected simulated order
                rejected_order = SimulatedOrder(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    size=0.0,  # Zero size for rejected orders
                    limit_price=limit_price,
                    stop_price=stop_price
                )
                rejected_order.status = OrderStatus.REJECTED
                rejected_order.rejection_reason = f"Risk Guard: {risk_check_result['reason']}"
                return rejected_order
            
            # Use approved size from risk check
            if risk_check_result["approved_size"] != size:
                size = risk_check_result["approved_size"]
                logger.info(f"Order size adjusted by risk check: {order_id} {symbol} {size}")
                
        except Exception as e:
            # Return rejected order on risk enforcement error
            error_order = SimulatedOrder(
                order_id=order_id,
                symbol=symbol, 
                side=side,
                order_type=order_type,
                size=0.0,
                limit_price=limit_price,
                stop_price=stop_price
            )
            error_order.status = OrderStatus.REJECTED
            error_order.rejection_reason = f"Risk enforcement error: {str(e)}"
            return error_order
        
        with self._lock:
            # Create simulated order
            order = SimulatedOrder(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                size=size,
                limit_price=limit_price,
                stop_price=stop_price
            )
            
            # Add submission latency
            order.submit_latency_ms = self.latency_model.get_order_submit_latency()
            
            # Determine queue position
            if market_conditions:
                is_aggressive = order_type == OrderType.MARKET
                order.queue_position = self.liquidity_model.get_queue_position(market_conditions, is_aggressive)
            
            # Set initial status
            if order_type == OrderType.MARKET:
                order.status = OrderStatus.PENDING
            else:
                order.status = OrderStatus.QUEUED
            
            # Track order
            self.active_orders[order_id] = order
            self.total_orders += 1
            
            logger.info(f"Order submitted: {order_id} {side} {size} {symbol} ({order_type.value})")
            
            return order
    
    def process_order_execution(
        self, 
        order_id: str, 
        market_conditions: MarketConditions,
        time_elapsed_ms: float = 0.0
    ) -> List[OrderFill]:
        """Process order execution based on market conditions"""
        
        with self._lock:
            if order_id not in self.active_orders:
                return []
            
            order = self.active_orders[order_id]
            new_fills = []
            
            # Skip if order is already complete
            if order.is_complete:
                return []
            
            # Check if order should execute
            should_execute, execution_price = self._should_order_execute(order, market_conditions)
            
            if should_execute:
                # Determine fill size
                fill_size = self._determine_fill_size(order, market_conditions)
                
                if fill_size > 0:
                    # Calculate slippage
                    slippage = self.slippage_model.calculate_slippage(order, market_conditions, fill_size)
                    final_price = execution_price + slippage
                    
                    # Determine fill type
                    fill_type = self._determine_fill_type(order, market_conditions, final_price)
                    
                    # Calculate fees
                    fill_value = fill_size * final_price
                    fee = self.fee_structure.calculate_fee(fill_value, fill_type)
                    
                    # Create fill
                    fill = OrderFill(
                        fill_id=f"{order_id}_{len(order.fills) + 1}",
                        order_id=order_id,
                        symbol=order.symbol,
                        side=order.side,
                        size=fill_size,
                        price=final_price,
                        fill_type=fill_type,
                        fee=fee,
                        timestamp=time.time() + time_elapsed_ms / 1000,
                        latency_ms=self.latency_model.get_fill_latency()
                    )
                    
                    # Update order
                    order.fills.append(fill)
                    order.filled_size += fill_size
                    order.remaining_size -= fill_size
                    order.total_fees += fee
                    
                    # Update average fill price
                    total_value = sum(f.size * f.price for f in order.fills)
                    order.average_fill_price = total_value / order.filled_size
                    
                    # Update timing
                    if order.first_fill_time is None:
                        order.first_fill_time = fill.timestamp
                    
                    # Update status
                    if order.remaining_size <= 0.001:  # Small tolerance for floating point
                        order.status = OrderStatus.FILLED
                        order.complete_time = fill.timestamp
                        order.remaining_size = 0.0  # Clean up floating point
                        
                        # Move to completed orders
                        self.completed_orders.append(order)
                        del self.active_orders[order_id]
                    else:
                        order.status = OrderStatus.PARTIAL
                    
                    # Track fill
                    self.order_fills.append(fill)
                    new_fills.append(fill)
                    
                    # Update statistics
                    self.total_fills += 1
                    self.total_fees += fee
                    self.total_slippage += abs(slippage)
                    
                    logger.info(f"Order fill: {order_id} {fill_size}@{final_price:.2f} (fee: ${fee:.4f})")
            
            return new_fills
    
    def _should_order_execute(self, order: SimulatedOrder, market_conditions: MarketConditions) -> Tuple[bool, float]:
        """Determine if order should execute and at what price"""
        
        if order.order_type == OrderType.MARKET:
            # Market orders execute immediately at market price
            if order.side == "buy":
                return True, market_conditions.ask_price
            else:
                return True, market_conditions.bid_price
        
        elif order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                return False, 0.0
            
            if order.side == "buy":
                # Buy limit executes if market ask <= limit price
                if market_conditions.ask_price <= order.limit_price:
                    return True, min(order.limit_price, market_conditions.ask_price)
            else:
                # Sell limit executes if market bid >= limit price
                if market_conditions.bid_price >= order.limit_price:
                    return True, max(order.limit_price, market_conditions.bid_price)
        
        return False, 0.0
    
    def _determine_fill_size(self, order: SimulatedOrder, market_conditions: MarketConditions) -> float:
        """Determine how much of the order should be filled"""
        
        # Check partial fill probability
        partial_fill_prob = self.liquidity_model.get_partial_fill_probability(order, market_conditions)
        
        if random.random() < partial_fill_prob:
            # Partial fill - fill between 10% and 90% of remaining
            fill_percentage = random.uniform(0.1, 0.9)
            return order.remaining_size * fill_percentage
        else:
            # Full fill
            return order.remaining_size
    
    def _determine_fill_type(self, order: SimulatedOrder, market_conditions: MarketConditions, fill_price: float) -> FillType:
        """Determine the type of fill for fee calculation"""
        
        if order.order_type == OrderType.MARKET:
            return FillType.AGGRESSIVE
        
        elif order.order_type == OrderType.LIMIT:
            # Check if we're providing or removing liquidity
            if order.side == "buy":
                if fill_price < market_conditions.ask_price:
                    return FillType.MAKER  # Providing liquidity
                else:
                    return FillType.TAKER  # Removing liquidity
            else:
                if fill_price > market_conditions.bid_price:
                    return FillType.MAKER  # Providing liquidity
                else:
                    return FillType.TAKER  # Removing liquidity
        
        return FillType.PASSIVE  # Default
    
    def get_execution_statistics(self) -> Dict:
        """Get comprehensive execution statistics"""
        
        with self._lock:
            if self.total_orders == 0:
                return {}
            
            # Calculate averages
            avg_fill_latency = 0.0
            avg_slippage = 0.0
            
            if self.order_fills:
                avg_fill_latency = sum(f.latency_ms for f in self.order_fills) / len(self.order_fills)
                avg_slippage = self.total_slippage / len(self.order_fills)
            
            # Fill rate statistics
            total_filled_orders = len([o for o in self.completed_orders if o.status == OrderStatus.FILLED])
            fill_rate = total_filled_orders / self.total_orders if self.total_orders > 0 else 0.0
            
            # Partial fill statistics
            partial_fills = len([o for o in self.completed_orders if len(o.fills) > 1])
            partial_fill_rate = partial_fills / self.total_orders if self.total_orders > 0 else 0.0
            
            return {
                'total_orders': self.total_orders,
                'total_fills': self.total_fills,
                'fill_rate': fill_rate,
                'partial_fill_rate': partial_fill_rate,
                'total_fees_usd': self.total_fees,
                'avg_fees_per_order': self.total_fees / max(self.total_orders, 1),
                'total_slippage_bps': self.total_slippage * 10000,
                'avg_slippage_bps': avg_slippage * 10000,
                'avg_fill_latency_ms': avg_fill_latency,
                'active_orders': len(self.active_orders),
                'completed_orders': len(self.completed_orders)
            }


# Global execution simulator
_execution_simulator: Optional[ExecutionSimulator] = None
_simulator_lock = threading.Lock()


def get_execution_simulator() -> ExecutionSimulator:
    """Get global execution simulator"""
    global _execution_simulator
    
    if _execution_simulator is None:
        with _simulator_lock:
            if _execution_simulator is None:
                _execution_simulator = ExecutionSimulator()
    
    return _execution_simulator