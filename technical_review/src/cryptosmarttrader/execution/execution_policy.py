"""
Enterprise Execution Policy System
Advanced execution controls with tradability gates, slippage budget, and order deduplication.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from decimal import Decimal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types with execution characteristics."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TWAP = "twap"
    ICEBERG = "iceberg"


class TimeInForce(Enum):
    """Time in force options."""
    GTC = "gtc"  # Good till canceled
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill
    GTD = "gtd"  # Good till date
    POST_ONLY = "post_only"  # Post only (maker)


class OrderStatus(Enum):
    """Order execution status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TradabilityGate(Enum):
    """Tradability gate types."""
    SPREAD_TOO_WIDE = "spread_too_wide"
    INSUFFICIENT_DEPTH = "insufficient_depth"
    LOW_VOLUME = "low_volume"
    HIGH_VOLATILITY = "high_volatility"
    MARKET_CLOSED = "market_closed"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class MarketConditions:
    """Current market conditions for execution decisions."""
    symbol: str
    bid: float
    ask: float
    spread_bps: float
    depth_bid: float  # USD depth at best bid
    depth_ask: float  # USD depth at best ask
    volume_1m: float  # 1-minute volume in USD
    volatility_1m: float  # 1-minute volatility
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionParams:
    """Execution parameters for order placement."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: OrderType = OrderType.LIMIT
    time_in_force: TimeInForce = TimeInForce.GTC
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    max_slippage_bps: float = 30.0  # 0.3% default slippage budget
    post_only: bool = False
    reduce_only: bool = False
    client_order_id: Optional[str] = None


@dataclass
class OrderExecution:
    """Order execution result with detailed tracking."""
    client_order_id: str
    exchange_order_id: Optional[str]
    symbol: str
    side: str
    quantity: float
    filled_quantity: float
    average_price: float
    status: OrderStatus
    slippage_bps: float
    fees: float
    execution_time_ms: float
    timestamp: datetime
    error_message: Optional[str] = None
    partial_fills: List[Dict] = field(default_factory=list)


class TradabilityLimits(BaseModel):
    """Tradability gate thresholds."""
    max_spread_bps: float = Field(default=50.0, description="Maximum spread in basis points")
    min_depth_usd: float = Field(default=1000.0, description="Minimum order book depth in USD")
    min_volume_1m_usd: float = Field(default=5000.0, description="Minimum 1-minute volume in USD")
    max_volatility_1m: float = Field(default=0.05, description="Maximum 1-minute volatility (5%)")
    max_slippage_budget_bps: float = Field(default=30.0, description="Maximum slippage budget (0.3%)")


class ExecutionPolicy:
    """
    Enterprise execution policy with comprehensive controls.
    
    Features:
    - Tradability gates with market condition checks
    - Slippage budget enforcement
    - Idempotent client order IDs with retry deduplication
    - Advanced order types (TWAP, post-only, iceberg)
    - Partial fill handling
    - Execution quality tracking
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/execution_policy.json")
        self.limits = self._load_limits()
        
        # Order tracking
        self.active_orders: Dict[str, OrderExecution] = {}
        self.completed_orders: Dict[str, OrderExecution] = {}
        self.deduplication_cache: Dict[str, datetime] = {}
        
        # Execution metrics
        self.execution_stats = {
            'total_orders': 0,
            'successful_executions': 0,
            'rejected_by_gates': 0,
            'slippage_violations': 0,
            'average_slippage_bps': 0.0,
            'average_execution_time_ms': 0.0
        }
        
        # Cache for market conditions
        self.market_conditions_cache: Dict[str, MarketConditions] = {}
        self.cache_ttl_seconds = 5  # 5-second cache
        
        logger.info("ExecutionPolicy initialized with comprehensive controls")
    
    def _load_limits(self) -> TradabilityLimits:
        """Load execution limits from configuration."""
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                return TradabilityLimits(**config_data)
            except Exception as e:
                logger.warning(f"Failed to load execution config: {e}, using defaults")
        
        return TradabilityLimits()
    
    def save_limits(self):
        """Save current execution limits."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(self.limits.dict(), f, indent=2)
        logger.info(f"Execution limits saved to {self.config_path}")
    
    def generate_client_order_id(self, params: ExecutionParams) -> str:
        """Generate deterministic, idempotent client order ID."""
        # Create deterministic hash from order parameters
        order_data = {
            'symbol': params.symbol,
            'side': params.side,
            'quantity': params.quantity,
            'order_type': params.order_type.value,
            'limit_price': params.limit_price,
            'timestamp_minute': datetime.now().strftime('%Y%m%d_%H%M')  # Minute precision for deduplication
        }
        
        order_string = json.dumps(order_data, sort_keys=True)
        hash_digest = hashlib.sha256(order_string.encode()).hexdigest()
        
        # Create readable client order ID
        client_order_id = f"CST_{params.symbol}_{params.side.upper()}_{hash_digest[:8]}"
        
        return client_order_id
    
    def check_order_deduplication(self, client_order_id: str) -> Tuple[bool, Optional[str]]:
        """Check if order was recently submitted (deduplication)."""
        # Clean old entries (older than 1 hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        expired_ids = [oid for oid, timestamp in self.deduplication_cache.items() 
                      if timestamp < cutoff_time]
        
        for oid in expired_ids:
            del self.deduplication_cache[oid]
        
        # Check if order was recently submitted
        if client_order_id in self.deduplication_cache:
            last_submission = self.deduplication_cache[client_order_id]
            time_since = datetime.now() - last_submission
            
            if time_since.total_seconds() < 60:  # 1-minute deduplication window
                return True, f"Order submitted {time_since.total_seconds():.0f} seconds ago"
        
        # Record this submission
        self.deduplication_cache[client_order_id] = datetime.now()
        return False, None
    
    def check_tradability_gates(self, conditions: MarketConditions) -> Tuple[bool, List[TradabilityGate]]:
        """Check all tradability gates for market conditions."""
        violations = []
        
        # Check spread
        if conditions.spread_bps > self.limits.max_spread_bps:
            violations.append(TradabilityGate.SPREAD_TOO_WIDE)
        
        # Check order book depth
        min_depth = min(conditions.depth_bid, conditions.depth_ask)
        if min_depth < self.limits.min_depth_usd:
            violations.append(TradabilityGate.INSUFFICIENT_DEPTH)
        
        # Check volume
        if conditions.volume_1m < self.limits.min_volume_1m_usd:
            violations.append(TradabilityGate.LOW_VOLUME)
        
        # Check volatility
        if conditions.volatility_1m > self.limits.max_volatility_1m:
            violations.append(TradabilityGate.HIGH_VOLATILITY)
        
        is_tradable = len(violations) == 0
        return is_tradable, violations
    
    def calculate_optimal_execution_strategy(self, params: ExecutionParams, 
                                           conditions: MarketConditions) -> ExecutionParams:
        """Calculate optimal execution strategy based on market conditions."""
        optimized_params = params
        
        # Large orders -> TWAP or Iceberg
        if params.quantity * conditions.ask > 10000:  # Orders > $10k
            if conditions.volume_1m > 50000:  # High volume -> TWAP
                optimized_params.order_type = OrderType.TWAP
            else:  # Low volume -> Iceberg
                optimized_params.order_type = OrderType.ICEBERG
        
        # Wide spreads -> Post-only to capture spread
        if conditions.spread_bps > 20:
            optimized_params.post_only = True
            optimized_params.time_in_force = TimeInForce.POST_ONLY
        
        # High volatility -> Tighter slippage budget
        if conditions.volatility_1m > 0.02:  # 2% volatility
            optimized_params.max_slippage_bps = min(
                optimized_params.max_slippage_bps, 
                15.0  # Tighter budget in volatile conditions
            )
        
        # Set optimal limit price
        if optimized_params.order_type == OrderType.LIMIT:
            if params.side == 'buy':
                # Aggressive but within slippage budget
                max_price = conditions.ask * (1 + optimized_params.max_slippage_bps / 10000)
                optimized_params.limit_price = min(max_price, conditions.ask * 1.001)  # Max 0.1% above ask
            else:  # sell
                min_price = conditions.bid * (1 - optimized_params.max_slippage_bps / 10000)
                optimized_params.limit_price = max(min_price, conditions.bid * 0.999)  # Max 0.1% below bid
        
        return optimized_params
    
    def validate_slippage_budget(self, params: ExecutionParams, execution_price: float, 
                                conditions: MarketConditions) -> Tuple[bool, float]:
        """Validate execution against slippage budget."""
        # Calculate reference price
        if params.side == 'buy':
            reference_price = conditions.ask
        else:
            reference_price = conditions.bid
        
        # Calculate actual slippage
        if params.side == 'buy':
            slippage_bps = ((execution_price - reference_price) / reference_price) * 10000
        else:
            slippage_bps = ((reference_price - execution_price) / reference_price) * 10000
        
        # Check against budget
        within_budget = slippage_bps <= params.max_slippage_bps
        
        return within_budget, slippage_bps
    
    async def execute_order(self, params: ExecutionParams, 
                           conditions: MarketConditions) -> OrderExecution:
        """Execute order with comprehensive controls and monitoring."""
        start_time = time.time()
        
        # Generate client order ID if not provided
        if not params.client_order_id:
            params.client_order_id = self.generate_client_order_id(params)
        
        # Check deduplication
        is_duplicate, duplicate_reason = self.check_order_deduplication(params.client_order_id)
        if is_duplicate:
            logger.warning(f"Duplicate order detected: {duplicate_reason}")
            return OrderExecution(
                client_order_id=params.client_order_id,
                exchange_order_id=None,
                symbol=params.symbol,
                side=params.side,
                quantity=params.quantity,
                filled_quantity=0.0,
                average_price=0.0,
                status=OrderStatus.REJECTED,
                slippage_bps=0.0,
                fees=0.0,
                execution_time_ms=0.0,
                timestamp=datetime.now(),
                error_message=f"Duplicate order: {duplicate_reason}"
            )
        
        # Check tradability gates
        is_tradable, gate_violations = self.check_tradability_gates(conditions)
        if not is_tradable:
            self.execution_stats['rejected_by_gates'] += 1
            violation_names = [gate.value for gate in gate_violations]
            logger.warning(f"Order rejected by tradability gates: {violation_names}")
            
            return OrderExecution(
                client_order_id=params.client_order_id,
                exchange_order_id=None,
                symbol=params.symbol,
                side=params.side,
                quantity=params.quantity,
                filled_quantity=0.0,
                average_price=0.0,
                status=OrderStatus.REJECTED,
                slippage_bps=0.0,
                fees=0.0,
                execution_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                error_message=f"Tradability gates failed: {violation_names}"
            )
        
        # Optimize execution strategy
        optimized_params = self.calculate_optimal_execution_strategy(params, conditions)
        
        # Execute order based on type
        try:
            if optimized_params.order_type == OrderType.TWAP:
                execution = await self._execute_twap_order(optimized_params, conditions)
            elif optimized_params.order_type == OrderType.ICEBERG:
                execution = await self._execute_iceberg_order(optimized_params, conditions)
            else:
                execution = await self._execute_standard_order(optimized_params, conditions)
            
            execution.execution_time_ms = (time.time() - start_time) * 1000
            
            # Validate slippage budget
            if execution.status == OrderStatus.FILLED and execution.average_price > 0:
                within_budget, actual_slippage = self.validate_slippage_budget(
                    optimized_params, execution.average_price, conditions
                )
                execution.slippage_bps = actual_slippage
                
                if not within_budget:
                    self.execution_stats['slippage_violations'] += 1
                    logger.warning(f"Slippage budget exceeded: {actual_slippage:.1f} bps > {optimized_params.max_slippage_bps:.1f} bps")
            
            # Update statistics
            self._update_execution_stats(execution)
            
            # Store execution record
            if execution.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                self.completed_orders[execution.client_order_id] = execution
            else:
                self.active_orders[execution.client_order_id] = execution
            
            return execution
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return OrderExecution(
                client_order_id=params.client_order_id,
                exchange_order_id=None,
                symbol=params.symbol,
                side=params.side,
                quantity=params.quantity,
                filled_quantity=0.0,
                average_price=0.0,
                status=OrderStatus.REJECTED,
                slippage_bps=0.0,
                fees=0.0,
                execution_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                error_message=str(e)
            )
    
    async def _execute_standard_order(self, params: ExecutionParams, 
                                    conditions: MarketConditions) -> OrderExecution:
        """Execute standard order (market/limit)."""
        # Simulate order execution (replace with actual exchange API calls)
        await asyncio.sleep(0.1)  # Simulate network latency
        
        # Simulate execution price
        if params.order_type == OrderType.MARKET:
            if params.side == 'buy':
                execution_price = conditions.ask * (1 + 0.0005)  # Small slippage
            else:
                execution_price = conditions.bid * (1 - 0.0005)
        else:  # LIMIT
            execution_price = params.limit_price or (
                conditions.ask if params.side == 'buy' else conditions.bid
            )
        
        # Simulate fees (0.1% taker fee)
        fees = params.quantity * execution_price * 0.001
        
        return OrderExecution(
            client_order_id=params.client_order_id,
            exchange_order_id=f"EX_{int(time.time() * 1000)}",
            symbol=params.symbol,
            side=params.side,
            quantity=params.quantity,
            filled_quantity=params.quantity,
            average_price=execution_price,
            status=OrderStatus.FILLED,
            slippage_bps=0.0,  # Will be calculated later
            fees=fees,
            execution_time_ms=0.0,  # Will be set by caller
            timestamp=datetime.now()
        )
    
    async def _execute_twap_order(self, params: ExecutionParams, 
                                conditions: MarketConditions) -> OrderExecution:
        """Execute TWAP (Time-Weighted Average Price) order."""
        # Split order into smaller chunks over time
        num_slices = min(10, max(3, int(params.quantity * conditions.ask / 1000)))  # 1 slice per $1k
        slice_size = params.quantity / num_slices
        slice_interval = 30  # 30 seconds between slices
        
        total_filled = 0.0
        total_cost = 0.0
        total_fees = 0.0
        partial_fills = []
        
        for i in range(num_slices):
            # Execute slice
            slice_params = ExecutionParams(
                symbol=params.symbol,
                side=params.side,
                quantity=slice_size,
                order_type=OrderType.LIMIT,
                limit_price=params.limit_price,
                max_slippage_bps=params.max_slippage_bps
            )
            
            slice_execution = await self._execute_standard_order(slice_params, conditions)
            
            if slice_execution.status == OrderStatus.FILLED:
                total_filled += slice_execution.filled_quantity
                total_cost += slice_execution.filled_quantity * slice_execution.average_price
                total_fees += slice_execution.fees
                
                partial_fills.append({
                    'slice': i + 1,
                    'quantity': slice_execution.filled_quantity,
                    'price': slice_execution.average_price,
                    'timestamp': slice_execution.timestamp.isoformat()
                })
            
            # Wait between slices (except last one)
            if i < num_slices - 1:
                await asyncio.sleep(min(slice_interval, 5))  # Cap at 5 seconds for demo
        
        average_price = total_cost / total_filled if total_filled > 0 else 0.0
        
        return OrderExecution(
            client_order_id=params.client_order_id,
            exchange_order_id=f"TWAP_{int(time.time() * 1000)}",
            symbol=params.symbol,
            side=params.side,
            quantity=params.quantity,
            filled_quantity=total_filled,
            average_price=average_price,
            status=OrderStatus.FILLED if total_filled == params.quantity else OrderStatus.PARTIALLY_FILLED,
            slippage_bps=0.0,
            fees=total_fees,
            execution_time_ms=0.0,
            timestamp=datetime.now(),
            partial_fills=partial_fills
        )
    
    async def _execute_iceberg_order(self, params: ExecutionParams, 
                                   conditions: MarketConditions) -> OrderExecution:
        """Execute iceberg order (hidden quantity)."""
        # Show only small portion of total order
        visible_size = min(params.quantity * 0.1, conditions.depth_ask / conditions.ask)
        remaining_quantity = params.quantity
        
        total_filled = 0.0
        total_cost = 0.0
        total_fees = 0.0
        partial_fills = []
        
        while remaining_quantity > 0:
            # Execute visible portion
            current_slice = min(visible_size, remaining_quantity)
            
            slice_params = ExecutionParams(
                symbol=params.symbol,
                side=params.side,
                quantity=current_slice,
                order_type=OrderType.LIMIT,
                limit_price=params.limit_price,
                post_only=True  # Iceberg orders are typically passive
            )
            
            slice_execution = await self._execute_standard_order(slice_params, conditions)
            
            if slice_execution.status == OrderStatus.FILLED:
                total_filled += slice_execution.filled_quantity
                total_cost += slice_execution.filled_quantity * slice_execution.average_price
                total_fees += slice_execution.fees
                remaining_quantity -= slice_execution.filled_quantity
                
                partial_fills.append({
                    'slice': len(partial_fills) + 1,
                    'quantity': slice_execution.filled_quantity,
                    'price': slice_execution.average_price,
                    'timestamp': slice_execution.timestamp.isoformat()
                })
            else:
                break  # Stop if slice fails
            
            await asyncio.sleep(2)  # Wait between iceberg reveals
        
        average_price = total_cost / total_filled if total_filled > 0 else 0.0
        
        return OrderExecution(
            client_order_id=params.client_order_id,
            exchange_order_id=f"ICE_{int(time.time() * 1000)}",
            symbol=params.symbol,
            side=params.side,
            quantity=params.quantity,
            filled_quantity=total_filled,
            average_price=average_price,
            status=OrderStatus.FILLED if total_filled == params.quantity else OrderStatus.PARTIALLY_FILLED,
            slippage_bps=0.0,
            fees=total_fees,
            execution_time_ms=0.0,
            timestamp=datetime.now(),
            partial_fills=partial_fills
        )
    
    def _update_execution_stats(self, execution: OrderExecution):
        """Update execution statistics."""
        self.execution_stats['total_orders'] += 1
        
        if execution.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
            self.execution_stats['successful_executions'] += 1
            
            # Update average slippage
            total_slippage = (self.execution_stats['average_slippage_bps'] * 
                            (self.execution_stats['successful_executions'] - 1) + 
                            execution.slippage_bps)
            self.execution_stats['average_slippage_bps'] = total_slippage / self.execution_stats['successful_executions']
            
            # Update average execution time
            total_time = (self.execution_stats['average_execution_time_ms'] * 
                         (self.execution_stats['successful_executions'] - 1) + 
                         execution.execution_time_ms)
            self.execution_stats['average_execution_time_ms'] = total_time / self.execution_stats['successful_executions']
    
    def get_execution_stats(self) -> Dict:
        """Get comprehensive execution statistics."""
        total_orders = self.execution_stats['total_orders']
        successful_rate = (self.execution_stats['successful_executions'] / total_orders * 100) if total_orders > 0 else 0.0
        rejection_rate = (self.execution_stats['rejected_by_gates'] / total_orders * 100) if total_orders > 0 else 0.0
        
        return {
            'total_orders': total_orders,
            'successful_executions': self.execution_stats['successful_executions'],
            'success_rate_percent': successful_rate,
            'rejected_by_gates': self.execution_stats['rejected_by_gates'],
            'rejection_rate_percent': rejection_rate,
            'slippage_violations': self.execution_stats['slippage_violations'],
            'average_slippage_bps': self.execution_stats['average_slippage_bps'],
            'average_execution_time_ms': self.execution_stats['average_execution_time_ms'],
            'active_orders': len(self.active_orders),
            'completed_orders': len(self.completed_orders),
            'deduplication_cache_size': len(self.deduplication_cache)
        }
    
    def cancel_order(self, client_order_id: str) -> bool:
        """Cancel active order."""
        if client_order_id in self.active_orders:
            order = self.active_orders[client_order_id]
            order.status = OrderStatus.CANCELED
            del self.active_orders[client_order_id]
            self.completed_orders[client_order_id] = order
            logger.info(f"Order canceled: {client_order_id}")
            return True
        
        logger.warning(f"Cannot cancel order {client_order_id}: not found in active orders")
        return False
    
    def get_order_status(self, client_order_id: str) -> Optional[OrderExecution]:
        """Get order status by client order ID."""
        return (self.active_orders.get(client_order_id) or 
                self.completed_orders.get(client_order_id))


# Execution utility functions
def create_market_conditions(symbol: str, bid: float, ask: float, 
                           volume_1m: float = 10000.0) -> MarketConditions:
    """Create market conditions for testing."""
    spread_bps = ((ask - bid) / bid) * 10000
    
    return MarketConditions(
        symbol=symbol,
        bid=bid,
        ask=ask,
        spread_bps=spread_bps,
        depth_bid=volume_1m * 0.1,  # 10% of 1m volume as depth
        depth_ask=volume_1m * 0.1,
        volume_1m=volume_1m,
        volatility_1m=0.02  # 2% default volatility
    )