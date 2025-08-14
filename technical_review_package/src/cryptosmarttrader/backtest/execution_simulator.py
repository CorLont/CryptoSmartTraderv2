"""
Advanced Execution Simulator
Models fees, partial fills, latency, queue position voor realistic backtest-live parity
"""

import numpy as np
import pandas as pd
import logging
import time
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class FillType(Enum):
    MAKER = "maker"
    TAKER = "taker"
    AGGRESSIVE = "aggressive"


@dataclass
class MarketMicrostructure:
    """Real-time market microstructure data"""
    symbol: str
    timestamp: float
    bid_price: float
    ask_price: float
    bid_size: float
    ask_size: float
    last_price: float
    spread_bps: float
    depth_5_levels: Dict[str, List[Tuple[float, float]]]  # {"bids": [(price, size)], "asks": [...]}
    recent_trades: List[Tuple[float, float, str]]  # [(price, size, side)]
    volume_1min: float
    volatility_1min: float


@dataclass
class OrderRequest:
    """Order request for simulation"""
    order_id: str
    symbol: str
    side: str  # "buy", "sell"
    order_type: str  # "market", "limit", "stop_limit"
    size: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # "GTC", "IOC", "FOK"
    post_only: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class Fill:
    """Individual fill execution"""
    fill_id: str
    order_id: str
    symbol: str
    side: str
    size: float
    price: float
    fee: float
    fee_currency: str
    fill_type: FillType
    timestamp: float
    latency_ms: float
    queue_position: int


@dataclass
class OrderResult:
    """Complete order execution result"""
    order_id: str
    symbol: str
    side: str
    requested_size: float
    filled_size: float
    average_price: float
    total_fee: float
    status: OrderStatus
    fills: List[Fill]
    total_latency_ms: float
    slippage_bps: float
    execution_quality_score: float  # 0-100
    rejection_reason: Optional[str] = None


@dataclass
class ExchangeConfig:
    """Exchange-specific configuration"""
    name: str
    maker_fee_bps: float = 10.0  # 0.1% maker fee
    taker_fee_bps: float = 25.0  # 0.25% taker fee
    min_order_size: float = 10.0  # Min $10 order
    max_order_size: float = 1000000.0  # Max $1M order
    base_latency_ms: float = 50.0  # Base latency
    latency_variance_ms: float = 20.0  # Latency variance
    partial_fill_probability: float = 0.15  # 15% chance of partial fill
    rejection_probability: float = 0.02  # 2% rejection rate
    queue_depth_factor: float = 0.1  # Queue position factor


class ExecutionSimulator:
    """
    Advanced execution simulator voor realistic backtest-live parity
    Models fees, partial fills, latency, queue position, market impact
    """
    
    def __init__(self, exchange_config: Optional[ExchangeConfig] = None):
        self.exchange_config = exchange_config or ExchangeConfig("kraken")
        self.logger = logging.getLogger(__name__)
        
        # Market state
        self.market_data: Dict[str, MarketMicrostructure] = {}
        self.order_book_depth: Dict[str, Dict] = {}
        
        # Execution tracking
        self.pending_orders: Dict[str, OrderRequest] = {}
        self.execution_history: List[OrderResult] = []
        self.fill_count = 0
        
        # Performance metrics
        self.total_slippage_bps = 0.0
        self.total_fees_paid = 0.0
        self.average_latency_ms = 0.0
        self.fill_rate = 1.0
        self.rejection_count = 0
    
    def update_market_data(self, market_data: MarketMicrostructure):
        """Update real-time market microstructure data"""
        self.market_data[market_data.symbol] = market_data
        
        # Update order book depth simulation
        self._simulate_order_book_depth(market_data)
    
    def _simulate_order_book_depth(self, market_data: MarketMicrostructure):
        """Simulate realistic order book depth"""
        symbol = market_data.symbol
        
        # Simulate depth based on recent volume and volatility
        base_depth = market_data.volume_1min / 60  # Per second volume
        vol_adjustment = max(0.5, 2.0 - market_data.volatility_1min)  # Lower depth in high vol
        
        # Generate realistic depth profile
        depth_levels = []
        for i in range(10):  # 10 levels each side
            level_depth = base_depth * np.exp(-i * 0.3) * vol_adjustment
            depth_levels.append(level_depth)
        
        self.order_book_depth[symbol] = {
            "bid_depths": depth_levels,
            "ask_depths": depth_levels,
            "last_update": market_data.timestamp
        }
    
    def simulate_order_execution(
        self, 
        order: OrderRequest,
        market_conditions: Optional[MarketMicrostructure] = None
    ) -> OrderResult:
        """
        Simulate realistic order execution met alle market microstructure effecten
        
        Args:
            order: Order request to execute
            market_conditions: Current market conditions (optional)
            
        Returns:
            OrderResult with realistic execution simulation
        """
        
        if market_conditions is None:
            market_conditions = self.market_data.get(order.symbol)
        
        if market_conditions is None:
            return OrderResult(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                requested_size=order.size,
                filled_size=0.0,
                average_price=0.0,
                total_fee=0.0,
                status=OrderStatus.REJECTED,
                fills=[],
                total_latency_ms=0.0,
                slippage_bps=0.0,
                execution_quality_score=0.0,
                rejection_reason="No market data available"
            )
        
        # Step 1: Pre-execution validation
        validation_result = self._validate_order(order, market_conditions)
        if validation_result:
            return validation_result
        
        # Step 2: Calculate execution latency
        execution_latency = self._calculate_execution_latency(order, market_conditions)
        
        # Step 3: Simulate market impact during latency
        adjusted_market = self._simulate_market_impact(order, market_conditions, execution_latency)
        
        # Step 4: Determine fill strategy
        fills = self._simulate_order_fills(order, adjusted_market, execution_latency)
        
        # Step 5: Calculate execution metrics
        result = self._calculate_execution_result(order, fills, market_conditions)
        
        # Step 6: Update tracking
        self.execution_history.append(result)
        self._update_performance_metrics(result)
        
        self.logger.info(
            f"🔧 Simulated execution: {order.order_id} {order.side} {order.size:.2f} "
            f"@ {result.average_price:.2f} (slippage: {result.slippage_bps:.1f} bps, "
            f"latency: {result.total_latency_ms:.1f}ms)"
        )
        
        return result
    
    def _validate_order(
        self, 
        order: OrderRequest, 
        market_conditions: MarketMicrostructure
    ) -> Optional[OrderResult]:
        """Validate order before execution"""
        
        # Size validation
        order_value = order.size * market_conditions.last_price
        if order_value < self.exchange_config.min_order_size:
            return OrderResult(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                requested_size=order.size,
                filled_size=0.0,
                average_price=0.0,
                total_fee=0.0,
                status=OrderStatus.REJECTED,
                fills=[],
                total_latency_ms=0.0,
                slippage_bps=0.0,
                execution_quality_score=0.0,
                rejection_reason=f"Order size too small: ${order_value:.2f} < ${self.exchange_config.min_order_size}"
            )
        
        if order_value > self.exchange_config.max_order_size:
            return OrderResult(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                requested_size=order.size,
                filled_size=0.0,
                average_price=0.0,
                total_fee=0.0,
                status=OrderStatus.REJECTED,
                fills=[],
                total_latency_ms=0.0,
                slippage_bps=0.0,
                execution_quality_score=0.0,
                rejection_reason=f"Order size too large: ${order_value:.2f} > ${self.exchange_config.max_order_size}"
            )
        
        # Random rejection simulation
        if random.random() < self.exchange_config.rejection_probability:
            rejection_reasons = [
                "Insufficient liquidity",
                "Price out of bounds",
                "System maintenance",
                "Risk management rejection"
            ]
            return OrderResult(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                requested_size=order.size,
                filled_size=0.0,
                average_price=0.0,
                total_fee=0.0,
                status=OrderStatus.REJECTED,
                fills=[],
                total_latency_ms=0.0,
                slippage_bps=0.0,
                execution_quality_score=0.0,
                rejection_reason=random.choice(rejection_reasons)
            )
        
        return None  # Order is valid
    
    def _calculate_execution_latency(
        self, 
        order: OrderRequest, 
        market_conditions: MarketMicrostructure
    ) -> float:
        """Calculate realistic execution latency"""
        
        # Base latency
        base_latency = self.exchange_config.base_latency_ms
        
        # Network variance
        network_variance = np.random.normal(0, self.exchange_config.latency_variance_ms)
        
        # Market volatility impact (higher vol = higher latency)
        vol_impact = market_conditions.volatility_1min * 10  # ms per vol unit
        
        # Order size impact (larger orders = higher latency)
        order_value = order.size * market_conditions.last_price
        size_impact = max(0, (order_value / 100000) * 5)  # 5ms per $100k
        
        # Queue position simulation
        queue_factor = random.uniform(0.5, 2.0)  # Random queue position
        
        total_latency = base_latency + network_variance + vol_impact + size_impact + queue_factor
        
        return max(10.0, total_latency)  # Minimum 10ms latency
    
    def _simulate_market_impact(
        self, 
        order: OrderRequest, 
        market_conditions: MarketMicrostructure,
        latency_ms: float
    ) -> MarketMicrostructure:
        """Simulate market movement during execution latency"""
        
        # Calculate time-based price movement
        time_factor = latency_ms / 1000.0  # Convert to seconds
        vol_per_second = market_conditions.volatility_1min / 60.0
        
        # Random price movement during latency
        price_change_pct = np.random.normal(0, vol_per_second * time_factor)
        new_price = market_conditions.last_price * (1 + price_change_pct)
        
        # Market impact based on order size vs available liquidity
        order_value = order.size * market_conditions.last_price
        depth_value = (market_conditions.bid_size + market_conditions.ask_size) * market_conditions.last_price
        
        if depth_value > 0:
            impact_ratio = order_value / depth_value
            impact_bps = min(50, impact_ratio * 100)  # Max 50 bps impact
            
            if order.side == "buy":
                # Buying pushes price up
                impact_adjustment = 1 + (impact_bps / 10000)
            else:
                # Selling pushes price down
                impact_adjustment = 1 - (impact_bps / 10000)
            
            new_price *= impact_adjustment
        
        # Create adjusted market conditions
        spread_adjustment = 1 + (market_conditions.volatility_1min * 0.1)  # Wider spreads in vol
        new_spread = market_conditions.spread_bps * spread_adjustment
        
        return MarketMicrostructure(
            symbol=market_conditions.symbol,
            timestamp=market_conditions.timestamp + (latency_ms / 1000),
            bid_price=new_price - (new_spread / 20000 * new_price),
            ask_price=new_price + (new_spread / 20000 * new_price),
            bid_size=market_conditions.bid_size * random.uniform(0.8, 1.2),
            ask_size=market_conditions.ask_size * random.uniform(0.8, 1.2),
            last_price=new_price,
            spread_bps=new_spread,
            depth_5_levels=market_conditions.depth_5_levels,
            recent_trades=market_conditions.recent_trades,
            volume_1min=market_conditions.volume_1min,
            volatility_1min=market_conditions.volatility_1min
        )
    
    def _simulate_order_fills(
        self, 
        order: OrderRequest, 
        market_conditions: MarketMicrostructure,
        latency_ms: float
    ) -> List[Fill]:
        """Simulate realistic order fills with partial fills"""
        
        fills = []
        remaining_size = order.size
        cumulative_latency = 0.0
        
        # Determine if this will be a partial fill scenario
        will_partial_fill = (
            random.random() < self.exchange_config.partial_fill_probability and
            order.order_type == "limit"
        )
        
        while remaining_size > 0:
            # Calculate fill size
            if will_partial_fill and len(fills) == 0:
                # First fill is partial
                fill_size = remaining_size * random.uniform(0.3, 0.8)
            else:
                # Complete remaining size
                fill_size = remaining_size
            
            # Determine fill price and type
            fill_price, fill_type = self._calculate_fill_price(
                order, market_conditions, fill_size
            )
            
            # Calculate fees
            fee_bps = (self.exchange_config.maker_fee_bps 
                      if fill_type == FillType.MAKER 
                      else self.exchange_config.taker_fee_bps)
            fee = (fill_size * fill_price * fee_bps) / 10000
            
            # Calculate individual fill latency
            fill_latency = latency_ms / max(1, len(fills) + 1) if len(fills) == 0 else latency_ms * 0.1
            cumulative_latency += fill_latency
            
            # Create fill
            fill = Fill(
                fill_id=f"{order.order_id}_{len(fills)+1}",
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                size=fill_size,
                price=fill_price,
                fee=fee,
                fee_currency="USD",
                fill_type=fill_type,
                timestamp=market_conditions.timestamp + (cumulative_latency / 1000),
                latency_ms=fill_latency,
                queue_position=random.randint(1, 50)
            )
            
            fills.append(fill)
            remaining_size -= fill_size
            
            # Break if partial fill scenario and we've had one fill
            if will_partial_fill and len(fills) == 1:
                break
            
            # Small chance of additional partial fills
            if remaining_size > 0 and random.random() < 0.1:
                continue
            else:
                break
        
        return fills
    
    def _calculate_fill_price(
        self, 
        order: OrderRequest, 
        market_conditions: MarketMicrostructure,
        fill_size: float
    ) -> Tuple[float, FillType]:
        """Calculate realistic fill price and type"""
        
        if order.order_type == "market":
            # Market orders are aggressive takers
            if order.side == "buy":
                fill_price = market_conditions.ask_price
            else:
                fill_price = market_conditions.bid_price
            
            # Add slippage for large orders
            order_value = fill_size * fill_price
            if order_value > 50000:  # Large order threshold
                slippage_factor = min(0.005, order_value / 10000000)  # Max 0.5% slippage
                if order.side == "buy":
                    fill_price *= (1 + slippage_factor)
                else:
                    fill_price *= (1 - slippage_factor)
            
            return fill_price, FillType.TAKER
        
        elif order.order_type == "limit":
            # Determine if limit order gets maker or taker treatment
            if order.limit_price is None:
                order.limit_price = market_conditions.last_price
            
            if order.side == "buy":
                if order.limit_price >= market_conditions.ask_price:
                    # Aggressive limit order (crosses spread)
                    fill_price = market_conditions.ask_price
                    return fill_price, FillType.AGGRESSIVE
                else:
                    # Passive limit order
                    fill_price = order.limit_price
                    return fill_price, FillType.MAKER
            else:  # sell
                if order.limit_price <= market_conditions.bid_price:
                    # Aggressive limit order
                    fill_price = market_conditions.bid_price
                    return fill_price, FillType.AGGRESSIVE
                else:
                    # Passive limit order
                    fill_price = order.limit_price
                    return fill_price, FillType.MAKER
        
        # Default fallback
        return market_conditions.last_price, FillType.TAKER
    
    def _calculate_execution_result(
        self, 
        order: OrderRequest, 
        fills: List[Fill],
        original_market: MarketMicrostructure
    ) -> OrderResult:
        """Calculate comprehensive execution result"""
        
        if not fills:
            return OrderResult(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                requested_size=order.size,
                filled_size=0.0,
                average_price=0.0,
                total_fee=0.0,
                status=OrderStatus.CANCELLED,
                fills=[],
                total_latency_ms=0.0,
                slippage_bps=0.0,
                execution_quality_score=0.0
            )
        
        # Calculate aggregate metrics
        total_filled = sum(fill.size for fill in fills)
        total_value = sum(fill.size * fill.price for fill in fills)
        average_price = total_value / total_filled if total_filled > 0 else 0.0
        total_fee = sum(fill.fee for fill in fills)
        total_latency = max(fill.latency_ms for fill in fills)
        
        # Calculate slippage
        benchmark_price = (original_market.bid_price + original_market.ask_price) / 2
        slippage_bps = ((average_price - benchmark_price) / benchmark_price) * 10000
        if order.side == "sell":
            slippage_bps = -slippage_bps  # Invert for sell orders
        
        # Determine status
        if total_filled >= order.size * 0.99:  # 99% fill threshold
            status = OrderStatus.FILLED
        elif total_filled > 0:
            status = OrderStatus.PARTIAL
        else:
            status = OrderStatus.CANCELLED
        
        # Calculate execution quality score (0-100)
        quality_score = self._calculate_execution_quality(
            order, fills, slippage_bps, total_latency, original_market
        )
        
        return OrderResult(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            requested_size=order.size,
            filled_size=total_filled,
            average_price=average_price,
            total_fee=total_fee,
            status=status,
            fills=fills,
            total_latency_ms=total_latency,
            slippage_bps=abs(slippage_bps),  # Always positive for reporting
            execution_quality_score=quality_score
        )
    
    def _calculate_execution_quality(
        self, 
        order: OrderRequest, 
        fills: List[Fill],
        slippage_bps: float,
        latency_ms: float,
        market_conditions: MarketMicrostructure
    ) -> float:
        """Calculate execution quality score (0-100)"""
        
        score = 100.0
        
        # Slippage penalty (higher slippage = lower score)
        slippage_penalty = min(30, abs(slippage_bps) / 2)  # Max 30 point penalty
        score -= slippage_penalty
        
        # Latency penalty (higher latency = lower score)
        latency_penalty = min(20, (latency_ms - 50) / 10)  # Max 20 point penalty
        score -= max(0, latency_penalty)
        
        # Partial fill penalty
        fill_ratio = sum(fill.size for fill in fills) / order.size
        if fill_ratio < 1.0:
            partial_penalty = (1.0 - fill_ratio) * 25  # Max 25 point penalty
            score -= partial_penalty
        
        # Fee efficiency (maker fills get bonus)
        maker_fills = sum(1 for fill in fills if fill.fill_type == FillType.MAKER)
        if len(fills) > 0:
            maker_ratio = maker_fills / len(fills)
            score += maker_ratio * 10  # Max 10 point bonus
        
        # Market impact consideration
        if market_conditions.spread_bps < 20:  # Tight spread = better execution environment
            score += 5
        
        return max(0, min(100, score))
    
    def _update_performance_metrics(self, result: OrderResult):
        """Update cumulative performance metrics"""
        
        # Update totals
        self.total_slippage_bps += result.slippage_bps
        self.total_fees_paid += result.total_fee
        
        # Update averages
        total_executions = len(self.execution_history)
        if total_executions > 0:
            self.average_latency_ms = (
                (self.average_latency_ms * (total_executions - 1) + result.total_latency_ms) 
                / total_executions
            )
        
        # Update fill rate
        filled_orders = sum(1 for r in self.execution_history if r.status == OrderStatus.FILLED)
        self.fill_rate = filled_orders / total_executions if total_executions > 0 else 0.0
        
        # Update rejection count
        if result.status == OrderStatus.REJECTED:
            self.rejection_count += 1
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive execution statistics"""
        
        if not self.execution_history:
            return {"error": "No execution history available"}
        
        total_executions = len(self.execution_history)
        filled_orders = [r for r in self.execution_history if r.status == OrderStatus.FILLED]
        partial_orders = [r for r in self.execution_history if r.status == OrderStatus.PARTIAL]
        
        return {
            "total_executions": total_executions,
            "fill_statistics": {
                "fill_rate": len(filled_orders) / total_executions,
                "partial_fill_rate": len(partial_orders) / total_executions,
                "rejection_rate": self.rejection_count / total_executions
            },
            "performance_metrics": {
                "average_slippage_bps": self.total_slippage_bps / total_executions,
                "average_latency_ms": self.average_latency_ms,
                "total_fees_paid": self.total_fees_paid,
                "average_execution_quality": np.mean([r.execution_quality_score for r in self.execution_history])
            },
            "cost_breakdown": {
                "maker_fee_percentage": len([f for r in self.execution_history for f in r.fills if f.fill_type == FillType.MAKER]) / max(1, sum(len(r.fills) for r in self.execution_history)),
                "average_fee_bps": (self.total_fees_paid / max(1, sum(r.filled_size * r.average_price for r in filled_orders))) * 10000 if filled_orders else 0
            }
        }


# Global execution simulator instance
_global_execution_simulator: Optional[ExecutionSimulator] = None


def get_global_execution_simulator() -> ExecutionSimulator:
    """Get or create global execution simulator"""
    global _global_execution_simulator
    if _global_execution_simulator is None:
        _global_execution_simulator = ExecutionSimulator()
        logger.info("✅ Global ExecutionSimulator initialized")
    return _global_execution_simulator


def reset_global_execution_simulator():
    """Reset global execution simulator (for testing)"""
    global _global_execution_simulator
    _global_execution_simulator = None