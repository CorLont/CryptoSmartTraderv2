"""
Execution Simulator

Realistic execution simulation that mirrors live trading conditions
including latencies, order queues, partial fills, and market impact.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging
import asyncio
import random

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"
    POST_ONLY = "post_only"
    FILL_OR_KILL = "fill_or_kill"
    IMMEDIATE_OR_CANCEL = "immediate_or_cancel"


class OrderStatus(Enum):
    """Order execution status"""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class FillType(Enum):
    """Fill type classification"""
    MAKER = "maker"
    TAKER = "taker"
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"


@dataclass
class OrderRequest:
    """Order execution request"""
    order_id: str
    timestamp: datetime
    pair: str
    side: str               # "buy" or "sell"
    order_type: OrderType
    size: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # GTC, IOC, FOK
    reduce_only: bool = False
    post_only: bool = False
    
    # Advanced parameters
    max_slippage_bps: Optional[int] = None
    min_fill_size: Optional[float] = None
    execution_algo: str = "default"  # default, twap, vwap, iceberg


@dataclass
class Fill:
    """Individual fill record"""
    fill_id: str
    timestamp: datetime
    order_id: str
    size: float
    price: float
    fee: float
    fee_currency: str
    fill_type: FillType
    liquidity: str          # "maker" or "taker"
    
    # Market context
    bid_at_fill: float
    ask_at_fill: float
    spread_bps: float
    market_impact_bps: float
    
    @property
    def notional_value(self) -> float:
        """Calculate notional value of fill"""
        return self.size * self.price


@dataclass
class OrderResult:
    """Complete order execution result"""
    order_id: str
    request: OrderRequest
    status: OrderStatus
    
    # Execution details
    fills: List[Fill]
    total_filled_size: float
    avg_fill_price: float
    total_fees: float
    
    # Timing
    submit_time: datetime
    first_fill_time: Optional[datetime]
    last_fill_time: Optional[datetime]
    completion_time: Optional[datetime]
    
    # Performance metrics
    expected_price: float
    realized_slippage_bps: float
    execution_shortfall_bps: float
    market_impact_bps: float
    timing_cost_bps: float
    
    # Quality scores
    fill_rate: float        # Proportion filled
    speed_score: float      # Execution speed (0-1)
    cost_score: float       # Cost efficiency (0-1)
    overall_quality: float  # Overall execution quality (0-1)
    
    @property
    def is_complete(self) -> bool:
        """Check if order is completely filled"""
        return self.status == OrderStatus.FILLED
    
    @property
    def execution_time_seconds(self) -> float:
        """Calculate total execution time"""
        if self.completion_time and self.submit_time:
            return (self.completion_time - self.submit_time).total_seconds()
        return 0.0


@dataclass
class ExecutionMetrics:
    """Execution performance metrics"""
    total_orders: int
    filled_orders: int
    partially_filled_orders: int
    cancelled_orders: int
    rejected_orders: int
    
    # Fill metrics
    fill_rate: float
    avg_fill_time_seconds: float
    avg_slippage_bps: float
    avg_market_impact_bps: float
    
    # Cost metrics
    total_fees: float
    maker_fill_rate: float
    taker_fill_rate: float
    avg_effective_spread_bps: float
    
    # Quality scores
    avg_execution_quality: float
    speed_percentile_90: float
    cost_percentile_90: float


class ExecutionSimulator:
    """
    Advanced execution simulator for backtest-live parity
    """
    
    def __init__(self):
        self.order_history = []
        self.active_orders = {}
        
        # Simulation parameters
        self.base_latency_ms = 50       # Base network latency
        self.latency_variance_ms = 20   # Latency variance
        self.queue_processing_ms = 10   # Queue processing time
        
        # Market microstructure
        self.min_spread_bps = 2         # Minimum bid-ask spread
        self.max_spread_bps = 50        # Maximum spread under stress
        self.impact_decay_seconds = 300  # Market impact decay time
        
        # Fee structure
        self.maker_fee_bps = 25         # 0.25% maker fee
        self.taker_fee_bps = 40         # 0.40% taker fee
        self.fee_tier_discounts = {     # Volume-based discounts
            1000000: 0.8,   # 20% discount for $1M+ volume
            5000000: 0.6,   # 40% discount for $5M+ volume
            10000000: 0.4   # 60% discount for $10M+ volume
        }
        
        # Execution quality factors
        self.partial_fill_probability = 0.15    # 15% chance of partial fill
        self.cancel_probability = 0.05          # 5% chance of cancellation
        self.reject_probability = 0.02          # 2% chance of rejection
        
        # Market state tracking
        self.recent_volume = {}         # Recent trading volume per pair
        self.market_stress_factor = 1.0  # Overall market stress multiplier
        
    async def execute_order(self, 
                          order_request: OrderRequest,
                          market_data: Dict[str, Any]) -> OrderResult:
        """Execute order with realistic simulation"""
        try:
            logger.info(f"Executing order: {order_request.order_id}")
            
            # Simulate order submission latency
            await self._simulate_latency("submission")
            
            # Validate order
            validation_result = self._validate_order(order_request, market_data)
            if not validation_result["valid"]:
                return self._create_rejected_order(order_request, validation_result["reason"])
            
            # Add to active orders
            self.active_orders[order_request.order_id] = {
                "request": order_request,
                "submit_time": datetime.now(),
                "remaining_size": order_request.size,
                "fills": []
            }
            
            # Execute based on order type
            if order_request.order_type == OrderType.MARKET:
                result = await self._execute_market_order(order_request, market_data)
            elif order_request.order_type == OrderType.LIMIT:
                result = await self._execute_limit_order(order_request, market_data)
            elif order_request.order_type == OrderType.POST_ONLY:
                result = await self._execute_post_only_order(order_request, market_data)
            else:
                result = await self._execute_advanced_order(order_request, market_data)
            
            # Store in history
            self.order_history.append(result)
            
            # Clean up active orders
            if order_request.order_id in self.active_orders:
                del self.active_orders[order_request.order_id]
            
            return result
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return self._create_error_order(order_request, str(e))
    
    async def _execute_market_order(self, 
                                  order_request: OrderRequest,
                                  market_data: Dict[str, Any]) -> OrderResult:
        """Execute market order with realistic fills"""
        try:
            pair_data = market_data.get(order_request.pair, {})
            current_price = pair_data.get("price", 0)
            bid = pair_data.get("bid", current_price * 0.999)
            ask = pair_data.get("ask", current_price * 1.001)
            
            fills = []
            remaining_size = order_request.size
            fill_counter = 0
            
            # Simulate aggressive execution across order book levels
            while remaining_size > 0 and fill_counter < 10:  # Max 10 fills
                # Calculate market impact
                impact_bps = self._calculate_market_impact(
                    order_request.pair, remaining_size, order_request.side
                )
                
                # Determine execution price
                if order_request.side == "buy":
                    execution_price = ask * (1 + impact_bps / 10000)
                else:
                    execution_price = bid * (1 - impact_bps / 10000)
                
                # Simulate partial fill
                available_liquidity = self._get_available_liquidity(
                    order_request.pair, execution_price, order_request.side
                )
                
                fill_size = min(remaining_size, available_liquidity)
                
                # Add some randomness to fill size
                if fill_counter > 0:  # Not first fill
                    fill_size *= random.uniform(0.7, 1.0)
                
                fill_size = max(0.001, fill_size)  # Minimum fill size
                
                # Create fill
                fill = Fill(
                    fill_id=f"{order_request.order_id}_fill_{fill_counter + 1}",
                    timestamp=datetime.now(),
                    order_id=order_request.order_id,
                    size=fill_size,
                    price=execution_price,
                    fee=self._calculate_fee(fill_size * execution_price, FillType.TAKER),
                    fee_currency="USD",
                    fill_type=FillType.TAKER,
                    liquidity="taker",
                    bid_at_fill=bid,
                    ask_at_fill=ask,
                    spread_bps=(ask - bid) / ((ask + bid) / 2) * 10000,
                    market_impact_bps=impact_bps
                )
                
                fills.append(fill)
                remaining_size -= fill_size
                fill_counter += 1
                
                # Simulate fill latency
                await self._simulate_latency("fill")
                
                # Update market state (impact on subsequent fills)
                bid *= (1 - impact_bps / 20000)  # Half impact on bid
                ask *= (1 + impact_bps / 20000)  # Half impact on ask
                
                # Break if nearly complete or hit partial fill probability
                if remaining_size < 0.001 or (fill_counter > 1 and random.random() < self.partial_fill_probability):
                    break
            
            # Calculate execution metrics
            total_filled = sum(f.size for f in fills)
            avg_price = sum(f.price * f.size for f in fills) / total_filled if total_filled > 0 else 0
            total_fees = sum(f.fee for f in fills)
            
            # Determine order status
            if remaining_size < 0.001:
                status = OrderStatus.FILLED
            elif fills:
                status = OrderStatus.PARTIALLY_FILLED
            else:
                status = OrderStatus.REJECTED
            
            # Calculate performance metrics
            expected_price = ask if order_request.side == "buy" else bid
            realized_slippage = (avg_price - expected_price) / expected_price * 10000
            if order_request.side == "sell":
                realized_slippage = -realized_slippage
            
            result = OrderResult(
                order_id=order_request.order_id,
                request=order_request,
                status=status,
                fills=fills,
                total_filled_size=total_filled,
                avg_fill_price=avg_price,
                total_fees=total_fees,
                submit_time=order_request.timestamp,
                first_fill_time=fills[0].timestamp if fills else None,
                last_fill_time=fills[-1].timestamp if fills else None,
                completion_time=datetime.now(),
                expected_price=expected_price,
                realized_slippage_bps=realized_slippage,
                execution_shortfall_bps=realized_slippage,  # For market orders
                market_impact_bps=sum(f.market_impact_bps * f.size for f in fills) / total_filled if total_filled > 0 else 0,
                timing_cost_bps=0,  # No timing cost for market orders
                fill_rate=total_filled / order_request.size,
                speed_score=1.0,  # Market orders are fast
                cost_score=max(0, 1 - abs(realized_slippage) / 100),  # Cost score based on slippage
                overall_quality=0.8 if status == OrderStatus.FILLED else 0.5
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Market order execution failed: {e}")
            return self._create_error_order(order_request, str(e))
    
    async def _execute_limit_order(self, 
                                 order_request: OrderRequest,
                                 market_data: Dict[str, Any]) -> OrderResult:
        """Execute limit order with queue simulation"""
        try:
            pair_data = market_data.get(order_request.pair, {})
            current_price = pair_data.get("price", 0)
            bid = pair_data.get("bid", current_price * 0.999)
            ask = pair_data.get("ask", current_price * 1.001)
            
            limit_price = order_request.price
            fills = []
            
            # Check if limit order can be filled immediately (aggressive)
            can_fill_immediately = (
                (order_request.side == "buy" and limit_price >= ask) or
                (order_request.side == "sell" and limit_price <= bid)
            )
            
            if can_fill_immediately:
                # Aggressive limit order - execute like market order but at limit price
                fill_price = min(limit_price, ask) if order_request.side == "buy" else max(limit_price, bid)
                
                fill = Fill(
                    fill_id=f"{order_request.order_id}_fill_1",
                    timestamp=datetime.now(),
                    order_id=order_request.order_id,
                    size=order_request.size,
                    price=fill_price,
                    fee=self._calculate_fee(order_request.size * fill_price, FillType.TAKER),
                    fee_currency="USD",
                    fill_type=FillType.TAKER,
                    liquidity="taker",
                    bid_at_fill=bid,
                    ask_at_fill=ask,
                    spread_bps=(ask - bid) / ((ask + bid) / 2) * 10000,
                    market_impact_bps=self._calculate_market_impact(order_request.pair, order_request.size, order_request.side)
                )
                
                fills.append(fill)
                status = OrderStatus.FILLED
            else:
                # Passive limit order - simulate queue waiting
                fill_probability = self._calculate_fill_probability(
                    order_request.pair, limit_price, order_request.side, order_request.size
                )
                
                # Simulate waiting time
                wait_time_seconds = random.uniform(5, 300)  # 5 seconds to 5 minutes
                await asyncio.sleep(wait_time_seconds / 1000)  # Convert to actual wait in simulation
                
                if random.random() < fill_probability:
                    # Order filled as maker
                    fill = Fill(
                        fill_id=f"{order_request.order_id}_fill_1",
                        timestamp=datetime.now(),
                        order_id=order_request.order_id,
                        size=order_request.size,
                        price=limit_price,
                        fee=self._calculate_fee(order_request.size * limit_price, FillType.MAKER),
                        fee_currency="USD",
                        fill_type=FillType.MAKER,
                        liquidity="maker",
                        bid_at_fill=bid,
                        ask_at_fill=ask,
                        spread_bps=(ask - bid) / ((ask + bid) / 2) * 10000,
                        market_impact_bps=0  # No market impact for maker fills
                    )
                    
                    fills.append(fill)
                    status = OrderStatus.FILLED
                else:
                    # Order not filled - could be cancelled or expired
                    if random.random() < 0.3:  # 30% chance of cancellation
                        status = OrderStatus.CANCELLED
                    else:
                        status = OrderStatus.PARTIALLY_FILLED  # Simulate partial fill
                        
                        partial_size = order_request.size * random.uniform(0.1, 0.8)
                        fill = Fill(
                            fill_id=f"{order_request.order_id}_fill_1",
                            timestamp=datetime.now(),
                            order_id=order_request.order_id,
                            size=partial_size,
                            price=limit_price,
                            fee=self._calculate_fee(partial_size * limit_price, FillType.MAKER),
                            fee_currency="USD",
                            fill_type=FillType.MAKER,
                            liquidity="maker",
                            bid_at_fill=bid,
                            ask_at_fill=ask,
                            spread_bps=(ask - bid) / ((ask + bid) / 2) * 10000,
                            market_impact_bps=0
                        )
                        
                        fills.append(fill)
            
            # Calculate metrics
            total_filled = sum(f.size for f in fills)
            avg_price = sum(f.price * f.size for f in fills) / total_filled if total_filled > 0 else 0
            total_fees = sum(f.fee for f in fills)
            
            expected_price = limit_price
            realized_slippage = (avg_price - expected_price) / expected_price * 10000 if expected_price > 0 else 0
            if order_request.side == "sell":
                realized_slippage = -realized_slippage
            
            result = OrderResult(
                order_id=order_request.order_id,
                request=order_request,
                status=status,
                fills=fills,
                total_filled_size=total_filled,
                avg_fill_price=avg_price,
                total_fees=total_fees,
                submit_time=order_request.timestamp,
                first_fill_time=fills[0].timestamp if fills else None,
                last_fill_time=fills[-1].timestamp if fills else None,
                completion_time=datetime.now(),
                expected_price=expected_price,
                realized_slippage_bps=realized_slippage,
                execution_shortfall_bps=0 if can_fill_immediately else realized_slippage,
                market_impact_bps=sum(f.market_impact_bps * f.size for f in fills) / total_filled if total_filled > 0 else 0,
                timing_cost_bps=0,
                fill_rate=total_filled / order_request.size,
                speed_score=0.8 if can_fill_immediately else 0.4,
                cost_score=0.9 if not can_fill_immediately else 0.7,  # Maker fills are cheaper
                overall_quality=0.9 if status == OrderStatus.FILLED else 0.3
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Limit order execution failed: {e}")
            return self._create_error_order(order_request, str(e))
    
    async def _execute_post_only_order(self, 
                                     order_request: OrderRequest,
                                     market_data: Dict[str, Any]) -> OrderResult:
        """Execute post-only order (guaranteed maker)"""
        try:
            pair_data = market_data.get(order_request.pair, {})
            current_price = pair_data.get("price", 0)
            bid = pair_data.get("bid", current_price * 0.999)
            ask = pair_data.get("ask", current_price * 1.001)
            
            limit_price = order_request.price
            
            # Check if post-only order would cross spread
            would_cross_spread = (
                (order_request.side == "buy" and limit_price >= ask) or
                (order_request.side == "sell" and limit_price <= bid)
            )
            
            if would_cross_spread:
                # Post-only order rejected to prevent taking liquidity
                return OrderResult(
                    order_id=order_request.order_id,
                    request=order_request,
                    status=OrderStatus.REJECTED,
                    fills=[],
                    total_filled_size=0,
                    avg_fill_price=0,
                    total_fees=0,
                    submit_time=order_request.timestamp,
                    first_fill_time=None,
                    last_fill_time=None,
                    completion_time=datetime.now(),
                    expected_price=limit_price,
                    realized_slippage_bps=0,
                    execution_shortfall_bps=float('inf'),  # Complete shortfall
                    market_impact_bps=0,
                    timing_cost_bps=0,
                    fill_rate=0,
                    speed_score=0,
                    cost_score=0,
                    overall_quality=0
                )
            
            # Execute as passive limit order with higher maker probability
            fill_probability = self._calculate_fill_probability(
                order_request.pair, limit_price, order_request.side, order_request.size
            ) * 1.2  # 20% boost for post-only
            
            fills = []
            if random.random() < min(fill_probability, 0.9):  # Max 90% fill rate
                fill = Fill(
                    fill_id=f"{order_request.order_id}_fill_1",
                    timestamp=datetime.now(),
                    order_id=order_request.order_id,
                    size=order_request.size,
                    price=limit_price,
                    fee=self._calculate_fee(order_request.size * limit_price, FillType.MAKER),
                    fee_currency="USD",
                    fill_type=FillType.MAKER,
                    liquidity="maker",
                    bid_at_fill=bid,
                    ask_at_fill=ask,
                    spread_bps=(ask - bid) / ((ask + bid) / 2) * 10000,
                    market_impact_bps=0  # No impact for maker fills
                )
                
                fills.append(fill)
                status = OrderStatus.FILLED
            else:
                status = OrderStatus.CANCELLED  # Not filled, eventually cancelled
            
            # Calculate metrics
            total_filled = sum(f.size for f in fills)
            avg_price = sum(f.price * f.size for f in fills) / total_filled if total_filled > 0 else 0
            total_fees = sum(f.fee for f in fills)
            
            result = OrderResult(
                order_id=order_request.order_id,
                request=order_request,
                status=status,
                fills=fills,
                total_filled_size=total_filled,
                avg_fill_price=avg_price,
                total_fees=total_fees,
                submit_time=order_request.timestamp,
                first_fill_time=fills[0].timestamp if fills else None,
                last_fill_time=fills[-1].timestamp if fills else None,
                completion_time=datetime.now(),
                expected_price=limit_price,
                realized_slippage_bps=0,  # No slippage at exact limit price
                execution_shortfall_bps=0 if status == OrderStatus.FILLED else float('inf'),
                market_impact_bps=0,
                timing_cost_bps=0,
                fill_rate=total_filled / order_request.size,
                speed_score=0.3,  # Post-only is slower
                cost_score=1.0 if fills else 0,  # Best cost when filled
                overall_quality=0.95 if status == OrderStatus.FILLED else 0.1
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Post-only order execution failed: {e}")
            return self._create_error_order(order_request, str(e))
    
    async def _execute_advanced_order(self, 
                                    order_request: OrderRequest,
                                    market_data: Dict[str, Any]) -> OrderResult:
        """Execute advanced order types (stop orders, etc.)"""
        # Simplified implementation - would be more sophisticated in practice
        logger.warning(f"Advanced order type {order_request.order_type} not fully implemented")
        return await self._execute_market_order(order_request, market_data)
    
    async def _simulate_latency(self, operation_type: str) -> None:
        """Simulate network and processing latency"""
        try:
            base_latency = self.base_latency_ms
            if operation_type == "submission":
                base_latency += self.queue_processing_ms
            elif operation_type == "fill":
                base_latency += random.uniform(5, 15)  # Fill processing variance
            
            # Add variance
            total_latency = base_latency + random.uniform(-self.latency_variance_ms, self.latency_variance_ms)
            total_latency = max(1, total_latency)  # Minimum 1ms
            
            # Convert to seconds for asyncio.sleep (scaled down for simulation)
            await asyncio.sleep(total_latency / 10000)  # Scale down for simulation
            
        except Exception as e:
            logger.error(f"Latency simulation failed: {e}")
    
    def _validate_order(self, order_request: OrderRequest, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate order request"""
        try:
            if order_request.size <= 0:
                return {"valid": False, "reason": "Invalid size"}
            
            if order_request.pair not in market_data:
                return {"valid": False, "reason": "Unknown trading pair"}
            
            pair_data = market_data.get(order_request.pair, {})
            current_price = pair_data.get("price", 0)
            
            if current_price <= 0:
                return {"valid": False, "reason": "No market price available"}
            
            # Check if limit price is reasonable
            if order_request.price:
                price_deviation = abs(order_request.price - current_price) / current_price
                if price_deviation > 0.5:  # More than 50% from market
                    return {"valid": False, "reason": "Limit price too far from market"}
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "reason": f"Validation error: {e}"}
    
    def _calculate_market_impact(self, pair: str, size: float, side: str) -> float:
        """Calculate market impact in basis points"""
        try:
            # Simplified market impact model
            # Real implementation would use order book depth and historical impact
            
            # Base impact proportional to size
            base_impact = size * 0.1  # 0.1 bps per unit size
            
            # Market stress multiplier
            stress_adjusted_impact = base_impact * self.market_stress_factor
            
            # Pair-specific liquidity adjustment
            liquidity_factor = self._get_liquidity_factor(pair)
            final_impact = stress_adjusted_impact / liquidity_factor
            
            # Cap impact at reasonable levels
            return min(final_impact, 200)  # Max 200 bps impact
            
        except Exception as e:
            logger.error(f"Market impact calculation failed: {e}")
            return 10  # Default 10 bps
    
    def _get_available_liquidity(self, pair: str, price: float, side: str) -> float:
        """Get available liquidity at price level"""
        try:
            # Simplified liquidity model
            base_liquidity = 1000  # Base liquidity amount
            
            # Apply liquidity factor for pair
            liquidity_factor = self._get_liquidity_factor(pair)
            available = base_liquidity * liquidity_factor
            
            # Add some randomness
            return available * random.uniform(0.5, 1.5)
            
        except Exception as e:
            logger.error(f"Liquidity calculation failed: {e}")
            return 100  # Default liquidity
    
    def _get_liquidity_factor(self, pair: str) -> float:
        """Get liquidity factor for pair"""
        # Simplified - would use actual market data in practice
        if "BTC" in pair:
            return 2.0  # High liquidity for BTC pairs
        elif "ETH" in pair:
            return 1.5  # Medium-high liquidity
        else:
            return 1.0  # Default liquidity
    
    def _calculate_fill_probability(self, pair: str, price: float, side: str, size: float) -> float:
        """Calculate probability of limit order fill"""
        try:
            # Simplified fill probability model
            # Real implementation would use historical fill rates and market microstructure
            
            base_probability = 0.6  # 60% base fill rate
            
            # Adjust for price aggressiveness (closer to market = higher probability)
            # This would require current bid/ask which we simplify here
            aggressiveness_factor = 1.0  # Simplified
            
            # Adjust for size (larger orders less likely to fill completely)
            size_factor = max(0.1, 1.0 - size * 0.0001)
            
            # Adjust for pair liquidity
            liquidity_factor = min(1.2, self._get_liquidity_factor(pair))
            
            final_probability = base_probability * aggressiveness_factor * size_factor * liquidity_factor
            
            return min(final_probability, 0.95)  # Max 95% fill probability
            
        except Exception as e:
            logger.error(f"Fill probability calculation failed: {e}")
            return 0.5  # Default 50%
    
    def _calculate_fee(self, notional_value: float, fill_type: FillType) -> float:
        """Calculate trading fees"""
        try:
            # Determine base fee rate
            if fill_type == FillType.MAKER:
                base_fee_bps = self.maker_fee_bps
            else:
                base_fee_bps = self.taker_fee_bps
            
            # Apply volume discounts (simplified)
            discount_factor = 1.0
            monthly_volume = sum(self.recent_volume.values())  # Simplified
            
            for volume_threshold, discount in sorted(self.fee_tier_discounts.items()):
                if monthly_volume >= volume_threshold:
                    discount_factor = discount
                    break
            
            effective_fee_bps = base_fee_bps * discount_factor
            
            return notional_value * effective_fee_bps / 10000
            
        except Exception as e:
            logger.error(f"Fee calculation failed: {e}")
            return notional_value * 0.004  # Default 0.4%
    
    def _create_rejected_order(self, order_request: OrderRequest, reason: str) -> OrderResult:
        """Create rejected order result"""
        return OrderResult(
            order_id=order_request.order_id,
            request=order_request,
            status=OrderStatus.REJECTED,
            fills=[],
            total_filled_size=0,
            avg_fill_price=0,
            total_fees=0,
            submit_time=order_request.timestamp,
            first_fill_time=None,
            last_fill_time=None,
            completion_time=datetime.now(),
            expected_price=0,
            realized_slippage_bps=0,
            execution_shortfall_bps=float('inf'),
            market_impact_bps=0,
            timing_cost_bps=0,
            fill_rate=0,
            speed_score=0,
            cost_score=0,
            overall_quality=0
        )
    
    def _create_error_order(self, order_request: OrderRequest, error: str) -> OrderResult:
        """Create error order result"""
        logger.error(f"Order error: {error}")
        return self._create_rejected_order(order_request, f"System error: {error}")
    
    def get_execution_metrics(self, days_back: int = 30) -> ExecutionMetrics:
        """Calculate execution performance metrics"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            recent_orders = [
                order for order in self.order_history
                if order.submit_time >= cutoff_time
            ]
            
            if not recent_orders:
                return ExecutionMetrics(
                    total_orders=0,
                    filled_orders=0,
                    partially_filled_orders=0,
                    cancelled_orders=0,
                    rejected_orders=0,
                    fill_rate=0.0,
                    avg_fill_time_seconds=0.0,
                    avg_slippage_bps=0.0,
                    avg_market_impact_bps=0.0,
                    total_fees=0.0,
                    maker_fill_rate=0.0,
                    taker_fill_rate=0.0,
                    avg_effective_spread_bps=0.0,
                    avg_execution_quality=0.0,
                    speed_percentile_90=0.0,
                    cost_percentile_90=0.0
                )
            
            # Count order statuses
            filled_orders = sum(1 for o in recent_orders if o.status == OrderStatus.FILLED)
            partially_filled = sum(1 for o in recent_orders if o.status == OrderStatus.PARTIALLY_FILLED)
            cancelled_orders = sum(1 for o in recent_orders if o.status == OrderStatus.CANCELLED)
            rejected_orders = sum(1 for o in recent_orders if o.status == OrderStatus.REJECTED)
            
            # Calculate metrics
            total_orders = len(recent_orders)
            fill_rate = (filled_orders + partially_filled) / total_orders if total_orders > 0 else 0
            
            # Execution times
            fill_times = [o.execution_time_seconds for o in recent_orders if o.execution_time_seconds > 0]
            avg_fill_time = np.mean(fill_times) if fill_times else 0
            
            # Slippage and impact
            slippages = [o.realized_slippage_bps for o in recent_orders if o.fills]
            market_impacts = [o.market_impact_bps for o in recent_orders if o.fills]
            
            avg_slippage = np.mean(slippages) if slippages else 0
            avg_impact = np.mean(market_impacts) if market_impacts else 0
            
            # Fee analysis
            total_fees = sum(o.total_fees for o in recent_orders)
            
            maker_fills = sum(len([f for f in o.fills if f.fill_type == FillType.MAKER]) for o in recent_orders)
            taker_fills = sum(len([f for f in o.fills if f.fill_type == FillType.TAKER]) for o in recent_orders)
            total_fills = maker_fills + taker_fills
            
            maker_rate = maker_fills / total_fills if total_fills > 0 else 0
            taker_rate = taker_fills / total_fills if total_fills > 0 else 0
            
            # Spread analysis
            spreads = []
            for order in recent_orders:
                for fill in order.fills:
                    spreads.append(fill.spread_bps)
            
            avg_spread = np.mean(spreads) if spreads else 0
            
            # Quality scores
            quality_scores = [o.overall_quality for o in recent_orders]
            avg_quality = np.mean(quality_scores) if quality_scores else 0
            
            # Percentiles
            speed_scores = [o.speed_score for o in recent_orders]
            cost_scores = [o.cost_score for o in recent_orders]
            
            speed_90th = np.percentile(speed_scores, 90) if speed_scores else 0
            cost_90th = np.percentile(cost_scores, 90) if cost_scores else 0
            
            return ExecutionMetrics(
                total_orders=total_orders,
                filled_orders=filled_orders,
                partially_filled_orders=partially_filled,
                cancelled_orders=cancelled_orders,
                rejected_orders=rejected_orders,
                fill_rate=fill_rate,
                avg_fill_time_seconds=avg_fill_time,
                avg_slippage_bps=avg_slippage,
                avg_market_impact_bps=avg_impact,
                total_fees=total_fees,
                maker_fill_rate=maker_rate,
                taker_fill_rate=taker_rate,
                avg_effective_spread_bps=avg_spread,
                avg_execution_quality=avg_quality,
                speed_percentile_90=speed_90th,
                cost_percentile_90=cost_90th
            )
            
        except Exception as e:
            logger.error(f"Execution metrics calculation failed: {e}")
            return ExecutionMetrics(
                total_orders=0,
                filled_orders=0,
                partially_filled_orders=0,
                cancelled_orders=0,
                rejected_orders=0,
                fill_rate=0.0,
                avg_fill_time_seconds=0.0,
                avg_slippage_bps=0.0,
                avg_market_impact_bps=0.0,
                total_fees=0.0,
                maker_fill_rate=0.0,
                taker_fill_rate=0.0,
                avg_effective_spread_bps=0.0,
                avg_execution_quality=0.0,
                speed_percentile_90=0.0,
                cost_percentile_90=0.0
            )