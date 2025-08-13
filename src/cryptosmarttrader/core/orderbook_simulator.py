#!/usr/bin/env python3
"""
L2 Orderbook Simulator
Simulates Level-2 order book depth, partial fills, fees, Time-in-Force, and latency
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import warnings
warnings.filterwarnings('ignore')

# Import core components
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.structured_logger import get_structured_logger

class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class TimeInForce(Enum):
    """Time-in-Force enumeration"""
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill
    DAY = "day"  # Day order

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class OrderBookLevel:
    """Single level in order book"""
    price: float
    volume: float
    order_count: int

@dataclass
class OrderBookSnapshot:
    """L2 order book snapshot"""
    timestamp: datetime
    bids: List[OrderBookLevel]  # Sorted by price descending
    asks: List[OrderBookLevel]  # Sorted by price ascending
    exchange: str = "simulated"

@dataclass
class Order:
    """Trading order"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # None for market orders
    time_in_force: TimeInForce = TimeInForce.GTC
    created_at: datetime = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    average_fill_price: float = 0.0
    fees_paid: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.remaining_quantity == 0.0:
            self.remaining_quantity = self.quantity

@dataclass
class Fill:
    """Order fill/execution"""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    price: float
    quantity: float
    fee: float
    timestamp: datetime
    liquidity_flag: str  # 'maker' or 'taker'

@dataclass
class ExchangeConfig:
    """Exchange-specific configuration"""
    name: str
    maker_fee: float = 0.001  # 0.1%
    taker_fee: float = 0.0015  # 0.15%
    min_order_size: float = 0.001
    price_precision: int = 8
    quantity_precision: int = 8
    latency_mean_ms: float = 50.0
    latency_std_ms: float = 20.0
    max_latency_ms: float = 500.0

class OrderBookSimulator:
    """L2 order book simulator with realistic market microstructure"""
    
    def __init__(self, symbol: str, exchange_config: Optional[ExchangeConfig] = None):
        self.symbol = symbol
        self.logger = get_structured_logger("OrderBookSimulator")
        
        # Exchange configuration
        self.exchange_config = exchange_config or ExchangeConfig("simulated")
        
        # Current order book state
        self.current_book: Optional[OrderBookSnapshot] = None
        self.bid_price = 0.0
        self.ask_price = 0.0
        self.mid_price = 0.0
        self.spread = 0.0
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.order_counter = 0
        
        # Market impact parameters
        self.impact_params = {
            'linear_impact': 0.001,  # Price impact per unit volume
            'sqrt_impact': 0.0005,   # Square root impact
            'permanent_impact': 0.3, # Fraction of impact that's permanent
            'recovery_half_life': 30.0  # Seconds for impact recovery
        }
        
        # Historical data for realistic simulation
        self.price_history: List[float] = []
        self.volume_history: List[float] = []
        
    def update_orderbook(self, snapshot: OrderBookSnapshot) -> None:
        """Update order book with new snapshot"""
        
        self.current_book = snapshot
        
        if snapshot.bids and snapshot.asks:
            self.bid_price = snapshot.bids[0].price
            self.ask_price = snapshot.asks[0].price
            self.mid_price = (self.bid_price + self.ask_price) / 2
            self.spread = self.ask_price - self.bid_price
            
            self.logger.debug(f"Updated order book: bid={self.bid_price:.6f}, "
                            f"ask={self.ask_price:.6f}, spread={self.spread:.6f}")
    
    def generate_realistic_orderbook(self, base_price: float, volatility: float = 0.01,
                                   depth_levels: int = 20) -> OrderBookSnapshot:
        """Generate realistic L2 order book snapshot"""
        
        try:
            # Calculate spread based on volatility
            spread_bps = max(1, int(volatility * 10000))  # Min 1 bps
            spread = base_price * spread_bps / 10000
            
            bid_price = base_price - spread / 2
            ask_price = base_price + spread / 2
            
            # Generate bid levels (descending price)
            bids = []
            for i in range(depth_levels):
                level_price = bid_price - (i * spread * 0.1)
                
                # Volume decreases with distance from best price
                base_volume = np.random.exponential(100) * (0.8 ** i)
                volume = max(0.01, base_volume)
                
                # Order count roughly proportional to volume
                order_count = max(1, int(volume / 10) + np.random.poisson(2))
                
                bids.append(OrderBookLevel(
                    price=round(level_price, self.exchange_config.price_precision),
                    volume=round(volume, self.exchange_config.quantity_precision),
                    order_count=order_count
                ))
            
            # Generate ask levels (ascending price)
            asks = []
            for i in range(depth_levels):
                level_price = ask_price + (i * spread * 0.1)
                
                # Volume decreases with distance from best price
                base_volume = np.random.exponential(100) * (0.8 ** i)
                volume = max(0.01, base_volume)
                
                # Order count roughly proportional to volume
                order_count = max(1, int(volume / 10) + np.random.poisson(2))
                
                asks.append(OrderBookLevel(
                    price=round(level_price, self.exchange_config.price_precision),
                    volume=round(volume, self.exchange_config.quantity_precision),
                    order_count=order_count
                ))
            
            snapshot = OrderBookSnapshot(
                timestamp=datetime.now(),
                bids=bids,
                asks=asks,
                exchange=self.exchange_config.name
            )
            
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to generate order book: {e}")
            # Return minimal order book
            return OrderBookSnapshot(
                timestamp=datetime.now(),
                bids=[OrderBookLevel(base_price * 0.999, 1.0, 1)],
                asks=[OrderBookLevel(base_price * 1.001, 1.0, 1)],
                exchange=self.exchange_config.name
            )
    
    def submit_order(self, side: OrderSide, order_type: OrderType, quantity: float,
                    price: Optional[float] = None, time_in_force: TimeInForce = TimeInForce.GTC) -> str:
        """Submit trading order to simulator"""
        
        self.order_counter += 1
        order_id = f"order_{self.symbol}_{self.order_counter}_{int(datetime.now().timestamp())}"
        
        order = Order(
            order_id=order_id,
            symbol=self.symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            created_at=datetime.now()
        )
        
        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            self.logger.warning(f"Order rejected: {order_id}")
            return order_id
        
        # Store order
        self.orders[order_id] = order
        
        # Process order immediately for market orders or IOC/FOK
        if order_type == OrderType.MARKET or time_in_force in [TimeInForce.IOC, TimeInForce.FOK]:
            self._process_order_immediately(order)
        
        self.logger.info(f"Order submitted: {order_id} - {side.value} {quantity} {self.symbol} "
                        f"at {price if price else 'market'}")
        
        return order_id
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order parameters"""
        
        try:
            # Check minimum order size
            if order.quantity < self.exchange_config.min_order_size:
                self.logger.warning(f"Order size {order.quantity} below minimum "
                                  f"{self.exchange_config.min_order_size}")
                return False
            
            # Check price precision
            if order.price is not None:
                price_decimals = str(order.price)[::-1].find('.')
                if price_decimals > self.exchange_config.price_precision:
                    self.logger.warning(f"Price precision {price_decimals} exceeds maximum "
                                      f"{self.exchange_config.price_precision}")
                    return False
            
            # Check quantity precision
            qty_decimals = str(order.quantity)[::-1].find('.')
            if qty_decimals > self.exchange_config.quantity_precision:
                self.logger.warning(f"Quantity precision {qty_decimals} exceeds maximum "
                                  f"{self.exchange_config.quantity_precision}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Order validation failed: {e}")
            return False
    
    def _process_order_immediately(self, order: Order) -> None:
        """Process order immediately (market orders, IOC, FOK)"""
        
        if not self.current_book:
            order.status = OrderStatus.REJECTED
            self.logger.warning(f"No order book available for order {order.order_id}")
            return
        
        try:
            # Add latency simulation
            latency_ms = self._# REMOVED: Mock data pattern not allowed in production)
            
            # Determine which side of book to match against
            book_levels = (self.current_book.asks if order.side == OrderSide.BUY 
                         else self.current_book.bids)
            
            if not book_levels:
                order.status = OrderStatus.REJECTED
                self.logger.warning(f"No liquidity available for order {order.order_id}")
                return
            
            # Execute fills against order book
            remaining_qty = order.remaining_quantity
            total_filled = 0.0
            total_cost = 0.0
            
            for level in book_levels:
                if remaining_qty <= 0:
                    break
                
                # Check price limit for limit orders
                if order.price is not None:
                    if (order.side == OrderSide.BUY and level.price > order.price) or \
                       (order.side == OrderSide.SELL and level.price < order.price):
                        break
                
                # Calculate fill quantity (limited by available volume)
                fill_qty = min(remaining_qty, level.volume)
                
                # Apply market impact
                impact_adjusted_price = self._apply_market_impact(level.price, fill_qty, order.side)
                
                # Calculate fees
                fee = self._calculate_fee(fill_qty * impact_adjusted_price, is_taker=True)
                
                # Create fill
                fill = Fill(
                    fill_id=f"fill_{order.order_id}_{len(self.fills)+1}",
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    price=impact_adjusted_price,
                    quantity=fill_qty,
                    fee=fee,
                    timestamp=datetime.now() + timedelta(milliseconds=latency_ms),
                    liquidity_flag="taker"
                )
                
                self.fills.append(fill)
                
                # Update order
                remaining_qty -= fill_qty
                total_filled += fill_qty
                total_cost += fill_qty * impact_adjusted_price
                order.fees_paid += fee
                
                self.logger.debug(f"Fill executed: {fill_qty} at {impact_adjusted_price:.6f}")
            
            # Update order status
            order.filled_quantity = total_filled
            order.remaining_quantity = remaining_qty
            
            if total_filled > 0:
                order.average_fill_price = total_cost / total_filled
            
            if remaining_qty <= 0:
                order.status = OrderStatus.FILLED
            elif total_filled > 0:
                order.status = OrderStatus.PARTIAL
            else:
                order.status = OrderStatus.CANCELLED  # No fills
            
            # Handle FOK orders
            if order.time_in_force == TimeInForce.FOK and order.status != OrderStatus.FILLED:
                order.status = OrderStatus.CANCELLED
                # Would need to reverse fills in real implementation
                self.logger.info(f"FOK order {order.order_id} cancelled - partial fill not allowed")
            
            self.logger.info(f"Order processed: {order.order_id} - {order.status.value}, "
                           f"filled {order.filled_quantity}/{order.quantity}")
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            self.logger.error(f"Order processing failed for {order.order_id}: {e}")
    
    def _# REMOVED: Mock data pattern not allowed in productionself) -> float:
        """Simulate exchange latency"""
        
        # Generate realistic latency with occasional spikes
        if np.random.random() < 0.05:  # 5% chance of high latency
            latency = np.random.exponential(self.exchange_config.latency_mean_ms * 5)
        else:
            latency = np.# REMOVED: Mock data pattern not allowed in production(self.exchange_config.latency_mean_ms, 
                                     self.exchange_config.latency_std_ms)
        
        # Cap at maximum latency
        latency = min(latency, self.exchange_config.max_latency_ms)
        latency = max(latency, 1.0)  # Minimum 1ms
        
        return latency
    
    def _apply_market_impact(self, base_price: float, quantity: float, side: OrderSide) -> float:
        """Apply market impact to execution price"""
        
        try:
            # Calculate impact based on quantity and recent volume
            recent_volume = np.mean(self.volume_history[-10:]) if self.volume_history else 1000.0
            relative_size = quantity / max(recent_volume, 1.0)
            
            # Linear + square root impact model
            linear_impact = self.impact_params['linear_impact'] * relative_size
            sqrt_impact = self.impact_params['sqrt_impact'] * np.sqrt(relative_size)
            total_impact = linear_impact + sqrt_impact
            
            # Apply impact direction based on order side
            if side == OrderSide.BUY:
                impacted_price = base_price * (1 + total_impact)
            else:
                impacted_price = base_price * (1 - total_impact)
            
            return round(impacted_price, self.exchange_config.price_precision)
            
        except Exception as e:
            self.logger.error(f"Market impact calculation failed: {e}")
            return base_price
    
    def _calculate_fee(self, notional_value: float, is_taker: bool = True) -> float:
        """Calculate trading fees"""
        
        fee_rate = (self.exchange_config.taker_fee if is_taker 
                   else self.exchange_config.maker_fee)
        
        return notional_value * fee_rate
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        
        if order_id not in self.orders:
            self.logger.warning(f"Order not found for cancellation: {order_id}")
            return False
        
        order = self.orders[order_id]
        
        if order.status not in [OrderStatus.PENDING, OrderStatus.PARTIAL]:
            self.logger.warning(f"Cannot cancel order {order_id} with status {order.status.value}")
            return False
        
        order.status = OrderStatus.CANCELLED
        self.logger.info(f"Order cancelled: {order_id}")
        
        return True
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order status and details"""
        
        if order_id not in self.orders:
            return None
        
        order = self.orders[order_id]
        order_fills = [fill for fill in self.fills if fill.order_id == order_id]
        
        return {
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'order_type': order.order_type.value,
            'quantity': order.quantity,
            'price': order.price,
            'time_in_force': order.time_in_force.value,
            'status': order.status.value,
            'filled_quantity': order.filled_quantity,
            'remaining_quantity': order.remaining_quantity,
            'average_fill_price': order.average_fill_price,
            'fees_paid': order.fees_paid,
            'created_at': order.created_at.isoformat(),
            'fills': len(order_fills)
        }
    
    def get_fill_history(self, order_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get fill history, optionally filtered by order"""
        
        fills = [fill for fill in self.fills if order_id is None or fill.order_id == order_id]
        
        return [
            {
                'fill_id': fill.fill_id,
                'order_id': fill.order_id,
                'symbol': fill.symbol,
                'side': fill.side.value,
                'price': fill.price,
                'quantity': fill.quantity,
                'fee': fill.fee,
                'timestamp': fill.timestamp.isoformat(),
                'liquidity_flag': fill.liquidity_flag
            }
            for fill in fills
        ]
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get market and simulator summary"""
        
        if not self.current_book:
            return {'error': 'No order book data available'}
        
        # Calculate order book statistics
        total_bid_volume = sum(level.volume for level in self.current_book.bids[:10])
        total_ask_volume = sum(level.volume for level in self.current_book.asks[:10])
        
        # Order statistics
        total_orders = len(self.orders)
        filled_orders = len([o for o in self.orders.values() if o.status == OrderStatus.FILLED])
        partial_orders = len([o for o in self.orders.values() if o.status == OrderStatus.PARTIAL])
        
        return {
            'symbol': self.symbol,
            'exchange': self.exchange_config.name,
            'timestamp': self.current_book.timestamp.isoformat(),
            'market_data': {
                'bid_price': self.bid_price,
                'ask_price': self.ask_price,
                'mid_price': self.mid_price,
                'spread': self.spread,
                'spread_bps': (self.spread / self.mid_price * 10000) if self.mid_price > 0 else 0,
                'bid_volume_top10': total_bid_volume,
                'ask_volume_top10': total_ask_volume
            },
            'trading_stats': {
                'total_orders': total_orders,
                'filled_orders': filled_orders,
                'partial_orders': partial_orders,
                'total_fills': len(self.fills),
                'fill_rate': filled_orders / max(total_orders, 1) * 100
            },
            'exchange_config': {
                'maker_fee': self.exchange_config.maker_fee,
                'taker_fee': self.exchange_config.taker_fee,
                'latency_mean_ms': self.exchange_config.latency_mean_ms
            }
        }

if __name__ == "__main__":
    async def test_orderbook_simulator():
        """Test order book simulator"""
        
        print("🔍 TESTING L2 ORDERBOOK SIMULATOR")
        print("=" * 60)
        
        # Create simulator
        exchange_config = ExchangeConfig(
            name="test_exchange",
            maker_fee=0.001,
            taker_fee=0.0015,
            latency_mean_ms=25.0
        )
        
        simulator = OrderBookSimulator("BTC/USD", exchange_config)
        
        print("📊 Generating realistic order book...")
        
        # Generate realistic order book
        orderbook = simulator.generate_realistic_orderbook(50000.0, volatility=0.005)
        simulator.update_orderbook(orderbook)
        
        market_summary = simulator.get_market_summary()
        print(f"   Bid: ${market_summary['market_data']['bid_price']:.2f}")
        print(f"   Ask: ${market_summary['market_data']['ask_price']:.2f}")
        print(f"   Spread: {market_summary['market_data']['spread_bps']:.1f} bps")
        
        print("\n📝 Testing order submission and execution...")
        
        # Test market buy order
        buy_order_id = simulator.submit_order(
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=0.1
        )
        
        # Test limit sell order
        sell_order_id = simulator.submit_order(
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=0.05,
            price=51000.0,
            time_in_force=TimeInForce.GTC
        )
        
        # Test IOC order
        ioc_order_id = simulator.submit_order(
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=0.2,
            price=49500.0,
            time_in_force=TimeInForce.IOC
        )
        
        print(f"   Submitted 3 test orders")
        
        # Check order statuses
        print("\n📋 Order execution results:")
        
        for order_id in [buy_order_id, sell_order_id, ioc_order_id]:
            status = simulator.get_order_status(order_id)
            if status:
                print(f"   {order_id}: {status['status']} - "
                      f"{status['filled_quantity']:.6f}/{status['quantity']:.6f}")
                if status['filled_quantity'] > 0:
                    print(f"      Avg price: ${status['average_fill_price']:.2f}, "
                          f"Fees: ${status['fees_paid']:.4f}")
        
        # Check fills
        fills = simulator.get_fill_history()
        print(f"\n💸 Total fills executed: {len(fills)}")
        
        for fill in fills:
            print(f"   {fill['side']} {fill['quantity']:.6f} at ${fill['price']:.2f} "
                  f"(fee: ${fill['fee']:.4f})")
        
        # Test order cancellation
        print("\n❌ Testing order cancellation...")
        cancelled = simulator.cancel_order(sell_order_id)
        print(f"   Cancellation result: {cancelled}")
        
        # Final summary
        print("\n📈 Final simulator summary:")
        final_summary = simulator.get_market_summary()
        trading_stats = final_summary['trading_stats']
        
        print(f"   Total orders: {trading_stats['total_orders']}")
        print(f"   Fill rate: {trading_stats['fill_rate']:.1f}%")
        print(f"   Total fills: {trading_stats['total_fills']}")
        
        print("\n✅ L2 ORDERBOOK SIMULATOR TEST COMPLETED")
        
        return len(fills) > 0 and trading_stats['fill_rate'] > 0
    
    # Run test
    import asyncio
    success = asyncio.run(test_orderbook_simulator())
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")