"""
Execution Simulator for CryptoSmartTrader
Advanced simulation of real-world execution conditions for backtest-live parity.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import random
import logging


class OrderType(Enum):
    """Order type classifications."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side classifications."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class OrderBook:
    """Order book state for simulation."""
    symbol: str
    timestamp: datetime
    bids: List[Tuple[float, float]]  # (price, quantity)
    asks: List[Tuple[float, float]]  # (price, quantity)
    mid_price: float
    spread_bps: float
    depth_usd: float


@dataclass
class ExecutionResult:
    """Result of order execution simulation."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    requested_quantity: float
    executed_quantity: float
    avg_fill_price: float
    total_fees: float
    slippage_bps: float
    latency_ms: float
    partial_fill: bool
    execution_time: datetime
    market_impact_bps: float
    queue_position: Optional[int]
    fills: List[Dict[str, Any]]
    metadata: Dict[str, Any]


@dataclass
class SimulationConfig:
    """Configuration for execution simulation."""
    # Fee structure
    maker_fee_bps: float = 10.0  # 0.1% maker fee
    taker_fee_bps: float = 25.0  # 0.25% taker fee
    
    # Latency modeling
    min_latency_ms: float = 50.0
    max_latency_ms: float = 500.0
    latency_volatility: float = 0.3
    
    # Slippage modeling
    base_slippage_bps: float = 5.0
    slippage_impact_factor: float = 0.1
    max_slippage_bps: float = 100.0
    
    # Partial fill modeling
    partial_fill_probability: float = 0.15
    min_fill_ratio: float = 0.5
    
    # Market impact
    impact_decay_halflife: float = 300.0  # seconds
    impact_per_volume_pct: float = 50.0  # bps per 1% of daily volume
    
    # Queue modeling
    queue_jump_probability: float = 0.05
    queue_decay_rate: float = 0.1


class ExecutionSimulator:
    """
    Enterprise execution simulator for backtest-live parity validation.
    
    Features:
    - Realistic order book modeling
    - Fee structure simulation (maker/taker)
    - Latency and queue position modeling
    - Partial fill simulation
    - Market impact calculation
    - Slippage estimation
    - Real-world execution constraints
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        
        # Simulation state
        self.order_books: Dict[str, OrderBook] = {}
        self.market_impact_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.execution_history: List[ExecutionResult] = []
        
        # Statistics tracking
        self.total_executed_volume = 0.0
        self.avg_slippage = 0.0
        self.fill_rate = 0.0
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("ExecutionSimulator initialized with realistic modeling")
    
    def simulate_order_execution(self,
                                order_id: str,
                                symbol: str,
                                side: OrderSide,
                                order_type: OrderType,
                                quantity: float,
                                limit_price: Optional[float] = None,
                                market_data: Optional[pd.DataFrame] = None,
                                volume_data: Optional[pd.Series] = None) -> ExecutionResult:
        """
        Simulate order execution with realistic market conditions.
        
        Args:
            order_id: Unique order identifier
            symbol: Trading symbol
            side: Order side (buy/sell)
            order_type: Order type
            quantity: Requested quantity
            limit_price: Limit price (if applicable)
            market_data: Recent market data for simulation
            volume_data: Volume data for impact calculation
            
        Returns:
            ExecutionResult with detailed execution simulation
        """
        execution_time = datetime.utcnow()
        
        # Generate or update order book
        order_book = self._generate_order_book(symbol, market_data, execution_time)
        
        # Simulate latency
        latency_ms = self._simulate_latency()
        
        # Calculate market impact
        daily_volume = volume_data.mean() if volume_data is not None else 1000000
        market_impact_bps = self._calculate_market_impact(symbol, quantity, daily_volume)
        
        # Simulate execution based on order type
        if order_type == OrderType.MARKET:
            result = self._simulate_market_order(
                order_id, symbol, side, quantity, order_book, 
                latency_ms, market_impact_bps, execution_time
            )
        elif order_type == OrderType.LIMIT:
            result = self._simulate_limit_order(
                order_id, symbol, side, quantity, limit_price,
                order_book, latency_ms, market_impact_bps, execution_time
            )
        else:
            # For now, treat other order types as limit orders
            result = self._simulate_limit_order(
                order_id, symbol, side, quantity, limit_price,
                order_book, latency_ms, market_impact_bps, execution_time
            )
        
        # Update simulation state
        self._update_market_impact_history(symbol, market_impact_bps, execution_time)
        self.execution_history.append(result)
        self._update_statistics(result)
        
        return result
    
    def _generate_order_book(self, symbol: str, market_data: Optional[pd.DataFrame], 
                           timestamp: datetime) -> OrderBook:
        """Generate realistic order book state."""
        
        # Get current price from market data or use cached
        if market_data is not None and not market_data.empty:
            current_price = float(market_data['close'].iloc[-1])
        else:
            # Use cached price or default
            cached_book = self.order_books.get(symbol)
            current_price = cached_book.mid_price if cached_book else 50000.0
        
        # Generate spread (wider during volatile periods)
        if market_data is not None and len(market_data) > 10:
            volatility = market_data['close'].pct_change().rolling(10).std().iloc[-1]
            base_spread_bps = 5.0 + (volatility * 1000)  # Scale volatility to bps
        else:
            base_spread_bps = 8.0
        
        spread_bps = min(50.0, max(2.0, base_spread_bps))
        spread_amount = current_price * (spread_bps / 10000)
        
        # Generate bid/ask prices and quantities
        bid_price = current_price - (spread_amount / 2)
        ask_price = current_price + (spread_amount / 2)
        
        # Generate depth (randomized but realistic)
        depth_levels = 10
        bids = []
        asks = []
        
        for i in range(depth_levels):
            # Exponentially decreasing quantities as we move away from mid
            quantity_factor = np.exp(-i * 0.3)
            base_quantity = random.uniform(0.5, 5.0) * quantity_factor
            
            # Bid side
            level_bid_price = bid_price - (i * spread_amount * 0.1)
            bids.append((level_bid_price, base_quantity))
            
            # Ask side  
            level_ask_price = ask_price + (i * spread_amount * 0.1)
            asks.append((level_ask_price, base_quantity))
        
        # Calculate total depth in USD
        total_bid_value = sum(price * qty for price, qty in bids)
        total_ask_value = sum(price * qty for price, qty in asks)
        depth_usd = (total_bid_value + total_ask_value) / 2
        
        order_book = OrderBook(
            symbol=symbol,
            timestamp=timestamp,
            bids=bids,
            asks=asks,
            mid_price=current_price,
            spread_bps=spread_bps,
            depth_usd=depth_usd
        )
        
        self.order_books[symbol] = order_book
        return order_book
    
    def _simulate_latency(self) -> float:
        """Simulate network and exchange latency."""
        # Log-normal distribution for realistic latency
        mu = np.log(100)  # Median 100ms
        sigma = self.config.latency_volatility
        
        latency = np.random.lognormal(mu, sigma)
        return max(self.config.min_latency_ms, 
                  min(self.config.max_latency_ms, latency))
    
    def _calculate_market_impact(self, symbol: str, quantity: float, 
                               daily_volume: float) -> float:
        """Calculate market impact in basis points."""
        if daily_volume <= 0:
            return 0.0
        
        # Impact as percentage of daily volume
        volume_percentage = abs(quantity) / daily_volume
        
        # Square root impact model
        impact_bps = self.config.impact_per_volume_pct * np.sqrt(volume_percentage * 100)
        
        return min(200.0, impact_bps)  # Cap at 2%
    
    def _simulate_market_order(self, order_id: str, symbol: str, side: OrderSide,
                             quantity: float, order_book: OrderBook,
                             latency_ms: float, market_impact_bps: float,
                             execution_time: datetime) -> ExecutionResult:
        """Simulate market order execution."""
        
        # Market orders execute immediately but walk the book
        executed_quantity = 0.0
        total_cost = 0.0
        fills = []
        
        # Choose appropriate side of book
        if side == OrderSide.BUY:
            book_levels = order_book.asks.copy()
        else:
            book_levels = order_book.bids.copy()
        
        remaining_quantity = quantity
        
        # Walk through order book levels
        for level_price, level_quantity in book_levels:
            if remaining_quantity <= 0:
                break
            
            # Execute against this level
            fill_quantity = min(remaining_quantity, level_quantity)
            
            # Apply market impact
            impact_adjusted_price = level_price
            if side == OrderSide.BUY:
                impact_adjusted_price *= (1 + market_impact_bps / 10000)
            else:
                impact_adjusted_price *= (1 - market_impact_bps / 10000)
            
            fill_cost = fill_quantity * impact_adjusted_price
            
            fills.append({
                'price': impact_adjusted_price,
                'quantity': fill_quantity,
                'level': len(fills),
                'timestamp': execution_time
            })
            
            executed_quantity += fill_quantity
            total_cost += fill_cost
            remaining_quantity -= fill_quantity
        
        # Check for partial fill
        partial_fill = executed_quantity < quantity
        if partial_fill and random.random() < self.config.partial_fill_probability:
            # Simulate partial fill by reducing executed quantity
            fill_ratio = random.uniform(self.config.min_fill_ratio, 1.0)
            executed_quantity *= fill_ratio
            total_cost *= fill_ratio
            
            # Adjust fills
            fills = fills[:max(1, int(len(fills) * fill_ratio))]
        
        # Calculate average fill price
        avg_fill_price = total_cost / executed_quantity if executed_quantity > 0 else 0.0
        
        # Calculate slippage
        reference_price = order_book.mid_price
        if side == OrderSide.BUY:
            slippage_bps = max(0, (avg_fill_price - reference_price) / reference_price * 10000)
        else:
            slippage_bps = max(0, (reference_price - avg_fill_price) / reference_price * 10000)
        
        # Calculate fees (market orders are taker)
        total_fees = total_cost * (self.config.taker_fee_bps / 10000)
        
        return ExecutionResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            requested_quantity=quantity,
            executed_quantity=executed_quantity,
            avg_fill_price=avg_fill_price,
            total_fees=total_fees,
            slippage_bps=slippage_bps,
            latency_ms=latency_ms,
            partial_fill=partial_fill,
            execution_time=execution_time,
            market_impact_bps=market_impact_bps,
            queue_position=None,
            fills=fills,
            metadata={
                'order_book_depth': order_book.depth_usd,
                'spread_bps': order_book.spread_bps,
                'levels_consumed': len(fills)
            }
        )
    
    def _simulate_limit_order(self, order_id: str, symbol: str, side: OrderSide,
                            quantity: float, limit_price: Optional[float],
                            order_book: OrderBook, latency_ms: float,
                            market_impact_bps: float, execution_time: datetime) -> ExecutionResult:
        """Simulate limit order execution."""
        
        if limit_price is None:
            # Convert to market order if no limit price
            return self._simulate_market_order(
                order_id, symbol, side, quantity, order_book,
                latency_ms, market_impact_bps, execution_time
            )
        
        # Check if limit order can execute immediately
        if side == OrderSide.BUY:
            best_ask = order_book.asks[0][0] if order_book.asks else float('inf')
            can_execute = limit_price >= best_ask
        else:
            best_bid = order_book.bids[0][0] if order_book.bids else 0.0
            can_execute = limit_price <= best_bid
        
        if can_execute:
            # Execute as aggressive limit order (taker)
            executed_quantity = quantity
            avg_fill_price = limit_price
            
            # Apply partial fill probability
            if random.random() < self.config.partial_fill_probability:
                fill_ratio = random.uniform(self.config.min_fill_ratio, 1.0)
                executed_quantity *= fill_ratio
            
            # Calculate slippage relative to mid price
            reference_price = order_book.mid_price
            if side == OrderSide.BUY:
                slippage_bps = max(0, (avg_fill_price - reference_price) / reference_price * 10000)
            else:
                slippage_bps = max(0, (reference_price - avg_fill_price) / reference_price * 10000)
            
            # Use taker fees
            total_fees = executed_quantity * avg_fill_price * (self.config.taker_fee_bps / 10000)
            
            fills = [{
                'price': avg_fill_price,
                'quantity': executed_quantity,
                'level': 0,
                'timestamp': execution_time
            }]
            
            queue_position = None
            partial_fill = executed_quantity < quantity
            
        else:
            # Order goes to queue (passive limit order)
            queue_position = self._simulate_queue_position(symbol, side, limit_price, order_book)
            
            # Simulate queue execution probability
            execution_probability = self._calculate_execution_probability(
                queue_position, latency_ms, order_book.spread_bps
            )
            
            if random.random() < execution_probability:
                # Order executes as maker
                executed_quantity = quantity
                avg_fill_price = limit_price
                
                # Apply partial fill
                if random.random() < self.config.partial_fill_probability * 0.5:  # Lower for maker
                    fill_ratio = random.uniform(self.config.min_fill_ratio, 1.0)
                    executed_quantity *= fill_ratio
                
                # Use maker fees (lower)
                total_fees = executed_quantity * avg_fill_price * (self.config.maker_fee_bps / 10000)
                
                # Minimal slippage for maker orders
                slippage_bps = 0.0
                
                fills = [{
                    'price': avg_fill_price,
                    'quantity': executed_quantity,
                    'level': 0,
                    'timestamp': execution_time
                }]
                
                partial_fill = executed_quantity < quantity
                
            else:
                # Order doesn't execute
                executed_quantity = 0.0
                avg_fill_price = 0.0
                total_fees = 0.0
                slippage_bps = 0.0
                fills = []
                partial_fill = True
        
        return ExecutionResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            requested_quantity=quantity,
            executed_quantity=executed_quantity,
            avg_fill_price=avg_fill_price,
            total_fees=total_fees,
            slippage_bps=slippage_bps,
            latency_ms=latency_ms,
            partial_fill=partial_fill,
            execution_time=execution_time,
            market_impact_bps=market_impact_bps,
            queue_position=queue_position,
            fills=fills,
            metadata={
                'limit_price': limit_price,
                'order_book_depth': order_book.depth_usd,
                'spread_bps': order_book.spread_bps,
                'can_execute_immediately': can_execute
            }
        )
    
    def _simulate_queue_position(self, symbol: str, side: OrderSide, 
                               limit_price: float, order_book: OrderBook) -> int:
        """Simulate queue position for limit orders."""
        
        # Find position in queue based on price level
        if side == OrderSide.BUY:
            relevant_levels = [price for price, qty in order_book.bids if price >= limit_price]
        else:
            relevant_levels = [price for price, qty in order_book.asks if price <= limit_price]
        
        # Queue position based on price improvement and random factors
        base_position = len(relevant_levels) + 1
        random_adjustment = random.randint(-2, 5)  # Some randomness
        
        return max(1, base_position + random_adjustment)
    
    def _calculate_execution_probability(self, queue_position: int, 
                                       latency_ms: float, spread_bps: float) -> float:
        """Calculate probability of limit order execution."""
        
        # Base probability decreases with queue position
        base_prob = 1.0 / (1.0 + queue_position * 0.1)
        
        # Higher latency reduces execution probability
        latency_penalty = max(0.1, 1.0 - (latency_ms - 100) / 1000)
        
        # Wider spreads increase execution probability
        spread_bonus = min(2.0, 1.0 + spread_bps / 50.0)
        
        # Random market activity factor
        market_activity = random.uniform(0.5, 1.5)
        
        execution_prob = base_prob * latency_penalty * spread_bonus * market_activity
        
        return min(0.95, max(0.05, execution_prob))
    
    def _update_market_impact_history(self, symbol: str, impact_bps: float, 
                                    timestamp: datetime):
        """Update market impact history for decay modeling."""
        if symbol not in self.market_impact_history:
            self.market_impact_history[symbol] = []
        
        # Add new impact
        self.market_impact_history[symbol].append((timestamp, impact_bps))
        
        # Remove old impacts (beyond decay period)
        cutoff_time = timestamp - timedelta(seconds=self.config.impact_decay_halflife * 5)
        self.market_impact_history[symbol] = [
            (t, impact) for t, impact in self.market_impact_history[symbol]
            if t > cutoff_time
        ]
    
    def _update_statistics(self, result: ExecutionResult):
        """Update simulation statistics."""
        if result.executed_quantity > 0:
            # Update volume tracking
            self.total_executed_volume += result.executed_quantity
            
            # Update average slippage
            if hasattr(self, '_slippage_sum'):
                self._slippage_sum += result.slippage_bps
                self._slippage_count += 1
            else:
                self._slippage_sum = result.slippage_bps
                self._slippage_count = 1
            
            self.avg_slippage = self._slippage_sum / self._slippage_count
        
        # Update fill rate
        total_orders = len(self.execution_history)
        filled_orders = len([r for r in self.execution_history if r.executed_quantity > 0])
        self.fill_rate = filled_orders / total_orders if total_orders > 0 else 0.0
    
    def get_simulation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive simulation statistics."""
        
        if not self.execution_history:
            return {'message': 'No executions recorded'}
        
        # Calculate detailed statistics
        total_orders = len(self.execution_history)
        market_orders = [r for r in self.execution_history if r.order_type == OrderType.MARKET]
        limit_orders = [r for r in self.execution_history if r.order_type == OrderType.LIMIT]
        
        filled_orders = [r for r in self.execution_history if r.executed_quantity > 0]
        partial_fills = [r for r in self.execution_history if r.partial_fill]
        
        # Slippage statistics
        slippages = [r.slippage_bps for r in filled_orders if r.slippage_bps > 0]
        
        # Fee statistics
        total_fees = sum(r.total_fees for r in filled_orders)
        
        # Latency statistics
        latencies = [r.latency_ms for r in self.execution_history]
        
        return {
            'total_orders': total_orders,
            'filled_orders': len(filled_orders),
            'fill_rate': self.fill_rate,
            'partial_fill_rate': len(partial_fills) / total_orders if total_orders > 0 else 0.0,
            'market_orders': len(market_orders),
            'limit_orders': len(limit_orders),
            'total_volume': self.total_executed_volume,
            'total_fees': total_fees,
            'average_slippage_bps': np.mean(slippages) if slippages else 0.0,
            'median_slippage_bps': np.median(slippages) if slippages else 0.0,
            'max_slippage_bps': max(slippages) if slippages else 0.0,
            'average_latency_ms': np.mean(latencies) if latencies else 0.0,
            'median_latency_ms': np.median(latencies) if latencies else 0.0,
            'fee_rate_bps': (total_fees / (self.total_executed_volume * 50000) * 10000) if self.total_executed_volume > 0 else 0.0
        }
    
    def reset_simulation(self):
        """Reset simulation state."""
        self.execution_history.clear()
        self.market_impact_history.clear()
        self.order_books.clear()
        self.total_executed_volume = 0.0
        self.avg_slippage = 0.0
        self.fill_rate = 0.0
        
        if hasattr(self, '_slippage_sum'):
            delattr(self, '_slippage_sum')
            delattr(self, '_slippage_count')


def create_execution_simulator(config: Optional[SimulationConfig] = None) -> ExecutionSimulator:
    """Create execution simulator with optional configuration."""
    return ExecutionSimulator(config)