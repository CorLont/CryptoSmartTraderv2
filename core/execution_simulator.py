#!/usr/bin/env python3
"""
Advanced Execution Simulator with Level-2 Order Book and Realistic Market Impact
Implements comprehensive backtesting with slippage, partial fills, latency, and fees
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import warnings
warnings.filterwarnings('ignore')

from core.logging_manager import get_logger

class OrderType(str, Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class OrderSide(str, Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(str, Enum):
    """Order execution status"""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class ExchangeType(str, Enum):
    """Exchange types with different characteristics"""
    MAJOR_CEX = "major_cex"  # Binance, Coinbase, Kraken
    MINOR_CEX = "minor_cex"  # Smaller centralized exchanges
    DEX = "dex"              # Decentralized exchanges

@dataclass
class OrderBookLevel:
    """Single level in order book"""
    price: float
    size: float
    timestamp: datetime

@dataclass
class OrderBook:
    """Level-2 order book representation"""
    symbol: str
    bids: List[OrderBookLevel]  # Sorted by price descending
    asks: List[OrderBookLevel]  # Sorted by price ascending
    timestamp: datetime
    spread_bps: float = 0.0
    
    def __post_init__(self):
        """Calculate spread after initialization"""
        if self.bids and self.asks:
            best_bid = self.bids[0].price
            best_ask = self.asks[0].price
            mid_price = (best_bid + best_ask) / 2
            self.spread_bps = ((best_ask - best_bid) / mid_price) * 10000

@dataclass
class ExecutionFill:
    """Individual fill from order execution"""
    price: float
    size: float
    fee: float
    timestamp: datetime
    fee_currency: str
    order_book_level: int  # Which level of order book was hit

@dataclass
class Order:
    """Trading order with execution tracking"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    size: float
    price: Optional[float] = None  # None for market orders
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    fills: List[ExecutionFill] = field(default_factory=list)
    remaining_size: float = 0.0
    average_fill_price: float = 0.0
    total_fees: float = 0.0
    slippage_bps: float = 0.0
    
    def __post_init__(self):
        """Initialize remaining size"""
        if self.remaining_size == 0.0:
            self.remaining_size = self.size

@dataclass
class MarketConditions:
    """Current market conditions affecting execution"""
    volatility: float  # Recent volatility measure
    volume_ratio: float  # Current volume vs average
    spread_ratio: float  # Current spread vs average
    depth_ratio: float  # Order book depth vs average
    market_impact_factor: float = 1.0  # Multiplier for impact calculations

@dataclass
class ExchangeProfile:
    """Exchange-specific execution characteristics"""
    name: str
    exchange_type: ExchangeType
    base_fee_bps: float  # Base trading fee in basis points
    fee_tiers: Dict[float, float]  # Volume -> fee rate mapping
    min_order_size: float
    max_order_size: float
    latency_ms: Tuple[float, float]  # (min, max) latency
    rate_limit_orders_per_second: float
    maintenance_probability: float  # Daily probability of maintenance
    partial_fill_probability: float  # Probability of partial fills
    slippage_multiplier: float = 1.0  # Exchange-specific slippage factor

class OrderBookSimulator:
    """Simulates realistic level-2 order book behavior"""
    
    def __init__(self):
        self.logger = get_logger()
        
    def generate_realistic_order_book(
        self, 
        symbol: str, 
        mid_price: float, 
        market_conditions: MarketConditions,
        depth_levels: int = 20
    ) -> OrderBook:
        """Generate realistic order book based on market conditions"""
        
        timestamp = datetime.now()
        
        # Calculate base spread based on volatility and market conditions
        base_spread_bps = 5.0 + (market_conditions.volatility * 100) * market_conditions.spread_ratio
        spread = (base_spread_bps / 10000) * mid_price
        
        best_bid = mid_price - spread / 2
        best_ask = mid_price + spread / 2
        
        # Generate bid levels
        bids = []
        current_price = best_bid
        for level in range(depth_levels):
            # Exponential decay in size with depth
            base_size = 1000 * np.exp(-level * 0.3) * market_conditions.depth_ratio
            # Add randomness
            size = base_size * (0.5 + np.random.random())
            
            bids.append(OrderBookLevel(
                price=current_price,
                size=size,
                timestamp=timestamp
            ))
            
            # Price decay with some randomness
            price_step = spread * (0.1 + np.random.random() * 0.2)
            current_price -= price_step
        
        # Generate ask levels
        asks = []
        current_price = best_ask
        for level in range(depth_levels):
            # Exponential decay in size with depth
            base_size = 1000 * np.exp(-level * 0.3) * market_conditions.depth_ratio
            # Add randomness
            size = base_size * (0.5 + np.random.random())
            
            asks.append(OrderBookLevel(
                price=current_price,
                size=size,
                timestamp=timestamp
            ))
            
            # Price increase with some randomness
            price_step = spread * (0.1 + np.random.random() * 0.2)
            current_price += price_step
        
        return OrderBook(
            symbol=symbol,
            bids=sorted(bids, key=lambda x: x.price, reverse=True),
            asks=sorted(asks, key=lambda x: x.price),
            timestamp=timestamp
        )
    
    def simulate_market_impact(
        self, 
        order: Order, 
        order_book: OrderBook, 
        market_conditions: MarketConditions
    ) -> float:
        """Calculate market impact based on order size and market depth"""
        
        # Get relevant side of order book
        levels = order_book.asks if order.side == OrderSide.BUY else order_book.bids
        
        if not levels:
            return 0.5  # High impact if no liquidity
        
        # Calculate total available liquidity in first few levels
        total_liquidity = sum(level.size for level in levels[:5])
        
        # Calculate order size as percentage of available liquidity
        liquidity_ratio = order.size / (total_liquidity + 1e-10)
        
        # Base impact factor
        base_impact = liquidity_ratio * 0.1  # 10% impact per 100% of liquidity
        
        # Adjust for market conditions
        volatility_adjustment = market_conditions.volatility * 0.5
        volume_adjustment = (1.0 / market_conditions.volume_ratio) * 0.3
        
        # Total market impact
        total_impact = (base_impact + volatility_adjustment + volume_adjustment) * market_conditions.market_impact_factor
        
        return min(total_impact, 0.05)  # Cap at 5% impact

class ExchangeSimulator:
    """Simulates exchange-specific execution behavior"""
    
    def __init__(self):
        self.logger = get_logger()
        self.exchange_profiles = self._initialize_exchange_profiles()
        self.order_book_simulator = OrderBookSimulator()
        
    def _initialize_exchange_profiles(self) -> Dict[str, ExchangeProfile]:
        """Initialize realistic exchange profiles"""
        
        return {
            'kraken': ExchangeProfile(
                name='kraken',
                exchange_type=ExchangeType.MAJOR_CEX,
                base_fee_bps=26.0,  # 0.26%
                fee_tiers={
                    0: 26.0,
                    50000: 24.0,
                    100000: 22.0,
                    250000: 20.0,
                    500000: 18.0,
                    1000000: 16.0
                },
                min_order_size=0.0001,
                max_order_size=1000000,
                latency_ms=(50, 200),
                rate_limit_orders_per_second=1.0,
                maintenance_probability=0.02,  # 2% daily chance
                partial_fill_probability=0.15,
                slippage_multiplier=1.0
            ),
            'binance': ExchangeProfile(
                name='binance',
                exchange_type=ExchangeType.MAJOR_CEX,
                base_fee_bps=10.0,  # 0.10%
                fee_tiers={
                    0: 10.0,
                    100000: 9.0,
                    500000: 8.0,
                    1000000: 7.0,
                    5000000: 6.0,
                    10000000: 5.0
                },
                min_order_size=0.00001,
                max_order_size=9000000,
                latency_ms=(20, 100),
                rate_limit_orders_per_second=10.0,
                maintenance_probability=0.01,  # 1% daily chance
                partial_fill_probability=0.10,
                slippage_multiplier=0.8
            ),
            'coinbase': ExchangeProfile(
                name='coinbase',
                exchange_type=ExchangeType.MAJOR_CEX,
                base_fee_bps=50.0,  # 0.50%
                fee_tiers={
                    0: 50.0,
                    10000: 35.0,
                    50000: 25.0,
                    100000: 15.0,
                    1000000: 10.0,
                    15000000: 5.0
                },
                min_order_size=0.001,
                max_order_size=10000000,
                latency_ms=(100, 300),
                rate_limit_orders_per_second=2.0,
                maintenance_probability=0.015,  # 1.5% daily chance
                partial_fill_probability=0.20,
                slippage_multiplier=1.2
            )
        }
    
    def calculate_trading_fee(
        self, 
        exchange_name: str, 
        trade_volume_30d: float, 
        trade_amount: float
    ) -> float:
        """Calculate trading fee based on volume tier"""
        
        if exchange_name not in self.exchange_profiles:
            return trade_amount * 0.001  # 0.1% default fee
        
        profile = self.exchange_profiles[exchange_name]
        
        # Find applicable fee tier
        fee_rate = profile.base_fee_bps
        for volume_threshold, tier_fee in sorted(profile.fee_tiers.items(), reverse=True):
            if trade_volume_30d >= volume_threshold:
                fee_rate = tier_fee
                break
        
        return trade_amount * (fee_rate / 10000)
    
    def simulate_execution_latency(self, exchange_name: str) -> float:
        """Simulate realistic execution latency"""
        
        if exchange_name not in self.exchange_profiles:
            return np.random.uniform(100, 500)  # Default latency
        
        profile = self.exchange_profiles[exchange_name]
        min_latency, max_latency = profile.latency_ms
        
        # Use gamma distribution for realistic latency simulation
        shape = 2.0
        scale = (max_latency - min_latency) / 4
        latency = min_latency + np.random.gamma(shape, scale)
        
        return min(latency, max_latency * 2)  # Cap at 2x max
    
    def check_maintenance_window(self, exchange_name: str) -> bool:
        """Check if exchange is in maintenance"""
        
        if exchange_name not in self.exchange_profiles:
            return False
        
        profile = self.exchange_profiles[exchange_name]
        return np.random.random() < profile.maintenance_probability / 24  # Hourly probability

class ExecutionSimulator:
    """Main execution simulator with comprehensive market microstructure"""
    
    def __init__(self):
        self.logger = get_logger()
        self.exchange_simulator = ExchangeSimulator()
        self.order_book_simulator = OrderBookSimulator()
        
        # Execution tracking
        self.executed_orders: List[Order] = []
        self.execution_history: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.total_slippage = 0.0
        self.total_fees = 0.0
        self.partial_fill_count = 0
        self.rejected_orders = 0
        
    def execute_order(
        self, 
        order: Order, 
        current_price: float,
        market_conditions: MarketConditions,
        exchange_name: str = 'kraken',
        user_volume_30d: float = 0.0
    ) -> Order:
        """Execute order with realistic market microstructure simulation"""
        
        execution_start = datetime.now()
        
        try:
            # Check if exchange is in maintenance
            if self.exchange_simulator.check_maintenance_window(exchange_name):
                order.status = OrderStatus.REJECTED
                self.rejected_orders += 1
                self.logger.warning(
                    f"Order rejected - exchange maintenance",
                    extra={'order_id': order.order_id, 'exchange': exchange_name}
                )
                return order
            
            # Simulate execution latency
            latency_ms = self.exchange_simulator.simulate_execution_latency(exchange_name)
            
            # Generate realistic order book
            order_book = self.order_book_simulator.generate_realistic_order_book(
                order.symbol, current_price, market_conditions
            )
            
            # Calculate market impact
            market_impact = self.order_book_simulator.simulate_market_impact(
                order, order_book, market_conditions
            )
            
            # Execute order against order book
            if order.order_type == OrderType.MARKET:
                order = self._execute_market_order(
                    order, order_book, market_conditions, exchange_name, 
                    user_volume_30d, market_impact
                )
            else:
                order = self._execute_limit_order(
                    order, order_book, market_conditions, exchange_name,
                    user_volume_30d, market_impact
                )
            
            # Record execution metrics
            execution_time = (datetime.now() - execution_start).total_seconds() * 1000
            
            self.execution_history.append({
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'size': order.size,
                'execution_time_ms': execution_time,
                'latency_ms': latency_ms,
                'market_impact': market_impact,
                'slippage_bps': order.slippage_bps,
                'total_fees': order.total_fees,
                'fill_count': len(order.fills),
                'status': order.status.value,
                'exchange': exchange_name,
                'timestamp': execution_start.isoformat()
            })
            
            self.executed_orders.append(order)
            
            # Update performance metrics
            self.total_slippage += order.slippage_bps
            self.total_fees += order.total_fees
            if order.status == OrderStatus.PARTIALLY_FILLED:
                self.partial_fill_count += 1
            
            self.logger.info(
                f"Order executed: {order.order_id}",
                extra={
                    'order_id': order.order_id,
                    'status': order.status.value,
                    'slippage_bps': order.slippage_bps,
                    'fees': order.total_fees,
                    'execution_time_ms': execution_time
                }
            )
            
        except Exception as e:
            order.status = OrderStatus.REJECTED
            self.rejected_orders += 1
            self.logger.error(
                f"Order execution failed: {e}",
                extra={'order_id': order.order_id, 'error': str(e)}
            )
        
        return order
    
    def _execute_market_order(
        self, 
        order: Order, 
        order_book: OrderBook,
        market_conditions: MarketConditions,
        exchange_name: str,
        user_volume_30d: float,
        market_impact: float
    ) -> Order:
        """Execute market order with level-by-level fills"""
        
        # Get relevant side of order book
        levels = order_book.asks if order.side == OrderSide.BUY else order_book.bids
        
        if not levels:
            order.status = OrderStatus.REJECTED
            return order
        
        remaining_size = order.size
        total_cost = 0.0
        
        # Execute against order book levels
        for level_idx, level in enumerate(levels):
            if remaining_size <= 0:
                break
            
            # Calculate fill size for this level
            available_size = level.size
            fill_size = min(remaining_size, available_size)
            
            # Apply market impact to price
            impact_adjustment = market_impact * (level_idx + 1) * 0.1  # Increasing impact with depth
            if order.side == OrderSide.BUY:
                fill_price = level.price * (1 + impact_adjustment)
            else:
                fill_price = level.price * (1 - impact_adjustment)
            
            # Calculate fees
            fill_value = fill_size * fill_price
            fee = self.exchange_simulator.calculate_trading_fee(
                exchange_name, user_volume_30d, fill_value
            )
            
            # Create fill
            fill = ExecutionFill(
                price=fill_price,
                size=fill_size,
                fee=fee,
                timestamp=datetime.now(),
                fee_currency=order.symbol.split('/')[1] if '/' in order.symbol else 'USD',
                order_book_level=level_idx
            )
            
            order.fills.append(fill)
            remaining_size -= fill_size
            total_cost += fill_value
            order.total_fees += fee
            
            # Check for partial fill simulation
            profile = self.exchange_simulator.exchange_profiles.get(exchange_name)
            if profile and np.random.random() < profile.partial_fill_probability:
                # Simulate partial fill - stop execution early
                break
        
        # Update order status
        order.remaining_size = remaining_size
        if remaining_size > 0:
            order.status = OrderStatus.PARTIALLY_FILLED
        else:
            order.status = OrderStatus.FILLED
        
        # Calculate average fill price and slippage
        if order.fills:
            filled_size = sum(fill.size for fill in order.fills)
            order.average_fill_price = sum(fill.price * fill.size for fill in order.fills) / filled_size
            
            # Calculate slippage vs mid price
            mid_price = (order_book.bids[0].price + order_book.asks[0].price) / 2
            if order.side == OrderSide.BUY:
                order.slippage_bps = ((order.average_fill_price - mid_price) / mid_price) * 10000
            else:
                order.slippage_bps = ((mid_price - order.average_fill_price) / mid_price) * 10000
        
        return order
    
    def _execute_limit_order(
        self, 
        order: Order, 
        order_book: OrderBook,
        market_conditions: MarketConditions,
        exchange_name: str,
        user_volume_30d: float,
        market_impact: float
    ) -> Order:
        """Execute limit order based on current market conditions"""
        
        # Get best price from opposite side
        if order.side == OrderSide.BUY:
            best_offer = order_book.asks[0].price if order_book.asks else float('inf')
            can_execute = order.price >= best_offer
        else:
            best_bid = order_book.bids[0].price if order_book.bids else 0.0
            can_execute = order.price <= best_bid
        
        if not can_execute:
            # Order goes to order book (not executed immediately)
            order.status = OrderStatus.PENDING
            return order
        
        # Execute as if market order at limit price
        fill_price = order.price
        fill_size = order.size
        
        # Calculate fees
        fill_value = fill_size * fill_price
        fee = self.exchange_simulator.calculate_trading_fee(
            exchange_name, user_volume_30d, fill_value
        )
        
        # Create fill
        fill = ExecutionFill(
            price=fill_price,
            size=fill_size,
            fee=fee,
            timestamp=datetime.now(),
            fee_currency=order.symbol.split('/')[1] if '/' in order.symbol else 'USD',
            order_book_level=0  # Limit orders execute at best level
        )
        
        order.fills.append(fill)
        order.remaining_size = 0.0
        order.status = OrderStatus.FILLED
        order.average_fill_price = fill_price
        order.total_fees = fee
        
        # Calculate slippage vs mid price
        mid_price = (order_book.bids[0].price + order_book.asks[0].price) / 2
        if order.side == OrderSide.BUY:
            order.slippage_bps = ((fill_price - mid_price) / mid_price) * 10000
        else:
            order.slippage_bps = ((mid_price - fill_price) / mid_price) * 10000
        
        return order
    
    def backtest_strategy(
        self, 
        trades: List[Dict[str, Any]], 
        market_data: pd.DataFrame,
        exchange_name: str = 'kraken',
        user_volume_30d: float = 100000
    ) -> Dict[str, Any]:
        """Run comprehensive backtest with realistic execution simulation"""
        
        backtest_results = {
            'start_time': datetime.now().isoformat(),
            'total_trades': len(trades),
            'executed_orders': [],
            'performance_metrics': {},
            'execution_statistics': {},
            'risk_metrics': {}
        }
        
        portfolio_value = 100000  # Starting portfolio value
        portfolio_history = []
        total_pnl = 0.0
        
        for i, trade in enumerate(trades):
            try:
                # Get market conditions for this trade
                market_row = market_data.iloc[i] if i < len(market_data) else market_data.iloc[-1]
                
                market_conditions = MarketConditions(
                    volatility=market_row.get('volatility', 0.02),
                    volume_ratio=market_row.get('volume_ratio', 1.0),
                    spread_ratio=market_row.get('spread_ratio', 1.0),
                    depth_ratio=market_row.get('depth_ratio', 1.0),
                    market_impact_factor=1.0
                )
                
                # Create order
                order = Order(
                    order_id=f"backtest_{i}",
                    symbol=trade.get('symbol', 'BTC/USD'),
                    side=OrderSide(trade['side']),
                    order_type=OrderType(trade.get('type', 'market')),
                    size=trade['size'],
                    price=trade.get('price'),
                    timestamp=market_row.get('timestamp', datetime.now())
                )
                
                # Execute order
                executed_order = self.execute_order(
                    order, 
                    market_row.get('price', market_row.get('close', 0)),
                    market_conditions,
                    exchange_name,
                    user_volume_30d
                )
                
                backtest_results['executed_orders'].append({
                    'order_id': executed_order.order_id,
                    'status': executed_order.status.value,
                    'average_fill_price': executed_order.average_fill_price,
                    'slippage_bps': executed_order.slippage_bps,
                    'total_fees': executed_order.total_fees,
                    'fill_count': len(executed_order.fills)
                })
                
                # Update portfolio value
                if executed_order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                    trade_pnl = self._calculate_trade_pnl(executed_order, trade)
                    total_pnl += trade_pnl
                    portfolio_value += trade_pnl
                
                portfolio_history.append({
                    'timestamp': market_row.get('timestamp', datetime.now()).isoformat(),
                    'portfolio_value': portfolio_value,
                    'pnl': total_pnl
                })
                
            except Exception as e:
                self.logger.error(f"Backtest trade {i} failed: {e}")
        
        # Calculate performance metrics
        backtest_results['performance_metrics'] = self._calculate_backtest_metrics(
            portfolio_history, backtest_results['executed_orders']
        )
        
        # Calculate execution statistics
        backtest_results['execution_statistics'] = self._calculate_execution_statistics()
        
        self.logger.info(
            f"Backtest completed: {len(trades)} trades",
            extra={
                'total_trades': len(trades),
                'successful_executions': len([o for o in backtest_results['executed_orders'] if o['status'] == 'filled']),
                'total_slippage_bps': sum(o['slippage_bps'] for o in backtest_results['executed_orders']),
                'total_fees': sum(o['total_fees'] for o in backtest_results['executed_orders'])
            }
        )
        
        return backtest_results
    
    def _calculate_trade_pnl(self, order: Order, trade: Dict[str, Any]) -> float:
        """Calculate P&L for executed trade (simplified)"""
        if order.status != OrderStatus.FILLED:
            return 0.0
        
        # This is a simplified P&L calculation
        # In practice, you'd need exit prices and position tracking
        base_pnl = order.average_fill_price * order.size * 0.01  # 1% assumed profit
        return base_pnl - order.total_fees
    
    def _calculate_backtest_metrics(
        self, 
        portfolio_history: List[Dict[str, Any]], 
        executed_orders: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate comprehensive backtest performance metrics"""
        
        if not portfolio_history:
            return {}
        
        values = [p['portfolio_value'] for p in portfolio_history]
        returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
        
        metrics = {
            'total_return': (values[-1] - values[0]) / values[0] if values[0] > 0 else 0.0,
            'volatility': np.std(returns) if returns else 0.0,
            'sharpe_ratio': (np.mean(returns) / np.std(returns)) if returns and np.std(returns) > 0 else 0.0,
            'max_drawdown': self._calculate_max_drawdown(values),
            'win_rate': len([r for r in returns if r > 0]) / len(returns) if returns else 0.0
        }
        
        return metrics
    
    def _calculate_max_drawdown(self, values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not values:
            return 0.0
        
        peak = values[0]
        max_dd = 0.0
        
        for value in values:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _calculate_execution_statistics(self) -> Dict[str, Any]:
        """Calculate execution quality statistics"""
        
        if not self.execution_history:
            return {}
        
        slippages = [h['slippage_bps'] for h in self.execution_history]
        fees = [h['total_fees'] for h in self.execution_history]
        latencies = [h['latency_ms'] for h in self.execution_history]
        
        return {
            'average_slippage_bps': np.mean(slippages) if slippages else 0.0,
            'median_slippage_bps': np.median(slippages) if slippages else 0.0,
            'max_slippage_bps': np.max(slippages) if slippages else 0.0,
            'total_fees': sum(fees),
            'average_fees': np.mean(fees) if fees else 0.0,
            'average_latency_ms': np.mean(latencies) if latencies else 0.0,
            'p95_latency_ms': np.percentile(latencies, 95) if latencies else 0.0,
            'partial_fill_rate': self.partial_fill_count / len(self.execution_history) if self.execution_history else 0.0,
            'rejection_rate': self.rejected_orders / len(self.execution_history) if self.execution_history else 0.0
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary"""
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_orders_executed': len(self.executed_orders),
            'execution_statistics': self._calculate_execution_statistics(),
            'supported_exchanges': list(self.exchange_simulator.exchange_profiles.keys()),
            'recent_executions': self.execution_history[-10:] if self.execution_history else []
        }

# Global instance
_execution_simulator = None

def get_execution_simulator() -> ExecutionSimulator:
    """Get global execution simulator instance"""
    global _execution_simulator
    if _execution_simulator is None:
        _execution_simulator = ExecutionSimulator()
    return _execution_simulator