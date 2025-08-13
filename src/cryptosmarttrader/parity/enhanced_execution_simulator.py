"""
Enhanced Execution Simulator - Fase D Implementation
Realistic execution simulation with comprehensive backtest-live parity monitoring.
"""

import asyncio
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

from ..core.structured_logger import get_logger
from ..execution.execution_policy import OrderType, OrderSide, OrderResult, OrderStatus


@dataclass
class MarketConditions:
    """Real-time market conditions for simulation."""
    
    symbol: str
    bid_price: float
    ask_price: float
    mid_price: float
    spread_bps: float
    volume_24h: float
    orderbook_depth_bid: float
    orderbook_depth_ask: float
    volatility: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionFees:
    """Fee structure for different order types."""
    
    maker_fee_bps: float = 10.0  # 0.10% maker fee
    taker_fee_bps: float = 25.0  # 0.25% taker fee
    
    def calculate_fee(self, quantity: float, price: float, is_maker: bool) -> float:
        """Calculate fee for execution."""
        notional = quantity * price
        fee_bps = self.maker_fee_bps if is_maker else self.taker_fee_bps
        return notional * (fee_bps / 10000.0)


@dataclass
class LatencyModel:
    """Latency modeling for realistic execution."""
    
    network_latency_ms: Tuple[float, float] = (5.0, 50.0)  # min, max
    exchange_processing_ms: Tuple[float, float] = (1.0, 10.0)  # min, max
    queue_delay_ms: Tuple[float, float] = (0.0, 100.0)  # min, max when busy
    
    def generate_total_latency(self, market_stress: float = 0.0) -> float:
        """Generate realistic total latency based on market conditions."""
        network = np.random.uniform(*self.network_latency_ms)
        processing = np.random.uniform(*self.exchange_processing_ms)
        queue = np.random.uniform(*self.queue_delay_ms) * market_stress
        
        return network + processing + queue


@dataclass
class SlippageModel:
    """Advanced slippage modeling."""
    
    base_slippage_bps: float = 2.0  # Base slippage 2 bps
    size_impact_factor: float = 0.01  # Size impact coefficient  
    volatility_impact_factor: float = 10.0  # Volatility impact
    
    def calculate_slippage(self, quantity: float, price: float, volatility: float, 
                          orderbook_depth: float) -> float:
        """Calculate realistic slippage based on market conditions."""
        notional = quantity * price
        
        # Size impact (square root law)
        size_impact = self.size_impact_factor * np.sqrt(notional / max(orderbook_depth, 1000))
        
        # Volatility impact
        vol_impact = self.volatility_impact_factor * volatility
        
        # Market impact (temporary)
        market_impact = np.random.exponential(0.5)  # Random market impact
        
        total_slippage_bps = self.base_slippage_bps + size_impact + vol_impact + market_impact
        return min(total_slippage_bps, 200.0)  # Cap at 200 bps


class EnhancedExecutionSimulator:
    """
    Enhanced Execution Simulator for Fase D Parity Monitoring.
    
    Features:
    - Realistic latency modeling
    - Advanced slippage calculation
    - Partial fill simulation
    - Market microstructure effects
    - Order book depth impact
    - Fee calculation (maker/taker)
    - Queue position modeling
    """
    
    def __init__(self):
        self.logger = get_logger("enhanced_execution_simulator")
        
        # Models
        self.fee_model = ExecutionFees()
        self.latency_model = LatencyModel()
        self.slippage_model = SlippageModel()
        
        # Simulation state
        self.simulated_executions: List[OrderResult] = []
        self.market_stress_level = 0.0  # 0.0 to 1.0
        
        # Statistics
        self.total_simulated_orders = 0
        self.total_slippage_bps = 0.0
        self.total_fees_paid = 0.0
        
        self.logger.info("EnhancedExecutionSimulator initialized for backtest-live parity")
    
    async def simulate_order_execution(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        market_conditions: Optional[MarketConditions] = None
    ) -> OrderResult:
        """
        Simulate realistic order execution with comprehensive modeling.
        """
        start_time = time.time()
        order_id = client_order_id or f"sim_{int(time.time() * 1000)}"
        
        # Get or generate market conditions
        conditions = market_conditions or await self._generate_market_conditions(symbol)
        
        try:
            # Step 1: Latency simulation
            execution_latency_ms = self.latency_model.generate_total_latency(self.market_stress_level)
            await asyncio.sleep(execution_latency_ms / 1000.0)  # Simulate latency
            
            # Step 2: Determine execution price and slippage
            execution_price, slippage_bps = self._calculate_execution_price(
                side, quantity, order_type, price, conditions
            )
            
            # Step 3: Partial fill simulation
            fill_ratio = self._simulate_fill_ratio(order_type, conditions.spread_bps, conditions.volume_24h)
            filled_quantity = quantity * fill_ratio
            
            # Step 4: Fee calculation
            is_maker = order_type == "limit" and np.random.random() < 0.7  # 70% chance maker for limit orders
            total_fees = self.fee_model.calculate_fee(filled_quantity, execution_price, is_maker)
            
            # Step 5: Create execution result
            status = OrderStatus.FILLED if fill_ratio >= 0.99 else OrderStatus.PARTIAL
            
            result = OrderResult(
                client_order_id=order_id,
                exchange_order_id=f"EXCH_{order_id}",
                status=status,
                filled_quantity=filled_quantity,
                avg_fill_price=execution_price,
                total_fees=total_fees,
                slippage_percent=slippage_bps / 10000.0,  # Convert bps to percentage
                execution_time_ms=int(execution_latency_ms),
                timestamp=datetime.now()
            )
            
            # Update statistics
            self.total_simulated_orders += 1
            self.total_slippage_bps += slippage_bps
            self.total_fees_paid += total_fees
            self.simulated_executions.append(result)
            
            # Log execution
            self.logger.info(
                f"Simulated execution completed",
                symbol=symbol,
                side=side,
                quantity=quantity,
                filled_quantity=filled_quantity,
                execution_price=execution_price,
                slippage_bps=slippage_bps,
                fees=total_fees,
                latency_ms=execution_latency_ms
            )
            
            return result
            
        except Exception as e:
            # Return failed execution
            self.logger.error(f"Execution simulation failed: {e}")
            
            return OrderResult(
                client_order_id=order_id,
                exchange_order_id=None,
                status=OrderStatus.REJECTED,
                filled_quantity=0.0,
                avg_fill_price=0.0,
                total_fees=0.0,
                slippage_percent=0.0,
                execution_time_ms=0,
                error_message=str(e),
                timestamp=datetime.now()
            )
    
    def _calculate_execution_price(
        self, 
        side: str, 
        quantity: float, 
        order_type: str, 
        limit_price: Optional[float],
        conditions: MarketConditions
    ) -> Tuple[float, float]:
        """Calculate realistic execution price and slippage."""
        
        if order_type == "market":
            # Market order - use ask for buy, bid for sell
            base_price = conditions.ask_price if side == "buy" else conditions.bid_price
            
            # Calculate slippage
            orderbook_depth = conditions.orderbook_depth_ask if side == "buy" else conditions.orderbook_depth_bid
            slippage_bps = self.slippage_model.calculate_slippage(
                quantity, base_price, conditions.volatility, orderbook_depth
            )
            
            # Apply slippage
            slippage_multiplier = 1 + (slippage_bps / 10000.0) if side == "buy" else 1 - (slippage_bps / 10000.0)
            execution_price = base_price * slippage_multiplier
            
        else:  # Limit order
            base_price = limit_price or conditions.mid_price
            
            # Limit orders have lower slippage but may not fill
            slippage_bps = self.slippage_model.base_slippage_bps + np.random.normal(0, 1.0)
            slippage_bps = max(0, slippage_bps)  # No negative slippage
            
            slippage_multiplier = 1 + (slippage_bps / 10000.0) if side == "buy" else 1 - (slippage_bps / 10000.0)
            execution_price = base_price * slippage_multiplier
        
        return execution_price, slippage_bps
    
    def _simulate_fill_ratio(self, order_type: str, spread_bps: float, volume_24h: float) -> float:
        """Simulate partial fill ratio based on market conditions."""
        
        if order_type == "market":
            # Market orders usually fill completely, but may be partial in stressed conditions
            base_fill_ratio = 0.98
            stress_penalty = self.market_stress_level * 0.2
            return min(1.0, base_fill_ratio - stress_penalty + np.random.normal(0, 0.02))
        
        else:  # Limit orders
            # Limit orders have variable fill ratios
            base_fill_ratio = 0.8
            
            # Wide spreads reduce fill probability
            spread_penalty = min(0.3, spread_bps / 100.0 * 0.1)
            
            # Low volume reduces fill probability  
            volume_penalty = 0.1 if volume_24h < 100000 else 0.0
            
            fill_ratio = base_fill_ratio - spread_penalty - volume_penalty + np.random.normal(0, 0.1)
            return max(0.0, min(1.0, fill_ratio))
    
    async def _generate_market_conditions(self, symbol: str) -> MarketConditions:
        """Generate realistic market conditions for simulation."""
        
        # Base prices (in production, this would come from live data)
        base_prices = {
            "BTC/USD": 45000.0,
            "ETH/USD": 3000.0,
            "ADA/USD": 0.5,
            "SOL/USD": 100.0
        }
        
        mid_price = base_prices.get(symbol, 100.0)
        
        # Generate realistic spread (2-20 bps)
        spread_bps = np.random.uniform(2.0, 20.0)
        spread_absolute = mid_price * (spread_bps / 10000.0)
        
        bid_price = mid_price - (spread_absolute / 2)
        ask_price = mid_price + (spread_absolute / 2)
        
        # Generate market depth and volatility
        volume_24h = np.random.uniform(500000, 5000000)  # $500k - $5M
        orderbook_depth_bid = np.random.uniform(10000, 100000)  # $10k - $100k
        orderbook_depth_ask = np.random.uniform(10000, 100000)  # $10k - $100k
        volatility = np.random.uniform(0.01, 0.05)  # 1% - 5% daily vol
        
        return MarketConditions(
            symbol=symbol,
            bid_price=bid_price,
            ask_price=ask_price,
            mid_price=mid_price,
            spread_bps=spread_bps,
            volume_24h=volume_24h,
            orderbook_depth_bid=orderbook_depth_bid,
            orderbook_depth_ask=orderbook_depth_ask,
            volatility=volatility
        )
    
    def update_market_stress(self, stress_level: float) -> None:
        """Update market stress level (0.0 to 1.0)."""
        self.market_stress_level = max(0.0, min(1.0, stress_level))
        self.logger.info(f"Market stress level updated to {self.market_stress_level:.2f}")
    
    def get_simulation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive simulation statistics."""
        
        if self.total_simulated_orders == 0:
            return {
                "total_orders": 0,
                "avg_slippage_bps": 0.0,
                "total_fees": 0.0,
                "fill_rate": 0.0,
                "avg_latency_ms": 0.0
            }
        
        filled_orders = [e for e in self.simulated_executions if e.status == OrderStatus.FILLED]
        
        return {
            "total_orders": self.total_simulated_orders,
            "avg_slippage_bps": self.total_slippage_bps / self.total_simulated_orders,
            "total_fees": self.total_fees_paid,
            "fill_rate": len(filled_orders) / self.total_simulated_orders,
            "avg_latency_ms": np.mean([e.execution_time_ms for e in self.simulated_executions]),
            "market_stress_level": self.market_stress_level,
            "recent_executions": len([e for e in self.simulated_executions 
                                    if (datetime.now() - e.timestamp).seconds < 3600])
        }
    
    def reset_simulation_state(self) -> None:
        """Reset simulation state for new session."""
        self.simulated_executions.clear()
        self.total_simulated_orders = 0
        self.total_slippage_bps = 0.0
        self.total_fees_paid = 0.0
        self.market_stress_level = 0.0
        
        self.logger.info("Simulation state reset")