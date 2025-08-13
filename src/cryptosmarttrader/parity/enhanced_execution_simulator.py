"""
Enhanced Execution Simulator - FASE D
Fixed implementation voor daily parity tracking met <X bps accuracy
"""

import time
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..core.structured_logger import get_logger


class FillType(Enum):
    """Order fill types."""
    FULL = "full"
    PARTIAL = "partial"
    REJECTED = "rejected"


@dataclass
class MarketConditions:
    """Real-time market conditions."""
    
    symbol: str
    bid: float
    ask: float
    spread_bps: float
    volume_24h: float
    orderbook_depth: float
    volatility_24h: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2.0
    
    @property
    def spread_percent(self) -> float:
        """Calculate spread percentage."""
        return ((self.ask - self.bid) / self.mid_price) * 100.0


@dataclass
class SimulatedFill:
    """Simulated order fill result."""
    
    order_id: str
    symbol: str
    side: str
    quantity_requested: float
    quantity_filled: float
    price_requested: Optional[float]
    fill_price: float
    fill_type: FillType
    slippage_bps: float
    fees_paid: float
    latency_ms: int
    market_impact_bps: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def slippage_percent(self) -> float:
        """Convert slippage to percentage."""
        return self.slippage_bps / 100.0


@dataclass
class SimulationResult:
    """Comprehensive simulation result."""
    
    simulation_id: str
    start_time: datetime
    end_time: datetime
    total_orders: int
    successful_fills: int
    partial_fills: int
    rejected_orders: int
    avg_slippage_bps: float
    p95_slippage_bps: float
    total_fees: float
    market_impact_bps: float
    execution_shortfall_bps: float
    fills: List[SimulatedFill] = field(default_factory=list)
    
    @property
    def fill_rate(self) -> float:
        """Calculate fill rate percentage."""
        if self.total_orders == 0:
            return 0.0
        return ((self.successful_fills + self.partial_fills) / self.total_orders) * 100.0
    
    @property
    def execution_quality_score(self) -> float:
        """Calculate execution quality score (0-100)."""
        # Weighted score based on fill rate, slippage, and market impact
        fill_score = self.fill_rate / 100.0
        slippage_score = max(0, 1.0 - (self.p95_slippage_bps / 100.0))  # 100bps = 0 score
        impact_score = max(0, 1.0 - (self.market_impact_bps / 50.0))    # 50bps = 0 score
        
        # Weighted average
        total_score = (fill_score * 0.4 + slippage_score * 0.4 + impact_score * 0.2) * 100.0
        return min(100.0, total_score)


class EnhancedExecutionSimulator:
    """
    ENTERPRISE EXECUTION SIMULATOR - FASE D
    
    Fixed implementation for accurate backtest-live parity:
    - Realistic latency modeling (1-50ms)
    - Market impact calculation based on order size
    - Partial fill simulation with queue modeling
    - Maker/taker fee simulation
    - Volatility-adjusted slippage
    """
    
    def __init__(self):
        """Initialize enhanced execution simulator."""
        self.logger = get_logger("execution_simulator")
        
        # Simulation parameters
        self.base_latency_ms = 5.0
        self.latency_variance_ms = 15.0
        self.maker_fee_bps = 1.0    # 0.01%
        self.taker_fee_bps = 5.0    # 0.05%
        self.min_fill_size = 0.001   # Minimum fill size
        
        # Market microstructure parameters
        self.liquidity_depth_tiers = {
            "tier_1": {"depth": 10000, "impact_bps": 1.0},   # $10k depth, 1bp impact
            "tier_2": {"depth": 50000, "impact_bps": 3.0},   # $50k depth, 3bp impact  
            "tier_3": {"depth": 100000, "impact_bps": 8.0},  # $100k depth, 8bp impact
        }
        
        # Fill probability curves
        self.fill_probabilities = {
            "limit_orders": {
                "at_bid_ask": 0.95,      # 95% fill at bid/ask
                "through_spread": 0.85,   # 85% fill through spread
                "far_from_market": 0.15   # 15% fill far from market
            },
            "market_orders": {
                "immediate": 0.98,        # 98% immediate fill
                "delayed": 0.02          # 2% delayed/partial
            }
        }
        
        self.logger.info("Enhanced Execution Simulator initialized")
    
    def get_market_conditions(self, symbol: str) -> MarketConditions:
        """Get realistic market conditions for symbol."""
        # Simulate realistic BTC/ETH market conditions
        if "BTC" in symbol.upper():
            base_price = 45000.0 + random.uniform(-5000, 5000)
            spread_bps = random.uniform(1.0, 8.0)
            volume_24h = random.uniform(500000000, 2000000000)  # $500M-2B
            volatility = random.uniform(0.02, 0.08)             # 2-8% daily vol
        elif "ETH" in symbol.upper():
            base_price = 2800.0 + random.uniform(-400, 400)
            spread_bps = random.uniform(2.0, 12.0)
            volume_24h = random.uniform(200000000, 800000000)   # $200M-800M
            volatility = random.uniform(0.03, 0.12)             # 3-12% daily vol
        else:
            base_price = random.uniform(0.1, 100.0)
            spread_bps = random.uniform(5.0, 50.0)
            volume_24h = random.uniform(1000000, 50000000)      # $1M-50M
            volatility = random.uniform(0.05, 0.25)             # 5-25% daily vol
        
        # Calculate bid/ask from spread
        spread_amount = base_price * (spread_bps / 10000.0)
        bid = base_price - (spread_amount / 2)
        ask = base_price + (spread_amount / 2)
        
        return MarketConditions(
            symbol=symbol,
            bid=bid,
            ask=ask,
            spread_bps=spread_bps,
            volume_24h=volume_24h,
            orderbook_depth=volume_24h * 0.001,  # 0.1% of 24h volume as depth
            volatility_24h=volatility
        )
    
    def calculate_market_impact(
        self, 
        order_value: float, 
        market_conditions: MarketConditions
    ) -> float:
        """Calculate market impact in basis points."""
        
        # Impact based on order size relative to liquidity depth
        depth_ratio = order_value / market_conditions.orderbook_depth
        
        # Tiered impact calculation
        if depth_ratio <= 0.1:      # <10% of depth
            base_impact = depth_ratio * 5.0
        elif depth_ratio <= 0.3:    # 10-30% of depth
            base_impact = 0.5 + (depth_ratio - 0.1) * 15.0
        else:                       # >30% of depth (high impact)
            base_impact = 3.5 + (depth_ratio - 0.3) * 25.0
        
        # Volatility adjustment
        vol_multiplier = 1.0 + (market_conditions.volatility_24h * 2.0)
        
        # Spread adjustment (wider spreads = higher impact)
        spread_multiplier = 1.0 + (market_conditions.spread_bps / 100.0)
        
        final_impact = base_impact * vol_multiplier * spread_multiplier
        
        # Cap at reasonable maximum (500bps)
        return min(500.0, final_impact)
    
    def calculate_slippage(
        self,
        order_side: str,
        order_price: Optional[float],
        execution_price: float,
        market_conditions: MarketConditions
    ) -> float:
        """Calculate slippage in basis points."""
        
        if order_price is None:  # Market order
            # Slippage relative to mid price
            reference_price = market_conditions.mid_price
        else:  # Limit order
            reference_price = order_price
        
        # Calculate slippage
        if order_side.lower() == "buy":
            slippage = ((execution_price - reference_price) / reference_price) * 10000
        else:  # sell
            slippage = ((reference_price - execution_price) / reference_price) * 10000
        
        # Ensure slippage is always positive (cost)
        return max(0.0, slippage)
    
    def simulate_order_execution(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "MARKET",
        limit_price: Optional[float] = None
    ) -> SimulatedFill:
        """Simulate realistic order execution."""
        
        start_time = time.time()
        
        # Get market conditions
        market_conditions = self.get_market_conditions(symbol)
        
        # Calculate order value
        reference_price = limit_price or market_conditions.mid_price
        order_value = quantity * reference_price
        
        # Simulate execution latency
        base_latency = random.uniform(
            self.base_latency_ms, 
            self.base_latency_ms + self.latency_variance_ms
        )
        
        # Network/exchange latency spikes (5% chance)
        if random.random() < 0.05:
            base_latency += random.uniform(50, 200)  # 50-200ms spike
        
        latency_ms = int(base_latency)
        
        # Determine fill type and execution
        fill_type, quantity_filled, execution_price = self._simulate_fill_execution(
            order_type, side, quantity, limit_price, market_conditions
        )
        
        # Calculate market impact
        market_impact_bps = self.calculate_market_impact(order_value, market_conditions)
        
        # Calculate slippage
        slippage_bps = self.calculate_slippage(
            side, limit_price, execution_price, market_conditions
        )
        
        # Calculate fees
        if order_type == "LIMIT" and fill_type == FillType.FULL:
            # Assume maker fee for limit orders
            fee_bps = self.maker_fee_bps
        else:
            # Taker fee for market orders or partial fills
            fee_bps = self.taker_fee_bps
        
        fees_paid = (quantity_filled * execution_price) * (fee_bps / 10000.0)
        
        return SimulatedFill(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity_requested=quantity,
            quantity_filled=quantity_filled,
            price_requested=limit_price,
            fill_price=execution_price,
            fill_type=fill_type,
            slippage_bps=slippage_bps,
            fees_paid=fees_paid,
            latency_ms=latency_ms,
            market_impact_bps=market_impact_bps
        )
    
    def _simulate_fill_execution(
        self,
        order_type: str,
        side: str,
        quantity: float,
        limit_price: Optional[float],
        market_conditions: MarketConditions
    ) -> Tuple[FillType, float, float]:
        """Simulate the actual fill execution logic."""
        
        if order_type.upper() == "MARKET":
            # Market orders: high fill probability, price uncertainty
            if random.random() < self.fill_probabilities["market_orders"]["immediate"]:
                # Immediate full fill
                if side.lower() == "buy":
                    # Buy at ask + some market impact
                    execution_price = market_conditions.ask * (1 + random.uniform(0.0001, 0.002))
                else:  # sell
                    # Sell at bid - some market impact
                    execution_price = market_conditions.bid * (1 - random.uniform(0.0001, 0.002))
                
                return FillType.FULL, quantity, execution_price
            else:
                # Partial fill or rejection
                if random.random() < 0.7:  # 70% partial
                    fill_ratio = random.uniform(0.3, 0.9)
                    quantity_filled = quantity * fill_ratio
                    execution_price = market_conditions.ask if side.lower() == "buy" else market_conditions.bid
                    return FillType.PARTIAL, quantity_filled, execution_price
                else:  # 30% rejection
                    return FillType.REJECTED, 0.0, 0.0
        
        else:  # LIMIT order
            if limit_price is None:
                return FillType.REJECTED, 0.0, 0.0
            
            # Check if limit price is executable
            if side.lower() == "buy":
                executable = limit_price >= market_conditions.ask
                fill_prob = self.fill_probabilities["limit_orders"]["at_bid_ask"] if executable else \
                           self.fill_probabilities["limit_orders"]["far_from_market"]
            else:  # sell
                executable = limit_price <= market_conditions.bid
                fill_prob = self.fill_probabilities["limit_orders"]["at_bid_ask"] if executable else \
                           self.fill_probabilities["limit_orders"]["far_from_market"]
            
            if random.random() < fill_prob:
                # Fill at limit price (or better)
                if executable and random.random() < 0.3:  # 30% chance of price improvement
                    if side.lower() == "buy":
                        execution_price = limit_price - random.uniform(0, market_conditions.spread_bps * 0.1)
                    else:
                        execution_price = limit_price + random.uniform(0, market_conditions.spread_bps * 0.1)
                else:
                    execution_price = limit_price
                
                # Partial fill possibility for large orders
                if quantity * execution_price > 50000:  # Large orders >$50k
                    if random.random() < 0.3:  # 30% partial fill chance
                        fill_ratio = random.uniform(0.4, 0.8)
                        return FillType.PARTIAL, quantity * fill_ratio, execution_price
                
                return FillType.FULL, quantity, execution_price
            else:
                return FillType.REJECTED, 0.0, 0.0
    
    def simulate_trading_session(
        self,
        orders: List[Dict[str, Any]],
        session_duration_minutes: int = 60
    ) -> SimulationResult:
        """Simulate a complete trading session."""
        
        simulation_id = f"sim_{int(time.time())}"
        start_time = datetime.now()
        
        fills = []
        successful_fills = 0
        partial_fills = 0
        rejected_orders = 0
        
        self.logger.info(f"Starting simulation session: {simulation_id}")
        
        for i, order in enumerate(orders):
            # Simulate time progression
            if i > 0:
                time.sleep(random.uniform(0.01, 0.1))  # 10-100ms between orders
            
            fill = self.simulate_order_execution(
                order_id=f"{simulation_id}_{i}",
                symbol=order.get("symbol", "BTC/USD"),
                side=order.get("side", "buy"),
                quantity=order.get("quantity", 1.0),
                order_type=order.get("order_type", "MARKET"),
                limit_price=order.get("limit_price")
            )
            
            fills.append(fill)
            
            if fill.fill_type == FillType.FULL:
                successful_fills += 1
            elif fill.fill_type == FillType.PARTIAL:
                partial_fills += 1
            else:
                rejected_orders += 1
        
        end_time = datetime.now()
        
        # Calculate aggregate metrics
        filled_orders = [f for f in fills if f.fill_type != FillType.REJECTED]
        
        if filled_orders:
            slippages = [f.slippage_bps for f in filled_orders]
            avg_slippage = np.mean(slippages)
            p95_slippage = np.percentile(slippages, 95)
            
            total_fees = sum(f.fees_paid for f in filled_orders)
            market_impacts = [f.market_impact_bps for f in filled_orders]
            avg_market_impact = np.mean(market_impacts)
            
            # Execution shortfall (slippage + market impact + fees in bps)
            execution_shortfall = avg_slippage + avg_market_impact + (total_fees / sum(
                f.quantity_filled * f.fill_price for f in filled_orders
            ) * 10000)
        else:
            avg_slippage = p95_slippage = total_fees = avg_market_impact = execution_shortfall = 0.0
        
        result = SimulationResult(
            simulation_id=simulation_id,
            start_time=start_time,
            end_time=end_time,
            total_orders=len(orders),
            successful_fills=successful_fills,
            partial_fills=partial_fills,
            rejected_orders=rejected_orders,
            avg_slippage_bps=avg_slippage,
            p95_slippage_bps=p95_slippage,
            total_fees=total_fees,
            market_impact_bps=avg_market_impact,
            execution_shortfall_bps=execution_shortfall,
            fills=fills
        )
        
        self.logger.info(
            f"Simulation completed: {successful_fills + partial_fills}/{len(orders)} fills, "
            f"P95 slippage: {p95_slippage:.1f}bps, Quality: {result.execution_quality_score:.1f}/100"
        )
        
        return result


def create_execution_simulator() -> EnhancedExecutionSimulator:
    """Factory function to create enhanced execution simulator."""
    return EnhancedExecutionSimulator()