"""
Execution Policy System

Advanced execution policies with spread/volume/depth gating, slippage budgeting,
post-only/TWAP strategies, and real-time telemetry tracking.
"""

import asyncio
import time
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ExecutionStrategy(Enum):
    """Execution strategy types"""
    MARKET = "market"           # Immediate market order
    POST_ONLY = "post_only"     # Maker-only orders
    TWAP = "twap"              # Time-weighted average price
    ICEBERG = "iceberg"        # Large order fragmentation
    ADAPTIVE = "adaptive"      # Dynamic strategy selection

class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"

class GatingReason(Enum):
    """Reasons for execution gating"""
    SPREAD_TOO_WIDE = "spread_too_wide"
    LOW_VOLUME = "low_volume"
    SHALLOW_DEPTH = "shallow_depth"
    HIGH_SLIPPAGE = "high_slippage"
    MARKET_CLOSED = "market_closed"
    RISK_LIMIT = "risk_limit"
    CIRCUIT_BREAKER = "circuit_breaker"

@dataclass
class MarketConditions:
    """Current market conditions for execution decision"""
    symbol: str
    timestamp: datetime
    
    # Spread data
    bid_price: float
    ask_price: float
    spread_bps: float
    
    # Volume data
    volume_24h: float
    volume_1h: float
    avg_trade_size: float
    
    # Order book depth
    bid_depth_5: float  # Total volume in top 5 bid levels
    ask_depth_5: float  # Total volume in top 5 ask levels
    bid_depth_10: float
    ask_depth_10: float
    
    # Market impact estimation
    estimated_slippage_bps: float
    market_impact_bps: float
    
    # Volatility
    volatility_1h: float
    volatility_24h: float

@dataclass
class ExecutionGates:
    """Execution gating thresholds"""
    
    # Spread gates
    max_spread_bps: float = 20.0  # Maximum allowed spread
    max_spread_percentage: float = 0.5  # 0.5% max spread
    
    # Volume gates
    min_volume_24h_usd: float = 100000.0  # $100k minimum daily volume
    min_volume_1h_usd: float = 5000.0     # $5k minimum hourly volume
    min_avg_trade_size_usd: float = 100.0  # $100 minimum average trade
    
    # Depth gates
    min_depth_5_levels_usd: float = 10000.0  # $10k minimum in top 5 levels
    min_depth_ratio: float = 2.0            # Depth must be 2x order size
    
    # Slippage gates
    max_slippage_bps: float = 50.0          # 50 bps maximum slippage
    max_market_impact_bps: float = 30.0     # 30 bps maximum market impact
    
    # Volatility gates
    max_volatility_1h: float = 0.05         # 5% max hourly volatility
    pause_on_high_volatility: bool = True

@dataclass
class SlippageBudget:
    """Slippage budget configuration"""
    
    # Budget limits
    daily_budget_bps: float = 100.0         # 100 bps daily budget
    hourly_budget_bps: float = 25.0         # 25 bps hourly budget
    per_trade_budget_bps: float = 50.0      # 50 bps per trade budget
    
    # Tracking
    daily_consumed_bps: float = 0.0
    hourly_consumed_bps: float = 0.0
    last_reset_daily: datetime = field(default_factory=datetime.now)
    last_reset_hourly: datetime = field(default_factory=datetime.now)
    
    # Compliance targets
    compliance_target_95th_percentile: float = 45.0  # 95th percentile target
    compliance_target_99th_percentile: float = 75.0  # 99th percentile target

@dataclass
class ExecutionOrder:
    """Execution order specification"""
    symbol: str
    side: OrderSide
    quantity: float
    target_price: Optional[float] = None
    
    # Execution parameters
    strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE
    max_slippage_bps: float = 50.0
    timeout_seconds: int = 300
    
    # TWAP parameters
    twap_duration_minutes: int = 10
    twap_slices: int = 5
    
    # Iceberg parameters
    iceberg_slice_size: float = 0.1  # 10% of order size per slice
    iceberg_randomization: float = 0.2  # 20% randomization
    
    # Metadata
    order_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 1  # 1=low, 5=high

@dataclass
class ExecutionResult:
    """Result of order execution"""
    order_id: str
    symbol: str
    side: OrderSide
    
    # Execution details
    requested_quantity: float
    filled_quantity: float
    average_price: float
    
    # Performance metrics
    realized_slippage_bps: float
    market_impact_bps: float
    execution_time_seconds: float
    
    # Cost breakdown
    trading_fees: float
    market_impact_cost: float
    total_cost_bps: float
    
    # Metadata
    strategy_used: ExecutionStrategy
    execution_timestamp: datetime
    market_conditions: MarketConditions
    
    # Status
    status: str = "completed"  # completed, partial, failed, cancelled
    error_message: Optional[str] = None

class ExecutionPolicyEngine:
    """
    Advanced execution policy engine with comprehensive gating and budget management
    """
    
    def __init__(self, 
                 environment_manager: Optional[Any] = None,
                 metrics_collector: Optional[Any] = None):
        
        self.environment_manager = environment_manager
        self.metrics_collector = metrics_collector
        
        # Configuration
        self.execution_gates = ExecutionGates()
        self.slippage_budget = SlippageBudget()
        
        # Execution tracking
        self.pending_orders: Dict[str, ExecutionOrder] = {}
        self.execution_history: List[ExecutionResult] = []
        self.slippage_history: List[Tuple[datetime, float]] = []
        
        # Real-time telemetry
        self.telemetry_data: Dict[str, Any] = {
            'total_executions': 0,
            'successful_executions': 0,
            'gated_executions': 0,
            'average_slippage_bps': 0.0,
            'slippage_95th_percentile': 0.0,
            'budget_utilization': 0.0
        }
        
        # Load environment-specific configuration
        self._load_environment_config()
        
        logger.info("Execution Policy Engine initialized")
    
    def _load_environment_config(self):
        """Load environment-specific execution configuration"""
        
        if not self.environment_manager:
            return
        
        # Get slippage budget from feature flags
        slippage_budget_bps = self.environment_manager.get_feature_flag_value(
            "slippage_budget_bps", default=50
        )
        if slippage_budget_bps:
            self.slippage_budget.per_trade_budget_bps = float(slippage_budget_bps)
            self.execution_gates.max_slippage_bps = float(slippage_budget_bps)
        
        # Check if strict execution policy is enabled
        strict_policy = self.environment_manager.is_feature_enabled("execution_policy_strict")
        if strict_policy:
            # Tighter gates for production
            self.execution_gates.max_spread_bps = 15.0
            self.execution_gates.max_slippage_bps = 30.0
            self.execution_gates.min_volume_24h_usd = 500000.0
        
        logger.info(f"Loaded execution config - Slippage budget: {self.slippage_budget.per_trade_budget_bps} bps")
    
    def evaluate_execution_feasibility(self, 
                                     order: ExecutionOrder, 
                                     market_conditions: MarketConditions) -> Tuple[bool, List[GatingReason]]:
        """Evaluate if order execution should proceed based on current conditions"""
        
        can_execute = True
        gating_reasons = []
        
        # Check spread gates
        if market_conditions.spread_bps > self.execution_gates.max_spread_bps:
            can_execute = False
            gating_reasons.append(GatingReason.SPREAD_TOO_WIDE)
        
        # Check volume gates
        if market_conditions.volume_24h < self.execution_gates.min_volume_24h_usd:
            can_execute = False
            gating_reasons.append(GatingReason.LOW_VOLUME)
        
        if market_conditions.volume_1h < self.execution_gates.min_volume_1h_usd:
            can_execute = False
            gating_reasons.append(GatingReason.LOW_VOLUME)
        
        # Check depth gates
        required_depth = order.quantity * (order.target_price or market_conditions.ask_price)
        min_required_depth = required_depth * self.execution_gates.min_depth_ratio
        
        available_depth = (market_conditions.bid_depth_5 if order.side == OrderSide.SELL 
                          else market_conditions.ask_depth_5)
        
        if available_depth < min_required_depth:
            can_execute = False
            gating_reasons.append(GatingReason.SHALLOW_DEPTH)
        
        # Check slippage gates
        if market_conditions.estimated_slippage_bps > self.execution_gates.max_slippage_bps:
            can_execute = False
            gating_reasons.append(GatingReason.HIGH_SLIPPAGE)
        
        # Check slippage budget
        if not self._check_slippage_budget(order.max_slippage_bps):
            can_execute = False
            gating_reasons.append(GatingReason.HIGH_SLIPPAGE)
        
        # Check volatility gates
        if (self.execution_gates.pause_on_high_volatility and 
            market_conditions.volatility_1h > self.execution_gates.max_volatility_1h):
            can_execute = False
            gating_reasons.append(GatingReason.CIRCUIT_BREAKER)
        
        # Record gating decision
        if not can_execute:
            self.telemetry_data['gated_executions'] += 1
            
            if self.metrics_collector:
                self.metrics_collector.record_execution_gate(
                    symbol=order.symbol,
                    side=order.side.value,
                    gating_reasons=[reason.value for reason in gating_reasons]
                )
        
        return can_execute, gating_reasons
    
    def _check_slippage_budget(self, required_slippage_bps: float) -> bool:
        """Check if slippage budget allows for execution"""
        
        # Reset budgets if needed
        self._reset_slippage_budgets()
        
        # Check daily budget
        if (self.slippage_budget.daily_consumed_bps + required_slippage_bps > 
            self.slippage_budget.daily_budget_bps):
            return False
        
        # Check hourly budget
        if (self.slippage_budget.hourly_consumed_bps + required_slippage_bps > 
            self.slippage_budget.hourly_budget_bps):
            return False
        
        return True
    
    def _reset_slippage_budgets(self):
        """Reset slippage budgets if time periods have elapsed"""
        
        now = datetime.now()
        
        # Reset daily budget
        if (now - self.slippage_budget.last_reset_daily).days >= 1:
            self.slippage_budget.daily_consumed_bps = 0.0
            self.slippage_budget.last_reset_daily = now
        
        # Reset hourly budget
        if (now - self.slippage_budget.last_reset_hourly).total_seconds() >= 3600:
            self.slippage_budget.hourly_consumed_bps = 0.0
            self.slippage_budget.last_reset_hourly = now
    
    def select_execution_strategy(self, 
                                order: ExecutionOrder, 
                                market_conditions: MarketConditions) -> ExecutionStrategy:
        """Select optimal execution strategy based on market conditions"""
        
        if order.strategy != ExecutionStrategy.ADAPTIVE:
            return order.strategy
        
        # Strategy selection logic
        order_value = order.quantity * (order.target_price or market_conditions.ask_price)
        
        # Large orders → TWAP or Iceberg
        if order_value > 50000:  # $50k+
            if market_conditions.volume_1h > order_value * 10:
                return ExecutionStrategy.TWAP
            else:
                return ExecutionStrategy.ICEBERG
        
        # Medium orders with good depth → Post-only
        elif (order_value > 5000 and 
              market_conditions.bid_depth_5 > order_value * 3 and
              market_conditions.spread_bps < 10):
            return ExecutionStrategy.POST_ONLY
        
        # Small orders or urgent → Market
        else:
            return ExecutionStrategy.MARKET
    
    async def execute_order(self, 
                          order: ExecutionOrder, 
                          market_conditions: MarketConditions) -> ExecutionResult:
        """Execute order using selected strategy"""
        
        start_time = time.time()
        
        # Evaluate execution feasibility
        can_execute, gating_reasons = self.evaluate_execution_feasibility(order, market_conditions)
        
        if not can_execute:
            return ExecutionResult(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                requested_quantity=order.quantity,
                filled_quantity=0.0,
                average_price=0.0,
                realized_slippage_bps=0.0,
                market_impact_bps=0.0,
                execution_time_seconds=time.time() - start_time,
                trading_fees=0.0,
                market_impact_cost=0.0,
                total_cost_bps=0.0,
                strategy_used=order.strategy,
                execution_timestamp=datetime.now(),
                market_conditions=market_conditions,
                status="gated",
                error_message=f"Execution gated: {[r.value for r in gating_reasons]}"
            )
        
        # Select execution strategy
        selected_strategy = self.select_execution_strategy(order, market_conditions)
        
        # Execute based on strategy
        if selected_strategy == ExecutionStrategy.MARKET:
            result = await self._execute_market_order(order, market_conditions)
        elif selected_strategy == ExecutionStrategy.POST_ONLY:
            result = await self._execute_post_only_order(order, market_conditions)
        elif selected_strategy == ExecutionStrategy.TWAP:
            result = await self._execute_twap_order(order, market_conditions)
        elif selected_strategy == ExecutionStrategy.ICEBERG:
            result = await self._execute_iceberg_order(order, market_conditions)
        else:
            result = await self._execute_market_order(order, market_conditions)  # Fallback
        
        # Update slippage budget
        self._consume_slippage_budget(result.realized_slippage_bps)
        
        # Record execution
        self._record_execution(result)
        
        return result
    
    async def _execute_market_order(self, 
                                  order: ExecutionOrder, 
                                  market_conditions: MarketConditions) -> ExecutionResult:
        """Execute immediate market order"""
        
        # Simulate market order execution
        execution_price = (market_conditions.ask_price if order.side == OrderSide.BUY 
                          else market_conditions.bid_price)
        
        # Calculate slippage
        reference_price = (market_conditions.bid_price + market_conditions.ask_price) / 2
        slippage_bps = abs(execution_price - reference_price) / reference_price * 10000
        
        # Add market impact
        order_value = order.quantity * execution_price
        impact_factor = min(order_value / market_conditions.volume_1h, 0.1)  # Cap at 10%
        market_impact_bps = impact_factor * 50  # Up to 50 bps impact
        
        total_slippage = slippage_bps + market_impact_bps
        
        return ExecutionResult(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            requested_quantity=order.quantity,
            filled_quantity=order.quantity,
            average_price=execution_price,
            realized_slippage_bps=total_slippage,
            market_impact_bps=market_impact_bps,
            execution_time_seconds=0.5,  # Fast execution
            trading_fees=order_value * 0.001,  # 0.1% fee
            market_impact_cost=order_value * (market_impact_bps / 10000),
            total_cost_bps=total_slippage + 10,  # Slippage + fees
            strategy_used=ExecutionStrategy.MARKET,
            execution_timestamp=datetime.now(),
            market_conditions=market_conditions,
            status="completed"
        )
    
    async def _execute_post_only_order(self, 
                                     order: ExecutionOrder, 
                                     market_conditions: MarketConditions) -> ExecutionResult:
        """Execute post-only (maker) order"""
        
        # Post at mid-price or better
        mid_price = (market_conditions.bid_price + market_conditions.ask_price) / 2
        
        # Wait for fill simulation
        await asyncio.sleep(5)  # Simulate waiting for fill
        
        # Assume filled at mid-price (best case for post-only)
        execution_price = mid_price
        
        # Minimal slippage for post-only
        slippage_bps = market_conditions.spread_bps / 4  # Quarter of spread
        
        order_value = order.quantity * execution_price
        
        return ExecutionResult(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            requested_quantity=order.quantity,
            filled_quantity=order.quantity,
            average_price=execution_price,
            realized_slippage_bps=slippage_bps,
            market_impact_bps=0.0,  # No market impact for post-only
            execution_time_seconds=5.0,
            trading_fees=order_value * 0.0005,  # Lower maker fees
            market_impact_cost=0.0,
            total_cost_bps=slippage_bps + 5,  # Slippage + lower fees
            strategy_used=ExecutionStrategy.POST_ONLY,
            execution_timestamp=datetime.now(),
            market_conditions=market_conditions,
            status="completed"
        )
    
    async def _execute_twap_order(self, 
                                order: ExecutionOrder, 
                                market_conditions: MarketConditions) -> ExecutionResult:
        """Execute TWAP (Time-Weighted Average Price) order"""
        
        slice_size = order.quantity / order.twap_slices
        slice_duration = order.twap_duration_minutes * 60 / order.twap_slices
        
        total_filled = 0.0
        weighted_price = 0.0
        total_slippage = 0.0
        
        for slice_num in range(order.twap_slices):
            # Execute slice
            await asyncio.sleep(slice_duration / 60)  # Scale down for simulation
            
            # Vary execution price slightly for each slice
            price_variation = np.random.normal(0, 0.001)  # 0.1% variation
            execution_price = (market_conditions.ask_price if order.side == OrderSide.BUY 
                             else market_conditions.bid_price) * (1 + price_variation)
            
            filled_quantity = slice_size
            total_filled += filled_quantity
            weighted_price += execution_price * filled_quantity
            
            # Calculate slice slippage
            reference_price = (market_conditions.bid_price + market_conditions.ask_price) / 2
            slice_slippage = abs(execution_price - reference_price) / reference_price * 10000
            total_slippage += slice_slippage
        
        average_price = weighted_price / total_filled
        average_slippage = total_slippage / order.twap_slices
        
        order_value = total_filled * average_price
        
        return ExecutionResult(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            requested_quantity=order.quantity,
            filled_quantity=total_filled,
            average_price=average_price,
            realized_slippage_bps=average_slippage,
            market_impact_bps=average_slippage * 0.6,  # 60% of slippage is impact
            execution_time_seconds=order.twap_duration_minutes * 60,
            trading_fees=order_value * 0.0008,  # Mid-tier fees
            market_impact_cost=order_value * (average_slippage * 0.6 / 10000),
            total_cost_bps=average_slippage + 8,
            strategy_used=ExecutionStrategy.TWAP,
            execution_timestamp=datetime.now(),
            market_conditions=market_conditions,
            status="completed"
        )
    
    async def _execute_iceberg_order(self, 
                                   order: ExecutionOrder, 
                                   market_conditions: MarketConditions) -> ExecutionResult:
        """Execute iceberg order (large order fragmentation)"""
        
        slice_size = order.quantity * order.iceberg_slice_size
        total_filled = 0.0
        weighted_price = 0.0
        total_slippage = 0.0
        execution_count = 0
        
        remaining_quantity = order.quantity
        
        while remaining_quantity > 0 and execution_count < 20:  # Max 20 slices
            # Randomize slice size
            randomization = np.random.uniform(-order.iceberg_randomization, order.iceberg_randomization)
            current_slice = min(slice_size * (1 + randomization), remaining_quantity)
            
            # Execute slice with slight delay
            await asyncio.sleep(np.random.uniform(1, 5))
            
            # Price impact accumulation
            cumulative_impact = execution_count * 2  # 2 bps per slice
            execution_price = (market_conditions.ask_price if order.side == OrderSide.BUY 
                             else market_conditions.bid_price) * (1 + cumulative_impact / 10000)
            
            total_filled += current_slice
            weighted_price += execution_price * current_slice
            remaining_quantity -= current_slice
            execution_count += 1
            
            # Calculate slice slippage
            reference_price = (market_conditions.bid_price + market_conditions.ask_price) / 2
            slice_slippage = abs(execution_price - reference_price) / reference_price * 10000
            total_slippage += slice_slippage
        
        average_price = weighted_price / total_filled
        average_slippage = total_slippage / execution_count
        market_impact = execution_count * 2  # Cumulative impact
        
        order_value = total_filled * average_price
        
        return ExecutionResult(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            requested_quantity=order.quantity,
            filled_quantity=total_filled,
            average_price=average_price,
            realized_slippage_bps=average_slippage,
            market_impact_bps=market_impact,
            execution_time_seconds=execution_count * 3,  # 3s per slice average
            trading_fees=order_value * 0.001,
            market_impact_cost=order_value * (market_impact / 10000),
            total_cost_bps=average_slippage + 10,
            strategy_used=ExecutionStrategy.ICEBERG,
            execution_timestamp=datetime.now(),
            market_conditions=market_conditions,
            status="completed"
        )
    
    def _consume_slippage_budget(self, slippage_bps: float):
        """Consume slippage budget for executed trade"""
        self.slippage_budget.daily_consumed_bps += slippage_bps
        self.slippage_budget.hourly_consumed_bps += slippage_bps
    
    def _record_execution(self, result: ExecutionResult):
        """Record execution for telemetry and analysis"""
        
        # Add to history
        self.execution_history.append(result)
        self.slippage_history.append((result.execution_timestamp, result.realized_slippage_bps))
        
        # Update telemetry
        self.telemetry_data['total_executions'] += 1
        if result.status == "completed":
            self.telemetry_data['successful_executions'] += 1
        
        # Calculate running statistics
        recent_slippages = [s for _, s in self.slippage_history[-100:]]  # Last 100 trades
        if recent_slippages:
            self.telemetry_data['average_slippage_bps'] = np.mean(recent_slippages)
            self.telemetry_data['slippage_95th_percentile'] = np.percentile(recent_slippages, 95)
        
        # Budget utilization
        self.telemetry_data['budget_utilization'] = (
            self.slippage_budget.daily_consumed_bps / self.slippage_budget.daily_budget_bps
        )
        
        # Record metrics
        if self.metrics_collector:
            self.metrics_collector.record_trade_execution(
                symbol=result.symbol,
                side=result.side.value,
                order_type=result.strategy_used.value,
                status=result.status
            )
            
            self.metrics_collector.record_slippage(
                symbol=result.symbol,
                order_type=result.strategy_used.value,
                side=result.side.value,
                slippage_bps=result.realized_slippage_bps
            )
        
        logger.info(f"Execution recorded: {result.symbol} {result.side.value} "
                   f"{result.filled_quantity:.4f} @ {result.average_price:.4f} "
                   f"(slippage: {result.realized_slippage_bps:.1f} bps)")
    
    def get_execution_telemetry(self) -> Dict[str, Any]:
        """Get real-time execution telemetry"""
        
        # Calculate compliance metrics
        recent_slippages = [s for _, s in self.slippage_history[-1000:]]  # Last 1000 trades
        
        compliance_95th = 0.0
        compliance_99th = 0.0
        if recent_slippages:
            p95 = np.percentile(recent_slippages, 95)
            p99 = np.percentile(recent_slippages, 99)
            
            compliance_95th = (p95 <= self.slippage_budget.compliance_target_95th_percentile)
            compliance_99th = (p99 <= self.slippage_budget.compliance_target_99th_percentile)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'execution_statistics': self.telemetry_data,
            
            'slippage_compliance': {
                '95th_percentile_target': self.slippage_budget.compliance_target_95th_percentile,
                '99th_percentile_target': self.slippage_budget.compliance_target_99th_percentile,
                '95th_percentile_actual': np.percentile(recent_slippages, 95) if recent_slippages else 0,
                '99th_percentile_actual': np.percentile(recent_slippages, 99) if recent_slippages else 0,
                '95th_percentile_compliant': compliance_95th,
                '99th_percentile_compliant': compliance_99th
            },
            
            'budget_status': {
                'daily_budget_bps': self.slippage_budget.daily_budget_bps,
                'daily_consumed_bps': self.slippage_budget.daily_consumed_bps,
                'daily_remaining_bps': self.slippage_budget.daily_budget_bps - self.slippage_budget.daily_consumed_bps,
                'hourly_budget_bps': self.slippage_budget.hourly_budget_bps,
                'hourly_consumed_bps': self.slippage_budget.hourly_consumed_bps,
                'hourly_remaining_bps': self.slippage_budget.hourly_budget_bps - self.slippage_budget.hourly_consumed_bps,
                'utilization_percentage': self.telemetry_data['budget_utilization'] * 100
            },
            
            'execution_gates': {
                'max_spread_bps': self.execution_gates.max_spread_bps,
                'max_slippage_bps': self.execution_gates.max_slippage_bps,
                'min_volume_24h_usd': self.execution_gates.min_volume_24h_usd,
                'min_depth_5_levels_usd': self.execution_gates.min_depth_5_levels_usd
            },
            
            'recent_performance': {
                'total_executions_24h': len([r for r in self.execution_history 
                                           if (datetime.now() - r.execution_timestamp).days < 1]),
                'success_rate_24h': (
                    len([r for r in self.execution_history 
                        if (datetime.now() - r.execution_timestamp).days < 1 and r.status == "completed"]) /
                    max(len([r for r in self.execution_history 
                           if (datetime.now() - r.execution_timestamp).days < 1]), 1) * 100
                )
            }
        }
    
    def validate_slippage_compliance(self) -> Dict[str, Any]:
        """Validate slippage compliance against targets"""
        
        recent_slippages = [s for _, s in self.slippage_history[-1000:]]
        
        if not recent_slippages:
            return {
                'compliant': True,
                'reason': 'No recent trades to analyze',
                'sample_size': 0
            }
        
        p95 = np.percentile(recent_slippages, 95)
        p99 = np.percentile(recent_slippages, 99)
        
        compliant_95th = p95 <= self.slippage_budget.compliance_target_95th_percentile
        compliant_99th = p99 <= self.slippage_budget.compliance_target_99th_percentile
        
        return {
            'compliant': compliant_95th and compliant_99th,
            'sample_size': len(recent_slippages),
            '95th_percentile': {
                'target': self.slippage_budget.compliance_target_95th_percentile,
                'actual': p95,
                'compliant': compliant_95th
            },
            '99th_percentile': {
                'target': self.slippage_budget.compliance_target_99th_percentile,
                'actual': p99,
                'compliant': compliant_99th
            },
            'average_slippage': np.mean(recent_slippages),
            'max_slippage': np.max(recent_slippages),
            'violations': len([s for s in recent_slippages if s > self.slippage_budget.per_trade_budget_bps])
        }