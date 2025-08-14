"""
Advanced Execution Simulator - FASE D IMPLEMENTATION
High-fidelity simulation of real execution with comprehensive tracking error monitoring
"""

import time
import json
import random
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import numpy as np
import logging
from pathlib import Path

from ..core.structured_logger import get_logger
from ..observability.metrics import PrometheusMetrics
from ..execution.hard_execution_policy import OrderRequest, MarketConditions, OrderSide

logger = get_logger(__name__)


class ExecutionStatus(Enum):
    """Execution status enumeration"""
    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class SlippageComponent(Enum):
    """Components contributing to slippage"""
    SPREAD = "spread"
    MARKET_IMPACT = "market_impact"
    LATENCY = "latency"
    QUEUE_POSITION = "queue_position"
    VOLATILITY = "volatility"
    TIMING = "timing"


@dataclass
class ExecutionFill:
    """Individual fill within an execution"""
    fill_id: str
    quantity: float
    price: float
    timestamp: datetime
    fee_rate: float
    fee_amount: float
    liquidity_flag: str  # "maker" or "taker"


@dataclass
class SimulatedExecution:
    """Result of simulated execution"""
    order_id: str
    symbol: str
    side: str
    requested_quantity: float
    filled_quantity: float
    average_fill_price: float
    total_fees: float
    execution_time_ms: int
    status: ExecutionStatus
    fills: List[ExecutionFill]
    slippage_bps: float
    slippage_breakdown: Dict[str, float]
    market_conditions: MarketConditions
    timestamp: datetime
    
    # Tracking error components
    expected_price: float
    actual_price: float
    price_impact_bps: float
    timing_impact_bps: float
    liquidity_impact_bps: float


@dataclass
class MarketMicrostructure:
    """Market microstructure parameters for realistic simulation"""
    
    # Order book parameters
    tick_size: float = 0.01
    min_size: float = 0.001
    max_size: float = 100.0
    
    # Latency parameters
    order_latency_ms: Tuple[float, float] = (5, 50)  # min, max latency
    cancel_latency_ms: Tuple[float, float] = (2, 20)
    market_data_latency_ms: float = 3.0
    
    # Queue dynamics
    queue_decay_rate: float = 0.1  # Orders ahead decay rate
    fill_probability: float = 0.85  # Probability of fill at touch
    
    # Fee structure
    maker_fee_rate: float = 0.0016  # 0.16% maker fee
    taker_fee_rate: float = 0.0026  # 0.26% taker fee
    
    # Market impact parameters
    impact_coefficient: float = 0.1  # Square root impact coefficient
    impact_decay_halflife: float = 300.0  # 5 minutes in seconds
    
    # Volatility parameters
    intraday_volatility: float = 0.02  # 2% daily volatility
    microstructure_noise: float = 0.0001  # Price noise


class AdvancedExecutionSimulator:
    """
    ADVANCED EXECUTION SIMULATOR - FASE D IMPLEMENTATION
    
    Features:
    ✅ High-fidelity execution modeling with fees/partial-fills/latency
    ✅ Realistic market microstructure simulation
    ✅ Queue effects and order book dynamics
    ✅ Comprehensive slippage component attribution
    ✅ Tracking error calculation in basis points
    ✅ Real-time drift monitoring with auto-disable
    ✅ Multi-strategy execution support
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Market microstructure model
        self.microstructure = MarketMicrostructure(
            tick_size=self.config.get('tick_size', 0.01),
            maker_fee_rate=self.config.get('maker_fee_rate', 0.0016),
            taker_fee_rate=self.config.get('taker_fee_rate', 0.0026),
            impact_coefficient=self.config.get('impact_coefficient', 0.1)
        )
        
        # Tracking error monitoring
        self.tracking_error_threshold_bps = self.config.get('tracking_error_threshold_bps', 20.0)
        self.auto_disable_threshold_bps = self.config.get('auto_disable_threshold_bps', 100.0)
        self.tracking_window_hours = self.config.get('tracking_window_hours', 24)
        
        # Execution history for tracking error calculation
        self.execution_history: List[SimulatedExecution] = []
        self.daily_tracking_errors: List[Tuple[datetime, float]] = []
        
        # Performance tracking
        self.simulation_enabled = True
        self.total_executions = 0
        self.total_volume_simulated = 0.0
        self.cumulative_tracking_error_bps = 0.0
        
        # Metrics integration
        self.metrics = PrometheusMetrics.get_instance()
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info("AdvancedExecutionSimulator initialized", extra={
            'tracking_error_threshold_bps': self.tracking_error_threshold_bps,
            'auto_disable_threshold_bps': self.auto_disable_threshold_bps,
            'maker_fee_rate': self.microstructure.maker_fee_rate,
            'impact_coefficient': self.microstructure.impact_coefficient
        })
    
    async def simulate_execution(
        self, 
        order_request: OrderRequest, 
        market_conditions: MarketConditions,
        expected_execution_price: Optional[float] = None
    ) -> SimulatedExecution:
        """
        Simulate realistic order execution with comprehensive modeling
        """
        
        if not self.simulation_enabled:
            raise RuntimeError("Execution simulator disabled due to high tracking error")
        
        start_time = time.time()
        
        with self._lock:
            self.total_executions += 1
        
        # Determine expected execution price
        if expected_execution_price is None:
            if order_request.side == OrderSide.BUY:
                expected_execution_price = market_conditions.ask_price
            else:
                expected_execution_price = market_conditions.bid_price
        
        # Simulate execution latency
        latency_ms = np.random.uniform(*self.microstructure.order_latency_ms)
        await asyncio.sleep(latency_ms / 1000.0)
        
        # Calculate market impact
        impact_bps = self._calculate_market_impact(order_request, market_conditions)
        
        # Calculate queue effects for limit orders
        queue_delay_ms = 0
        queue_slippage_bps = 0
        if order_request.price is not None:  # Limit order
            queue_delay_ms, queue_slippage_bps = self._simulate_queue_effects(
                order_request, market_conditions
            )
        
        # Simulate order fills
        fills = self._simulate_fills(order_request, market_conditions, impact_bps)
        
        # Calculate execution metrics
        total_filled = sum(fill.quantity for fill in fills)
        if total_filled > 0:
            avg_price = sum(fill.price * fill.quantity for fill in fills) / total_filled
            total_fees = sum(fill.fee_amount for fill in fills)
        else:
            avg_price = expected_execution_price
            total_fees = 0.0
        
        # Calculate slippage components
        slippage_breakdown = self._calculate_slippage_breakdown(
            expected_execution_price, avg_price, impact_bps, queue_slippage_bps, market_conditions
        )
        
        total_slippage_bps = sum(slippage_breakdown.values())
        
        # Determine execution status
        fill_ratio = total_filled / order_request.quantity
        if fill_ratio >= 0.99:
            status = ExecutionStatus.FILLED
        elif fill_ratio > 0:
            status = ExecutionStatus.PARTIAL
        else:
            status = ExecutionStatus.REJECTED
        
        execution_time_ms = int((time.time() - start_time) * 1000 + latency_ms + queue_delay_ms)
        
        # Create execution result
        execution = SimulatedExecution(
            order_id=order_request.client_order_id or f"sim_{int(time.time()*1000)}",
            symbol=order_request.symbol,
            side=order_request.side.value,
            requested_quantity=order_request.quantity,
            filled_quantity=total_filled,
            average_fill_price=avg_price,
            total_fees=total_fees,
            execution_time_ms=execution_time_ms,
            status=status,
            fills=fills,
            slippage_bps=total_slippage_bps,
            slippage_breakdown=slippage_breakdown,
            market_conditions=market_conditions,
            timestamp=datetime.now(),
            expected_price=expected_execution_price,
            actual_price=avg_price,
            price_impact_bps=impact_bps,
            timing_impact_bps=slippage_breakdown.get('timing', 0.0),
            liquidity_impact_bps=slippage_breakdown.get('market_impact', 0.0)
        )
        
        # Record execution for tracking error calculation
        self._record_execution(execution)
        
        # Update metrics
        self._update_metrics(execution)
        
        logger.info("Execution simulated", extra={
            'order_id': execution.order_id,
            'symbol': execution.symbol,
            'filled_quantity': execution.filled_quantity,
            'avg_price': execution.average_fill_price,
            'slippage_bps': execution.slippage_bps,
            'execution_time_ms': execution.execution_time_ms,
            'status': execution.status.value
        })
        
        return execution
    
    def _calculate_market_impact(self, order_request: OrderRequest, market_conditions: MarketConditions) -> float:
        """Calculate market impact in basis points"""
        
        # Order value
        order_value = order_request.quantity * (order_request.price or market_conditions.ask_price)
        
        # Available depth
        if order_request.side == OrderSide.BUY:
            available_depth = market_conditions.ask_depth_usd
        else:
            available_depth = market_conditions.bid_depth_usd
        
        # Square root market impact model
        participation_rate = order_value / max(available_depth, 1000)
        base_impact = self.microstructure.impact_coefficient * np.sqrt(participation_rate)
        
        # Volatility adjustment
        volatility_adj = market_conditions.volatility_24h / 0.02  # Normalize to 2% vol
        adjusted_impact = base_impact * volatility_adj
        
        # Convert to basis points
        impact_bps = adjusted_impact * 10000
        
        return min(impact_bps, 500.0)  # Cap at 500 bps
    
    def _simulate_queue_effects(self, order_request: OrderRequest, market_conditions: MarketConditions) -> Tuple[float, float]:
        """Simulate order queue position and waiting time effects"""
        
        # For post-only orders, simulate queue position
        if hasattr(order_request, 'time_in_force') and 'post' in str(order_request.time_in_force).lower():
            
            # Estimate queue position based on price aggressiveness
            if order_request.side == OrderSide.BUY:
                price_aggressiveness = (order_request.price - market_conditions.bid_price) / market_conditions.bid_price
            else:
                price_aggressiveness = (market_conditions.ask_price - order_request.price) / order_request.price
            
            # Queue position affects fill probability and time
            queue_position = max(1, int(20 * (1 - price_aggressiveness)))
            
            # Simulate queue decay and fill time
            expected_fill_time_ms = queue_position * 1000 * np.random.exponential(1.0)
            queue_delay_ms = min(expected_fill_time_ms, 30000)  # Max 30 seconds
            
            # Queue slippage from adverse selection
            queue_slippage_bps = queue_position * 0.5  # 0.5 bps per position
            
            return queue_delay_ms, queue_slippage_bps
        
        return 0.0, 0.0
    
    def _simulate_fills(self, order_request: OrderRequest, market_conditions: MarketConditions, impact_bps: float) -> List[ExecutionFill]:
        """Simulate realistic order fills with partial execution"""
        
        fills = []
        remaining_quantity = order_request.quantity
        fill_count = 0
        
        # Determine if maker or taker
        is_maker = hasattr(order_request, 'time_in_force') and 'post' in str(order_request.time_in_force).lower()
        fee_rate = self.microstructure.maker_fee_rate if is_maker else self.microstructure.taker_fee_rate
        
        # Base fill price
        if order_request.side == OrderSide.BUY:
            base_price = market_conditions.ask_price
        else:
            base_price = market_conditions.bid_price
        
        # Apply market impact
        impact_adjustment = (impact_bps / 10000) * base_price
        if order_request.side == OrderSide.BUY:
            fill_price = base_price + impact_adjustment
        else:
            fill_price = base_price - impact_adjustment
        
        # Simulate multiple fills for large orders
        max_fills = max(1, int(order_request.quantity / 0.1))  # Split large orders
        fill_sizes = self._generate_fill_sizes(remaining_quantity, max_fills)
        
        for fill_size in fill_sizes:
            if remaining_quantity <= 0:
                break
                
            # Add microstructure noise to fill price
            noise_bps = np.random.normal(0, self.microstructure.microstructure_noise * 10000)
            noisy_price = fill_price * (1 + noise_bps / 10000)
            
            # Round to tick size
            tick_size = self.microstructure.tick_size
            rounded_price = round(noisy_price / tick_size) * tick_size
            
            actual_fill_size = min(fill_size, remaining_quantity)
            fee_amount = actual_fill_size * rounded_price * fee_rate
            
            fill = ExecutionFill(
                fill_id=f"fill_{fill_count}_{int(time.time()*1000)}",
                quantity=actual_fill_size,
                price=rounded_price,
                timestamp=datetime.now(),
                fee_rate=fee_rate,
                fee_amount=fee_amount,
                liquidity_flag="maker" if is_maker else "taker"
            )
            
            fills.append(fill)
            remaining_quantity -= actual_fill_size
            fill_count += 1
            
            # Simulate partial fill probability
            if random.random() > self.microstructure.fill_probability:
                break
        
        return fills
    
    def _generate_fill_sizes(self, total_quantity: float, max_fills: int) -> List[float]:
        """Generate realistic fill size distribution"""
        
        if max_fills == 1:
            return [total_quantity]
        
        # Use exponential distribution for fill sizes
        fills = []
        remaining = total_quantity
        
        for i in range(max_fills - 1):
            if remaining <= 0:
                break
                
            # Exponentially decreasing fill sizes
            avg_fill_size = remaining / (max_fills - i)
            fill_size = min(np.random.exponential(avg_fill_size), remaining * 0.5)
            fills.append(max(fill_size, self.microstructure.min_size))
            remaining -= fill_size
        
        if remaining > 0:
            fills.append(remaining)
        
        return fills
    
    def _calculate_slippage_breakdown(
        self, 
        expected_price: float, 
        actual_price: float, 
        impact_bps: float,
        queue_slippage_bps: float,
        market_conditions: MarketConditions
    ) -> Dict[str, float]:
        """Calculate detailed slippage component breakdown"""
        
        total_slippage_bps = abs(actual_price - expected_price) / expected_price * 10000
        
        # Component attribution
        spread_component = market_conditions.spread_bps / 2  # Half spread baseline
        impact_component = impact_bps
        queue_component = queue_slippage_bps
        
        # Timing component (from latency effects)
        timing_component = max(0, total_slippage_bps - spread_component - impact_component - queue_component)
        
        # Volatility component
        volatility_component = market_conditions.volatility_24h * 100  # Convert to bps equivalent
        
        return {
            SlippageComponent.SPREAD.value: spread_component,
            SlippageComponent.MARKET_IMPACT.value: impact_component,
            SlippageComponent.QUEUE_POSITION.value: queue_component,
            SlippageComponent.TIMING.value: timing_component,
            SlippageComponent.VOLATILITY.value: min(volatility_component, 5.0),  # Cap volatility component
            SlippageComponent.LATENCY.value: max(0, total_slippage_bps * 0.1)  # 10% attributed to latency
        }
    
    def _record_execution(self, execution: SimulatedExecution):
        """Record execution for tracking error calculation"""
        
        with self._lock:
            self.execution_history.append(execution)
            self.total_volume_simulated += execution.filled_quantity * execution.average_fill_price
            
            # Keep last 1000 executions
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]
            
            # Update cumulative tracking error
            self.cumulative_tracking_error_bps += execution.slippage_bps
    
    def _update_metrics(self, execution: SimulatedExecution):
        """Update Prometheus metrics"""
        
        # Record execution metrics
        self.metrics.slippage_bps.labels(
            exchange="simulated",
            symbol=execution.symbol,
            side=execution.side
        ).observe(execution.slippage_bps)
        
        self.metrics.latency_ms.labels(
            operation="execution_simulation",
            exchange="simulated",
            endpoint="simulate_execution"
        ).observe(execution.execution_time_ms)
        
        # Record component-wise slippage
        for component, value in execution.slippage_breakdown.items():
            self.metrics.estimated_slippage_bps.labels(
                symbol=f"{execution.symbol}_{component}"
            ).observe(value)
    
    def calculate_daily_tracking_error(self) -> float:
        """Calculate daily tracking error in basis points"""
        
        cutoff_time = datetime.now() - timedelta(hours=self.tracking_window_hours)
        recent_executions = [
            ex for ex in self.execution_history 
            if ex.timestamp >= cutoff_time
        ]
        
        if not recent_executions:
            return 0.0
        
        # Volume-weighted tracking error
        total_volume = sum(ex.filled_quantity * ex.average_fill_price for ex in recent_executions)
        if total_volume == 0:
            return 0.0
        
        weighted_tracking_error = sum(
            ex.slippage_bps * (ex.filled_quantity * ex.average_fill_price)
            for ex in recent_executions
        ) / total_volume
        
        return weighted_tracking_error
    
    def check_tracking_error_threshold(self) -> Tuple[bool, float, str]:
        """Check if tracking error exceeds thresholds"""
        
        daily_tracking_error = self.calculate_daily_tracking_error()
        
        # Record daily tracking error
        self.daily_tracking_errors.append((datetime.now(), daily_tracking_error))
        
        # Keep last 30 days
        if len(self.daily_tracking_errors) > 30:
            self.daily_tracking_errors = self.daily_tracking_errors[-30:]
        
        # Check thresholds
        if daily_tracking_error > self.auto_disable_threshold_bps:
            self.simulation_enabled = False
            reason = f"Auto-disabled: tracking error {daily_tracking_error:.1f} bps > threshold {self.auto_disable_threshold_bps} bps"
            
            logger.critical("Execution simulator auto-disabled", extra={
                'daily_tracking_error_bps': daily_tracking_error,
                'threshold_bps': self.auto_disable_threshold_bps,
                'total_executions': self.total_executions
            })
            
            return False, daily_tracking_error, reason
        
        elif daily_tracking_error > self.tracking_error_threshold_bps:
            reason = f"Warning: tracking error {daily_tracking_error:.1f} bps > threshold {self.tracking_error_threshold_bps} bps"
            
            logger.warning("High tracking error detected", extra={
                'daily_tracking_error_bps': daily_tracking_error,
                'threshold_bps': self.tracking_error_threshold_bps
            })
            
            return True, daily_tracking_error, reason
        
        return True, daily_tracking_error, "Tracking error within acceptable range"
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        
        daily_tracking_error = self.calculate_daily_tracking_error()
        
        # Calculate component attribution from recent executions
        recent_executions = self.execution_history[-100:] if self.execution_history else []
        
        component_attribution = {}
        if recent_executions:
            for component in SlippageComponent:
                component_values = [
                    ex.slippage_breakdown.get(component.value, 0) 
                    for ex in recent_executions
                ]
                component_attribution[component.value] = {
                    'mean': np.mean(component_values),
                    'p95': np.percentile(component_values, 95) if component_values else 0
                }
        
        return {
            'simulation_enabled': self.simulation_enabled,
            'total_executions': self.total_executions,
            'total_volume_simulated': self.total_volume_simulated,
            'daily_tracking_error_bps': daily_tracking_error,
            'tracking_error_threshold_bps': self.tracking_error_threshold_bps,
            'auto_disable_threshold_bps': self.auto_disable_threshold_bps,
            'cumulative_tracking_error_bps': self.cumulative_tracking_error_bps,
            'component_attribution': component_attribution,
            'recent_executions_count': len(recent_executions),
            'avg_execution_time_ms': np.mean([ex.execution_time_ms for ex in recent_executions]) if recent_executions else 0,
            'avg_slippage_bps': np.mean([ex.slippage_bps for ex in recent_executions]) if recent_executions else 0
        }
    
    def enable_simulation(self, operator: str):
        """Manually re-enable simulation (requires operator intervention)"""
        
        self.simulation_enabled = True
        
        logger.warning("Execution simulator manually re-enabled", extra={
            'operator': operator,
            'timestamp': datetime.now().isoformat()
        })
    
    def export_execution_history(self, filepath: str):
        """Export execution history for analysis"""
        
        history_data = []
        for execution in self.execution_history:
            history_data.append({
                'order_id': execution.order_id,
                'symbol': execution.symbol,
                'side': execution.side,
                'requested_quantity': execution.requested_quantity,
                'filled_quantity': execution.filled_quantity,
                'average_fill_price': execution.average_fill_price,
                'slippage_bps': execution.slippage_bps,
                'slippage_breakdown': execution.slippage_breakdown,
                'execution_time_ms': execution.execution_time_ms,
                'status': execution.status.value,
                'timestamp': execution.timestamp.isoformat()
            })
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        logger.info("Execution history exported", extra={
            'filepath': filepath,
            'records_count': len(history_data)
        })


# Factory function
def create_execution_simulator(config: Dict[str, Any] = None) -> AdvancedExecutionSimulator:
    """Create configured execution simulator"""
    return AdvancedExecutionSimulator(config)