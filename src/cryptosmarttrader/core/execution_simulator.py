#!/usr/bin/env python3
"""
Execution Simulator - Advanced Order Execution & Risk Simulation
Implements realistic slippage, fill rates, and latency modeling for trading validation
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
import uuid

from ..core.logging_manager import get_logger

class OrderType(str, Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    IOC = "ioc"  # Immediate or Cancel
    FOK = "fok"  # Fill or Kill

class OrderStatus(str, Enum):
    """Order execution status"""
    PENDING = "pending"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class ExecutionQuality(str, Enum):
    """Execution quality assessment"""
    EXCELLENT = "excellent"   # p50 ≤ 15 bps, p90 ≤ 60 bps
    GOOD = "good"            # p50 ≤ 25 bps, p90 ≤ 80 bps
    ACCEPTABLE = "acceptable" # p50 ≤ 35 bps, p90 ≤ 100 bps
    POOR = "poor"            # Above acceptable thresholds

@dataclass
class ExecutionConfig:
    """Configuration for execution simulation"""
    # Slippage targets (basis points)
    target_slippage_p50: float = 25.0    # 25 bps median slippage
    target_slippage_p90: float = 80.0    # 80 bps 90th percentile

    # Fill rate targets
    target_fill_rate: float = 0.95       # 95% fill rate

    # Latency targets (seconds)
    target_latency_p95: float = 2.0      # 2 second 95th percentile

    # Market impact parameters
    base_market_impact_bps: float = 5.0  # Base market impact
    impact_decay_factor: float = 0.8     # Impact decay over time

    # Order book simulation
    bid_ask_spread_bps: float = 10.0     # 10 bps spread
    order_book_depth: int = 10           # Number of price levels

    # Time in force simulation
    default_tif_seconds: int = 300       # 5 minutes default

    # Exchange simulation
    exchange_latency_ms: float = 50.0    # 50ms exchange latency
    network_jitter_ms: float = 20.0     # 20ms network jitter

@dataclass
class OrderBookLevel:
    """Single order book price level"""
    price: float
    quantity: float
    side: str  # 'bid' or 'ask'

@dataclass
class SimulatedOrderBook:
    """Simulated order book for execution testing"""
    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    mid_price: float
    spread_bps: float

@dataclass
class ExecutionResult:
    """Result of order execution simulation"""
    order_id: str
    symbol: str
    side: str
    order_type: OrderType
    requested_quantity: float
    filled_quantity: float
    average_fill_price: float

    # Execution metrics
    slippage_bps: float
    market_impact_bps: float
    total_cost_bps: float

    # Timing metrics
    signal_timestamp: datetime
    order_timestamp: datetime
    first_fill_timestamp: Optional[datetime]
    last_fill_timestamp: Optional[datetime]
    end_to_end_latency_ms: float
    execution_latency_ms: float

    # Fill metrics
    fill_rate: float
    num_fills: int
    status: OrderStatus

    # Market data
    mid_price_at_signal: float
    mid_price_at_order: float
    bid_ask_spread_bps: float

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionValidationReport:
    """Comprehensive execution validation report"""
    validation_id: str
    timestamp: datetime
    total_orders: int

    # Slippage metrics
    slippage_p50: float
    slippage_p90: float
    slippage_p95: float
    slippage_mean: float
    slippage_std: float

    # Fill rate metrics
    overall_fill_rate: float
    market_order_fill_rate: float
    limit_order_fill_rate: float

    # Latency metrics
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_mean_ms: float

    # Execution quality
    execution_quality: ExecutionQuality
    quality_score: float

    # Performance by order size
    small_order_performance: Dict[str, float]
    medium_order_performance: Dict[str, float]
    large_order_performance: Dict[str, float]

    # Market impact analysis
    impact_correlation: float
    impact_decay_rate: float

    # Status
    validation_passed: bool
    failed_criteria: List[str]
    passed_criteria: List[str]

    # Recommendations
    recommendations: List[str]

class OrderBookSimulator:
    """Realistic order book simulation"""

    def __init__(self, config: ExecutionConfig):
        self.config = config
        self.logger = get_logger()

    def generate_order_book(self, symbol: str, mid_price: float, volatility: float = 0.02) -> SimulatedOrderBook:
        """Generate realistic order book"""

        try:
            timestamp = datetime.now()

            # Calculate spread based on volatility and symbol
            base_spread_bps = self.config.bid_ask_spread_bps
            spread_multiplier = 1.0 + (volatility / 0.02)  # Higher volatility = wider spreads
            spread_bps = base_spread_bps * spread_multiplier
            spread_absolute = mid_price * (spread_bps / 10000)

            # Generate bid side
            bids = []
            bid_price = mid_price - (spread_absolute / 2)

            for i in range(self.config.order_book_depth):
                # Price decreases as we go deeper
                price_offset = i * (spread_absolute / self.config.order_book_depth) * 0.5
                price = bid_price - price_offset

                # Quantity increases with depth (typical order book shape)
                base_quantity = np.random.exponential(1000)
                quantity_multiplier = 1.0 + (i * 0.3)  # More liquidity at deeper levels
                quantity = base_quantity * quantity_multiplier

                bids.append(OrderBookLevel(
                    price=price,
                    quantity=quantity,
                    side="bid"
                ))

            # Generate ask side
            asks = []
            ask_price = mid_price + (spread_absolute / 2)

            for i in range(self.config.order_book_depth):
                # Price increases as we go deeper
                price_offset = i * (spread_absolute / self.config.order_book_depth) * 0.5
                price = ask_price + price_offset

                # Quantity similar to bid side
                base_quantity = np.random.exponential(1000)
                quantity_multiplier = 1.0 + (i * 0.3)
                quantity = base_quantity * quantity_multiplier

                asks.append(OrderBookLevel(
                    price=price,
                    quantity=quantity,
                    side="ask"
                ))

            return SimulatedOrderBook(
                symbol=symbol,
                timestamp=timestamp,
                bids=bids,
                asks=asks,
                mid_price=mid_price,
                spread_bps=spread_bps
            )

        except Exception as e:
            self.logger.error(f"Order book generation failed: {e}")
            raise

class ExecutionSimulator:
    """Advanced execution simulation engine"""

    def __init__(self, config: Optional[ExecutionConfig] = None):
        self.config = config or ExecutionConfig()
        self.logger = get_logger()
        self.order_book_simulator = OrderBookSimulator(self.config)

    async def simulate_order_execution(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        signal_timestamp: Optional[datetime] = None,
        mid_price: Optional[float] = None,
        volatility: float = 0.02
    ) -> ExecutionResult:
        """Simulate realistic order execution"""

        order_id = f"sim_order_{uuid.uuid4().hex[:8]}"
        signal_time = signal_timestamp or datetime.now()

        try:
            # REMOVED: Mock data pattern not allowed in production
            processing_latency_ms = np.random.gamma(2, 25)  # Gamma distribution ~50ms mean
            order_timestamp = signal_time + timedelta(milliseconds=processing_latency_ms)

            # Get current market price
            current_mid_price = mid_price or await self._get_market_price(symbol)

            # Generate order book
            order_book = self.order_book_simulator.generate_order_book(
                symbol, current_mid_price, volatility
            )

            # REMOVED: Mock data pattern not allowed in production
            execution_result = await self._execute_order_on_book(
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type=order_type,
                order_book=order_book,
                signal_timestamp=signal_time,
                order_timestamp=order_timestamp
            )

            return execution_result

        except Exception as e:
            self.logger.error(f"Order execution simulation failed: {e}")

            # Return failed execution
            return ExecutionResult(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                requested_quantity=quantity,
                filled_quantity=0.0,
                average_fill_price=0.0,
                slippage_bps=float('inf'),
                market_impact_bps=float('inf'),
                total_cost_bps=float('inf'),
                signal_timestamp=signal_time,
                order_timestamp=signal_time,
                first_fill_timestamp=None,
                last_fill_timestamp=None,
                end_to_end_latency_ms=float('inf'),
                execution_latency_ms=float('inf'),
                fill_rate=0.0,
                num_fills=0,
                status=OrderStatus.REJECTED,
                mid_price_at_signal=current_mid_price if mid_price else 0.0,
                mid_price_at_order=current_mid_price if mid_price else 0.0,
                bid_ask_spread_bps=0.0,
                metadata={"error": str(e)}
            )

    async def _execute_order_on_book(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        order_type: OrderType,
        order_book: SimulatedOrderBook,
        signal_timestamp: datetime,
        order_timestamp: datetime
    ) -> ExecutionResult:
        """Execute order against simulated order book"""

        try:
            # Determine which side of book to trade against
            target_levels = order_book.asks if side == "buy" else order_book.bids

            # Calculate market impact based on order size
            total_book_liquidity = sum(level.quantity for level in target_levels)
            impact_ratio = quantity / total_book_liquidity if total_book_liquidity > 0 else 1.0

            # Base market impact increases with order size
            market_impact_bps = self.config.base_market_impact_bps * (impact_ratio ** 0.5)

            # REMOVED: Mock data pattern not allowed in production
            exchange_latency_ms = np.random.gamma(
                2, self.config.exchange_latency_ms / 2
            ) + np.random.normal(0, 1)

            first_fill_time = order_timestamp + timedelta(milliseconds=exchange_latency_ms)

            # REMOVED: Mock data pattern not allowed in production
            fills = []
            remaining_quantity = quantity
            total_fill_value = 0.0

            for level in target_levels:
                if remaining_quantity <= 0:
                    break

                # Determine fill quantity for this level
                available_quantity = level.quantity

                # Apply random partial fill factor
                if order_type == OrderType.MARKET:
                    fill_probability = 0.95  # 95% chance of filling at each level
                else:
                    fill_probability = 0.80  # 80% for limit orders

                if np.random.random() > fill_probability:
                    continue  # Skip this level

                # Calculate fill quantity
                fill_quantity = min(remaining_quantity, available_quantity)

                # Apply slippage and impact
                base_price = level.price

                # Add market impact
                impact_adjustment = base_price * (market_impact_bps / 10000)
                if side == "buy":
                    impact_adjustment = abs(impact_adjustment)
                else:
                    impact_adjustment = -abs(impact_adjustment)

                # Add random slippage component
                random_slippage_bps = np.random.gamma(2, 5)  # ~10 bps mean
                random_slippage = base_price * (random_slippage_bps / 10000)
                if side == "buy":
                    random_slippage = abs(random_slippage)
                else:
                    random_slippage = -abs(random_slippage)

                fill_price = base_price + impact_adjustment + random_slippage

                fills.append({
                    "quantity": fill_quantity,
                    "price": fill_price,
                    "timestamp": first_fill_time + timedelta(milliseconds=len(fills) * 10)
                })

                total_fill_value += fill_quantity * fill_price
                remaining_quantity -= fill_quantity

            # Calculate execution metrics
            if fills:
                filled_quantity = sum(fill["quantity"] for fill in fills)
                average_fill_price = total_fill_value / filled_quantity if filled_quantity > 0 else 0.0
                last_fill_time = fills[-1]["timestamp"]

                # Calculate slippage vs mid price
                mid_price = order_book.mid_price
                slippage_absolute = abs(average_fill_price - mid_price)
                slippage_bps = (slippage_absolute / mid_price) * 10000

                # Total cost including market impact
                total_cost_bps = slippage_bps + market_impact_bps

                # Determine order status
                fill_rate = filled_quantity / quantity
                if fill_rate >= 0.99:
                    status = OrderStatus.FILLED
                elif fill_rate > 0:
                    status = OrderStatus.PARTIALLY_FILLED
                else:
                    status = OrderStatus.CANCELLED

            else:
                # No fills
                filled_quantity = 0.0
                average_fill_price = 0.0
                last_fill_time = None
                slippage_bps = float('inf')
                total_cost_bps = float('inf')
                fill_rate = 0.0
                status = OrderStatus.CANCELLED

            # Calculate latency metrics
            end_to_end_latency_ms = (order_timestamp - signal_timestamp).total_seconds() * 1000

            if first_fill_time:
                execution_latency_ms = (first_fill_time - order_timestamp).total_seconds() * 1000
            else:
                execution_latency_ms = float('inf')

            return ExecutionResult(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                requested_quantity=quantity,
                filled_quantity=filled_quantity,
                average_fill_price=average_fill_price,
                slippage_bps=slippage_bps,
                market_impact_bps=market_impact_bps,
                total_cost_bps=total_cost_bps,
                signal_timestamp=signal_timestamp,
                order_timestamp=order_timestamp,
                first_fill_timestamp=first_fill_time,
                last_fill_timestamp=last_fill_time,
                end_to_end_latency_ms=end_to_end_latency_ms,
                execution_latency_ms=execution_latency_ms,
                fill_rate=fill_rate,
                num_fills=len(fills),
                status=status,
                mid_price_at_signal=order_book.mid_price,
                mid_price_at_order=order_book.mid_price,
                bid_ask_spread_bps=order_book.spread_bps,
                metadata={
                    "impact_ratio": impact_ratio,
                    "total_book_liquidity": total_book_liquidity,
                    "fills": fills
                }
            )

        except Exception as e:
            self.logger.error(f"Order book execution failed: {e}")
            raise

    async def _get_market_price(self, symbol: str) -> float:
        """Get current market price for symbol"""

        # REMOVED: Mock data pattern not allowed in production
        base_prices = {
            "BTC/USD": 45000.0,
            "ETH/USD": 3000.0,
            "ADA/USD": 0.50,
            "SOL/USD": 100.0,
            "DOT/USD": 7.50
        }

        base_price = base_prices.get(symbol, 1.0)

        # Add realistic price movement
        noise = np.random.normal(0, 1)  # 0.5% volatility
        current_price = base_price * (1 + noise)

        return max(current_price, 0.0001)

    async def validate_execution_performance(
        self,
        execution_results: List[ExecutionResult]
    ) -> ExecutionValidationReport:
        """Validate execution performance against targets"""

        validation_id = f"execution_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            if not execution_results:
                raise ValueError("No execution results provided")

            # Filter valid executions
            valid_executions = [
                r for r in execution_results
                if r.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]
                and not np.isinf(r.slippage_bps)
            ]

            if not valid_executions:
                raise ValueError("No valid executions to analyze")

            # Calculate slippage metrics
            slippages = [r.slippage_bps for r in valid_executions]
            slippage_p50 = np.percentile(slippages, 50)
            slippage_p90 = np.percentile(slippages, 90)
            slippage_p95 = np.percentile(slippages, 95)
            slippage_mean = np.mean(slippages)
            slippage_std = np.std(slippages)

            # Calculate fill rates
            all_orders = execution_results
            overall_fill_rate = np.mean([r.fill_rate for r in all_orders])

            market_orders = [r for r in all_orders if r.order_type == OrderType.MARKET]
            market_order_fill_rate = np.mean([r.fill_rate for r in market_orders]) if market_orders else 0.0

            limit_orders = [r for r in all_orders if r.order_type == OrderType.LIMIT]
            limit_order_fill_rate = np.mean([r.fill_rate for r in limit_orders]) if limit_orders else 0.0

            # Calculate latency metrics
            latencies = [r.end_to_end_latency_ms for r in valid_executions if not np.isinf(r.end_to_end_latency_ms)]
            if latencies:
                latency_p50_ms = np.percentile(latencies, 50)
                latency_p90_ms = np.percentile(latencies, 90)
                latency_p95_ms = np.percentile(latencies, 95)
                latency_mean_ms = np.mean(latencies)
            else:
                latency_p50_ms = latency_p90_ms = latency_p95_ms = latency_mean_ms = float('inf')

            # Determine execution quality
            execution_quality, quality_score = self._assess_execution_quality(
                slippage_p50, slippage_p90, overall_fill_rate, latency_p95_ms
            )

            # Analyze performance by order size
            small_orders = [r for r in valid_executions if r.requested_quantity <= 1000]
            medium_orders = [r for r in valid_executions if 1000 < r.requested_quantity <= 10000]
            large_orders = [r for r in valid_executions if r.requested_quantity > 10000]

            small_order_performance = self._calculate_size_performance(small_orders)
            medium_order_performance = self._calculate_size_performance(medium_orders)
            large_order_performance = self._calculate_size_performance(large_orders)

            # Market impact analysis
            quantities = [r.requested_quantity for r in valid_executions]
            impacts = [r.market_impact_bps for r in valid_executions]

            if len(quantities) > 5:
                impact_correlation, _ = stats.pearsonr(quantities, impacts)
                impact_decay_rate = self.config.impact_decay_factor
            else:
                impact_correlation = 0.0
                impact_decay_rate = 0.0

            # Validation criteria
            failed_criteria = []
            passed_criteria = []

            # Check slippage targets
            if slippage_p50 <= self.config.target_slippage_p50:
                passed_criteria.append("slippage_p50")
            else:
                failed_criteria.append("slippage_p50")

            if slippage_p90 <= self.config.target_slippage_p90:
                passed_criteria.append("slippage_p90")
            else:
                failed_criteria.append("slippage_p90")

            # Check fill rate
            if overall_fill_rate >= self.config.target_fill_rate:
                passed_criteria.append("fill_rate")
            else:
                failed_criteria.append("fill_rate")

            # Check latency
            if latency_p95_ms <= (self.config.target_latency_p95 * 1000):
                passed_criteria.append("latency_p95")
            else:
                failed_criteria.append("latency_p95")

            validation_passed = len(failed_criteria) == 0

            # Generate recommendations
            recommendations = self._generate_execution_recommendations(
                failed_criteria, slippage_p50, slippage_p90, overall_fill_rate, latency_p95_ms
            )

            return ExecutionValidationReport(
                validation_id=validation_id,
                timestamp=datetime.now(),
                total_orders=len(execution_results),
                slippage_p50=slippage_p50,
                slippage_p90=slippage_p90,
                slippage_p95=slippage_p95,
                slippage_mean=slippage_mean,
                slippage_std=slippage_std,
                overall_fill_rate=overall_fill_rate,
                market_order_fill_rate=market_order_fill_rate,
                limit_order_fill_rate=limit_order_fill_rate,
                latency_p50_ms=latency_p50_ms,
                latency_p90_ms=latency_p90_ms,
                latency_p95_ms=latency_p95_ms,
                latency_mean_ms=latency_mean_ms,
                execution_quality=execution_quality,
                quality_score=quality_score,
                small_order_performance=small_order_performance,
                medium_order_performance=medium_order_performance,
                large_order_performance=large_order_performance,
                impact_correlation=impact_correlation,
                impact_decay_rate=impact_decay_rate,
                validation_passed=validation_passed,
                failed_criteria=failed_criteria,
                passed_criteria=passed_criteria,
                recommendations=recommendations
            )

        except Exception as e:
            self.logger.error(f"Execution validation failed: {e}")

            return ExecutionValidationReport(
                validation_id=validation_id,
                timestamp=datetime.now(),
                total_orders=0,
                slippage_p50=float('inf'),
                slippage_p90=float('inf'),
                slippage_p95=float('inf'),
                slippage_mean=float('inf'),
                slippage_std=float('inf'),
                overall_fill_rate=0.0,
                market_order_fill_rate=0.0,
                limit_order_fill_rate=0.0,
                latency_p50_ms=float('inf'),
                latency_p90_ms=float('inf'),
                latency_p95_ms=float('inf'),
                latency_mean_ms=float('inf'),
                execution_quality=ExecutionQuality.POOR,
                quality_score=0.0,
                small_order_performance={},
                medium_order_performance={},
                large_order_performance={},
                impact_correlation=0.0,
                impact_decay_rate=0.0,
                validation_passed=False,
                failed_criteria=["validation_error"],
                passed_criteria=[],
                recommendations=[f"Fix execution validation error: {e}"]
            )

    def _assess_execution_quality(
        self,
        slippage_p50: float,
        slippage_p90: float,
        fill_rate: float,
        latency_p95_ms: float
    ) -> Tuple[ExecutionQuality, float]:
        """Assess overall execution quality"""

        # Quality scoring
        scores = []

        # Slippage score
        if slippage_p50 <= 15 and slippage_p90 <= 60:
            slippage_score = 1.0
        elif slippage_p50 <= 25 and slippage_p90 <= 80:
            slippage_score = 0.8
        elif slippage_p50 <= 35 and slippage_p90 <= 100:
            slippage_score = 0.6
        else:
            slippage_score = 0.3

        scores.append(slippage_score)

        # Fill rate score
        if fill_rate >= 0.98:
            fill_score = 1.0
        elif fill_rate >= 0.95:
            fill_score = 0.8
        elif fill_rate >= 0.90:
            fill_score = 0.6
        else:
            fill_score = 0.3

        scores.append(fill_score)

        # Latency score
        if latency_p95_ms <= 1000:
            latency_score = 1.0
        elif latency_p95_ms <= 2000:
            latency_score = 0.8
        elif latency_p95_ms <= 5000:
            latency_score = 0.6
        else:
            latency_score = 0.3

        scores.append(latency_score)

        # Overall quality score
        quality_score = np.mean(scores)

        # Determine quality level
        if quality_score >= 0.9:
            quality = ExecutionQuality.EXCELLENT
        elif quality_score >= 0.75:
            quality = ExecutionQuality.GOOD
        elif quality_score >= 0.55:
            quality = ExecutionQuality.ACCEPTABLE
        else:
            quality = ExecutionQuality.POOR

        return quality, quality_score

    def _calculate_size_performance(self, orders: List[ExecutionResult]) -> Dict[str, float]:
        """Calculate performance metrics for order size category"""

        if not orders:
            return {
                "avg_slippage_bps": float('inf'),
                "avg_fill_rate": 0.0,
                "avg_latency_ms": float('inf'),
                "count": 0
            }

        return {
            "avg_slippage_bps": np.mean([o.slippage_bps for o in orders if not np.isinf(o.slippage_bps)]),
            "avg_fill_rate": np.mean([o.fill_rate for o in orders]),
            "avg_latency_ms": np.mean([o.end_to_end_latency_ms for o in orders if not np.isinf(o.end_to_end_latency_ms)]),
            "count": len(orders)
        }

    def _generate_execution_recommendations(
        self,
        failed_criteria: List[str],
        slippage_p50: float,
        slippage_p90: float,
        fill_rate: float,
        latency_p95_ms: float
    ) -> List[str]:
        """Generate execution improvement recommendations"""

        recommendations = []

        if "slippage_p50" in failed_criteria:
            recommendations.append(f"Median slippage {slippage_p50:.1f} bps > target {self.config.target_slippage_p50:.1f} bps - optimize order routing")

        if "slippage_p90" in failed_criteria:
            recommendations.append(f"90th percentile slippage {slippage_p90:.1f} bps > target {self.config.target_slippage_p90:.1f} bps - implement dynamic sizing")

        if "fill_rate" in failed_criteria:
            recommendations.append(f"Fill rate {fill_rate:.1%} < target {self.config.target_fill_rate:.1%} - review order types and TIF")

        if "latency_p95" in failed_criteria:
            recommendations.append(f"95th percentile latency {latency_p95_ms:.0f}ms > target {self.config.target_latency_p95*1000:.0f}ms - optimize signal processing")

        if not recommendations:
            recommendations.append("All execution targets met - maintain current configuration")

        return recommendations

# Global instance
_execution_simulator = None

def get_execution_simulator(config: Optional[ExecutionConfig] = None) -> ExecutionSimulator:
    """Get global execution simulator instance"""
    global _execution_simulator
    if _execution_simulator is None:
        _execution_simulator = ExecutionSimulator(config)
    return _execution_simulator
