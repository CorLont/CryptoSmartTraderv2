#!/usr/bin/env python3
"""
Advanced Position Manager with Automated Kill Switch and Health-Based Controls
Implements real-time position monitoring, automated flattening, and health score integration
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
import warnings

warnings.filterwarnings("ignore")

from core.logging_manager import get_logger
from core.portfolio_risk_manager import (
    get_portfolio_risk_manager,
    Position,
    PositionAction,
    RiskStatus,
)
from core.data_quality_manager import get_data_quality_manager
from core.execution_simulator import get_execution_simulator, Order, OrderSide, OrderType


class HealthStatus(str, Enum):
    """System health status levels"""

    HEALTHY = "healthy"  # All systems operational
    DEGRADED = "degraded"  # Some issues but operational
    IMPAIRED = "impaired"  # Significant issues
    CRITICAL = "critical"  # Critical issues - emergency actions needed
    OFFLINE = "offline"  # System offline - kill all positions


class PositionCommand(str, Enum):
    """Position management commands"""

    OPEN = "open"
    CLOSE = "close"
    REDUCE = "reduce"
    INCREASE = "increase"
    FLATTEN_ALL = "flatten_all"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class HealthScore:
    """System health score components"""

    timestamp: datetime
    data_quality_score: float  # 0-1
    connectivity_score: float  # 0-1
    latency_score: float  # 0-1
    execution_score: float  # 0-1
    liquidity_score: float  # 0-1
    overall_score: float  # 0-1
    go_nogo_threshold: float = 0.7  # Threshold for operations

    @property
    def is_go(self) -> bool:
        """Check if system is GO for operations"""
        return self.overall_score >= self.go_nogo_threshold

    @property
    def health_status(self) -> HealthStatus:
        """Get health status based on score"""
        if self.overall_score >= 0.9:
            return HealthStatus.HEALTHY
        elif self.overall_score >= 0.7:
            return HealthStatus.DEGRADED
        elif self.overall_score >= 0.5:
            return HealthStatus.IMPAIRED
        elif self.overall_score >= 0.2:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.OFFLINE


@dataclass
class PositionOrder:
    """Position management order"""

    order_id: str
    symbol: str
    command: PositionCommand
    target_size: float
    current_size: float
    urgency: str  # 'low', 'medium', 'high', 'emergency'
    reason: str
    timestamp: datetime
    status: str = "pending"
    execution_result: Optional[Dict[str, Any]] = None


@dataclass
class EmergencyAction:
    """Emergency position action record"""

    action_id: str
    trigger_reason: str
    health_score: float
    positions_before: Dict[str, float]
    positions_after: Dict[str, float]
    execution_time_seconds: float
    timestamp: datetime
    success: bool


class HealthMonitor:
    """Monitors system health for position management decisions"""

    def __init__(self):
        self.logger = get_logger()
        self.health_history = []
        self.last_health_check = None

    def calculate_health_score(
        self,
        market_data: Dict[str, Any],
        execution_metrics: Dict[str, Any],
        data_quality_summary: Dict[str, Any],
    ) -> HealthScore:
        """Calculate comprehensive system health score"""

        timestamp = datetime.now()

        # 1. Data Quality Score (30% weight)
        data_quality_score = data_quality_summary.get("overall_completeness", 0)

        # 2. Connectivity Score (20% weight)
        # Based on API response times and success rates
        api_success_rate = execution_metrics.get("api_success_rate", 0.95)
        connectivity_score = min(api_success_rate, 1.0)

        # 3. Latency Score (15% weight)
        # Based on execution latency
        avg_latency = execution_metrics.get("avg_latency_ms", 100)
        latency_score = max(0, 1 - (avg_latency - 50) / 1000)  # Penalty for >50ms

        # 4. Execution Score (20% weight)
        # Based on fill rates and slippage
        fill_rate = execution_metrics.get("fill_rate", 0.95)
        avg_slippage = execution_metrics.get("avg_slippage_bps", 10)
        slippage_penalty = min(avg_slippage / 100, 0.5)  # Penalty for high slippage
        execution_score = fill_rate * (1 - slippage_penalty)

        # 5. Liquidity Score (15% weight)
        # Based on market depth and volume
        avg_liquidity = market_data.get("avg_liquidity_score", 0.5)
        liquidity_score = avg_liquidity

        # Calculate weighted overall score
        overall_score = (
            data_quality_score * 0.30
            + connectivity_score * 0.20
            + latency_score * 0.15
            + execution_score * 0.20
            + liquidity_score * 0.15
        )

        health_score = HealthScore(
            timestamp=timestamp,
            data_quality_score=data_quality_score,
            connectivity_score=connectivity_score,
            latency_score=latency_score,
            execution_score=execution_score,
            liquidity_score=liquidity_score,
            overall_score=overall_score,
        )

        # Store health history
        self.health_history.append(health_score)
        if len(self.health_history) > 1000:  # Keep last 1000 records
            self.health_history = self.health_history[-1000:]

        self.last_health_check = timestamp

        self.logger.info(
            f"Health score calculated: {overall_score:.2f}",
            extra={
                "overall_score": overall_score,
                "health_status": health_score.health_status.value,
                "is_go": health_score.is_go,
                "data_quality": data_quality_score,
                "connectivity": connectivity_score,
                "latency": latency_score,
                "execution": execution_score,
                "liquidity": liquidity_score,
            },
        )

        return health_score

    def check_health_degradation(self, lookback_minutes: int = 30) -> bool:
        """Check if health has been degrading over time"""

        if len(self.health_history) < 2:
            return False

        cutoff_time = datetime.now() - timedelta(minutes=lookback_minutes)
        recent_scores = [h.overall_score for h in self.health_history if h.timestamp >= cutoff_time]

        if len(recent_scores) < 2:
            return False

        # Check for declining trend
        trend = np.polyfit(range(len(recent_scores)), recent_scores, 1)[0]

        # Significant negative trend
        return trend < -0.01  # Declining by more than 1% per measurement


class PositionManager:
    """Advanced position manager with health-based controls"""

    def __init__(self):
        self.logger = get_logger()
        self.risk_manager = get_portfolio_risk_manager()
        self.data_quality_manager = get_data_quality_manager()
        self.execution_simulator = get_execution_simulator()
        self.health_monitor = HealthMonitor()

        # Position tracking
        self.current_positions: Dict[str, Position] = {}
        self.pending_orders: List[PositionOrder] = []
        self.emergency_actions: List[EmergencyAction] = []

        # Control state
        self.auto_flatten_enabled = True
        self.emergency_mode = False
        self.last_health_check = None

        # Thresholds
        self.health_thresholds = {
            "auto_flatten": 0.5,  # Auto-flatten below 50% health
            "position_freeze": 0.6,  # Freeze new positions below 60%
            "emergency_stop": 0.3,  # Emergency stop below 30%
        }

        self.logger.info("Position Manager initialized with health-based controls")

    async def monitor_positions(self):
        """Continuous position monitoring with health checks"""

        while True:
            try:
                # Get current market data
                market_data = await self._get_market_data()

                # Get execution metrics
                execution_metrics = self.execution_simulator.get_execution_summary()

                # Get data quality summary
                data_quality_summary = self.data_quality_manager.get_quality_summary()

                # Calculate health score
                health_score = self.health_monitor.calculate_health_score(
                    market_data, execution_metrics, data_quality_summary
                )

                # Check for emergency conditions
                await self._check_emergency_conditions(health_score)

                # Update position risk metrics
                await self._update_position_metrics(market_data)

                # Process any pending orders
                await self._process_pending_orders()

                # Check kill switch conditions
                if self.risk_manager.check_kill_switch_conditions(
                    self.current_positions, market_data
                ):
                    await self._execute_emergency_flatten("Risk manager kill switch activated")

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                self.logger.error(f"Position monitoring error: {e}")
                await asyncio.sleep(30)  # Wait longer on error

    async def _check_emergency_conditions(self, health_score: HealthScore):
        """Check for emergency conditions requiring immediate action"""

        # Emergency stop condition
        if health_score.overall_score < self.health_thresholds["emergency_stop"]:
            await self._execute_emergency_flatten(
                f"Emergency stop: health score {health_score.overall_score:.2f} below threshold {self.health_thresholds['emergency_stop']}"
            )
            return

        # Auto-flatten condition
        if (
            health_score.overall_score < self.health_thresholds["auto_flatten"]
            and self.auto_flatten_enabled
            and not self.emergency_mode
        ):
            await self._execute_auto_flatten(
                f"Auto-flatten: health score {health_score.overall_score:.2f} below threshold {self.health_thresholds['auto_flatten']}"
            )
            return

        # Check for health degradation
        if self.health_monitor.check_health_degradation():
            self.logger.warning(
                "Health degradation detected - restricting new positions",
                extra={"health_score": health_score.overall_score},
            )

    async def _execute_emergency_flatten(self, reason: str):
        """Execute emergency position flattening"""

        if self.emergency_mode:
            return  # Already in emergency mode

        start_time = datetime.now()
        self.emergency_mode = True

        self.logger.critical(
            f"EMERGENCY FLATTEN ACTIVATED: {reason}",
            extra={
                "reason": reason,
                "positions_count": len(self.current_positions),
                "total_exposure": sum(
                    abs(pos.market_value) for pos in self.current_positions.values()
                ),
            },
        )

        positions_before = {
            symbol: pos.market_value for symbol, pos in self.current_positions.items()
        }

        # Flatten all positions immediately
        flatten_orders = []
        for symbol, position in self.current_positions.items():
            if abs(position.size) > 1e-8:  # Position exists
                # Create emergency close order
                order = PositionOrder(
                    order_id=f"emergency_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    symbol=symbol,
                    command=PositionCommand.CLOSE,
                    target_size=0.0,
                    current_size=position.size,
                    urgency="emergency",
                    reason=reason,
                    timestamp=datetime.now(),
                )
                flatten_orders.append(order)

        # Execute all flatten orders
        success = True
        for order in flatten_orders:
            try:
                await self._execute_position_order(order)
                if order.status != "completed":
                    success = False
            except Exception as e:
                self.logger.error(f"Emergency flatten failed for {order.symbol}: {e}")
                success = False

        positions_after = {
            symbol: pos.market_value for symbol, pos in self.current_positions.items()
        }
        execution_time = (datetime.now() - start_time).total_seconds()

        # Record emergency action
        emergency_action = EmergencyAction(
            action_id=f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            trigger_reason=reason,
            health_score=self.health_monitor.health_history[-1].overall_score
            if self.health_monitor.health_history
            else 0,
            positions_before=positions_before,
            positions_after=positions_after,
            execution_time_seconds=execution_time,
            timestamp=start_time,
            success=success,
        )

        self.emergency_actions.append(emergency_action)

        self.logger.critical(
            f"Emergency flatten completed in {execution_time:.2f}s",
            extra={
                "success": success,
                "execution_time_seconds": execution_time,
                "positions_closed": len(flatten_orders),
            },
        )

    async def _execute_auto_flatten(self, reason: str):
        """Execute automatic position flattening for risk management"""

        self.logger.warning(
            f"AUTO-FLATTEN TRIGGERED: {reason}",
            extra={"reason": reason, "positions_count": len(self.current_positions)},
        )

        # Get position recommendations from risk manager
        recommendations = self.risk_manager.get_position_recommendations(self.current_positions)

        # Execute recommended actions
        for symbol, recommendation in recommendations.items():
            if recommendation["action"] in ["flatten", "reduce"]:
                target_value = recommendation["suggested_target"]
                current_position = self.current_positions.get(symbol)

                if current_position:
                    # Calculate target size
                    target_size = (
                        target_value / current_position.current_price
                        if current_position.current_price > 0
                        else 0
                    )

                    order = PositionOrder(
                        order_id=f"auto_flatten_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        symbol=symbol,
                        command=PositionCommand.REDUCE
                        if recommendation["action"] == "reduce"
                        else PositionCommand.CLOSE,
                        target_size=target_size,
                        current_size=current_position.size,
                        urgency=recommendation["urgency"],
                        reason=reason,
                        timestamp=datetime.now(),
                    )

                    self.pending_orders.append(order)

    async def _update_position_metrics(self, market_data: Dict[str, Any]):
        """Update position metrics with latest market data"""

        for symbol, position in self.current_positions.items():
            # Update current price
            price_key = f"{symbol}_price"
            if price_key in market_data:
                position.current_price = market_data[price_key]
                position.market_value = position.size * position.current_price
                position.unrealized_pnl = (
                    position.current_price - position.entry_price
                ) * position.size

            # Update data quality score
            quality_summary = self.data_quality_manager.get_quality_summary()
            position.data_quality_score = quality_summary.get("overall_completeness", 0)

            # Update liquidity score
            liquidity_key = f"{symbol}_liquidity_score"
            if liquidity_key in market_data:
                position.liquidity_score = market_data[liquidity_key]

            position.last_updated = datetime.now()

    async def _process_pending_orders(self):
        """Process pending position orders"""

        completed_orders = []

        for order in self.pending_orders:
            try:
                await self._execute_position_order(order)
                if order.status in ["completed", "failed"]:
                    completed_orders.append(order)
            except Exception as e:
                self.logger.error(f"Failed to process order {order.order_id}: {e}")
                order.status = "failed"
                completed_orders.append(order)

        # Remove completed orders
        for order in completed_orders:
            self.pending_orders.remove(order)

    async def _execute_position_order(self, order: PositionOrder):
        """Execute individual position order"""

        try:
            symbol = order.symbol
            current_position = self.current_positions.get(symbol)

            if not current_position and order.command != PositionCommand.OPEN:
                order.status = "failed"
                order.execution_result = {"error": "Position does not exist"}
                return

            # Calculate order size
            if order.command == PositionCommand.CLOSE:
                order_size = -current_position.size  # Close entire position
            elif order.command == PositionCommand.REDUCE:
                size_diff = current_position.size - order.target_size
                order_size = -size_diff if size_diff > 0 else 0
            elif order.command == PositionCommand.INCREASE:
                size_diff = order.target_size - current_position.size
                order_size = size_diff if size_diff > 0 else 0
            elif order.command == PositionCommand.OPEN:
                order_size = order.target_size
            else:
                order.status = "failed"
                order.execution_result = {"error": "Unknown command"}
                return

            if abs(order_size) < 1e-8:  # No action needed
                order.status = "completed"
                order.execution_result = {"message": "No action required"}
                return

            # Create execution order
            side = OrderSide.BUY if order_size > 0 else OrderSide.SELL
            execution_order = Order(
                order_id=order.order_id,
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET,
                size=abs(order_size),
                timestamp=order.timestamp,
            )

            # Execute via execution simulator
            current_price = (
                current_position.current_price if current_position else 100
            )  # Default price
            from core.execution_simulator import MarketConditions

            market_conditions = MarketConditions(
                volatility=0.02, volume_ratio=1.0, spread_ratio=1.0, depth_ratio=1.0
            )

            executed_order = self.execution_simulator.execute_order(
                execution_order, current_price, market_conditions
            )

            # Update position based on execution
            if executed_order.status.value in ["filled", "partially_filled"]:
                self._update_position_from_execution(symbol, executed_order)
                order.status = "completed"
                order.execution_result = {
                    "fill_price": executed_order.average_fill_price,
                    "fill_size": sum(fill.size for fill in executed_order.fills),
                    "fees": executed_order.total_fees,
                    "slippage_bps": executed_order.slippage_bps,
                }
            else:
                order.status = "failed"
                order.execution_result = {
                    "error": f"Execution failed: {executed_order.status.value}"
                }

            self.logger.info(
                f"Position order executed: {order.order_id}",
                extra={
                    "symbol": symbol,
                    "command": order.command.value,
                    "status": order.status,
                    "order_size": order_size,
                },
            )

        except Exception as e:
            order.status = "failed"
            order.execution_result = {"error": str(e)}
            self.logger.error(f"Position order execution failed: {e}")

    def _update_position_from_execution(self, symbol: str, executed_order: Order):
        """Update position based on execution results"""

        filled_size = sum(fill.size for fill in executed_order.fills)
        if executed_order.side == OrderSide.SELL:
            filled_size = -filled_size

        avg_price = executed_order.average_fill_price

        if symbol in self.current_positions:
            # Update existing position
            position = self.current_positions[symbol]

            # Calculate new position size
            new_size = position.size + filled_size

            if abs(new_size) < 1e-8:  # Position closed
                del self.current_positions[symbol]
            else:
                # Update position
                position.size = new_size
                position.current_price = avg_price
                position.market_value = new_size * avg_price
                position.last_updated = datetime.now()
        else:
            # Create new position
            self.current_positions[symbol] = Position(
                symbol=symbol,
                size=filled_size,
                market_value=filled_size * avg_price,
                unrealized_pnl=0.0,
                entry_price=avg_price,
                current_price=avg_price,
                weight=0.0,  # Will be calculated later
                daily_volume=0.0,
                adv_utilization=0.0,
                correlation_max=0.0,
                data_quality_score=1.0,
                liquidity_score=1.0,
                last_updated=datetime.now(),
            )

    async def _get_market_data(self) -> Dict[str, Any]:
        """Get current market data (placeholder implementation)"""

        # In practice, this would fetch real market data
        return {
            "avg_liquidity_score": 0.7,
            "api_success_rate": 0.95,
            "avg_latency_ms": 120,
            "fill_rate": 0.98,
            "avg_slippage_bps": 15,
        }

    def get_position_summary(self) -> Dict[str, Any]:
        """Get comprehensive position summary"""

        current_health = (
            self.health_monitor.health_history[-1] if self.health_monitor.health_history else None
        )

        return {
            "timestamp": datetime.now().isoformat(),
            "total_positions": len(self.current_positions),
            "pending_orders": len(self.pending_orders),
            "emergency_mode": self.emergency_mode,
            "auto_flatten_enabled": self.auto_flatten_enabled,
            "current_health_score": current_health.overall_score if current_health else 0,
            "health_status": current_health.health_status.value if current_health else "unknown",
            "is_go_for_operations": current_health.is_go if current_health else False,
            "positions": {
                symbol: {
                    "size": pos.size,
                    "market_value": pos.market_value,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "data_quality_score": pos.data_quality_score,
                    "liquidity_score": pos.liquidity_score,
                }
                for symbol, pos in self.current_positions.items()
            },
            "emergency_actions_count": len(self.emergency_actions),
            "last_emergency": self.emergency_actions[-1].timestamp.isoformat()
            if self.emergency_actions
            else None,
        }


# Global instance
_position_manager = None


def get_position_manager() -> PositionManager:
    """Get global position manager instance"""
    global _position_manager
    if _position_manager is None:
        _position_manager = PositionManager()
    return _position_manager
