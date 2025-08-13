"""
Partial Fill Handler

Manages partial order fills and implements intelligent order management
strategies for incomplete executions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class FillStatus(Enum):
    """Fill status for orders"""
    UNFILLED = "unfilled"
    PARTIAL = "partial"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class FillStrategy(Enum):
    """Strategies for handling partial fills"""
    ACCUMULATE = "accumulate"       # Keep accumulating fills
    CANCEL_AND_RETRY = "cancel_retry"  # Cancel and place new order
    SIZE_REDUCTION = "size_reduction"  # Reduce remaining size
    TIME_PRIORITY = "time_priority"    # Focus on time-sensitive fills


@dataclass
class OrderFill:
    """Individual order fill"""
    fill_id: str
    timestamp: datetime
    filled_quantity: float
    fill_price: float
    fee_paid: float
    fee_type: str  # maker/taker

    # Market conditions at fill
    market_price: float
    spread_bp: float
    slippage_bp: float = 0.0

    def __post_init__(self):
        if self.slippage_bp == 0.0:
            # Calculate slippage
            if self.market_price > 0:
                self.slippage_bp = abs(self.fill_price - self.market_price) / self.market_price * 10000


@dataclass
class PartialOrder:
    """Order with partial fill tracking"""
    order_id: str
    pair: str
    side: str  # buy/sell
    original_quantity: float
    target_price: float
    order_type: str  # limit/market/stop

    # Fill tracking
    fills: List[OrderFill] = field(default_factory=list)
    remaining_quantity: float = 0.0
    status: FillStatus = FillStatus.UNFILLED

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    last_fill_at: Optional[datetime] = None

    # Strategy parameters
    max_fill_time_minutes: int = 30
    min_fill_size: float = 0.0
    allow_partial_completion: bool = True

    def __post_init__(self):
        if self.remaining_quantity == 0.0:
            self.remaining_quantity = self.original_quantity

    @property
    def filled_quantity(self) -> float:
        """Total filled quantity"""
        return sum(fill.filled_quantity for fill in self.fills)

    @property
    def fill_rate(self) -> float:
        """Fill rate (0-1)"""
        return self.filled_quantity / self.original_quantity if self.original_quantity > 0 else 0

    @property
    def average_fill_price(self) -> float:
        """Volume-weighted average fill price"""
        if not self.fills:
            return 0.0

        total_value = sum(fill.filled_quantity * fill.fill_price for fill in self.fills)
        total_quantity = sum(fill.filled_quantity for fill in self.fills)

        return total_value / total_quantity if total_quantity > 0 else 0.0

    @property
    def total_fees(self) -> float:
        """Total fees paid"""
        return sum(fill.fee_paid for fill in self.fills)

    @property
    def maker_ratio(self) -> float:
        """Ratio of maker fills"""
        if not self.fills:
            return 0.0

        maker_fills = sum(1 for fill in self.fills if fill.fee_type == "maker")
        return maker_fills / len(self.fills)

    def is_expired(self) -> bool:
        """Check if order has expired"""
        time_since_creation = datetime.now() - self.created_at
        return time_since_creation.total_seconds() / 60 > self.max_fill_time_minutes


@dataclass
class FillRecommendation:
    """Recommendation for handling partial fill"""
    strategy: FillStrategy
    action: str  # "wait", "cancel", "modify", "complete"
    reasoning: str
    confidence: float

    # Action parameters
    suggested_price: Optional[float] = None
    suggested_size: Optional[float] = None
    timeout_minutes: Optional[int] = None

    # Expected outcomes
    expected_fill_rate: float = 0.0
    expected_time_minutes: int = 0


class PartialFillHandler:
    """
    Intelligent handler for partial order fills
    """

    def __init__(self):
        self.active_orders = {}  # order_id -> PartialOrder
        self.completed_orders = []
        self.fill_history = []

        # Performance tracking
        self.strategy_performance = {}

        # Configuration
        self.default_timeout_minutes = 30
        self.min_meaningful_fill_pct = 0.05  # 5% minimum meaningful fill

    def track_order(self,
                   order_id: str,
                   pair: str,
                   side: str,
                   quantity: float,
                   target_price: float,
                   order_type: str = "limit",
                   **kwargs) -> PartialOrder:
        """Start tracking a new order"""
        try:
            order = PartialOrder(
                order_id=order_id,
                pair=pair,
                side=side,
                original_quantity=quantity,
                target_price=target_price,
                order_type=order_type,
                max_fill_time_minutes=kwargs.get("timeout_minutes", self.default_timeout_minutes),
                min_fill_size=kwargs.get("min_fill_size", quantity * self.min_meaningful_fill_pct),
                allow_partial_completion=kwargs.get("allow_partial", True)
            )

            self.active_orders[order_id] = order

            logger.info(f"Started tracking order {order_id}: {quantity} {pair} @ {target_price}")

            return order

        except Exception as e:
            logger.error(f"Failed to track order {order_id}: {e}")
            raise

    def record_fill(self,
                   order_id: str,
                   filled_quantity: float,
                   fill_price: float,
                   fee_paid: float,
                   fee_type: str,
                   market_data: Dict[str, Any]) -> Optional[PartialOrder]:
        """Record a fill for an order"""
        try:
            if order_id not in self.active_orders:
                logger.warning(f"Fill recorded for unknown order: {order_id}")
                return None

            order = self.active_orders[order_id]

            # Create fill record
            fill = OrderFill(
                fill_id=f"{order_id}_{len(order.fills)+1}",
                timestamp=datetime.now(),
                filled_quantity=filled_quantity,
                fill_price=fill_price,
                fee_paid=fee_paid,
                fee_type=fee_type,
                market_price=market_data.get("mid_price", fill_price),
                spread_bp=market_data.get("spread_bp", 0)
            )

            # Add fill to order
            order.fills.append(fill)
            order.last_fill_at = datetime.now()
            order.remaining_quantity = max(0, order.remaining_quantity - filled_quantity)

            # Update status
            if order.remaining_quantity <= 0:
                order.status = FillStatus.COMPLETE
                self._complete_order(order_id)
            elif order.filled_quantity >= order.min_fill_size:
                order.status = FillStatus.PARTIAL

            # Record for analytics
            self._record_fill_analytics(order, fill)

            logger.info(f"Fill recorded for {order_id}: {filled_quantity} @ {fill_price} "
                       f"({order.fill_rate:.1%} complete)")

            return order

        except Exception as e:
            logger.error(f"Failed to record fill for {order_id}: {e}")
            return None

    def get_fill_recommendation(self,
                              order_id: str,
                              current_market_data: Dict[str, Any],
                              urgency: str = "normal") -> Optional[FillRecommendation]:
        """Get recommendation for handling partial fill"""
        try:
            if order_id not in self.active_orders:
                return None

            order = self.active_orders[order_id]

            # Analyze current situation
            analysis = self._analyze_fill_situation(order, current_market_data, urgency)

            # Determine strategy
            strategy = self._determine_fill_strategy(order, analysis, urgency)

            # Generate recommendation
            recommendation = self._build_fill_recommendation(order, strategy, analysis)

            return recommendation

        except Exception as e:
            logger.error(f"Failed to generate fill recommendation for {order_id}: {e}")
            return None

    def handle_expired_orders(self) -> List[str]:
        """Handle expired orders and return list of expired order IDs"""
        try:
            expired_orders = []

            for order_id, order in list(self.active_orders.items()):
                if order.is_expired():
                    # Decide how to handle expiration
                    if order.fill_rate >= 0.8:  # 80% filled
                        # Accept partial completion
                        order.status = FillStatus.COMPLETE
                        self._complete_order(order_id)
                        logger.info(f"Order {order_id} completed with {order.fill_rate:.1%} fill")
                    else:
                        # Mark as expired
                        order.status = FillStatus.CANCELLED
                        expired_orders.append(order_id)
                        self._expire_order(order_id)
                        logger.warning(f"Order {order_id} expired with only {order.fill_rate:.1%} fill")

            return expired_orders

        except Exception as e:
            logger.error(f"Failed to handle expired orders: {e}")
            return []

    def get_fill_analytics(self,
                          order_id: Optional[str] = None,
                          days_back: int = 7) -> Dict[str, Any]:
        """Get fill performance analytics"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)

            # Filter relevant data
            if order_id:
                if order_id in self.active_orders:
                    orders = [self.active_orders[order_id]]
                else:
                    orders = [order for order in self.completed_orders if order.order_id == order_id]
            else:
                # All recent orders
                orders = (list(self.active_orders.values()) +
                         [order for order in self.completed_orders if order.created_at >= cutoff_time])

            if not orders:
                return {"status": "No orders found"}

            # Calculate analytics
            analytics = self._calculate_fill_analytics(orders)

            return analytics

        except Exception as e:
            logger.error(f"Failed to generate fill analytics: {e}")
            return {"status": "Error", "error": str(e)}

    def _analyze_fill_situation(self,
                              order: PartialOrder,
                              market_data: Dict[str, Any],
                              urgency: str) -> Dict[str, Any]:
        """Analyze current fill situation"""
        try:
            current_time = datetime.now()
            time_elapsed = (current_time - order.created_at).total_seconds() / 60  # minutes
            time_remaining = order.max_fill_time_minutes - time_elapsed

            # Market condition analysis
            current_price = market_data.get("mid_price", order.target_price)
            spread_bp = market_data.get("spread_bp", 0)
            depth = market_data.get("depth_quote", 0)

            # Price movement analysis
            price_change_bp = 0
            if order.fills:
                last_fill_price = order.fills[-1].fill_price
                price_change_bp = (current_price - last_fill_price) / last_fill_price * 10000

            # Fill velocity analysis
            fill_velocity = 0  # fills per minute
            if order.fills and time_elapsed > 0:
                fill_velocity = len(order.fills) / time_elapsed

            # Market condition classification
            market_condition = "normal"
            if spread_bp > 30:
                market_condition = "illiquid"
            elif market_data.get("volatility_1h", 0.02) > 0.05:
                market_condition = "volatile"
            elif depth < order.remaining_quantity * current_price * 2:
                market_condition = "thin"

            return {
                "time_elapsed_minutes": time_elapsed,
                "time_remaining_minutes": time_remaining,
                "fill_rate": order.fill_rate,
                "price_change_bp": price_change_bp,
                "current_spread_bp": spread_bp,
                "fill_velocity": fill_velocity,
                "market_condition": market_condition,
                "price_favorable": self._is_price_favorable(order, current_price),
                "liquidity_adequate": depth > order.remaining_quantity * current_price
            }

        except Exception as e:
            logger.error(f"Fill situation analysis failed: {e}")
            return {}

    def _determine_fill_strategy(self,
                               order: PartialOrder,
                               analysis: Dict[str, Any],
                               urgency: str) -> FillStrategy:
        """Determine optimal fill strategy"""
        try:
            fill_rate = analysis.get("fill_rate", 0)
            time_remaining = analysis.get("time_remaining_minutes", 0)
            market_condition = analysis.get("market_condition", "normal")
            price_favorable = analysis.get("price_favorable", True)

            # High urgency - prioritize completion
            if urgency == "high":
                if fill_rate < 0.5:
                    return FillStrategy.CANCEL_AND_RETRY
                else:
                    return FillStrategy.SIZE_REDUCTION

            # Low time remaining - need to act
            if time_remaining < 5:  # Less than 5 minutes
                if fill_rate >= 0.8:
                    return FillStrategy.ACCUMULATE  # Close to completion
                else:
                    return FillStrategy.SIZE_REDUCTION  # Reduce remaining size

            # Market conditions
            if market_condition == "illiquid":
                return FillStrategy.TIME_PRIORITY  # Patient approach
            elif market_condition == "volatile":
                if fill_rate >= 0.5:
                    return FillStrategy.SIZE_REDUCTION  # Take what we can get
                else:
                    return FillStrategy.CANCEL_AND_RETRY  # Reset in volatile market

            # Price movement considerations
            if not price_favorable:
                if fill_rate >= 0.7:
                    return FillStrategy.ACCUMULATE  # Stay with current orders
                else:
                    return FillStrategy.CANCEL_AND_RETRY  # Reset at better price

            # Default strategy - continue accumulating
            return FillStrategy.ACCUMULATE

        except Exception as e:
            logger.error(f"Fill strategy determination failed: {e}")
            return FillStrategy.ACCUMULATE

    def _build_fill_recommendation(self,
                                 order: PartialOrder,
                                 strategy: FillStrategy,
                                 analysis: Dict[str, Any]) -> FillRecommendation:
        """Build detailed fill recommendation"""
        try:
            current_price = analysis.get("current_price", order.target_price)

            if strategy == FillStrategy.ACCUMULATE:
                return FillRecommendation(
                    strategy=strategy,
                    action="wait",
                    reasoning="Continue with current strategy - conditions favorable",
                    confidence=0.7,
                    expected_fill_rate=min(1.0, order.fill_rate + 0.3),
                    expected_time_minutes=int(analysis.get("time_remaining_minutes", 15) * 0.8)
                )

            elif strategy == FillStrategy.CANCEL_AND_RETRY:
                new_price = self._suggest_new_price(order, current_price, analysis)

                return FillRecommendation(
                    strategy=strategy,
                    action="cancel",
                    reasoning="Market conditions changed - retry with updated parameters",
                    confidence=0.8,
                    suggested_price=new_price,
                    suggested_size=order.remaining_quantity,
                    timeout_minutes=max(5, int(analysis.get("time_remaining_minutes", 10))),
                    expected_fill_rate=0.9,
                    expected_time_minutes=10
                )

            elif strategy == FillStrategy.SIZE_REDUCTION:
                reduced_size = max(order.min_fill_size, order.remaining_quantity * 0.7)

                return FillRecommendation(
                    strategy=strategy,
                    action="modify",
                    reasoning="Reduce size to improve fill probability",
                    confidence=0.8,
                    suggested_size=reduced_size,
                    expected_fill_rate=min(1.0, order.fill_rate + 0.4),
                    expected_time_minutes=8
                )

            elif strategy == FillStrategy.TIME_PRIORITY:
                return FillRecommendation(
                    strategy=strategy,
                    action="wait",
                    reasoning="Illiquid market - patient approach recommended",
                    confidence=0.6,
                    timeout_minutes=max(10, int(analysis.get("time_remaining_minutes", 20))),
                    expected_fill_rate=min(1.0, order.fill_rate + 0.2),
                    expected_time_minutes=15
                )

            # Fallback
            return FillRecommendation(
                strategy=FillStrategy.ACCUMULATE,
                action="wait",
                reasoning="Default strategy",
                confidence=0.5,
                expected_fill_rate=order.fill_rate + 0.1,
                expected_time_minutes=10
            )

        except Exception as e:
            logger.error(f"Fill recommendation building failed: {e}")
            return FillRecommendation(
                strategy=FillStrategy.ACCUMULATE,
                action="wait",
                reasoning="Error in analysis - using safe default",
                confidence=0.3
            )

    def _is_price_favorable(self, order: PartialOrder, current_price: float) -> bool:
        """Check if current price is favorable for order"""
        try:
            if order.side == "buy":
                # For buy orders, favorable if current price <= target price
                return current_price <= order.target_price * 1.01  # 1% tolerance
            else:
                # For sell orders, favorable if current price >= target price
                return current_price >= order.target_price * 0.99  # 1% tolerance

        except Exception as e:
            logger.error(f"Price favorability check failed: {e}")
            return True  # Conservative default

    def _suggest_new_price(self,
                          order: PartialOrder,
                          current_price: float,
                          analysis: Dict[str, Any]) -> float:
        """Suggest new price for order modification"""
        try:
            spread_bp = analysis.get("current_spread_bp", 10)

            if order.side == "buy":
                # For buy orders, suggest price slightly below current bid
                offset_bp = max(1, spread_bp * 0.3)
                suggested_price = current_price * (1 - offset_bp / 10000)
            else:
                # For sell orders, suggest price slightly above current ask
                offset_bp = max(1, spread_bp * 0.3)
                suggested_price = current_price * (1 + offset_bp / 10000)

            return suggested_price

        except Exception as e:
            logger.error(f"Price suggestion failed: {e}")
            return order.target_price

    def _complete_order(self, order_id: str) -> None:
        """Move order to completed status"""
        try:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                self.completed_orders.append(order)
                del self.active_orders[order_id]

                logger.info(f"Order {order_id} completed: {order.fill_rate:.1%} filled, "
                           f"avg price: {order.average_fill_price:.2f}")
        except Exception as e:
            logger.error(f"Failed to complete order {order_id}: {e}")

    def _expire_order(self, order_id: str) -> None:
        """Handle expired order"""
        try:
            if order_id in self.active_orders:
                order = self.active_orders[order_id]
                self.completed_orders.append(order)
                del self.active_orders[order_id]
        except Exception as e:
            logger.error(f"Failed to expire order {order_id}: {e}")

    def _record_fill_analytics(self, order: PartialOrder, fill: OrderFill) -> None:
        """Record fill for analytics"""
        try:
            fill_record = {
                "timestamp": fill.timestamp,
                "order_id": order.order_id,
                "pair": order.pair,
                "side": order.side,
                "fill_size": fill.filled_quantity,
                "fill_price": fill.fill_price,
                "fee_type": fill.fee_type,
                "fee_paid": fill.fee_paid,
                "slippage_bp": fill.slippage_bp,
                "order_fill_rate": order.fill_rate
            }

            self.fill_history.append(fill_record)

            # Keep only recent history
            if len(self.fill_history) > 10000:
                self.fill_history = self.fill_history[-10000:]

        except Exception as e:
            logger.error(f"Failed to record fill analytics: {e}")

    def _calculate_fill_analytics(self, orders: List[PartialOrder]) -> Dict[str, Any]:
        """Calculate comprehensive fill analytics"""
        try:
            if not orders:
                return {"status": "No orders to analyze"}

            # Basic statistics
            total_orders = len(orders)
            completed_orders = [o for o in orders if o.status == FillStatus.COMPLETE]
            partial_orders = [o for o in orders if o.status == FillStatus.PARTIAL]

            # Fill rates
            fill_rates = [order.fill_rate for order in orders]
            avg_fill_rate = np.mean(fill_rates)

            # Completion rates
            completion_rate = len(completed_orders) / total_orders if total_orders > 0 else 0

            # Fee analysis
            maker_ratios = [order.maker_ratio for order in orders if order.fills]
            avg_maker_ratio = np.mean(maker_ratios) if maker_ratios else 0

            # Timing analysis
            fill_times = []
            for order in completed_orders:
                if order.last_fill_at:
                    fill_time = (order.last_fill_at - order.created_at).total_seconds() / 60
                    fill_times.append(fill_time)

            avg_fill_time = np.mean(fill_times) if fill_times else 0

            # Price improvement analysis
            price_improvements = []
            for order in orders:
                if order.fills:
                    avg_price = order.average_fill_price
                    target_price = order.target_price

                    if order.side == "buy" and avg_price < target_price:
                        improvement_bp = (target_price - avg_price) / target_price * 10000
                        price_improvements.append(improvement_bp)
                    elif order.side == "sell" and avg_price > target_price:
                        improvement_bp = (avg_price - target_price) / target_price * 10000
                        price_improvements.append(improvement_bp)

            avg_price_improvement = np.mean(price_improvements) if price_improvements else 0

            return {
                "total_orders": total_orders,
                "completion_rate": completion_rate,
                "avg_fill_rate": avg_fill_rate,
                "avg_maker_ratio": avg_maker_ratio,
                "avg_fill_time_minutes": avg_fill_time,
                "avg_price_improvement_bp": avg_price_improvement,
                "orders_by_status": {
                    "complete": len(completed_orders),
                    "partial": len(partial_orders),
                    "cancelled": len([o for o in orders if o.status == FillStatus.CANCELLED])
                }
            }

        except Exception as e:
            logger.error(f"Fill analytics calculation failed: {e}")
            return {"status": "Error", "error": str(e)}
