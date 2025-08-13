
# Import from hard execution policy (FASE C implementation)
from .hard_execution_policy import (
    HardExecutionPolicy,
    OrderRequest, MarketConditions, ExecutionResult,
    OrderSide, TimeInForce, ExecutionDecision,
    get_execution_policy, reset_execution_policy
)

# Backward compatibility alias
ExecutionPolicy = HardExecutionPolicy

# Re-export all components
__all__ = [
    'ExecutionPolicy', 'HardExecutionPolicy',
    'OrderRequest', 'MarketConditions', 'ExecutionResult',
    'OrderSide', 'TimeInForce', 'ExecutionDecision',
    'get_execution_policy', 'reset_execution_policy'
]


"""ExecutionPolicy - Trading execution with guardrails, slippage control, and order idempotency."""

import asyncio
import time
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import threading
import uuid

from ..core.structured_logger import get_logger


class OrderStatus(Enum):
    """Order execution status."""

    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    """Time in force options."""

    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill
    POST_ONLY = "post_only"  # Post Only (maker)


class OrderType(Enum):
    """Order type options."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


@dataclass
class TradabilityGate:
    """Tradability assessment criteria."""

    min_volume_24h: float = 100000.0  # $100k minimum daily volume
    max_spread_percent: float = 0.5  # 0.5% maximum spread
    min_orderbook_depth: float = 10000.0  # $10k minimum depth
    max_price_impact_percent: float = 1.0  # 1% maximum price impact
    min_liquidity_score: float = 0.6  # 0.6 minimum liquidity score


@dataclass
class SlippageBudget:
    """Slippage budget configuration."""

    max_slippage_percent: float = 0.3  # 0.3% maximum slippage
    warning_threshold_percent: float = 0.2  # 0.2% warning threshold
    adaptive_sizing: bool = True  # Reduce size on high slippage
    emergency_stop_percent: float = 1.0  # 1% emergency stop


@dataclass
class OrderRequest:
    """Order execution request."""

    client_order_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    confidence_score: Optional[float] = None
    strategy_id: Optional[str] = None
    max_slippage_percent: Optional[float] = None
    post_only: bool = False
    reduce_only: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OrderResult:
    """Order execution result."""

    client_order_id: str
    exchange_order_id: Optional[str]
    status: OrderStatus
    filled_quantity: float
    avg_fill_price: float
    total_fees: float
    slippage_percent: float
    execution_time_ms: int
    error_message: Optional[str] = None
    retry_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MarketConditions:
    """Current market conditions for execution assessment."""

    bid_price: float
    ask_price: float
    mid_price: float
    spread_percent: float
    volume_24h: float
    orderbook_depth_bid: float
    orderbook_depth_ask: float
    price_volatility: float
    liquidity_score: float
    timestamp: datetime = field(default_factory=datetime.now)


class ExecutionPolicy:
    """Enterprise trading execution policy with comprehensive guardrails."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize execution policy system."""
        self.logger = get_logger("execution_policy")

        # Load configuration
        self.tradability_gate = self._load_tradability_config(config_path)
        self.slippage_budget = self._load_slippage_config(config_path)

        # Order tracking and idempotency
        self.order_cache: Dict[str, OrderResult] = {}
        self.execution_history: List[OrderResult] = []
        self.retry_tracker: Dict[str, int] = {}

        # Deduplication window (60 minutes)
        self.dedup_window_minutes = 60
        self.order_hashes: Dict[str, datetime] = {}

        # Execution metrics
        self.execution_stats = {
            "total_orders": 0,
            "successful_orders": 0,
            "rejected_orders": 0,
            "average_slippage": 0.0,
            "average_execution_time": 0.0,
            "retry_rate": 0.0,
        }

        # Market conditions cache
        self.market_conditions: Dict[str, MarketConditions] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Persistence
        self.data_path = Path("data/execution_policy")
        self.data_path.mkdir(parents=True, exist_ok=True)

        # Network timeout and retry configuration
        self.network_timeout_seconds = 30
        self.max_retries = 3
        self.retry_delay_base = 1.0  # Exponential backoff base

        self.logger.info(
            "ExecutionPolicy initialized",
            tradability_gate=self.tradability_gate.__dict__,
            slippage_budget=self.slippage_budget.__dict__,
        )

    def _load_tradability_config(self, config_path: Optional[str]) -> TradabilityGate:
        """Load tradability gate configuration."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                return TradabilityGate(**config.get("tradability_gate", {}))
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                self.logger.warning(f"Failed to load tradability config: {e}")

        return TradabilityGate()  # Use defaults

    def _load_slippage_config(self, config_path: Optional[str]) -> SlippageBudget:
        """Load slippage budget configuration."""
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    config = json.load(f)
                return SlippageBudget(**config.get("slippage_budget", {}))
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                self.logger.warning(f"Failed to load slippage config: {e}")

        return SlippageBudget()  # Use defaults

    def generate_client_order_id(
        self, symbol: str, side: str, quantity: float, strategy_id: Optional[str] = None
    ) -> str:
        """Generate deterministic client order ID for idempotency."""
        # Create deterministic hash from order parameters
        order_data = (
            f"{symbol}:{side}:{quantity}:{strategy_id}:{datetime.now().strftime('%Y%m%d%H%M')}"
        )
        order_hash = hashlib.sha256(order_data.encode()).hexdigest()[:16]

        # Format: CST_YYYYMMDD_HASH_UUID
        timestamp = datetime.now().strftime("%Y%m%d")
        unique_id = str(uuid.uuid4()).split("-")[0]

        return f"CST_{timestamp}_{order_hash}_{unique_id}"

    def check_order_deduplication(self, order_request: OrderRequest) -> bool:
        """Check if order is duplicate within deduplication window."""
        # Create content hash for deduplication
        content = f"{order_request.symbol}:{order_request.side}:{order_request.quantity}:{order_request.price}"
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        with self._lock:
            now = datetime.now()

            # Clean expired hashes
            expired_hashes = [
                h
                for h, ts in self.order_hashes.items()
                if (now - ts).total_seconds() > self.dedup_window_minutes * 60
            ]
            for h in expired_hashes:
                del self.order_hashes[h]

            # Check for duplicate
            if content_hash in self.order_hashes:
                self.logger.warning(
                    "Duplicate order detected within deduplication window",
                    content_hash=content_hash[:8],
                    original_time=self.order_hashes[content_hash],
                )
                return True

            # Register new order
            self.order_hashes[content_hash] = now
            return False

    def update_market_conditions(self, symbol: str, conditions: MarketConditions) -> None:
        """Update market conditions for symbol."""
        with self._lock:
            self.market_conditions[symbol] = conditions

    def assess_tradability(self, symbol: str) -> Tuple[bool, List[str]]:
        """Assess if symbol meets tradability criteria."""
        if symbol not in self.market_conditions:
            return False, ["No market data available"]

        conditions = self.market_conditions[symbol]
        issues = []

        # Volume check
        if conditions.volume_24h < self.tradability_gate.min_volume_24h:
            issues.append(
                f"Volume {conditions.volume_24h:.0f} below minimum {self.tradability_gate.min_volume_24h:.0f}"
            )

        # Spread check
        if conditions.spread_percent > self.tradability_gate.max_spread_percent:
            issues.append(
                f"Spread {conditions.spread_percent:.3f}% above maximum {self.tradability_gate.max_spread_percent:.3f}%"
            )

        # Orderbook depth check
        min_depth = min(conditions.orderbook_depth_bid, conditions.orderbook_depth_ask)
        if min_depth < self.tradability_gate.min_orderbook_depth:
            issues.append(
                f"Orderbook depth {min_depth:.0f} below minimum {self.tradability_gate.min_orderbook_depth:.0f}"
            )

        # Liquidity score check
        if conditions.liquidity_score < self.tradability_gate.min_liquidity_score:
            issues.append(
                f"Liquidity score {conditions.liquidity_score:.2f} below minimum {self.tradability_gate.min_liquidity_score:.2f}"
            )

        tradable = len(issues) == 0
        return tradable, issues

    def estimate_slippage(self, order_request: OrderRequest) -> float:
        """Estimate expected slippage for order."""
        symbol = order_request.symbol
        if symbol not in self.market_conditions:
            return 1.0  # Conservative estimate

        conditions = self.market_conditions[symbol]

        # Base slippage from spread
        base_slippage = conditions.spread_percent / 2

        # Size impact estimation
        relevant_depth = (
            conditions.orderbook_depth_bid
            if order_request.side == "sell"
            else conditions.orderbook_depth_ask
        )

        size_impact = min(
            2.0, (order_request.quantity * conditions.mid_price) / relevant_depth * 100
        )

        # Volatility adjustment
        volatility_adjustment = conditions.price_volatility * 0.5

        # Market order additional impact
        market_impact = 0.1 if order_request.order_type == OrderType.MARKET else 0.0

        total_slippage = base_slippage + size_impact + volatility_adjustment + market_impact

        return min(5.0, total_slippage)  # Cap at 5%

    def adjust_order_size(self, order_request: OrderRequest, estimated_slippage: float) -> float:
        """Adjust order size based on slippage budget."""
        if not self.slippage_budget.adaptive_sizing:
            return order_request.quantity

        max_slippage = (
            order_request.max_slippage_percent or self.slippage_budget.max_slippage_percent
        )

        if estimated_slippage <= max_slippage:
            return order_request.quantity

        # Reduce size proportionally
        size_factor = max_slippage / estimated_slippage
        adjusted_quantity = order_request.quantity * size_factor

        self.logger.info(
            "Order size adjusted due to slippage",
            original_quantity=order_request.quantity,
            adjusted_quantity=adjusted_quantity,
            estimated_slippage=estimated_slippage,
            max_slippage=max_slippage,
        )

        return adjusted_quantity

    def validate_order_request(self, order_request: OrderRequest) -> Tuple[bool, List[str]]:
        """Comprehensive order request validation."""
        issues = []

        # Basic validation
        if order_request.quantity <= 0:
            issues.append("Invalid quantity: must be positive")

        if order_request.side not in ["buy", "sell"]:
            issues.append("Invalid side: must be 'buy' or 'sell'")

        # Price validation for limit orders
        if order_request.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order_request.price is None or order_request.price <= 0:
                issues.append("Limit orders require valid price")

        # Tradability check
        tradable, tradability_issues = self.assess_tradability(order_request.symbol)
        if not tradable:
            issues.extend(tradability_issues)

        # Slippage check
        estimated_slippage = self.estimate_slippage(order_request)
        max_slippage = (
            order_request.max_slippage_percent or self.slippage_budget.max_slippage_percent
        )

        if estimated_slippage > self.slippage_budget.emergency_stop_percent:
            issues.append(
                f"Estimated slippage {estimated_slippage:.2f}% exceeds emergency stop {self.slippage_budget.emergency_stop_percent:.2f}%"
            )

        # Deduplication check
        if self.check_order_deduplication(order_request):
            issues.append("Duplicate order within deduplication window")

        return len(issues) == 0, issues

    async def execute_order_with_retry(
        self, order_request: OrderRequest, exchange_client
    ) -> OrderResult:
        """Execute order with exponential backoff retry logic."""
        start_time = time.time()

        # Validate order first
        valid, validation_issues = self.validate_order_request(order_request)
        if not valid:
            return OrderResult(
                client_order_id=order_request.client_order_id,
                exchange_order_id=None,
                status=OrderStatus.REJECTED,
                filled_quantity=0.0,
                avg_fill_price=0.0,
                total_fees=0.0,
                slippage_percent=0.0,
                execution_time_ms=0,
                error_message=f"Validation failed: {'; '.join(validation_issues)}",
            )

        # Adjust order size if needed
        estimated_slippage = self.estimate_slippage(order_request)
        adjusted_quantity = self.adjust_order_size(order_request, estimated_slippage)

        if adjusted_quantity != order_request.quantity:
            order_request.quantity = adjusted_quantity

        # Execute with retry logic
        last_error = None
        retry_count = 0

        for attempt in range(self.max_retries + 1):
            try:
                result = await self._execute_single_order(order_request, exchange_client)

                # Record successful execution
                execution_time_ms = int((time.time() - start_time) * 1000)
                result.execution_time_ms = execution_time_ms
                result.retry_count = retry_count

                with self._lock:
                    self.order_cache[order_request.client_order_id] = result
                    self.execution_history.append(result)
                    self._update_execution_stats(result)

                self.logger.info(
                    "Order executed successfully",
                    client_order_id=order_request.client_order_id,
                    status=result.status.value,
                    filled_quantity=result.filled_quantity,
                    slippage_percent=result.slippage_percent,
                    retry_count=retry_count,
                )

                return result

            except (ConnectionError, TimeoutError, asyncio.TimeoutError) as e:
                last_error = str(e)
                retry_count += 1

                if attempt < self.max_retries:
                    delay = self.retry_delay_base * (2**attempt)  # Exponential backoff
                    self.logger.warning(
                        f"Order execution failed, retrying in {delay}s",
                        client_order_id=order_request.client_order_id,
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(
                        "Order execution failed after all retries",
                        client_order_id=order_request.client_order_id,
                        final_error=str(e),
                    )

            except Exception as e:
                # Non-retryable error
                self.logger.error(
                    "Order execution failed with non-retryable error",
                    client_order_id=order_request.client_order_id,
                    error=str(e),
                )
                last_error = str(e)
                break

        # Failed execution
        return OrderResult(
            client_order_id=order_request.client_order_id,
            exchange_order_id=None,
            status=OrderStatus.REJECTED,
            filled_quantity=0.0,
            avg_fill_price=0.0,
            total_fees=0.0,
            slippage_percent=0.0,
            execution_time_ms=int((time.time() - start_time) * 1000),
            error_message=f"Execution failed after {retry_count} retries: {last_error}",
            retry_count=retry_count,
        )

    async def _execute_single_order(
        self, order_request: OrderRequest, exchange_client
    ) -> OrderResult:
        """Execute single order attempt."""
        # This would integrate with actual exchange client
        # For now, return simulated result

        # Simulate network delay
        await asyncio.sleep(0.1)

        # Get market conditions for slippage calculation
        conditions = self.market_conditions.get(order_request.symbol)
        if not conditions:
            raise ValueError("No market conditions available")

        # Calculate execution price and slippage
        if order_request.order_type == OrderType.MARKET:
            if order_request.side == "buy":
                execution_price = conditions.ask_price
                expected_price = conditions.mid_price
            else:
                execution_price = conditions.bid_price
                expected_price = conditions.mid_price
        else:
            execution_price = order_request.price or conditions.mid_price
            expected_price = conditions.mid_price

        # Calculate slippage
        slippage_percent = abs(execution_price - expected_price) / expected_price * 100

        # Simulate fees (0.1% taker fee)
        total_fees = order_request.quantity * execution_price * 0.001

        return OrderResult(
            client_order_id=order_request.client_order_id,
            exchange_order_id=f"EXC_{int(time.time())}_{order_request.client_order_id[-8:]}",
            status=OrderStatus.FILLED,
            filled_quantity=order_request.quantity,
            avg_fill_price=execution_price,
            total_fees=total_fees,
            slippage_percent=slippage_percent,
            execution_time_ms=100.0,  # Simulated execution time
        )

    def _update_execution_stats(self, result: OrderResult) -> None:
        """Update execution statistics."""
        self.execution_stats["total_orders"] += 1

        if result.status == OrderStatus.FILLED:
            self.execution_stats["successful_orders"] += 1

            # Update moving averages
            n = self.execution_stats["successful_orders"]
            self.execution_stats["average_slippage"] = (
                self.execution_stats["average_slippage"] * (n - 1) + result.slippage_percent
            ) / n
            self.execution_stats["average_execution_time"] = (
                self.execution_stats["average_execution_time"] * (n - 1) + result.execution_time_ms
            ) / n
        else:
            self.execution_stats["rejected_orders"] += 1

        # Update retry rate
        total_retries = sum(r.retry_count for r in self.execution_history[-100:])  # Last 100 orders
        recent_orders = min(100, len(self.execution_history))
        self.execution_stats["retry_rate"] = total_retries / max(1, recent_orders)

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get current execution metrics."""
        with self._lock:
            return {
                **self.execution_stats,
                "cache_size": len(self.order_cache),
                "history_size": len(self.execution_history),
                "dedup_entries": len(self.order_hashes),
                "market_conditions_count": len(self.market_conditions),
            }

    def get_order_status(self, client_order_id: str) -> Optional[OrderResult]:
        """Get order status by client order ID."""
        with self._lock:
            return self.order_cache.get(client_order_id)

    def cancel_order(self, client_order_id: str) -> bool:
        """Cancel pending order."""
        # Implementation would cancel order on exchange
        with self._lock:
            if client_order_id in self.order_cache:
                result = self.order_cache[client_order_id]
                if result.status in [
                    OrderStatus.PENDING,
                    OrderStatus.SUBMITTED,
                    OrderStatus.PARTIAL,
                ]:
                    result.status = OrderStatus.CANCELLED
                    self.logger.info("Order cancelled", client_order_id=client_order_id)
                    return True
        return False


def create_execution_policy(config_path: Optional[str] = None) -> ExecutionPolicy:
    """Factory function to create ExecutionPolicy instance."""
    return ExecutionPolicy(config_path=config_path)
