#!/usr/bin/env python3
"""
Orderbook Simulator
Realistic Level-2 orderbook simulation with market impact, slippage, and partial fills
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
import warnings
import heapq

warnings.filterwarnings("ignore")


class OrderType(Enum):
    """Order types"""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order sides"""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""

    PENDING = "pending"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class OrderBookLevel:
    """Single level in orderbook"""

    price: float
    quantity: float
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())


@dataclass
class Order:
    """Order representation"""

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())
    latency_ms: float = 0.0
    fees_paid: float = 0.0
    slippage: float = 0.0


@dataclass
class Fill:
    """Order fill representation"""

    order_id: str
    fill_id: str
    price: float
    quantity: float
    fee: float
    timestamp: datetime
    is_maker: bool = False


@dataclass
class OrderBookSnapshot:
    """Complete orderbook snapshot"""

    symbol: str
    timestamp: datetime
    bids: List[OrderBookLevel]  # Sorted descending by price
    asks: List[OrderBookLevel]  # Sorted ascending by price
    last_price: float
    volume_24h: float = 0.0


class ExchangeConfig:
    """Exchange-specific configuration"""

    def __init__(self, exchange_name: str = "kraken"):
        self.exchange_name = exchange_name

        # Fee structure (maker/taker)
        if exchange_name.lower() == "kraken":
            self.maker_fee = 0.0016  # 0.16%
            self.taker_fee = 0.0026  # 0.26%
            self.min_order_size = 0.0001
            self.price_precision = 2
            self.quantity_precision = 8

        elif exchange_name.lower() == "binance":
            self.maker_fee = 0.001  # 0.1%
            self.taker_fee = 0.001  # 0.1%
            self.min_order_size = 0.00001
            self.price_precision = 2
            self.quantity_precision = 8

        elif exchange_name.lower() == "coinbase":
            self.maker_fee = 0.005  # 0.5%
            self.taker_fee = 0.005  # 0.5%
            self.min_order_size = 0.001
            self.price_precision = 2
            self.quantity_precision = 8

        else:  # Generic exchange
            self.maker_fee = 0.002  # 0.2%
            self.taker_fee = 0.003  # 0.3%
            self.min_order_size = 0.001
            self.price_precision = 2
            self.quantity_precision = 8

        # Latency characteristics
        self.base_latency_ms = self._get_base_latency()
        self.latency_std_ms = self.base_latency_ms * 0.5

        # Market impact parameters
        self.market_impact_coefficient = 0.001  # Impact per % of volume
        self.liquidity_recovery_rate = 0.95  # How fast liquidity recovers

    def _get_base_latency(self) -> float:
        """Get base latency for exchange"""
        latency_map = {
            "kraken": 50.0,  # 50ms average
            "binance": 20.0,  # 20ms average
            "coinbase": 30.0,  # 30ms average
            "generic": 40.0,  # 40ms average
        }
        return latency_map.get(self.exchange_name.lower(), 40.0)


class SlippageModel:
    """Advanced slippage model considering market impact and liquidity"""

    def __init__(self, exchange_config: ExchangeConfig):
        self.config = exchange_config
        self.logger = logging.getLogger(__name__)

    def calculate_slippage(
        self,
        orderbook: OrderBookSnapshot,
        order_quantity: float,
        order_side: OrderSide,
        volatility: float = 0.02,
    ) -> Tuple[float, List[Tuple[float, float]]]:
        """Calculate slippage and price levels for order execution"""

        # Get relevant side of orderbook
        if order_side == OrderSide.BUY:
            levels = orderbook.asks  # Buy from asks
        else:
            levels = orderbook.bids  # Sell to bids

        if not levels:
            # No liquidity - high slippage
            mid_price = orderbook.last_price
            slippage_rate = 0.01 + volatility  # 1% + volatility
            slippage_price = mid_price * (
                1 + slippage_rate if order_side == OrderSide.BUY else 1 - slippage_rate
            )
            return slippage_rate, [(slippage_price, order_quantity)]

        # Calculate market impact
        total_available_liquidity = sum(level.quantity for level in levels[:10])  # Top 10 levels
        impact_ratio = order_quantity / max(total_available_liquidity, order_quantity)

        # Base market impact
        base_impact = self.config.market_impact_coefficient * impact_ratio

        # Volatility adjustment
        volatility_impact = volatility * 0.5

        # Total impact
        total_impact = base_impact + volatility_impact

        # Calculate execution levels
        remaining_quantity = order_quantity
        execution_levels = []
        cumulative_impact = 0.0

        for i, level in enumerate(levels):
            if remaining_quantity <= 0:
                break

            # Progressive impact (deeper levels have more impact)
            level_impact_multiplier = 1.0 + (i * 0.1)  # 10% additional impact per level
            level_impact = total_impact * level_impact_multiplier

            # Apply impact to price
            if order_side == OrderSide.BUY:
                adjusted_price = level.price * (1 + level_impact)
            else:
                adjusted_price = level.price * (1 - level_impact)

            # Determine fill quantity at this level
            available_quantity = level.quantity
            fill_quantity = min(remaining_quantity, available_quantity)

            execution_levels.append((adjusted_price, fill_quantity))
            remaining_quantity -= fill_quantity
            cumulative_impact += level_impact * (fill_quantity / order_quantity)

        # If order is larger than available liquidity
        if remaining_quantity > 0:
            # Execute remaining at highly slipped price
            worst_price = levels[-1].price if levels else orderbook.last_price
            extreme_impact = total_impact * 3  # 3x impact for liquidity exhaustion

            if order_side == OrderSide.BUY:
                slipped_price = worst_price * (1 + extreme_impact)
            else:
                slipped_price = worst_price * (1 - extreme_impact)

            execution_levels.append((slipped_price, remaining_quantity))
            cumulative_impact += extreme_impact * (remaining_quantity / order_quantity)

        return cumulative_impact, execution_levels


class OrderBookSimulator:
    """Realistic Level-2 orderbook simulator"""

    def __init__(
        self,
        exchange_config: ExchangeConfig,
        initial_price: float = 50000.0,
        spread_bps: int = 10,  # 10 basis points = 0.1%
    ):
        self.config = exchange_config
        self.initial_price = initial_price
        self.spread_bps = spread_bps

        # Initialize orderbook
        self.orderbook = self._create_initial_orderbook()

        # Slippage model
        self.slippage_model = SlippageModel(exchange_config)

        # Order tracking
        self.orders: Dict[str, Order] = {}
        self.fills: List[Fill] = []
        self.next_order_id = 1
        self.next_fill_id = 1

        # Market state
        self.current_volatility = 0.02  # 2% daily volatility
        self.liquidity_factor = 1.0  # Multiplier for available liquidity

        self.logger = logging.getLogger(__name__)

    def _create_initial_orderbook(self) -> OrderBookSnapshot:
        """Create realistic initial orderbook"""

        mid_price = self.initial_price
        spread = mid_price * (self.spread_bps / 10000)

        # Create bid levels (descending prices)
        bids = []
        for i in range(20):  # 20 levels
            price = mid_price - spread / 2 - i * (spread * 0.1)
            # Liquidity decreases with distance from mid
            base_quantity = 1.0 + np.random.exponential(2.0)
            quantity = base_quantity * (1.0 / (1 + i * 0.1))
            bids.append(OrderBookLevel(price, quantity))

        # Create ask levels (ascending prices)
        asks = []
        for i in range(20):  # 20 levels
            price = mid_price + spread / 2 + i * (spread * 0.1)
            base_quantity = 1.0 + np.random.exponential(2.0)
            quantity = base_quantity * (1.0 / (1 + i * 0.1))
            asks.append(OrderBookLevel(price, quantity))

        return OrderBookSnapshot(
            symbol="BTC/USD",
            timestamp=datetime.utcnow(),
            bids=bids,
            asks=asks,
            last_price=mid_price,
        )

    def update_orderbook(self, price_change: float, volatility: float = None):
        """Update orderbook based on price movement"""

        if volatility is not None:
            self.current_volatility = volatility

        # Update last price
        new_price = self.orderbook.last_price * (1 + price_change)
        self.orderbook.last_price = new_price

        # Adjust all levels proportionally with some noise
        noise_factor = self.current_volatility * 0.1

        # Update bids
        for level in self.orderbook.bids:
            price_adjustment = price_change + np.random.normal(0, noise_factor)
            level.price *= 1 + price_adjustment

            # Quantity may change due to market activity
            quantity_change = np.random.normal(0, 0.05)  # 5% std
            level.quantity *= max(0.1, 1 + quantity_change)

        # Update asks
        for level in self.orderbook.asks:
            price_adjustment = price_change + np.random.normal(0, noise_factor)
            level.price *= 1 + price_adjustment

            quantity_change = np.random.normal(0, 0.05)
            level.quantity *= max(0.1, 1 + quantity_change)

        # Re-sort to maintain order
        self.orderbook.bids.sort(key=lambda x: x.price, reverse=True)
        self.orderbook.asks.sort(key=lambda x: x.price)

        # Update timestamp
        self.orderbook.timestamp = datetime.utcnow()

    def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
    ) -> Order:
        """Submit order to simulator - HARD WIRED TO GATEWAY"""
        
        # MANDATORY GATEWAY ENFORCEMENT
        try:
            from src.cryptosmarttrader.core.mandatory_execution_gateway import enforce_mandatory_gateway, UniversalOrderRequest
            
            gateway_order = UniversalOrderRequest(
                symbol=symbol,
                side=side.value,
                size=quantity,
                order_type=order_type.value,
                limit_price=price,
                stop_price=stop_price,
                strategy_id="orderbook_simulator_trading",
                source_module="trading.orderbook_simulator",
                source_function="submit_order"
            )
            
            gateway_result = enforce_mandatory_gateway(gateway_order)
            
            if not gateway_result.approved:
                # Return rejected order
                order_id = f"rejected_{self.next_order_id}"
                self.next_order_id += 1
                
                rejected_order = Order(
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    order_type=order_type,
                    quantity=quantity,
                    price=price,
                    stop_price=stop_price,
                    latency_ms=0.0
                )
                rejected_order.status = OrderStatus.REJECTED
                self.logger.warning(f"Order {order_id} rejected by gateway: {gateway_result.reason}")
                return rejected_order
            
            # Use approved size
            quantity = gateway_result.approved_size
            
        except Exception as e:
            # Return error order
            order_id = f"error_{self.next_order_id}"
            self.next_order_id += 1
            
            error_order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                price=price,
                stop_price=stop_price,
                latency_ms=0.0
            )
            error_order.status = OrderStatus.REJECTED
            self.logger.error(f"Gateway error for order {order_id}: {str(e)}")
            return error_order

        # Generate order ID
        order_id = f"order_{self.next_order_id}"
        self.next_order_id += 1

        # Simulate network latency
        latency_ms = max(
            0, np.random.normal(self.config.base_latency_ms, self.config.latency_std_ms)
        )

        # Create order
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            latency_ms=latency_ms,
        )

        # Validate order
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            self.logger.warning(f"Order {order_id} rejected")
            return order

        # Store order
        self.orders[order_id] = order

        # Execute order if market order or triggered
        if order_type == OrderType.MARKET:
            self._execute_market_order(order)
        elif order_type == OrderType.LIMIT:
            self._process_limit_order(order)

        return order

    def _validate_order(self, order: Order) -> bool:
        """Validate order parameters"""

        # Check minimum size
        if order.quantity < self.config.min_order_size:
            return False

        # Check price precision
        if order.price is not None:
            if round(order.price, self.config.price_precision) != order.price:
                return False

        # Check quantity precision
        if round(order.quantity, self.config.quantity_precision) != order.quantity:
            return False

        return True

    def _execute_market_order(self, order: Order):
        """Execute market order with realistic slippage"""

        # Calculate slippage and execution levels
        slippage_rate, execution_levels = self.slippage_model.calculate_slippage(
            self.orderbook, order.quantity, order.side, self.current_volatility
        )

        order.slippage = slippage_rate

        # Execute fills at each level
        total_filled = 0.0
        total_cost = 0.0
        total_fees = 0.0

        for price, quantity in execution_levels:
            if total_filled >= order.quantity:
                break

            # Create fill
            fill_quantity = min(quantity, order.quantity - total_filled)

            # Calculate fees (always taker for market orders)
            fee_rate = self.config.taker_fee
            fee = fill_quantity * price * fee_rate

            # Create fill record
            fill_id = f"fill_{self.next_fill_id}"
            self.next_fill_id += 1

            fill = Fill(
                order_id=order.order_id,
                fill_id=fill_id,
                price=price,
                quantity=fill_quantity,
                fee=fee,
                timestamp=datetime.utcnow(),
                is_maker=False,  # Market orders are always taker
            )

            self.fills.append(fill)

            # Update totals
            total_filled += fill_quantity
            total_cost += fill_quantity * price
            total_fees += fee

        # Update order
        order.filled_quantity = total_filled
        order.fees_paid = total_fees

        if total_filled >= order.quantity:
            order.status = OrderStatus.FILLED
            order.avg_fill_price = total_cost / total_filled
        else:
            order.status = OrderStatus.PARTIAL
            order.avg_fill_price = total_cost / total_filled if total_filled > 0 else 0

        # Update orderbook liquidity (consumed liquidity)
        self._update_liquidity_after_execution(order.side, execution_levels)

    def _process_limit_order(self, order: Order):
        """Process limit order (simplified - would need full matching engine)"""

        # For now, assume immediate execution if price crosses
        if order.side == OrderSide.BUY:
            if order.price >= self.orderbook.asks[0].price:
                # Execute as market order (price improvement)
                self._execute_market_order(order)
            else:
                # Order remains pending (would be added to orderbook)
                order.status = OrderStatus.PENDING
        else:  # SELL
            if order.price <= self.orderbook.bids[0].price:
                # Execute as market order
                self._execute_market_order(order)
            else:
                order.status = OrderStatus.PENDING

    def _update_liquidity_after_execution(
        self, side: OrderSide, execution_levels: List[Tuple[float, float]]
    ):
        """Update orderbook liquidity after order execution"""

        # Remove consumed liquidity
        relevant_levels = self.orderbook.asks if side == OrderSide.BUY else self.orderbook.bids

        level_index = 0
        for exec_price, exec_quantity in execution_levels:
            if level_index < len(relevant_levels):
                level = relevant_levels[level_index]
                level.quantity = max(0, level.quantity - exec_quantity)

                # Remove empty levels
                if level.quantity <= 0:
                    relevant_levels.pop(level_index)
                else:
                    level_index += 1

        # Simulate liquidity recovery over time (would be more complex in reality)
        self.liquidity_factor *= self.config.liquidity_recovery_rate

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """Get current order status"""
        return self.orders.get(order_id)

    def get_fills_for_order(self, order_id: str) -> List[Fill]:
        """Get all fills for specific order"""
        return [fill for fill in self.fills if fill.order_id == order_id]

    def get_best_bid_ask(self) -> Tuple[float, float]:
        """Get current best bid and ask prices"""

        if not self.orderbook.bids or not self.orderbook.asks:
            return self.orderbook.last_price, self.orderbook.last_price

        best_bid = self.orderbook.bids[0].price
        best_ask = self.orderbook.asks[0].price

        return best_bid, best_ask

    def get_market_impact_estimate(self, quantity: float, side: OrderSide) -> Dict[str, float]:
        """Estimate market impact for proposed order"""

        slippage_rate, execution_levels = self.slippage_model.calculate_slippage(
            self.orderbook, quantity, side, self.current_volatility
        )

        # Calculate weighted average execution price
        total_quantity = sum(level[1] for level in execution_levels)
        weighted_price = sum(level[0] * level[1] for level in execution_levels) / total_quantity

        # Reference price (best bid/ask)
        if side == OrderSide.BUY:
            reference_price = self.orderbook.asks[0].price
        else:
            reference_price = self.orderbook.bids[0].price

        # Price impact
        price_impact = abs(weighted_price - reference_price) / reference_price

        return {
            "estimated_avg_price": weighted_price,
            "reference_price": reference_price,
            "price_impact_bps": price_impact * 10000,
            "slippage_rate": slippage_rate,
            "execution_levels": len(execution_levels),
            "total_quantity": total_quantity,
        }


def create_orderbook_simulator(
    exchange: str = "kraken", initial_price: float = 50000.0, spread_bps: int = 10
) -> OrderBookSimulator:
    """Create configured orderbook simulator"""

    config = ExchangeConfig(exchange)
    return OrderBookSimulator(config, initial_price, spread_bps)


def simulate_realistic_execution(
    order_size: float,
    side: str,
    current_price: float,
    exchange: str = "kraken",
    volatility: float = 0.02,
) -> Dict[str, Any]:
    """High-level function to simulate realistic order execution"""

    # Create simulator
    simulator = create_orderbook_simulator(exchange, current_price)
    simulator.current_volatility = volatility

    # Convert side
    order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

    # Submit market order
    order = simulator.submit_order(
        symbol="BTC/USD", side=order_side, order_type=OrderType.MARKET, quantity=order_size
    )

    # Get fills
    fills = simulator.get_fills_for_order(order.order_id)

    # Calculate execution summary
    if fills:
        total_quantity = sum(fill.quantity for fill in fills)
        total_cost = sum(fill.price * fill.quantity for fill in fills)
        total_fees = sum(fill.fee for fill in fills)
        avg_price = total_cost / total_quantity if total_quantity > 0 else 0

        return {
            "order_id": order.order_id,
            "status": order.status.value,
            "requested_quantity": order.quantity,
            "filled_quantity": total_quantity,
            "avg_fill_price": avg_price,
            "total_fees": total_fees,
            "slippage": order.slippage,
            "latency_ms": order.latency_ms,
            "fills": [
                {
                    "price": fill.price,
                    "quantity": fill.quantity,
                    "fee": fill.fee,
                    "timestamp": fill.timestamp.isoformat(),
                }
                for fill in fills
            ],
        }
    else:
        return {
            "order_id": order.order_id,
            "status": order.status.value,
            "error": "No fills executed",
        }
