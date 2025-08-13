"""
Order Book Data Integration and Liquidity Analysis
Addresses: order book depth, spoofing detection, liquidity gaps
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import json
import time

from utils.daily_logger import get_daily_logger


@dataclass
class OrderBookLevel:
    """Single order book level"""

    price: float
    size: float
    side: str  # 'bid' or 'ask'
    exchange: str
    timestamp: datetime


@dataclass
class OrderBookSnapshot:
    """Complete order book snapshot"""

    symbol: str
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    spread: float
    mid_price: float
    timestamp: datetime
    exchange: str


@dataclass
class LiquidityMetrics:
    """Comprehensive liquidity analysis"""

    symbol: str
    bid_ask_spread: float
    spread_percentage: float
    market_depth_1000: float  # Depth for $1000 order
    market_depth_10000: float  # Depth for $10000 order
    order_book_imbalance: float  # -1 to 1
    liquidity_score: float  # 0 to 1
    spoofing_risk: float  # 0 to 1
    large_wall_detected: bool
    wall_price: Optional[float]
    wall_size: float
    timestamp: datetime


class SpoofingDetector:
    """Detect order book manipulation and spoofing"""

    def __init__(self):
        self.logger = get_daily_logger().get_logger("api_calls")
        self.order_history = {}
        self.detection_window = 60  # 1 minute

    def detect_spoofing(
        self, current_snapshot: OrderBookSnapshot, historical_snapshots: List[OrderBookSnapshot]
    ) -> float:
        """Detect spoofing patterns in order book"""

        spoofing_score = 0.0
        checks_performed = 0

        # Check for large orders that appear and disappear quickly
        spoofing_score += self._detect_phantom_walls(historical_snapshots)
        checks_performed += 1

        # Check for coordinated bidding/asking patterns
        spoofing_score += self._detect_coordinated_orders(current_snapshot, historical_snapshots)
        checks_performed += 1

        # Check for unusual order sizes
        spoofing_score += self._detect_unusual_orders(current_snapshot)
        checks_performed += 1

        # Check for rapid order placement/cancellation
        spoofing_score += self._detect_rapid_modifications(historical_snapshots)
        checks_performed += 1

        return min(spoofing_score / checks_performed, 1.0) if checks_performed > 0 else 0.0

    def _detect_phantom_walls(self, snapshots: List[OrderBookSnapshot]) -> float:
        """Detect large orders that appear and disappear without execution"""

        if len(snapshots) < 3:
            return 0.0

        phantom_score = 0.0

        # Look for large orders in recent snapshots
        for i in range(len(snapshots) - 2):
            current = snapshots[i]
            next_snapshot = snapshots[i + 1]

            # Check bids
            for bid in current.bids[:5]:  # Top 5 levels
                if bid.size > 10000:  # Large order
                    # Check if order disappeared in next snapshot
                    found = any(
                        abs(next_bid.price - bid.price) < 0.001 and next_bid.size > bid.size * 0.5
                        for next_bid in next_snapshot.bids[:5]
                    )

                    if not found:
                        phantom_score += 0.2  # Phantom wall detected

            # Check asks
            for ask in current.asks[:5]:
                if ask.size > 10000:  # Large order
                    found = any(
                        abs(next_ask.price - ask.price) < 0.001 and next_ask.size > ask.size * 0.5
                        for next_ask in next_snapshot.asks[:5]
                    )

                    if not found:
                        phantom_score += 0.2

        return min(phantom_score, 1.0)

    def _detect_coordinated_orders(
        self, current: OrderBookSnapshot, historical: List[OrderBookSnapshot]
    ) -> float:
        """Detect coordinated bidding/asking patterns"""

        if len(historical) < 5:
            return 0.0

        # Look for unusual patterns in order placement
        price_levels = {}

        for snapshot in historical[-5:]:  # Last 5 snapshots
            for bid in snapshot.bids[:10]:
                price_key = round(bid.price, 2)
                if price_key not in price_levels:
                    price_levels[price_key] = []
                price_levels[price_key].append(bid.size)

        # Check for repeated exact sizes (indication of bots)
        coordination_score = 0.0
        for price, sizes in price_levels.items():
            if len(sizes) >= 3:
                # Check if sizes are suspiciously similar
                size_std = np.std(sizes)
                size_mean = np.mean(sizes)
                if size_mean > 0 and size_std / size_mean < 0.1:  # Very low variation
                    coordination_score += 0.3

        return min(coordination_score, 1.0)

    def _detect_unusual_orders(self, snapshot: OrderBookSnapshot) -> float:
        """Detect unusually large orders that might be manipulation"""

        all_sizes = [level.size for level in snapshot.bids + snapshot.asks]

        if not all_sizes:
            return 0.0

        median_size = np.median(all_sizes)
        unusual_score = 0.0

        for size in all_sizes:
            if size > median_size * 50:  # 50x larger than median
                unusual_score += 0.2

        return min(unusual_score, 1.0)

    def _detect_rapid_modifications(self, snapshots: List[OrderBookSnapshot]) -> float:
        """Detect rapid order modifications"""

        if len(snapshots) < 3:
            return 0.0

        # Count significant changes between consecutive snapshots
        rapid_changes = 0

        for i in range(len(snapshots) - 1):
            current = snapshots[i]
            next_snapshot = snapshots[i + 1]

            # Time difference
            time_diff = (next_snapshot.timestamp - current.timestamp).total_seconds()

            if time_diff < 5:  # Less than 5 seconds
                # Count order book changes
                changes = 0

                # Compare top 5 levels
                for j in range(min(5, len(current.bids), len(next_snapshot.bids))):
                    if abs(current.bids[j].price - next_snapshot.bids[j].price) > 0.001:
                        changes += 1

                if changes >= 3:  # Many changes in short time
                    rapid_changes += 1

        return min(rapid_changes / max(len(snapshots) - 1, 1), 1.0)


class LiquidityAnalyzer:
    """Comprehensive liquidity analysis"""

    def __init__(self):
        self.logger = get_daily_logger().get_logger("api_calls")

    def analyze_liquidity(
        self, order_book: OrderBookSnapshot, volume_24h: float = 0
    ) -> LiquidityMetrics:
        """Comprehensive liquidity analysis"""

        # Basic spread metrics
        spread = order_book.spread
        spread_pct = (spread / order_book.mid_price) * 100 if order_book.mid_price > 0 else 0

        # Market depth analysis
        depth_1000 = self._calculate_market_depth(order_book, 1000)
        depth_10000 = self._calculate_market_depth(order_book, 10000)

        # Order book imbalance
        imbalance = self._calculate_order_book_imbalance(order_book)

        # Detect large walls
        wall_detected, wall_price, wall_size = self._detect_large_walls(order_book)

        # Calculate overall liquidity score
        liquidity_score = self._calculate_liquidity_score(
            spread_pct, depth_1000, depth_10000, imbalance, volume_24h
        )

        return LiquidityMetrics(
            symbol=order_book.symbol,
            bid_ask_spread=spread,
            spread_percentage=spread_pct,
            market_depth_1000=depth_1000,
            market_depth_10000=depth_10000,
            order_book_imbalance=imbalance,
            liquidity_score=liquidity_score,
            spoofing_risk=0.0,  # Will be set by spoofing detector
            large_wall_detected=wall_detected,
            wall_price=wall_price,
            wall_size=wall_size,
            timestamp=order_book.timestamp,
        )

    def _calculate_market_depth(self, order_book: OrderBookSnapshot, target_amount: float) -> float:
        """Calculate market depth for given order size"""

        # Calculate bid side depth
        bid_depth = 0
        remaining_amount = target_amount / 2  # Half for each side

        for bid in order_book.bids:
            order_value = bid.price * bid.size
            if remaining_amount <= order_value:
                # This level can fulfill remaining amount
                required_shares = remaining_amount / bid.price
                price_impact = 0  # No additional impact
                bid_depth = required_shares / bid.size if bid.size > 0 else 0
                break
            else:
                remaining_amount -= order_value

        # Calculate ask side depth
        ask_depth = 0
        remaining_amount = target_amount / 2

        for ask in order_book.asks:
            order_value = ask.price * ask.size
            if remaining_amount <= order_value:
                required_shares = remaining_amount / ask.price
                ask_depth = required_shares / ask.size if ask.size > 0 else 0
                break
            else:
                remaining_amount -= order_value

        # Return average depth (0 to 1, where 1 = can fulfill order at best price)
        return (bid_depth + ask_depth) / 2

    def _calculate_order_book_imbalance(self, order_book: OrderBookSnapshot) -> float:
        """Calculate order book imbalance (-1 to 1)"""

        # Calculate total bid and ask volumes (top 10 levels)
        bid_volume = sum(bid.size for bid in order_book.bids[:10])
        ask_volume = sum(ask.size for ask in order_book.asks[:10])

        total_volume = bid_volume + ask_volume

        if total_volume == 0:
            return 0.0

        # Imbalance: positive = more bids, negative = more asks
        imbalance = (bid_volume - ask_volume) / total_volume

        return imbalance

    def _detect_large_walls(
        self, order_book: OrderBookSnapshot
    ) -> Tuple[bool, Optional[float], float]:
        """Detect large walls in order book"""

        all_levels = order_book.bids + order_book.asks

        if not all_levels:
            return False, None, 0.0

        # Calculate average size
        sizes = [level.size for level in all_levels]
        median_size = np.median(sizes)

        # Look for orders significantly larger than median
        wall_threshold = median_size * 10  # 10x median

        for level in all_levels:
            if level.size > wall_threshold:
                return True, level.price, level.size

        return False, None, 0.0

    def _calculate_liquidity_score(
        self,
        spread_pct: float,
        depth_1000: float,
        depth_10000: float,
        imbalance: float,
        volume_24h: float,
    ) -> float:
        """Calculate overall liquidity score (0 to 1)"""

        score = 0.0
        weight_sum = 0.0

        # Spread component (lower spread = higher score)
        if spread_pct <= 0.1:  # <= 0.1%
            spread_score = 1.0
        elif spread_pct <= 0.5:  # <= 0.5%
            spread_score = 0.8
        elif spread_pct <= 1.0:  # <= 1.0%
            spread_score = 0.6
        else:
            spread_score = 0.4

        score += spread_score * 0.3
        weight_sum += 0.3

        # Depth component
        depth_score = (depth_1000 + depth_10000) / 2
        score += depth_score * 0.3
        weight_sum += 0.3

        # Imbalance component (balanced book = higher score)
        imbalance_score = 1.0 - abs(imbalance)
        score += imbalance_score * 0.2
        weight_sum += 0.2

        # Volume component
        if volume_24h > 1000000:  # > $1M daily volume
            volume_score = 1.0
        elif volume_24h > 100000:  # > $100K
            volume_score = 0.8
        else:
            volume_score = 0.5

        score += volume_score * 0.2
        weight_sum += 0.2

        return score / weight_sum if weight_sum > 0 else 0.5


class OrderBookDataProvider:
    """Provides real-time order book data from multiple exchanges"""

    def __init__(self):
        self.logger = get_daily_logger().get_logger("api_calls")
        self.session = None
        self.supported_exchanges = ["kraken", "coinbase"]
        self.cache = {}
        self.cache_ttl = 5  # 5 seconds

    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()

    async def get_order_book(
        self, symbol: str, exchange: str = "kraken", depth: int = 20
    ) -> Optional[OrderBookSnapshot]:
        """Get order book snapshot"""

        # Check cache
        cache_key = f"{exchange}_{symbol}_{depth}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_data

        try:
            if exchange == "kraken":
                snapshot = await self._get_kraken_order_book(symbol, depth)
            elif exchange == "coinbase":
                snapshot = await self._get_coinbase_order_book(symbol, depth)
            else:
                self.logger.warning(f"Unsupported exchange: {exchange}")
                return None

            # Cache result
            if snapshot:
                self.cache[cache_key] = (snapshot, time.time())

            return snapshot

        except Exception as e:
            self.logger.error(f"Error getting order book for {symbol} on {exchange}: {e}")
            return None

    async def _get_kraken_order_book(self, symbol: str, depth: int) -> Optional[OrderBookSnapshot]:
        """Get order book from Kraken"""

        # Convert symbol format for Kraken
        kraken_symbol = symbol.replace("/", "")

        try:
            url = f"https://api.kraken.com/0/public/Depth"
            params = {"pair": kraken_symbol, "count": depth}

            if not self.session:
                return None

            async with self.session.get(url, params=params) as response:
                if response.status != 200:
                    return None

                data = await response.json()

                if "error" in data and data["error"]:
                    self.logger.error(f"Kraken API error: {data['error']}")
                    return None

                if "result" not in data:
                    return None

                # Parse order book data
                pair_data = list(data["result"].values())[0]

                bids = []
                for bid_data in pair_data.get("bids", []):
                    bids.append(
                        OrderBookLevel(
                            price=float(bid_data[0]),
                            size=float(bid_data[1]),
                            side="bid",
                            exchange="kraken",
                            timestamp=datetime.now(),
                        )
                    )

                asks = []
                for ask_data in pair_data.get("asks", []):
                    asks.append(
                        OrderBookLevel(
                            price=float(ask_data[0]),
                            size=float(ask_data[1]),
                            side="ask",
                            exchange="kraken",
                            timestamp=datetime.now(),
                        )
                    )

                # Calculate spread and mid price
                if bids and asks:
                    best_bid = max(bids, key=lambda x: x.price)
                    best_ask = min(asks, key=lambda x: x.price)
                    spread = best_ask.price - best_bid.price
                    mid_price = (best_bid.price + best_ask.price) / 2
                else:
                    spread = 0
                    mid_price = 0

                return OrderBookSnapshot(
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    spread=spread,
                    mid_price=mid_price,
                    timestamp=datetime.now(),
                    exchange="kraken",
                )

        except Exception as e:
            self.logger.error(f"Error fetching Kraken order book: {e}")
            return None

    async def _get_coinbase_order_book(
        self, symbol: str, depth: int
    ) -> Optional[OrderBookSnapshot]:
        """Get order book from Coinbase"""

        # For now, return None (would implement actual Coinbase API)
        return None


class OrderBookAnalyzer:
    """Main order book analysis coordinator"""

    def __init__(self):
        self.logger = get_daily_logger().get_logger("api_calls")
        self.data_provider = OrderBookDataProvider()
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.spoofing_detector = SpoofingDetector()
        self.historical_snapshots = {}

    async def analyze_symbol_liquidity(
        self, symbol: str, exchanges: List[str] = None, volume_24h: float = 0
    ) -> Dict[str, LiquidityMetrics]:
        """Analyze liquidity across multiple exchanges"""

        if exchanges is None:
            exchanges = ["kraken"]

        results = {}

        async with self.data_provider:
            for exchange in exchanges:
                try:
                    # Get current order book
                    order_book = await self.data_provider.get_order_book(symbol, exchange)

                    if not order_book:
                        continue

                    # Get historical snapshots for spoofing detection
                    history_key = f"{exchange}_{symbol}"
                    historical = self.historical_snapshots.get(history_key, [])

                    # Analyze liquidity
                    liquidity_metrics = self.liquidity_analyzer.analyze_liquidity(
                        order_book, volume_24h
                    )

                    # Detect spoofing
                    spoofing_risk = self.spoofing_detector.detect_spoofing(order_book, historical)
                    liquidity_metrics.spoofing_risk = spoofing_risk

                    results[exchange] = liquidity_metrics

                    # Update historical snapshots
                    historical.append(order_book)
                    # Keep only recent snapshots (last 10)
                    self.historical_snapshots[history_key] = historical[-10:]

                    self.logger.info(f"Liquidity analysis complete for {symbol} on {exchange}")

                except Exception as e:
                    self.logger.error(f"Error analyzing {symbol} on {exchange}: {e}")
                    continue

        return results

    def get_status(self) -> Dict:
        """Get analyzer status"""
        return {
            "component": "order_book_analyzer",
            "status": "operational",
            "supported_exchanges": self.data_provider.supported_exchanges,
            "spoofing_detection": True,
            "liquidity_analysis": True,
            "cached_symbols": len(self.historical_snapshots),
        }


# Global instance
order_book_analyzer = OrderBookAnalyzer()
