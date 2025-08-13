#!/usr/bin/env python3
"""
Order Book Imbalance & Spoof Detection System
Better timing & detection of fake orders through L2 depth analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import asyncio
import aiohttp
import logging
from dataclasses import dataclass
import ccxt.async_support as ccxt
from collections import deque
import warnings
warnings.filterwarnings('ignore')

@dataclass
class OrderBookSnapshot:
    """L2 order book snapshot"""
    timestamp: datetime
    symbol: str
    bids: List[Tuple[float, float]]  # (price, size)
    asks: List[Tuple[float, float]]  # (price, size)
    mid_price: float
    spread_bps: float

@dataclass
class ImbalanceSignal:
    """Order book imbalance signal"""
    symbol: str
    timestamp: datetime
    signal_type: str  # 'bid_imbalance', 'ask_imbalance', 'spoof_detected', 'liquidity_gap'
    imbalance_ratio: float  # -1 to 1, negative = ask heavy, positive = bid heavy
    depth_weighted_imbalance: float
    volume_weighted_imbalance: float
    spoof_score: float  # 0-1, probability of spoofing
    liquidity_score: float  # 0-1, overall liquidity quality
    signal_strength: float
    confidence: float
    expected_move_bps: float
    time_horizon_minutes: int

class OrderBookAnalyzer:
    """Analyzes order book for imbalances and spoofing"""

    def __init__(self, lookback_snapshots: int = 20):
        self.lookback_snapshots = lookback_snapshots
        self.orderbook_history = {}  # symbol -> deque of snapshots
        self.logger = logging.getLogger(__name__)

        # Analysis parameters
        self.params = {
            'depth_levels': 10,          # Analyze top 10 levels
            'min_imbalance_threshold': 0.3,  # 30% minimum imbalance
            'spoof_detection_threshold': 0.7,  # 70% spoof probability
            'large_order_multiplier': 3.0,    # 3x average for large order
            'cancel_rate_threshold': 0.8,     # 80% cancel rate for spoof
            'min_liquidity_score': 0.4        # 40% minimum liquidity
        }

    def add_orderbook_snapshot(self, snapshot: OrderBookSnapshot):
        """Add new order book snapshot to history"""

        symbol = snapshot.symbol

        if symbol not in self.orderbook_history:
            self.orderbook_history[symbol] = deque(maxlen=self.lookback_snapshots)

        self.orderbook_history[symbol].append(snapshot)

    def calculate_basic_imbalance(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Calculate basic bid/ask imbalance metrics"""

        bids = snapshot.bids[:self.params['depth_levels']]
        asks = snapshot.asks[:self.params['depth_levels']]

        if not bids or not asks:
            return {'imbalance_ratio': 0, 'bid_volume': 0, 'ask_volume': 0}

        # Volume-based imbalance
        bid_volume = sum(size for price, size in bids)
        ask_volume = sum(size for price, size in asks)
        total_volume = bid_volume + ask_volume

        if total_volume == 0:
            return {'imbalance_ratio': 0, 'bid_volume': 0, 'ask_volume': 0}

        # Imbalance ratio: +1 = all bids, -1 = all asks, 0 = balanced
        imbalance_ratio = (bid_volume - ask_volume) / total_volume

        return {
            'imbalance_ratio': imbalance_ratio,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'total_volume': total_volume
        }

    def calculate_depth_weighted_imbalance(self, snapshot: OrderBookSnapshot) -> float:
        """Calculate imbalance weighted by distance from mid price"""

        bids = snapshot.bids[:self.params['depth_levels']]
        asks = snapshot.asks[:self.params['depth_levels']]
        mid_price = snapshot.mid_price

        if not bids or not asks or mid_price == 0:
            return 0.0

        # Weight by inverse distance from mid (closer orders have more weight)
        weighted_bid_volume = 0
        weighted_ask_volume = 0

        for price, size in bids:
            distance = abs(price - mid_price) / mid_price
            weight = 1 / (1 + distance * 10)  # Exponential decay with distance
            weighted_bid_volume += size * weight

        for price, size in asks:
            distance = abs(price - mid_price) / mid_price
            weight = 1 / (1 + distance * 10)
            weighted_ask_volume += size * weight

        total_weighted = weighted_bid_volume + weighted_ask_volume

        if total_weighted == 0:
            return 0.0

        return (weighted_bid_volume - weighted_ask_volume) / total_weighted

    def detect_large_orders(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """Detect unusually large orders that might be spoofs"""

        bids = snapshot.bids[:self.params['depth_levels']]
        asks = snapshot.asks[:self.params['depth_levels']]

        if not bids or not asks:
            return {'large_bid_detected': False, 'large_ask_detected': False}

        # Calculate average order sizes
        bid_sizes = [size for price, size in bids]
        ask_sizes = [size for price, size in asks]

        avg_bid_size = np.mean(bid_sizes) if bid_sizes else 0
        avg_ask_size = np.mean(ask_sizes) if ask_sizes else 0

        # Detect large orders (significantly above average)
        large_bid_threshold = avg_bid_size * self.params['large_order_multiplier']
        large_ask_threshold = avg_ask_size * self.params['large_order_multiplier']

        large_bid_detected = any(size >= large_bid_threshold for size in bid_sizes)
        large_ask_detected = any(size >= large_ask_threshold for size in ask_sizes)

        # Find the largest orders
        max_bid_size = max(bid_sizes) if bid_sizes else 0
        max_ask_size = max(ask_sizes) if ask_sizes else 0

        return {
            'large_bid_detected': large_bid_detected,
            'large_ask_detected': large_ask_detected,
            'max_bid_size': max_bid_size,
            'max_ask_size': max_ask_size,
            'avg_bid_size': avg_bid_size,
            'avg_ask_size': avg_ask_size,
            'bid_size_ratio': max_bid_size / avg_bid_size if avg_bid_size > 0 else 1,
            'ask_size_ratio': max_ask_size / avg_ask_size if avg_ask_size > 0 else 1
        }

    def detect_spoofing_patterns(self, symbol: str) -> Dict[str, float]:
        """Detect spoofing patterns using historical snapshots"""

        if symbol not in self.orderbook_history:
            return {'spoof_score': 0, 'cancel_rate': 0, 'order_persistence': 1}

        snapshots = list(self.orderbook_history[symbol])

        if len(snapshots) < 5:
            return {'spoof_score': 0, 'cancel_rate': 0, 'order_persistence': 1}

        # Track large order persistence
        large_order_appearances = {}  # (price, size) -> count of appearances
        total_large_orders = 0

        for snapshot in snapshots:
            large_order_info = self.detect_large_orders(snapshot)

            # Track large bid orders
            if large_order_info['large_bid_detected']:
                for price, size in snapshot.bids[:3]:  # Top 3 levels
                    if size >= large_order_info['avg_bid_size'] * self.params['large_order_multiplier']:
                        key = (round(price, 2), round(size, 2))
                        large_order_appearances[key] = large_order_appearances.get(key, 0) + 1
                        total_large_orders += 1

            # Track large ask orders
            if large_order_info['large_ask_detected']:
                for price, size in snapshot.asks[:3]:
                    if size >= large_order_info['avg_ask_size'] * self.params['large_order_multiplier']:
                        key = (round(price, 2), round(size, 2))
                        large_order_appearances[key] = large_order_appearances.get(key, 0) + 1
                        total_large_orders += 1

        if total_large_orders == 0:
            return {'spoof_score': 0, 'cancel_rate': 0, 'order_persistence': 1}

        # Calculate cancel rate (low persistence = high cancel rate = potential spoof)
        persistent_orders = sum(1 for count in large_order_appearances.values() if count >= len(snapshots) * 0.3)
        cancel_rate = 1 - (persistent_orders / len(large_order_appearances)) if large_order_appearances else 0

        # Calculate average order persistence
        avg_persistence = np.mean(list(large_order_appearances.values())) / len(snapshots) if large_order_appearances else 1

        # Spoof score based on cancel rate and order size patterns
        spoof_score = min(1.0, cancel_rate * 1.2)  # High cancel rate = potential spoof

        return {
            'spoof_score': spoof_score,
            'cancel_rate': cancel_rate,
            'order_persistence': avg_persistence,
            'unique_large_orders': len(large_order_appearances),
            'total_large_order_observations': total_large_orders
        }

    def calculate_liquidity_metrics(self, snapshot: OrderBookSnapshot) -> Dict[str, float]:
        """Calculate liquidity quality metrics"""

        bids = snapshot.bids[:self.params['depth_levels']]
        asks = snapshot.asks[:self.params['depth_levels']]

        if not bids or not asks:
            return {'liquidity_score': 0, 'depth_score': 0, 'tightness_score': 0}

        # Tightness (spread quality)
        spread_score = max(0, 1 - snapshot.spread_bps / 100)  # Normalize by 100 bps

        # Depth (volume quality)
        total_volume = sum(size for price, size in bids + asks)
        depth_score = min(1.0, total_volume / 1000000)  # Normalize by 1M volume

        # Resilience (order distribution)
        bid_levels = len([1 for price, size in bids if size > 0])
        ask_levels = len([1 for price, size in asks if size > 0])
        distribution_score = min(1.0, (bid_levels + ask_levels) / (self.params['depth_levels'] * 2))

        # Combined liquidity score
        liquidity_score = (spread_score * 0.4 + depth_score * 0.4 + distribution_score * 0.2)

        return {
            'liquidity_score': liquidity_score,
            'depth_score': depth_score,
            'tightness_score': spread_score,
            'distribution_score': distribution_score,
            'total_depth_volume': total_volume,
            'active_levels': bid_levels + ask_levels
        }

    def generate_imbalance_signals(self, symbol: str) -> List[ImbalanceSignal]:
        """Generate trading signals from order book analysis"""

        if symbol not in self.orderbook_history or len(self.orderbook_history[symbol]) < 3:
            return []

        latest_snapshot = self.orderbook_history[symbol][-1]
        signals = []

        # Basic imbalance analysis
        basic_imbalance = self.calculate_basic_imbalance(latest_snapshot)
        depth_weighted_imbalance = self.calculate_depth_weighted_imbalance(latest_snapshot)

        # Large order detection
        large_orders = self.detect_large_orders(latest_snapshot)

        # Spoofing detection
        spoof_analysis = self.detect_spoofing_patterns(symbol)

        # Liquidity analysis
        liquidity_metrics = self.calculate_liquidity_metrics(latest_snapshot)

        # Generate signals based on analysis

        # 1. Imbalance signals
        imbalance_strength = abs(basic_imbalance['imbalance_ratio'])

        if imbalance_strength >= self.params['min_imbalance_threshold']:
            signal_type = 'bid_imbalance' if basic_imbalance['imbalance_ratio'] > 0 else 'ask_imbalance'

            # Estimate expected price move
            expected_move_bps = min(50, imbalance_strength * 100)  # Cap at 50 bps

            # Adjust for liquidity quality
            liquidity_adjustment = liquidity_metrics['liquidity_score']
            confidence = min(0.9, imbalance_strength * liquidity_adjustment + 0.2)

            signal = ImbalanceSignal(
                symbol=symbol,
                timestamp=latest_snapshot.timestamp,
                signal_type=signal_type,
                imbalance_ratio=basic_imbalance['imbalance_ratio'],
                depth_weighted_imbalance=depth_weighted_imbalance,
                volume_weighted_imbalance=basic_imbalance['imbalance_ratio'],
                spoof_score=spoof_analysis['spoof_score'],
                liquidity_score=liquidity_metrics['liquidity_score'],
                signal_strength=imbalance_strength,
                confidence=confidence,
                expected_move_bps=expected_move_bps,
                time_horizon_minutes=5
            )

            signals.append(signal)

        # 2. Spoof detection signals
        if spoof_analysis['spoof_score'] >= self.params['spoof_detection_threshold']:

            # High spoof probability -> fade the large orders
            signal_direction = 'ask_imbalance' if large_orders['large_bid_detected'] else 'bid_imbalance'

            signal = ImbalanceSignal(
                symbol=symbol,
                timestamp=latest_snapshot.timestamp,
                signal_type='spoof_detected',
                imbalance_ratio=-basic_imbalance['imbalance_ratio'] * 0.5,  # Fade the imbalance
                depth_weighted_imbalance=depth_weighted_imbalance,
                volume_weighted_imbalance=basic_imbalance['imbalance_ratio'],
                spoof_score=spoof_analysis['spoof_score'],
                liquidity_score=liquidity_metrics['liquidity_score'],
                signal_strength=spoof_analysis['spoof_score'],
                confidence=spoof_analysis['spoof_score'],
                expected_move_bps=20,  # Conservative move expectation
                time_horizon_minutes=10
            )

            signals.append(signal)

        # 3. Liquidity gap signals
        if liquidity_metrics['liquidity_score'] < self.params['min_liquidity_score']:

            signal = ImbalanceSignal(
                symbol=symbol,
                timestamp=latest_snapshot.timestamp,
                signal_type='liquidity_gap',
                imbalance_ratio=basic_imbalance['imbalance_ratio'],
                depth_weighted_imbalance=depth_weighted_imbalance,
                volume_weighted_imbalance=basic_imbalance['imbalance_ratio'],
                spoof_score=spoof_analysis['spoof_score'],
                liquidity_score=liquidity_metrics['liquidity_score'],
                signal_strength=1 - liquidity_metrics['liquidity_score'],
                confidence=0.6,  # Medium confidence for liquidity gaps
                expected_move_bps=abs(basic_imbalance['imbalance_ratio']) * 30,
                time_horizon_minutes=15
            )

            signals.append(signal)

        return signals

class OrderBookDataCollector:
    """Collects real-time order book data"""

    def __init__(self, max_depth: int = 20):
        self.max_depth = max_depth
        self.exchanges = {}
        self.logger = logging.getLogger(__name__)
        self._initialize_exchanges()

    def _initialize_exchanges(self):
        """Initialize exchanges for order book data"""
        try:
            # Binance
            self.exchanges['binance'] = ccxt.binance({
                'enableRateLimit': True,
                'options': {'fetchOrderBookMaxRetries': 3}
            })

            # Kraken
            self.exchanges['kraken'] = ccxt.kraken({
                'enableRateLimit': True
            })

        except Exception as e:
            self.logger.warning(f"Exchange initialization warning: {e}")

    async def fetch_orderbook_snapshot(self, symbol: str, exchange_name: str = 'binance') -> Optional[OrderBookSnapshot]:
        """Fetch single order book snapshot"""

        if exchange_name not in self.exchanges:
            return None

        exchange = self.exchanges[exchange_name]

        try:
            await exchange.load_markets()
            trading_symbol = f"{symbol}/USDT"

            if trading_symbol not in exchange.markets:
                return None

            # Fetch order book
            orderbook = await exchange.fetch_order_book(trading_symbol, limit=self.max_depth)

            bids = [(float(price), float(size)) for price, size in orderbook['bids'][:self.max_depth]]
            asks = [(float(price), float(size)) for price, size in orderbook['asks'][:self.max_depth]]

            if not bids or not asks:
                return None

            # Calculate mid price and spread
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid_price = (best_bid + best_ask) / 2
            spread_bps = ((best_ask - best_bid) / mid_price) * 10000

            snapshot = OrderBookSnapshot(
                timestamp=datetime.utcnow(),
                symbol=symbol,
                bids=bids,
                asks=asks,
                mid_price=mid_price,
                spread_bps=spread_bps
            )

            return snapshot

        except Exception as e:
            self.logger.error(f"Error fetching orderbook for {symbol} from {exchange_name}: {e}")
            return None

    async def close_connections(self):
        """Close exchange connections"""
        for exchange in self.exchanges.values():
            await exchange.close()

class OrderBookImbalanceSystem:
    """Complete order book imbalance and spoof detection system"""

    def __init__(self, update_interval_seconds: int = 5):
        self.data_collector = OrderBookDataCollector()
        self.analyzer = OrderBookAnalyzer()
        self.update_interval = update_interval_seconds
        self.logger = logging.getLogger(__name__)

        # System state
        self.active_symbols = set()
        self.latest_signals = {}
        self.running = False

    async def start_monitoring(self, symbols: List[str]):
        """Start monitoring order books for imbalance signals"""

        self.active_symbols = set(symbols)
        self.running = True

        self.logger.info(f"Starting order book monitoring for {len(symbols)} symbols")

        while self.running:
            try:
                # Collect snapshots for all symbols
                tasks = []
                for symbol in self.active_symbols:
                    task = self.data_collector.fetch_orderbook_snapshot(symbol)
                    tasks.append(task)

                snapshots = await asyncio.gather(*tasks, return_exceptions=True)

                # Process snapshots and generate signals
                for i, snapshot in enumerate(snapshots):
                    if isinstance(snapshot, OrderBookSnapshot):
                        symbol = list(self.active_symbols)[i]

                        # Add to analyzer
                        self.analyzer.add_orderbook_snapshot(snapshot)

                        # Generate signals
                        signals = self.analyzer.generate_imbalance_signals(symbol)

                        if signals:
                            self.latest_signals[symbol] = signals
                            self.logger.info(f"Generated {len(signals)} signals for {symbol}")

                # Wait before next update
                await asyncio.sleep(self.update_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.update_interval)

    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False

    def get_latest_signals(self, symbol: str = None) -> Dict[str, List[ImbalanceSignal]]:
        """Get latest imbalance signals"""

        if symbol:
            return {symbol: self.latest_signals.get(symbol, [])}
        else:
            return self.latest_signals.copy()

    def create_imbalance_features(self, signals: Dict[str, List[ImbalanceSignal]]) -> pd.DataFrame:
        """Create ML features from imbalance signals"""

        features_data = []

        for symbol, symbol_signals in signals.items():
            if not symbol_signals:
                continue

            # Use latest signal per symbol
            latest_signal = symbol_signals[-1]

            features = {
                'symbol': symbol,
                'orderbook_imbalance_ratio': latest_signal.imbalance_ratio,
                'depth_weighted_imbalance': latest_signal.depth_weighted_imbalance,
                'spoof_score': latest_signal.spoof_score,
                'liquidity_score': latest_signal.liquidity_score,
                'imbalance_signal_strength': latest_signal.signal_strength,
                'imbalance_confidence': latest_signal.confidence,
                'expected_move_bps': latest_signal.expected_move_bps,
                'is_bid_imbalance': latest_signal.signal_type == 'bid_imbalance',
                'is_ask_imbalance': latest_signal.signal_type == 'ask_imbalance',
                'is_spoof_detected': latest_signal.signal_type == 'spoof_detected',
                'is_liquidity_gap': latest_signal.signal_type == 'liquidity_gap',
                'time_horizon_minutes': latest_signal.time_horizon_minutes
            }

            features_data.append(features)

        return pd.DataFrame(features_data)

    async def cleanup(self):
        """Cleanup resources"""
        self.stop_monitoring()
        await self.data_collector.close_connections()

async def main():
    """Test order book imbalance system"""

    system = OrderBookImbalanceSystem(update_interval_seconds=10)

    test_symbols = ['BTC', 'ETH', 'ADA']

    try:
        # Start monitoring for 30 seconds
        monitor_task = asyncio.create_task(system.start_monitoring(test_symbols))

        await asyncio.sleep(30)

        # Get results
        signals = system.get_latest_signals()

        print(f"Collected signals for {len(signals)} symbols")
        for symbol, symbol_signals in signals.items():
            print(f"{symbol}: {len(symbol_signals)} signals")
            for signal in symbol_signals:
                print(f"  {signal.signal_type}: strength={signal.signal_strength:.3f}, "
                      f"confidence={signal.confidence:.3f}")

        # Create features
        features_df = system.create_imbalance_features(signals)
        print(f"Generated features DataFrame with {len(features_df)} rows")

    finally:
        await system.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
