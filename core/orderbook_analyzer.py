#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Advanced Order Book Analyzer
Real-time liquidity analysis, spoofing detection, and market depth calculation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
import asyncio
import aiohttp
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class LiquidityLevel(Enum):
    ULTRA_HIGH = "ultra_high"    # Deep, stable liquidity
    HIGH = "high"                # Good liquidity for most trades
    MEDIUM = "medium"            # Adequate for small-medium trades
    LOW = "low"                  # Limited liquidity, potential slippage
    VERY_LOW = "very_low"        # Poor liquidity, high slippage risk

class OrderBookPattern(Enum):
    NORMAL = "normal"
    SPOOFING = "spoofing"        # Large orders that disappear
    ICEBERG = "iceberg"          # Hidden large orders
    WALL = "wall"                # Large defensive orders
    SQUEEZE = "squeeze"          # Liquidity crunch
    ACCUMULATION = "accumulation" # Gradual building
    DISTRIBUTION = "distribution" # Gradual selling

@dataclass
class OrderBookSnapshot:
    """Order book snapshot data"""
    timestamp: datetime
    bids: List[Tuple[float, float]]  # (price, size) pairs
    asks: List[Tuple[float, float]]  # (price, size) pairs
    mid_price: float
    spread: float
    total_bid_volume: float
    total_ask_volume: float

@dataclass
class LiquidityMetrics:
    """Comprehensive liquidity metrics"""
    liquidity_level: LiquidityLevel
    bid_ask_spread: float
    spread_percentage: float
    market_depth: float
    price_impact_1k: float      # Price impact for $1K trade
    price_impact_10k: float     # Price impact for $10K trade
    price_impact_100k: float    # Price impact for $100K trade
    order_book_imbalance: float # Bid/ask volume imbalance
    liquidity_score: float      # Overall liquidity score (0-1)
    detected_patterns: List[OrderBookPattern]
    spoofing_probability: float
    timestamp: datetime

@dataclass
class OrderBookConfig:
    """Configuration for order book analysis"""
    max_depth_levels: int = 20
    min_update_interval: float = 0.1  # seconds
    spoofing_detection_window: int = 60  # seconds
    large_order_threshold: float = 10000  # USD value
    price_impact_thresholds: List[float] = field(default_factory=lambda: [1000, 10000, 100000])
    imbalance_threshold: float = 0.3
    liquidity_score_weights: Dict[str, float] = field(default_factory=lambda: {
        'spread': 0.3,
        'depth': 0.3,
        'stability': 0.2,
        'imbalance': 0.2
    })

class OrderBookAnalyzer:
    """Advanced order book analyzer with real-time liquidity assessment"""
    
    def __init__(self, config: Optional[OrderBookConfig] = None):
        self.config = config or OrderBookConfig()
        self.logger = logging.getLogger(f"{__name__}.OrderBookAnalyzer")
        
        # Order book history for pattern detection
        self.order_book_history: deque = deque(maxlen=1000)
        self.large_order_history: deque = deque(maxlen=500)
        
        # Real-time metrics
        self.current_metrics: Optional[LiquidityMetrics] = None
        self.metrics_history: deque = deque(maxlen=1000)
        
        # Pattern detection state
        self.suspicious_orders: Dict[str, Dict] = {}  # Track potential spoofing
        self.order_lifetime_tracking: Dict[str, Dict] = {}
        
        # Performance tracking
        self.analysis_stats = {
            'total_snapshots_processed': 0,
            'spoofing_events_detected': 0,
            'last_update_time': None,
            'average_processing_time': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info("Order Book Analyzer initialized with advanced pattern detection")
    
    def analyze_order_book(self, order_book_data: Dict[str, Any]) -> LiquidityMetrics:
        """
        Analyze order book data and return comprehensive liquidity metrics
        
        Args:
            order_book_data: Dictionary containing bids, asks, and metadata
            
        Returns:
            LiquidityMetrics with comprehensive analysis
        """
        with self._lock:
            start_time = datetime.now()
            
            try:
                # Parse order book data
                snapshot = self._parse_order_book_data(order_book_data)
                
                # Store in history
                self.order_book_history.append(snapshot)
                
                # Calculate liquidity metrics
                metrics = self._calculate_liquidity_metrics(snapshot)
                
                # Detect patterns and anomalies
                patterns = self._detect_order_book_patterns(snapshot)
                metrics.detected_patterns = patterns
                
                # Spoofing detection
                spoofing_prob = self._detect_spoofing(snapshot)
                metrics.spoofing_probability = spoofing_prob
                
                # Update current metrics
                self.current_metrics = metrics
                self.metrics_history.append(metrics)
                
                # Update performance stats
                processing_time = (datetime.now() - start_time).total_seconds()
                self._update_performance_stats(processing_time)
                
                return metrics
                
            except Exception as e:
                self.logger.error(f"Order book analysis failed: {e}")
                return self._get_default_metrics()
    
    def _parse_order_book_data(self, data: Dict[str, Any]) -> OrderBookSnapshot:
        """Parse order book data into standardized format"""
        try:
            # Extract bids and asks
            bids = data.get('bids', [])
            asks = data.get('asks', [])
            
            # Convert to list of tuples if needed
            if isinstance(bids, dict):
                bids = [(float(price), float(size)) for price, size in bids.items()]
            elif isinstance(bids, list) and bids and isinstance(bids[0], list):
                bids = [(float(item[0]), float(item[1])) for item in bids]
            
            if isinstance(asks, dict):
                asks = [(float(price), float(size)) for price, size in asks.items()]
            elif isinstance(asks, list) and asks and isinstance(asks[0], list):
                asks = [(float(item[0]), float(item[1])) for item in asks]
            
            # Sort bids (highest first) and asks (lowest first)
            bids = sorted(bids, key=lambda x: x[0], reverse=True)
            asks = sorted(asks, key=lambda x: x[0])
            
            # Limit depth
            bids = bids[:self.config.max_depth_levels]
            asks = asks[:self.config.max_depth_levels]
            
            # Calculate basic metrics
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            mid_price = (best_bid + best_ask) / 2 if best_bid > 0 and best_ask > 0 else 0
            spread = best_ask - best_bid if best_bid > 0 and best_ask > 0 else 0
            
            total_bid_volume = sum(size for _, size in bids)
            total_ask_volume = sum(size for _, size in asks)
            
            return OrderBookSnapshot(
                timestamp=datetime.now(),
                bids=bids,
                asks=asks,
                mid_price=mid_price,
                spread=spread,
                total_bid_volume=total_bid_volume,
                total_ask_volume=total_ask_volume
            )
            
        except Exception as e:
            self.logger.error(f"Failed to parse order book data: {e}")
            raise
    
    def _calculate_liquidity_metrics(self, snapshot: OrderBookSnapshot) -> LiquidityMetrics:
        """Calculate comprehensive liquidity metrics"""
        try:
            # Basic spread metrics
            spread_percentage = (snapshot.spread / snapshot.mid_price * 100) if snapshot.mid_price > 0 else 100
            
            # Market depth calculation
            market_depth = self._calculate_market_depth(snapshot)
            
            # Price impact analysis
            price_impact_1k = self._calculate_price_impact(snapshot, 1000)
            price_impact_10k = self._calculate_price_impact(snapshot, 10000)
            price_impact_100k = self._calculate_price_impact(snapshot, 100000)
            
            # Order book imbalance
            total_volume = snapshot.total_bid_volume + snapshot.total_ask_volume
            imbalance = ((snapshot.total_bid_volume - snapshot.total_ask_volume) / 
                        max(total_volume, 1e-8)) if total_volume > 0 else 0
            
            # Overall liquidity score
            liquidity_score = self._calculate_liquidity_score(
                spread_percentage, market_depth, price_impact_10k, abs(imbalance)
            )
            
            # Liquidity level classification
            liquidity_level = self._classify_liquidity_level(liquidity_score, spread_percentage)
            
            return LiquidityMetrics(
                liquidity_level=liquidity_level,
                bid_ask_spread=snapshot.spread,
                spread_percentage=spread_percentage,
                market_depth=market_depth,
                price_impact_1k=price_impact_1k,
                price_impact_10k=price_impact_10k,
                price_impact_100k=price_impact_100k,
                order_book_imbalance=imbalance,
                liquidity_score=liquidity_score,
                detected_patterns=[],
                spoofing_probability=0.0,
                timestamp=snapshot.timestamp
            )
            
        except Exception as e:
            self.logger.error(f"Liquidity metrics calculation failed: {e}")
            return self._get_default_metrics()
    
    def _calculate_market_depth(self, snapshot: OrderBookSnapshot) -> float:
        """Calculate market depth as total volume within reasonable price range"""
        if not snapshot.bids or not snapshot.asks:
            return 0.0
        
        try:
            # Define reasonable price range (±2% from mid price)
            price_range = 0.02
            lower_bound = snapshot.mid_price * (1 - price_range)
            upper_bound = snapshot.mid_price * (1 + price_range)
            
            # Sum volume within range
            depth_volume = 0.0
            
            # Bid side
            for price, volume in snapshot.bids:
                if price >= lower_bound:
                    depth_volume += volume * price  # USD value
                else:
                    break
            
            # Ask side
            for price, volume in snapshot.asks:
                if price <= upper_bound:
                    depth_volume += volume * price  # USD value
                else:
                    break
            
            return depth_volume
            
        except Exception:
            return 0.0
    
    def _calculate_price_impact(self, snapshot: OrderBookSnapshot, trade_size_usd: float) -> float:
        """Calculate price impact for a given trade size"""
        if not snapshot.bids or not snapshot.asks or snapshot.mid_price <= 0:
            return 1.0  # 100% impact if no liquidity
        
        try:
            # Calculate for market buy (consuming asks)
            remaining_size = trade_size_usd
            weighted_price = 0.0
            total_volume = 0.0
            
            for price, volume in snapshot.asks:
                volume_usd = volume * price
                if remaining_size <= volume_usd:
                    # Partial fill
                    needed_volume = remaining_size / price
                    weighted_price += price * needed_volume
                    total_volume += needed_volume
                    break
                else:
                    # Full level consumed
                    weighted_price += price * volume
                    total_volume += volume
                    remaining_size -= volume_usd
            
            if total_volume > 0:
                avg_fill_price = weighted_price / total_volume
                price_impact = abs(avg_fill_price - snapshot.mid_price) / snapshot.mid_price
                return min(price_impact, 1.0)  # Cap at 100%
            else:
                return 1.0  # No liquidity available
                
        except Exception:
            return 1.0
    
    def _calculate_liquidity_score(self, spread_pct: float, depth: float, price_impact: float, imbalance: float) -> float:
        """Calculate overall liquidity score (0-1, higher is better)"""
        try:
            weights = self.config.liquidity_score_weights
            
            # Spread component (lower spread is better)
            spread_score = max(0, 1 - spread_pct / 5.0)  # Normalize to 5% spread
            
            # Depth component (higher depth is better)
            depth_score = min(1, depth / 100000)  # Normalize to $100K depth
            
            # Price impact component (lower impact is better)
            impact_score = max(0, 1 - price_impact * 10)  # Normalize to 10% impact
            
            # Imbalance component (lower imbalance is better)
            imbalance_score = max(0, 1 - imbalance * 2)  # Normalize to 50% imbalance
            
            # Weighted combination
            liquidity_score = (
                weights['spread'] * spread_score +
                weights['depth'] * depth_score +
                weights['stability'] * impact_score +
                weights['imbalance'] * imbalance_score
            )
            
            return max(0.0, min(1.0, liquidity_score))
            
        except Exception:
            return 0.5
    
    def _classify_liquidity_level(self, score: float, spread_pct: float) -> LiquidityLevel:
        """Classify liquidity level based on score and spread"""
        if score >= 0.8 and spread_pct <= 0.1:
            return LiquidityLevel.ULTRA_HIGH
        elif score >= 0.6 and spread_pct <= 0.3:
            return LiquidityLevel.HIGH
        elif score >= 0.4 and spread_pct <= 0.6:
            return LiquidityLevel.MEDIUM
        elif score >= 0.2 and spread_pct <= 1.0:
            return LiquidityLevel.LOW
        else:
            return LiquidityLevel.VERY_LOW
    
    def _detect_order_book_patterns(self, snapshot: OrderBookSnapshot) -> List[OrderBookPattern]:
        """Detect patterns in order book structure"""
        patterns = []
        
        try:
            # Wall detection (large orders at key levels)
            if self._detect_walls(snapshot):
                patterns.append(OrderBookPattern.WALL)
            
            # Iceberg detection (unusual volume patterns)
            if self._detect_iceberg_orders(snapshot):
                patterns.append(OrderBookPattern.ICEBERG)
            
            # Squeeze detection (low liquidity)
            if self._detect_liquidity_squeeze(snapshot):
                patterns.append(OrderBookPattern.SQUEEZE)
            
            # Accumulation/Distribution patterns
            accumulation_dist = self._detect_accumulation_distribution(snapshot)
            if accumulation_dist:
                patterns.append(accumulation_dist)
            
            if not patterns:
                patterns.append(OrderBookPattern.NORMAL)
            
            return patterns
            
        except Exception as e:
            self.logger.warning(f"Pattern detection failed: {e}")
            return [OrderBookPattern.NORMAL]
    
    def _detect_walls(self, snapshot: OrderBookSnapshot) -> bool:
        """Detect large defensive orders (walls)"""
        try:
            if not snapshot.bids or not snapshot.asks:
                return False
            
            # Check for unusually large orders
            avg_bid_size = snapshot.total_bid_volume / len(snapshot.bids) if snapshot.bids else 0
            avg_ask_size = snapshot.total_ask_volume / len(snapshot.asks) if snapshot.asks else 0
            
            # Look for orders 5x larger than average
            threshold_multiplier = 5
            
            for price, size in snapshot.bids[:5]:  # Check top 5 levels
                if size > avg_bid_size * threshold_multiplier:
                    return True
            
            for price, size in snapshot.asks[:5]:
                if size > avg_ask_size * threshold_multiplier:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _detect_iceberg_orders(self, snapshot: OrderBookSnapshot) -> bool:
        """Detect potential iceberg orders (hidden large orders)"""
        try:
            # Iceberg orders show consistent refilling at same price levels
            # This requires historical comparison
            if len(self.order_book_history) < 10:
                return False
            
            recent_snapshots = list(self.order_book_history)[-10:]
            
            # Check for price levels that consistently show similar volumes
            consistent_levels = 0
            
            for price, size in snapshot.bids[:5]:
                similar_count = 0
                for hist_snapshot in recent_snapshots:
                    for hist_price, hist_size in hist_snapshot.bids[:5]:
                        if abs(hist_price - price) / price < 0.001:  # Same price level
                            if abs(hist_size - size) / max(size, hist_size) < 0.1:  # Similar size
                                similar_count += 1
                                break
                
                if similar_count >= 5:  # Consistent across multiple snapshots
                    consistent_levels += 1
            
            return consistent_levels >= 2
            
        except Exception:
            return False
    
    def _detect_liquidity_squeeze(self, snapshot: OrderBookSnapshot) -> bool:
        """Detect liquidity squeeze conditions"""
        try:
            # Low total volume and wide spread indicate squeeze
            total_volume_usd = (snapshot.total_bid_volume + snapshot.total_ask_volume) * snapshot.mid_price
            spread_percentage = (snapshot.spread / snapshot.mid_price * 100) if snapshot.mid_price > 0 else 100
            
            # Thresholds for squeeze detection
            low_volume_threshold = 50000  # $50K total volume
            wide_spread_threshold = 0.5   # 0.5% spread
            
            return total_volume_usd < low_volume_threshold and spread_percentage > wide_spread_threshold
            
        except Exception:
            return False
    
    def _detect_accumulation_distribution(self, snapshot: OrderBookSnapshot) -> Optional[OrderBookPattern]:
        """Detect accumulation or distribution patterns"""
        try:
            if len(self.order_book_history) < 20:
                return None
            
            # Analyze bid/ask volume trends over time
            recent_snapshots = list(self.order_book_history)[-20:]
            
            bid_volumes = [s.total_bid_volume for s in recent_snapshots]
            ask_volumes = [s.total_ask_volume for s in recent_snapshots]
            
            # Calculate trends
            bid_trend = np.polyfit(range(len(bid_volumes)), bid_volumes, 1)[0] if len(bid_volumes) > 1 else 0
            ask_trend = np.polyfit(range(len(ask_volumes)), ask_volumes, 1)[0] if len(ask_volumes) > 1 else 0
            
            # Normalize by average volume
            avg_bid_volume = np.mean(bid_volumes) if bid_volumes else 1
            avg_ask_volume = np.mean(ask_volumes) if ask_volumes else 1
            
            bid_trend_normalized = bid_trend / avg_bid_volume if avg_bid_volume > 0 else 0
            ask_trend_normalized = ask_trend / avg_ask_volume if avg_ask_volume > 0 else 0
            
            # Classification thresholds
            trend_threshold = 0.02  # 2% trend
            
            if bid_trend_normalized > trend_threshold and ask_trend_normalized < -trend_threshold:
                return OrderBookPattern.ACCUMULATION
            elif ask_trend_normalized > trend_threshold and bid_trend_normalized < -trend_threshold:
                return OrderBookPattern.DISTRIBUTION
            
            return None
            
        except Exception:
            return None
    
    def _detect_spoofing(self, snapshot: OrderBookSnapshot) -> float:
        """Detect potential spoofing activity"""
        try:
            if len(self.order_book_history) < 5:
                return 0.0
            
            spoofing_indicators = 0
            total_indicators = 4
            
            # 1. Large orders that appear and disappear quickly
            large_order_disappearance = self._check_large_order_disappearance()
            if large_order_disappearance:
                spoofing_indicators += 1
            
            # 2. Unusual order size relative to normal activity
            unusual_sizes = self._check_unusual_order_sizes(snapshot)
            if unusual_sizes:
                spoofing_indicators += 1
            
            # 3. Orders placed far from market price then pulled
            far_orders = self._check_far_market_orders()
            if far_orders:
                spoofing_indicators += 1
            
            # 4. Rapid order placement and cancellation patterns
            rapid_patterns = self._check_rapid_order_patterns()
            if rapid_patterns:
                spoofing_indicators += 1
            
            return spoofing_indicators / total_indicators
            
        except Exception:
            return 0.0
    
    def _check_large_order_disappearance(self) -> bool:
        """Check for large orders that appeared then disappeared"""
        try:
            if len(self.order_book_history) < 3:
                return False
            
            recent_snapshots = list(self.order_book_history)[-3:]
            
            # Look for large orders in previous snapshots that are missing now
            for prev_snapshot in recent_snapshots[:-1]:
                current_snapshot = recent_snapshots[-1]
                
                # Check bids
                for prev_price, prev_size in prev_snapshot.bids:
                    if prev_size * prev_price > self.config.large_order_threshold:
                        # Check if this large order still exists
                        found = any(abs(price - prev_price) / prev_price < 0.001 
                                  for price, _ in current_snapshot.bids)
                        if not found:
                            return True
            
            return False
            
        except Exception:
            return False
    
    def _check_unusual_order_sizes(self, snapshot: OrderBookSnapshot) -> bool:
        """Check for unusually large orders compared to historical average"""
        try:
            if len(self.order_book_history) < 10:
                return False
            
            # Calculate historical average order sizes
            recent_snapshots = list(self.order_book_history)[-10:]
            all_order_sizes = []
            
            for hist_snapshot in recent_snapshots:
                all_order_sizes.extend([size for _, size in hist_snapshot.bids])
                all_order_sizes.extend([size for _, size in hist_snapshot.asks])
            
            if not all_order_sizes:
                return False
            
            avg_size = np.mean(all_order_sizes)
            std_size = np.std(all_order_sizes)
            
            # Check current snapshot for outliers (>3 standard deviations)
            threshold = avg_size + 3 * std_size
            
            for _, size in snapshot.bids + snapshot.asks:
                if size > threshold:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _check_far_market_orders(self) -> bool:
        """Check for orders placed far from market then removed"""
        # This would require more sophisticated order tracking
        # For now, return False (placeholder)
        return False
    
    def _check_rapid_order_patterns(self) -> bool:
        """Check for rapid order placement and cancellation patterns"""
        # This would require order-level tracking over time
        # For now, return False (placeholder)
        return False
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics"""
        self.analysis_stats['total_snapshots_processed'] += 1
        self.analysis_stats['last_update_time'] = datetime.now()
        
        # Update moving average of processing time
        current_avg = self.analysis_stats['average_processing_time']
        count = self.analysis_stats['total_snapshots_processed']
        new_avg = (current_avg * (count - 1) + processing_time) / count
        self.analysis_stats['average_processing_time'] = new_avg
    
    def _get_default_metrics(self) -> LiquidityMetrics:
        """Get default metrics when analysis fails"""
        return LiquidityMetrics(
            liquidity_level=LiquidityLevel.MEDIUM,
            bid_ask_spread=0.0,
            spread_percentage=1.0,
            market_depth=0.0,
            price_impact_1k=0.1,
            price_impact_10k=0.2,
            price_impact_100k=0.5,
            order_book_imbalance=0.0,
            liquidity_score=0.5,
            detected_patterns=[OrderBookPattern.NORMAL],
            spoofing_probability=0.0,
            timestamp=datetime.now()
        )
    
    def get_liquidity_summary(self) -> Dict[str, Any]:
        """Get comprehensive liquidity summary"""
        with self._lock:
            if self.current_metrics is None:
                return {'status': 'no_data_available'}
            
            metrics = self.current_metrics
            
            summary = {
                'liquidity_level': metrics.liquidity_level.value,
                'liquidity_score': metrics.liquidity_score,
                'spread_percentage': metrics.spread_percentage,
                'market_depth_usd': metrics.market_depth,
                'price_impact_analysis': {
                    '1k_usd': f"{metrics.price_impact_1k * 100:.2f}%",
                    '10k_usd': f"{metrics.price_impact_10k * 100:.2f}%", 
                    '100k_usd': f"{metrics.price_impact_100k * 100:.2f}%"
                },
                'order_book_imbalance': f"{metrics.order_book_imbalance * 100:.1f}%",
                'detected_patterns': [p.value for p in metrics.detected_patterns],
                'spoofing_probability': f"{metrics.spoofing_probability * 100:.1f}%",
                'analysis_timestamp': metrics.timestamp.isoformat(),
                'performance_stats': self.analysis_stats.copy()
            }
            
            # Add recommendations
            summary['recommendations'] = self._generate_trading_recommendations(metrics)
            
            return summary
    
    def _generate_trading_recommendations(self, metrics: LiquidityMetrics) -> Dict[str, Any]:
        """Generate trading recommendations based on liquidity analysis"""
        recommendations = {
            'max_recommended_size': 0,
            'execution_strategy': 'market',
            'risk_level': 'medium',
            'warnings': []
        }
        
        # Size recommendations based on price impact
        if metrics.price_impact_100k < 0.02:  # < 2% impact
            recommendations['max_recommended_size'] = 100000
        elif metrics.price_impact_10k < 0.02:
            recommendations['max_recommended_size'] = 10000
        elif metrics.price_impact_1k < 0.02:
            recommendations['max_recommended_size'] = 1000
        else:
            recommendations['max_recommended_size'] = 500
        
        # Execution strategy
        if metrics.liquidity_level in [LiquidityLevel.ULTRA_HIGH, LiquidityLevel.HIGH]:
            recommendations['execution_strategy'] = 'market'
            recommendations['risk_level'] = 'low'
        elif metrics.liquidity_level == LiquidityLevel.MEDIUM:
            recommendations['execution_strategy'] = 'limit_with_patience'
            recommendations['risk_level'] = 'medium'
        else:
            recommendations['execution_strategy'] = 'iceberg_orders'
            recommendations['risk_level'] = 'high'
            recommendations['warnings'].append('Low liquidity - consider smaller position sizes')
        
        # Pattern-based warnings
        if OrderBookPattern.SPOOFING in metrics.detected_patterns:
            recommendations['warnings'].append('Potential spoofing detected - be cautious')
        if OrderBookPattern.SQUEEZE in metrics.detected_patterns:
            recommendations['warnings'].append('Liquidity squeeze - expect higher slippage')
        if metrics.spoofing_probability > 0.5:
            recommendations['warnings'].append('High spoofing probability - monitor order book closely')
        
        return recommendations
    
    def get_analysis_statistics(self) -> Dict[str, Any]:
        """Get order book analysis statistics"""
        with self._lock:
            return {
                'snapshots_analyzed': len(self.order_book_history),
                'metrics_generated': len(self.metrics_history),
                'performance_stats': self.analysis_stats.copy(),
                'current_liquidity_level': self.current_metrics.liquidity_level.value if self.current_metrics else 'unknown',
                'average_liquidity_score': np.mean([m.liquidity_score for m in self.metrics_history]) if self.metrics_history else 0.0,
                'spoofing_events_detected': self.analysis_stats['spoofing_events_detected']
            }


# Singleton order book analyzer
_orderbook_analyzer = None
_analyzer_lock = threading.Lock()

def get_orderbook_analyzer(config: Optional[OrderBookConfig] = None) -> OrderBookAnalyzer:
    """Get the singleton order book analyzer"""
    global _orderbook_analyzer
    
    with _analyzer_lock:
        if _orderbook_analyzer is None:
            _orderbook_analyzer = OrderBookAnalyzer(config)
        return _orderbook_analyzer

def analyze_liquidity(order_book_data: Dict[str, Any]) -> LiquidityMetrics:
    """Convenient function to analyze order book liquidity"""
    analyzer = get_orderbook_analyzer()
    return analyzer.analyze_order_book(order_book_data)