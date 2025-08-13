"""
Market Regime Detection and Switching System
Detects trend/mean-revert/chop/high-vol regimes and adapts trading parameters
"""

import time
import math
import statistics
from typing import Dict, List, Optional, Tuple, Deque
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import threading
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime types"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting"
    CHOPPY = "choppy"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    BREAKOUT = "breakout"
    CONSOLIDATION = "consolidation"


class RegimeConfidence(Enum):
    """Confidence levels for regime detection"""
    VERY_LOW = "very_low"      # < 30%
    LOW = "low"                # 30-50%
    MEDIUM = "medium"          # 50-70%
    HIGH = "high"              # 70-85%
    VERY_HIGH = "very_high"    # > 85%


@dataclass
class RegimeMetrics:
    """Metrics used for regime detection"""
    
    # Trend metrics
    trend_strength: float       # -1 (strong down) to 1 (strong up)
    trend_consistency: float    # 0 to 1, higher = more consistent
    
    # Mean reversion metrics
    mean_reversion_score: float # 0 to 1, higher = more mean reverting
    deviation_from_mean: float  # Standard deviations from mean
    
    # Volatility metrics
    realized_volatility: float  # Annualized realized volatility
    volatility_regime: str      # "low", "normal", "high", "extreme"
    
    # Choppiness metrics
    choppiness_index: float     # 0 to 100, higher = more choppy
    directional_movement: float # 0 to 1, higher = more directional
    
    # Momentum metrics
    momentum_strength: float    # -1 to 1
    momentum_acceleration: float # Rate of momentum change
    
    # Market microstructure
    bid_ask_spread_regime: str  # "tight", "normal", "wide"
    volume_regime: str          # "low", "normal", "high"
    
    # Time-based
    calculation_time: datetime = field(default_factory=datetime.now)
    lookback_periods: int = 50  # Number of periods used


@dataclass
class RegimeParameters:
    """Trading parameters for each regime"""
    
    # Position sizing adjustments
    sizing_multiplier: float = 1.0      # Multiply base size by this
    max_position_size: float = 0.20     # Maximum position size
    min_position_size: float = 0.01     # Minimum position size
    
    # Stop loss parameters
    stop_loss_multiplier: float = 1.0   # Multiply base stop by this
    trailing_stop_enabled: bool = True  # Enable trailing stops
    breakeven_move_ratio: float = 2.0   # Move to breakeven after 2R
    
    # Take profit parameters
    take_profit_multiplier: float = 1.0 # Multiply base TP by this
    profit_scaling_enabled: bool = True # Enable profit scaling
    partial_profit_levels: List[float] = field(default_factory=lambda: [1.5, 3.0, 5.0])
    
    # Entry throttling
    max_entries_per_hour: int = 10      # Maximum entries per hour
    min_time_between_entries: int = 300 # Minimum seconds between entries
    entry_confidence_threshold: float = 0.6  # Minimum signal confidence
    
    # Risk adjustments
    risk_multiplier: float = 1.0        # Overall risk adjustment
    correlation_threshold: float = 0.7  # Max correlation for new positions
    drawdown_scaling: bool = True       # Scale down during drawdown


@dataclass
class RegimeDetectionResult:
    """Result of regime detection"""
    primary_regime: MarketRegime
    secondary_regime: Optional[MarketRegime]
    confidence: RegimeConfidence
    confidence_score: float
    regime_duration: int           # Bars in current regime
    regime_change_detected: bool   # True if regime just changed
    metrics: RegimeMetrics
    parameters: RegimeParameters
    regime_history: List[Tuple[MarketRegime, datetime, float]]


class PriceDataBuffer:
    """Efficient price data storage for regime analysis"""
    
    def __init__(self, max_size: int = 200):
        self.max_size = max_size
        self.prices: Deque[float] = deque(maxlen=max_size)
        self.volumes: Deque[float] = deque(maxlen=max_size)
        self.timestamps: Deque[float] = deque(maxlen=max_size)
        self.highs: Deque[float] = deque(maxlen=max_size)
        self.lows: Deque[float] = deque(maxlen=max_size)
        self._lock = threading.Lock()
    
    def add_price_point(self, price: float, volume: float = 0.0, 
                       high: float = None, low: float = None, timestamp: float = None):
        """Add new price point"""
        with self._lock:
            self.prices.append(price)
            self.volumes.append(volume)
            self.timestamps.append(timestamp or time.time())
            self.highs.append(high or price)
            self.lows.append(low or price)
    
    def get_returns(self, periods: int = None) -> List[float]:
        """Calculate returns over specified periods"""
        with self._lock:
            if len(self.prices) < 2:
                return []
            
            prices = list(self.prices)
            if periods:
                prices = prices[-periods-1:]
            
            returns = []
            for i in range(1, len(prices)):
                if prices[i-1] > 0:
                    returns.append((prices[i] / prices[i-1]) - 1.0)
            
            return returns
    
    def get_price_range(self, periods: int = None) -> Tuple[float, float]:
        """Get high and low over specified periods"""
        with self._lock:
            if not self.highs or not self.lows:
                return 0.0, 0.0
            
            highs = list(self.highs)
            lows = list(self.lows)
            
            if periods:
                highs = highs[-periods:]
                lows = lows[-periods:]
            
            return max(highs), min(lows)


class RegimeIndicators:
    """Calculate technical indicators for regime detection"""
    
    @staticmethod
    def calculate_trend_strength(prices: List[float], periods: int = 20) -> float:
        """Calculate trend strength using linear regression slope"""
        if len(prices) < periods:
            return 0.0
        
        recent_prices = prices[-periods:]
        n = len(recent_prices)
        
        # Linear regression slope
        x_sum = sum(range(n))
        y_sum = sum(recent_prices)
        xy_sum = sum(i * price for i, price in enumerate(recent_prices))
        x2_sum = sum(i * i for i in range(n))
        
        denominator = n * x2_sum - x_sum * x_sum
        if denominator == 0:
            return 0.0
        
        slope = (n * xy_sum - x_sum * y_sum) / denominator
        
        # Normalize slope relative to price level
        avg_price = statistics.mean(recent_prices)
        if avg_price > 0:
            normalized_slope = slope / avg_price * 100  # Percentage slope
            # Convert to -1 to 1 range
            return max(-1.0, min(1.0, normalized_slope / 5.0))  # 5% = max strength
        
        return 0.0
    
    @staticmethod
    def calculate_choppiness_index(highs: List[float], lows: List[float], 
                                 closes: List[float], periods: int = 14) -> float:
        """Calculate Choppiness Index (0-100)"""
        if len(highs) < periods or len(lows) < periods or len(closes) < periods:
            return 50.0  # Neutral
        
        # True Range calculation
        true_ranges = []
        for i in range(1, min(len(highs), len(lows), len(closes))):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            true_ranges.append(max(high_low, high_close, low_close))
        
        if len(true_ranges) < periods:
            return 50.0
        
        # Choppiness calculation
        atr_sum = sum(true_ranges[-periods:])
        high_period = max(highs[-periods:])
        low_period = min(lows[-periods:])
        
        if high_period <= low_period or atr_sum <= 0:
            return 50.0
        
        choppiness = 100 * math.log10(atr_sum / (high_period - low_period)) / math.log10(periods)
        return max(0.0, min(100.0, choppiness))
    
    @staticmethod
    def calculate_realized_volatility(returns: List[float], periods: int = 20) -> float:
        """Calculate annualized realized volatility"""
        if len(returns) < periods:
            return 0.0
        
        recent_returns = returns[-periods:]
        if not recent_returns:
            return 0.0
        
        variance = statistics.variance(recent_returns) if len(recent_returns) > 1 else 0.0
        daily_vol = math.sqrt(variance)
        
        # Annualize (assuming daily data)
        return daily_vol * math.sqrt(252)
    
    @staticmethod
    def calculate_mean_reversion_score(prices: List[float], periods: int = 20) -> float:
        """Calculate mean reversion tendency (0-1)"""
        if len(prices) < periods + 5:
            return 0.5  # Neutral
        
        recent_prices = prices[-periods:]
        mean_price = statistics.mean(recent_prices)
        
        # Calculate how often price reverts to mean after deviation
        reversion_count = 0
        total_opportunities = 0
        
        for i in range(2, len(recent_prices)):
            prev_deviation = recent_prices[i-1] - mean_price
            curr_deviation = recent_prices[i] - mean_price
            
            # Check if price moved back toward mean
            if abs(prev_deviation) > abs(curr_deviation):
                reversion_count += 1
            total_opportunities += 1
        
        if total_opportunities == 0:
            return 0.5
        
        return reversion_count / total_opportunities
    
    @staticmethod
    def calculate_momentum_strength(prices: List[float], short_period: int = 10, 
                                  long_period: int = 30) -> float:
        """Calculate momentum strength (-1 to 1)"""
        if len(prices) < long_period:
            return 0.0
        
        short_ma = statistics.mean(prices[-short_period:])
        long_ma = statistics.mean(prices[-long_period:])
        current_price = prices[-1]
        
        # Calculate momentum relative to moving averages
        if long_ma > 0:
            ma_momentum = (short_ma - long_ma) / long_ma
            price_momentum = (current_price - long_ma) / long_ma
            
            # Combine and normalize
            combined_momentum = (ma_momentum + price_momentum) / 2
            return max(-1.0, min(1.0, combined_momentum * 10))  # 10% = max momentum
        
        return 0.0


class RegimeDetector:
    """Main regime detection and switching system"""
    
    def __init__(self, symbol: str = "default"):
        self.symbol = symbol
        self.price_buffer = PriceDataBuffer(max_size=200)
        self.current_regime = MarketRegime.CONSOLIDATION
        self.regime_confidence = RegimeConfidence.MEDIUM
        self.regime_start_time = datetime.now()
        self.regime_history: List[Tuple[MarketRegime, datetime, float]] = []
        
        # Regime parameters
        self.regime_parameters = self._initialize_regime_parameters()
        
        # Detection thresholds
        self.trend_threshold = 0.3
        self.high_vol_threshold = 0.4  # 40% annualized
        self.choppiness_threshold = 61.8  # Golden ratio
        self.mean_reversion_threshold = 0.7
        
        self._lock = threading.Lock()
    
    def _initialize_regime_parameters(self) -> Dict[MarketRegime, RegimeParameters]:
        """Initialize trading parameters for each regime"""
        
        return {
            MarketRegime.TRENDING_UP: RegimeParameters(
                sizing_multiplier=1.2,          # Larger positions in trends
                stop_loss_multiplier=0.8,       # Tighter stops
                take_profit_multiplier=1.5,     # Wider targets
                trailing_stop_enabled=True,
                max_entries_per_hour=8,
                entry_confidence_threshold=0.5,
                risk_multiplier=1.1
            ),
            
            MarketRegime.TRENDING_DOWN: RegimeParameters(
                sizing_multiplier=0.8,          # Smaller short positions
                stop_loss_multiplier=0.7,       # Very tight stops
                take_profit_multiplier=1.3,
                trailing_stop_enabled=True,
                max_entries_per_hour=6,
                entry_confidence_threshold=0.6,
                risk_multiplier=0.9
            ),
            
            MarketRegime.MEAN_REVERTING: RegimeParameters(
                sizing_multiplier=1.0,
                stop_loss_multiplier=1.2,       # Wider stops
                take_profit_multiplier=0.8,     # Quicker profits
                trailing_stop_enabled=False,    # No trailing in mean reversion
                max_entries_per_hour=12,
                entry_confidence_threshold=0.7,
                risk_multiplier=1.0,
                partial_profit_levels=[1.0, 2.0, 3.0]  # Quick scaling
            ),
            
            MarketRegime.CHOPPY: RegimeParameters(
                sizing_multiplier=0.6,          # Much smaller positions
                stop_loss_multiplier=0.9,
                take_profit_multiplier=0.7,     # Quick profits
                trailing_stop_enabled=False,
                max_entries_per_hour=4,         # Fewer entries
                entry_confidence_threshold=0.8, # High confidence required
                risk_multiplier=0.7
            ),
            
            MarketRegime.HIGH_VOLATILITY: RegimeParameters(
                sizing_multiplier=0.5,          # Small positions
                stop_loss_multiplier=1.5,       # Wide stops
                take_profit_multiplier=1.8,     # Wide targets
                trailing_stop_enabled=True,
                max_entries_per_hour=3,
                entry_confidence_threshold=0.9,
                risk_multiplier=0.6
            ),
            
            MarketRegime.LOW_VOLATILITY: RegimeParameters(
                sizing_multiplier=1.3,          # Larger positions
                stop_loss_multiplier=0.6,       # Tight stops
                take_profit_multiplier=0.9,
                trailing_stop_enabled=True,
                max_entries_per_hour=15,
                entry_confidence_threshold=0.4,
                risk_multiplier=1.2
            ),
            
            MarketRegime.BREAKOUT: RegimeParameters(
                sizing_multiplier=1.5,          # Large positions
                stop_loss_multiplier=0.7,
                take_profit_multiplier=2.0,     # Big targets
                trailing_stop_enabled=True,
                max_entries_per_hour=5,
                entry_confidence_threshold=0.6,
                risk_multiplier=1.3
            ),
            
            MarketRegime.CONSOLIDATION: RegimeParameters(
                sizing_multiplier=0.8,
                stop_loss_multiplier=1.0,
                take_profit_multiplier=1.0,
                trailing_stop_enabled=True,
                max_entries_per_hour=10,
                entry_confidence_threshold=0.5,
                risk_multiplier=1.0
            )
        }
    
    def update_price_data(self, price: float, volume: float = 0.0,
                         high: float = None, low: float = None,
                         timestamp: float = None):
        """Update price data and trigger regime detection"""
        
        self.price_buffer.add_price_point(price, volume, high, low, timestamp)
        
        # Trigger regime detection if we have enough data
        if len(self.price_buffer.prices) >= 50:
            self.detect_regime()
    
    def detect_regime(self) -> RegimeDetectionResult:
        """Detect current market regime"""
        
        with self._lock:
            # Get price data
            prices = list(self.price_buffer.prices)
            highs = list(self.price_buffer.highs)
            lows = list(self.price_buffer.lows)
            
            if len(prices) < 50:
                return self._create_default_result()
            
            # Calculate metrics
            returns = self.price_buffer.get_returns()
            
            metrics = RegimeMetrics(
                trend_strength=RegimeIndicators.calculate_trend_strength(prices),
                trend_consistency=self._calculate_trend_consistency(prices),
                mean_reversion_score=RegimeIndicators.calculate_mean_reversion_score(prices),
                deviation_from_mean=self._calculate_deviation_from_mean(prices),
                realized_volatility=RegimeIndicators.calculate_realized_volatility(returns),
                volatility_regime=self._classify_volatility_regime(returns),
                choppiness_index=RegimeIndicators.calculate_choppiness_index(highs, lows, prices),
                directional_movement=self._calculate_directional_movement(prices),
                momentum_strength=RegimeIndicators.calculate_momentum_strength(prices),
                momentum_acceleration=self._calculate_momentum_acceleration(prices),
                bid_ask_spread_regime="normal",  # Placeholder
                volume_regime="normal"           # Placeholder
            )
            
            # Detect regime based on metrics
            regime, confidence_score = self._classify_regime(metrics)
            confidence = self._classify_confidence(confidence_score)
            
            # Check for regime change
            regime_changed = regime != self.current_regime
            if regime_changed:
                # Log regime change
                self.regime_history.append((
                    self.current_regime, 
                    self.regime_start_time, 
                    (datetime.now() - self.regime_start_time).total_seconds() / 3600
                ))
                
                self.current_regime = regime
                self.regime_confidence = confidence
                self.regime_start_time = datetime.now()
                
                logger.info(f"Regime change detected for {self.symbol}: {regime.value} (confidence: {confidence.value})")
            
            # Calculate regime duration
            regime_duration_hours = (datetime.now() - self.regime_start_time).total_seconds() / 3600
            
            # Get parameters for current regime
            parameters = self.regime_parameters.get(regime, RegimeParameters())
            
            return RegimeDetectionResult(
                primary_regime=regime,
                secondary_regime=self._detect_secondary_regime(metrics),
                confidence=confidence,
                confidence_score=confidence_score,
                regime_duration=int(regime_duration_hours * 60),  # Convert to minutes
                regime_change_detected=regime_changed,
                metrics=metrics,
                parameters=parameters,
                regime_history=self.regime_history[-10:]  # Last 10 regime changes
            )
    
    def _classify_regime(self, metrics: RegimeMetrics) -> Tuple[MarketRegime, float]:
        """Classify regime based on metrics"""
        
        scores = {}
        
        # Trending regimes
        if abs(metrics.trend_strength) > self.trend_threshold:
            if metrics.trend_strength > 0:
                trend_score = metrics.trend_strength * metrics.trend_consistency
                scores[MarketRegime.TRENDING_UP] = trend_score * 0.8 + metrics.momentum_strength * 0.2
            else:
                trend_score = abs(metrics.trend_strength) * metrics.trend_consistency
                scores[MarketRegime.TRENDING_DOWN] = trend_score * 0.8 + abs(metrics.momentum_strength) * 0.2
        
        # Mean reverting regime
        if metrics.mean_reversion_score > self.mean_reversion_threshold:
            reversion_score = metrics.mean_reversion_score * (1 - metrics.choppiness_index / 100)
            scores[MarketRegime.MEAN_REVERTING] = reversion_score
        
        # Volatility regimes
        if metrics.realized_volatility > self.high_vol_threshold:
            vol_score = min(1.0, metrics.realized_volatility / self.high_vol_threshold)
            scores[MarketRegime.HIGH_VOLATILITY] = vol_score
        elif metrics.realized_volatility < self.high_vol_threshold * 0.5:
            low_vol_score = 1.0 - (metrics.realized_volatility / (self.high_vol_threshold * 0.5))
            scores[MarketRegime.LOW_VOLATILITY] = low_vol_score
        
        # Choppy regime
        if metrics.choppiness_index > self.choppiness_threshold:
            chop_score = (metrics.choppiness_index - self.choppiness_threshold) / (100 - self.choppiness_threshold)
            scores[MarketRegime.CHOPPY] = chop_score
        
        # Breakout regime
        if (abs(metrics.trend_strength) > 0.5 and 
            metrics.momentum_acceleration > 0.3 and 
            metrics.realized_volatility > self.high_vol_threshold * 0.8):
            breakout_score = (abs(metrics.trend_strength) + metrics.momentum_acceleration) / 2
            scores[MarketRegime.BREAKOUT] = breakout_score
        
        # Default to consolidation if no strong signals
        if not scores:
            scores[MarketRegime.CONSOLIDATION] = 0.5
        
        # Find regime with highest score
        best_regime = max(scores, key=scores.get)
        best_score = scores[best_regime]
        
        return best_regime, best_score
    
    def _detect_secondary_regime(self, metrics: RegimeMetrics) -> Optional[MarketRegime]:
        """Detect secondary regime characteristics"""
        
        # Simple heuristic for secondary regime
        if metrics.realized_volatility > self.high_vol_threshold:
            return MarketRegime.HIGH_VOLATILITY
        elif metrics.choppiness_index > self.choppiness_threshold:
            return MarketRegime.CHOPPY
        
        return None
    
    def _classify_confidence(self, score: float) -> RegimeConfidence:
        """Classify confidence level based on score"""
        
        if score < 0.3:
            return RegimeConfidence.VERY_LOW
        elif score < 0.5:
            return RegimeConfidence.LOW
        elif score < 0.7:
            return RegimeConfidence.MEDIUM
        elif score < 0.85:
            return RegimeConfidence.HIGH
        else:
            return RegimeConfidence.VERY_HIGH
    
    def _calculate_trend_consistency(self, prices: List[float], periods: int = 20) -> float:
        """Calculate how consistent the trend is"""
        
        if len(prices) < periods + 1:
            return 0.0
        
        returns = []
        for i in range(len(prices) - periods, len(prices)):
            if i > 0 and prices[i-1] > 0:
                returns.append((prices[i] / prices[i-1]) - 1.0)
        
        if not returns:
            return 0.0
        
        # Count returns in same direction as overall trend
        total_return = sum(returns)
        same_direction = sum(1 for r in returns if (r > 0) == (total_return > 0))
        
        return same_direction / len(returns)
    
    def _calculate_deviation_from_mean(self, prices: List[float], periods: int = 20) -> float:
        """Calculate current deviation from mean in standard deviations"""
        
        if len(prices) < periods:
            return 0.0
        
        recent_prices = prices[-periods:]
        mean_price = statistics.mean(recent_prices)
        std_price = statistics.stdev(recent_prices) if len(recent_prices) > 1 else 0.0
        
        if std_price > 0:
            return (prices[-1] - mean_price) / std_price
        
        return 0.0
    
    def _calculate_directional_movement(self, prices: List[float], periods: int = 14) -> float:
        """Calculate directional movement index"""
        
        if len(prices) < periods + 1:
            return 0.0
        
        up_moves = 0
        down_moves = 0
        
        for i in range(len(prices) - periods, len(prices)):
            if i > 0:
                move = prices[i] - prices[i-1]
                if move > 0:
                    up_moves += move
                elif move < 0:
                    down_moves += abs(move)
        
        total_movement = up_moves + down_moves
        if total_movement > 0:
            return abs(up_moves - down_moves) / total_movement
        
        return 0.0
    
    def _calculate_momentum_acceleration(self, prices: List[float]) -> float:
        """Calculate momentum acceleration"""
        
        if len(prices) < 20:
            return 0.0
        
        # Calculate momentum at two different points
        recent_momentum = RegimeIndicators.calculate_momentum_strength(prices[-10:])
        older_momentum = RegimeIndicators.calculate_momentum_strength(prices[-20:-10])
        
        return recent_momentum - older_momentum
    
    def _create_default_result(self) -> RegimeDetectionResult:
        """Create default result when insufficient data"""
        
        return RegimeDetectionResult(
            primary_regime=MarketRegime.CONSOLIDATION,
            secondary_regime=None,
            confidence=RegimeConfidence.LOW,
            confidence_score=0.3,
            regime_duration=0,
            regime_change_detected=False,
            metrics=RegimeMetrics(
                trend_strength=0.0,
                trend_consistency=0.0,
                mean_reversion_score=0.5,
                deviation_from_mean=0.0,
                realized_volatility=0.0,
                volatility_regime="normal",
                choppiness_index=50.0,
                directional_movement=0.0,
                momentum_strength=0.0,
                momentum_acceleration=0.0,
                bid_ask_spread_regime="normal",
                volume_regime="normal"
            ),
            parameters=RegimeParameters(),
            regime_history=[]
        )
    
    def get_regime_summary(self) -> Dict:
        """Get comprehensive regime summary"""
        
        with self._lock:
            return {
                'symbol': self.symbol,
                'current_regime': self.current_regime.value,
                'confidence': self.regime_confidence.value,
                'regime_duration_minutes': int((datetime.now() - self.regime_start_time).total_seconds() / 60),
                'regime_changes_today': len([h for h in self.regime_history 
                                           if h[1].date() == datetime.now().date()]),
                'data_points': len(self.price_buffer.prices),
                'regime_parameters': {
                    'sizing_multiplier': self.regime_parameters[self.current_regime].sizing_multiplier,
                    'stop_loss_multiplier': self.regime_parameters[self.current_regime].stop_loss_multiplier,
                    'take_profit_multiplier': self.regime_parameters[self.current_regime].take_profit_multiplier,
                    'max_entries_per_hour': self.regime_parameters[self.current_regime].max_entries_per_hour,
                    'risk_multiplier': self.regime_parameters[self.current_regime].risk_multiplier
                }
            }


# Global regime detectors
_regime_detectors: Dict[str, RegimeDetector] = {}
_detector_lock = threading.Lock()


def get_regime_detector(symbol: str = "default") -> RegimeDetector:
    """Get regime detector for symbol"""
    global _regime_detectors
    
    if symbol not in _regime_detectors:
        with _detector_lock:
            if symbol not in _regime_detectors:
                _regime_detectors[symbol] = RegimeDetector(symbol)
    
    return _regime_detectors[symbol]


def detect_current_regime(symbol: str, price: float, volume: float = 0.0,
                         high: float = None, low: float = None) -> RegimeDetectionResult:
    """Convenience function for regime detection"""
    
    detector = get_regime_detector(symbol)
    detector.update_price_data(price, volume, high, low)
    return detector.detect_regime()


if __name__ == "__main__":
    # Example usage
    detector = RegimeDetector("BTC/USD")
    
    # Simulate some price data
    import random
    base_price = 50000.0
    
    for i in range(100):
        # Add some trend and noise
        trend = i * 50  # Upward trend
        noise = random.gauss(0, 500)  # Random noise
        price = base_price + trend + noise
        
        detector.update_price_data(price, volume=1000.0)
    
    # Get regime detection result
    result = detector.detect_regime()
    
    print(f"Detected regime: {result.primary_regime.value}")
    print(f"Confidence: {result.confidence.value} ({result.confidence_score:.2f})")
    print(f"Trend strength: {result.metrics.trend_strength:.2f}")
    print(f"Volatility: {result.metrics.realized_volatility:.2f}")
    print(f"Choppiness: {result.metrics.choppiness_index:.1f}")
    
    # Show adjusted parameters
    params = result.parameters
    print(f"\nAdjusted Parameters:")
    print(f"Sizing multiplier: {params.sizing_multiplier:.1f}")
    print(f"Stop loss multiplier: {params.stop_loss_multiplier:.1f}")
    print(f"Take profit multiplier: {params.take_profit_multiplier:.1f}")
    print(f"Max entries/hour: {params.max_entries_per_hour}")
