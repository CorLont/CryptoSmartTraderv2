"""
Market Regime Detection System
Detects trend/mean-reversion/chop/high-vol regimes voor portfolio throttling
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import time
from scipy import stats
from .kelly_vol_sizing import MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class RegimeMetrics:
    """Market regime metrics"""
    hurst_exponent: float
    trend_strength: float  # ADX-like
    volatility_regime: float  # Current vol vs historical
    market_breadth: float  # % assets in uptrend
    momentum_persistence: float
    mean_reversion_speed: float
    regime_confidence: float  # 0-1


@dataclass
class RegimeThresholds:
    """Thresholds voor regime classification"""
    trend_hurst_min: float = 0.6
    trend_strength_min: float = 0.7
    chop_hurst_max: float = 0.4
    chop_strength_max: float = 0.3
    high_vol_threshold: float = 2.0  # 2x historical vol
    crisis_vol_threshold: float = 3.0  # 3x historical vol
    breadth_trend_min: float = 0.7


class RegimeDetector:
    """
    Advanced market regime detection voor portfolio throttling
    """
    
    def __init__(self, lookback_periods: int = 50):
        self.lookback_periods = lookback_periods
        self.logger = logging.getLogger(__name__)
        
        # Historical data storage
        self.price_history: Dict[str, List[float]] = {}
        self.returns_history: Dict[str, List[float]] = {}
        self.timestamps: List[float] = []
        
        # Regime state
        self.current_regime = MarketRegime.TREND
        self.regime_confidence = 0.5
        self.regime_history: List[Tuple[float, MarketRegime, float]] = []
        
        # Thresholds
        self.thresholds = RegimeThresholds()
        
        # Market state
        self.market_volatility = 0.2  # Default 20% annual vol
        self.historical_volatility = 0.2
        self.last_update = 0.0
    
    def update_market_data(self, prices: Dict[str, float], timestamp: float = None):
        """Update market data voor regime detection"""
        if timestamp is None:
            timestamp = time.time()
        
        self.timestamps.append(timestamp)
        
        # Update price history
        for symbol, price in prices.items():
            if symbol not in self.price_history:
                self.price_history[symbol] = []
                self.returns_history[symbol] = []
            
            self.price_history[symbol].append(price)
            
            # Calculate return
            if len(self.price_history[symbol]) > 1:
                prev_price = self.price_history[symbol][-2]
                return_val = (price - prev_price) / prev_price
                self.returns_history[symbol].append(return_val)
            
            # Maintain lookback window
            if len(self.price_history[symbol]) > self.lookback_periods:
                self.price_history[symbol] = self.price_history[symbol][-self.lookback_periods:]
                self.returns_history[symbol] = self.returns_history[symbol][-self.lookback_periods:]
        
        # Maintain timestamp window
        if len(self.timestamps) > self.lookback_periods:
            self.timestamps = self.timestamps[-self.lookback_periods:]
        
        self.last_update = timestamp
        
        # Detect regime if enough data
        if len(self.timestamps) >= 20:  # Minimum data required
            self._detect_regime()
    
    def calculate_hurst_exponent(self, returns: List[float]) -> float:
        """
        Calculate Hurst exponent voor trend persistence
        
        H > 0.5: trending (persistent)
        H < 0.5: mean-reverting (anti-persistent)  
        H ≈ 0.5: random walk
        """
        if len(returns) < 10:
            return 0.5
        
        returns_array = np.array(returns)
        n = len(returns_array)
        
        # Remove any NaN values
        returns_array = returns_array[~np.isnan(returns_array)]
        if len(returns_array) < 10:
            return 0.5
        
        # Calculate cumulative sum
        cumsum = np.cumsum(returns_array - np.mean(returns_array))
        
        # Calculate range and standard deviation for different time horizons
        lags = range(2, min(n//4, 20))
        rs_values = []
        
        for lag in lags:
            # Split into sub-periods
            n_periods = n // lag
            if n_periods < 2:
                continue
            
            ranges = []
            stds = []
            
            for i in range(n_periods):
                start_idx = i * lag
                end_idx = (i + 1) * lag
                
                period_cumsum = cumsum[start_idx:end_idx]
                period_range = np.max(period_cumsum) - np.min(period_cumsum)
                period_std = np.std(returns_array[start_idx:end_idx])
                
                if period_std > 0:
                    ranges.append(period_range)
                    stds.append(period_std)
            
            if len(ranges) > 0:
                mean_range = np.mean(ranges)
                mean_std = np.mean(stds)
                if mean_std > 0:
                    rs_values.append(mean_range / mean_std)
        
        if len(rs_values) < 3:
            return 0.5
        
        # Linear regression of log(R/S) vs log(lag)
        log_lags = np.log(lags[:len(rs_values)])
        log_rs = np.log(rs_values)
        
        # Remove any infinite values
        valid_mask = np.isfinite(log_lags) & np.isfinite(log_rs)
        if np.sum(valid_mask) < 3:
            return 0.5
        
        slope, _, _, _, _ = stats.linregress(log_lags[valid_mask], log_rs[valid_mask])
        
        return max(0.1, min(0.9, slope))  # Bound between 0.1 and 0.9
    
    def calculate_trend_strength(self, prices: List[float]) -> float:
        """
        Calculate trend strength (ADX-like indicator)
        
        Returns 0-1 where 1 = strong trend, 0 = no trend
        """
        if len(prices) < 14:
            return 0.0
        
        prices_array = np.array(prices)
        
        # Calculate directional movement
        highs = prices_array[1:]
        lows = prices_array[:-1]
        closes = prices_array
        
        # True range
        tr1 = highs - lows[:-1]
        tr2 = np.abs(highs - closes[:-2])
        tr3 = np.abs(lows[:-1] - closes[:-2])
        true_range = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional movement
        plus_dm = np.where(highs - lows[:-1] > lows[:-1] - highs, 
                          np.maximum(highs - lows[:-1], 0), 0)
        minus_dm = np.where(lows[:-1] - highs > highs - lows[:-1],
                           np.maximum(lows[:-1] - highs, 0), 0)
        
        # 14-period smoothed values
        period = min(14, len(true_range))
        atr = np.mean(true_range[-period:])
        plus_di = 100 * np.mean(plus_dm[-period:]) / atr if atr > 0 else 0
        minus_di = 100 * np.mean(minus_dm[-period:]) / atr if atr > 0 else 0
        
        # ADX calculation
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
        
        return min(1.0, dx / 100.0)
    
    def calculate_volatility_regime(self, returns: List[float]) -> float:
        """
        Calculate volatility regime (current vs historical)
        
        Returns ratio of current vol to historical vol
        """
        if len(returns) < 10:
            return 1.0
        
        returns_array = np.array(returns)
        returns_array = returns_array[~np.isnan(returns_array)]
        
        if len(returns_array) < 10:
            return 1.0
        
        # Recent volatility (last 10 periods)
        recent_vol = np.std(returns_array[-10:]) * np.sqrt(252)  # Annualized
        
        # Historical volatility (full period)
        historical_vol = np.std(returns_array) * np.sqrt(252)  # Annualized
        
        if historical_vol > 0:
            vol_ratio = recent_vol / historical_vol
        else:
            vol_ratio = 1.0
        
        return vol_ratio
    
    def calculate_market_breadth(self) -> float:
        """
        Calculate market breadth (% assets in uptrend)
        """
        if len(self.returns_history) == 0:
            return 0.5
        
        uptrend_count = 0
        total_count = 0
        
        for symbol, returns in self.returns_history.items():
            if len(returns) >= 10:
                # Simple trend: positive average return over last 10 periods
                recent_avg_return = np.mean(returns[-10:])
                if recent_avg_return > 0:
                    uptrend_count += 1
                total_count += 1
        
        if total_count > 0:
            return uptrend_count / total_count
        return 0.5
    
    def _detect_regime(self):
        """Detect current market regime"""
        
        # Use market-wide returns (average across all assets)
        all_returns = []
        all_prices = []
        
        for symbol in self.returns_history:
            if len(self.returns_history[symbol]) >= 10:
                all_returns.extend(self.returns_history[symbol])
                all_prices.extend(self.price_history[symbol])
        
        if len(all_returns) < 20:
            self.logger.warning("⚠️  Insufficient data for regime detection")
            return
        
        # Calculate regime metrics
        hurst = self.calculate_hurst_exponent(all_returns)
        trend_strength = self.calculate_trend_strength(all_prices)
        vol_regime = self.calculate_volatility_regime(all_returns)
        market_breadth = self.calculate_market_breadth()
        
        regime_metrics = RegimeMetrics(
            hurst_exponent=hurst,
            trend_strength=trend_strength,
            volatility_regime=vol_regime,
            market_breadth=market_breadth,
            momentum_persistence=hurst,  # Simplified
            mean_reversion_speed=1 - hurst,  # Simplified
            regime_confidence=0.8  # Default confidence
        )
        
        # Classify regime
        new_regime, confidence = self._classify_regime(regime_metrics)
        
        # Update regime if confidence is high enough
        if confidence > 0.6:
            self.current_regime = new_regime
            self.regime_confidence = confidence
            
            # Add to history
            self.regime_history.append((self.last_update, new_regime, confidence))
            
            # Maintain history window
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            self.logger.info(
                f"🌍 Regime detected: {new_regime.value} "
                f"(confidence: {confidence:.1%}, Hurst: {hurst:.2f}, "
                f"Trend: {trend_strength:.2f}, Vol: {vol_regime:.1f}x)"
            )
    
    def _classify_regime(self, metrics: RegimeMetrics) -> Tuple[MarketRegime, float]:
        """
        Classify market regime based on metrics
        
        Returns (regime, confidence)
        """
        confidence_scores = {}
        
        # CRISIS regime (highest priority)
        if metrics.volatility_regime >= self.thresholds.crisis_vol_threshold:
            confidence_scores[MarketRegime.CRISIS] = min(1.0, metrics.volatility_regime / 3.0)
        
        # HIGH_VOL regime
        elif metrics.volatility_regime >= self.thresholds.high_vol_threshold:
            confidence_scores[MarketRegime.HIGH_VOL] = min(1.0, metrics.volatility_regime / 2.0)
        
        # TREND regime
        trend_score = 0.0
        if (metrics.hurst_exponent >= self.thresholds.trend_hurst_min and
            metrics.trend_strength >= self.thresholds.trend_strength_min and
            metrics.market_breadth >= self.thresholds.breadth_trend_min):
            
            trend_score = (metrics.hurst_exponent + metrics.trend_strength + metrics.market_breadth) / 3.0
            confidence_scores[MarketRegime.TREND] = trend_score
        
        # MEAN_REVERSION regime
        mr_score = 0.0
        if (metrics.hurst_exponent < 0.5 and
            metrics.trend_strength < 0.5):
            
            mr_score = (1 - metrics.hurst_exponent) * (1 - metrics.trend_strength)
            confidence_scores[MarketRegime.MEAN_REVERSION] = mr_score
        
        # CHOP regime (default for unclear conditions)
        if (metrics.hurst_exponent <= self.thresholds.chop_hurst_max or
            metrics.trend_strength <= self.thresholds.chop_strength_max):
            
            chop_score = 1 - max(metrics.hurst_exponent, metrics.trend_strength)
            confidence_scores[MarketRegime.CHOP] = chop_score
        
        # Select regime with highest confidence
        if confidence_scores:
            best_regime = max(confidence_scores.items(), key=lambda x: x[1])
            return best_regime[0], best_regime[1]
        
        # Default fallback
        return MarketRegime.CHOP, 0.5
    
    def get_regime_status(self) -> Dict[str, Any]:
        """Get comprehensive regime status"""
        
        # Calculate current metrics if data available
        current_metrics = None
        if len(self.returns_history) > 0:
            all_returns = []
            all_prices = []
            
            for symbol in self.returns_history:
                if len(self.returns_history[symbol]) >= 10:
                    all_returns.extend(self.returns_history[symbol][-20:])  # Recent data
                    all_prices.extend(self.price_history[symbol][-20:])
            
            if len(all_returns) >= 10:
                current_metrics = {
                    "hurst_exponent": self.calculate_hurst_exponent(all_returns),
                    "trend_strength": self.calculate_trend_strength(all_prices),
                    "volatility_regime": self.calculate_volatility_regime(all_returns),
                    "market_breadth": self.calculate_market_breadth()
                }
        
        return {
            "current_regime": self.current_regime.value,
            "regime_confidence": self.regime_confidence,
            "last_update": self.last_update,
            "data_points": len(self.timestamps),
            "current_metrics": current_metrics,
            "thresholds": {
                "trend_hurst_min": self.thresholds.trend_hurst_min,
                "trend_strength_min": self.thresholds.trend_strength_min,
                "high_vol_threshold": self.thresholds.high_vol_threshold,
                "crisis_vol_threshold": self.thresholds.crisis_vol_threshold
            },
            "regime_history": [
                {"timestamp": ts, "regime": regime.value, "confidence": conf}
                for ts, regime, conf in self.regime_history[-10:]  # Last 10 regimes
            ]
        }


# Global regime detector instance
_global_regime_detector: Optional[RegimeDetector] = None


def get_global_regime_detector() -> RegimeDetector:
    """Get or create global regime detector"""
    global _global_regime_detector
    if _global_regime_detector is None:
        _global_regime_detector = RegimeDetector()
        logger.info("✅ Global RegimeDetector initialized")
    return _global_regime_detector


def reset_global_regime_detector():
    """Reset global regime detector (for testing)"""
    global _global_regime_detector
    _global_regime_detector = None