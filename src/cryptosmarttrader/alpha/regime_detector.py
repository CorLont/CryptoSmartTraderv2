"""
Regime Detection System for CryptoSmartTrader
Advanced market regime classification with strategy switching and throttling.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Technical analysis libraries
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


class MarketRegime(Enum):
    """Market regime classifications."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    MEAN_REVERTING = "mean_reverting" 
    CHOPPY = "choppy"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class RegimeSignal:
    """Regime detection signal."""
    regime: MarketRegime
    confidence: float
    strength: float
    timestamp: datetime
    indicators: Dict[str, float]
    metadata: Dict[str, Any]


class RegimeDetector:
    """
    Enterprise regime detection system with multiple indicators.
    
    Features:
    - Hurst exponent for trend strength measurement
    - ADX for directional movement strength
    - Volatility regime classification
    - Market breadth analysis
    - Momentum scoring
    - Confidence weighting
    """
    
    def __init__(self, lookback_periods: int = 100, min_confidence: float = 0.6):
        self.lookback_periods = lookback_periods
        self.min_confidence = min_confidence
        self.logger = logging.getLogger(__name__)
        
        # Regime thresholds
        self.hurst_trend_threshold = 0.6
        self.hurst_mean_revert_threshold = 0.4
        self.adx_strong_trend_threshold = 30
        self.adx_weak_trend_threshold = 20
        self.volatility_high_threshold = 0.04  # 4% daily volatility
        self.volatility_low_threshold = 0.015  # 1.5% daily volatility
        
        self.logger.info("RegimeDetector initialized with enterprise indicators")
    
    def detect_regime(self, price_data: pd.Series, volume_data: Optional[pd.Series] = None) -> RegimeSignal:
        """
        Detect current market regime using multiple indicators.
        
        Args:
            price_data: Time series of prices (should have datetime index)
            volume_data: Optional volume data for additional analysis
            
        Returns:
            RegimeSignal with regime classification and confidence
        """
        if len(price_data) < self.lookback_periods:
            raise ValueError(f"Insufficient data: need {self.lookback_periods}, got {len(price_data)}")
        
        # Calculate returns
        returns = price_data.pct_change().dropna()
        
        # Calculate regime indicators
        indicators = {}
        
        # 1. Hurst Exponent (trend persistence)
        hurst_exp = self._calculate_hurst_exponent(price_data.iloc[-self.lookback_periods:])
        indicators['hurst_exponent'] = hurst_exp
        
        # 2. ADX (Average Directional Index)
        adx_value = self._calculate_adx(price_data.iloc[-self.lookback_periods:])
        indicators['adx'] = adx_value
        
        # 3. Volatility metrics
        volatility = self._calculate_volatility_metrics(returns.iloc[-self.lookback_periods:])
        indicators.update(volatility)
        
        # 4. Momentum indicators
        momentum = self._calculate_momentum_indicators(price_data.iloc[-self.lookback_periods:])
        indicators.update(momentum)
        
        # 5. Market breadth (if volume available)
        if volume_data is not None:
            breadth = self._calculate_market_breadth(price_data.iloc[-self.lookback_periods:], 
                                                   volume_data.iloc[-self.lookback_periods:])
            indicators.update(breadth)
        
        # Classify regime based on indicators
        regime, confidence, strength = self._classify_regime(indicators)
        
        return RegimeSignal(
            regime=regime,
            confidence=confidence,
            strength=strength,
            timestamp=datetime.utcnow(),
            indicators=indicators,
            metadata={
                'lookback_periods': self.lookback_periods,
                'data_points_used': len(price_data)
            }
        )
    
    def _calculate_hurst_exponent(self, price_series: pd.Series) -> float:
        """Calculate Hurst exponent for trend persistence."""
        try:
            log_prices = np.log(price_series)
            
            # Calculate R/S statistic for different lags
            lags = range(2, min(50, len(log_prices) // 4))
            rs_values = []
            
            for lag in lags:
                # Split into non-overlapping windows
                n_windows = len(log_prices) // lag
                if n_windows < 2:
                    continue
                    
                rs_window = []
                for i in range(n_windows):
                    window = log_prices.iloc[i*lag:(i+1)*lag]
                    if len(window) < lag:
                        continue
                    
                    # Calculate mean
                    mean_val = window.mean()
                    
                    # Calculate cumulative deviations
                    deviations = (window - mean_val).cumsum()
                    
                    # Calculate range and standard deviation
                    range_val = deviations.max() - deviations.min()
                    std_val = window.std()
                    
                    if std_val > 0:
                        rs_window.append(range_val / std_val)
                
                if rs_window:
                    rs_values.append(np.mean(rs_window))
            
            if len(rs_values) < 3:
                return 0.5  # Random walk default
            
            # Fit log(R/S) vs log(lag)
            log_lags = np.log(list(lags[:len(rs_values)]))
            log_rs = np.log(rs_values)
            
            # Linear regression to get Hurst exponent
            hurst = np.polyfit(log_lags, log_rs, 1)[0]
            
            # Clamp to reasonable range
            return max(0.0, min(1.0, hurst))
            
        except Exception as e:
            self.logger.warning(f"Error calculating Hurst exponent: {e}")
            return 0.5
    
    def _calculate_adx(self, price_data: pd.Series) -> float:
        """Calculate Average Directional Index."""
        try:
            if not HAS_TALIB:
                # Simple ADX approximation
                returns = price_data.pct_change().dropna()
                pos_moves = returns.where(returns > 0, 0)
                neg_moves = returns.where(returns < 0, 0).abs()
                
                # Exponential moving averages
                alpha = 2.0 / 15.0  # 14-period equivalent
                pos_dm = pos_moves.ewm(alpha=alpha).mean()
                neg_dm = neg_moves.ewm(alpha=alpha).mean()
                
                # Directional indicators
                total_dm = pos_dm + neg_dm
                dx = abs(pos_dm - neg_dm) / total_dm.where(total_dm != 0, 1)
                adx = dx.ewm(alpha=alpha).mean().iloc[-1] * 100
                
                return float(adx)
            else:
                # Use TA-Lib if available
                high = price_data * 1.001  # Approximate high
                low = price_data * 0.999   # Approximate low
                close = price_data
                
                adx = talib.ADX(high.values, low.values, close.values, timeperiod=14)
                return float(adx[-1]) if not np.isnan(adx[-1]) else 25.0
                
        except Exception as e:
            self.logger.warning(f"Error calculating ADX: {e}")
            return 25.0  # Neutral value
    
    def _calculate_volatility_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate volatility-based metrics."""
        try:
            # Realized volatility (annualized)
            realized_vol = returns.std() * np.sqrt(365)
            
            # GARCH-like volatility (exponentially weighted)
            ewm_vol = returns.ewm(halflife=10).std() * np.sqrt(365)
            
            # Volatility of volatility
            vol_returns = returns.rolling(10).std()
            vol_of_vol = vol_returns.std()
            
            # Parkinson estimator (if we had high/low data)
            # For now, use realized vol
            parkinson_vol = realized_vol
            
            return {
                'realized_volatility': float(realized_vol),
                'ewm_volatility': float(ewm_vol.iloc[-1]),
                'volatility_of_volatility': float(vol_of_vol),
                'parkinson_volatility': float(parkinson_vol)
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating volatility metrics: {e}")
            return {
                'realized_volatility': 0.02,
                'ewm_volatility': 0.02,
                'volatility_of_volatility': 0.01,
                'parkinson_volatility': 0.02
            }
    
    def _calculate_momentum_indicators(self, price_data: pd.Series) -> Dict[str, float]:
        """Calculate momentum-based indicators."""
        try:
            # Simple momentum (rate of change)
            momentum_10 = (price_data.iloc[-1] / price_data.iloc[-11] - 1) * 100
            momentum_20 = (price_data.iloc[-1] / price_data.iloc[-21] - 1) * 100
            
            # RSI-like indicator
            returns = price_data.pct_change().dropna()
            gains = returns.where(returns > 0, 0)
            losses = returns.where(returns < 0, 0).abs()
            
            avg_gains = gains.ewm(alpha=1/14).mean()
            avg_losses = losses.ewm(alpha=1/14).mean()
            
            rs = avg_gains / avg_losses.replace(0, 1)
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # Moving average convergence
            ema_12 = price_data.ewm(span=12).mean()
            ema_26 = price_data.ewm(span=26).mean()
            macd = ((ema_12 - ema_26) / price_data * 100).iloc[-1]
            
            return {
                'momentum_10d': float(momentum_10),
                'momentum_20d': float(momentum_20),
                'rsi': float(rsi),
                'macd_signal': float(macd)
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating momentum indicators: {e}")
            return {
                'momentum_10d': 0.0,
                'momentum_20d': 0.0,
                'rsi': 50.0,
                'macd_signal': 0.0
            }
    
    def _calculate_market_breadth(self, price_data: pd.Series, volume_data: pd.Series) -> Dict[str, float]:
        """Calculate market breadth indicators."""
        try:
            # Volume-weighted price momentum
            returns = price_data.pct_change().dropna()
            volume_aligned = volume_data.iloc[1:]  # Align with returns
            
            vwap_momentum = (returns * volume_aligned).sum() / volume_aligned.sum()
            
            # Volume trend
            volume_ma_short = volume_data.rolling(5).mean()
            volume_ma_long = volume_data.rolling(20).mean()
            volume_trend = (volume_ma_short.iloc[-1] / volume_ma_long.iloc[-1] - 1) * 100
            
            # Price-volume divergence
            price_trend = (price_data.iloc[-5:].mean() / price_data.iloc[-20:-5].mean() - 1) * 100
            pv_divergence = abs(price_trend - volume_trend)
            
            return {
                'vwap_momentum': float(vwap_momentum),
                'volume_trend': float(volume_trend),
                'price_volume_divergence': float(pv_divergence)
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating market breadth: {e}")
            return {
                'vwap_momentum': 0.0,
                'volume_trend': 0.0,
                'price_volume_divergence': 5.0
            }
    
    def _classify_regime(self, indicators: Dict[str, float]) -> Tuple[MarketRegime, float, float]:
        """Classify market regime based on indicators."""
        
        # Extract key indicators
        hurst = indicators.get('hurst_exponent', 0.5)
        adx = indicators.get('adx', 25.0)
        volatility = indicators.get('realized_volatility', 0.02)
        momentum_10 = indicators.get('momentum_10d', 0.0)
        momentum_20 = indicators.get('momentum_20d', 0.0)
        rsi = indicators.get('rsi', 50.0)
        
        # Score each regime
        regime_scores = {}
        
        # Trending Up
        trend_up_score = 0.0
        if hurst > self.hurst_trend_threshold:
            trend_up_score += 0.3
        if adx > self.adx_strong_trend_threshold:
            trend_up_score += 0.3
        if momentum_10 > 2.0 and momentum_20 > 1.0:
            trend_up_score += 0.25
        if rsi > 55:
            trend_up_score += 0.15
        
        regime_scores[MarketRegime.TRENDING_UP] = trend_up_score
        
        # Trending Down
        trend_down_score = 0.0
        if hurst > self.hurst_trend_threshold:
            trend_down_score += 0.3
        if adx > self.adx_strong_trend_threshold:
            trend_down_score += 0.3
        if momentum_10 < -2.0 and momentum_20 < -1.0:
            trend_down_score += 0.25
        if rsi < 45:
            trend_down_score += 0.15
        
        regime_scores[MarketRegime.TRENDING_DOWN] = trend_down_score
        
        # Mean Reverting
        mean_revert_score = 0.0
        if hurst < self.hurst_mean_revert_threshold:
            mean_revert_score += 0.4
        if adx < self.adx_weak_trend_threshold:
            mean_revert_score += 0.3
        if abs(momentum_10) < 1.0:
            mean_revert_score += 0.2
        if 45 <= rsi <= 55:
            mean_revert_score += 0.1
        
        regime_scores[MarketRegime.MEAN_REVERTING] = mean_revert_score
        
        # Choppy
        choppy_score = 0.0
        if 0.4 <= hurst <= 0.6:
            choppy_score += 0.3
        if self.adx_weak_trend_threshold <= adx <= self.adx_strong_trend_threshold:
            choppy_score += 0.3
        if abs(momentum_10 - momentum_20) > 2.0:  # Conflicting signals
            choppy_score += 0.25
        if volatility > self.volatility_high_threshold * 0.7:
            choppy_score += 0.15
        
        regime_scores[MarketRegime.CHOPPY] = choppy_score
        
        # High Volatility
        high_vol_score = 0.0
        if volatility > self.volatility_high_threshold:
            high_vol_score += 0.5
        if indicators.get('volatility_of_volatility', 0.01) > 0.02:
            high_vol_score += 0.3
        if abs(momentum_10) > 5.0:
            high_vol_score += 0.2
        
        regime_scores[MarketRegime.HIGH_VOLATILITY] = high_vol_score
        
        # Low Volatility
        low_vol_score = 0.0
        if volatility < self.volatility_low_threshold:
            low_vol_score += 0.5
        if indicators.get('volatility_of_volatility', 0.01) < 0.005:
            low_vol_score += 0.3
        if abs(momentum_10) < 0.5:
            low_vol_score += 0.2
        
        regime_scores[MarketRegime.LOW_VOLATILITY] = low_vol_score
        
        # Find best regime
        best_regime = max(regime_scores, key=regime_scores.get)
        confidence = regime_scores[best_regime]
        
        # Calculate regime strength
        second_best_score = sorted(regime_scores.values(), reverse=True)[1]
        strength = confidence - second_best_score if len(regime_scores) > 1 else confidence
        
        # Ensure minimum confidence
        if confidence < self.min_confidence:
            # Default to choppy if confidence is low
            best_regime = MarketRegime.CHOPPY
            confidence = self.min_confidence
        
        return best_regime, confidence, strength
    
    def get_regime_strategy_config(self, regime: MarketRegime) -> Dict[str, Any]:
        """Get recommended strategy configuration for regime."""
        
        strategy_configs = {
            MarketRegime.TRENDING_UP: {
                'strategy_type': 'momentum',
                'position_sizing': 'aggressive',
                'rebalance_frequency': 'daily',
                'stop_loss_pct': 3.0,
                'take_profit_pct': 8.0,
                'max_positions': 8,
                'volatility_target': 0.20,
                'throttling_factor': 1.0
            },
            MarketRegime.TRENDING_DOWN: {
                'strategy_type': 'momentum_short',
                'position_sizing': 'conservative',
                'rebalance_frequency': 'daily',
                'stop_loss_pct': 2.5,
                'take_profit_pct': 6.0,
                'max_positions': 5,
                'volatility_target': 0.15,
                'throttling_factor': 0.7
            },
            MarketRegime.MEAN_REVERTING: {
                'strategy_type': 'mean_reversion',
                'position_sizing': 'moderate',
                'rebalance_frequency': 'hourly',
                'stop_loss_pct': 2.0,
                'take_profit_pct': 4.0,
                'max_positions': 12,
                'volatility_target': 0.12,
                'throttling_factor': 0.8
            },
            MarketRegime.CHOPPY: {
                'strategy_type': 'range_bound',
                'position_sizing': 'conservative',
                'rebalance_frequency': '4hourly',
                'stop_loss_pct': 1.5,
                'take_profit_pct': 3.0,
                'max_positions': 6,
                'volatility_target': 0.10,
                'throttling_factor': 0.5
            },
            MarketRegime.HIGH_VOLATILITY: {
                'strategy_type': 'defensive',
                'position_sizing': 'minimal',
                'rebalance_frequency': 'hourly',
                'stop_loss_pct': 1.0,
                'take_profit_pct': 2.0,
                'max_positions': 3,
                'volatility_target': 0.08,
                'throttling_factor': 0.3
            },
            MarketRegime.LOW_VOLATILITY: {
                'strategy_type': 'momentum',
                'position_sizing': 'aggressive',
                'rebalance_frequency': 'daily',
                'stop_loss_pct': 4.0,
                'take_profit_pct': 10.0,
                'max_positions': 15,
                'volatility_target': 0.25,
                'throttling_factor': 1.2
            }
        }
        
        return strategy_configs.get(regime, strategy_configs[MarketRegime.CHOPPY])


def create_regime_detector(lookback_periods: int = 100, min_confidence: float = 0.6) -> RegimeDetector:
    """Create regime detector with specified parameters."""
    return RegimeDetector(lookback_periods, min_confidence)