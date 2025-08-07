#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Market Regime Detection Engine
Advanced market regime detection and volatility classification with auto-model switching
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import threading
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    SIDEWAYS_LOW_VOL = "sideways_low_vol"
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"
    CRASH = "crash"
    BUBBLE = "bubble"
    RECOVERY = "recovery"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    UNKNOWN = "unknown"

class VolatilityRegime(Enum):
    ULTRA_LOW = "ultra_low"     # < 10% annualized
    LOW = "low"                 # 10-30% annualized
    MEDIUM = "medium"           # 30-60% annualized
    HIGH = "high"               # 60-100% annualized
    EXTREME = "extreme"         # > 100% annualized

@dataclass
class RegimeDetectionConfig:
    """Configuration for market regime detection"""
    lookback_periods: List[int] = field(default_factory=lambda: [20, 50, 100, 200])
    volatility_window: int = 20
    trend_window: int = 50
    volume_window: int = 20
    regime_min_duration: int = 5  # Minimum periods for regime change
    confidence_threshold: float = 0.6
    volatility_quantiles: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8])
    enable_ml_clustering: bool = True
    n_regime_clusters: int = 6
    update_frequency: int = 10  # Update every N periods

@dataclass
class RegimeSignal:
    """Market regime detection signal"""
    regime: MarketRegime
    volatility_regime: VolatilityRegime
    confidence: float
    timestamp: datetime
    indicators: Dict[str, float] = field(default_factory=dict)
    regime_duration: int = 0
    trend_strength: float = 0.0
    volatility_percentile: float = 0.0
    volume_profile: str = "normal"

class MarketRegimeDetector:
    """Advanced market regime detection and classification system"""
    
    def __init__(self, config: Optional[RegimeDetectionConfig] = None):
        self.config = config or RegimeDetectionConfig()
        self.logger = logging.getLogger(f"{__name__}.MarketRegimeDetector")
        
        # State tracking
        self.current_regime = MarketRegime.UNKNOWN
        self.current_volatility_regime = VolatilityRegime.MEDIUM
        self.regime_history: List[RegimeSignal] = []
        self.regime_start_time = datetime.now()
        self.regime_duration = 0
        
        # ML models for regime detection
        self.regime_clusterer = None
        self.volatility_scaler = StandardScaler()
        self.regime_features_cache = {}
        
        # Historical data for regime classification
        self.price_history = []
        self.volume_history = []
        self.volatility_history = []
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Model switching callbacks
        self.regime_change_callbacks: List[Callable] = []
        
        self.logger.info("Market Regime Detector initialized with ML clustering")
    
    def add_regime_change_callback(self, callback: Callable[[RegimeSignal], None]):
        """Add callback for regime changes"""
        with self._lock:
            self.regime_change_callbacks.append(callback)
    
    def update_market_data(self, price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None) -> RegimeSignal:
        """
        Update market data and detect current regime
        
        Args:
            price_data: DataFrame with OHLCV data
            volume_data: Optional volume data
            
        Returns:
            Current regime signal
        """
        with self._lock:
            try:
                # Validate and process input data
                if price_data.empty:
                    return self._get_current_regime_signal()
                
                # Extract key market features
                market_features = self._extract_market_features(price_data, volume_data)
                
                # Detect current regime
                regime_signal = self._detect_regime(market_features, price_data)
                
                # Check for regime change
                if regime_signal.regime != self.current_regime:
                    self._handle_regime_change(regime_signal)
                
                # Update regime duration
                regime_signal.regime_duration = self.regime_duration
                
                # Store in history
                self.regime_history.append(regime_signal)
                if len(self.regime_history) > 1000:  # Limit history size
                    self.regime_history = self.regime_history[-1000:]
                
                # Update cache
                self._update_historical_cache(price_data, volume_data)
                
                return regime_signal
                
            except Exception as e:
                self.logger.error(f"Regime detection update failed: {e}")
                return self._get_current_regime_signal()
    
    def _extract_market_features(self, price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame]) -> Dict[str, float]:
        """Extract comprehensive market features for regime detection"""
        features = {}
        
        try:
            # Ensure we have required columns
            if 'close' not in price_data.columns:
                if 'price' in price_data.columns:
                    price_data = price_data.rename(columns={'price': 'close'})
                else:
                    # Use first numeric column as price
                    numeric_cols = price_data.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        price_data = price_data.rename(columns={numeric_cols[0]: 'close'})
                    else:
                        raise ValueError("No numeric price data found")
            
            prices = price_data['close'].dropna()
            if len(prices) < 20:
                return self._get_default_features()
            
            # Price-based features
            returns = prices.pct_change().dropna()
            
            # Trend features
            for window in self.config.lookback_periods:
                if len(prices) >= window:
                    # Trend strength
                    price_change = (prices.iloc[-1] - prices.iloc[-window]) / prices.iloc[-window]
                    features[f'trend_{window}'] = price_change
                    
                    # Linear regression slope
                    x = np.arange(window)
                    y = prices.iloc[-window:].values
                    slope = np.polyfit(x, y, 1)[0] / np.mean(y)  # Normalized slope
                    features[f'slope_{window}'] = slope
            
            # Volatility features
            if len(returns) >= self.config.volatility_window:
                vol_window = min(self.config.volatility_window, len(returns))
                volatility = returns.iloc[-vol_window:].std() * np.sqrt(252)  # Annualized
                features['volatility'] = volatility
                
                # Rolling volatility trend
                if len(returns) >= vol_window * 2:
                    recent_vol = returns.iloc[-vol_window:].std()
                    older_vol = returns.iloc[-vol_window*2:-vol_window].std()
                    vol_trend = (recent_vol - older_vol) / older_vol if older_vol > 0 else 0
                    features['volatility_trend'] = vol_trend
                
                # Volatility clustering (GARCH-like)
                abs_returns = np.abs(returns.iloc[-vol_window:])
                vol_clustering = abs_returns.autocorr(lag=1) if len(abs_returns) > 1 else 0
                features['volatility_clustering'] = vol_clustering or 0
            
            # Momentum features
            if len(returns) >= 20:
                # RSI-like momentum
                gains = returns[returns > 0].sum()
                losses = -returns[returns < 0].sum()
                rsi = 100 - (100 / (1 + gains / max(losses, 1e-8)))
                features['momentum_rsi'] = rsi
                
                # Price momentum
                momentum_periods = [5, 10, 20]
                for period in momentum_periods:
                    if len(returns) >= period:
                        momentum = returns.iloc[-period:].sum()
                        features[f'momentum_{period}'] = momentum
            
            # Volume features (if available)
            if volume_data is not None and not volume_data.empty:
                volume = volume_data.iloc[:, 0] if volume_data.shape[1] > 0 else volume_data
                if len(volume) >= self.config.volume_window:
                    vol_window = min(self.config.volume_window, len(volume))
                    
                    # Volume trend
                    recent_vol = volume.iloc[-vol_window//2:].mean()
                    older_vol = volume.iloc[-vol_window:-vol_window//2].mean()
                    vol_trend = (recent_vol - older_vol) / older_vol if older_vol > 0 else 0
                    features['volume_trend'] = vol_trend
                    
                    # Volume volatility
                    vol_vol = volume.iloc[-vol_window:].std() / volume.iloc[-vol_window:].mean()
                    features['volume_volatility'] = vol_vol or 0
            
            # Market structure features
            if len(prices) >= 50:
                # Support/resistance levels
                recent_prices = prices.iloc[-50:]
                price_range = recent_prices.max() - recent_prices.min()
                current_position = (prices.iloc[-1] - recent_prices.min()) / max(price_range, 1e-8)
                features['price_position'] = current_position
                
                # Drawdown
                rolling_max = prices.rolling(window=50, min_periods=1).max()
                drawdown = (prices - rolling_max) / rolling_max
                max_drawdown = drawdown.iloc[-50:].min()
                features['max_drawdown'] = max_drawdown
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Feature extraction failed: {e}")
            return self._get_default_features()
    
    def _get_default_features(self) -> Dict[str, float]:
        """Get default features when extraction fails"""
        return {
            'trend_20': 0.0,
            'trend_50': 0.0,
            'volatility': 0.5,
            'momentum_rsi': 50.0,
            'volume_trend': 0.0,
            'price_position': 0.5
        }
    
    def _detect_regime(self, features: Dict[str, float], price_data: pd.DataFrame) -> RegimeSignal:
        """Detect current market regime based on features"""
        try:
            # Rule-based regime detection
            regime = self._rule_based_regime_detection(features)
            
            # ML-based regime detection (if enabled and trained)
            if self.config.enable_ml_clustering and self.regime_clusterer is not None:
                ml_regime = self._ml_based_regime_detection(features)
                # Combine rule-based and ML-based detection
                regime = self._combine_regime_predictions(regime, ml_regime, features)
            
            # Detect volatility regime
            volatility_regime = self._detect_volatility_regime(features)
            
            # Calculate confidence
            confidence = self._calculate_regime_confidence(features, regime)
            
            # Create regime signal
            signal = RegimeSignal(
                regime=regime,
                volatility_regime=volatility_regime,
                confidence=confidence,
                timestamp=datetime.now(),
                indicators=features.copy(),
                trend_strength=self._calculate_trend_strength(features),
                volatility_percentile=self._calculate_volatility_percentile(features.get('volatility', 0.5)),
                volume_profile=self._analyze_volume_profile(features)
            )
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return RegimeSignal(
                regime=MarketRegime.UNKNOWN,
                volatility_regime=VolatilityRegime.MEDIUM,
                confidence=0.0,
                timestamp=datetime.now()
            )
    
    def _rule_based_regime_detection(self, features: Dict[str, float]) -> MarketRegime:
        """Rule-based regime detection using market indicators"""
        trend_20 = features.get('trend_20', 0)
        trend_50 = features.get('trend_50', 0)
        volatility = features.get('volatility', 0.5)
        momentum_rsi = features.get('momentum_rsi', 50)
        max_drawdown = features.get('max_drawdown', 0)
        volume_trend = features.get('volume_trend', 0)
        
        # Crash detection
        if trend_20 < -0.15 and volatility > 0.8 and max_drawdown < -0.2:
            return MarketRegime.CRASH
        
        # Bubble detection
        if trend_20 > 0.3 and volatility > 0.6 and momentum_rsi > 80:
            return MarketRegime.BUBBLE
        
        # Bull trending
        if trend_20 > 0.05 and trend_50 > 0.02 and momentum_rsi > 55:
            return MarketRegime.BULL_TRENDING
        
        # Bear trending
        if trend_20 < -0.05 and trend_50 < -0.02 and momentum_rsi < 45:
            return MarketRegime.BEAR_TRENDING
        
        # Recovery
        if trend_20 > 0.1 and max_drawdown < -0.1 and volume_trend > 0.1:
            return MarketRegime.RECOVERY
        
        # Accumulation
        if abs(trend_20) < 0.03 and volatility < 0.4 and volume_trend > 0:
            return MarketRegime.ACCUMULATION
        
        # Distribution
        if abs(trend_20) < 0.03 and volatility < 0.4 and volume_trend < -0.05:
            return MarketRegime.DISTRIBUTION
        
        # Sideways markets
        if abs(trend_20) < 0.05 and abs(trend_50) < 0.03:
            if volatility < 0.4:
                return MarketRegime.SIDEWAYS_LOW_VOL
            else:
                return MarketRegime.SIDEWAYS_HIGH_VOL
        
        return MarketRegime.UNKNOWN
    
    def _ml_based_regime_detection(self, features: Dict[str, float]) -> MarketRegime:
        """ML-based regime detection using clustering"""
        try:
            if self.regime_clusterer is None:
                return MarketRegime.UNKNOWN
            
            # Prepare feature vector
            feature_names = ['trend_20', 'trend_50', 'volatility', 'momentum_rsi', 'volume_trend', 'max_drawdown']
            feature_vector = np.array([features.get(name, 0) for name in feature_names]).reshape(1, -1)
            
            # Scale features
            scaled_features = self.volatility_scaler.transform(feature_vector)
            
            # Predict cluster
            cluster = self.regime_clusterer.predict(scaled_features)[0]
            
            # Map cluster to regime (this mapping would be learned during training)
            cluster_to_regime = {
                0: MarketRegime.BULL_TRENDING,
                1: MarketRegime.BEAR_TRENDING,
                2: MarketRegime.SIDEWAYS_LOW_VOL,
                3: MarketRegime.SIDEWAYS_HIGH_VOL,
                4: MarketRegime.CRASH,
                5: MarketRegime.RECOVERY
            }
            
            return cluster_to_regime.get(cluster, MarketRegime.UNKNOWN)
            
        except Exception as e:
            self.logger.warning(f"ML regime detection failed: {e}")
            return MarketRegime.UNKNOWN
    
    def _combine_regime_predictions(self, rule_regime: MarketRegime, ml_regime: MarketRegime, features: Dict[str, float]) -> MarketRegime:
        """Combine rule-based and ML-based regime predictions"""
        # If both agree, high confidence
        if rule_regime == ml_regime:
            return rule_regime
        
        # Priority rules for critical regimes
        if rule_regime == MarketRegime.CRASH or ml_regime == MarketRegime.CRASH:
            # Crash detection requires high confidence
            volatility = features.get('volatility', 0.5)
            if volatility > 0.8:
                return MarketRegime.CRASH
        
        if rule_regime == MarketRegime.BUBBLE or ml_regime == MarketRegime.BUBBLE:
            # Bubble detection
            trend_20 = features.get('trend_20', 0)
            if trend_20 > 0.2:
                return MarketRegime.BUBBLE
        
        # Default to rule-based for disagreement
        return rule_regime
    
    def _detect_volatility_regime(self, features: Dict[str, float]) -> VolatilityRegime:
        """Detect current volatility regime"""
        volatility = features.get('volatility', 0.5)
        
        if volatility < 0.1:
            return VolatilityRegime.ULTRA_LOW
        elif volatility < 0.3:
            return VolatilityRegime.LOW
        elif volatility < 0.6:
            return VolatilityRegime.MEDIUM
        elif volatility < 1.0:
            return VolatilityRegime.HIGH
        else:
            return VolatilityRegime.EXTREME
    
    def _calculate_regime_confidence(self, features: Dict[str, float], regime: MarketRegime) -> float:
        """Calculate confidence in regime detection"""
        try:
            # Base confidence on feature consistency
            trend_20 = features.get('trend_20', 0)
            trend_50 = features.get('trend_50', 0)
            volatility = features.get('volatility', 0.5)
            momentum_rsi = features.get('momentum_rsi', 50)
            
            confidence = 0.5  # Base confidence
            
            # Trend consistency
            if regime in [MarketRegime.BULL_TRENDING, MarketRegime.RECOVERY]:
                if trend_20 > 0 and trend_50 > 0:
                    confidence += 0.2
                if momentum_rsi > 50:
                    confidence += 0.1
            elif regime in [MarketRegime.BEAR_TRENDING, MarketRegime.CRASH]:
                if trend_20 < 0 and trend_50 < 0:
                    confidence += 0.2
                if momentum_rsi < 50:
                    confidence += 0.1
            
            # Volatility consistency
            if regime == MarketRegime.CRASH and volatility > 0.8:
                confidence += 0.2
            elif regime == MarketRegime.SIDEWAYS_LOW_VOL and volatility < 0.3:
                confidence += 0.2
            
            # Duration consistency (longer duration = higher confidence)
            if self.regime_duration > 10:
                confidence += min(0.2, self.regime_duration / 100)
            
            return min(1.0, max(0.0, confidence))
            
        except Exception:
            return 0.5
    
    def _calculate_trend_strength(self, features: Dict[str, float]) -> float:
        """Calculate overall trend strength"""
        trend_20 = features.get('trend_20', 0)
        trend_50 = features.get('trend_50', 0)
        momentum_rsi = features.get('momentum_rsi', 50)
        
        # Combine multiple trend indicators
        trend_strength = (abs(trend_20) + abs(trend_50)) / 2
        momentum_strength = abs(momentum_rsi - 50) / 50
        
        return (trend_strength + momentum_strength) / 2
    
    def _calculate_volatility_percentile(self, current_volatility: float) -> float:
        """Calculate volatility percentile compared to history"""
        if len(self.volatility_history) < 20:
            return 0.5
        
        percentile = np.percentile(self.volatility_history, 
                                  100 * np.sum(np.array(self.volatility_history) <= current_volatility) / len(self.volatility_history))
        return percentile / 100
    
    def _analyze_volume_profile(self, features: Dict[str, float]) -> str:
        """Analyze volume profile"""
        volume_trend = features.get('volume_trend', 0)
        volume_volatility = features.get('volume_volatility', 0)
        
        if volume_trend > 0.1:
            return "increasing"
        elif volume_trend < -0.1:
            return "decreasing"
        elif volume_volatility > 1.0:
            return "volatile"
        else:
            return "normal"
    
    def _handle_regime_change(self, new_signal: RegimeSignal):
        """Handle regime change and notify callbacks"""
        old_regime = self.current_regime
        self.current_regime = new_signal.regime
        self.current_volatility_regime = new_signal.volatility_regime
        self.regime_start_time = datetime.now()
        self.regime_duration = 0
        
        self.logger.info(f"Regime change detected: {old_regime.value} -> {new_signal.regime.value} (confidence: {new_signal.confidence:.2f})")
        
        # Notify callbacks
        for callback in self.regime_change_callbacks:
            try:
                callback(new_signal)
            except Exception as e:
                self.logger.error(f"Regime change callback failed: {e}")
    
    def _update_historical_cache(self, price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame]):
        """Update historical data cache"""
        try:
            if not price_data.empty:
                prices = price_data['close'] if 'close' in price_data.columns else price_data.iloc[:, 0]
                self.price_history.extend(prices.tolist())
                
                # Calculate volatility
                if len(prices) > 1:
                    returns = prices.pct_change().dropna()
                    if len(returns) > 0:
                        volatility = returns.std() * np.sqrt(252)
                        self.volatility_history.append(volatility)
            
            if volume_data is not None and not volume_data.empty:
                volumes = volume_data.iloc[:, 0] if volume_data.shape[1] > 0 else volume_data
                self.volume_history.extend(volumes.tolist())
            
            # Limit cache size
            max_history = 5000
            if len(self.price_history) > max_history:
                self.price_history = self.price_history[-max_history:]
            if len(self.volume_history) > max_history:
                self.volume_history = self.volume_history[-max_history:]
            if len(self.volatility_history) > max_history:
                self.volatility_history = self.volatility_history[-max_history:]
            
            # Update regime duration
            self.regime_duration += 1
            
        except Exception as e:
            self.logger.warning(f"Historical cache update failed: {e}")
    
    def _get_current_regime_signal(self) -> RegimeSignal:
        """Get current regime signal"""
        return RegimeSignal(
            regime=self.current_regime,
            volatility_regime=self.current_volatility_regime,
            confidence=0.5,
            timestamp=datetime.now(),
            regime_duration=self.regime_duration
        )
    
    def train_ml_models(self, historical_data: pd.DataFrame) -> bool:
        """Train ML models for regime detection"""
        try:
            if historical_data.empty or len(historical_data) < 100:
                self.logger.warning("Insufficient data for ML model training")
                return False
            
            # Extract features for training
            features_list = []
            for i in range(50, len(historical_data) - 50, 10):  # Sample every 10 periods
                window_data = historical_data.iloc[i-50:i+50]
                features = self._extract_market_features(window_data, None)
                if features:
                    features_list.append([
                        features.get('trend_20', 0),
                        features.get('trend_50', 0),
                        features.get('volatility', 0.5),
                        features.get('momentum_rsi', 50),
                        features.get('volume_trend', 0),
                        features.get('max_drawdown', 0)
                    ])
            
            if len(features_list) < 50:
                return False
            
            X = np.array(features_list)
            
            # Scale features
            X_scaled = self.volatility_scaler.fit_transform(X)
            
            # Train clustering model
            self.regime_clusterer = GaussianMixture(
                n_components=self.config.n_regime_clusters,
                random_state=42
            )
            self.regime_clusterer.fit(X_scaled)
            
            self.logger.info(f"ML regime detection models trained on {len(features_list)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"ML model training failed: {e}")
            return False
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get regime detection statistics"""
        with self._lock:
            stats = {
                'current_regime': self.current_regime.value,
                'current_volatility_regime': self.current_volatility_regime.value,
                'regime_duration': self.regime_duration,
                'regime_start_time': self.regime_start_time.isoformat(),
                'total_regime_changes': len(self.regime_history),
                'ml_models_trained': self.regime_clusterer is not None,
                'historical_data_points': {
                    'prices': len(self.price_history),
                    'volumes': len(self.volume_history),
                    'volatilities': len(self.volatility_history)
                }
            }
            
            # Regime distribution in history
            if self.regime_history:
                regime_counts = {}
                for signal in self.regime_history[-100:]:  # Last 100 signals
                    regime_counts[signal.regime.value] = regime_counts.get(signal.regime.value, 0) + 1
                stats['recent_regime_distribution'] = regime_counts
            
            return stats
    
    def get_regime_recommendations(self) -> Dict[str, Any]:
        """Get recommendations based on current regime"""
        current_signal = self._get_current_regime_signal()
        
        recommendations = {
            'regime': current_signal.regime.value,
            'volatility_regime': current_signal.volatility_regime.value,
            'confidence': current_signal.confidence,
            'recommended_actions': [],
            'risk_adjustments': {},
            'model_preferences': {}
        }
        
        # Regime-specific recommendations
        if current_signal.regime == MarketRegime.BULL_TRENDING:
            recommendations['recommended_actions'] = ['increase_long_exposure', 'momentum_strategies']
            recommendations['risk_adjustments'] = {'position_size': 1.2, 'stop_loss': 0.95}
            recommendations['model_preferences'] = {'trend_following': 1.5, 'mean_reversion': 0.5}
            
        elif current_signal.regime == MarketRegime.BEAR_TRENDING:
            recommendations['recommended_actions'] = ['reduce_exposure', 'defensive_strategies']
            recommendations['risk_adjustments'] = {'position_size': 0.6, 'stop_loss': 0.98}
            recommendations['model_preferences'] = {'trend_following': 1.3, 'momentum': 0.7}
            
        elif current_signal.regime == MarketRegime.CRASH:
            recommendations['recommended_actions'] = ['emergency_exit', 'cash_preservation']
            recommendations['risk_adjustments'] = {'position_size': 0.2, 'stop_loss': 0.99}
            recommendations['model_preferences'] = {'all_models': 0.1}  # Minimal exposure
            
        elif current_signal.regime == MarketRegime.SIDEWAYS_LOW_VOL:
            recommendations['recommended_actions'] = ['mean_reversion', 'range_trading']
            recommendations['risk_adjustments'] = {'position_size': 1.0, 'stop_loss': 0.96}
            recommendations['model_preferences'] = {'mean_reversion': 1.5, 'trend_following': 0.7}
            
        elif current_signal.regime == MarketRegime.SIDEWAYS_HIGH_VOL:
            recommendations['recommended_actions'] = ['volatility_strategies', 'breakout_preparation']
            recommendations['risk_adjustments'] = {'position_size': 0.8, 'stop_loss': 0.97}
            recommendations['model_preferences'] = {'volatility_based': 1.3, 'breakout': 1.2}
        
        return recommendations


# Singleton regime detector
_regime_detector = None
_detector_lock = threading.Lock()

def get_market_regime_detector(config: Optional[RegimeDetectionConfig] = None) -> MarketRegimeDetector:
    """Get the singleton market regime detector"""
    global _regime_detector
    
    with _detector_lock:
        if _regime_detector is None:
            _regime_detector = MarketRegimeDetector(config)
        return _regime_detector

def detect_market_regime(price_data: pd.DataFrame, volume_data: Optional[pd.DataFrame] = None) -> RegimeSignal:
    """Convenient function to detect market regime"""
    detector = get_market_regime_detector()
    return detector.update_market_data(price_data, volume_data)