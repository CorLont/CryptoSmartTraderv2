#!/usr/bin/env python3
"""
Market Regime Detection
Hidden Markov Models and rule-based classification for dynamic market regime identification
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone, timedelta
import warnings
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

class MarketRegime(Enum):
    """Market regime types"""
    BULL_MARKET = "bull_market"
    BEAR_MARKET = "bear_market"
    CONSOLIDATION = "consolidation"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TREND_REVERSAL = "trend_reversal"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"

@dataclass
class RegimeState:
    """Current market regime state"""
    regime: MarketRegime
    confidence: float
    volatility_level: float
    trend_strength: float
    momentum: float
    regime_duration_days: int
    transition_probability: float
    supporting_indicators: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())

@dataclass
class RegimeTransition:
    """Market regime transition"""
    from_regime: MarketRegime
    to_regime: MarketRegime
    transition_date: datetime
    confidence: float
    trigger_indicators: List[str]
    market_impact_score: float

class HMMRegimeDetector:
    """Hidden Markov Model for regime detection"""
    
    def __init__(self, n_regimes: int = 4, random_state: int = 42):
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.regime_mapping = {}
        self.feature_names = []
        
        self.logger = logging.getLogger(__name__)
    
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare regime detection features"""
        
        features = pd.DataFrame(index=data.index)
        
        # Price-based features
        if 'close' in data.columns:
            # Returns and volatility
            features['returns'] = data['close'].pct_change()
            features['volatility'] = features['returns'].rolling(20).std()
            features['log_volatility'] = np.log(features['volatility'] + 1e-8)
            
            # Trend features
            features['sma_20'] = data['close'].rolling(20).mean()
            features['sma_50'] = data['close'].rolling(50).mean()
            features['price_vs_sma20'] = (data['close'] - features['sma_20']) / features['sma_20']
            features['price_vs_sma50'] = (data['close'] - features['sma_50']) / features['sma_50']
            features['sma_slope'] = features['sma_20'].pct_change(5)
            
            # Momentum features
            features['rsi'] = self._calculate_rsi(data['close'], 14)
            features['macd'], features['macd_signal'], _ = self._calculate_macd(data['close'])
            features['macd_histogram'] = features['macd'] - features['macd_signal']
            
            # Volatility regime features
            features['vol_regime'] = (features['volatility'] / features['volatility'].rolling(100).mean()) - 1
            features['vol_zscore'] = (features['volatility'] - features['volatility'].rolling(50).mean()) / features['volatility'].rolling(50).std()
            
        # Volume features
        if 'volume' in data.columns:
            features['volume_sma'] = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma']
            features['volume_trend'] = features['volume_sma'].pct_change(5)
        
        # Market structure features
        if all(col in data.columns for col in ['high', 'low', 'close']):
            features['true_range'] = self._calculate_true_range(data)
            features['atr'] = features['true_range'].rolling(14).mean()
            features['atr_regime'] = (features['atr'] / features['atr'].rolling(50).mean()) - 1
        
        # Cross-asset correlation proxy (use price correlation as proxy)
        if 'close' in data.columns:
            features['price_autocorr'] = features['returns'].rolling(20).apply(
                lambda x: x.autocorr(lag=1) if len(x.dropna()) > 10 else 0
            )
        
        # Clean features
        features = features.dropna()
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        return features.values
    
    def fit(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fit HMM regime detector"""
        
        # Prepare features
        X = self.prepare_features(data)
        
        if len(X) < 100:
            raise ValueError("Insufficient data for regime detection (need at least 100 observations)")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            random_state=self.random_state,
            n_iter=100
        )
        
        self.model.fit(X_scaled)
        
        # Get regime predictions
        regime_states = self.model.predict(X_scaled)
        regime_probs = self.model.predict_proba(X_scaled)
        
        # Interpret regimes based on characteristics
        self.regime_mapping = self._interpret_regimes(X, regime_states)
        
        self.is_fitted = True
        
        # Calculate fit metrics
        log_likelihood = self.model.score(X_scaled)
        
        return {
            'log_likelihood': log_likelihood,
            'n_regimes': self.n_regimes,
            'regime_mapping': self.regime_mapping,
            'transition_matrix': self.model.transmat_.tolist(),
            'regime_probabilities': regime_probs[-10:].tolist()  # Last 10 observations
        }
    
    def predict(self, data: pd.DataFrame) -> Tuple[List[MarketRegime], np.ndarray]:
        """Predict market regimes"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare features
        X = self.prepare_features(data)
        X_scaled = self.scaler.transform(X)
        
        # Get regime predictions
        regime_states = self.model.predict(X_scaled)
        regime_probs = self.model.predict_proba(X_scaled)
        
        # Map to regime types
        regimes = [self.regime_mapping.get(state, MarketRegime.UNKNOWN) for state in regime_states]
        
        return regimes, regime_probs
    
    def _interpret_regimes(self, X: np.ndarray, regime_states: np.ndarray) -> Dict[int, MarketRegime]:
        """Interpret HMM states as market regimes"""
        
        regime_characteristics = {}
        
        for regime_id in range(self.n_regimes):
            mask = regime_states == regime_id
            regime_data = X[mask]
            
            if len(regime_data) == 0:
                continue
            
            # Calculate regime characteristics
            characteristics = {}
            
            for i, feature_name in enumerate(self.feature_names):
                if i < regime_data.shape[1]:
                    characteristics[feature_name] = np.mean(regime_data[:, i])
            
            regime_characteristics[regime_id] = characteristics
        
        # Map regimes based on characteristics
        regime_mapping = {}
        
        for regime_id, chars in regime_characteristics.items():
            # Bull market: positive returns, low volatility, strong trend
            if (chars.get('returns', 0) > 0.001 and
                chars.get('volatility', 0) < np.median([c.get('volatility', 0) for c in regime_characteristics.values()]) and
                chars.get('price_vs_sma20', 0) > 0):
                regime_mapping[regime_id] = MarketRegime.BULL_MARKET
            
            # Bear market: negative returns, increasing volatility
            elif (chars.get('returns', 0) < -0.001 and
                  chars.get('price_vs_sma20', 0) < 0):
                regime_mapping[regime_id] = MarketRegime.BEAR_MARKET
            
            # High volatility: high volatility regardless of direction
            elif chars.get('volatility', 0) > np.percentile([c.get('volatility', 0) for c in regime_characteristics.values()], 75):
                regime_mapping[regime_id] = MarketRegime.HIGH_VOLATILITY
            
            # Low volatility: low volatility, sideways movement
            elif chars.get('volatility', 0) < np.percentile([c.get('volatility', 0) for c in regime_characteristics.values()], 25):
                regime_mapping[regime_id] = MarketRegime.LOW_VOLATILITY
            
            # Default to consolidation
            else:
                regime_mapping[regime_id] = MarketRegime.CONSOLIDATION
        
        return regime_mapping
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(
        self, 
        prices: pd.Series, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""
        
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        return macd, macd_signal, macd_histogram
    
    def _calculate_true_range(self, data: pd.DataFrame) -> pd.Series:
        """Calculate True Range"""
        
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift(1))
        low_close = np.abs(data['low'] - data['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        return pd.Series(true_range, index=data.index)

class RuleBasedRegimeDetector:
    """Rule-based regime detector for real-time classification"""
    
    def __init__(self):
        self.volatility_threshold_high = 0.03  # 3% daily volatility
        self.volatility_threshold_low = 0.01   # 1% daily volatility
        self.trend_strength_threshold = 0.02   # 2% trend strength
        self.momentum_threshold = 0.15         # RSI-based momentum
        
        self.logger = logging.getLogger(__name__)
    
    def detect_regime(self, data: pd.DataFrame, lookback_days: int = 30) -> RegimeState:
        """Detect current market regime using rules"""
        
        if len(data) < lookback_days:
            return RegimeState(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                volatility_level=0.0,
                trend_strength=0.0,
                momentum=0.0,
                regime_duration_days=0,
                transition_probability=0.0
            )
        
        # Calculate indicators
        recent_data = data.tail(lookback_days)
        
        # Volatility
        returns = recent_data['close'].pct_change().dropna()
        current_volatility = returns.std()
        
        # Trend strength
        sma_short = recent_data['close'].rolling(10).mean()
        sma_long = recent_data['close'].rolling(20).mean()
        trend_strength = (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
        
        # Momentum (RSI)
        rsi = self._calculate_rsi(recent_data['close'])
        current_rsi = rsi.iloc[-1]
        momentum = (current_rsi - 50) / 50  # Normalize to [-1, 1]
        
        # Price position relative to moving averages
        current_price = recent_data['close'].iloc[-1]
        sma_20 = recent_data['close'].rolling(20).mean().iloc[-1]
        price_position = (current_price - sma_20) / sma_20
        
        # Regime classification logic
        regime, confidence = self._classify_regime(
            current_volatility, trend_strength, momentum, price_position
        )
        
        # Supporting indicators
        supporting_indicators = {
            'volatility': current_volatility,
            'trend_strength': trend_strength,
            'momentum': momentum,
            'price_position': price_position,
            'rsi': current_rsi
        }
        
        # Estimate regime duration (simplified)
        regime_duration = self._estimate_regime_duration(data, regime)
        
        # Transition probability (simplified)
        transition_prob = self._estimate_transition_probability(
            current_volatility, trend_strength, momentum
        )
        
        return RegimeState(
            regime=regime,
            confidence=confidence,
            volatility_level=current_volatility,
            trend_strength=abs(trend_strength),
            momentum=momentum,
            regime_duration_days=regime_duration,
            transition_probability=transition_prob,
            supporting_indicators=supporting_indicators
        )
    
    def _classify_regime(
        self, 
        volatility: float, 
        trend_strength: float, 
        momentum: float,
        price_position: float
    ) -> Tuple[MarketRegime, float]:
        """Classify regime based on indicators"""
        
        # High volatility regime
        if volatility > self.volatility_threshold_high:
            if abs(trend_strength) > self.trend_strength_threshold:
                if trend_strength > 0:
                    return MarketRegime.BULL_MARKET, 0.8
                else:
                    return MarketRegime.BEAR_MARKET, 0.8
            else:
                return MarketRegime.HIGH_VOLATILITY, 0.7
        
        # Low volatility regime
        elif volatility < self.volatility_threshold_low:
            if abs(trend_strength) < self.trend_strength_threshold * 0.5:
                return MarketRegime.LOW_VOLATILITY, 0.7
            else:
                return MarketRegime.CONSOLIDATION, 0.6
        
        # Medium volatility - check trend and momentum
        else:
            # Strong uptrend
            if (trend_strength > self.trend_strength_threshold and 
                momentum > self.momentum_threshold and
                price_position > 0.02):
                return MarketRegime.BULL_MARKET, 0.7
            
            # Strong downtrend
            elif (trend_strength < -self.trend_strength_threshold and
                  momentum < -self.momentum_threshold and
                  price_position < -0.02):
                return MarketRegime.BEAR_MARKET, 0.7
            
            # Potential reversal
            elif abs(momentum) > 0.3:  # Extreme RSI
                return MarketRegime.TREND_REVERSAL, 0.6
            
            # Default to consolidation
            else:
                return MarketRegime.CONSOLIDATION, 0.5
    
    def _estimate_regime_duration(self, data: pd.DataFrame, current_regime: MarketRegime) -> int:
        """Estimate how long current regime has been active"""
        
        # Simplified: look for regime consistency over recent periods
        if len(data) < 10:
            return 1
        
        consistent_days = 1
        
        # Check recent 30 days for regime consistency
        for i in range(min(30, len(data) - 1)):
            recent_subset = data.iloc[-(i+2):-(i+1) if i > 0 else None]
            if len(recent_subset) > 10:
                temp_regime = self.detect_regime(recent_subset, lookback_days=min(10, len(recent_subset)))
                if temp_regime.regime == current_regime:
                    consistent_days += 1
                else:
                    break
        
        return consistent_days
    
    def _estimate_transition_probability(
        self, 
        volatility: float, 
        trend_strength: float, 
        momentum: float
    ) -> float:
        """Estimate probability of regime transition"""
        
        # Higher transition probability when:
        # 1. Extreme momentum (RSI overbought/oversold)
        # 2. Volatility spikes
        # 3. Trend weakening
        
        transition_factors = []
        
        # Momentum extremes
        if abs(momentum) > 0.4:
            transition_factors.append(abs(momentum))
        
        # Volatility spikes
        if volatility > self.volatility_threshold_high:
            transition_factors.append(min(volatility / self.volatility_threshold_high, 1.0))
        
        # Trend weakening (if very weak trend)
        if abs(trend_strength) < self.trend_strength_threshold * 0.3:
            transition_factors.append(0.3)
        
        if transition_factors:
            return min(np.mean(transition_factors), 0.9)
        else:
            return 0.1  # Base transition probability
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

class MarketRegimeDetector:
    """Comprehensive market regime detection system"""
    
    def __init__(
        self,
        use_hmm: bool = True,
        use_rules: bool = True,
        hmm_regimes: int = 4,
        update_frequency_days: int = 7
    ):
        self.use_hmm = use_hmm
        self.use_rules = use_rules
        self.hmm_regimes = hmm_regimes
        self.update_frequency_days = update_frequency_days
        
        # Initialize detectors
        self.hmm_detector = HMMRegimeDetector(n_regimes=hmm_regimes) if use_hmm else None
        self.rule_detector = RuleBasedRegimeDetector() if use_rules else None
        
        # State tracking
        self.last_hmm_update = None
        self.regime_history = []
        self.transition_history = []
        
        self.logger = logging.getLogger(__name__)
    
    def fit_historical_regimes(self, historical_data: pd.DataFrame) -> Dict[str, Any]:
        """Fit regime detector on historical data"""
        
        results = {}
        
        if self.hmm_detector:
            try:
                hmm_results = self.hmm_detector.fit(historical_data)
                results['hmm'] = hmm_results
                self.last_hmm_update = datetime.utcnow()
                self.logger.info("HMM regime detector fitted successfully")
            except Exception as e:
                self.logger.error(f"HMM fitting failed: {e}")
                results['hmm'] = {'error': str(e)}
        
        return results
    
    def detect_current_regime(self, data: pd.DataFrame) -> RegimeState:
        """Detect current market regime"""
        
        regime_states = []
        
        # Rule-based detection (always available)
        if self.rule_detector:
            rule_regime = self.rule_detector.detect_regime(data)
            regime_states.append(('rule_based', rule_regime))
        
        # HMM-based detection (if fitted)
        if self.hmm_detector and self.hmm_detector.is_fitted:
            try:
                regimes, probs = self.hmm_detector.predict(data)
                if regimes:
                    # Get most recent regime
                    recent_regime = regimes[-1]
                    recent_prob = probs[-1] if len(probs) > 0 else np.array([0.5] * self.hmm_regimes)
                    confidence = np.max(recent_prob)
                    
                    # Create HMM regime state
                    hmm_regime_state = RegimeState(
                        regime=recent_regime,
                        confidence=confidence,
                        volatility_level=rule_regime.volatility_level if rule_regime else 0.0,
                        trend_strength=rule_regime.trend_strength if rule_regime else 0.0,
                        momentum=rule_regime.momentum if rule_regime else 0.0,
                        regime_duration_days=0,  # Would need more complex calculation
                        transition_probability=1.0 - confidence
                    )
                    
                    regime_states.append(('hmm', hmm_regime_state))
            except Exception as e:
                self.logger.error(f"HMM prediction failed: {e}")
        
        # Combine regime predictions
        if not regime_states:
            return RegimeState(
                regime=MarketRegime.UNKNOWN,
                confidence=0.0,
                volatility_level=0.0,
                trend_strength=0.0,
                momentum=0.0,
                regime_duration_days=0,
                transition_probability=1.0
            )
        
        # If multiple detectors, use ensemble
        if len(regime_states) > 1:
            combined_regime = self._combine_regime_predictions(regime_states)
        else:
            combined_regime = regime_states[0][1]
        
        # Store in history
        self.regime_history.append({
            'timestamp': datetime.utcnow(),
            'regime': combined_regime.regime,
            'confidence': combined_regime.confidence,
            'detection_methods': [method for method, _ in regime_states]
        })
        
        # Check for regime transitions
        if len(self.regime_history) > 1:
            self._check_regime_transition(self.regime_history[-2], self.regime_history[-1])
        
        return combined_regime
    
    def _combine_regime_predictions(self, regime_states: List[Tuple[str, RegimeState]]) -> RegimeState:
        """Combine predictions from multiple detectors"""
        
        # Weight different methods
        method_weights = {
            'rule_based': 0.6,  # More weight to rule-based for real-time
            'hmm': 0.4          # HMM for historical patterns
        }
        
        # Calculate weighted regime probabilities
        regime_scores = {}
        total_weight = 0
        
        for method, regime_state in regime_states:
            weight = method_weights.get(method, 0.5)
            regime = regime_state.regime
            confidence = regime_state.confidence
            
            if regime not in regime_scores:
                regime_scores[regime] = 0
            
            regime_scores[regime] += weight * confidence
            total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            for regime in regime_scores:
                regime_scores[regime] /= total_weight
        
        # Select best regime
        best_regime = max(regime_scores, key=regime_scores.get)
        best_confidence = regime_scores[best_regime]
        
        # Use rule-based state as template and update regime/confidence
        template_state = None
        for method, regime_state in regime_states:
            if method == 'rule_based':
                template_state = regime_state
                break
        
        if template_state is None:
            template_state = regime_states[0][1]
        
        # Create combined state
        combined_state = RegimeState(
            regime=best_regime,
            confidence=best_confidence,
            volatility_level=template_state.volatility_level,
            trend_strength=template_state.trend_strength,
            momentum=template_state.momentum,
            regime_duration_days=template_state.regime_duration_days,
            transition_probability=1.0 - best_confidence,
            supporting_indicators=template_state.supporting_indicators
        )
        
        return combined_state
    
    def _check_regime_transition(self, prev_regime: Dict, curr_regime: Dict):
        """Check for regime transitions and record them"""
        
        if prev_regime['regime'] != curr_regime['regime']:
            transition = RegimeTransition(
                from_regime=prev_regime['regime'],
                to_regime=curr_regime['regime'],
                transition_date=curr_regime['timestamp'],
                confidence=(prev_regime['confidence'] + curr_regime['confidence']) / 2,
                trigger_indicators=['regime_change_detected'],
                market_impact_score=self._calculate_transition_impact(
                    prev_regime['regime'], curr_regime['regime']
                )
            )
            
            self.transition_history.append(transition)
            
            self.logger.info(f"Regime transition detected: {prev_regime['regime'].value} → {curr_regime['regime'].value}")
    
    def _calculate_transition_impact(self, from_regime: MarketRegime, to_regime: MarketRegime) -> float:
        """Calculate market impact score of regime transition"""
        
        # Define impact scores for different transitions
        impact_matrix = {
            (MarketRegime.BULL_MARKET, MarketRegime.BEAR_MARKET): 0.9,
            (MarketRegime.BEAR_MARKET, MarketRegime.BULL_MARKET): 0.9,
            (MarketRegime.LOW_VOLATILITY, MarketRegime.HIGH_VOLATILITY): 0.8,
            (MarketRegime.HIGH_VOLATILITY, MarketRegime.LOW_VOLATILITY): 0.7,
            (MarketRegime.CONSOLIDATION, MarketRegime.BULL_MARKET): 0.6,
            (MarketRegime.CONSOLIDATION, MarketRegime.BEAR_MARKET): 0.6,
        }
        
        return impact_matrix.get((from_regime, to_regime), 0.5)
    
    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get comprehensive regime statistics"""
        
        if not self.regime_history:
            return {'error': 'No regime history available'}
        
        recent_regimes = [r['regime'] for r in self.regime_history[-30:]]  # Last 30 observations
        
        # Regime distribution
        regime_counts = {}
        for regime in recent_regimes:
            regime_counts[regime.value] = regime_counts.get(regime.value, 0) + 1
        
        # Current regime info
        current_regime = self.regime_history[-1]
        
        # Transition frequency
        transition_count = len(self.transition_history)
        avg_regime_duration = len(self.regime_history) / max(transition_count, 1)
        
        return {
            'current_regime': current_regime['regime'].value,
            'current_confidence': current_regime['confidence'],
            'regime_distribution': regime_counts,
            'total_transitions': transition_count,
            'average_regime_duration': avg_regime_duration,
            'recent_transitions': [
                {
                    'from': t.from_regime.value,
                    'to': t.to_regime.value,
                    'date': t.transition_date.isoformat(),
                    'impact': t.market_impact_score
                }
                for t in self.transition_history[-5:]  # Last 5 transitions
            ]
        }

def create_regime_detector(
    use_hmm: bool = True,
    use_rules: bool = True,
    hmm_regimes: int = 4
) -> MarketRegimeDetector:
    """Create configured market regime detector"""
    
    return MarketRegimeDetector(
        use_hmm=use_hmm,
        use_rules=use_rules,
        hmm_regimes=hmm_regimes
    )

def detect_market_regime(
    data: pd.DataFrame,
    method: str = "combined"
) -> RegimeState:
    """High-level function to detect market regime"""
    
    if method == "rules":
        detector = RuleBasedRegimeDetector()
        return detector.detect_regime(data)
    
    elif method == "hmm":
        detector = HMMRegimeDetector()
        detector.fit(data)
        regimes, probs = detector.predict(data)
        
        if regimes:
            recent_regime = regimes[-1]
            recent_prob = probs[-1] if len(probs) > 0 else np.array([0.5] * 4)
            confidence = np.max(recent_prob)
            
            return RegimeState(
                regime=recent_regime,
                confidence=confidence,
                volatility_level=0.0,
                trend_strength=0.0,
                momentum=0.0,
                regime_duration_days=0,
                transition_probability=1.0 - confidence
            )
    
    else:  # combined
        detector = create_regime_detector()
        detector.fit_historical_regimes(data)
        return detector.detect_current_regime(data)