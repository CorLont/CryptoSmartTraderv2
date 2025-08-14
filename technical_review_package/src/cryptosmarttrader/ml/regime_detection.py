"""
Advanced Regime Detection System

Multi-factor regime classification system identifying market phases
(trend, mean-reversion, chop) with adaptive strategy switching.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Try to import talib, fallback to manual calculations if not available
try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger.warning("talib not available, using fallback calculations")


class MarketRegime(Enum):
    """Market regime types"""

    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    MEAN_REVERSION = "mean_reversion"
    CHOP = "chop"
    BREAKOUT = "breakout"
    VOLATILITY_SPIKE = "volatility_spike"


class RegimeConfidence(Enum):
    """Confidence levels for regime classification"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class RegimeFeatures:
    """Market regime feature set"""

    symbol: str
    timestamp: datetime

    # Trend indicators
    adx: float = 0.0
    adx_trend: float = 0.0
    ema_slope_short: float = 0.0
    ema_slope_long: float = 0.0
    price_momentum: float = 0.0

    # Mean reversion indicators
    rsi: float = 50.0
    bollinger_position: float = 0.5
    price_distance_from_mean: float = 0.0

    # Volatility indicators
    atr_normalized: float = 0.0
    volatility_ratio: float = 1.0
    volume_profile: float = 1.0

    # Market structure
    support_resistance_strength: float = 0.0
    breakout_probability: float = 0.0
    liquidity_score: float = 1.0

    # Cross-market factors
    correlation_strength: float = 0.0
    market_breadth: float = 0.0


@dataclass
class RegimeClassification:
    """Regime classification result"""

    regime: MarketRegime
    confidence: RegimeConfidence
    probability: float

    # Supporting evidence
    features: RegimeFeatures
    regime_probabilities: Dict[str, float] = field(default_factory=dict)

    # Duration tracking
    regime_start: Optional[datetime] = None
    regime_duration_minutes: int = 0

    @property
    def is_trending(self) -> bool:
        return self.regime in [MarketRegime.TREND_UP, MarketRegime.TREND_DOWN]

    @property
    def is_mean_reverting(self) -> bool:
        return self.regime == MarketRegime.MEAN_REVERSION

    @property
    def is_choppy(self) -> bool:
        return self.regime == MarketRegime.CHOP

    @property
    def trend_direction(self) -> int:
        """Return 1 for up trend, -1 for down trend, 0 for no trend"""
        if self.regime == MarketRegime.TREND_UP:
            return 1
        elif self.regime == MarketRegime.TREND_DOWN:
            return -1
        return 0


class RegimeDetector:
    """
    Advanced regime detection system using multiple indicators
    """

    def __init__(self, lookback_periods: int = 100):
        self.lookback_periods = lookback_periods

        # Model for regime classification
        self.model = RandomForestClassifier(
            n_estimators=50, max_depth=8, min_samples_split=10, random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False

        # Regime tracking
        self.regime_history: List[RegimeClassification] = []
        self.current_regime: Optional[RegimeClassification] = None

        # Configuration thresholds
        self.thresholds = {
            "adx_trending": 25.0,
            "adx_strong_trend": 40.0,
            "rsi_oversold": 30.0,
            "rsi_overbought": 70.0,
            "volatility_spike": 2.0,
            "mean_reversion_threshold": 0.02,
            "chop_adx_threshold": 20.0,
        }

    def calculate_features(self, price_data: pd.DataFrame) -> RegimeFeatures:
        """Calculate comprehensive regime features from price data"""

        try:
            # Ensure we have required columns
            required_cols = ["open", "high", "low", "close", "volume"]
            for col in required_cols:
                if col not in price_data.columns:
                    raise ValueError(f"Missing required column: {col}")

            if len(price_data) < 50:
                raise ValueError("Insufficient data for feature calculation")

            # Get latest values
            close = price_data["close"].values
            high = price_data["high"].values
            low = price_data["low"].values
            volume = price_data["volume"].values

            # Trend indicators
            if TALIB_AVAILABLE:
                adx_values = talib.ADX(high, low, close, timeperiod=14)
                adx = adx_values[-1] if not np.isnan(adx_values[-1]) else 0.0
                adx_trend = adx_values[-1] - adx_values[-5] if len(adx_values) >= 5 else 0.0

                # EMA slopes
                ema_12 = talib.EMA(close, timeperiod=12)
                ema_26 = talib.EMA(close, timeperiod=26)
                ema_slope_short = (
                    (ema_12[-1] - ema_12[-5]) / ema_12[-5] if len(ema_12) >= 5 else 0.0
                )
                ema_slope_long = (ema_26[-1] - ema_26[-5]) / ema_26[-5] if len(ema_26) >= 5 else 0.0

                # Mean reversion indicators
                rsi_values = talib.RSI(close, timeperiod=14)
                rsi = rsi_values[-1] if not np.isnan(rsi_values[-1]) else 50.0

                # Bollinger Bands
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20)
                bb_position = (close[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                bb_position = np.clip(bb_position, 0, 1)

                # Distance from mean
                sma_20 = talib.SMA(close, timeperiod=20)
                price_distance = (
                    (close[-1] - sma_20[-1]) / sma_20[-1] if not np.isnan(sma_20[-1]) else 0.0
                )

                # Volatility indicators
                atr_values = talib.ATR(high, low, close, timeperiod=14)
                atr_current = atr_values[-1] if not np.isnan(atr_values[-1]) else 0.0
                atr_normalized = atr_current / close[-1] if close[-1] > 0 else 0.0
            else:
                # Fallback calculations
                adx = self._calculate_adx_fallback(high, low, close)
                adx_trend = 0.0

                # Simple EMA calculation
                ema_12 = self._calculate_ema_fallback(close, 12)
                ema_26 = self._calculate_ema_fallback(close, 26)
                ema_slope_short = (
                    (ema_12[-1] - ema_12[-5]) / ema_12[-5] if len(ema_12) >= 5 else 0.0
                )
                ema_slope_long = (ema_26[-1] - ema_26[-5]) / ema_26[-5] if len(ema_26) >= 5 else 0.0

                # Simple RSI calculation
                rsi = self._calculate_rsi_fallback(close)

                # Simple Bollinger Bands
                sma_20 = np.mean(close[-20:]) if len(close) >= 20 else np.mean(close)
                std_20 = np.std(close[-20:]) if len(close) >= 20 else np.std(close)
                bb_upper = sma_20 + 2 * std_20
                bb_lower = sma_20 - 2 * std_20
                bb_position = (
                    (close[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
                )
                bb_position = np.clip(bb_position, 0, 1)

                price_distance = (close[-1] - sma_20) / sma_20 if sma_20 > 0 else 0.0

                # Simple ATR
                atr_current = self._calculate_atr_fallback(high, low, close)
                atr_normalized = atr_current / close[-1] if close[-1] > 0 else 0.0

            # Price momentum
            price_momentum = (close[-1] - close[-10]) / close[-10] if len(close) >= 10 else 0.0

            # Volatility ratio (current vs historical)
            volatility_20 = np.std(close[-20:]) if len(close) >= 20 else np.std(close)
            volatility_5 = np.std(close[-5:]) if len(close) >= 5 else np.std(close)
            volatility_ratio = volatility_5 / volatility_20 if volatility_20 > 0 else 1.0

            # Volume profile
            avg_volume = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
            current_volume = volume[-1]
            volume_profile = current_volume / avg_volume if avg_volume > 0 else 1.0

            # Support/Resistance strength (simplified)
            recent_highs = np.max(high[-10:]) if len(high) >= 10 else high[-1]
            recent_lows = np.min(low[-10:]) if len(low) >= 10 else low[-1]
            sr_strength = (recent_highs - recent_lows) / close[-1] if close[-1] > 0 else 0.0

            # Breakout probability (price near recent extremes)
            price_percentile = (
                (close[-1] - recent_lows) / (recent_highs - recent_lows)
                if recent_highs > recent_lows
                else 0.5
            )
            breakout_prob = 1.0 if price_percentile > 0.9 or price_percentile < 0.1 else 0.0

            # Create features object
            features = RegimeFeatures(
                symbol=getattr(price_data, "symbol", "BTC/USD"),
                timestamp=datetime.now(),
                adx=float(adx),
                adx_trend=float(adx_trend),
                ema_slope_short=float(ema_slope_short),
                ema_slope_long=float(ema_slope_long),
                price_momentum=float(price_momentum),
                rsi=float(rsi),
                bollinger_position=float(bb_position),
                price_distance_from_mean=float(price_distance),
                atr_normalized=float(atr_normalized),
                volatility_ratio=float(volatility_ratio),
                volume_profile=float(volume_profile),
                support_resistance_strength=float(sr_strength),
                breakout_probability=float(breakout_prob),
                liquidity_score=1.0,  # Would be calculated from order book
                correlation_strength=0.0,  # Would be calculated from cross-market data
                market_breadth=0.0,  # Would be calculated from broader market
            )

            return features

        except Exception as e:
            logger.error(f"Feature calculation failed: {e}")
            # Return default features
            return RegimeFeatures(symbol="Unknown", timestamp=datetime.now())

    def classify_regime_rules_based(self, features: RegimeFeatures) -> RegimeClassification:
        """Rule-based regime classification as fallback"""

        try:
            # Rule-based classification logic
            regime_scores = {
                MarketRegime.TREND_UP: 0.0,
                MarketRegime.TREND_DOWN: 0.0,
                MarketRegime.MEAN_REVERSION: 0.0,
                MarketRegime.CHOP: 0.0,
                MarketRegime.BREAKOUT: 0.0,
                MarketRegime.VOLATILITY_SPIKE: 0.0,
            }

            # Trend detection
            if features.adx > self.thresholds["adx_trending"]:
                if features.ema_slope_short > 0 and features.price_momentum > 0:
                    regime_scores[MarketRegime.TREND_UP] += 0.4
                elif features.ema_slope_short < 0 and features.price_momentum < 0:
                    regime_scores[MarketRegime.TREND_DOWN] += 0.4

            # ADX strength boost
            if features.adx > self.thresholds["adx_strong_trend"]:
                if features.price_momentum > 0:
                    regime_scores[MarketRegime.TREND_UP] += 0.3
                else:
                    regime_scores[MarketRegime.TREND_DOWN] += 0.3

            # Mean reversion signals
            if (
                features.rsi > self.thresholds["rsi_overbought"]
                or features.rsi < self.thresholds["rsi_oversold"]
            ):
                regime_scores[MarketRegime.MEAN_REVERSION] += 0.3

            if abs(features.price_distance_from_mean) > self.thresholds["mean_reversion_threshold"]:
                regime_scores[MarketRegime.MEAN_REVERSION] += 0.2

            # Chop/sideways detection
            if features.adx < self.thresholds["chop_adx_threshold"]:
                regime_scores[MarketRegime.CHOP] += 0.4

            if abs(features.price_momentum) < 0.01:  # Very low momentum
                regime_scores[MarketRegime.CHOP] += 0.2

            # Breakout detection
            if features.breakout_probability > 0.5:
                regime_scores[MarketRegime.BREAKOUT] += 0.5

            # Volatility spike
            if features.volatility_ratio > self.thresholds["volatility_spike"]:
                regime_scores[MarketRegime.VOLATILITY_SPIKE] += 0.6

            # Find dominant regime
            max_regime = max(regime_scores.keys(), key=lambda k: regime_scores[k])
            max_score = regime_scores[max_regime]

            # Determine confidence
            if max_score > 0.7:
                confidence = RegimeConfidence.HIGH
            elif max_score > 0.4:
                confidence = RegimeConfidence.MEDIUM
            else:
                confidence = RegimeConfidence.LOW

            # Convert scores to probabilities
            total_score = sum(regime_scores.values()) or 1.0
            regime_probs = {
                regime.value: score / total_score for regime, score in regime_scores.items()
            }

            return RegimeClassification(
                regime=max_regime,
                confidence=confidence,
                probability=max_score,
                features=features,
                regime_probabilities=regime_probs,
            )

        except Exception as e:
            logger.error(f"Rule-based classification failed: {e}")
            return RegimeClassification(
                regime=MarketRegime.CHOP,
                confidence=RegimeConfidence.LOW,
                probability=0.5,
                features=features,
            )

    def classify_regime_ml(self, features: RegimeFeatures) -> RegimeClassification:
        """ML-based regime classification (when trained)"""

        if not self.is_trained:
            return self.classify_regime_rules_based(features)

        try:
            # Extract feature vector
            feature_vector = self._features_to_vector(features)
            feature_scaled = self.scaler.transform([feature_vector])

            # Get predictions
            probabilities = self.model.predict_proba(feature_scaled)[0]
            predicted_class = self.model.predict(feature_scaled)[0]

            # Map to regime
            regime_mapping = {
                0: MarketRegime.TREND_UP,
                1: MarketRegime.TREND_DOWN,
                2: MarketRegime.MEAN_REVERSION,
                3: MarketRegime.CHOP,
                4: MarketRegime.BREAKOUT,
                5: MarketRegime.VOLATILITY_SPIKE,
            }

            regime = regime_mapping.get(predicted_class, MarketRegime.CHOP)
            confidence_score = max(probabilities)

            if confidence_score > 0.8:
                confidence = RegimeConfidence.HIGH
            elif confidence_score > 0.6:
                confidence = RegimeConfidence.MEDIUM
            else:
                confidence = RegimeConfidence.LOW

            # Create probability dictionary
            regime_probs = {
                regime_val.value: prob
                for regime_val, prob in zip(regime_mapping.values(), probabilities)
            }

            return RegimeClassification(
                regime=regime,
                confidence=confidence,
                probability=confidence_score,
                features=features,
                regime_probabilities=regime_probs,
            )

        except Exception as e:
            logger.error(f"ML classification failed: {e}")
            return self.classify_regime_rules_based(features)

    def detect_regime(self, price_data: pd.DataFrame) -> RegimeClassification:
        """Main regime detection method"""

        try:
            # Calculate features
            features = self.calculate_features(price_data)

            # Classify regime
            if self.is_trained:
                classification = self.classify_regime_ml(features)
            else:
                classification = self.classify_regime_rules_based(features)

            # Update regime duration if same regime
            if self.current_regime and self.current_regime.regime == classification.regime:
                classification.regime_start = self.current_regime.regime_start
                if classification.regime_start:
                    classification.regime_duration_minutes = int(
                        (datetime.now() - classification.regime_start).total_seconds() / 60
                    )
            else:
                # New regime detected
                classification.regime_start = datetime.now()
                classification.regime_duration_minutes = 0

            # Update current regime
            self.current_regime = classification

            # Add to history
            self.regime_history.append(classification)

            # Limit history size
            if len(self.regime_history) > 1000:
                self.regime_history = self.regime_history[-500:]

            logger.info(
                f"Regime detected: {classification.regime.value} ({classification.confidence.value})"
            )

            return classification

        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            # Return default classification
            return RegimeClassification(
                regime=MarketRegime.CHOP,
                confidence=RegimeConfidence.LOW,
                probability=0.5,
                features=RegimeFeatures(symbol="Unknown", timestamp=datetime.now()),
            )

    def _features_to_vector(self, features: RegimeFeatures) -> List[float]:
        """Convert features to ML input vector"""

        return [
            features.adx,
            features.adx_trend,
            features.ema_slope_short,
            features.ema_slope_long,
            features.price_momentum,
            features.rsi,
            features.bollinger_position,
            features.price_distance_from_mean,
            features.atr_normalized,
            features.volatility_ratio,
            features.volume_profile,
            features.support_resistance_strength,
            features.breakout_probability,
            features.liquidity_score,
            features.correlation_strength,
            features.market_breadth,
        ]

    def train_model(self, training_data: List[Tuple[RegimeFeatures, MarketRegime]]):
        """Train ML model on historical regime data"""

        try:
            if len(training_data) < 50:
                logger.warning("Insufficient training data for ML model")
                return False

            # Prepare training data
            X = []
            y = []

            regime_to_int = {
                MarketRegime.TREND_UP: 0,
                MarketRegime.TREND_DOWN: 1,
                MarketRegime.MEAN_REVERSION: 2,
                MarketRegime.CHOP: 3,
                MarketRegime.BREAKOUT: 4,
                MarketRegime.VOLATILITY_SPIKE: 5,
            }

            for features, regime in training_data:
                feature_vector = self._features_to_vector(features)
                X.append(feature_vector)
                y.append(regime_to_int[regime])

            X = np.array(X)
            y = np.array(y)

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train model
            self.model.fit(X_scaled, y)
            self.is_trained = True

            # Calculate accuracy
            predictions = self.model.predict(X_scaled)
            accuracy = np.mean(predictions == y)

            logger.info(f"Regime detection model trained - Accuracy: {accuracy:.2%}")
            return True

        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return False

    def get_regime_statistics(self, days_back: int = 30) -> Dict[str, Any]:
        """Get regime statistics over specified period"""

        cutoff_time = datetime.now() - timedelta(days=days_back)
        recent_regimes = [r for r in self.regime_history if r.features.timestamp >= cutoff_time]

        if not recent_regimes:
            return {"status": "no_data"}

        # Count regime occurrences
        regime_counts = {}
        for classification in recent_regimes:
            regime = classification.regime.value
            regime_counts[regime] = regime_counts.get(regime, 0) + 1

        # Calculate percentages
        total_classifications = len(recent_regimes)
        regime_percentages = {
            regime: (count / total_classifications) * 100 for regime, count in regime_counts.items()
        }

        # Calculate average confidence by regime
        regime_confidences = {}
        for regime in regime_counts.keys():
            regime_classifications = [r for r in recent_regimes if r.regime.value == regime]
            avg_confidence = np.mean([r.probability for r in regime_classifications])
            regime_confidences[regime] = avg_confidence

        # Recent regime transitions
        transitions = []
        for i in range(1, min(10, len(recent_regimes))):
            if recent_regimes[i].regime != recent_regimes[i - 1].regime:
                transitions.append(
                    {
                        "from": recent_regimes[i - 1].regime.value,
                        "to": recent_regimes[i].regime.value,
                        "timestamp": recent_regimes[i].features.timestamp.isoformat(),
                    }
                )

        return {
            "period_days": days_back,
            "total_classifications": total_classifications,
            "regime_distribution": regime_percentages,
            "average_confidences": regime_confidences,
            "recent_transitions": transitions,
            "current_regime": self.current_regime.regime.value if self.current_regime else None,
            "current_confidence": self.current_regime.confidence.value
            if self.current_regime
            else None,
            "model_trained": self.is_trained,
        }

    def get_regime_forecast(
        self, price_data: pd.DataFrame, periods_ahead: int = 5
    ) -> List[Dict[str, Any]]:
        """Simple regime forecasting based on current trends"""

        try:
            current_classification = self.detect_regime(price_data)

            forecasts = []

            # Simple persistence forecast with decay
            for i in range(1, periods_ahead + 1):
                persistence_prob = current_classification.probability * (0.9**i)

                forecast = {
                    "periods_ahead": i,
                    "most_likely_regime": current_classification.regime.value,
                    "probability": persistence_prob,
                    "confidence": "low" if persistence_prob < 0.5 else "medium",
                }

                forecasts.append(forecast)

            return forecasts

        except Exception as e:
            logger.error(f"Regime forecasting failed: {e}")
            return []

    def _calculate_adx_fallback(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> float:
        """Simplified ADX calculation without talib"""
        try:
            if len(close) < 14:
                return 0.0

            # Simple trend strength calculation
            up_moves = np.where(high[1:] > high[:-1], high[1:] - high[:-1], 0)
            down_moves = np.where(low[1:] < low[:-1], low[:-1] - low[1:], 0)

            avg_up = np.mean(up_moves[-14:])
            avg_down = np.mean(down_moves[-14:])

            if avg_up + avg_down == 0:
                return 0.0

            # Simplified directional index
            dx = abs(avg_up - avg_down) / (avg_up + avg_down) * 100
            return min(dx, 100.0)

        except Exception:
            return 0.0

    def _calculate_ema_fallback(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Simple EMA calculation"""
        try:
            alpha = 2.0 / (period + 1)
            ema = np.zeros_like(prices)
            ema[0] = prices[0]

            for i in range(1, len(prices)):
                ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]

            return ema
        except Exception:
            return np.array([prices[-1]] * len(prices))

    def _calculate_rsi_fallback(self, prices: np.ndarray, period: int = 14) -> float:
        """Simple RSI calculation"""
        try:
            if len(prices) < period + 1:
                return 50.0

            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)

            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])

            if avg_loss == 0:
                return 100.0

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            return rsi
        except Exception:
            return 50.0

    def _calculate_atr_fallback(
        self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14
    ) -> float:
        """Simple ATR calculation"""
        try:
            if len(close) < 2:
                return 0.0

            tr_values = []
            for i in range(1, len(close)):
                tr1 = high[i] - low[i]
                tr2 = abs(high[i] - close[i - 1])
                tr3 = abs(low[i] - close[i - 1])
                tr = max(tr1, tr2, tr3)
                tr_values.append(tr)

            if not tr_values:
                return 0.0

            return np.mean(tr_values[-period:])
        except Exception:
            return 0.0
