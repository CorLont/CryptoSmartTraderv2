"""
Base Models for Ensemble Learning

Defines interfaces and implementations for base models that provide
orthogonal information sources for meta-learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Protocol, runtime_checkable
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class ModelPrediction:
    """Standardized prediction output from base models"""
    model_name: str
    symbol: str
    timestamp: datetime

    # Core prediction
    probability: float           # Win probability (0-1)
    confidence: float           # Model confidence (0-1)
    direction: str              # 'up' or 'down'

    # Feature importance and explanation
    feature_importance: Dict[str, float]
    explanation: str

    # Model-specific metadata
    model_version: str
    training_data_end: Optional[datetime]

    # Signal decay properties
    ttl_hours: float            # Time-to-live for this signal
    decay_factor: float         # How fast signal decays (0-1)


@runtime_checkable
class BaseModelInterface(Protocol):
    """Interface that all base models must implement"""

    @property
    def model_name(self) -> str:
        """Unique model identifier"""
        ...

    @property
    def model_type(self) -> str:
        """Model category: 'technical', 'sentiment', 'regime', 'onchain'"""
        ...

    def predict(self,
               symbol: str,
               market_data: Dict[str, Any],
               lookback_hours: int = 24) -> Optional[ModelPrediction]:
        """Generate prediction for given symbol and market data"""
        ...

    def get_feature_importance(self) -> Dict[str, float]:
        """Get current feature importance weights"""
        ...

    def is_ready(self) -> bool:
        """Check if model is ready to make predictions"""
        ...


class TechnicalAnalysisModel:
    """
    Technical Analysis base model

    Provides signals based on price action, volume, and technical indicators
    """

    def __init__(self):
        self.model_name = "technical_analysis"
        self.model_type = "technical"
        self.model_version = "1.0.0"

        # TA indicator weights (can be learned)
        self.indicator_weights = {
            'rsi': 0.15,
            'macd': 0.20,
            'bb_position': 0.15,
            'volume_profile': 0.20,
            'momentum': 0.15,
            'support_resistance': 0.15
        }

    def predict(self,
               symbol: str,
               market_data: Dict[str, Any],
               lookback_hours: int = 24) -> Optional[ModelPrediction]:
        """Generate TA-based prediction"""
        try:
            # Extract price data
            if 'price_data' not in market_data:
                logger.warning(f"No price data for TA prediction: {symbol}")
                return None

            price_df = market_data['price_data']
            if len(price_df) < 50:  # Need enough data for TA
                return None

            # Calculate technical indicators
            indicators = self._calculate_indicators(price_df)

            # Generate signals from each indicator
            signals = self._generate_signals(indicators)

            # Combine signals using weights
            combined_score = sum(
                signals[indicator] * weight
                for indicator, weight in self.indicator_weights.items()
                if indicator in signals
            )

            # Convert to probability (sigmoid transformation)
            probability = 1 / (1 + np.exp(-combined_score * 5))  # Scale factor 5

            # Calculate confidence based on signal strength and consistency
            signal_strengths = [abs(signals.get(ind, 0)) for ind in self.indicator_weights.keys()]
            confidence = min(1.0, np.mean(signal_strengths) * 1.5)

            # Determine direction
            direction = 'up' if combined_score > 0 else 'down'

            # Generate explanation
            strong_signals = {k: v for k, v in signals.items() if abs(v) > 0.3}
            explanation = f"TA signals: {', '.join([f'{k}:{v:.2f}' for k, v in strong_signals.items()])}"

            return ModelPrediction(
                model_name=self.model_name,
                symbol=symbol,
                timestamp=datetime.now(),
                probability=probability,
                confidence=confidence,
                direction=direction,
                feature_importance=dict(self.indicator_weights),
                explanation=explanation,
                model_version=self.model_version,
                training_data_end=None,
                ttl_hours=4.0,  # TA signals decay in 4 hours
                decay_factor=0.7
            )

        except Exception as e:
            logger.error(f"TA prediction failed for {symbol}: {e}")
            return None

    def _calculate_indicators(self, price_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate technical indicators"""
        try:
            indicators = {}

            # Ensure we have OHLCV columns
            if not all(col in price_df.columns for col in ['close', 'high', 'low', 'volume']):
                return {}

            close = price_df['close'].values
            high = price_df['high'].values
            low = price_df['low'].values
            volume = price_df['volume'].values

            # RSI (14-period)
            rsi = self._calculate_rsi(close, 14)
            indicators['rsi'] = (rsi - 50) / 50  # Normalize to [-1, 1]

            # MACD
            macd_line, signal_line = self._calculate_macd(close)
            indicators['macd'] = np.tanh((macd_line - signal_line) * 1000)  # Normalize

            # Bollinger Bands position
            bb_upper, bb_lower, bb_mid = self._calculate_bollinger_bands(close, 20, 2)
            bb_position = (close[-1] - bb_mid) / (bb_upper - bb_lower)
            indicators['bb_position'] = np.clip(bb_position, -1, 1)

            # Volume profile (relative to recent average)
            avg_volume = np.mean(volume[-20:])
            current_volume = volume[-1]
            indicators['volume_profile'] = np.tanh((current_volume - avg_volume) / avg_volume)

            # Momentum (rate of change)
            if len(close) >= 10:
                momentum = (close[-1] - close[-10]) / close[-10]
                indicators['momentum'] = np.tanh(momentum * 20)
            else:
                indicators['momentum'] = 0.0

            # Support/Resistance (simplified)
            indicators['support_resistance'] = self._calculate_support_resistance_signal(close, high, low)

            return indicators

        except Exception as e:
            logger.error(f"Indicator calculation failed: {e}")
            return {}

    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
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

    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD indicator"""
        try:
            if len(prices) < slow + signal:
                return 0.0, 0.0

            # Exponential moving averages
            ema_fast = self._ema(prices, fast)
            ema_slow = self._ema(prices, slow)

            macd_line = ema_fast - ema_slow
            signal_line = self._ema(np.array([macd_line] * signal), signal)  # Simplified

            return macd_line, signal_line

        except Exception:
            return 0.0, 0.0

    def _ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate exponential moving average"""
        try:
            if len(prices) < period:
                return np.mean(prices)

            alpha = 2 / (period + 1)
            ema = prices[0]

            for price in prices[1:]:
                ema = alpha * price + (1 - alpha) * ema

            return ema

        except Exception:
            return np.mean(prices) if len(prices) > 0 else 0.0

    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2) -> tuple:
        """Calculate Bollinger Bands"""
        try:
            if len(prices) < period:
                mean_price = np.mean(prices)
                return mean_price, mean_price, mean_price

            recent_prices = prices[-period:]
            middle = np.mean(recent_prices)
            std = np.std(recent_prices)

            upper = middle + (std_dev * std)
            lower = middle - (std_dev * std)

            return upper, lower, middle

        except Exception:
            mean_price = np.mean(prices) if len(prices) > 0 else 0.0
            return mean_price, mean_price, mean_price

    def _calculate_support_resistance_signal(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> float:
        """Calculate support/resistance signal"""
        try:
            if len(close) < 20:
                return 0.0

            current_price = close[-1]
            recent_highs = high[-20:]
            recent_lows = low[-20:]

            # Find support and resistance levels
            resistance = np.percentile(recent_highs, 95)
            support = np.percentile(recent_lows, 5)

            # Calculate position relative to S/R
            if resistance == support:
                return 0.0

            position = (current_price - support) / (resistance - support)

            # Convert to signal (-1 to 1)
            # Near resistance = negative (sell), near support = positive (buy)
            signal = 1 - 2 * position

            return np.clip(signal, -1, 1)

        except Exception:
            return 0.0

    def _generate_signals(self, indicators: Dict[str, float]) -> Dict[str, float]:
        """Convert indicators to normalized signals"""
        signals = {}

        for indicator, value in indicators.items():
            if indicator == 'rsi':
                # RSI already normalized
                signals[indicator] = value
            elif indicator == 'macd':
                # MACD already normalized
                signals[indicator] = value
            elif indicator == 'bb_position':
                # BB position already normalized
                signals[indicator] = value
            elif indicator == 'volume_profile':
                # Volume already normalized
                signals[indicator] = value
            elif indicator == 'momentum':
                # Momentum already normalized
                signals[indicator] = value
            elif indicator == 'support_resistance':
                # S/R already normalized
                signals[indicator] = value
            else:
                # Default normalization
                signals[indicator] = np.clip(value, -1, 1)

        return signals

    def get_feature_importance(self) -> Dict[str, float]:
        """Return current indicator weights"""
        return dict(self.indicator_weights)

    def is_ready(self) -> bool:
        """TA model is always ready"""
        return True


class SentimentModel:
    """
    Sentiment Analysis base model

    Provides signals based on social sentiment, news sentiment, and market sentiment
    """

    def __init__(self):
        self.model_name = "sentiment_analysis"
        self.model_type = "sentiment"
        self.model_version = "1.0.0"

        # Sentiment source weights
        self.sentiment_weights = {
            'social_sentiment': 0.4,
            'news_sentiment': 0.3,
            'market_sentiment': 0.3
        }

    def predict(self,
               symbol: str,
               market_data: Dict[str, Any],
               lookback_hours: int = 24) -> Optional[ModelPrediction]:
        """Generate sentiment-based prediction"""
        try:
            # Extract sentiment data
            sentiment_data = market_data.get('sentiment_data', {})

            if not sentiment_data:
                logger.warning(f"No sentiment data for prediction: {symbol}")
                return None

            # Calculate sentiment signals
            signals = self._calculate_sentiment_signals(sentiment_data)

            if not signals:
                return None

            # Combine signals using weights
            combined_score = sum(
                signals[source] * weight
                for source, weight in self.sentiment_weights.items()
                if source in signals
            )

            # Convert to probability
            probability = 1 / (1 + np.exp(-combined_score * 3))  # Scale factor 3

            # Calculate confidence based on sentiment strength and freshness
            signal_strengths = [abs(signals.get(source, 0)) for source in self.sentiment_weights.keys()]
            confidence = min(1.0, np.mean(signal_strengths) * 1.2)

            # Determine direction
            direction = 'up' if combined_score > 0 else 'down'

            # Generate explanation
            strong_signals = {k: v for k, v in signals.items() if abs(v) > 0.2}
            explanation = f"Sentiment: {', '.join([f'{k}:{v:.2f}' for k, v in strong_signals.items()])}"

            return ModelPrediction(
                model_name=self.model_name,
                symbol=symbol,
                timestamp=datetime.now(),
                probability=probability,
                confidence=confidence,
                direction=direction,
                feature_importance=dict(self.sentiment_weights),
                explanation=explanation,
                model_version=self.model_version,
                training_data_end=None,
                ttl_hours=2.0,  # Sentiment signals decay in 2 hours
                decay_factor=0.8
            )

        except Exception as e:
            logger.error(f"Sentiment prediction failed for {symbol}: {e}")
            return None

    def _calculate_sentiment_signals(self, sentiment_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate sentiment signals from various sources"""
        try:
            signals = {}

            # Social sentiment (Twitter, Reddit, etc.)
            social_sentiment = sentiment_data.get('social_sentiment', 0.5)
            signals['social_sentiment'] = (social_sentiment - 0.5) * 2  # Convert to [-1, 1]

            # News sentiment
            news_sentiment = sentiment_data.get('news_sentiment', 0.5)
            signals['news_sentiment'] = (news_sentiment - 0.5) * 2  # Convert to [-1, 1]

            # Market sentiment (fear/greed index, etc.)
            market_sentiment = sentiment_data.get('market_sentiment', 0.5)
            signals['market_sentiment'] = (market_sentiment - 0.5) * 2  # Convert to [-1, 1]

            return signals

        except Exception as e:
            logger.error(f"Sentiment signal calculation failed: {e}")
            return {}

    def get_feature_importance(self) -> Dict[str, float]:
        """Return current sentiment weights"""
        return dict(self.sentiment_weights)

    def is_ready(self) -> bool:
        """Sentiment model is ready if we have basic sentiment data"""
        return True


class RegimeModel:
    """
    Market Regime base model

    Provides signals based on market regime classification
    """

    def __init__(self):
        self.model_name = "regime_classifier"
        self.model_type = "regime"
        self.model_version = "1.0.0"

        # Regime-based signal mappings
        self.regime_signals = {
            'trend_up': 0.8,
            'trend_down': -0.8,
            'mean_reversion': 0.0,
            'high_vol_chop': -0.6,
            'low_vol_drift': 0.2,
            'risk_off': -0.9
        }

    def predict(self,
               symbol: str,
               market_data: Dict[str, Any],
               lookback_hours: int = 24) -> Optional[ModelPrediction]:
        """Generate regime-based prediction"""
        try:
            # Extract regime classification
            regime_data = market_data.get('regime_data', {})

            if not regime_data:
                logger.warning(f"No regime data for prediction: {symbol}")
                return None

            # Get current regime
            current_regime = regime_data.get('primary_regime', 'mean_reversion')
            regime_confidence = regime_data.get('confidence', 0.5)

            # Get base signal from regime
            base_signal = self.regime_signals.get(current_regime, 0.0)

            # Adjust signal by regime confidence
            adjusted_signal = base_signal * regime_confidence

            # Convert to probability
            probability = 1 / (1 + np.exp(-adjusted_signal * 4))  # Scale factor 4

            # Confidence is the regime classification confidence
            confidence = regime_confidence

            # Determine direction
            direction = 'up' if adjusted_signal > 0 else 'down'

            # Generate explanation
            explanation = f"Regime: {current_regime} (conf: {regime_confidence:.2f})"

            return ModelPrediction(
                model_name=self.model_name,
                symbol=symbol,
                timestamp=datetime.now(),
                probability=probability,
                confidence=confidence,
                direction=direction,
                feature_importance={'regime_classification': 1.0},
                explanation=explanation,
                model_version=self.model_version,
                training_data_end=None,
                ttl_hours=6.0,  # Regime signals last longer (6 hours)
                decay_factor=0.9
            )

        except Exception as e:
            logger.error(f"Regime prediction failed for {symbol}: {e}")
            return None

    def get_feature_importance(self) -> Dict[str, float]:
        """Return regime importance (always 1.0 as it's a single feature)"""
        return {'regime_classification': 1.0}

    def is_ready(self) -> bool:
        """Regime model is ready if we have regime classification"""
        return True
