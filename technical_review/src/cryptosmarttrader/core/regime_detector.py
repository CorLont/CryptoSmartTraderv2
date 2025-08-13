"""Advanced regime detection system for market state classification and strategy switching."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
from pathlib import Path
import asyncio
import threading

from .structured_logger import get_logger


class MarketRegime(Enum):
    """Market regime classifications."""
    BULL_TRENDING = "bull_trending"      # Strong upward trend
    BEAR_TRENDING = "bear_trending"      # Strong downward trend
    SIDEWAYS_LOW_VOL = "sideways_low_vol"    # Range-bound, low volatility
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"  # Range-bound, high volatility
    BREAKOUT = "breakout"                # Momentum breakout phase
    REVERSAL = "reversal"                # Trend reversal phase


@dataclass
class RegimeMetrics:
    """Quantitative metrics for regime classification."""
    trend_strength: float           # -1 (bear) to +1 (bull)
    volatility_percentile: float    # 0 to 100
    momentum_score: float           # -1 to +1
    mean_reversion_score: float     # 0 to 1
    volume_profile: float           # Relative volume strength
    hurst_exponent: float           # 0 to 1 (0.5 = random walk)
    adx_strength: float             # 0 to 100 (trend strength)
    rsi_divergence: float           # RSI vs price divergence
    correlation_breakdown: float    # Cross-asset correlation change
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RegimeTransition:
    """Record of regime transition."""
    from_regime: MarketRegime
    to_regime: MarketRegime
    confidence: float
    trigger_metrics: Dict[str, float]
    timestamp: datetime
    duration_hours: float = 0.0


class RegimeDetector:
    """Advanced market regime detection and classification system."""

    def __init__(self, lookback_periods: int = 252):
        """Initialize regime detector."""
        self.logger = get_logger("regime_detector")
        self.lookback_periods = lookback_periods

        # Current state
        self.current_regime = MarketRegime.SIDEWAYS_LOW_VOL
        self.regime_confidence = 0.5
        self.regime_start_time = datetime.now()

        # Historical data storage
        self.price_history: Dict[str, pd.DataFrame] = {}
        self.volume_history: Dict[str, pd.DataFrame] = {}
        self.regime_history: List[RegimeTransition] = []

        # Regime detection parameters
        self.regime_thresholds = {
            'trend_strength_bull': 0.3,
            'trend_strength_bear': -0.3,
            'volatility_high': 75.0,  # 75th percentile
            'volatility_low': 25.0,   # 25th percentile
            'momentum_threshold': 0.2,
            'hurst_trend_threshold': 0.6,
            'adx_trending_threshold': 25.0,
            'min_confidence': 0.7
        }

        # Regime-specific strategy mappings
        self.regime_strategies = {
            MarketRegime.BULL_TRENDING: {
                'primary_strategy': 'momentum_long',
                'secondary_strategy': 'breakout_continuation',
                'position_sizing': 1.0,
                'risk_multiplier': 1.2,
                'rebalance_frequency': 'daily'
            },
            MarketRegime.BEAR_TRENDING: {
                'primary_strategy': 'momentum_short',
                'secondary_strategy': 'breakdown_continuation',
                'position_sizing': 0.8,
                'risk_multiplier': 0.8,
                'rebalance_frequency': 'daily'
            },
            MarketRegime.SIDEWAYS_LOW_VOL: {
                'primary_strategy': 'mean_reversion',
                'secondary_strategy': 'range_trading',
                'position_sizing': 0.6,
                'risk_multiplier': 0.7,
                'rebalance_frequency': 'weekly'
            },
            MarketRegime.SIDEWAYS_HIGH_VOL: {
                'primary_strategy': 'volatility_capture',
                'secondary_strategy': 'short_gamma',
                'position_sizing': 0.5,
                'risk_multiplier': 0.6,
                'rebalance_frequency': 'intraday'
            },
            MarketRegime.BREAKOUT: {
                'primary_strategy': 'breakout_momentum',
                'secondary_strategy': 'trend_following',
                'position_sizing': 1.3,
                'risk_multiplier': 1.5,
                'rebalance_frequency': 'hourly'
            },
            MarketRegime.REVERSAL: {
                'primary_strategy': 'contrarian',
                'secondary_strategy': 'mean_reversion_aggressive',
                'position_sizing': 0.4,
                'risk_multiplier': 0.5,
                'rebalance_frequency': 'hourly'
            }
        }

        # Thread safety
        self._lock = threading.RLock()

        # Persistence
        self.data_path = Path("data/regime_detection")
        self.data_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("Regime detector initialized",
                        lookback_periods=lookback_periods,
                        current_regime=self.current_regime.value)

    def update_market_data(self, symbol: str, price_data: pd.DataFrame,
                          volume_data: Optional[pd.DataFrame] = None) -> None:
        """Update market data for regime analysis."""
        with self._lock:
            # Ensure data has required columns
            required_columns = ['open', 'high', 'low', 'close', 'timestamp']
            if not all(col in price_data.columns for col in required_columns):
                self.logger.warning(f"Missing required columns for {symbol}")
                return

            # Store latest data (keep only lookback periods)
            self.price_history[symbol] = price_data.tail(self.lookback_periods).copy()

            if volume_data is not None:
                self.volume_history[symbol] = volume_data.tail(self.lookback_periods).copy()

            self.logger.debug(f"Updated market data for {symbol}",
                            records=len(price_data),
                            latest_price=price_data['close'].iloc[-1])

    def calculate_regime_metrics(self, symbol: str) -> Optional[RegimeMetrics]:
        """Calculate comprehensive regime metrics for a symbol."""
        if symbol not in self.price_history:
            return None

        df = self.price_history[symbol].copy()
        if len(df) < 50:  # Minimum data required
            return None

        try:
            # Calculate technical indicators
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['volatility'] = df['returns'].rolling(20).std() * np.sqrt(252)

            # Trend strength (slope of linear regression)
            x = np.arange(len(df))
            trend_slope = np.polyfit(x[-50:], df['close'].iloc[-50:], 1)[0]
            trend_strength = np.tanh(trend_slope / df['close'].iloc[-1] * 100)

            # Volatility percentile
            current_vol = df['volatility'].iloc[-1]
            vol_percentile = (df['volatility'].rank(pct=True).iloc[-1] * 100)

            # Momentum score (rate of change)
            momentum_20 = (df['close'].iloc[-1] / df['close'].iloc[-21] - 1)
            momentum_score = np.tanh(momentum_20 * 10)

            # Mean reversion score (distance from moving average)
            ma_50 = df['close'].rolling(50).mean().iloc[-1]
            mean_reversion = abs(df['close'].iloc[-1] - ma_50) / ma_50

            # Volume profile (relative volume strength)
            if symbol in self.volume_history:
                vol_df = self.volume_history[symbol]
                avg_volume = vol_df['volume'].rolling(20).mean().iloc[-1]
                current_volume = vol_df['volume'].iloc[-1]
                volume_profile = current_volume / avg_volume if avg_volume > 0 else 1.0
            else:
                volume_profile = 1.0

            # Hurst exponent calculation
            hurst_exponent = self._calculate_hurst_exponent(df['log_returns'].dropna())

            # ADX calculation (simplified)
            adx_strength = self._calculate_adx(df)

            # RSI divergence
            rsi = self._calculate_rsi(df['close'])
            price_change = (df['close'].iloc[-1] / df['close'].iloc[-21] - 1)
            rsi_change = (rsi.iloc[-1] - rsi.iloc[-21]) / 100
            rsi_divergence = abs(price_change - rsi_change)

            # Correlation breakdown (simplified - would need multiple assets)
            correlation_breakdown = 0.5  # Placeholder

            return RegimeMetrics(
                trend_strength=trend_strength,
                volatility_percentile=vol_percentile,
                momentum_score=momentum_score,
                mean_reversion_score=mean_reversion,
                volume_profile=volume_profile,
                hurst_exponent=hurst_exponent,
                adx_strength=adx_strength,
                rsi_divergence=rsi_divergence,
                correlation_breakdown=correlation_breakdown
            )

        except Exception as e:
            self.logger.error(f"Error calculating regime metrics for {symbol}: {e}")
            return None

    def _calculate_hurst_exponent(self, returns: pd.Series) -> float:
        """Calculate Hurst exponent for trend persistence."""
        try:
            if len(returns) < 100:
                return 0.5

            # Remove NaN values
            returns = returns.dropna()
            if len(returns) < 50:
                return 0.5

            # Calculate Hurst exponent using R/S analysis
            lags = range(2, min(100, len(returns) // 4))
            rs_values = []

            for lag in lags:
                # Split series into chunks
                chunks = [returns[i:i+lag] for i in range(0, len(returns), lag)]
                chunks = [chunk for chunk in chunks if len(chunk) == lag]

                if len(chunks) < 2:
                    continue

                rs_chunk = []
                for chunk in chunks:
                    mean_chunk = chunk.mean()
                    cumsum = (chunk - mean_chunk).cumsum()
                    r = cumsum.max() - cumsum.min()
                    s = chunk.std()
                    if s > 0:
                        rs_chunk.append(r / s)

                if rs_chunk:
                    rs_values.append(np.mean(rs_chunk))

            if len(rs_values) < 10:
                return 0.5

            # Linear regression to find Hurst exponent
            log_lags = np.log(lags[:len(rs_values)])
            log_rs = np.log(rs_values)
            hurst = np.polyfit(log_lags, log_rs, 1)[0]

            # Clamp to reasonable range
            return max(0.0, min(1.0, hurst))

        except Exception:
            return 0.5

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index."""
        try:
            high = df['high']
            low = df['low']
            close = df['close']

            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Directional movements
            dm_plus = high - high.shift()
            dm_minus = low.shift() - low

            dm_plus[dm_plus < 0] = 0
            dm_minus[dm_minus < 0] = 0
            dm_plus[(dm_plus - dm_minus) < 0] = 0
            dm_minus[(dm_minus - dm_plus) < 0] = 0

            # Smoothed averages
            atr = tr.rolling(period).mean()
            di_plus = 100 * (dm_plus.rolling(period).mean() / atr)
            di_minus = 100 * (dm_minus.rolling(period).mean() / atr)

            # ADX calculation
            dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
            adx = dx.rolling(period).mean()

            return adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 0.0

        except Exception:
            return 0.0

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception:
            return pd.Series([50] * len(prices), index=prices.index)

    def classify_regime(self, metrics: RegimeMetrics) -> Tuple[MarketRegime, float]:
        """Classify market regime based on metrics."""
        scores = {}

        # Bull trending conditions
        bull_score = 0.0
        if metrics.trend_strength > self.regime_thresholds['trend_strength_bull']:
            bull_score += 0.3
        if metrics.momentum_score > self.regime_thresholds['momentum_threshold']:
            bull_score += 0.2
        if metrics.hurst_exponent > self.regime_thresholds['hurst_trend_threshold']:
            bull_score += 0.2
        if metrics.adx_strength > self.regime_thresholds['adx_trending_threshold']:
            bull_score += 0.2
        if metrics.volume_profile > 1.2:  # Above average volume
            bull_score += 0.1
        scores[MarketRegime.BULL_TRENDING] = bull_score

        # Bear trending conditions
        bear_score = 0.0
        if metrics.trend_strength < self.regime_thresholds['trend_strength_bear']:
            bear_score += 0.3
        if metrics.momentum_score < -self.regime_thresholds['momentum_threshold']:
            bear_score += 0.2
        if metrics.hurst_exponent > self.regime_thresholds['hurst_trend_threshold']:
            bear_score += 0.2
        if metrics.adx_strength > self.regime_thresholds['adx_trending_threshold']:
            bear_score += 0.2
        if metrics.volume_profile > 1.2:
            bear_score += 0.1
        scores[MarketRegime.BEAR_TRENDING] = bear_score

        # Sideways low volatility
        sideways_low_score = 0.0
        if (abs(metrics.trend_strength) < 0.1 and
            metrics.volatility_percentile < self.regime_thresholds['volatility_low']):
            sideways_low_score += 0.4
        if metrics.mean_reversion_score > 0.3:
            sideways_low_score += 0.2
        if metrics.adx_strength < 20:
            sideways_low_score += 0.2
        if abs(metrics.momentum_score) < 0.1:
            sideways_low_score += 0.2
        scores[MarketRegime.SIDEWAYS_LOW_VOL] = sideways_low_score

        # Sideways high volatility
        sideways_high_score = 0.0
        if (abs(metrics.trend_strength) < 0.1 and
            metrics.volatility_percentile > self.regime_thresholds['volatility_high']):
            sideways_high_score += 0.4
        if metrics.rsi_divergence > 0.5:
            sideways_high_score += 0.2
        if metrics.volume_profile > 1.5:
            sideways_high_score += 0.2
        if metrics.adx_strength < 25:
            sideways_high_score += 0.2
        scores[MarketRegime.SIDEWAYS_HIGH_VOL] = sideways_high_score

        # Breakout conditions
        breakout_score = 0.0
        if metrics.volatility_percentile > 80 and abs(metrics.momentum_score) > 0.3:
            breakout_score += 0.4
        if metrics.volume_profile > 2.0:
            breakout_score += 0.3
        if metrics.adx_strength > 30:
            breakout_score += 0.2
        if metrics.hurst_exponent > 0.7:
            breakout_score += 0.1
        scores[MarketRegime.BREAKOUT] = breakout_score

        # Reversal conditions
        reversal_score = 0.0
        if metrics.rsi_divergence > 0.7:
            reversal_score += 0.3
        if metrics.correlation_breakdown > 0.6:
            reversal_score += 0.2
        if (metrics.trend_strength > 0.4 and metrics.momentum_score < -0.2) or \
           (metrics.trend_strength < -0.4 and metrics.momentum_score > 0.2):
            reversal_score += 0.3
        if metrics.volume_profile > 1.8:
            reversal_score += 0.2
        scores[MarketRegime.REVERSAL] = reversal_score

        # Find highest scoring regime
        best_regime = max(scores.keys(), key=lambda k: scores[k])
        confidence = scores[best_regime]

        return best_regime, confidence

    def update_regime(self, symbol: str = "BTC/USDT") -> Optional[MarketRegime]:
        """Update current market regime classification."""
        metrics = self.calculate_regime_metrics(symbol)
        if not metrics:
            return None

        new_regime, confidence = self.classify_regime(metrics)

        # Only update if confidence is high enough
        if confidence < self.regime_thresholds['min_confidence']:
            return self.current_regime

        # Check for regime transition
        if new_regime != self.current_regime:
            transition = RegimeTransition(
                from_regime=self.current_regime,
                to_regime=new_regime,
                confidence=confidence,
                trigger_metrics={
                    'trend_strength': metrics.trend_strength,
                    'volatility_percentile': metrics.volatility_percentile,
                    'momentum_score': metrics.momentum_score,
                    'adx_strength': metrics.adx_strength
                },
                timestamp=datetime.now(),
                duration_hours=(datetime.now() - self.regime_start_time).total_seconds() / 3600
            )

            self.regime_history.append(transition)

            self.logger.info(f"Regime transition: {self.current_regime.value} → {new_regime.value}",
                           confidence=confidence,
                           duration_hours=transition.duration_hours)

            self.current_regime = new_regime
            self.regime_confidence = confidence
            self.regime_start_time = datetime.now()
        else:
            # Update confidence for current regime
            self.regime_confidence = confidence

        return self.current_regime

    def get_current_strategy_config(self) -> Dict[str, Any]:
        """Get strategy configuration for current regime."""
        return self.regime_strategies.get(self.current_regime,
                                        self.regime_strategies[MarketRegime.SIDEWAYS_LOW_VOL])

    def get_regime_transition_history(self, hours: int = 168) -> List[RegimeTransition]:
        """Get regime transitions from last N hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [t for t in self.regime_history if t.timestamp >= cutoff_time]

    def calculate_regime_stability(self, hours: int = 24) -> float:
        """Calculate regime stability score (fewer transitions = more stable)."""
        recent_transitions = self.get_regime_transition_history(hours)
        if not recent_transitions:
            return 1.0

        # Penalize frequent transitions
        stability = max(0.0, 1.0 - len(recent_transitions) / 10)
        return stability

    def get_regime_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for each regime."""
        if not self.regime_history:
            return {}

        regime_stats = {}
        for regime in MarketRegime:
            regime_transitions = [t for t in self.regime_history if t.to_regime == regime]

            if regime_transitions:
                durations = [t.duration_hours for t in regime_transitions if t.duration_hours > 0]
                confidences = [t.confidence for t in regime_transitions]

                regime_stats[regime.value] = {
                    'occurrences': len(regime_transitions),
                    'avg_duration_hours': np.mean(durations) if durations else 0.0,
                    'avg_confidence': np.mean(confidences),
                    'total_time_percentage': sum(durations) / (24 * 7) * 100 if durations else 0.0  # % of week
                }

        return regime_stats

    def save_regime_state(self) -> None:
        """Save current regime state and history."""
        state = {
            'current_regime': self.current_regime.value,
            'regime_confidence': self.regime_confidence,
            'regime_start_time': self.regime_start_time.isoformat(),
            'regime_history': [
                {
                    'from_regime': t.from_regime.value,
                    'to_regime': t.to_regime.value,
                    'confidence': t.confidence,
                    'trigger_metrics': t.trigger_metrics,
                    'timestamp': t.timestamp.isoformat(),
                    'duration_hours': t.duration_hours
                }
                for t in self.regime_history[-100:]  # Keep last 100 transitions
            ]
        }

        try:
            with open(self.data_path / "regime_state.json", 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save regime state: {e}")

    def load_regime_state(self) -> None:
        """Load previous regime state and history."""
        try:
            state_file = self.data_path / "regime_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)

                self.current_regime = MarketRegime(state['current_regime'])
                self.regime_confidence = state['regime_confidence']
                self.regime_start_time = datetime.fromisoformat(state['regime_start_time'])

                # Reconstruct regime history
                self.regime_history = []
                for t_data in state['regime_history']:
                    transition = RegimeTransition(
                        from_regime=MarketRegime(t_data['from_regime']),
                        to_regime=MarketRegime(t_data['to_regime']),
                        confidence=t_data['confidence'],
                        trigger_metrics=t_data['trigger_metrics'],
                        timestamp=datetime.fromisoformat(t_data['timestamp']),
                        duration_hours=t_data['duration_hours']
                    )
                    self.regime_history.append(transition)

                self.logger.info("Regime state loaded from disk")
        except Exception as e:
            self.logger.warning(f"Failed to load regime state: {e}")


def create_regime_detector(lookback_periods: int = 252) -> RegimeDetector:
    """Factory function to create RegimeDetector instance."""
    return RegimeDetector(lookback_periods=lookback_periods)
