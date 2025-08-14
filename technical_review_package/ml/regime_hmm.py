#!/usr/bin/env python3
"""
Regime Detection using Hidden Markov Models (HMM)
Market regime classification for adaptive trading strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import warnings

warnings.filterwarnings("ignore")

# Try to import hmmlearn, fallback to custom implementation if not available
try:
    from hmmlearn.hmm import GaussianHMM

    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    print("hmmlearn not available - using fallback regime detection")


def fit_hmm(returns: np.ndarray, n_components: int = 3) -> np.ndarray:
    """
    Fit HMM model to returns for regime detection

    Args:
        returns: Time series of returns
        n_components: Number of market regimes (default: 3 = bear/neutral/bull)

    Returns:
        regimes: Array of regime labels
    """

    if HMMLEARN_AVAILABLE:
        try:
            model = GaussianHMM(
                n_components=n_components, covariance_type="diag", n_iter=200, random_state=42
            )

            # Fit model
            model.fit(returns.reshape(-1, 1))

            # Predict regimes
            regimes = model.predict(returns.reshape(-1, 1))

            return regimes

        except Exception as e:
            print(f"HMM fitting failed: {e}, using fallback")
            return _fallback_regime_detection(returns, n_components)
    else:
        return _fallback_regime_detection(returns, n_components)


def _fallback_regime_detection(returns: np.ndarray, n_components: int = 3) -> np.ndarray:
    """
    Fallback regime detection using volatility and trend analysis
    """

    # Calculate rolling statistics
    window = min(30, len(returns) // 4)

    rolling_mean = pd.Series(returns).rolling(window).mean()
    rolling_std = pd.Series(returns).rolling(window).std()

    # Define regimes based on volatility and trend
    regimes = np.zeros(len(returns), dtype=int)

    # High volatility = regime 0 (bear/crisis)
    # Low volatility + positive trend = regime 2 (bull)
    # Everything else = regime 1 (neutral)

    high_vol_threshold = np.percentile(rolling_std.dropna(), 75)
    positive_trend_threshold = 0

    for i in range(len(returns)):
        if i >= window:
            vol = rolling_std.iloc[i]
            trend = rolling_mean.iloc[i]

            if vol > high_vol_threshold:
                regimes[i] = 0  # High volatility (bear)
            elif trend > positive_trend_threshold and vol < high_vol_threshold:
                regimes[i] = 2  # Low vol + positive trend (bull)
            else:
                regimes[i] = 1  # Neutral

    return regimes


class MarketRegimeDetector:
    """
    Advanced market regime detection with HMM and additional features
    """

    def __init__(self, n_regimes: int = 3, lookback_window: int = 252):
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window

        self.model = None
        self.regime_labels = {0: "bear", 1: "neutral", 2: "bull"}
        self.regime_stats = {}

    def fit(self, prices: np.ndarray) -> "MarketRegimeDetector":
        """
        Fit regime detection model to price data
        """

        # Calculate returns
        returns = np.diff(np.log(prices))

        # Fit HMM model
        regimes = fit_hmm(returns, self.n_regimes)

        # Calculate regime statistics
        self._calculate_regime_stats(returns, regimes)

        # Store for prediction
        self.last_returns = returns[-self.lookback_window :]
        self.last_regimes = regimes[-self.lookback_window :]

        return self

    def predict_current_regime(self, recent_returns: np.ndarray) -> Dict[str, Any]:
        """
        Predict current market regime
        """

        if len(recent_returns) == 0:
            return {"regime": 1, "regime_name": "neutral", "confidence": 0.5}

        # Use most recent returns for prediction
        current_regimes = fit_hmm(recent_returns, self.n_regimes)

        # Get most recent regime
        current_regime = current_regimes[-1]
        regime_name = self.regime_labels.get(current_regime, "unknown")

        # Calculate confidence based on regime stability
        recent_window = min(10, len(current_regimes))
        recent_regime_consistency = np.mean(current_regimes[-recent_window:] == current_regime)

        return {
            "regime": int(current_regime),
            "regime_name": regime_name,
            "confidence": float(recent_regime_consistency),
            "regime_sequence": current_regimes[-recent_window:].tolist(),
        }

    def _calculate_regime_stats(self, returns: np.ndarray, regimes: np.ndarray):
        """
        Calculate statistics for each regime
        """

        for regime_id in range(self.n_regimes):
            regime_mask = regimes == regime_id
            regime_returns = returns[regime_mask]

            if len(regime_returns) > 0:
                stats = {
                    "mean_return": np.mean(regime_returns),
                    "volatility": np.std(regime_returns),
                    "frequency": np.mean(regime_mask),
                    "sharpe_ratio": np.mean(regime_returns) / max(np.std(regime_returns), 1e-8),
                    "max_drawdown": self._calculate_max_drawdown(regime_returns),
                }
            else:
                stats = {
                    "mean_return": 0,
                    "volatility": 0,
                    "frequency": 0,
                    "sharpe_ratio": 0,
                    "max_drawdown": 0,
                }

            regime_name = self.regime_labels[regime_id]
            self.regime_stats[regime_name] = stats

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown for returns
        """

        if len(returns) == 0:
            return 0

        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max

        return float(np.min(drawdown))

    def get_regime_summary(self) -> Dict[str, Any]:
        """
        Get summary of all detected regimes
        """

        return {
            "n_regimes": self.n_regimes,
            "regime_stats": self.regime_stats,
            "regime_labels": self.regime_labels,
        }


def detect_crypto_market_regime(prices: List[float], symbol: str = "BTC") -> Dict[str, Any]:
    """
    Detect current crypto market regime for a given symbol
    """

    if len(prices) < 30:
        return {
            "symbol": symbol,
            "regime": 1,
            "regime_name": "neutral",
            "confidence": 0.5,
            "error": "Insufficient data for regime detection",
        }

    try:
        # Convert to numpy array
        prices_array = np.array(prices)

        # Initialize detector
        detector = MarketRegimeDetector(n_regimes=3, lookback_window=min(252, len(prices) // 2))

        # Fit and predict
        detector.fit(prices_array)

        # Use recent returns for current prediction
        recent_returns = np.diff(np.log(prices_array[-30:]))
        current_regime = detector.predict_current_regime(recent_returns)

        result = {
            "symbol": symbol,
            "regime": current_regime["regime"],
            "regime_name": current_regime["regime_name"],
            "confidence": current_regime["confidence"],
            "regime_stats": detector.get_regime_summary(),
            "data_points": len(prices),
        }

        return result

    except Exception as e:
        return {
            "symbol": symbol,
            "regime": 1,
            "regime_name": "neutral",
            "confidence": 0.5,
            "error": str(e),
        }


if __name__ == "__main__":
    print("📊 TESTING MARKET REGIME DETECTION")
    print("=" * 50)

    # Generate synthetic market data with different regimes
    np.random.seed(42)

    # Bear market (high vol, negative trend)
    bear_returns = np.random.normal(-0.02, 0.05, 100)

    # Bull market (low vol, positive trend)
    bull_returns = np.random.normal(0.015, 0.02, 100)

    # Neutral market (medium vol, no trend)
    neutral_returns = np.random.normal(0.001, 0.03, 100)

    # Combine into single series
    all_returns = np.concatenate([bear_returns, neutral_returns, bull_returns])

    # Convert to price series
    prices = np.exp(np.cumsum(all_returns)) * 100

    print(f"📈 Generated {len(prices)} price points")

    # Test regime detection
    print("\n🔍 Testing regime detection...")

    # Fit HMM model directly
    regimes = fit_hmm(all_returns, n_components=3)

    print(f"   Detected regimes: {np.unique(regimes)}")

    # Count regime frequency
    for regime_id in np.unique(regimes):
        count = np.sum(regimes == regime_id)
        percentage = count / len(regimes) * 100
        print(f"   Regime {regime_id}: {count} periods ({percentage:.1f}%)")

    # Test MarketRegimeDetector
    print("\n🎯 Testing MarketRegimeDetector...")

    detector = MarketRegimeDetector(n_regimes=3)
    detector.fit(prices)

    # Get current regime
    recent_returns = np.diff(np.log(prices[-30:]))
    current_regime = detector.predict_current_regime(recent_returns)

    print(
        f"   Current regime: {current_regime['regime_name']} (confidence: {current_regime['confidence']:.3f})"
    )

    # Show regime statistics
    print(f"\n📊 Regime statistics:")
    regime_summary = detector.get_regime_summary()

    for regime_name, stats in regime_summary["regime_stats"].items():
        print(f"   {regime_name.upper()} regime:")
        print(f"      Mean return: {stats['mean_return']:.4f}")
        print(f"      Volatility: {stats['volatility']:.4f}")
        print(f"      Frequency: {stats['frequency']:.3f}")
        print(f"      Sharpe ratio: {stats['sharpe_ratio']:.3f}")

    # Test crypto regime detection function
    print(f"\n🪙 Testing crypto regime detection...")

    crypto_regime = detect_crypto_market_regime(prices.tolist(), "TEST")

    print(f"   Symbol: {crypto_regime['symbol']}")
    print(f"   Current regime: {crypto_regime['regime_name']}")
    print(f"   Confidence: {crypto_regime['confidence']:.3f}")

    print("\n✅ Regime detection test completed")
