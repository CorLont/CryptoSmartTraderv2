"""
Enhanced Technical Analysis Agent
Addresses: compute bottlenecks, indicator inflation, regime detection
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import json
import time

from utils.daily_logger import get_daily_logger


@dataclass
class TechnicalSignal:
    """Enhanced technical signal with regime context"""

    coin: str
    timeframe: str
    signal_strength: float  # -1 to 1
    confidence: float  # 0 to 1
    regime: str  # bull, bear, sideways, volatile
    active_indicators: List[str]
    regime_confidence: float
    computation_time: float
    timestamp: datetime


@dataclass
class MarketRegime:
    """Market regime classification"""

    regime_type: str  # bull, bear, sideways, volatile
    confidence: float
    trend_strength: float
    volatility_level: float
    indicators_used: List[str]


class IndicatorSelection:
    """Dynamic indicator selection based on market conditions"""

    def __init__(self):
        self.indicator_registry = {
            "trend": {
                "sma_20": {"weight": 1.0, "regimes": ["bull", "bear"]},
                "ema_12": {"weight": 1.2, "regimes": ["bull", "bear"]},
                "macd": {"weight": 1.5, "regimes": ["bull", "bear"]},
                "adx": {"weight": 1.3, "regimes": ["bull", "bear"]},
                "parabolic_sar": {"weight": 1.1, "regimes": ["bull", "bear"]},
            },
            "momentum": {
                "rsi": {"weight": 1.0, "regimes": ["all"]},
                "stoch": {"weight": 0.9, "regimes": ["sideways", "volatile"]},
                "williams_r": {"weight": 0.8, "regimes": ["sideways"]},
                "roc": {"weight": 1.1, "regimes": ["bull", "bear"]},
            },
            "volatility": {
                "bollinger": {"weight": 1.2, "regimes": ["volatile", "sideways"]},
                "atr": {"weight": 1.0, "regimes": ["all"]},
                "keltner": {"weight": 0.9, "regimes": ["volatile"]},
            },
            "volume": {
                "volume_sma": {"weight": 1.0, "regimes": ["all"]},
                "obv": {"weight": 1.1, "regimes": ["bull", "bear"]},
                "mfi": {"weight": 0.9, "regimes": ["all"]},
                "vwap": {"weight": 1.3, "regimes": ["all"]},
            },
        }

    def select_indicators(self, regime: str, max_indicators: int = 12) -> List[str]:
        """Select optimal indicators for current regime"""
        candidates = []

        for category, indicators in self.indicator_registry.items():
            for indicator, config in indicators.items():
                regimes = config["regimes"]
                if regime in regimes or "all" in regimes:
                    score = config["weight"]
                    if regime in regimes:
                        score *= 1.2  # Boost regime-specific indicators
                    candidates.append((indicator, score))

        # Sort by score and take top indicators
        candidates.sort(key=lambda x: x[1], reverse=True)
        selected = [indicator for indicator, _ in candidates[:max_indicators]]

        return selected


class RegimeDetector:
    """Advanced market regime detection"""

    def __init__(self):
        self.logger = get_daily_logger().get_logger("technical")

    def detect_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Detect current market regime"""
        if len(df) < 50:
            return MarketRegime("sideways", 0.5, 0.0, 0.5, [])

        # Calculate regime indicators
        indicators_used = []

        # Trend strength using ADX
        adx = self._calculate_adx(df)
        current_adx = adx.iloc[-1] if not adx.empty else 25
        indicators_used.append("ADX")

        # Price momentum
        price_change_20 = (df["close"].iloc[-1] / df["close"].iloc[-20] - 1) * 100
        price_change_50 = (df["close"].iloc[-1] / df["close"].iloc[-50] - 1) * 100
        indicators_used.extend(["Price_20D", "Price_50D"])

        # Volatility
        returns = df["close"].pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252) * 100
        indicators_used.append("Volatility")

        # Volume trend
        volume_ma = df["volume"].rolling(20).mean()
        volume_trend = (volume_ma.iloc[-1] / volume_ma.iloc[-10] - 1) * 100
        indicators_used.append("Volume_Trend")

        # Regime classification logic
        regime_type = "sideways"
        confidence = 0.5
        trend_strength = current_adx / 100
        volatility_level = min(volatility / 50, 1.0)  # Normalize

        if current_adx > 25 and price_change_20 > 5:
            regime_type = "bull"
            confidence = min((current_adx - 25) / 50 + abs(price_change_20) / 20, 1.0)
        elif current_adx > 25 and price_change_20 < -5:
            regime_type = "bear"
            confidence = min((current_adx - 25) / 50 + abs(price_change_20) / 20, 1.0)
        elif volatility > 40:
            regime_type = "volatile"
            confidence = min(volatility / 80, 1.0)
        else:
            regime_type = "sideways"
            confidence = 1.0 - min(current_adx / 50, 1.0)

        return MarketRegime(
            regime_type=regime_type,
            confidence=confidence,
            trend_strength=trend_strength,
            volatility_level=volatility_level,
            indicators_used=indicators_used,
        )

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index"""
        try:
            high = df["high"]
            low = df["low"]
            close = df["close"]

            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # Directional Movement
            dm_plus = np.where(
                (high - high.shift()) > (low.shift() - low), np.maximum(high - high.shift(), 0), 0
            )
            dm_minus = np.where(
                (low.shift() - low) > (high - high.shift()), np.maximum(low.shift() - low, 0), 0
            )

            # Smoothed averages
            tr_smooth = pd.Series(tr).rolling(period).mean()
            dm_plus_smooth = pd.Series(dm_plus).rolling(period).mean()
            dm_minus_smooth = pd.Series(dm_minus).rolling(period).mean()

            # Directional Indicators
            di_plus = (dm_plus_smooth / tr_smooth) * 100
            di_minus = (dm_minus_smooth / tr_smooth) * 100

            # ADX
            dx = abs(di_plus - di_minus) / (di_plus + di_minus) * 100
            adx = dx.rolling(period).mean()

            return adx
        except Exception:
            return pd.Series([25] * len(df), index=df.index)


class ParallelTechnicalComputer:
    """High-performance parallel technical analysis computation"""

    def __init__(self, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.thread_executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_executor = ProcessPoolExecutor(max_workers=self.max_workers // 2)
        self.logger = get_daily_logger().get_logger("technical")

    async def compute_indicators_parallel(
        self, coin_data: Dict[str, pd.DataFrame], selected_indicators: List[str]
    ) -> Dict[str, Dict]:
        """Compute indicators for multiple coins in parallel"""

        start_time = time.time()

        # Group coins into batches for processing
        coins = list(coin_data.keys())
        batch_size = max(1, len(coins) // self.max_workers)
        batches = [coins[i : i + batch_size] for i in range(0, len(coins), batch_size)]

        # Process batches in parallel
        loop = asyncio.get_event_loop()
        tasks = []

        for batch in batches:
            task = loop.run_in_executor(
                self.thread_executor,
                self._compute_batch_indicators,
                batch,
                {coin: coin_data[coin] for coin in batch},
                selected_indicators,
            )
            tasks.append(task)

        # Collect results
        batch_results = await asyncio.gather(*tasks)

        # Merge results
        all_results = {}
        for batch_result in batch_results:
            all_results.update(batch_result)

        computation_time = time.time() - start_time
        self.logger.info(f"Computed indicators for {len(coins)} coins in {computation_time:.3f}s")

        return all_results

    def _compute_batch_indicators(
        self, coins: List[str], coin_data: Dict[str, pd.DataFrame], indicators: List[str]
    ) -> Dict[str, Dict]:
        """Compute indicators for a batch of coins"""
        results = {}

        for coin in coins:
            try:
                df = coin_data[coin]
                if len(df) < 20:  # Skip insufficient data
                    continue

                indicators_result = {}

                for indicator in indicators:
                    value = self._compute_single_indicator(df, indicator)
                    if value is not None:
                        indicators_result[indicator] = value

                results[coin] = indicators_result

            except Exception as e:
                logging.error(f"Error computing indicators for {coin}: {e}")
                continue

        return results

    def _compute_single_indicator(self, df: pd.DataFrame, indicator: str) -> Optional[float]:
        """Compute single indicator efficiently"""
        try:
            if indicator == "rsi":
                return self._rsi(df["close"])
            elif indicator == "macd":
                return self._macd_signal(df["close"])
            elif indicator == "sma_20":
                return df["close"].rolling(20).mean().iloc[-1]
            elif indicator == "ema_12":
                return df["close"].ewm(span=12).mean().iloc[-1]
            elif indicator == "bollinger":
                return self._bollinger_position(df["close"])
            elif indicator == "atr":
                return self._atr(df)
            elif indicator == "volume_sma":
                return df["volume"].rolling(20).mean().iloc[-1]
            elif indicator == "obv":
                return self._obv(df)
            # Add more indicators as needed
            else:
                return None
        except Exception:
            return None

    def _rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def _macd_signal(self, prices: pd.Series) -> float:
        """Calculate MACD signal"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return (macd - signal).iloc[-1]

    def _bollinger_position(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate position within Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * 2)
        lower = sma - (std * 2)
        current_price = prices.iloc[-1]

        # Return position: 0 = lower band, 0.5 = middle, 1 = upper band
        return (current_price - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])

    def _atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range"""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.rolling(period).mean().iloc[-1]

    def _obv(self, df: pd.DataFrame) -> float:
        """Calculate On-Balance Volume"""
        obv = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
        return obv.iloc[-1]


class EnhancedTechnicalAgent:
    """Professional technical analysis with regime detection and optimization"""

    def __init__(self):
        self.logger = get_daily_logger().get_logger("technical")
        self.regime_detector = RegimeDetector()
        self.indicator_selector = IndicatorSelection()
        self.parallel_computer = ParallelTechnicalComputer()
        self.cache = {}
        self.cache_ttl = 60  # 1 minute

    async def analyze_multiple_coins(
        self, coin_data: Dict[str, pd.DataFrame], timeframes: List[str]
    ) -> Dict[str, Dict[str, TechnicalSignal]]:
        """Analyze multiple coins across timeframes efficiently"""

        results = {}

        for timeframe in timeframes:
            self.logger.info(f"Analyzing {len(coin_data)} coins for {timeframe}")

            # Detect regime for each coin
            regimes = {}
            for coin, df in coin_data.items():
                regime = self.regime_detector.detect_regime(df)
                regimes[coin] = regime

            # Select indicators based on most common regime
            regime_counts = {}
            for regime in regimes.values():
                regime_counts[regime.regime_type] = regime_counts.get(regime.regime_type, 0) + 1

            dominant_regime = max(regime_counts, key=regime_counts.get)
            selected_indicators = self.indicator_selector.select_indicators(dominant_regime)

            self.logger.info(
                f"Selected {len(selected_indicators)} indicators for {dominant_regime} regime"
            )

            # Compute indicators in parallel
            start_time = time.time()
            indicator_results = await self.parallel_computer.compute_indicators_parallel(
                coin_data, selected_indicators
            )
            computation_time = time.time() - start_time

            # Generate signals
            for coin in coin_data.keys():
                if coin not in results:
                    results[coin] = {}

                if coin in indicator_results:
                    signal = self._generate_signal(
                        coin,
                        timeframe,
                        indicator_results[coin],
                        regimes.get(coin),
                        selected_indicators,
                        computation_time,
                    )
                    results[coin][timeframe] = signal

        return results

    def _generate_signal(
        self,
        coin: str,
        timeframe: str,
        indicators: Dict[str, float],
        regime: Optional[MarketRegime],
        active_indicators: List[str],
        computation_time: float,
    ) -> TechnicalSignal:
        """Generate trading signal from indicators"""

        if not indicators or not regime:
            return TechnicalSignal(
                coin=coin,
                timeframe=timeframe,
                signal_strength=0.0,
                confidence=0.0,
                regime="unknown",
                active_indicators=[],
                regime_confidence=0.0,
                computation_time=computation_time,
                timestamp=datetime.now(),
            )

        # Calculate signal based on regime-weighted indicators
        signal_components = []

        # RSI signal
        if "rsi" in indicators:
            rsi = indicators["rsi"]
            if rsi < 30:
                signal_components.append(0.6)  # Oversold - buy signal
            elif rsi > 70:
                signal_components.append(-0.6)  # Overbought - sell signal
            else:
                signal_components.append(0.0)

        # MACD signal
        if "macd" in indicators:
            macd = indicators["macd"]
            signal_components.append(np.tanh(macd * 2))  # Normalize to -1,1

        # Bollinger position
        if "bollinger" in indicators:
            bb_pos = indicators["bollinger"]
            if bb_pos < 0.2:
                signal_components.append(0.4)
            elif bb_pos > 0.8:
                signal_components.append(-0.4)
            else:
                signal_components.append(0.0)

        # Calculate final signal
        if signal_components:
            signal_strength = np.mean(signal_components)
            confidence = (
                min(len(signal_components) / len(active_indicators), 1.0) * regime.confidence
            )
        else:
            signal_strength = 0.0
            confidence = 0.0

        return TechnicalSignal(
            coin=coin,
            timeframe=timeframe,
            signal_strength=signal_strength,
            confidence=confidence,
            regime=regime.regime_type,
            active_indicators=active_indicators,
            regime_confidence=regime.confidence,
            computation_time=computation_time,
            timestamp=datetime.now(),
        )

    def get_status(self) -> Dict:
        """Get agent status"""
        return {
            "agent": "enhanced_technical",
            "status": "operational",
            "max_workers": self.parallel_computer.max_workers,
            "cache_size": len(self.cache),
            "regime_detection": True,
            "dynamic_indicators": True,
            "parallel_processing": True,
        }


# Global instance
technical_agent = EnhancedTechnicalAgent()
