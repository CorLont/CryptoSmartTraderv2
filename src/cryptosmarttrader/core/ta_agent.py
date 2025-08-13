#!/usr/bin/env python3
"""
Technical Analysis Agent - Enterprise lightweight TA with authentic indicators

Async TA agent providing SMA, RSI, MACD, and Bollinger Bands with proper mathematical
implementations: EMA-based MACD, Wilder-smoothing RSI, and authentic Bollinger Bands.
"""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import logging

try:
    from ..core.consolidated_logging_manager import get_consolidated_logger
except ImportError:

    def get_consolidated_logger(name: str) -> logging.Logger:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


# Technical Analysis Library fallback
try:
    import talib

    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False


@dataclass
class TASignal:
    """Technical analysis signal result"""

    indicator: str
    signal_type: str  # 'buy', 'sell', 'neutral'
    strength: float  # 0.0 to 1.0
    value: float
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TAResult:
    """Complete technical analysis result"""

    symbol: str
    timeframe: str
    indicators: Dict[str, Any]
    signals: List[TASignal]
    overall_signal: str
    confidence: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TechnicalAnalysisAgent:
    """
    Enterprise Technical Analysis Agent with authentic indicators

    Provides proper MACD (EMA-based), RSI (Wilder-smoothing), Bollinger Bands
    (authentic calculation), and SMA with no synthetic fallback data.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TA agent

        Args:
            config: Optional TA configuration
        """
        self.logger = get_consolidated_logger("TechnicalAnalysisAgent")

        # Load configuration
        self.config = self._load_config(config)

        # TA state
        self.last_analysis: Optional[TAResult] = None
        self.analysis_history: List[TAResult] = []

        # Performance tracking
        self.analysis_count = 0
        self.total_analysis_time = 0.0

        self.logger.info(
            f"Technical Analysis Agent initialized {'with TA-Lib' if TALIB_AVAILABLE else 'with native implementation'}"
        )

    def _load_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load TA configuration with enterprise defaults"""

        default_config = {
            # Indicator parameters
            "indicators": {
                "sma": {"periods": [20, 50, 200], "enabled": True},
                "rsi": {
                    "period": 14,
                    "overbought": 70,
                    "oversold": 30,
                    "enabled": True,
                    "use_wilder_smoothing": True,  # Authentic RSI calculation
                },
                "macd": {
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9,
                    "enabled": True,
                    "use_ema": True,  # Authentic MACD with EMA
                },
                "bollinger": {
                    "period": 20,
                    "std_dev": 2.0,
                    "enabled": True,
                    "min_data_points": 20,  # No fallback below this
                },
            },
            # Signal generation
            "signals": {
                "strength_thresholds": {"strong": 0.8, "moderate": 0.6, "weak": 0.4},
                "rsi_divergence_enabled": True,
                "macd_crossover_enabled": True,
                "bollinger_breakout_enabled": True,
            },
            # Data integrity
            "data_quality": {
                "min_data_points": 50,  # Minimum data for reliable analysis
                "max_missing_percentage": 0.05,  # 5% max missing data
                "require_recent_data": True,
                "max_age_hours": 24,
            },
            # Performance
            "async_enabled": True,
            "concurrent_indicators": True,
            "cache_results": True,
            "cache_ttl_seconds": 300,
        }

        if config:
            self._deep_merge_dict(default_config, config)

        return default_config

    async def analyze_symbol(
        self, symbol: str, price_data: pd.DataFrame, timeframe: str = "1h"
    ) -> TAResult:
        """
        Perform comprehensive technical analysis on symbol

        Args:
            symbol: Trading symbol
            price_data: OHLCV data with columns: open, high, low, close, volume
            timeframe: Analysis timeframe

        Returns:
            TAResult with indicators and signals
        """

        start_time = datetime.now(timezone.utc)
        self.logger.info(f"Starting TA analysis for {symbol} ({timeframe})")

        try:
            # Validate input data
            if not self._validate_price_data(price_data):
                raise ValueError("Invalid or insufficient price data")

            # Calculate indicators
            if self.config["async_enabled"] and self.config["concurrent_indicators"]:
                indicators = await self._calculate_indicators_async(price_data)
            else:
                indicators = self._calculate_indicators_sync(price_data)

            # Generate signals
            signals = self._generate_signals(indicators, price_data)

            # Determine overall signal
            overall_signal, confidence = self._determine_overall_signal(signals)

            # Create result
            result = TAResult(
                symbol=symbol,
                timeframe=timeframe,
                indicators=indicators,
                signals=signals,
                overall_signal=overall_signal,
                confidence=confidence,
                timestamp=start_time,
            )

            # Update state
            self.last_analysis = result
            self.analysis_history.append(result)
            self._cleanup_history()

            # Update performance metrics
            self.analysis_count += 1
            analysis_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            self.total_analysis_time += analysis_time

            self.logger.info(
                f"TA analysis completed for {symbol}: {overall_signal} "
                f"(confidence: {confidence:.1%}, duration: {analysis_time:.2f}s)"
            )

            return result

        except Exception as e:
            self.logger.error(f"TA analysis failed for {symbol}: {e}")
            raise

    def _validate_price_data(self, data: pd.DataFrame) -> bool:
        """Validate price data quality and completeness"""

        required_columns = ["open", "high", "low", "close"]

        # Check required columns
        if not all(col in data.columns for col in required_columns):
            self.logger.error(
                f"Missing required columns. Expected: {required_columns}, Got: {list(data.columns)}"
            )
            return False

        # Check minimum data points
        min_points = self.config["data_quality"]["min_data_points"]
        if len(data) < min_points:
            self.logger.error(f"Insufficient data points: {len(data)} < {min_points}")
            return False

        # Check for missing data
        missing_percentage = data[required_columns].isnull().sum().sum() / (
            len(data) * len(required_columns)
        max_missing = self.config["data_quality"]["max_missing_percentage"]

        if missing_percentage > max_missing:
            self.logger.error(
                f"Too much missing data: {missing_percentage:.1%} > {max_missing:.1%}"
            )
            return False

        # Check data recency if required
        if self.config["data_quality"]["require_recent_data"]:
            if hasattr(data.index, "max"):
                latest_time = data.index.max()
                if pd.notna(latest_time):
                    age_hours = (datetime.now() - latest_time).total_seconds() / 3600
                    max_age = self.config["data_quality"]["max_age_hours"]

                    if age_hours > max_age:
                        self.logger.warning(f"Data may be stale: {age_hours:.1f}h > {max_age}h")

        return True

    async def _calculate_indicators_async(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate indicators asynchronously"""

        tasks = []

        if self.config["indicators"]["sma"]["enabled"]:
            tasks.append(self._calculate_sma_async(data))

        if self.config["indicators"]["rsi"]["enabled"]:
            tasks.append(self._calculate_rsi_async(data))

        if self.config["indicators"]["macd"]["enabled"]:
            tasks.append(self._calculate_macd_async(data))

        if self.config["indicators"]["bollinger"]["enabled"]:
            tasks.append(self._calculate_bollinger_async(data))

        # Execute all calculations concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Combine results
        indicators = {}
        for result in results:
            if isinstance(result, Exception):
                self.logger.warning(f"Indicator calculation failed: {result}")
            elif isinstance(result, dict):
                indicators.update(result)

        return indicators

    def _calculate_indicators_sync(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate indicators synchronously"""

        indicators = {}

        try:
            if self.config["indicators"]["sma"]["enabled"]:
                indicators.update(self._calculate_sma(data))
        except Exception as e:
            self.logger.warning(f"SMA calculation failed: {e}")

        try:
            if self.config["indicators"]["rsi"]["enabled"]:
                indicators.update(self._calculate_rsi(data))
        except Exception as e:
            self.logger.warning(f"RSI calculation failed: {e}")

        try:
            if self.config["indicators"]["macd"]["enabled"]:
                indicators.update(self._calculate_macd(data))
        except Exception as e:
            self.logger.warning(f"MACD calculation failed: {e}")

        try:
            if self.config["indicators"]["bollinger"]["enabled"]:
                indicators.update(self._calculate_bollinger(data))
        except Exception as e:
            self.logger.warning(f"Bollinger Bands calculation failed: {e}")

        return indicators

    async def _calculate_sma_async(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Simple Moving Averages asynchronously"""
        return await asyncio.to_thread(self._calculate_sma, data)

    def _calculate_sma(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Simple Moving Averages"""

        close_prices = data["close"]
        periods = self.config["indicators"]["sma"]["periods"]

        sma_results = {}

        for period in periods:
            if len(close_prices) >= period:
                if TALIB_AVAILABLE:
                    sma_values = talib.SMA(close_prices.values, timeperiod=period)
                else:
                    sma_values = close_prices.rolling(window=period).mean()

                sma_results[f"sma_{period}"] = {
                    "values": sma_values,
                    "current": float(sma_values.iloc[-1])
                    if hasattr(sma_values, "iloc")
                    else float(sma_values[-1]),
                    "period": period,
                }
            else:
                self.logger.warning(
                    f"Insufficient data for SMA-{period}: {len(close_prices)} < {period}"
                )

        return {"sma": sma_results}

    async def _calculate_rsi_async(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate RSI asynchronously"""
        return await asyncio.to_thread(self._calculate_rsi, data)

    def _calculate_rsi(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate RSI with WILDER SMOOTHING (authentic implementation)"""

        close_prices = data["close"]
        period = self.config["indicators"]["rsi"]["period"]
        use_wilder = self.config["indicators"]["rsi"]["use_wilder_smoothing"]

        if len(close_prices) < period + 1:
            self.logger.warning(f"Insufficient data for RSI: {len(close_prices)} < {period + 1}")
            return {"rsi": None}

        try:
            if TALIB_AVAILABLE:
                # TA-Lib uses Wilder's smoothing by default
                rsi_values = talib.RSI(close_prices.values, timeperiod=period)
            else:
                # Manual calculation with Wilder smoothing option
                price_changes = close_prices.diff()
                gains = price_changes.where(price_changes > 0, 0)
                losses = -price_changes.where(price_changes < 0, 0)

                if use_wilder:
                    # Wilder's smoothing (authentic RSI)
                    alpha = 1.0 / period
                    avg_gains = gains.ewm(alpha=alpha, adjust=False).mean()
                    avg_losses = losses.ewm(alpha=alpha, adjust=False).mean()
                else:
                    # Simple moving average (approximation)
                    avg_gains = gains.rolling(window=period).mean()
                    avg_losses = losses.rolling(window=period).mean()

                rs = avg_gains / avg_losses
                rsi_values = 100 - (100 / (1 + rs))

            current_rsi = (
                float(rsi_values.iloc[-1]) if hasattr(rsi_values, "iloc") else float(rsi_values[-1])

            return {
                "rsi": {
                    "values": rsi_values,
                    "current": current_rsi,
                    "period": period,
                    "overbought": self.config["indicators"]["rsi"]["overbought"],
                    "oversold": self.config["indicators"]["rsi"]["oversold"],
                    "wilder_smoothing": use_wilder,
                }
            }

        except Exception as e:
            self.logger.error(f"RSI calculation failed: {e}")
            return {"rsi": None}

    async def _calculate_macd_async(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate MACD asynchronously"""
        return await asyncio.to_thread(self._calculate_macd, data)

    def _calculate_macd(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate MACD with AUTHENTIC EMA IMPLEMENTATION"""

        close_prices = data["close"]
        fast_period = self.config["indicators"]["macd"]["fast_period"]
        slow_period = self.config["indicators"]["macd"]["slow_period"]
        signal_period = self.config["indicators"]["macd"]["signal_period"]
        use_ema = self.config["indicators"]["macd"]["use_ema"]

        if len(close_prices) < slow_period + signal_period:
            self.logger.warning(
                f"Insufficient data for MACD: {len(close_prices)} < {slow_period + signal_period}"
            )
            return {"macd": None}

        try:
            if TALIB_AVAILABLE:
                # TA-Lib uses EMA by default (authentic MACD)
                macd_line, macd_signal, macd_histogram = talib.MACD(
                    close_prices.values,
                    fastperiod=fast_period,
                    slowperiod=slow_period,
                    signalperiod=signal_period,
                )
            else:
                if use_ema:
                    # Authentic MACD with EMA
                    fast_ema = close_prices.ewm(span=fast_period).mean()
                    slow_ema = close_prices.ewm(span=slow_period).mean()
                    macd_line = fast_ema - slow_ema
                    macd_signal = macd_line.ewm(span=signal_period).mean()
                else:
                    # Simple moving average approximation (not recommended)
                    fast_sma = close_prices.rolling(window=fast_period).mean()
                    slow_sma = close_prices.rolling(window=slow_period).mean()
                    macd_line = fast_sma - slow_sma
                    macd_signal = macd_line.rolling(window=signal_period).mean()

                macd_histogram = macd_line - macd_signal

            current_macd = (
                float(macd_line.iloc[-1]) if hasattr(macd_line, "iloc") else float(macd_line[-1])
            current_signal = (
                float(macd_signal.iloc[-1])
                if hasattr(macd_signal, "iloc")
                else float(macd_signal[-1])
            current_histogram = (
                float(macd_histogram.iloc[-1])
                if hasattr(macd_histogram, "iloc")
                else float(macd_histogram[-1])

            return {
                "macd": {
                    "macd_line": macd_line,
                    "signal_line": macd_signal,
                    "histogram": macd_histogram,
                    "current_macd": current_macd,
                    "current_signal": current_signal,
                    "current_histogram": current_histogram,
                    "fast_period": fast_period,
                    "slow_period": slow_period,
                    "signal_period": signal_period,
                    "uses_ema": use_ema,
                }
            }

        except Exception as e:
            self.logger.error(f"MACD calculation failed: {e}")
            return {"macd": None}

    async def _calculate_bollinger_async(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Bollinger Bands asynchronously"""
        return await asyncio.to_thread(self._calculate_bollinger, data)

    def _calculate_bollinger(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate Bollinger Bands with AUTHENTIC CALCULATION (no fallback dummy data)"""

        close_prices = data["close"]
        period = self.config["indicators"]["bollinger"]["period"]
        std_dev_multiplier = self.config["indicators"]["bollinger"]["std_dev"]
        min_data_points = self.config["indicators"]["bollinger"]["min_data_points"]

        # AUTHENTIC IMPLEMENTATION: Return None if insufficient data
        if len(close_prices) < min_data_points:
            self.logger.warning(
                f"Insufficient data for Bollinger Bands: {len(close_prices)} < {min_data_points}"
            )
            return {"bollinger": None}

        try:
            if TALIB_AVAILABLE:
                upper_band, middle_band, lower_band = talib.BBANDS(
                    close_prices.values,
                    timeperiod=period,
                    nbdevup=std_dev_multiplier,
                    nbdevdn=std_dev_multiplier,
                    matype=0,  # Simple Moving Average
                )
            else:
                # Manual calculation with authentic statistics
                middle_band = close_prices.rolling(window=period).mean()
                rolling_std = close_prices.rolling(window=period).std()
                upper_band = middle_band + (rolling_std * std_dev_multiplier)
                lower_band = middle_band - (rolling_std * std_dev_multiplier)

            # Get current values (only if we have valid data)
            current_upper = (
                float(upper_band.iloc[-1]) if hasattr(upper_band, "iloc") else float(upper_band[-1])
            current_middle = (
                float(middle_band.iloc[-1])
                if hasattr(middle_band, "iloc")
                else float(middle_band[-1])
            current_lower = (
                float(lower_band.iloc[-1]) if hasattr(lower_band, "iloc") else float(lower_band[-1])
            current_price = float(close_prices.iloc[-1])

            # Calculate band width and position
            band_width = (current_upper - current_lower) / current_middle
            price_position = (current_price - current_lower) / (current_upper - current_lower)

            return {
                "bollinger": {
                    "upper_band": upper_band,
                    "middle_band": middle_band,
                    "lower_band": lower_band,
                    "current_upper": current_upper,
                    "current_middle": current_middle,
                    "current_lower": current_lower,
                    "band_width": band_width,
                    "price_position": price_position,
                    "period": period,
                    "std_dev": std_dev_multiplier,
                }
            }

        except Exception as e:
            self.logger.error(f"Bollinger Bands calculation failed: {e}")
            return {"bollinger": None}

    def _generate_signals(self, indicators: Dict[str, Any], data: pd.DataFrame) -> List[TASignal]:
        """Generate trading signals from indicators"""

        signals = []
        current_price = float(data["close"].iloc[-1])

        # RSI signals
        if indicators.get("rsi") and indicators["rsi"]["current"] is not None:
            rsi_signal = self._generate_rsi_signal(indicators["rsi"], current_price)
            if rsi_signal:
                signals.append(rsi_signal)

        # MACD signals
        if indicators.get("macd") and indicators["macd"]["current_macd"] is not None:
            macd_signal = self._generate_macd_signal(indicators["macd"], current_price)
            if macd_signal:
                signals.append(macd_signal)

        # Bollinger signals
        if indicators.get("bollinger") and indicators["bollinger"]["current_upper"] is not None:
            bb_signal = self._generate_bollinger_signal(indicators["bollinger"], current_price)
            if bb_signal:
                signals.append(bb_signal)

        # SMA signals
        if indicators.get("sma"):
            sma_signal = self._generate_sma_signal(indicators["sma"], current_price)
            if sma_signal:
                signals.append(sma_signal)

        return signals

    def _generate_rsi_signal(
        self, rsi_data: Dict[str, Any], current_price: float
    ) -> Optional[TASignal]:
        """Generate RSI-based signal"""

        current_rsi = rsi_data["current"]
        overbought = rsi_data["overbought"]
        oversold = rsi_data["oversold"]

        if current_rsi >= overbought:
            strength = min(1.0, (current_rsi - overbought) / (100 - overbought))
            return TASignal(
                indicator="rsi",
                signal_type="sell",
                strength=strength,
                value=current_rsi,
                message=f"RSI overbought at {current_rsi:.1f} (>{overbought})",
                metadata={"rsi_value": current_rsi, "threshold": overbought},
            )
        elif current_rsi <= oversold:
            strength = min(1.0, (oversold - current_rsi) / oversold)
            return TASignal(
                indicator="rsi",
                signal_type="buy",
                strength=strength,
                value=current_rsi,
                message=f"RSI oversold at {current_rsi:.1f} (<{oversold})",
                metadata={"rsi_value": current_rsi, "threshold": oversold},
            )

        return None

    def _generate_macd_signal(
        self, macd_data: Dict[str, Any], current_price: float
    ) -> Optional[TASignal]:
        """Generate MACD-based signal"""

        current_macd = macd_data["current_macd"]
        current_signal = macd_data["current_signal"]
        current_histogram = macd_data["current_histogram"]

        # MACD line crossing signal line
        if current_macd > current_signal and current_histogram > 0:
            strength = min(1.0, abs(current_histogram) / (abs(current_macd) + 1e-8))
            return TASignal(
                indicator="macd",
                signal_type="buy",
                strength=strength,
                value=current_macd,
                message=f"MACD bullish crossover (MACD: {current_macd:.4f} > Signal: {current_signal:.4f})",
                metadata={
                    "macd": current_macd,
                    "signal": current_signal,
                    "histogram": current_histogram,
                },
            )
        elif current_macd < current_signal and current_histogram < 0:
            strength = min(1.0, abs(current_histogram) / (abs(current_macd) + 1e-8))
            return TASignal(
                indicator="macd",
                signal_type="sell",
                strength=strength,
                value=current_macd,
                message=f"MACD bearish crossover (MACD: {current_macd:.4f} < Signal: {current_signal:.4f})",
                metadata={
                    "macd": current_macd,
                    "signal": current_signal,
                    "histogram": current_histogram,
                },
            )

        return None

    def _generate_bollinger_signal(
        self, bb_data: Dict[str, Any], current_price: float
    ) -> Optional[TASignal]:
        """Generate Bollinger Bands signal"""

        upper_band = bb_data["current_upper"]
        lower_band = bb_data["current_lower"]
        price_position = bb_data["price_position"]

        if current_price >= upper_band:
            strength = min(1.0, price_position)
            return TASignal(
                indicator="bollinger",
                signal_type="sell",
                strength=strength,
                value=current_price,
                message=f"Price above upper Bollinger Band ({current_price:.2f} >= {upper_band:.2f})",
                metadata={
                    "price": current_price,
                    "upper_band": upper_band,
                    "price_position": price_position,
                },
            )
        elif current_price <= lower_band:
            strength = min(1.0, 1 - price_position)
            return TASignal(
                indicator="bollinger",
                signal_type="buy",
                strength=strength,
                value=current_price,
                message=f"Price below lower Bollinger Band ({current_price:.2f} <= {lower_band:.2f})",
                metadata={
                    "price": current_price,
                    "lower_band": lower_band,
                    "price_position": price_position,
                },
            )

        return None

    def _generate_sma_signal(
        self, sma_data: Dict[str, Any], current_price: float
    ) -> Optional[TASignal]:
        """Generate SMA-based signal"""

        # Use multiple SMAs for trend confirmation
        sma_values = []
        for key, data in sma_data.items():
            if isinstance(data, dict) and "current" in data:
                sma_values.append((data["period"], data["current"]))

        if len(sma_values) >= 2:
            # Sort by period
            sma_values.sort(key=lambda x: x[0])

            # Check if price is above/below multiple SMAs
            above_count = sum(1 for _, sma_val in sma_values if current_price > sma_val)
            below_count = len(sma_values) - above_count

            if above_count == len(sma_values):
                strength = 0.6  # Moderate strength for SMA signals
                return TASignal(
                    indicator="sma",
                    signal_type="buy",
                    strength=strength,
                    value=current_price,
                    message=f"Price above all SMAs - uptrend confirmed",
                    metadata={"sma_values": dict(sma_values), "above_count": above_count},
                )
            elif below_count == len(sma_values):
                strength = 0.6
                return TASignal(
                    indicator="sma",
                    signal_type="sell",
                    strength=strength,
                    value=current_price,
                    message=f"Price below all SMAs - downtrend confirmed",
                    metadata={"sma_values": dict(sma_values), "below_count": below_count},
                )

        return None

    def _determine_overall_signal(self, signals: List[TASignal]) -> Tuple[str, float]:
        """Determine overall signal from individual signals"""

        if not signals:
            return "neutral", 0.0

        # Weighted signal calculation
        buy_weight = 0.0
        sell_weight = 0.0
        total_weight = 0.0

        for signal in signals:
            weight = signal.strength
            total_weight += weight

            if signal.signal_type == "buy":
                buy_weight += weight
            elif signal.signal_type == "sell":
                sell_weight += weight

        if total_weight == 0:
            return "neutral", 0.0

        # Calculate confidence
        buy_ratio = buy_weight / total_weight
        sell_ratio = sell_weight / total_weight

        # Determine overall signal
        if buy_ratio > sell_ratio and buy_ratio > 0.6:
            return "buy", buy_ratio
        elif sell_ratio > buy_ratio and sell_ratio > 0.6:
            return "sell", sell_ratio
        else:
            return "neutral", max(buy_ratio, sell_ratio)

    def _cleanup_history(self):
        """Clean up analysis history"""

        max_history = 50
        if len(self.analysis_history) > max_history:
            self.analysis_history = self.analysis_history[-max_history:]

    def _deep_merge_dict(self, base: Dict, update: Dict) -> Dict:
        """Deep merge two dictionaries"""

        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(base[key], value)
            else:
                base[key] = value

        return base

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get TA agent summary"""

        return {
            "agent_status": "active",
            "talib_available": TALIB_AVAILABLE,
            "last_analysis": self.last_analysis.timestamp.isoformat()
            if self.last_analysis
            else None,
            "total_analyses": self.analysis_count,
            "average_analysis_time": self.total_analysis_time / max(1, self.analysis_count),
            "indicators_enabled": {
                "sma": self.config["indicators"]["sma"]["enabled"],
                "rsi": self.config["indicators"]["rsi"]["enabled"],
                "macd": self.config["indicators"]["macd"]["enabled"],
                "bollinger": self.config["indicators"]["bollinger"]["enabled"],
            },
        }


# Utility functions


async def quick_ta_analysis(symbol: str, price_data: pd.DataFrame) -> Dict[str, Any]:
    """Perform quick TA analysis"""

    agent = TechnicalAnalysisAgent()
    result = await agent.analyze_symbol(symbol, price_data)

    return {
        "symbol": result.symbol,
        "overall_signal": result.overall_signal,
        "confidence": result.confidence,
        "signal_count": len(result.signals),
        "timestamp": result.timestamp.isoformat(),
    }


if __name__ == "__main__":
    # Test TA agent

    async def test_ta_agent():
        print("Testing Technical Analysis Agent")

        # Create sample data
        dates = pd.date_range("2024-01-01", periods=100, freq="H")
        np.random.seed(42)

        # Generate realistic price data
        base_price = 100.0
        returns = np.random.normal(0, 1)
        prices = [base_price]

        for i in range(1, 100):
            price = prices[-1] * (1 + returns[i])
            prices.append(price)

        # Create OHLCV data
        data = pd.DataFrame(
            {
                "open": prices,
                "high": [p * (1 + abs(np.random.normal(0, 1))) for p in prices],
                "low": [p * (1 - abs(np.random.normal(0, 1))) for p in prices],
                "close": prices,
                "volume": np.random.normal(0, 1),
            },
            index=dates,
        )

        # Test TA analysis
        agent = TechnicalAnalysisAgent()
        result = await agent.analyze_symbol("TEST/USD", data)

        print(f"\nTA Analysis Result:")
        print(f"Symbol: {result.symbol}")
        print(f"Overall Signal: {result.overall_signal}")
        print(f"Confidence: {result.confidence:.1%}")
        print(f"Indicators: {list(result.indicators.keys())}")
        print(f"Signals: {len(result.signals)}")

        for signal in result.signals:
            print(f"  {signal.indicator}: {signal.signal_type} (strength: {signal.strength:.1%})")

        print("\n✅ TA AGENT TEST COMPLETE")

    asyncio.run(test_ta_agent())
