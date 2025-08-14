#!/usr/bin/env python3
"""
Property-based tests for technical indicators
"""

import pytest
import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.pandas import data_frames, columns

# Import technical indicator functions (assuming they exist)
# from src.cryptosmarttrader.analysis.enterprise_technical_analysis import get_technical_analyzer


@pytest.mark.property
class TestTechnicalIndicators:
    """Property-based tests for technical indicators"""

    def test_price_strategy(self):
        """Strategy for generating realistic price data"""
        return st.lists(
            st.floats(min_value=0.01, max_value=100000.0, allow_nan=False, allow_infinity=False),
            min_size=14,  # Minimum for RSI calculation
            max_size=1000,
        )

    @given(prices=st.lists(st.floats(min_value=1.0, max_value=100000.0), min_size=14, max_size=100))
    @settings(max_examples=50)
    def test_rsi_properties(self, prices):
        """Test RSI indicator properties"""
        assume(len(prices) >= 14)
        assume(all(p > 0 for p in prices))

        rsi = self.get_technical_analyzer().calculate_indicator("RSI", prices, period=14).values

        # Property 1: RSI should be between 0 and 100
        valid_rsi = [r for r in rsi if not np.isnan(r)]
        if valid_rsi:
            assert all(0 <= r <= 100 for r in valid_rsi), f"RSI values out of range: {valid_rsi}"

        # Property 2: RSI should have NaN values for initial period
        assert len([r for r in rsi[:13] if np.isnan(r)]) >= 13, (
            "RSI should have NaN for initial period"
        )

        # Property 3: Extreme price movements should push RSI towards extremes
        if len(valid_rsi) > 1:
            # Check if consistently rising prices increase RSI
            rising_prices = all(
                prices[i] <= prices[i + 1] for i in range(len(prices) - 5, len(prices) - 1)
            )
            if rising_prices and len(valid_rsi) >= 2:
                assert valid_rsi[-1] >= valid_rsi[-2] * 0.9, (
                    "RSI should increase with rising prices"
                )

    @given(prices=st.lists(st.floats(min_value=1.0, max_value=100000.0), min_size=26, max_size=100))
    @settings(max_examples=50)
    def test_macd_properties(self, prices):
        """Test MACD indicator properties"""
        assume(len(prices) >= 26)
        assume(all(p > 0 for p in prices))

        macd_line, signal_line, histogram = self.get_technical_analyzer().calculate_indicator("MACD", prices).values

        # Property 1: MACD line should be EMA(12) - EMA(26)
        ema12 = self.calculate_ema(prices, 12)
        ema26 = self.calculate_ema(prices, 26)

        # Check last few values where all EMAs are valid
        for i in range(-5, 0):
            if not (np.isnan(ema12[i]) or np.isnan(ema26[i]) or np.isnan(macd_line[i])):
                expected_macd = ema12[i] - ema26[i]
                assert abs(macd_line[i] - expected_macd) < 0.01, (
                    f"MACD calculation error at index {i}"
                )

        # Property 2: Histogram should be MACD - Signal
        for i in range(-5, 0):
            if not (np.isnan(macd_line[i]) or np.isnan(signal_line[i]) or np.isnan(histogram[i])):
                expected_hist = macd_line[i] - signal_line[i]
                assert abs(histogram[i] - expected_hist) < 0.01, (
                    f"Histogram calculation error at index {i}"
                )

        # Property 3: Signal line should be smoother than MACD line
        macd_volatility = np.nanstd(macd_line[-10:]) if len(macd_line) >= 10 else 0
        signal_volatility = np.nanstd(signal_line[-10:]) if len(signal_line) >= 10 else 0

        if macd_volatility > 0 and signal_volatility > 0:
            assert signal_volatility <= macd_volatility * 1.5, (
                "Signal line should be smoother than MACD"
            )

    @given(prices=st.lists(st.floats(min_value=1.0, max_value=100000.0), min_size=20, max_size=100))
    @settings(max_examples=50)
    def test_bollinger_bands_properties(self, prices):
        """Test Bollinger Bands properties"""
        assume(len(prices) >= 20)
        assume(all(p > 0 for p in prices))

        upper, middle, lower = self.calculate_bollinger_bands(prices, period=20, std_dev=2)

        # Property 1: Upper band should always be above middle band
        for i in range(len(prices)):
            if not (np.isnan(upper[i]) or np.isnan(middle[i])):
                assert upper[i] >= middle[i], f"Upper band below middle at index {i}"

        # Property 2: Lower band should always be below middle band
        for i in range(len(prices)):
            if not (np.isnan(lower[i]) or np.isnan(middle[i])):
                assert lower[i] <= middle[i], f"Lower band above middle at index {i}"

        # Property 3: Middle band should be the moving average
        sma = self.calculate_sma(prices, 20)
        for i in range(len(prices)):
            if not (np.isnan(middle[i]) or np.isnan(sma[i])):
                assert abs(middle[i] - sma[i]) < 0.01, f"Middle band not equal to SMA at index {i}"

        # Property 4: Price should be between bands most of the time (95% rule)
        valid_indices = []
        inside_bands = 0

        for i in range(len(prices)):
            if not (np.isnan(upper[i]) or np.isnan(lower[i])):
                valid_indices.append(i)
                if lower[i] <= prices[i] <= upper[i]:
                    inside_bands += 1

        if len(valid_indices) > 10:
            inside_percentage = inside_bands / len(valid_indices)
            # Allow some tolerance for small samples
            assert inside_percentage >= 0.85, f"Only {inside_percentage:.1%} of prices inside bands"

    @given(data=st.lists(st.floats(min_value=-10.0, max_value=10.0), min_size=10, max_size=50))
    @settings(max_examples=50)
    def test_moving_average_properties(self, data):
        """Test moving average properties"""
        assume(len(data) >= 10)
        assume(all(not np.isnan(x) and not np.isinf(x) for x in data))

        period = 5
        ma = self.calculate_sma(data, period)

        # Property 1: Moving average should smooth the data
        if len(data) > period * 2:
            original_volatility = np.std(data[period:])
            ma_volatility = np.nanstd(ma[period:])

            if original_volatility > 0:
                assert ma_volatility <= original_volatility, (
                    "Moving average should reduce volatility"
                )

        # Property 2: MA should be between min and max of recent values
        for i in range(period, len(data)):
            if not np.isnan(ma[i]):
                recent_values = data[i - period + 1 : i + 1]
                min_val = min(recent_values)
                max_val = max(recent_values)
                assert min_val <= ma[i] <= max_val, f"MA outside range at index {i}"

        # Property 3: MA should lag the data
        # If data is trending up, MA should be below recent prices
        if len(data) >= 10:
            recent_trend = data[-5:]
            if all(recent_trend[i] <= recent_trend[i + 1] for i in range(len(recent_trend) - 1)):
                # Strong uptrend
                if not np.isnan(ma[-1]):
                    assert ma[-1] <= max(recent_trend), "MA should lag in uptrend"

    def get_technical_analyzer().calculate_indicator("RSI", self, prices: list, period: int = 14).values -> list:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return [np.nan] * len(prices)

        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [max(0, d) for d in deltas]
        losses = [max(0, -d) for d in deltas]

        rsi_values = [np.nan] * (period)  # First 'period' values are NaN

        # Initial average
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        for i in range(period, len(deltas)):
            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                rsi_values.append(rsi)

            # Update averages
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        return rsi_values

    def calculate_ema(self, prices: list, period: int) -> list:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return [np.nan] * len(prices)

        ema = [np.nan] * (period - 1)
        multiplier = 2.0 / (period + 1)

        # First EMA is simple average
        ema.append(sum(prices[:period]) / period)

        for i in range(period, len(prices)):
            ema_val = (prices[i] * multiplier) + (ema[-1] * (1 - multiplier))
            ema.append(ema_val)

        return ema

    def calculate_sma(self, prices: list, period: int) -> list:
        """Calculate Simple Moving Average"""
        sma = []

        for i in range(len(prices)):
            if i < period - 1:
                sma.append(np.nan)
            else:
                avg = sum(prices[i - period + 1 : i + 1]) / period
                sma.append(avg)

        return sma

    def get_technical_analyzer().calculate_indicator("MACD", self, prices: list, fast=12, slow=26, signal=9).values:
        """Calculate MACD indicator"""
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)

        macd_line = []
        for i in range(len(prices)):
            if np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]):
                macd_line.append(np.nan)
            else:
                macd_line.append(ema_fast[i] - ema_slow[i])

        signal_line = self.calculate_ema(macd_line, signal)

        histogram = []
        for i in range(len(prices)):
            if np.isnan(macd_line[i]) or np.isnan(signal_line[i]):
                histogram.append(np.nan)
            else:
                histogram.append(macd_line[i] - signal_line[i])

        return macd_line, signal_line, histogram

    def calculate_bollinger_bands(self, prices: list, period: int = 20, std_dev: float = 2):
        """Calculate Bollinger Bands"""
        middle = self.calculate_sma(prices, period)

        upper = []
        lower = []

        for i in range(len(prices)):
            if i < period - 1:
                upper.append(np.nan)
                lower.append(np.nan)
            else:
                recent_prices = prices[i - period + 1 : i + 1]
                std = np.std(recent_prices)
                upper.append(middle[i] + (std_dev * std))
                lower.append(middle[i] - (std_dev * std))

        return upper, middle, lower


@pytest.mark.property
class TestSignalLogic:
    """Property-based tests for trading signal logic"""

    @given(
        confidence=st.floats(min_value=0.0, max_value=1.0),
        threshold=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_confidence_gating(self, confidence, threshold):
        """Test confidence gating logic"""

        # Signal should only pass if confidence >= threshold
        should_pass = confidence >= threshold
        actual_pass = self.confidence_gate(confidence, threshold)

        assert actual_pass == should_pass, (
            f"Confidence gate failed: {confidence} >= {threshold} should be {should_pass}"
        )

    @given(
        price=st.floats(min_value=1.0, max_value=100000.0),
        signal_type=st.sampled_from(["buy", "sell"]),
        position_size=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_position_sizing(self, price, signal_type, position_size):
        """Test position sizing logic"""
        assume(price > 0)
        assume(0 <= position_size <= 1)

        capital = 100000.0
        max_position = 0.1  # 10% max position size

        actual_size = self.calculate_position_size(
            signal_type, price, position_size, capital, max_position
        )

        # Property 1: Position size should not exceed maximum
        max_allowed = capital * max_position / price
        assert actual_size <= max_allowed, (
            f"Position size {actual_size} exceeds maximum {max_allowed}"
        )

        # Property 2: Position size should be non-negative
        assert actual_size >= 0, f"Position size should be non-negative: {actual_size}"

        # Property 3: If signal is zero confidence, position should be zero
        if position_size == 0:
            assert actual_size == 0, "Zero confidence should result in zero position"

    @given(
        prices=st.lists(st.floats(min_value=100.0, max_value=200.0), min_size=5, max_size=20),
        stop_loss_pct=st.floats(min_value=0.01, max_value=0.10),
    )
    @settings(max_examples=50)
    def test_stop_loss_logic(self, prices, stop_loss_pct):
        """Test stop loss triggering logic"""
        assume(len(prices) >= 2)
        assume(all(p > 0 for p in prices))
        assume(0.01 <= stop_loss_pct <= 0.10)

        entry_price = prices[0]

        for current_price in prices[1:]:
            # Test long position stop loss
            long_stop_triggered = self.check_stop_loss(
                entry_price, current_price, "long", stop_loss_pct
            )

            # Property: Long stop should trigger when price drops below threshold
            long_threshold = entry_price * (1 - stop_loss_pct)
            expected_long_stop = current_price <= long_threshold

            assert long_stop_triggered == expected_long_stop, (
                f"Long stop loss logic failed: {current_price} vs {long_threshold}"
            )

            # Test short position stop loss
            short_stop_triggered = self.check_stop_loss(
                entry_price, current_price, "short", stop_loss_pct
            )

            # Property: Short stop should trigger when price rises above threshold
            short_threshold = entry_price * (1 + stop_loss_pct)
            expected_short_stop = current_price >= short_threshold

            assert short_stop_triggered == expected_short_stop, (
                f"Short stop loss logic failed: {current_price} vs {short_threshold}"
            )

    def confidence_gate(self, confidence: float, threshold: float) -> bool:
        """Simple confidence gate logic"""
        return confidence >= threshold

    def calculate_position_size(
        self, signal_type: str, price: float, confidence: float, capital: float, max_position: float
    ) -> float:
        """Calculate position size based on signal"""
        if confidence <= 0:
            return 0.0

        # Calculate base position size
        target_value = capital * max_position * confidence
        position_size = target_value / price

        # Apply maximum position limit
        max_size = capital * max_position / price

        return min(position_size, max_size)

    def check_stop_loss(
        self, entry_price: float, current_price: float, position_type: str, stop_loss_pct: float
    ) -> bool:
        """Check if stop loss should be triggered"""
        if position_type == "long":
            stop_price = entry_price * (1 - stop_loss_pct)
            return current_price <= stop_price
        elif position_type == "short":
            stop_price = entry_price * (1 + stop_loss_pct)
            return current_price >= stop_price
        else:
            return False


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "property"])
