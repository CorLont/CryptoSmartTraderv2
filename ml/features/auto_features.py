#!/usr/bin/env python3
"""
Auto Features Module
Advanced feature engineering with temporal safety
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging


class AutoFeatures:
    """Advanced automatic feature engineering"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.feature_names = []

    def create_technical_features(self, df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
        """Create technical analysis features"""

        result = df.copy()

        # Price-based features
        result["returns_1h"] = result[price_col].pct_change(1)
        result["returns_24h"] = result[price_col].pct_change(24)

        # Moving averages
        result["sma_12"] = result[price_col].rolling(12, min_periods=12).mean()
        result["sma_24"] = result[price_col].rolling(24, min_periods=24).mean()
        result["ema_12"] = result[price_col].ewm(span=12, min_periods=12).mean()

        # Volatility features
        result["volatility_12h"] = result["returns_1h"].rolling(12, min_periods=12).std()
        result["volatility_24h"] = result["returns_1h"].rolling(24, min_periods=24).std()

        # RSI
        result["rsi_14"] = self._calculate_rsi(result[price_col], 14)

        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(result[price_col], 20)
        result["bb_position"] = (result[price_col] - bb_lower) / (bb_upper - bb_lower)

        return result

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI with temporal safety"""

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> tuple:
        """Calculate Bollinger Bands"""

        sma = prices.rolling(period, min_periods=period).mean()
        std = prices.rolling(period, min_periods=period).std()

        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)

        return upper_band, lower_band
