#!/usr/bin/env python3
"""
Temporal Feature Engineering
Safe temporal feature creation with leak prevention
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TemporalFeatureConfig:
    """Configuration for temporal feature engineering"""
    lookback_periods: List[int] = None  # [1, 3, 7, 14, 30] periods
    forward_looking: bool = False  # NEVER allow forward-looking features
    lag_features: bool = True
    rolling_features: bool = True
    diff_features: bool = True
    technical_indicators: bool = True
    time_features: bool = True

    def __post_init__(self):
        if self.lookback_periods is None:
            self.lookback_periods = [1, 3, 7, 14, 30]

        # CRITICAL: Never allow forward-looking features
        if self.forward_looking:
            raise ValueError("Forward-looking features are prohibited to prevent look-ahead bias")

class TemporalFeatureEngineer:
    """Safe temporal feature engineering with leak prevention"""

    def __init__(self, config: TemporalFeatureConfig = None):
        self.config = config or TemporalFeatureConfig()
        self.logger = logging.getLogger(__name__)

        # Feature validation settings
        self.validation_settings = {
            'max_future_correlation': 0.05,  # Max correlation with future values
            'min_temporal_consistency': 0.95,  # Min consistency across time
            'require_lag_validation': True,  # Validate all lag features
        }

    def create_temporal_features(
        self,
        df: pd.DataFrame,
        value_cols: List[str],
        timestamp_col: str = 'timestamp',
        validate_leakage: bool = True
    ) -> pd.DataFrame:
        """Create temporal features with comprehensive leak prevention"""

        # Validate input data
        self._validate_input_data(df, value_cols, timestamp_col)

        # Sort by timestamp to ensure chronological order
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)

        # Create feature DataFrame
        feature_df = df_sorted.copy()

        # Add time-based features
        if self.config.time_features:
            feature_df = self._add_time_features(feature_df, timestamp_col)

        # Add lag features for each value column
        if self.config.lag_features:
            for col in value_cols:
                feature_df = self._add_lag_features(feature_df, col)

        # Add rolling window features
        if self.config.rolling_features:
            for col in value_cols:
                feature_df = self._add_rolling_features(feature_df, col)

        # Add difference features
        if self.config.diff_features:
            for col in value_cols:
                feature_df = self._add_diff_features(feature_df, col)

        # Add technical indicators
        if self.config.technical_indicators:
            for col in value_cols:
                feature_df = self._add_technical_indicators(feature_df, col)

        # Validate features for temporal leakage
        if validate_leakage:
            validated_df = self._validate_temporal_features(feature_df, value_cols, timestamp_col)
            return validated_df

        return feature_df

    def _validate_input_data(
        self,
        df: pd.DataFrame,
        value_cols: List[str],
        timestamp_col: str
    ):
        """Validate input data for feature engineering"""

        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found")

        for col in value_cols:
            if col not in df.columns:
                raise ValueError(f"Value column '{col}' not found")

        # Check for chronological ordering
        timestamps = df[timestamp_col]
        if not timestamps.is_monotonic_increasing:
            self.logger.warning("Data not in chronological order - will be sorted")

        # Check for sufficient data
        min_required = max(self.config.lookback_periods) + 10
        if len(df) < min_required:
            raise ValueError(f"Insufficient data: {len(df)} rows < {min_required} required")

    def _add_time_features(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Add time-based features (hour, day, month, etc.)"""

        timestamps = pd.to_datetime(df[timestamp_col])

        # Basic time features
        df[f'{timestamp_col}_hour'] = timestamps.dt.hour
        df[f'{timestamp_col}_day'] = timestamps.dt.day
        df[f'{timestamp_col}_day_of_week'] = timestamps.dt.dayofweek
        df[f'{timestamp_col}_month'] = timestamps.dt.month
        df[f'{timestamp_col}_quarter'] = timestamps.dt.quarter
        df[f'{timestamp_col}_year'] = timestamps.dt.year

        # Cyclical features (better for ML)
        df[f'{timestamp_col}_hour_sin'] = np.sin(2 * np.pi * timestamps.dt.hour / 24)
        df[f'{timestamp_col}_hour_cos'] = np.cos(2 * np.pi * timestamps.dt.hour / 24)
        df[f'{timestamp_col}_day_sin'] = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7)
        df[f'{timestamp_col}_day_cos'] = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7)
        df[f'{timestamp_col}_month_sin'] = np.sin(2 * np.pi * timestamps.dt.month / 12)
        df[f'{timestamp_col}_month_cos'] = np.cos(2 * np.pi * timestamps.dt.month / 12)

        # Market session features (for trading)
        hour = timestamps.dt.hour
        df[f'{timestamp_col}_is_market_hours'] = ((hour >= 9) & (hour <= 16)).astype(int)
        df[f'{timestamp_col}_is_overnight'] = ((hour >= 17) | (hour <= 8)).astype(int)
        df[f'{timestamp_col}_is_weekend'] = (timestamps.dt.dayofweek >= 5).astype(int)

        return df

    def _add_lag_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Add lag features (historical values)"""

        for lag in self.config.lookback_periods:
            # Simple lag
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

            # Lag change
            df[f'{col}_lag_{lag}_change'] = df[col] - df[col].shift(lag)

            # Lag percentage change
            df[f'{col}_lag_{lag}_pct_change'] = (df[col] - df[col].shift(lag)) / (df[col].shift(lag) + 1e-8)

        return df

    def _add_rolling_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Add rolling window features"""

        for window in self.config.lookback_periods:
            # Basic rolling statistics
            df[f'{col}_rolling_{window}_mean'] = df[col].rolling(window=window, min_periods=1).mean()
            df[f'{col}_rolling_{window}_std'] = df[col].rolling(window=window, min_periods=1).std()
            df[f'{col}_rolling_{window}_min'] = df[col].rolling(window=window, min_periods=1).min()
            df[f'{col}_rolling_{window}_max'] = df[col].rolling(window=window, min_periods=1).max()

            # Percentile features
            df[f'{col}_rolling_{window}_q25'] = df[col].rolling(window=window, min_periods=1).quantile(0.25)
            df[f'{col}_rolling_{window}_q75'] = df[col].rolling(window=window, min_periods=1).quantile(0.75)

            # Position relative to rolling window
            rolling_mean = df[f'{col}_rolling_{window}_mean']
            rolling_std = df[f'{col}_rolling_{window}_std']

            df[f'{col}_rolling_{window}_zscore'] = (df[col] - rolling_mean) / (rolling_std + 1e-8)
            df[f'{col}_rolling_{window}_relative'] = df[col] / (rolling_mean + 1e-8)

            # Rolling trend
            df[f'{col}_rolling_{window}_trend'] = df[col] / df[f'{col}_rolling_{window}_mean'].shift(1)

        return df

    def _add_diff_features(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Add difference and change features"""

        # First difference
        df[f'{col}_diff_1'] = df[col].diff(1)

        # Multiple period differences
        for period in [3, 7, 14]:
            if period <= len(df):
                df[f'{col}_diff_{period}'] = df[col].diff(period)
                df[f'{col}_pct_change_{period}'] = df[col].pct_change(period)

        # Acceleration (second difference)
        df[f'{col}_acceleration'] = df[f'{col}_diff_1'].diff(1)

        # Cumulative changes
        df[f'{col}_cumulative_change'] = (df[col] / df[col].iloc[0] - 1) if len(df) > 0 else 0

        return df

    def _add_technical_indicators(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Add technical analysis indicators"""

        # RSI (Relative Strength Index)
        for period in [14, 21]:
            rsi = self._calculate_rsi(df[col], period)
            df[f'{col}_rsi_{period}'] = rsi

        # MACD (Moving Average Convergence Divergence)
        macd, macd_signal, macd_histogram = self._calculate_macd(df[col])
        df[f'{col}_macd'] = macd
        df[f'{col}_macd_signal'] = macd_signal
        df[f'{col}_macd_histogram'] = macd_histogram

        # Bollinger Bands
        for period in [20, 50]:
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df[col], period)
            df[f'{col}_bb_{period}_upper'] = bb_upper
            df[f'{col}_bb_{period}_middle'] = bb_middle
            df[f'{col}_bb_{period}_lower'] = bb_lower
            df[f'{col}_bb_{period}_position'] = (df[col] - bb_lower) / (bb_upper - bb_lower + 1e-8)

        # Support and Resistance levels
        df[f'{col}_support'] = df[col].rolling(window=20, min_periods=1).min()
        df[f'{col}_resistance'] = df[col].rolling(window=20, min_periods=1).max()
        df[f'{col}_support_distance'] = (df[col] - df[f'{col}_support']) / (df[col] + 1e-8)
        df[f'{col}_resistance_distance'] = (df[f'{col}_resistance'] - df[col]) / (df[col] + 1e-8)

        return df

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""

        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()

        rs = gain / (loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_macd(
        self,
        series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD indicator"""

        ema_fast = series.ewm(span=fast_period, min_periods=1).mean()
        ema_slow = series.ewm(span=slow_period, min_periods=1).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, min_periods=1).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _calculate_bollinger_bands(
        self,
        series: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""

        middle = series.rolling(window=period, min_periods=1).mean()
        std = series.rolling(window=period, min_periods=1).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    def _validate_temporal_features(
        self,
        df: pd.DataFrame,
        value_cols: List[str],
        timestamp_col: str
    ) -> pd.DataFrame:
        """Validate features for temporal leakage"""

        # Find all created feature columns
        original_cols = set([timestamp_col] + value_cols)
        feature_cols = [col for col in df.columns if col not in original_cols]

        validated_df = df.copy()
        removed_features = []

        for feature_col in feature_cols:
            # Check for forward correlation (look-ahead bias)
            if self._check_forward_correlation(df, feature_col, value_cols):
                self.logger.warning(f"Removing feature {feature_col}: Forward correlation detected")
                validated_df = validated_df.drop(columns=[feature_col])
                removed_features.append(feature_col)
                continue

            # Check for temporal consistency
            if not self._check_temporal_consistency(df, feature_col):
                self.logger.warning(f"Removing feature {feature_col}: Temporal inconsistency detected")
                validated_df = validated_df.drop(columns=[feature_col])
                removed_features.append(feature_col)
                continue

        if removed_features:
            self.logger.info(f"Removed {len(removed_features)} features due to temporal violations")

        return validated_df

    def _check_forward_correlation(
        self,
        df: pd.DataFrame,
        feature_col: str,
        value_cols: List[str]
    ) -> bool:
        """Check if feature has suspicious correlation with future values"""

        if feature_col not in df.columns:
            return False

        feature_values = df[feature_col].dropna()

        if len(feature_values) < 10:
            return False  # Not enough data to check

        # Check correlation with future values of target columns
        for value_col in value_cols:
            for shift in [1, 3, 7]:  # Check 1, 3, 7 periods ahead
                if shift >= len(df):
                    continue

                future_values = df[value_col].shift(-shift).dropna()

                # Align the series
                min_len = min(len(feature_values), len(future_values))
                if min_len < 10:
                    continue

                aligned_feature = feature_values.iloc[:min_len]
                aligned_future = future_values.iloc[:min_len]

                # Calculate correlation
                try:
                    correlation = aligned_feature.corr(aligned_future)
                    if abs(correlation) > self.validation_settings['max_future_correlation']:
                        return True  # Suspicious forward correlation
                except:
                    continue

        return False

    def _check_temporal_consistency(self, df: pd.DataFrame, feature_col: str) -> bool:
        """Check if feature maintains temporal consistency"""

        if feature_col not in df.columns:
            return False

        feature_values = df[feature_col].dropna()

        if len(feature_values) < 20:
            return True  # Not enough data to check consistency

        # Check for sudden jumps or inconsistencies
        try:
            # Calculate rolling statistics
            rolling_mean = feature_values.rolling(window=10, min_periods=5).mean()
            rolling_std = feature_values.rolling(window=10, min_periods=5).std()

            # Check for outliers (values > 3 std deviations from rolling mean)
            outliers = np.abs(feature_values - rolling_mean) > (3 * rolling_std)
            outlier_ratio = outliers.sum() / len(feature_values)

            # If too many outliers, feature might be inconsistent
            if outlier_ratio > 0.1:  # More than 10% outliers
                return False

            # Check for sudden level shifts
            diff = feature_values.diff().abs()
            median_diff = diff.median()
            large_jumps = diff > (median_diff * 10)
            jump_ratio = large_jumps.sum() / len(feature_values)

            if jump_ratio > 0.05:  # More than 5% large jumps
                return False

        except:
            return True  # If calculation fails, assume consistent

        return True

def create_temporal_features(
    df: pd.DataFrame,
    value_cols: List[str],
    timestamp_col: str = 'timestamp',
    lookback_periods: List[int] = None,
    include_technical: bool = True
) -> pd.DataFrame:
    """High-level function to create temporal features"""

    config = TemporalFeatureConfig(
        lookback_periods=lookback_periods or [1, 3, 7, 14, 30],
        lag_features=True,
        rolling_features=True,
        diff_features=True,
        technical_indicators=include_technical,
        time_features=True
    )

    engineer = TemporalFeatureEngineer(config)
    return engineer.create_temporal_features(df, value_cols, timestamp_col)

def validate_feature_leakage(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    timestamp_col: str = 'timestamp'
) -> Dict[str, Any]:
    """Validate features for potential look-ahead bias"""

    violations = []
    suspicious_features = []

    for feature_col in feature_cols:
        if feature_col not in df.columns:
            continue

        # Check correlation with future target values
        for shift in [1, 3, 7]:
            if shift >= len(df):
                continue

            try:
                feature_values = df[feature_col].dropna()
                future_target = df[target_col].shift(-shift).dropna()

                min_len = min(len(feature_values), len(future_target))
                if min_len < 10:
                    continue

                correlation = feature_values.iloc[:min_len].corr(future_target.iloc[:min_len])

                if abs(correlation) > 0.1:  # Suspicious correlation
                    violations.append(f"{feature_col}: High correlation ({correlation:.3f}) with future target (+{shift})")
                    if feature_col not in suspicious_features:
                        suspicious_features.append(feature_col)

            except:
                continue

    return {
        'is_valid': len(violations) == 0,
        'violations': violations,
        'suspicious_features': suspicious_features,
        'recommendation': 'Features are safe' if len(violations) == 0 else 'Review suspicious features'
    }
