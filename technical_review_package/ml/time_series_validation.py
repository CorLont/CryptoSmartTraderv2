#!/usr/bin/env python3
"""
Time Series Splits - Proper temporal validation
"""

from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np
from typing import Tuple, List


def create_time_series_splits(
    df: pd.DataFrame, n_splits: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create proper time series splits"""

    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must have 'timestamp' column")

    # Sort by timestamp
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(df_sorted))

    return splits


def validate_target_scaling(targets: pd.Series, max_reasonable_return: float = 2.0) -> bool:
    """Validate target scaling is reasonable"""

    abs_target_99 = targets.abs().quantile(0.99)

    if abs_target_99 > max_reasonable_return:
        raise ValueError(
            f"Target scaling issue: 99th percentile {abs_target_99:.3f} > {max_reasonable_return}"
        )

    return True


def create_returns_target(prices: pd.Series, horizon_hours: int) -> pd.Series:
    """Create properly scaled returns target"""

    future_prices = prices.shift(-horizon_hours)
    returns = (future_prices / prices) - 1.0

    # Validate scaling
    validate_target_scaling(returns.dropna())

    return returns
