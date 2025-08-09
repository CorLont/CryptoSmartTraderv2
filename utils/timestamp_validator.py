#!/usr/bin/env python3
"""
Timestamp Validation Utility
Ensures proper timezone handling and candle alignment
"""

import pandas as pd
from datetime import datetime, timezone
from typing import Union, Dict, Any

def normalize_timestamp(ts: Union[str, pd.Timestamp, datetime], target_tz: str = 'UTC') -> pd.Timestamp:
    """Normalize timestamp to UTC with proper timezone handling"""
    
    if isinstance(ts, str):
        ts = pd.to_datetime(ts)
    
    if isinstance(ts, datetime):
        ts = pd.Timestamp(ts)
    
    # Add timezone if missing
    if ts.tz is None:
        ts = ts.tz_localize('UTC')
    
    # Convert to target timezone
    if target_tz != 'UTC':
        ts = ts.tz_convert(target_tz)
    
    return ts

def align_to_candle_boundary(ts: pd.Timestamp, freq: str = '1H') -> pd.Timestamp:
    """Align timestamp to candle boundary (e.g., hourly)"""
    return ts.floor(freq)

def validate_timestamp_sequence(df: pd.DataFrame, ts_col: str = 'ts') -> Dict[str, Any]:
    """Validate timestamp sequence in DataFrame"""
    
    issues = []
    
    if ts_col not in df.columns:
        return {"valid": False, "issues": [f"Timestamp column '{ts_col}' not found"]}
    
    ts_series = df[ts_col]
    
    # Check timezone
    if hasattr(ts_series.dtype, 'tz') and ts_series.dt.tz is None:
        issues.append("Missing timezone information")
    
    # Check sorting
    if not ts_series.is_monotonic_increasing:
        issues.append("Timestamps not in ascending order")
    
    # Check for duplicates
    duplicates = ts_series.duplicated().sum()
    if duplicates > 0:
        issues.append(f"{duplicates} duplicate timestamps")
    
    # Check alignment (hourly candles)
    if not ts_series.empty:
        misaligned = (ts_series != ts_series.dt.floor('1H')).sum()
        if misaligned > 0:
            issues.append(f"{misaligned} timestamps not aligned to hourly candles")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "total_timestamps": len(ts_series),
        "duplicates": duplicates
    }

def fix_dataframe_timestamps(df: pd.DataFrame, ts_col: str = 'ts') -> pd.DataFrame:
    """Fix common timestamp issues in DataFrame"""
    
    df_fixed = df.copy()
    
    if ts_col in df_fixed.columns:
        # Normalize timestamps
        df_fixed[ts_col] = df_fixed[ts_col].apply(normalize_timestamp)
        
        # Align to candle boundaries
        df_fixed[ts_col] = df_fixed[ts_col].apply(align_to_candle_boundary)
        
        # Remove duplicates (keep last)
        df_fixed = df_fixed.drop_duplicates(subset=[ts_col], keep='last')
        
        # Sort by timestamp
        df_fixed = df_fixed.sort_values(ts_col)
    
    return df_fixed
