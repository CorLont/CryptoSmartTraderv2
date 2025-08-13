#!/usr/bin/env python3
"""
Timestamp Validator - UTC normalization and temporal integrity
"""

import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, List, Union


def normalize_timestamp(
    timestamp: Union[str, datetime, pd.Timestamp], target_timezone: str = "UTC"
) -> pd.Timestamp:
    """Normalize timestamp to UTC"""

    if pd.isna(timestamp):
        return pd.NaT

    # Convert to pandas timestamp
    if isinstance(timestamp, str):
        ts = pd.to_datetime(timestamp)
    elif isinstance(timestamp, datetime):
        ts = pd.Timestamp(timestamp)
    else:
        ts = pd.Timestamp(timestamp)

    # Convert to UTC if not already
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    return ts


def validate_timestamp_sequence(timestamps: pd.Series) -> Dict[str, Any]:
    """Validate timestamp sequence for temporal integrity"""

    if timestamps.empty:
        return {"valid": False, "issues": ["Empty timestamp series"]}

    issues = []

    # Check for missing timestamps
    missing_count = timestamps.isna().sum()
    if missing_count > 0:
        issues.append(f"{missing_count} missing timestamps")

    # Check for duplicates
    duplicate_count = timestamps.duplicated().sum()
    if duplicate_count > 0:
        issues.append(f"{duplicate_count} duplicate timestamps")

    # Check for chronological order
    sorted_timestamps = timestamps.dropna().sort_values()
    if not timestamps.dropna().equals(sorted_timestamps):
        issues.append("Timestamps not in chronological order")

    # Check for reasonable time gaps
    if len(timestamps.dropna()) > 1:
        time_diffs = timestamps.dropna().diff().dropna()
        if time_diffs.max() > pd.Timedelta(days=7):
            issues.append("Large time gaps detected (>7 days)")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "total_timestamps": len(timestamps),
        "missing_timestamps": missing_count,
        "duplicate_timestamps": duplicate_count,
    }
