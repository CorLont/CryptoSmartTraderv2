#!/usr/bin/env python3
"""
Temporal Integrity Validator
Lightweight validator for temporal integrity in the ML pipeline
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass

@dataclass
class TemporalValidationResult:
    """Simplified temporal validation result"""
    is_valid: bool
    violations: List[str]
    aligned_data: Optional[pd.DataFrame]
    recommendation: str

class TemporalIntegrityValidator:
    """Lightweight temporal integrity validator"""

    def __init__(self, timeframe: str = "1h"):
        self.timeframe = timeframe
        self.logger = logging.getLogger(__name__)

        # Timeframe intervals in seconds
        self.intervals = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600,
            "4h": 14400,
            "1d": 86400
        }

        self.interval_seconds = self.intervals.get(timeframe, 3600)

    def validate_agent_data(self, df: pd.DataFrame, agent_name: str = "agent") -> TemporalValidationResult:
        """Validate temporal integrity of agent data"""

        violations = []

        # Find timestamp column
        timestamp_col = self._find_timestamp_column(df)
        if timestamp_col is None:
            return TemporalValidationResult(
                is_valid=False,
                violations=[f"No timestamp column found in {agent_name}"],
                aligned_data=None,
                recommendation="Add timestamp column with UTC datetime values"
            )

        timestamps = df[timestamp_col]

        # Check 1: UTC compliance
        timezone_violations = self._check_timezone_compliance(timestamps, agent_name)
        violations.extend(timezone_violations)

        # Check 2: Chronological order
        if not timestamps.is_monotonic_increasing:
            violations.append(f"{agent_name}: Timestamps not in chronological order")

        # Check 3: Future data leakage
        current_time = datetime.utcnow().replace(tzinfo=timezone.utc)
        future_count = 0
        for ts in timestamps:
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts > current_time:
                    future_count += 1

        if future_count > 0:
            violations.append(f"{agent_name}: {future_count} timestamps in the future")

        # Check 4: Duplicate timestamps
        duplicates = timestamps.duplicated().sum()
        if duplicates > 0:
            violations.append(f"{agent_name}: {duplicates} duplicate timestamps")

        # Check 5: Regular intervals
        if len(timestamps) > 1:
            intervals = timestamps.diff().dt.total_seconds().dropna()
            irregular_count = sum(1 for interval in intervals if abs(interval - self.interval_seconds) > 60)
            if irregular_count > len(intervals) * 0.1:  # More than 10% irregular
                violations.append(f"{agent_name}: {irregular_count} irregular intervals")

        # Attempt to align timestamps to candle boundaries
        aligned_data = self._align_to_candles(df, timestamp_col)

        is_valid = len(violations) == 0
        recommendation = "Safe for ML operations" if is_valid else "Fix temporal violations before ML use"

        return TemporalValidationResult(
            is_valid=is_valid,
            violations=violations,
            aligned_data=aligned_data,
            recommendation=recommendation
        )

    def validate_multi_agent(self, agent_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Validate temporal integrity across multiple agents"""

        all_violations = []
        agent_results = {}
        aligned_agents = {}

        # Validate each agent individually
        for agent_name, df in agent_data.items():
            result = self.validate_agent_data(df, agent_name)
            agent_results[agent_name] = result

            if result.violations:
                all_violations.extend(result.violations)

            if result.aligned_data is not None:
                aligned_agents[agent_name] = result.aligned_data

        # Check cross-agent timestamp alignment
        if len(aligned_agents) > 1:
            timestamp_sets = {}
            for agent_name, df in aligned_agents.items():
                timestamp_col = self._find_timestamp_column(df)
                if timestamp_col:
                    timestamp_sets[agent_name] = set(df[timestamp_col])

            if len(timestamp_sets) > 1:
                # Compare timestamp sets
                reference_agent = list(timestamp_sets.keys())[0]
                reference_timestamps = timestamp_sets[reference_agent]

                for agent_name, timestamps in timestamp_sets.items():
                    if agent_name != reference_agent:
                        if timestamps != reference_timestamps:
                            missing = len(reference_timestamps - timestamps)
                            extra = len(timestamps - reference_timestamps)
                            all_violations.append(
                                f"Cross-agent mismatch: {agent_name} vs {reference_agent} "
                                f"({missing} missing, {extra} extra)"
                            )

        overall_valid = len(all_violations) == 0

        return {
            'overall_valid': overall_valid,
            'violations': all_violations,
            'agent_results': agent_results,
            'aligned_agents': aligned_agents,
            'recommendation': 'Safe for ML operations' if overall_valid else 'Fix violations before proceeding'
        }

    def _find_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find timestamp column in DataFrame"""

        candidates = ['timestamp', 'time', 'datetime', 'date']

        for col in candidates:
            if col in df.columns:
                return col

        # Check for datetime-like columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col

        return None

    def _check_timezone_compliance(self, timestamps: pd.Series, agent_name: str) -> List[str]:
        """Check timezone compliance"""

        violations = []

        for i, ts in enumerate(timestamps.head(5)):  # Check first 5
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    violations.append(f"{agent_name}: Timezone-naive timestamp at index {i}")
                elif ts.tzinfo != timezone.utc:
                    violations.append(f"{agent_name}: Non-UTC timezone at index {i}")

        return violations

    def _align_to_candles(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Align timestamps to candle boundaries"""

        aligned_df = df.copy()

        try:
            for idx, row in df.iterrows():
                original_ts = row[timestamp_col]

                if isinstance(original_ts, datetime):
                    # Ensure UTC
                    if original_ts.tzinfo is None:
                        original_ts = original_ts.replace(tzinfo=timezone.utc)
                    elif original_ts.tzinfo != timezone.utc:
                        original_ts = original_ts.astimezone(timezone.utc)

                    # Align to candle boundary (floor to nearest interval)
                    epoch = datetime(2000, 1, 1, tzinfo=timezone.utc)
                    seconds_since_epoch = (original_ts - epoch).total_seconds()
                    aligned_seconds = (seconds_since_epoch // self.interval_seconds) * self.interval_seconds
                    aligned_ts = epoch + timedelta(seconds=aligned_seconds)

                    aligned_df.at[idx, timestamp_col] = aligned_ts

        except Exception as e:
            self.logger.warning(f"Alignment failed: {e}")
            return df

        return aligned_df

def validate_temporal_integrity(
    agent_data: Dict[str, pd.DataFrame],
    timeframe: str = "1h"
) -> Dict[str, Any]:
    """High-level function to validate temporal integrity"""

    validator = TemporalIntegrityValidator(timeframe)
    return validator.validate_multi_agent(agent_data)

def align_agent_timestamps(
    df: pd.DataFrame,
    timeframe: str = "1h"
) -> pd.DataFrame:
    """Align agent timestamps to candle boundaries"""

    validator = TemporalIntegrityValidator(timeframe)
    result = validator.validate_agent_data(df)

    return result.aligned_data if result.aligned_data is not None else df
