#!/usr/bin/env python3
"""
Timestamp Synchronizer
Ensures all agents use exact UTC timeframes aligned on candle boundaries
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class TimeframeType(Enum):
    """Supported timeframe types"""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAY_1 = "1d"
    WEEK_1 = "1w"

@dataclass
class TimeAlignment:
    """Timestamp alignment result"""
    original_timestamp: datetime
    aligned_timestamp: datetime
    timeframe: TimeframeType
    candle_start: datetime
    candle_end: datetime
    alignment_offset_seconds: float
    is_perfectly_aligned: bool

@dataclass
class SynchronizationReport:
    """Agent synchronization report"""
    agent_name: str
    total_timestamps: int
    aligned_timestamps: int
    misaligned_timestamps: int
    max_misalignment_seconds: float
    timeframe_consistency: bool
    utc_compliance: bool
    candle_boundary_violations: List[datetime]
    synchronization_quality: float  # 0-1 score

class TimestampSynchronizer:
    """Synchronizes timestamps across all agents to exact UTC candle boundaries"""

    def __init__(self, primary_timeframe: TimeframeType = TimeframeType.HOUR_1):
        self.primary_timeframe = primary_timeframe
        self.logger = logging.getLogger(__name__)

        # Timeframe intervals in seconds
        self.timeframe_seconds = {
            TimeframeType.MINUTE_1: 60,
            TimeframeType.MINUTE_5: 300,
            TimeframeType.MINUTE_15: 900,
            TimeframeType.HOUR_1: 3600,
            TimeframeType.HOUR_4: 14400,
            TimeframeType.DAY_1: 86400,
            TimeframeType.WEEK_1: 604800
        }

        # UTC reference epoch (2000-01-01 00:00:00 UTC)
        self.utc_epoch = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    def align_timestamp_to_candle(
        self,
        timestamp: Union[datetime, str, int, float],
        timeframe: TimeframeType,
        alignment_mode: str = "floor"  # "floor", "ceil", "nearest"
    ) -> TimeAlignment:
        """Align timestamp to exact candle boundary"""

        # Convert to UTC datetime
        utc_timestamp = self._normalize_to_utc(timestamp)

        # Get timeframe interval
        interval_seconds = self.timeframe_seconds[timeframe]

        # Calculate seconds since epoch
        seconds_since_epoch = (utc_timestamp - self.utc_epoch).total_seconds()

        # Calculate candle alignment
        if alignment_mode == "floor":
            aligned_seconds = (seconds_since_epoch // interval_seconds) * interval_seconds
        elif alignment_mode == "ceil":
            aligned_seconds = ((seconds_since_epoch // interval_seconds) + 1) * interval_seconds
        elif alignment_mode == "nearest":
            remainder = seconds_since_epoch % interval_seconds
            if remainder < interval_seconds / 2:
                aligned_seconds = (seconds_since_epoch // interval_seconds) * interval_seconds
            else:
                aligned_seconds = ((seconds_since_epoch // interval_seconds) + 1) * interval_seconds
        else:
            raise ValueError(f"Invalid alignment mode: {alignment_mode}")

        # Calculate aligned timestamp
        aligned_timestamp = self.utc_epoch + timedelta(seconds=aligned_seconds)

        # Calculate candle boundaries
        candle_start = aligned_timestamp
        candle_end = candle_start + timedelta(seconds=interval_seconds)

        # Calculate alignment offset
        alignment_offset = (aligned_timestamp - utc_timestamp).total_seconds()
        is_perfectly_aligned = abs(alignment_offset) < 0.001  # Within 1ms

        return TimeAlignment(
            original_timestamp=utc_timestamp,
            aligned_timestamp=aligned_timestamp,
            timeframe=timeframe,
            candle_start=candle_start,
            candle_end=candle_end,
            alignment_offset_seconds=alignment_offset,
            is_perfectly_aligned=is_perfectly_aligned
        )

    def synchronize_agent_timestamps(
        self,
        agent_data: Dict[str, pd.DataFrame],
        timeframe: TimeframeType = None,
        strict_mode: bool = True
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, SynchronizationReport]]:
        """Synchronize timestamps across all agents"""

        if timeframe is None:
            timeframe = self.primary_timeframe

        synchronized_data = {}
        sync_reports = {}

        for agent_name, df in agent_data.items():
            sync_df, report = self._synchronize_single_agent(
                df, agent_name, timeframe, strict_mode
            )

            synchronized_data[agent_name] = sync_df
            sync_reports[agent_name] = report

        # Validate cross-agent alignment
        alignment_validation = self._validate_cross_agent_alignment(
            synchronized_data, timeframe
        )

        if strict_mode and not alignment_validation['all_aligned']:
            misaligned_agents = alignment_validation['misaligned_agents']
            raise ValueError(f"Cross-agent alignment failed: {misaligned_agents}")

        return synchronized_data, sync_reports

    def _synchronize_single_agent(
        self,
        df: pd.DataFrame,
        agent_name: str,
        timeframe: TimeframeType,
        strict_mode: bool
    ) -> Tuple[pd.DataFrame, SynchronizationReport]:
        """Synchronize timestamps for single agent"""

        if 'timestamp' not in df.columns:
            # Try common timestamp column names
            timestamp_cols = ['timestamp', 'time', 'datetime', 'date']
            timestamp_col = None

            for col in timestamp_cols:
                if col in df.columns:
                    timestamp_col = col
                    break

            if timestamp_col is None:
                raise ValueError(f"No timestamp column found in {agent_name} data")
        else:
            timestamp_col = 'timestamp'

        sync_df = df.copy()
        alignments = []
        candle_violations = []

        # Align each timestamp
        for idx, row in df.iterrows():
            original_ts = row[timestamp_col]

            try:
                alignment = self.align_timestamp_to_candle(original_ts, timeframe, "floor")
                alignments.append(alignment)

                # Update DataFrame with aligned timestamp
                sync_df.at[idx, timestamp_col] = alignment.aligned_timestamp

                # Check for violations
                if not alignment.is_perfectly_aligned:
                    if abs(alignment.alignment_offset_seconds) > 60:  # More than 1 minute off
                        candle_violations.append(alignment.original_timestamp)

            except Exception as e:
                self.logger.error(f"Failed to align timestamp {original_ts} in {agent_name}: {e}")
                if strict_mode:
                    raise
                # In non-strict mode, keep original timestamp
                alignments.append(None)

        # Calculate synchronization metrics
        valid_alignments = [a for a in alignments if a is not None]
        total_timestamps = len(df)
        aligned_timestamps = len(valid_alignments)
        misaligned_timestamps = total_timestamps - aligned_timestamps

        if valid_alignments:
            max_misalignment = max(abs(a.alignment_offset_seconds) for a in valid_alignments)
            perfect_alignments = sum(1 for a in valid_alignments if a.is_perfectly_aligned)
            sync_quality = perfect_alignments / len(valid_alignments)
        else:
            max_misalignment = float('inf')
            sync_quality = 0.0

        # Check UTC compliance
        utc_compliance = self._check_utc_compliance(sync_df[timestamp_col])

        # Check timeframe consistency
        timeframe_consistency = self._check_timeframe_consistency(
            sync_df[timestamp_col], timeframe
        )

        report = SynchronizationReport(
            agent_name=agent_name,
            total_timestamps=total_timestamps,
            aligned_timestamps=aligned_timestamps,
            misaligned_timestamps=misaligned_timestamps,
            max_misalignment_seconds=max_misalignment,
            timeframe_consistency=timeframe_consistency,
            utc_compliance=utc_compliance,
            candle_boundary_violations=candle_violations,
            synchronization_quality=sync_quality
        )

        return sync_df, report

    def _normalize_to_utc(self, timestamp: Union[datetime, str, int, float]) -> datetime:
        """Normalize various timestamp formats to UTC datetime"""

        if isinstance(timestamp, datetime):
            # If timezone-naive, assume UTC
            if timestamp.tzinfo is None:
                return timestamp.replace(tzinfo=timezone.utc)
            # Convert to UTC if timezone-aware
            return timestamp.astimezone(timezone.utc)

        elif isinstance(timestamp, str):
            # Parse string timestamp
            try:
                # Try ISO format first
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.astimezone(timezone.utc)
            except:
                # Try pandas parsing
                dt = pd.to_datetime(timestamp, utc=True)
                return dt.to_pydatetime()

        elif isinstance(timestamp, (int, float)):
            # Unix timestamp
            if timestamp > 1e10:  # Milliseconds
                timestamp = timestamp / 1000
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)

        else:
            raise ValueError(f"Unsupported timestamp type: {type(timestamp)}")

    def _check_utc_compliance(self, timestamps: pd.Series) -> bool:
        """Check if all timestamps are UTC compliant"""

        for ts in timestamps:
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    return False  # Timezone-naive timestamps not allowed
                if ts.tzinfo != timezone.utc:
                    return False  # Non-UTC timezones not allowed

        return True

    def _check_timeframe_consistency(
        self,
        timestamps: pd.Series,
        expected_timeframe: TimeframeType
    ) -> bool:
        """Check if timestamps are consistent with expected timeframe"""

        if len(timestamps) < 2:
            return True

        expected_interval = self.timeframe_seconds[expected_timeframe]

        # Check intervals between consecutive timestamps
        for i in range(1, len(timestamps)):
            prev_ts = timestamps.iloc[i-1]
            curr_ts = timestamps.iloc[i]

            if isinstance(prev_ts, datetime) and isinstance(curr_ts, datetime):
                interval = (curr_ts - prev_ts).total_seconds()

                # Allow some tolerance (±1 second)
                if abs(interval - expected_interval) > 1:
                    return False

        return True

    def _validate_cross_agent_alignment(
        self,
        agent_data: Dict[str, pd.DataFrame],
        timeframe: TimeframeType
    ) -> Dict[str, Any]:
        """Validate that all agents have aligned timestamps"""

        if len(agent_data) < 2:
            return {'all_aligned': True, 'misaligned_agents': []}

        # Get reference timestamps from first agent
        agent_names = list(agent_data.keys())
        reference_agent = agent_names[0]
        reference_df = agent_data[reference_agent]

        if 'timestamp' not in reference_df.columns:
            return {'all_aligned': False, 'error': 'No timestamp column in reference agent'}

        reference_timestamps = set(reference_df['timestamp'])
        misaligned_agents = []

        # Compare with other agents
        for agent_name in agent_names[1:]:
            agent_df = agent_data[agent_name]

            if 'timestamp' not in agent_df.columns:
                misaligned_agents.append(f"{agent_name}: No timestamp column")
                continue

            agent_timestamps = set(agent_df['timestamp'])

            # Check for timestamp mismatches
            if reference_timestamps != agent_timestamps:
                missing_in_agent = reference_timestamps - agent_timestamps
                extra_in_agent = agent_timestamps - reference_timestamps

                if missing_in_agent or extra_in_agent:
                    misaligned_agents.append(
                        f"{agent_name}: {len(missing_in_agent)} missing, {len(extra_in_agent)} extra"
                    )

        return {
            'all_aligned': len(misaligned_agents) == 0,
            'misaligned_agents': misaligned_agents,
            'reference_agent': reference_agent,
            'total_agents': len(agent_names)
        }

    def create_synchronized_candle_index(
        self,
        start_time: datetime,
        end_time: datetime,
        timeframe: TimeframeType
    ) -> pd.DatetimeIndex:
        """Create perfectly aligned candle index for given time range"""

        # Align start and end times to candle boundaries
        start_alignment = self.align_timestamp_to_candle(start_time, timeframe, "floor")
        end_alignment = self.align_timestamp_to_candle(end_time, timeframe, "ceil")

        # Create frequency string for pandas
        freq_map = {
            TimeframeType.MINUTE_1: "1T",
            TimeframeType.MINUTE_5: "5T",
            TimeframeType.MINUTE_15: "15T",
            TimeframeType.HOUR_1: "1H",
            TimeframeType.HOUR_4: "4H",
            TimeframeType.DAY_1: "1D",
            TimeframeType.WEEK_1: "1W"
        }

        freq = freq_map[timeframe]

        # Create aligned datetime index
        candle_index = pd.date_range(
            start=start_alignment.aligned_timestamp,
            end=end_alignment.aligned_timestamp,
            freq=freq,
            tz='UTC'
        )

        return candle_index

    def validate_temporal_integrity(
        self,
        agent_data: Dict[str, pd.DataFrame],
        timeframe: TimeframeType = None
    ) -> Dict[str, Any]:
        """Comprehensive temporal integrity validation"""

        if timeframe is None:
            timeframe = self.primary_timeframe

        validation_results = {
            'overall_status': 'unknown',
            'critical_violations': [],
            'warnings': [],
            'agent_reports': {},
            'cross_agent_alignment': {},
            'temporal_integrity_score': 0.0
        }

        try:
            # Synchronize and validate
            synchronized_data, sync_reports = self.synchronize_agent_timestamps(
                agent_data, timeframe, strict_mode=False
            )

            # Store individual agent reports
            validation_results['agent_reports'] = {
                name: {
                    'sync_quality': report.synchronization_quality,
                    'utc_compliance': report.utc_compliance,
                    'timeframe_consistency': report.timeframe_consistency,
                    'violations': len(report.candle_boundary_violations),
                    'max_misalignment_sec': report.max_misalignment_seconds
                }
                for name, report in sync_reports.items()
            }

            # Cross-agent validation
            alignment_validation = self._validate_cross_agent_alignment(
                synchronized_data, timeframe
            )
            validation_results['cross_agent_alignment'] = alignment_validation

            # Identify critical violations
            critical_violations = []
            warnings = []

            for agent_name, report in sync_reports.items():
                # Critical: More than 1 minute misalignment
                if report.max_misalignment_seconds > 60:
                    critical_violations.append(
                        f"{agent_name}: {report.max_misalignment_seconds:.1f}s max misalignment"
                    )

                # Critical: Not UTC compliant
                if not report.utc_compliance:
                    critical_violations.append(f"{agent_name}: Not UTC compliant")

                # Critical: Timeframe inconsistency
                if not report.timeframe_consistency:
                    critical_violations.append(f"{agent_name}: Inconsistent timeframe")

                # Warning: Low sync quality
                if report.synchronization_quality < 0.95:
                    warnings.append(
                        f"{agent_name}: {report.synchronization_quality:.1%} sync quality"
                    )

            # Cross-agent critical violations
            if not alignment_validation['all_aligned']:
                critical_violations.extend([
                    f"Cross-agent misalignment: {agent}"
                    for agent in alignment_validation['misaligned_agents']
                ])

            # Calculate overall temporal integrity score
            individual_scores = [
                report.synchronization_quality for report in sync_reports.values()
            ]
            cross_agent_score = 1.0 if alignment_validation['all_aligned'] else 0.0

            if individual_scores:
                temporal_integrity_score = (
                    np.mean(individual_scores) * 0.7 +  # Individual agent quality
                    cross_agent_score * 0.3             # Cross-agent alignment
                )
            else:
                temporal_integrity_score = 0.0

            # Overall status
            if critical_violations:
                overall_status = 'critical_violations'
            elif warnings:
                overall_status = 'warnings'
            else:
                overall_status = 'passed'

            validation_results.update({
                'overall_status': overall_status,
                'critical_violations': critical_violations,
                'warnings': warnings,
                'temporal_integrity_score': temporal_integrity_score
            })

        except Exception as e:
            validation_results.update({
                'overall_status': 'error',
                'critical_violations': [f"Validation error: {str(e)}"],
                'temporal_integrity_score': 0.0
            })

        return validation_results

def create_timestamp_synchronizer(timeframe: str = "1h") -> TimestampSynchronizer:
    """Create timestamp synchronizer with specified timeframe"""

    timeframe_map = {
        "1m": TimeframeType.MINUTE_1,
        "5m": TimeframeType.MINUTE_5,
        "15m": TimeframeType.MINUTE_15,
        "1h": TimeframeType.HOUR_1,
        "4h": TimeframeType.HOUR_4,
        "1d": TimeframeType.DAY_1,
        "1w": TimeframeType.WEEK_1
    }

    tf = timeframe_map.get(timeframe, TimeframeType.HOUR_1)
    return TimestampSynchronizer(primary_timeframe=tf)

def align_agents_to_candles(
    agent_data: Dict[str, pd.DataFrame],
    timeframe: str = "1h",
    strict_mode: bool = True
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Any]]:
    """High-level function to align all agents to candle boundaries"""

    synchronizer = create_timestamp_synchronizer(timeframe)

    # Get timeframe enum
    timeframe_map = {
        "1m": TimeframeType.MINUTE_1,
        "5m": TimeframeType.MINUTE_5,
        "15m": TimeframeType.MINUTE_15,
        "1h": TimeframeType.HOUR_1,
        "4h": TimeframeType.HOUR_4,
        "1d": TimeframeType.DAY_1,
        "1w": TimeframeType.WEEK_1
    }
    tf = timeframe_map.get(timeframe, TimeframeType.HOUR_1)

    # Synchronize
    synchronized_data, sync_reports = synchronizer.synchronize_agent_timestamps(
        agent_data, tf, strict_mode
    )

    # Validate temporal integrity
    validation_results = synchronizer.validate_temporal_integrity(
        synchronized_data, tf
    )

    return synchronized_data, validation_results
