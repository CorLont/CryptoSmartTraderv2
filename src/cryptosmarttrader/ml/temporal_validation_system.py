#!/usr/bin/env python3
"""
Temporal Validation System
Complete system for preventing timestamp misalignment and temporal violations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings("ignore")

from utils.timestamp_synchronizer import (
    TimestampSynchronizer,
    TimeframeType,
    SynchronizationReport,
    create_timestamp_synchronizer,
)


class TemporalViolationType(Enum):
    """Types of temporal violations"""

    TIMESTAMP_MISALIGNMENT = "timestamp_misalignment"
    TIMEZONE_INCONSISTENCY = "timezone_inconsistency"
    CANDLE_BOUNDARY_VIOLATION = "candle_boundary_violation"
    AGENT_DESYNCHRONIZATION = "agent_desynchronization"
    FUTURE_DATA_LEAKAGE = "future_data_leakage"
    IRREGULAR_INTERVALS = "irregular_intervals"


@dataclass
class TemporalViolation:
    """Single temporal violation record"""

    violation_type: TemporalViolationType
    severity: str  # 'critical', 'warning', 'info'
    agent_name: str
    timestamp: datetime
    description: str
    recommended_fix: str
    impact_assessment: str


@dataclass
class TemporalValidationReport:
    """Comprehensive temporal validation report"""

    validation_timestamp: datetime
    overall_status: str  # 'passed', 'warnings', 'critical_violations', 'failed'
    violations: List[TemporalViolation]
    critical_violations: int
    warning_violations: int
    agent_sync_scores: Dict[str, float]
    cross_agent_alignment_score: float
    temporal_integrity_score: float
    recommendation: str
    safe_for_ml_training: bool
    safe_for_prediction: bool


class TemporalValidationSystem:
    """Complete temporal validation system for multi-agent synchronization"""

    def __init__(self, primary_timeframe: str = "1h", strict_mode: bool = True):
        self.primary_timeframe = primary_timeframe
        self.strict_mode = strict_mode
        self.logger = logging.getLogger(__name__)

        # Initialize synchronizer
        self.synchronizer = create_timestamp_synchronizer(primary_timeframe)

        # Validation thresholds
        self.thresholds = {
            "max_misalignment_seconds": 1.0,  # Max 1 second misalignment
            "min_sync_quality": 0.99,  # 99% sync quality required
            "min_cross_agent_alignment": 0.95,  # 95% cross-agent alignment
            "min_temporal_integrity": 0.98,  # 98% overall integrity
        }

        # Agent expectations
        self.expected_agents = {
            "technical_analysis": {"required": True, "timeframe": primary_timeframe},
            "sentiment_analysis": {"required": True, "timeframe": primary_timeframe},
            "onchain_analysis": {"required": False, "timeframe": primary_timeframe},
            "whale_detection": {"required": False, "timeframe": primary_timeframe},
            "market_data": {"required": True, "timeframe": primary_timeframe},
        }

    def validate_complete_system(
        self, agent_data: Dict[str, pd.DataFrame], target_timeframe: Optional[str] = None
    ) -> TemporalValidationReport:
        """Complete temporal validation of all agents"""

        validation_start = datetime.utcnow()
        timeframe = target_timeframe or self.primary_timeframe

        violations = []
        agent_sync_scores = {}

        try:
            # Step 1: Individual agent validation
            for agent_name, df in agent_data.items():
                agent_violations, sync_score = self._validate_single_agent(
                    agent_name, df, timeframe
                )
                violations.extend(agent_violations)
                agent_sync_scores[agent_name] = sync_score

            # Step 2: Cross-agent synchronization validation
            cross_violations, cross_alignment_score = self._validate_cross_agent_sync(
                agent_data, timeframe
            )
            violations.extend(cross_violations)

            # Step 3: ML safety validation
            ml_violations = self._validate_ml_safety(agent_data, timeframe)
            violations.extend(ml_violations)

            # Step 4: Calculate overall scores
            temporal_integrity_score = self._calculate_temporal_integrity_score(
                agent_sync_scores, cross_alignment_score, violations
            )

            # Step 5: Categorize violations
            critical_violations = [v for v in violations if v.severity == "critical"]
            warning_violations = [v for v in violations if v.severity == "warning"]

            # Step 6: Determine overall status
            if len(critical_violations) > 0:
                overall_status = "critical_violations"
                recommendation = (
                    "BLOCK: Critical temporal violations must be fixed before proceeding"
                )
                safe_for_ml = False
                safe_for_prediction = False
            elif len(warning_violations) > 0:
                overall_status = "warnings"
                recommendation = "CAUTION: Review warnings before production use"
                safe_for_ml = temporal_integrity_score > self.thresholds["min_temporal_integrity"]
                safe_for_prediction = safe_for_ml
            else:
                overall_status = "passed"
                recommendation = "SAFE: All temporal validations passed"
                safe_for_ml = True
                safe_for_prediction = True

        except Exception as e:
            self.logger.error(f"Temporal validation failed: {e}")

            # Create error violation
            error_violation = TemporalViolation(
                violation_type=TemporalViolationType.AGENT_DESYNCHRONIZATION,
                severity="critical",
                agent_name="SYSTEM",
                timestamp=validation_start,
                description=f"Temporal validation system error: {str(e)}",
                recommended_fix="Fix underlying data issues and retry validation",
                impact_assessment="Cannot determine temporal safety",
            )

            violations = [error_violation]
            overall_status = "failed"
            recommendation = "FAILED: Fix system errors before validation"
            agent_sync_scores = {}
            cross_alignment_score = 0.0
            temporal_integrity_score = 0.0
            safe_for_ml = False
            safe_for_prediction = False

        return TemporalValidationReport(
            validation_timestamp=validation_start,
            overall_status=overall_status,
            violations=violations,
            critical_violations=len([v for v in violations if v.severity == "critical"]),
            warning_violations=len([v for v in violations if v.severity == "warning"]),
            agent_sync_scores=agent_sync_scores,
            cross_agent_alignment_score=cross_alignment_score,
            temporal_integrity_score=temporal_integrity_score,
            recommendation=recommendation,
            safe_for_ml_training=safe_for_ml,
            safe_for_prediction=safe_for_prediction,
        )

    def _validate_single_agent(
        self, agent_name: str, df: pd.DataFrame, timeframe: str
    ) -> Tuple[List[TemporalViolation], float]:
        """Validate temporal integrity of single agent"""

        violations = []

        # Find timestamp column
        timestamp_col = self._find_timestamp_column(df)
        if timestamp_col is None:
            violations.append(
                TemporalViolation(
                    violation_type=TemporalViolationType.TIMESTAMP_MISALIGNMENT,
                    severity="critical",
                    agent_name=agent_name,
                    timestamp=datetime.utcnow(),
                    description=f"No timestamp column found in {agent_name} data",
                    recommended_fix="Add 'timestamp' column with UTC datetime values",
                    impact_assessment="Cannot synchronize agent with others",
                )
            )
            return violations, 0.0

        timestamps = df[timestamp_col]

        # Validation 1: Check timezone consistency
        timezone_violations = self._check_timezone_consistency(timestamps, agent_name)
        violations.extend(timezone_violations)

        # Validation 2: Check candle boundary alignment
        alignment_violations, sync_score = self._check_candle_alignment(
            timestamps, agent_name, timeframe
        )
        violations.extend(alignment_violations)

        # Validation 3: Check interval regularity
        interval_violations = self._check_interval_regularity(timestamps, agent_name, timeframe)
        violations.extend(interval_violations)

        # Validation 4: Check for future data leakage
        leakage_violations = self._check_future_leakage(timestamps, agent_name)
        violations.extend(leakage_violations)

        return violations, sync_score

    def _validate_cross_agent_sync(
        self, agent_data: Dict[str, pd.DataFrame], timeframe: str
    ) -> Tuple[List[TemporalViolation], float]:
        """Validate synchronization across agents"""

        violations = []

        # Get all agent timestamps
        agent_timestamps = {}
        for agent_name, df in agent_data.items():
            timestamp_col = self._find_timestamp_column(df)
            if timestamp_col:
                agent_timestamps[agent_name] = set(df[timestamp_col])

        if len(agent_timestamps) < 2:
            return violations, 1.0

        # Find reference agent (one with most timestamps)
        reference_agent = max(agent_timestamps.keys(), key=lambda k: len(agent_timestamps[k]))
        reference_timestamps = agent_timestamps[reference_agent]

        # Compare other agents with reference
        total_comparisons = 0
        aligned_comparisons = 0

        for agent_name, timestamps in agent_timestamps.items():
            if agent_name == reference_agent:
                continue

            total_comparisons += 1

            # Check timestamp alignment
            missing_timestamps = reference_timestamps - timestamps
            extra_timestamps = timestamps - reference_timestamps

            if missing_timestamps or extra_timestamps:
                violations.append(
                    TemporalViolation(
                        violation_type=TemporalViolationType.AGENT_DESYNCHRONIZATION,
                        severity="critical",
                        agent_name=agent_name,
                        timestamp=datetime.utcnow(),
                        description=f"Timestamp mismatch with {reference_agent}: "
                        f"{len(missing_timestamps)} missing, {len(extra_timestamps)} extra",
                        recommended_fix=f"Synchronize {agent_name} timestamps with {reference_agent}",
                        impact_assessment="Cannot merge agent data reliably",
                    )
                )
            else:
                aligned_comparisons += 1

        # Calculate cross-agent alignment score
        alignment_score = aligned_comparisons / total_comparisons if total_comparisons > 0 else 1.0

        return violations, alignment_score

    def _validate_ml_safety(
        self, agent_data: Dict[str, pd.DataFrame], timeframe: str
    ) -> List[TemporalViolation]:
        """Validate ML training/prediction safety"""

        violations = []

        # Check for common ML temporal safety issues
        for agent_name, df in agent_data.items():
            timestamp_col = self._find_timestamp_column(df)
            if timestamp_col is None:
                continue

            timestamps = df[timestamp_col]

            # Check 1: Ensure chronological ordering
            if not timestamps.is_monotonic_increasing:
                violations.append(
                    TemporalViolation(
                        violation_type=TemporalViolationType.IRREGULAR_INTERVALS,
                        severity="critical",
                        agent_name=agent_name,
                        timestamp=datetime.utcnow(),
                        description=f"{agent_name} timestamps not in chronological order",
                        recommended_fix="Sort data by timestamp before ML processing",
                        impact_assessment="Can cause look-ahead bias in ML models",
                    )
                )

            # Check 2: Look for duplicate timestamps
            duplicate_timestamps = timestamps.duplicated().sum()
            if duplicate_timestamps > 0:
                violations.append(
                    TemporalViolation(
                        violation_type=TemporalViolationType.IRREGULAR_INTERVALS,
                        severity="warning",
                        agent_name=agent_name,
                        timestamp=datetime.utcnow(),
                        description=f"{agent_name} has {duplicate_timestamps} duplicate timestamps",
                        recommended_fix="Remove or aggregate duplicate timestamp entries",
                        impact_assessment="May cause data leakage in cross-validation",
                    )
                )

            # Check 3: Large gaps in data
            if len(timestamps) > 1:
                time_diffs = timestamps.diff().dt.total_seconds()
                expected_interval = self.synchronizer.timeframe_seconds[
                    self._get_timeframe_enum(timeframe)
                ]

                large_gaps = (time_diffs > expected_interval * 2).sum()
                if large_gaps > len(timestamps) * 0.05:  # More than 5% large gaps
                    violations.append(
                        TemporalViolation(
                            violation_type=TemporalViolationType.IRREGULAR_INTERVALS,
                            severity="warning",
                            agent_name=agent_name,
                            timestamp=datetime.utcnow(),
                            description=f"{agent_name} has {large_gaps} large time gaps",
                            recommended_fix="Fill gaps or use gap-aware ML techniques",
                            impact_assessment="May reduce ML model performance",
                        )
                    )

        return violations

    def _find_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find timestamp column in DataFrame"""

        timestamp_candidates = ["timestamp", "time", "datetime", "date"]

        for col in timestamp_candidates:
            if col in df.columns:
                return col

        # Check for datetime-like columns
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col

        return None

    def _check_timezone_consistency(
        self, timestamps: pd.Series, agent_name: str
    ) -> List[TemporalViolation]:
        """Check timezone consistency"""

        violations = []

        for i, ts in enumerate(timestamps.head(10)):  # Check first 10 timestamps
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    violations.append(
                        TemporalViolation(
                            violation_type=TemporalViolationType.TIMEZONE_INCONSISTENCY,
                            severity="critical",
                            agent_name=agent_name,
                            timestamp=ts,
                            description=f"Timezone-naive timestamp at index {i}",
                            recommended_fix="Convert all timestamps to UTC",
                            impact_assessment="Cannot reliably synchronize with other agents",
                        )
                    )
                elif ts.tzinfo != timezone.utc:
                    violations.append(
                        TemporalViolation(
                            violation_type=TemporalViolationType.TIMEZONE_INCONSISTENCY,
                            severity="warning",
                            agent_name=agent_name,
                            timestamp=ts,
                            description=f"Non-UTC timezone at index {i}: {ts.tzinfo}",
                            recommended_fix="Convert to UTC timezone",
                            impact_assessment="May cause synchronization errors",
                        )
                    )

        return violations

    def _check_candle_alignment(
        self, timestamps: pd.Series, agent_name: str, timeframe: str
    ) -> Tuple[List[TemporalViolation], float]:
        """Check alignment to candle boundaries"""

        violations = []
        timeframe_enum = self._get_timeframe_enum(timeframe)

        aligned_count = 0
        total_count = 0
        max_misalignment = 0.0

        for ts in timestamps:
            if isinstance(ts, datetime):
                total_count += 1

                # Check alignment
                alignment = self.synchronizer.align_timestamp_to_candle(ts, timeframe_enum, "floor")

                misalignment = abs(alignment.alignment_offset_seconds)
                max_misalignment = max(max_misalignment, misalignment)

                if alignment.is_perfectly_aligned:
                    aligned_count += 1
                elif misalignment > self.thresholds["max_misalignment_seconds"]:
                    violations.append(
                        TemporalViolation(
                            violation_type=TemporalViolationType.CANDLE_BOUNDARY_VIOLATION,
                            severity="critical" if misalignment > 60 else "warning",
                            agent_name=agent_name,
                            timestamp=ts,
                            description=f"Timestamp misaligned by {misalignment:.2f} seconds",
                            recommended_fix="Align timestamp to candle boundary",
                            impact_assessment="Creates label leakage in ML training",
                        )
                    )

        sync_score = aligned_count / total_count if total_count > 0 else 0.0

        return violations, sync_score

    def _check_interval_regularity(
        self, timestamps: pd.Series, agent_name: str, timeframe: str
    ) -> List[TemporalViolation]:
        """Check interval regularity"""

        violations = []

        if len(timestamps) < 2:
            return violations

        timeframe_enum = self._get_timeframe_enum(timeframe)
        expected_interval = self.synchronizer.timeframe_seconds[timeframe_enum]

        # Check intervals between consecutive timestamps
        time_diffs = timestamps.diff().dt.total_seconds().dropna()

        irregular_intervals = 0
        for i, interval in enumerate(time_diffs):
            deviation = abs(interval - expected_interval)

            if deviation > self.thresholds["max_misalignment_seconds"]:
                irregular_intervals += 1

                if irregular_intervals <= 5:  # Report first 5 violations
                    violations.append(
                        TemporalViolation(
                            violation_type=TemporalViolationType.IRREGULAR_INTERVALS,
                            severity="warning",
                            agent_name=agent_name,
                            timestamp=timestamps.iloc[i + 1],
                            description=f"Irregular interval: {interval:.1f}s (expected {expected_interval}s)",
                            recommended_fix="Ensure consistent timeframe intervals",
                            impact_assessment="May affect time series model performance",
                        )
                    )

        return violations

    def _check_future_leakage(
        self, timestamps: pd.Series, agent_name: str
    ) -> List[TemporalViolation]:
        """Check for future data leakage"""

        violations = []
        current_time = datetime.utcnow().replace(tzinfo=timezone.utc)

        future_timestamps = 0
        for ts in timestamps:
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)

                if ts > current_time:
                    future_timestamps += 1

        if future_timestamps > 0:
            violations.append(
                TemporalViolation(
                    violation_type=TemporalViolationType.FUTURE_DATA_LEAKAGE,
                    severity="critical",
                    agent_name=agent_name,
                    timestamp=current_time,
                    description=f"{future_timestamps} timestamps in the future",
                    recommended_fix="Remove future timestamps from dataset",
                    impact_assessment="Creates impossible look-ahead bias",
                )
            )

        return violations

    def _get_timeframe_enum(self, timeframe: str) -> TimeframeType:
        """Convert timeframe string to enum"""

        timeframe_map = {
            "1m": TimeframeType.MINUTE_1,
            "5m": TimeframeType.MINUTE_5,
            "15m": TimeframeType.MINUTE_15,
            "1h": TimeframeType.HOUR_1,
            "4h": TimeframeType.HOUR_4,
            "1d": TimeframeType.DAY_1,
            "1w": TimeframeType.WEEK_1,
        }

        return timeframe_map.get(timeframe, TimeframeType.HOUR_1)

    def _calculate_temporal_integrity_score(
        self,
        agent_sync_scores: Dict[str, float],
        cross_alignment_score: float,
        violations: List[TemporalViolation],
    ) -> float:
        """Calculate overall temporal integrity score"""

        # Base score from agent synchronization
        if agent_sync_scores:
            avg_agent_score = np.mean(list(agent_sync_scores.values()))
        else:
            avg_agent_score = 0.0

        # Penalty for violations
        critical_penalty = len([v for v in violations if v.severity == "critical"]) * 0.1
        warning_penalty = len([v for v in violations if v.severity == "warning"]) * 0.05

        # Combined score
        integrity_score = (
            avg_agent_score * 0.5  # Agent sync quality
            + cross_alignment_score * 0.3  # Cross-agent alignment
            + (1.0 - critical_penalty - warning_penalty) * 0.2  # Violation penalty
        )

        return max(0.0, min(1.0, integrity_score))

    def synchronize_and_validate(
        self, agent_data: Dict[str, pd.DataFrame], target_timeframe: Optional[str] = None
    ) -> Tuple[Dict[str, pd.DataFrame], TemporalValidationReport]:
        """Synchronize agents and validate temporal integrity"""

        timeframe = target_timeframe or self.primary_timeframe

        # First synchronize timestamps
        try:
            synchronized_data, sync_reports = self.synchronizer.synchronize_agent_timestamps(
                agent_data, self._get_timeframe_enum(timeframe), self.strict_mode
            )
        except Exception as e:
            # If synchronization fails, return original data with error report
            error_report = TemporalValidationReport(
                validation_timestamp=datetime.utcnow(),
                overall_status="failed",
                violations=[
                    TemporalViolation(
                        violation_type=TemporalViolationType.AGENT_DESYNCHRONIZATION,
                        severity="critical",
                        agent_name="SYSTEM",
                        timestamp=datetime.utcnow(),
                        description=f"Synchronization failed: {str(e)}",
                        recommended_fix="Fix agent data issues and retry",
                        impact_assessment="Cannot proceed with ML operations",
                    )
                ],
                critical_violations=1,
                warning_violations=0,
                agent_sync_scores={},
                cross_agent_alignment_score=0.0,
                temporal_integrity_score=0.0,
                recommendation="FIX: Resolve synchronization errors",
                safe_for_ml_training=False,
                safe_for_prediction=False,
            )

            return agent_data, error_report

        # Then validate synchronized data
        validation_report = self.validate_complete_system(synchronized_data, timeframe)

        return synchronized_data, validation_report


def create_temporal_validation_system(
    timeframe: str = "1h", strict_mode: bool = True
) -> TemporalValidationSystem:
    """Create temporal validation system"""

    return TemporalValidationSystem(primary_timeframe=timeframe, strict_mode=strict_mode)


def validate_agent_timestamps(
    agent_data: Dict[str, pd.DataFrame], timeframe: str = "1h", strict_mode: bool = True
) -> TemporalValidationReport:
    """High-level function to validate agent timestamps"""

    validator = create_temporal_validation_system(timeframe, strict_mode)
    return validator.validate_complete_system(agent_data, timeframe)
