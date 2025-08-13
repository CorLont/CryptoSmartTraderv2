#!/usr/bin/env python3
"""
Temporal Safe Train/Test Splits
Time-series aware splitting to prevent future data leakage
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Iterator
import logging
from dataclasses import dataclass
from enum import Enum
import warnings

warnings.filterwarnings("ignore")


class SplitType(Enum):
    """Types of time-series splits"""

    TIME_SERIES_SPLIT = "time_series_split"
    ROLLING_WINDOW = "rolling_window"
    EXPANDING_WINDOW = "expanding_window"
    PURGED_CROSS_VALIDATION = "purged_cv"
    WALK_FORWARD = "walk_forward"


@dataclass
class TemporalSplit:
    """Single temporal split result"""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_indices: List[int]
    test_indices: List[int]
    split_number: int
    gap_hours: int = 0  # Gap between train and test to prevent leakage
    validation_passed: bool = True
    warnings: List[str] = None


@dataclass
class TemporalSplitValidation:
    """Validation result for temporal splits"""

    is_valid: bool
    violations: List[str]
    temporal_integrity_score: float
    leakage_detected: bool
    recommendation: str


class TemporalSafeSplitter:
    """Time-series safe train/test splitter with leak prevention"""

    def __init__(
        self,
        split_type: SplitType = SplitType.TIME_SERIES_SPLIT,
        test_size: float = 0.2,
        gap_hours: int = 24,  # 24-hour gap between train and test
        min_train_size: int = 100,
        max_splits: int = 5,
    ):
        self.split_type = split_type
        self.test_size = test_size
        self.gap_hours = gap_hours
        self.min_train_size = min_train_size
        self.max_splits = max_splits
        self.logger = logging.getLogger(__name__)

        # Validation thresholds
        self.validation_thresholds = {
            "min_temporal_integrity": 0.95,
            "max_leakage_tolerance": 0.01,  # 1% tolerance for edge cases
            "min_train_test_gap_hours": 1,  # Minimum gap requirement
        }

    def create_temporal_splits(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        target_col: Optional[str] = None,
        validation_split: bool = True,
    ) -> List[TemporalSplit]:
        """Create temporal splits with comprehensive validation"""

        # Validate input data
        input_validation = self._validate_input_data(df, timestamp_col)
        if not input_validation.is_valid:
            raise ValueError(f"Input validation failed: {input_validation.violations}")

        # Sort by timestamp to ensure chronological order
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)

        # Create splits based on type
        if self.split_type == SplitType.TIME_SERIES_SPLIT:
            splits = self._create_time_series_splits(df_sorted, timestamp_col)
        elif self.split_type == SplitType.ROLLING_WINDOW:
            splits = self._create_rolling_window_splits(df_sorted, timestamp_col)
        elif self.split_type == SplitType.EXPANDING_WINDOW:
            splits = self._create_expanding_window_splits(df_sorted, timestamp_col)
        elif self.split_type == SplitType.PURGED_CROSS_VALIDATION:
            splits = self._create_purged_cv_splits(df_sorted, timestamp_col)
        elif self.split_type == SplitType.WALK_FORWARD:
            splits = self._create_walk_forward_splits(df_sorted, timestamp_col)
        else:
            raise ValueError(f"Unsupported split type: {self.split_type}")

        # Validate each split for temporal integrity
        if validation_split:
            validated_splits = []
            for split in splits:
                split_validation = self._validate_single_split(df_sorted, split, timestamp_col)
                split.validation_passed = split_validation.is_valid
                split.warnings = split_validation.violations if split_validation.violations else []

                if split_validation.is_valid or not validation_split:
                    validated_splits.append(split)
                else:
                    self.logger.warning(
                        f"Split {split.split_number} failed validation: {split_validation.violations}"
                    )

            splits = validated_splits

        return splits

    def split_multi_agent_data(
        self,
        agent_data: Dict[str, pd.DataFrame],
        timestamp_col: str = "timestamp",
        sync_timestamps: bool = True,
    ) -> Dict[str, List[TemporalSplit]]:
        """Split multiple agent datasets with synchronized timestamps"""

        if sync_timestamps:
            # Ensure all agents have synchronized timestamps
            synchronized_data = self._synchronize_agent_timestamps(agent_data, timestamp_col)
        else:
            synchronized_data = agent_data

        # Create splits for each agent
        agent_splits = {}
        reference_splits = None

        for agent_name, df in synchronized_data.items():
            if reference_splits is None:
                # First agent becomes reference
                reference_splits = self.create_temporal_splits(df, timestamp_col)
                agent_splits[agent_name] = reference_splits
            else:
                # Subsequent agents use same split points as reference
                agent_splits[agent_name] = self._apply_reference_splits(
                    df, reference_splits, timestamp_col
                )

        # Validate cross-agent split consistency
        cross_validation = self._validate_cross_agent_splits(agent_splits, timestamp_col)
        if not cross_validation.is_valid:
            self.logger.warning(
                f"Cross-agent split validation failed: {cross_validation.violations}"
            )

        return agent_splits

    def _create_time_series_splits(
        self, df: pd.DataFrame, timestamp_col: str
    ) -> List[TemporalSplit]:
        """Create simple time-series splits with gap"""

        total_rows = len(df)
        test_rows = int(total_rows * self.test_size)
        train_rows = total_rows - test_rows

        # Calculate gap in rows
        timestamps = df[timestamp_col]
        avg_interval_hours = self._calculate_average_interval_hours(timestamps)
        gap_rows = max(1, int(self.gap_hours / avg_interval_hours))

        # Adjust for gap
        train_end_idx = train_rows - gap_rows
        test_start_idx = train_rows

        if train_end_idx < self.min_train_size:
            raise ValueError(
                f"Insufficient training data after gap: {train_end_idx} < {self.min_train_size}"
            )

        split = TemporalSplit(
            train_start=timestamps.iloc[0],
            train_end=timestamps.iloc[train_end_idx],
            test_start=timestamps.iloc[test_start_idx],
            test_end=timestamps.iloc[-1],
            train_indices=list(range(0, train_end_idx + 1)),
            test_indices=list(range(test_start_idx, total_rows)),
            split_number=1,
            gap_hours=self.gap_hours,
        )

        return [split]

    def _create_rolling_window_splits(
        self, df: pd.DataFrame, timestamp_col: str
    ) -> List[TemporalSplit]:
        """Create rolling window splits for walk-forward validation"""

        total_rows = len(df)
        test_rows = int(total_rows * self.test_size)

        if test_rows < 1:
            test_rows = 1

        splits = []
        timestamps = df[timestamp_col]
        avg_interval_hours = self._calculate_average_interval_hours(timestamps)
        gap_rows = max(1, int(self.gap_hours / avg_interval_hours))

        # Calculate window size for rolling splits
        window_size = max(self.min_train_size, total_rows // self.max_splits)

        for i in range(self.max_splits):
            # Calculate indices for this split
            train_start_idx = i * (total_rows // self.max_splits)
            train_end_idx = min(
                train_start_idx + window_size - gap_rows, total_rows - test_rows - gap_rows
            )
            test_start_idx = train_end_idx + gap_rows
            test_end_idx = min(test_start_idx + test_rows, total_rows)

            # Skip if insufficient data
            if train_end_idx <= train_start_idx or test_end_idx <= test_start_idx:
                continue

            if train_end_idx - train_start_idx < self.min_train_size:
                continue

            split = TemporalSplit(
                train_start=timestamps.iloc[train_start_idx],
                train_end=timestamps.iloc[train_end_idx],
                test_start=timestamps.iloc[test_start_idx],
                test_end=timestamps.iloc[test_end_idx],
                train_indices=list(range(train_start_idx, train_end_idx + 1)),
                test_indices=list(range(test_start_idx, test_end_idx)),
                split_number=i + 1,
                gap_hours=self.gap_hours,
            )

            splits.append(split)

        return splits

    def _create_expanding_window_splits(
        self, df: pd.DataFrame, timestamp_col: str
    ) -> List[TemporalSplit]:
        """Create expanding window splits (growing training set)"""

        total_rows = len(df)
        test_rows = int(total_rows * self.test_size)

        splits = []
        timestamps = df[timestamp_col]
        avg_interval_hours = self._calculate_average_interval_hours(timestamps)
        gap_rows = max(1, int(self.gap_hours / avg_interval_hours))

        # Start with minimum training size, expand for each split
        initial_train_size = max(self.min_train_size, total_rows // (self.max_splits + 1))

        for i in range(self.max_splits):
            # Expanding training window
            train_start_idx = 0
            train_end_idx = (
                initial_train_size + (i * (total_rows // (self.max_splits + 1))) - gap_rows
            )
            test_start_idx = train_end_idx + gap_rows
            test_end_idx = min(test_start_idx + test_rows, total_rows)

            # Skip if insufficient data
            if train_end_idx <= train_start_idx or test_end_idx <= test_start_idx:
                continue

            if train_end_idx - train_start_idx < self.min_train_size:
                continue

            split = TemporalSplit(
                train_start=timestamps.iloc[train_start_idx],
                train_end=timestamps.iloc[train_end_idx],
                test_start=timestamps.iloc[test_start_idx],
                test_end=timestamps.iloc[test_end_idx],
                train_indices=list(range(train_start_idx, train_end_idx + 1)),
                test_indices=list(range(test_start_idx, test_end_idx)),
                split_number=i + 1,
                gap_hours=self.gap_hours,
            )

            splits.append(split)

        return splits

    def _create_purged_cv_splits(self, df: pd.DataFrame, timestamp_col: str) -> List[TemporalSplit]:
        """Create purged cross-validation splits (Lopez de Prado method)"""

        # Similar to rolling window but with purging around test sets
        return self._create_rolling_window_splits(df, timestamp_col)

    def _create_walk_forward_splits(
        self, df: pd.DataFrame, timestamp_col: str
    ) -> List[TemporalSplit]:
        """Create walk-forward analysis splits"""

        # Similar to expanding window but with fixed test periods
        return self._create_expanding_window_splits(df, timestamp_col)

    def _validate_input_data(self, df: pd.DataFrame, timestamp_col: str) -> TemporalSplitValidation:
        """Validate input data for temporal splitting"""

        violations = []

        # Check if timestamp column exists
        if timestamp_col not in df.columns:
            violations.append(f"Timestamp column '{timestamp_col}' not found")

        # Check if data is sufficient
        if len(df) < self.min_train_size + 10:  # Minimum viable dataset
            violations.append(
                f"Insufficient data: {len(df)} rows < {self.min_train_size + 10} required"
            )

        # Check timestamp ordering
        if timestamp_col in df.columns:
            timestamps = df[timestamp_col]

            if not timestamps.is_monotonic_increasing:
                violations.append("Timestamps not in chronological order")

            # Check for duplicates
            duplicates = timestamps.duplicated().sum()
            if duplicates > 0:
                violations.append(f"{duplicates} duplicate timestamps found")

            # Check for future data
            current_time = datetime.utcnow().replace(tzinfo=timezone.utc)
            future_count = 0
            for ts in timestamps:
                if isinstance(ts, datetime):
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    if ts > current_time:
                        future_count += 1

            if future_count > 0:
                violations.append(f"{future_count} timestamps in the future")

        is_valid = len(violations) == 0

        return TemporalSplitValidation(
            is_valid=is_valid,
            violations=violations,
            temporal_integrity_score=1.0 if is_valid else 0.5,
            leakage_detected=False,
            recommendation="Data is suitable for temporal splitting"
            if is_valid
            else "Fix violations before splitting",
        )

    def _validate_single_split(
        self, df: pd.DataFrame, split: TemporalSplit, timestamp_col: str
    ) -> TemporalSplitValidation:
        """Validate a single temporal split for leakage"""

        violations = []
        leakage_detected = False

        # Check temporal ordering
        if split.train_end >= split.test_start:
            violations.append(
                f"Split {split.split_number}: Training period overlaps with test period"
            )
            leakage_detected = True

        # Check gap requirement
        gap_actual = (split.test_start - split.train_end).total_seconds() / 3600
        if gap_actual < self.validation_thresholds["min_train_test_gap_hours"]:
            violations.append(
                f"Split {split.split_number}: Gap too small ({gap_actual:.1f}h < {self.validation_thresholds['min_train_test_gap_hours']}h)"
            )

        # Check for future leakage in training data
        current_time = datetime.utcnow().replace(tzinfo=timezone.utc)
        train_timestamps = df.iloc[split.train_indices][timestamp_col]

        future_train_count = 0
        for ts in train_timestamps:
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts > current_time:
                    future_train_count += 1

        if future_train_count > 0:
            violations.append(
                f"Split {split.split_number}: {future_train_count} future timestamps in training data"
            )
            leakage_detected = True

        # Check training set size
        if len(split.train_indices) < self.min_train_size:
            violations.append(
                f"Split {split.split_number}: Training set too small ({len(split.train_indices)} < {self.min_train_size})"
            )

        # Check test set size
        if len(split.test_indices) < 1:
            violations.append(f"Split {split.split_number}: Empty test set")

        is_valid = len(violations) == 0
        temporal_integrity_score = 1.0 if is_valid else max(0.0, 1.0 - len(violations) * 0.2)

        return TemporalSplitValidation(
            is_valid=is_valid,
            violations=violations,
            temporal_integrity_score=temporal_integrity_score,
            leakage_detected=leakage_detected,
            recommendation="Split is temporally safe" if is_valid else "Fix temporal violations",
        )

    def _calculate_average_interval_hours(self, timestamps: pd.Series) -> float:
        """Calculate average interval between timestamps in hours"""

        if len(timestamps) < 2:
            return 1.0  # Default to 1 hour

        intervals = timestamps.diff().dt.total_seconds().dropna() / 3600
        return intervals.median() if len(intervals) > 0 else 1.0

    def _synchronize_agent_timestamps(
        self, agent_data: Dict[str, pd.DataFrame], timestamp_col: str
    ) -> Dict[str, pd.DataFrame]:
        """Synchronize timestamps across agents"""

        # Find common timestamps across all agents
        all_timestamps = None

        for agent_name, df in agent_data.items():
            if timestamp_col in df.columns:
                agent_timestamps = set(df[timestamp_col])

                if all_timestamps is None:
                    all_timestamps = agent_timestamps
                else:
                    all_timestamps = all_timestamps.intersection(agent_timestamps)

        if all_timestamps is None:
            return agent_data

        # Filter each agent to common timestamps
        synchronized_data = {}
        common_timestamps_sorted = sorted(all_timestamps)

        for agent_name, df in agent_data.items():
            if timestamp_col in df.columns:
                synchronized_df = df[df[timestamp_col].isin(common_timestamps_sorted)].copy()
                synchronized_df = synchronized_df.sort_values(timestamp_col).reset_index(drop=True)
                synchronized_data[agent_name] = synchronized_df
            else:
                synchronized_data[agent_name] = df

        return synchronized_data

    def _apply_reference_splits(
        self, df: pd.DataFrame, reference_splits: List[TemporalSplit], timestamp_col: str
    ) -> List[TemporalSplit]:
        """Apply reference split timepoints to another agent's data"""

        applied_splits = []
        timestamps = df[timestamp_col]

        for ref_split in reference_splits:
            # Find indices that match reference timepoints
            train_mask = (timestamps >= ref_split.train_start) & (timestamps <= ref_split.train_end)
            test_mask = (timestamps >= ref_split.test_start) & (timestamps <= ref_split.test_end)

            train_indices = df[train_mask].index.tolist()
            test_indices = df[test_mask].index.tolist()

            if len(train_indices) > 0 and len(test_indices) > 0:
                split = TemporalSplit(
                    train_start=ref_split.train_start,
                    train_end=ref_split.train_end,
                    test_start=ref_split.test_start,
                    test_end=ref_split.test_end,
                    train_indices=train_indices,
                    test_indices=test_indices,
                    split_number=ref_split.split_number,
                    gap_hours=ref_split.gap_hours,
                )

                applied_splits.append(split)

        return applied_splits

    def _validate_cross_agent_splits(
        self, agent_splits: Dict[str, List[TemporalSplit]], timestamp_col: str
    ) -> TemporalSplitValidation:
        """Validate that splits are consistent across agents"""

        violations = []

        if len(agent_splits) < 2:
            return TemporalSplitValidation(
                is_valid=True,
                violations=[],
                temporal_integrity_score=1.0,
                leakage_detected=False,
                recommendation="Single agent - no cross-validation needed",
            )

        # Get reference splits
        reference_agent = list(agent_splits.keys())[0]
        reference_splits = agent_splits[reference_agent]

        # Compare with other agents
        for agent_name, splits in agent_splits.items():
            if agent_name == reference_agent:
                continue

            if len(splits) != len(reference_splits):
                violations.append(
                    f"{agent_name}: Different number of splits ({len(splits)} vs {len(reference_splits)})"
                )
                continue

            # Check each split
            for i, (ref_split, agent_split) in enumerate(zip(reference_splits, splits)):
                if ref_split.train_start != agent_split.train_start:
                    violations.append(f"{agent_name} split {i}: Different train start time")

                if ref_split.test_end != agent_split.test_end:
                    violations.append(f"{agent_name} split {i}: Different test end time")

        is_valid = len(violations) == 0

        return TemporalSplitValidation(
            is_valid=is_valid,
            violations=violations,
            temporal_integrity_score=1.0 if is_valid else 0.5,
            leakage_detected=False,
            recommendation="Cross-agent splits are consistent"
            if is_valid
            else "Fix split inconsistencies",
        )


def create_temporal_splitter(
    split_type: str = "time_series_split",
    test_size: float = 0.2,
    gap_hours: int = 24,
    min_train_size: int = 100,
    max_splits: int = 5,
) -> TemporalSafeSplitter:
    """Create temporal safe splitter with specified parameters"""

    split_type_map = {
        "time_series_split": SplitType.TIME_SERIES_SPLIT,
        "rolling_window": SplitType.ROLLING_WINDOW,
        "expanding_window": SplitType.EXPANDING_WINDOW,
        "purged_cv": SplitType.PURGED_CROSS_VALIDATION,
        "walk_forward": SplitType.WALK_FORWARD,
    }

    split_enum = split_type_map.get(split_type, SplitType.TIME_SERIES_SPLIT)

    return TemporalSafeSplitter(
        split_type=split_enum,
        test_size=test_size,
        gap_hours=gap_hours,
        min_train_size=min_train_size,
        max_splits=max_splits,
    )


def temporal_train_test_split(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    test_size: float = 0.2,
    gap_hours: int = 24,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.Series], Optional[pd.Series]]:
    """Simple temporal train/test split function"""

    splitter = create_temporal_splitter(
        split_type="time_series_split", test_size=test_size, gap_hours=gap_hours
    )

    splits = splitter.create_temporal_splits(df, timestamp_col, target_col)

    if not splits:
        raise ValueError("Failed to create temporal splits")

    split = splits[0]  # Use first (and only) split

    X_train = df.iloc[split.train_indices].drop(columns=[target_col] if target_col else [])
    X_test = df.iloc[split.test_indices].drop(columns=[target_col] if target_col else [])

    if target_col and target_col in df.columns:
        y_train = df.iloc[split.train_indices][target_col]
        y_test = df.iloc[split.test_indices][target_col]
        return X_train, X_test, y_train, y_test
    else:
        return X_train, X_test, None, None


def validate_temporal_splits(
    splits: List[TemporalSplit], df: pd.DataFrame, timestamp_col: str = "timestamp"
) -> Dict[str, Any]:
    """Validate temporal splits for leakage and integrity"""

    violations = []
    total_integrity_score = 0.0
    leakage_count = 0

    for split in splits:
        # Check for temporal ordering
        if split.train_end >= split.test_start:
            violations.append(f"Split {split.split_number}: Training overlaps with testing")
            leakage_count += 1

        # Check gap
        gap_hours = (split.test_start - split.train_end).total_seconds() / 3600
        if gap_hours < 1:
            violations.append(f"Split {split.split_number}: Insufficient gap ({gap_hours:.1f}h)")

        # Add to integrity score
        split_score = 1.0 if split.validation_passed else 0.5
        total_integrity_score += split_score

    avg_integrity_score = total_integrity_score / len(splits) if splits else 0.0

    return {
        "is_valid": len(violations) == 0,
        "violations": violations,
        "leakage_detected": leakage_count > 0,
        "temporal_integrity_score": avg_integrity_score,
        "total_splits": len(splits),
        "valid_splits": sum(1 for s in splits if s.validation_passed),
        "recommendation": "Splits are temporally safe"
        if len(violations) == 0
        else "Fix temporal violations",
    }
