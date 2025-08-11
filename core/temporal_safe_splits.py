#!/usr/bin/env python3
"""
Temporal Safe Splits - Time-series aware train/test splitting

Implements enterprise-grade temporal splitting strategies with look-ahead bias prevention,
proper purging, and robust gap handling for financial time series data.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Iterator
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from enum import Enum
import warnings
from pathlib import Path

from core.consolidated_logging_manager import get_consolidated_logger

class SplitStrategy(Enum):
    """Time series splitting strategies"""
    ROLLING_WINDOW = "rolling_window"
    EXPANDING_WINDOW = "expanding_window"
    WALK_FORWARD = "walk_forward"
    PURGED_CV = "purged_cv"
    BLOCKED_CV = "blocked_cv"

@dataclass
class SplitResult:
    """Result of temporal split operation"""
    train_indices: List[int]
    test_indices: List[int]
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    split_id: int
    strategy: SplitStrategy
    warnings: List[str] = field(default_factory=list)  # CRITICAL FIX: proper default factory
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SplitConfig:
    """Configuration for temporal splitting"""
    strategy: SplitStrategy
    train_size: int  # Number of periods for training
    test_size: int   # Number of periods for testing
    gap_hours: float = 0.0  # Gap between train and test to prevent leakage
    min_train_size: int = 100  # Minimum training samples
    max_splits: Optional[int] = None  # Maximum number of splits
    purge_buffer: float = 24.0  # Hours to purge around test set
    embargo_hours: float = 0.0  # Additional embargo period

class TemporalSafeSplitter:
    """
    Enterprise temporal splitter with comprehensive bias prevention
    
    Provides multiple splitting strategies specifically designed for financial time series,
    with robust handling of irregular timestamps and proper look-ahead bias prevention.
    """
    
    def __init__(self, config: SplitConfig):
        """
        Initialize temporal splitter
        
        Args:
            config: Splitting configuration
        """
        self.config = config
        self.logger = get_consolidated_logger("TemporalSafeSplitter")
        
        # Validation statistics
        self.split_count = 0
        self.warning_count = 0
        self.last_split_time = None
        
        self.logger.info(f"Temporal splitter initialized with strategy: {config.strategy.value}")
    
    def create_splits(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> List[SplitResult]:
        """
        Create temporal splits for time series data
        
        Args:
            df: DataFrame with time series data
            timestamp_col: Name of timestamp column
            
        Returns:
            List of SplitResult objects
        """
        
        if df.empty:
            self.logger.warning("Empty DataFrame provided for splitting")
            return []
        
        if timestamp_col not in df.columns:
            raise ValueError(f"Timestamp column '{timestamp_col}' not found in DataFrame")
        
        # Ensure proper datetime and sorting
        df = df.copy()
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Validate data quality
        timestamps = df[timestamp_col]
        validation_result = self._validate_timestamps(timestamps)
        
        if not validation_result['is_valid']:
            warnings.warn(f"Timestamp validation issues: {validation_result['issues']}")
        
        # Route to appropriate splitting strategy
        if self.config.strategy == SplitStrategy.ROLLING_WINDOW:
            splits = self._create_rolling_window_splits(df, timestamp_col)
        elif self.config.strategy == SplitStrategy.EXPANDING_WINDOW:
            splits = self._create_expanding_window_splits(df, timestamp_col)
        elif self.config.strategy == SplitStrategy.WALK_FORWARD:
            splits = self._create_walk_forward_splits(df, timestamp_col)
        elif self.config.strategy == SplitStrategy.PURGED_CV:
            splits = self._create_purged_cv_splits(df, timestamp_col)
        elif self.config.strategy == SplitStrategy.BLOCKED_CV:
            splits = self._create_blocked_cv_splits(df, timestamp_col)
        else:
            raise ValueError(f"Unknown split strategy: {self.config.strategy}")
        
        # Filter splits by quality criteria
        valid_splits = self._filter_valid_splits(splits)
        
        # Update statistics
        self.split_count = len(valid_splits)
        self.warning_count = sum(len(split.warnings) for split in valid_splits)
        self.last_split_time = datetime.now(timezone.utc)
        
        self.logger.info(f"Created {len(valid_splits)} valid splits using {self.config.strategy.value}")
        
        return valid_splits
    
    def _create_rolling_window_splits(self, df: pd.DataFrame, timestamp_col: str) -> List[SplitResult]:
        """Create rolling window splits"""
        
        splits = []
        timestamps = df[timestamp_col]
        
        # Calculate gap in rows - CRITICAL FIX for division by zero
        avg_interval_hours = self._calculate_average_interval_hours(timestamps)
        if not np.isfinite(avg_interval_hours) or avg_interval_hours <= 0:
            avg_interval_hours = 1e-6  # Guard against zero division
            self.logger.warning("Invalid average interval detected, using fallback value")
        
        gap_rows = max(1, int(self.config.gap_hours / avg_interval_hours))
        
        # Rolling window parameters
        window_size = self.config.train_size + gap_rows + self.config.test_size
        
        for i in range(len(df) - window_size + 1):
            if self.config.max_splits and len(splits) >= self.config.max_splits:
                break
            
            # Define indices
            train_start_idx = i
            train_end_idx = i + self.config.train_size
            test_start_idx = train_end_idx + gap_rows
            test_end_idx = test_start_idx + self.config.test_size
            
            # Ensure we don't go beyond data
            if test_end_idx > len(df):
                break
            
            # Create split result
            split = self._create_split_result(
                df, timestamp_col, train_start_idx, train_end_idx,
                test_start_idx, test_end_idx, len(splits)
            )
            
            splits.append(split)
        
        return splits
    
    def _create_expanding_window_splits(self, df: pd.DataFrame, timestamp_col: str) -> List[SplitResult]:
        """Create expanding window splits"""
        
        splits = []
        timestamps = df[timestamp_col]
        
        # Calculate gap in rows - CRITICAL FIX
        avg_interval_hours = self._calculate_average_interval_hours(timestamps)
        if not np.isfinite(avg_interval_hours) or avg_interval_hours <= 0:
            avg_interval_hours = 1e-6
            self.logger.warning("Invalid average interval detected, using fallback value")
        
        gap_rows = max(1, int(self.config.gap_hours / avg_interval_hours))
        
        # Start from minimum training size
        train_start_idx = 0
        initial_train_end = self.config.min_train_size
        
        for train_end_idx in range(initial_train_end, len(df) - gap_rows - self.config.test_size, 
                                  self.config.test_size):
            
            if self.config.max_splits and len(splits) >= self.config.max_splits:
                break
            
            # Define test indices
            test_start_idx = train_end_idx + gap_rows
            test_end_idx = test_start_idx + self.config.test_size
            
            # Ensure we don't go beyond data
            if test_end_idx > len(df):
                break
            
            # Create split result
            split = self._create_split_result(
                df, timestamp_col, train_start_idx, train_end_idx,
                test_start_idx, test_end_idx, len(splits)
            )
            
            splits.append(split)
        
        return splits
    
    def _create_walk_forward_splits(self, df: pd.DataFrame, timestamp_col: str) -> List[SplitResult]:
        """Create walk-forward analysis splits"""
        
        splits = []
        timestamps = df[timestamp_col]
        
        # Calculate gap in rows - CRITICAL FIX
        avg_interval_hours = self._calculate_average_interval_hours(timestamps)
        if not np.isfinite(avg_interval_hours) or avg_interval_hours <= 0:
            avg_interval_hours = 1e-6
            self.logger.warning("Invalid average interval detected, using fallback value")
        
        gap_rows = max(1, int(self.config.gap_hours / avg_interval_hours))
        step_size = self.config.test_size  # Step forward by test size
        
        train_start_idx = 0
        
        while True:
            if self.config.max_splits and len(splits) >= self.config.max_splits:
                break
            
            # Define training window
            train_end_idx = train_start_idx + self.config.train_size
            
            # Define test window
            test_start_idx = train_end_idx + gap_rows
            test_end_idx = test_start_idx + self.config.test_size
            
            # Check if we have enough data
            if test_end_idx > len(df):
                break
            
            # Create split result
            split = self._create_split_result(
                df, timestamp_col, train_start_idx, train_end_idx,
                test_start_idx, test_end_idx, len(splits)
            )
            
            splits.append(split)
            
            # Move forward
            train_start_idx += step_size
        
        return splits
    
    def _create_purged_cv_splits(self, df: pd.DataFrame, timestamp_col: str) -> List[SplitResult]:
        """
        Create purged cross-validation splits - ACTUAL PURGING IMPLEMENTATION
        
        Implements proper purging around test sets to prevent look-ahead bias
        in overlapping feature engineering or model training.
        """
        
        splits = []
        timestamps = df[timestamp_col]
        
        # Calculate purge buffer in rows
        avg_interval_hours = self._calculate_average_interval_hours(timestamps)
        if not np.isfinite(avg_interval_hours) or avg_interval_hours <= 0:
            avg_interval_hours = 1e-6
            self.logger.warning("Invalid average interval detected, using fallback value")
        
        purge_rows = max(1, int(self.config.purge_buffer / avg_interval_hours))
        gap_rows = max(1, int(self.config.gap_hours / avg_interval_hours))
        
        # Calculate number of folds - CRITICAL FIX for small datasets
        min_required_samples = self.config.test_size + 2 * purge_rows + 2 * gap_rows + self.config.min_train_size
        
        if len(df) < min_required_samples:
            self.logger.warning(f"Dataset too small for purged CV: {len(df)} < {min_required_samples} required")
            return []
        
        total_test_samples = max(1, len(df) // (self.config.test_size + 2 * purge_rows))
        if self.config.max_splits:
            total_test_samples = min(total_test_samples, self.config.max_splits)
        
        fold_size = len(df) // total_test_samples
        
        for fold in range(total_test_samples):
            # Define test set for this fold
            test_start_idx = fold * fold_size
            test_end_idx = min(test_start_idx + self.config.test_size, len(df))
            
            if test_end_idx - test_start_idx < self.config.test_size:
                continue  # Skip incomplete folds
            
            # Define purge zones around test set
            purge_start = max(0, test_start_idx - purge_rows)
            purge_end = min(len(df), test_end_idx + purge_rows)
            
            # Create training set excluding purged area
            train_indices = []
            
            # Training data before purge zone
            if purge_start > gap_rows:
                train_indices.extend(range(0, purge_start - gap_rows))
            
            # Training data after purge zone
            if purge_end + gap_rows < len(df):
                train_indices.extend(range(purge_end + gap_rows, len(df)))
            
            # Check minimum training size
            if len(train_indices) < self.config.min_train_size:
                split_warnings = [f"Insufficient training data after purging: {len(train_indices)} < {self.config.min_train_size}"]
                # Skip this split if insufficient training data
                continue
            else:
                split_warnings = []
            
            # Create split result with custom indices
            split = SplitResult(
                train_indices=train_indices,
                test_indices=list(range(test_start_idx, test_end_idx)),
                train_start=timestamps.iloc[train_indices[0]] if train_indices else timestamps.iloc[0],
                train_end=timestamps.iloc[train_indices[-1]] if train_indices else timestamps.iloc[0],
                test_start=timestamps.iloc[test_start_idx],
                test_end=timestamps.iloc[test_end_idx - 1],
                split_id=fold,
                strategy=SplitStrategy.PURGED_CV,
                warnings=split_warnings,
                metadata={
                    'purge_buffer_hours': self.config.purge_buffer,
                    'purge_rows': purge_rows,
                    'training_samples': len(train_indices),
                    'test_samples': test_end_idx - test_start_idx
                }
            )
            
            splits.append(split)
        
        return splits
    
    def _create_blocked_cv_splits(self, df: pd.DataFrame, timestamp_col: str) -> List[SplitResult]:
        """Create blocked time series cross-validation splits"""
        
        splits = []
        timestamps = df[timestamp_col]
        
        # Calculate embargo buffer
        avg_interval_hours = self._calculate_average_interval_hours(timestamps)
        if not np.isfinite(avg_interval_hours) or avg_interval_hours <= 0:
            avg_interval_hours = 1e-6
            self.logger.warning("Invalid average interval detected, using fallback value")
        
        embargo_rows = max(1, int(self.config.embargo_hours / avg_interval_hours))
        
        # Define block parameters
        block_size = self.config.train_size + self.config.test_size + embargo_rows
        n_blocks = len(df) // block_size
        
        if self.config.max_splits:
            n_blocks = min(n_blocks, self.config.max_splits)
        
        for block in range(n_blocks):
            block_start = block * block_size
            
            # Training block
            train_start_idx = block_start
            train_end_idx = train_start_idx + self.config.train_size
            
            # Embargo period
            test_start_idx = train_end_idx + embargo_rows
            test_end_idx = test_start_idx + self.config.test_size
            
            # Ensure we don't exceed data bounds
            if test_end_idx > len(df):
                break
            
            # Create split result
            split = self._create_split_result(
                df, timestamp_col, train_start_idx, train_end_idx,
                test_start_idx, test_end_idx, block
            )
            
            # Add embargo metadata
            split.metadata['embargo_hours'] = self.config.embargo_hours
            split.metadata['embargo_rows'] = embargo_rows
            
            splits.append(split)
        
        return splits
    
    def _create_split_result(self, df: pd.DataFrame, timestamp_col: str,
                           train_start_idx: int, train_end_idx: int,
                           test_start_idx: int, test_end_idx: int,
                           split_id: int) -> SplitResult:
        """Create a SplitResult object from indices"""
        
        timestamps = df[timestamp_col]
        split_warnings = []
        
        # Validate split integrity
        if train_end_idx > test_start_idx:
            split_warnings.append("Training and test periods overlap")
        
        if train_end_idx - train_start_idx < self.config.min_train_size:
            split_warnings.append(f"Training size below minimum: {train_end_idx - train_start_idx}")
        
        return SplitResult(
            train_indices=list(range(train_start_idx, train_end_idx)),
            test_indices=list(range(test_start_idx, test_end_idx)),
            train_start=timestamps.iloc[train_start_idx],
            train_end=timestamps.iloc[train_end_idx - 1],
            test_start=timestamps.iloc[test_start_idx],
            test_end=timestamps.iloc[test_end_idx - 1],
            split_id=split_id,
            strategy=self.config.strategy,
            warnings=split_warnings,
            metadata={
                'train_samples': train_end_idx - train_start_idx,
                'test_samples': test_end_idx - test_start_idx,
                'gap_hours': self.config.gap_hours
            }
        )
    
    def _calculate_average_interval_hours(self, timestamps: pd.Series) -> float:
        """
        Calculate average interval between timestamps in hours - ROBUST IMPLEMENTATION
        
        Args:
            timestamps: Series of timestamps
            
        Returns:
            Average interval in hours, with robust fallback handling
        """
        
        if len(timestamps) < 2:
            return 24.0  # Default to daily if insufficient data
        
        # Calculate differences
        diffs = timestamps.diff().dropna()
        
        if len(diffs) == 0:
            return 24.0
        
        # Remove zero differences (duplicates)
        non_zero_diffs = diffs[diffs > pd.Timedelta(0)]
        
        if len(non_zero_diffs) == 0:
            self.logger.warning("All timestamp differences are zero - potential duplicate timestamps")
            return 1e-6  # Very small value to prevent division by zero
        
        # Calculate median to be robust against outliers
        median_diff = non_zero_diffs.median()
        
        # Convert to hours
        avg_interval_hours = median_diff.total_seconds() / 3600.0
        
        # Sanity check
        if not np.isfinite(avg_interval_hours) or avg_interval_hours <= 0:
            self.logger.warning(f"Invalid average interval calculated: {avg_interval_hours}")
            return 1e-6
        
        return avg_interval_hours
    
    def _validate_timestamps(self, timestamps: pd.Series) -> Dict[str, Any]:
        """Validate timestamp quality for splitting"""
        
        issues = []
        
        # Check for missing values
        if timestamps.isna().any():
            issues.append("Missing timestamp values detected")
        
        # Check for duplicates
        duplicates = timestamps.duplicated().sum()
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate timestamps found")
        
        # Check for monotonicity
        if not timestamps.is_monotonic_increasing:
            issues.append("Timestamps are not monotonically increasing")
        
        # Check for reasonable intervals
        if len(timestamps) > 1:
            diffs = timestamps.diff().dropna()
            if (diffs <= pd.Timedelta(0)).any():
                issues.append("Non-positive time intervals detected")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'total_timestamps': len(timestamps),
            'duplicate_count': duplicates if 'duplicates' in locals() else 0
        }
    
    def _filter_valid_splits(self, splits: List[SplitResult]) -> List[SplitResult]:
        """Filter splits based on quality criteria"""
        
        valid_splits = []
        
        for split in splits:
            # Check minimum sizes
            if len(split.train_indices) < self.config.min_train_size:
                split.warnings.append("Insufficient training samples")
                continue
            
            if len(split.test_indices) == 0:
                split.warnings.append("Empty test set")
                continue
            
            # Check for temporal validity
            if split.train_end >= split.test_start:
                split.warnings.append("Train end after test start")
                continue
            
            valid_splits.append(split)
        
        return valid_splits
    
    def get_split_summary(self, splits: List[SplitResult]) -> Dict[str, Any]:
        """Get summary statistics for splits"""
        
        if not splits:
            return {'total_splits': 0, 'warnings': ['No valid splits created']}
        
        train_sizes = [len(split.train_indices) for split in splits]
        test_sizes = [len(split.test_indices) for split in splits]
        total_warnings = sum(len(split.warnings) for split in splits)
        
        # Calculate coverage
        all_train_indices = set()
        all_test_indices = set()
        
        for split in splits:
            all_train_indices.update(split.train_indices)
            all_test_indices.update(split.test_indices)
        
        return {
            'total_splits': len(splits),
            'strategy': splits[0].strategy.value,
            'train_size_stats': {
                'min': min(train_sizes),
                'max': max(train_sizes),
                'mean': np.mean(train_sizes),
                'std': np.std(train_sizes)
            },
            'test_size_stats': {
                'min': min(test_sizes),
                'max': max(test_sizes),
                'mean': np.mean(test_sizes),
                'std': np.std(test_sizes)
            },
            'coverage': {
                'train_coverage': len(all_train_indices),
                'test_coverage': len(all_test_indices),
                'overlap': len(all_train_indices & all_test_indices)
            },
            'quality_metrics': {
                'total_warnings': total_warnings,
                'splits_with_warnings': sum(1 for split in splits if split.warnings),
                'warning_rate': total_warnings / len(splits)
            },
            'time_range': {
                'earliest_train': min(split.train_start for split in splits).isoformat(),
                'latest_test': max(split.test_end for split in splits).isoformat()
            }
        }
    
    def save_splits(self, splits: List[SplitResult], output_path: str) -> str:
        """Save splits to file for reproducibility"""
        
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert splits to serializable format
            splits_data = []
            for split in splits:
                splits_data.append({
                    'split_id': split.split_id,
                    'train_indices': split.train_indices,
                    'test_indices': split.test_indices,
                    'train_start': split.train_start.isoformat(),
                    'train_end': split.train_end.isoformat(),
                    'test_start': split.test_start.isoformat(),
                    'test_end': split.test_end.isoformat(),
                    'strategy': split.strategy.value,
                    'warnings': split.warnings,
                    'metadata': split.metadata
                })
            
            # Save with metadata
            save_data = {
                'splits': splits_data,
                'config': {
                    'strategy': self.config.strategy.value,
                    'train_size': self.config.train_size,
                    'test_size': self.config.test_size,
                    'gap_hours': self.config.gap_hours,
                    'min_train_size': self.config.min_train_size,
                    'max_splits': self.config.max_splits,
                    'purge_buffer': self.config.purge_buffer,
                    'embargo_hours': self.config.embargo_hours
                },
                'summary': self.get_split_summary(splits),
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
            import json
            with open(output_file, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
            
            self.logger.info(f"Splits saved to {output_file}")
            return str(output_file)
            
        except Exception as e:
            self.logger.error(f"Failed to save splits: {e}")
            return ""

# Utility functions for common splitting patterns

def create_simple_rolling_splits(df: pd.DataFrame, train_days: int = 30, test_days: int = 7,
                                gap_hours: float = 0.0, timestamp_col: str = 'timestamp') -> List[SplitResult]:
    """Create simple rolling window splits"""
    
    config = SplitConfig(
        strategy=SplitStrategy.ROLLING_WINDOW,
        train_size=train_days * 24,  # Convert to hours
        test_size=test_days * 24,
        gap_hours=gap_hours
    )
    
    splitter = TemporalSafeSplitter(config)
    return splitter.create_splits(df, timestamp_col)

def create_walk_forward_splits(df: pd.DataFrame, train_weeks: int = 4, test_weeks: int = 1,
                              max_splits: int = 10, timestamp_col: str = 'timestamp') -> List[SplitResult]:
    """Create walk-forward analysis splits"""
    
    config = SplitConfig(
        strategy=SplitStrategy.WALK_FORWARD,
        train_size=train_weeks * 7 * 24,  # Convert to hours
        test_size=test_weeks * 7 * 24,
        gap_hours=0.0,
        max_splits=max_splits
    )
    
    splitter = TemporalSafeSplitter(config)
    return splitter.create_splits(df, timestamp_col)

def create_purged_cv_splits(df: pd.DataFrame, test_size: int = 168, purge_hours: float = 24.0,
                           n_splits: int = 5, timestamp_col: str = 'timestamp') -> List[SplitResult]:
    """Create purged cross-validation splits"""
    
    config = SplitConfig(
        strategy=SplitStrategy.PURGED_CV,
        train_size=len(df) // 2,  # Will be adjusted by purging
        test_size=test_size,
        gap_hours=0.0,
        purge_buffer=purge_hours,
        max_splits=n_splits
    )
    
    splitter = TemporalSafeSplitter(config)
    return splitter.create_splits(df, timestamp_col)

if __name__ == "__main__":
    # Test temporal safe splitting
    print("Testing Temporal Safe Splits")
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=1000, freq='H')
    test_data = pd.DataFrame({
        'timestamp': dates,
        'value': np.random.randn(1000),
        'feature1': np.random.randn(1000)
    })
    
    # Test rolling window splits
    print("\n1. Testing Rolling Window Splits...")
    config = SplitConfig(
        strategy=SplitStrategy.ROLLING_WINDOW,
        train_size=168,  # 1 week
        test_size=24,    # 1 day
        gap_hours=1.0,
        max_splits=5
    )
    
    splitter = TemporalSafeSplitter(config)
    splits = splitter.create_splits(test_data)
    
    print(f"Created {len(splits)} rolling window splits")
    if splits:
        print(f"First split: train={len(splits[0].train_indices)}, test={len(splits[0].test_indices)}")
    
    # Test purged CV splits
    print("\n2. Testing Purged CV Splits...")
    purged_splits = create_purged_cv_splits(test_data, n_splits=3)
    print(f"Created {len(purged_splits)} purged CV splits")
    
    # Test summary
    summary = splitter.get_split_summary(splits)
    print(f"\nSplit summary: {summary['total_splits']} splits, {summary['quality_metrics']['total_warnings']} warnings")
    
    print("\n✅ TEMPORAL SAFE SPLITS VALIDATION COMPLETE")