#!/usr/bin/env python3
"""
Temporal Integrity Validator - Enterprise-grade temporal validation and alignment
Fixes: datatype assumptions, UTC coverage, future-check bypass, inefficient alignment
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
from pathlib import Path
import json

from ..core.consolidated_logging_manager import get_consolidated_logger


class TemporalIntegrityValidator:
    """Enterprise temporal validator with comprehensive integrity checks and efficient alignment"""

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.logger = get_consolidated_logger("TemporalIntegrityValidator")

        # Validation statistics
        self.validation_count = 0
        self.violation_count = 0
        self.alignment_count = 0

        # Performance tracking
        self.last_validation_time = None
        self.average_validation_time = 0

        self.logger.info("Temporal Integrity Validator initialized")

    def validate_temporal_integrity(
        self, df: pd.DataFrame, agent_name: str = "Unknown", timestamp_col: str = "timestamp"
    ) -> Dict[str, Any]:
        """
        Comprehensive temporal integrity validation with early datatype normalization

        Args:
            df: DataFrame to validate
            agent_name: Name of the agent/source for logging
            timestamp_col: Name of the timestamp column

        Returns:
            Dict with validation results and statistics
        """
        start_time = pd.Timestamp.utcnow()
        violations = []
        warnings_list = []

        try:
            if df.empty:
                return self._create_validation_result(
                    agent_name, True, [], ["Empty DataFrame provided"], 0
                )

            if timestamp_col not in df.columns:
                violations.append(f"{agent_name}: timestamp column '{timestamp_col}' not found")
                return self._create_validation_result(
                    agent_name, False, violations, warnings_list, 0
                )

            # Early datatype normalization - CRITICAL FIX
            df = df.copy()
            original_dtype = df[timestamp_col].dtype

            try:
                # Force conversion to UTC datetime with error handling
                df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")
                timestamps = df[timestamp_col]

                # Check for conversion failures
                if timestamps.isna().any():
                    invalid_count = timestamps.isna().sum()
                    violations.append(
                        f"{agent_name}: {invalid_count} invalid timestamp values after conversion"
                    )

                    if self.strict_mode:
                        return self._create_validation_result(
                            agent_name, False, violations, warnings_list, 0
                        )
                    else:
                        # Drop invalid timestamps in non-strict mode
                        df = df.dropna(subset=[timestamp_col])
                        timestamps = df[timestamp_col]
                        warnings_list.append(f"Dropped {invalid_count} invalid timestamps")

            except Exception as e:
                violations.append(f"{agent_name}: timestamp conversion failed - {str(e)}")
                return self._create_validation_result(
                    agent_name, False, violations, warnings_list, 0
                )

            # Comprehensive temporal validations
            validation_results = []

            # 1. Monotonicity check (now safe with proper datetime)
            monotonic_result = self._check_monotonicity(timestamps, agent_name)
            validation_results.append(monotonic_result)
            if not monotonic_result["passed"]:
                violations.extend(monotonic_result["violations"])

            # 2. UTC alignment check (comprehensive, not just first 5)
            utc_result = self._check_utc_alignment(timestamps, agent_name)
            validation_results.append(utc_result)
            if not utc_result["passed"]:
                violations.extend(utc_result["violations"])

            # 3. Future timestamp check (vectorized, no bypass)
            future_result = self._check_future_timestamps(timestamps, agent_name)
            validation_results.append(future_result)
            if not future_result["passed"]:
                violations.extend(future_result["violations"])

            # 4. Duplicate timestamp check
            duplicate_result = self._check_duplicate_timestamps(timestamps, agent_name)
            validation_results.append(duplicate_result)
            if not duplicate_result["passed"]:
                violations.extend(duplicate_result["violations"])

            # 5. Temporal gaps analysis
            gaps_result = self._analyze_temporal_gaps(timestamps, agent_name)
            validation_results.append(gaps_result)
            if gaps_result["warnings"]:
                warnings_list.extend(gaps_result["warnings"])

            # Update statistics
            self.validation_count += 1
            if violations:
                self.violation_count += 1

            # Performance tracking
            end_time = pd.Timestamp.utcnow()
            validation_duration = (end_time - start_time).total_seconds()
            self._update_performance_stats(validation_duration)

            is_valid = len(violations) == 0

            result = self._create_validation_result(
                agent_name, is_valid, violations, warnings_list, len(df)
            )

            # Add detailed validation results
            result["validation_details"] = validation_results
            result["original_dtype"] = str(original_dtype)
            result["converted_dtype"] = str(timestamps.dtype)
            result["validation_duration"] = validation_duration

            # Log results
            if is_valid:
                self.logger.info(f"Temporal validation passed for {agent_name}: {len(df)} records")
            else:
                self.logger.warning(
                    f"Temporal validation failed for {agent_name}: {len(violations)} violations"
                )

            return result

        except Exception as e:
            self.logger.error(f"Temporal validation error for {agent_name}: {e}")
            return self._create_validation_result(
                agent_name, False, [f"Validation error: {str(e)}"], [], 0
            )

    def _check_monotonicity(self, timestamps: pd.Series, agent_name: str) -> Dict[str, Any]:
        """Check if timestamps are monotonically increasing"""

        try:
            if len(timestamps) < 2:
                return {
                    "check": "monotonicity",
                    "passed": True,
                    "violations": [],
                    "details": "Insufficient data for monotonicity check",
                }

            # Safe monotonicity check with proper datetime
            is_monotonic = timestamps.is_monotonic_increasing

            if not is_monotonic:
                # Find specific violations
                diffs = timestamps.diff()
                negative_diffs = diffs[diffs < pd.Timedelta(0)]

                return {
                    "check": "monotonicity",
                    "passed": False,
                    "violations": [f"{agent_name}: {len(negative_diffs)} non-monotonic timestamps"],
                    "details": f"First violation at index {negative_diffs.index[0] if len(negative_diffs) > 0 else 'unknown'}",
                }

            return {
                "check": "monotonicity",
                "passed": True,
                "violations": [],
                "details": "All timestamps are monotonically increasing",
            }

        except Exception as e:
            return {
                "check": "monotonicity",
                "passed": False,
                "violations": [f"{agent_name}: monotonicity check failed - {str(e)}"],
                "details": str(e),
            }

    def _check_utc_alignment(self, timestamps: pd.Series, agent_name: str) -> Dict[str, Any]:
        """Comprehensive UTC alignment check (not just first 5 rows)"""

        try:
            # Check all timestamps for proper UTC timezone
            non_utc_count = 0

            for ts in timestamps:
                if pd.notna(ts):
                    if ts.tz is None:
                        non_utc_count += 1
                    elif ts.tz != timezone.utc:
                        non_utc_count += 1

            if non_utc_count > 0:
                return {
                    "check": "utc_alignment",
                    "passed": False,
                    "violations": [
                        f"{agent_name}: {non_utc_count} timestamps not properly UTC-aligned"
                    ],
                    "details": f"Found {non_utc_count}/{len(timestamps)} non-UTC timestamps",
                }

            return {
                "check": "utc_alignment",
                "passed": True,
                "violations": [],
                "details": "All timestamps are properly UTC-aligned",
            }

        except Exception as e:
            return {
                "check": "utc_alignment",
                "passed": False,
                "violations": [f"{agent_name}: UTC alignment check failed - {str(e)}"],
                "details": str(e),
            }

    def _check_future_timestamps(self, timestamps: pd.Series, agent_name: str) -> Dict[str, Any]:
        """Vectorized future timestamp check - no bypass possible"""

        try:
            # Vectorized future check - CRITICAL FIX
            current_utc = pd.Timestamp.utcnow()
            future_mask = timestamps > current_utc
            future_count = future_mask.sum()

            if future_count > 0:
                # Get some examples of future timestamps
                future_timestamps = timestamps[future_mask].head(3)
                future_examples = [ts.isoformat() for ts in future_timestamps]

                return {
                    "check": "future_timestamps",
                    "passed": False,
                    "violations": [f"{agent_name}: {future_count} future timestamps detected"],
                    "details": f"Examples: {future_examples}",
                }

            return {
                "check": "future_timestamps",
                "passed": True,
                "violations": [],
                "details": "No future timestamps detected",
            }

        except Exception as e:
            return {
                "check": "future_timestamps",
                "passed": False,
                "violations": [f"{agent_name}: future timestamp check failed - {str(e)}"],
                "details": str(e),
            }

    def _check_duplicate_timestamps(self, timestamps: pd.Series, agent_name: str) -> Dict[str, Any]:
        """Check for duplicate timestamps"""

        try:
            duplicate_count = timestamps.duplicated().sum()

            if duplicate_count > 0:
                # Get examples of duplicates
                duplicates = timestamps[timestamps.duplicated(keep=False)]
                duplicate_values = duplicates.value_counts().head(3)

                return {
                    "check": "duplicate_timestamps",
                    "passed": False,
                    "violations": [f"{agent_name}: {duplicate_count} duplicate timestamps"],
                    "details": f"Most common duplicates: {duplicate_values.to_dict()}",
                }

            return {
                "check": "duplicate_timestamps",
                "passed": True,
                "violations": [],
                "details": "No duplicate timestamps found",
            }

        except Exception as e:
            return {
                "check": "duplicate_timestamps",
                "passed": False,
                "violations": [f"{agent_name}: duplicate check failed - {str(e)}"],
                "details": str(e),
            }

    def _analyze_temporal_gaps(self, timestamps: pd.Series, agent_name: str) -> Dict[str, Any]:
        """Analyze temporal gaps and irregular intervals"""

        try:
            if len(timestamps) < 2:
                return {
                    "check": "temporal_gaps",
                    "passed": True,
                    "warnings": [],
                    "details": "Insufficient data for gap analysis",
                }

            # Calculate intervals
            intervals = timestamps.diff().dropna()

            # Basic statistics
            median_interval = intervals.median()
            mean_interval = intervals.mean()
            std_interval = intervals.std()

            # Identify unusual gaps (more than 3 standard deviations from mean)
            threshold = mean_interval + 3 * std_interval
            large_gaps = intervals[intervals > threshold]

            warnings_list = []
            if len(large_gaps) > 0:
                warnings_list.append(
                    f"{agent_name}: {len(large_gaps)} unusually large temporal gaps detected"
                )

            # Check for very irregular intervals
            if std_interval > mean_interval:
                warnings_list.append(f"{agent_name}: highly irregular time intervals (std > mean)")

            return {
                "check": "temporal_gaps",
                "passed": True,  # Gaps are warnings, not violations
                "warnings": warnings_list,
                "details": {
                    "median_interval": str(median_interval),
                    "mean_interval": str(mean_interval),
                    "std_interval": str(std_interval),
                    "large_gaps_count": len(large_gaps),
                },
            }

        except Exception as e:
            return {
                "check": "temporal_gaps",
                "passed": False,
                "warnings": [f"{agent_name}: gap analysis failed - {str(e)}"],
                "details": str(e),
            }

    def align_timestamps_efficient(
        self, df: pd.DataFrame, target_freq: str = "H", timestamp_col: str = "timestamp"
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Efficient timestamp alignment using vectorized operations instead of iterrows()

        Args:
            df: DataFrame to align
            target_freq: Target frequency ('H', '15T', '1D', etc.)
            timestamp_col: Timestamp column name

        Returns:
            Tuple of (aligned_df, alignment_report)
        """

        start_time = pd.Timestamp.utcnow()

        try:
            if df.empty:
                return df, {"status": "empty", "aligned_count": 0}

            # Early normalization
            df = df.copy()
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True, errors="coerce")

            # Remove invalid timestamps
            original_count = len(df)
            df = df.dropna(subset=[timestamp_col])
            valid_count = len(df)

            if df.empty:
                return df, {
                    "status": "no_valid_timestamps",
                    "original_count": original_count,
                    "aligned_count": 0,
                }

            # Efficient alignment using pandas resampling - CRITICAL PERFORMANCE FIX
            df_indexed = df.set_index(timestamp_col)

            # Create target time range
            start_time_data = df_indexed.index.min()
            end_time_data = df_indexed.index.max()

            # Generate target frequency range
            target_range = pd.date_range(
                start=start_time_data.floor(target_freq),
                end=end_time_data.ceil(target_freq),
                freq=target_freq,
                tz=timezone.utc,
            )

            # Vectorized alignment using reindex with forward fill
            aligned_df = df_indexed.reindex(target_range, method="ffill")

            # Reset index to get timestamp column back
            aligned_df = aligned_df.reset_index()
            aligned_df = aligned_df.rename(columns={"index": timestamp_col})

            # Update statistics
            self.alignment_count += 1

            # Performance tracking
            end_time_proc = pd.Timestamp.utcnow()
            alignment_duration = (end_time_proc - start_time).total_seconds()

            alignment_report = {
                "status": "success",
                "original_count": original_count,
                "valid_count": valid_count,
                "aligned_count": len(aligned_df),
                "target_frequency": target_freq,
                "alignment_duration": alignment_duration,
                "start_time": start_time_data.isoformat(),
                "end_time": end_time_data.isoformat(),
                "method": "vectorized_reindex",
            }

            self.logger.info(
                f"Efficient alignment completed: {original_count} → {len(aligned_df)} records"
            )

            return aligned_df, alignment_report

        except Exception as e:
            self.logger.error(f"Timestamp alignment failed: {e}")
            return df, {
                "status": "error",
                "error": str(e),
                "original_count": len(df) if "df" in locals() else 0,
                "aligned_count": 0,
            }

    def _create_validation_result(
        self,
        agent_name: str,
        is_valid: bool,
        violations: List[str],
        warnings: List[str],
        record_count: int,
    ) -> Dict[str, Any]:
        """Create standardized validation result"""

        return {
            "agent_name": agent_name,
            "is_valid": is_valid,
            "violations": violations,
            "warnings": warnings,
            "record_count": record_count,
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "validator_stats": {
                "total_validations": self.validation_count,
                "total_violations": self.violation_count,
                "total_alignments": self.alignment_count,
            },
        }

    def _update_performance_stats(self, duration: float):
        """Update performance statistics"""

        if self.average_validation_time == 0:
            self.average_validation_time = duration
        else:
            # Running average
            self.average_validation_time = (self.average_validation_time * 0.9) + (duration * 0.1)

        self.last_validation_time = pd.Timestamp.utcnow()

    def get_validator_status(self) -> Dict[str, Any]:
        """Get validator status and statistics"""

        return {
            "strict_mode": self.strict_mode,
            "validation_count": self.validation_count,
            "violation_count": self.violation_count,
            "alignment_count": self.alignment_count,
            "violation_rate": self.violation_count / max(self.validation_count, 1),
            "average_validation_time": self.average_validation_time,
            "last_validation": self.last_validation_time.isoformat()
            if self.last_validation_time
            else None,
        }

    def save_validation_report(
        self, validation_result: Dict[str, Any], output_dir: str = "logs/temporal_validation"
    ) -> str:
        """Save detailed validation report to file"""

        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            timestamp_str = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
            agent_name = validation_result.get("agent_name", "unknown")
            filename = f"temporal_validation_{agent_name}_{timestamp_str}.json"

            file_path = output_path / filename

            with open(file_path, "w") as f:
                json.dump(validation_result, f, indent=2, default=str)

            self.logger.info(f"Validation report saved: {file_path}")
            return str(file_path)

        except Exception as e:
            self.logger.error(f"Failed to save validation report: {e}")
            return ""


# Global validator instance
_temporal_validator: Optional[TemporalIntegrityValidator] = None


def get_temporal_validator(strict_mode: bool = True) -> TemporalIntegrityValidator:
    """Get global temporal validator instance"""
    global _temporal_validator

    if _temporal_validator is None:
        _temporal_validator = TemporalIntegrityValidator(strict_mode)

    return _temporal_validator


def validate_temporal_data(
    df: pd.DataFrame, agent_name: str = "Unknown", timestamp_col: str = "timestamp"
) -> Dict[str, Any]:
    """Convenience function for temporal validation"""
    validator = get_temporal_validator()
    return validator.validate_temporal_integrity(df, agent_name, timestamp_col)


def align_temporal_data(
    df: pd.DataFrame, target_freq: str = "H", timestamp_col: str = "timestamp"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Convenience function for temporal alignment"""
    validator = get_temporal_validator()
    return validator.align_timestamps_efficient(df, target_freq, timestamp_col)


if __name__ == "__main__":
    # Test temporal integrity validator
    print("Testing Temporal Integrity Validator")

    # Create test data with various temporal issues
    test_data = pd.DataFrame(
        {
            "timestamp": [
                "2024-01-01 10:00:00",
                "2024-01-01 11:00:00",
                "2024-01-01 09:30:00",  # Non-monotonic
                "2024-01-01 12:00:00",
                "2024-01-01 12:00:00",  # Duplicate
                "2025-01-01 10:00:00",  # Future
            ],
            "value": [100, 101, 99, 102, 102, 200],
        }
    )

    validator = TemporalIntegrityValidator(strict_mode=False)

    print("Test data:")
    print(test_data)

    # Test validation
    result = validator.validate_temporal_integrity(test_data, "test_agent")
    print(f"\nValidation result:")
    print(f"Valid: {result['is_valid']}")
    print(f"Violations: {result['violations']}")
    print(f"Warnings: {result['warnings']}")

    # Test alignment
    aligned_df, alignment_report = validator.align_timestamps_efficient(test_data, "1H")
    print(f"\nAlignment report:")
    print(f"Status: {alignment_report['status']}")
    print(
        f"Original: {alignment_report.get('original_count', 0)} → Aligned: {alignment_report.get('aligned_count', 0)}"
    )

    # Show status
    status = validator.get_validator_status()
    print(f"\nValidator status: {status}")
