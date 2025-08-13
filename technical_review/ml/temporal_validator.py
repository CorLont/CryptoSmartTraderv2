#!/usr/bin/env python3
"""
Temporal Validation - Prevent look-ahead bias and feature leakage
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings


@dataclass
class ValidationResult:
    """Validation result container"""
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class TemporalValidator:
    """
    Temporal validation to prevent look-ahead bias and feature leakage
    
    Critical for financial time series:
    - No future information in features
    - Proper time-series cross-validation
    - Feature leakage detection
    - Data integrity checks
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_temporal_integrity(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        feature_cols: List[str] = None,
        target_col: str = 'target',
        max_lag_minutes: int = 5
    ) -> ValidationResult:
        """
        Validate temporal integrity of dataset
        
        Args:
            df: Dataset to validate
            timestamp_col: Timestamp column name
            feature_cols: Feature column names
            target_col: Target column name
            max_lag_minutes: Maximum allowed lag for features
            
        Returns:
            ValidationResult with pass/fail and details
        """
        
        try:
            # Ensure timestamp column exists
            if timestamp_col not in df.columns:
                return ValidationResult(
                    passed=False,
                    message=f"Timestamp column '{timestamp_col}' not found"
                )
            
            # Convert to datetime if needed
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
            # Sort by timestamp
            df_sorted = df.sort_values(timestamp_col)
            
            # Check for duplicate timestamps
            duplicates = df_sorted[timestamp_col].duplicated().sum()
            if duplicates > 0:
                self.logger.warning(f"Found {duplicates} duplicate timestamps")
            
            # Check for gaps in time series
            time_diffs = df_sorted[timestamp_col].diff().dropna()
            median_diff = time_diffs.median()
            large_gaps = (time_diffs > median_diff * 10).sum()
            
            if large_gaps > 0:
                self.logger.warning(f"Found {large_gaps} large time gaps")
            
            # Validate feature columns
            if feature_cols is None:
                feature_cols = [col for col in df.columns 
                               if col not in [timestamp_col, target_col]]
            
            # Check for future leakage
            leakage_results = self._check_feature_leakage(
                df_sorted, timestamp_col, feature_cols, max_lag_minutes
            )
            
            # Check for NaN values
            nan_counts = df[feature_cols + [target_col]].isnull().sum()
            high_nan_cols = nan_counts[nan_counts > len(df) * 0.1].index.tolist()
            
            # Compile results
            details = {
                "total_rows": len(df),
                "duplicate_timestamps": duplicates,
                "large_time_gaps": large_gaps,
                "median_time_diff_seconds": median_diff.total_seconds() if pd.notnull(median_diff) else 0,
                "feature_leakage_detected": leakage_results["has_leakage"],
                "leakage_details": leakage_results["details"],
                "high_nan_columns": high_nan_cols,
                "timestamp_range": {
                    "start": df_sorted[timestamp_col].min().isoformat(),
                    "end": df_sorted[timestamp_col].max().isoformat()
                }
            }
            
            # Determine overall pass/fail
            passed = (
                not leakage_results["has_leakage"] and
                duplicates == 0 and
                len(high_nan_cols) == 0
            )
            
            message = "Temporal validation passed" if passed else "Temporal validation failed"
            
            return ValidationResult(
                passed=passed,
                message=message,
                details=details
            )
            
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Validation error: {str(e)}",
                details={"error": str(e)}
            )
    
    def _check_feature_leakage(
        self,
        df: pd.DataFrame,
        timestamp_col: str,
        feature_cols: List[str],
        max_lag_minutes: int
    ) -> Dict[str, Any]:
        """
        Check for feature leakage by analyzing correlations with future values
        """
        
        leakage_details = {}
        has_leakage = False
        
        for feature in feature_cols:
            if feature not in df.columns:
                continue
            
            try:
                # Create shifted versions to check correlation with future
                df_temp = df[[timestamp_col, feature]].copy()
                
                # Shift feature forward (future values)
                for shift_minutes in [1, 5, 15, 30, 60]:
                    shifted_col = f"{feature}_future_{shift_minutes}m"
                    df_temp[shifted_col] = df_temp[feature].shift(-shift_minutes)
                
                # Calculate correlations
                correlations = df_temp.select_dtypes(include=[np.number]).corr()[feature]
                
                # Check for high correlation with future values
                future_corrs = {
                    col: correlations[col] for col in correlations.index
                    if col.startswith(f"{feature}_future_") and not pd.isna(correlations[col])
                }
                
                high_future_corr = any(abs(corr) > 0.95 for corr in future_corrs.values())
                
                if high_future_corr:
                    has_leakage = True
                    leakage_details[feature] = {
                        "future_correlations": future_corrs,
                        "max_future_correlation": max(future_corrs.values(), key=abs) if future_corrs else 0
                    }
                
            except Exception as e:
                self.logger.warning(f"Could not check leakage for {feature}: {e}")
        
        return {
            "has_leakage": has_leakage,
            "details": leakage_details
        }
    
    def create_time_series_splits(
        self,
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        n_splits: int = 5,
        test_size_days: int = 7,
        gap_days: int = 1
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time-series cross-validation splits
        
        Args:
            df: Dataset
            timestamp_col: Timestamp column
            n_splits: Number of splits
            test_size_days: Test set size in days
            gap_days: Gap between train and test
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        
        # Sort by timestamp
        df_sorted = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Convert to datetime
        if not pd.api.types.is_datetime64_any_dtype(df_sorted[timestamp_col]):
            df_sorted[timestamp_col] = pd.to_datetime(df_sorted[timestamp_col])
        
        splits = []
        total_days = (df_sorted[timestamp_col].max() - df_sorted[timestamp_col].min()).days
        
        # Calculate split points
        split_size_days = (total_days - test_size_days - gap_days) // n_splits
        
        for i in range(n_splits):
            # Calculate dates for this split
            test_start_days = split_size_days * (i + 1) + gap_days
            test_end_days = test_start_days + test_size_days
            
            start_date = df_sorted[timestamp_col].min()
            train_end_date = start_date + timedelta(days=split_size_days * (i + 1))
            test_start_date = start_date + timedelta(days=test_start_days)
            test_end_date = start_date + timedelta(days=test_end_days)
            
            # Get indices
            train_mask = df_sorted[timestamp_col] <= train_end_date
            test_mask = (
                (df_sorted[timestamp_col] >= test_start_date) &
                (df_sorted[timestamp_col] <= test_end_date)
            )
            
            train_indices = df_sorted[train_mask].index.values
            test_indices = df_sorted[test_mask].index.values
            
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))
        
        self.logger.info(f"Created {len(splits)} time-series splits")
        return splits
    
    def validate_feature_engineering(
        self,
        df: pd.DataFrame,
        feature_engineering_func,
        timestamp_col: str = 'timestamp'
    ) -> ValidationResult:
        """
        Validate that feature engineering doesn't introduce leakage
        
        Args:
            df: Original dataset
            feature_engineering_func: Function that applies feature engineering
            timestamp_col: Timestamp column
            
        Returns:
            ValidationResult
        """
        
        try:
            # Apply feature engineering
            df_engineered = feature_engineering_func(df.copy())
            
            # Get new feature columns
            original_cols = set(df.columns)
            new_cols = set(df_engineered.columns) - original_cols
            
            if not new_cols:
                return ValidationResult(
                    passed=True,
                    message="No new features created",
                    details={"new_features": []}
                )
            
            # Validate temporal integrity of engineered features
            validation_result = self.validate_temporal_integrity(
                df_engineered,
                timestamp_col=timestamp_col,
                feature_cols=list(new_cols)
            )
            
            validation_result.details["new_features"] = list(new_cols)
            return validation_result
            
        except Exception as e:
            return ValidationResult(
                passed=False,
                message=f"Feature engineering validation failed: {str(e)}",
                details={"error": str(e)}
            )
    
    def check_data_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: List[str],
        drift_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Check for data drift between reference and current datasets
        
        Args:
            reference_df: Reference dataset (training)
            current_df: Current dataset (inference)
            feature_cols: Features to check
            drift_threshold: Threshold for drift detection
            
        Returns:
            Drift analysis results
        """
        
        drift_results = {
            "has_drift": False,
            "drift_score": 0.0,
            "feature_drifts": {},
            "recommendation": "no_action"
        }
        
        try:
            total_drift_score = 0.0
            num_features = 0
            
            for feature in feature_cols:
                if feature not in reference_df.columns or feature not in current_df.columns:
                    continue
                
                # Remove NaN values
                ref_values = reference_df[feature].dropna()
                curr_values = current_df[feature].dropna()
                
                if len(ref_values) == 0 or len(curr_values) == 0:
                    continue
                
                # Calculate drift metrics
                if pd.api.types.is_numeric_dtype(ref_values):
                    # For numeric features: use KS test or statistical measures
                    ref_mean = ref_values.mean()
                    curr_mean = curr_values.mean()
                    ref_std = ref_values.std()
                    curr_std = curr_values.std()
                    
                    # Normalized difference in means
                    mean_drift = abs(curr_mean - ref_mean) / (ref_std + 1e-8)
                    # Ratio of standard deviations
                    std_drift = abs(curr_std - ref_std) / (ref_std + 1e-8)
                    
                    feature_drift = max(mean_drift, std_drift)
                    
                else:
                    # For categorical features: use value distribution differences
                    ref_dist = ref_values.value_counts(normalize=True)
                    curr_dist = curr_values.value_counts(normalize=True)
                    
                    # Calculate distribution difference
                    all_values = set(ref_dist.index) | set(curr_dist.index)
                    feature_drift = sum(
                        abs(ref_dist.get(val, 0) - curr_dist.get(val, 0))
                        for val in all_values
                    ) / 2
                
                drift_results["feature_drifts"][feature] = {
                    "drift_score": feature_drift,
                    "has_drift": feature_drift > drift_threshold
                }
                
                total_drift_score += feature_drift
                num_features += 1
            
            # Calculate overall drift score
            if num_features > 0:
                drift_results["drift_score"] = total_drift_score / num_features
                drift_results["has_drift"] = drift_results["drift_score"] > drift_threshold
                
                # Recommendations based on drift level
                if drift_results["drift_score"] > 0.3:
                    drift_results["recommendation"] = "retrain_immediately"
                elif drift_results["drift_score"] > 0.2:
                    drift_results["recommendation"] = "reduce_confidence"
                elif drift_results["drift_score"] > drift_threshold:
                    drift_results["recommendation"] = "monitor_closely"
                else:
                    drift_results["recommendation"] = "no_action"
            
            return drift_results
            
        except Exception as e:
            self.logger.error(f"Drift detection failed: {e}")
            return {
                "has_drift": True,  # Conservative: assume drift on error
                "drift_score": 1.0,
                "feature_drifts": {},
                "recommendation": "retrain_immediately",
                "error": str(e)
            }


# Global validator instance
temporal_validator = TemporalValidator()