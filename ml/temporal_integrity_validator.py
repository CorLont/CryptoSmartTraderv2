#!/usr/bin/env python3
"""
Temporal Integrity Validator
Prevents look-ahead bias and data leakage in time series ML models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TemporalViolation:
    """Detected temporal integrity violation"""
    violation_type: str  # 'look_ahead', 'future_leak', 'missing_shift', 'invalid_order'
    severity: str  # 'critical', 'warning', 'info'
    column_name: str
    description: str
    affected_rows: int
    recommendation: str

@dataclass
class TemporalValidationResult:
    """Result of temporal validation"""
    is_valid: bool
    violations: List[TemporalViolation]
    critical_count: int
    warning_count: int
    recommendations: List[str]

class TemporalIntegrityValidator:
    """Validates temporal integrity in time series data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Known temporal patterns that indicate violations
        self.violation_patterns = {
            'target_without_shift': r'^target_\d+[hdw]?$',
            'future_feature': r'^future_',
            'lookahead_feature': r'_(t\+\d+|next_|forward_)',
            'instant_feature': r'_(t0|same_period|concurrent)'
        }
    
    def validate_dataset(
        self, 
        df: pd.DataFrame,
        timestamp_col: str = 'timestamp',
        target_cols: List[str] = None,
        feature_cols: List[str] = None
    ) -> TemporalValidationResult:
        """Comprehensive temporal validation of dataset"""
        
        violations = []
        
        # Auto-detect target and feature columns if not provided
        if target_cols is None:
            target_cols = [col for col in df.columns if 'target_' in col.lower()]
        
        if feature_cols is None:
            feature_cols = [col for col in df.columns 
                          if col not in target_cols and col != timestamp_col]
        
        # Validation 1: Check timestamp ordering
        violations.extend(self._validate_timestamp_ordering(df, timestamp_col))
        
        # Validation 2: Check target calculation timing
        violations.extend(self._validate_target_timing(df, target_cols, timestamp_col))
        
        # Validation 3: Check feature-target temporal alignment
        violations.extend(self._validate_feature_target_alignment(
            df, feature_cols, target_cols, timestamp_col
        ))
        
        # Validation 4: Check for future information leakage
        violations.extend(self._detect_future_leakage(df, feature_cols, timestamp_col))
        
        # Validation 5: Check rolling window calculations
        violations.extend(self._validate_rolling_calculations(df, feature_cols))
        
        # Count violations by severity
        critical_count = sum(1 for v in violations if v.severity == 'critical')
        warning_count = sum(1 for v in violations if v.severity == 'warning')
        
        # Generate recommendations
        recommendations = self._generate_recommendations(violations)
        
        return TemporalValidationResult(
            is_valid=critical_count == 0,
            violations=violations,
            critical_count=critical_count,
            warning_count=warning_count,
            recommendations=recommendations
        )
    
    def _validate_timestamp_ordering(
        self, 
        df: pd.DataFrame, 
        timestamp_col: str
    ) -> List[TemporalViolation]:
        """Validate timestamp column ordering"""
        
        violations = []
        
        if timestamp_col not in df.columns:
            violations.append(TemporalViolation(
                violation_type='missing_timestamp',
                severity='critical',
                column_name=timestamp_col,
                description=f"Timestamp column '{timestamp_col}' not found",
                affected_rows=len(df),
                recommendation=f"Add timestamp column '{timestamp_col}' to dataset"
            ))
            return violations
        
        # Check for non-monotonic timestamps
        timestamps = pd.to_datetime(df[timestamp_col])
        non_monotonic = (timestamps.diff() < pd.Timedelta(0)).sum()
        
        if non_monotonic > 0:
            violations.append(TemporalViolation(
                violation_type='invalid_order',
                severity='critical',
                column_name=timestamp_col,
                description=f"Found {non_monotonic} non-monotonic timestamp entries",
                affected_rows=non_monotonic,
                recommendation="Sort dataset by timestamp before feature engineering"
            ))
        
        # Check for duplicate timestamps
        duplicates = timestamps.duplicated().sum()
        
        if duplicates > 0:
            violations.append(TemporalViolation(
                violation_type='duplicate_timestamps',
                severity='warning',
                column_name=timestamp_col,
                description=f"Found {duplicates} duplicate timestamps",
                affected_rows=duplicates,
                recommendation="Remove or aggregate duplicate timestamp entries"
            ))
        
        return violations
    
    def _validate_target_timing(
        self, 
        df: pd.DataFrame, 
        target_cols: List[str], 
        timestamp_col: str
    ) -> List[TemporalViolation]:
        """Validate that targets are properly time-shifted"""
        
        violations = []
        
        for target_col in target_cols:
            if target_col not in df.columns:
                continue
            
            # Check if target column appears to be calculated from same-period data
            violation = self._check_target_calculation(df, target_col, timestamp_col)
            if violation:
                violations.append(violation)
        
        return violations
    
    def _check_target_calculation(
        self, 
        df: pd.DataFrame, 
        target_col: str, 
        timestamp_col: str
    ) -> Optional[TemporalViolation]:
        """Check if target is calculated with proper time shift"""
        
        # Extract horizon from target name (e.g., target_1d, target_3h, target_7d)
        import re
        
        horizon_match = re.search(r'target_(\d+)([hdw]?)', target_col.lower())
        if not horizon_match:
            return TemporalViolation(
                violation_type='invalid_target_name',
                severity='warning',
                column_name=target_col,
                description=f"Target column '{target_col}' doesn't follow naming convention",
                affected_rows=len(df),
                recommendation="Use naming convention: target_Nh, target_Nd, target_Nw"
            )
        
        horizon_value = int(horizon_match.group(1))
        horizon_unit = horizon_match.group(2) or 'h'  # Default to hours
        
        # Convert to hours for calculation
        unit_multipliers = {'h': 1, 'd': 24, 'w': 168}
        horizon_hours = horizon_value * unit_multipliers[horizon_unit]
        
        # Check if target values are suspiciously correlated with current features
        # This is a heuristic check for look-ahead bias
        
        # Get price-related columns for correlation check
        price_cols = [col for col in df.columns 
                     if any(word in col.lower() for word in ['price', 'close', 'value'])]
        
        if price_cols and len(df) > 10:
            current_price = df[price_cols[0]] if price_cols else df.iloc[:, 1]
            target_values = df[target_col].dropna()
            
            if len(target_values) > 10:
                # Calculate correlation between current price and target
                correlation = current_price.corr(target_values)
                
                # High correlation suggests possible look-ahead bias
                if abs(correlation) > 0.95:
                    return TemporalViolation(
                        violation_type='look_ahead',
                        severity='critical',
                        column_name=target_col,
                        description=f"Suspiciously high correlation ({correlation:.3f}) between current features and {target_col}",
                        affected_rows=len(target_values),
                        recommendation=f"Ensure {target_col} is calculated using data from {horizon_hours}h in the future with proper shift(-{horizon_hours})"
                    )
        
        return None
    
    def _validate_feature_target_alignment(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str], 
        target_cols: List[str], 
        timestamp_col: str
    ) -> List[TemporalViolation]:
        """Validate temporal alignment between features and targets"""
        
        violations = []
        
        # Check for features that might contain future information
        suspicious_features = []
        
        for feature_col in feature_cols:
            # Check feature naming for future-leaking patterns
            feature_lower = feature_col.lower()
            
            if any(pattern in feature_lower for pattern in ['future', 'next', 'forward', 'ahead']):
                suspicious_features.append(feature_col)
            
            # Check for features calculated with forward-looking windows
            if 'shift(' in feature_col or 'lead(' in feature_col:
                violations.append(TemporalViolation(
                    violation_type='future_leak',
                    severity='critical',
                    column_name=feature_col,
                    description=f"Feature '{feature_col}' appears to use forward-looking calculation",
                    affected_rows=len(df),
                    recommendation=f"Replace forward shift with backward shift for {feature_col}"
                ))
        
        if suspicious_features:
            violations.append(TemporalViolation(
                violation_type='future_leak',
                severity='warning',
                column_name=', '.join(suspicious_features[:3]),
                description=f"Found {len(suspicious_features)} features with suspicious future-looking names",
                affected_rows=len(df),
                recommendation="Review feature calculation logic for temporal consistency"
            ))
        
        return violations
    
    def _detect_future_leakage(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str], 
        timestamp_col: str
    ) -> List[TemporalViolation]:
        """Detect potential future information leakage in features"""
        
        violations = []
        
        if len(df) < 5:
            return violations
        
        # Check for perfect prediction patterns (unrealistic accuracy indicators)
        for feature_col in feature_cols:
            if feature_col not in df.columns:
                continue
            
            feature_values = df[feature_col].dropna()
            
            if len(feature_values) < 5:
                continue
            
            # Check for suspiciously stable features (might be forward-filled)
            stability_ratio = (feature_values == feature_values.shift(1)).mean()
            
            if stability_ratio > 0.99:
                violations.append(TemporalViolation(
                    violation_type='future_leak',
                    severity='warning',
                    column_name=feature_col,
                    description=f"Feature '{feature_col}' shows {stability_ratio:.1%} stability (possible forward-fill)",
                    affected_rows=int(len(feature_values) * stability_ratio),
                    recommendation=f"Check if {feature_col} is improperly forward-filled or calculated"
                ))
            
            # Check for unrealistic precision in calculations
            if feature_values.dtype in ['float64', 'float32']:
                # Count decimal places
                decimal_places = feature_values.astype(str).str.split('.').str[1].str.len()
                avg_decimals = decimal_places.mean()
                
                if avg_decimals > 10:
                    violations.append(TemporalViolation(
                        violation_type='future_leak',
                        severity='info',
                        column_name=feature_col,
                        description=f"Feature '{feature_col}' has unusual precision ({avg_decimals:.1f} decimals)",
                        affected_rows=len(feature_values),
                        recommendation=f"Review calculation method for {feature_col} - high precision might indicate data leakage"
                    ))
        
        return violations
    
    def _validate_rolling_calculations(
        self, 
        df: pd.DataFrame, 
        feature_cols: List[str]
    ) -> List[TemporalViolation]:
        """Validate rolling window calculations for temporal consistency"""
        
        violations = []
        
        # Look for features that might be rolling calculations
        rolling_features = [col for col in feature_cols 
                          if any(word in col.lower() for word in 
                               ['rolling', 'ma', 'sma', 'ema', 'mean', 'std', 'var'])]
        
        for feature_col in rolling_features:
            if feature_col not in df.columns:
                continue
            
            feature_values = df[feature_col].dropna()
            
            if len(feature_values) < 10:
                continue
            
            # Check for NaN values at the beginning (expected for rolling calculations)
            initial_nans = df[feature_col].head(10).isna().sum()
            
            if initial_nans == 0 and 'rolling' in feature_col.lower():
                violations.append(TemporalViolation(
                    violation_type='missing_shift',
                    severity='warning',
                    column_name=feature_col,
                    description=f"Rolling feature '{feature_col}' has no initial NaN values",
                    affected_rows=len(df),
                    recommendation=f"Ensure {feature_col} uses proper rolling window with min_periods parameter"
                ))
        
        return violations
    
    def _generate_recommendations(self, violations: List[TemporalViolation]) -> List[str]:
        """Generate actionable recommendations based on violations"""
        
        recommendations = []
        
        # Critical violations
        critical_violations = [v for v in violations if v.severity == 'critical']
        
        if critical_violations:
            recommendations.append("CRITICAL: Fix temporal integrity violations before model training")
            
            look_ahead_violations = [v for v in critical_violations if v.violation_type == 'look_ahead']
            if look_ahead_violations:
                recommendations.append("• Recalculate targets using proper future shift: target = price.shift(-horizon)")
            
            future_leak_violations = [v for v in critical_violations if v.violation_type == 'future_leak']
            if future_leak_violations:
                recommendations.append("• Remove or fix features that use future information")
            
            order_violations = [v for v in critical_violations if v.violation_type == 'invalid_order']
            if order_violations:
                recommendations.append("• Sort dataset by timestamp before any feature engineering")
        
        # Warning violations
        warning_violations = [v for v in violations if v.severity == 'warning']
        
        if warning_violations:
            recommendations.append("Review and address warning-level violations:")
            
            for violation in warning_violations[:3]:  # Top 3 warnings
                recommendations.append(f"• {violation.recommendation}")
        
        # General recommendations
        if violations:
            recommendations.extend([
                "Implement temporal data integrity checks in your ML pipeline",
                "Use walk-forward validation to detect temporal leakage",
                "Consider using TemporalDataBuilder for safe feature engineering"
            ])
        
        return recommendations

class TemporalDataBuilder:
    """Builds temporally-safe datasets with proper time shifts"""
    
    def __init__(self, timestamp_col: str = 'timestamp'):
        self.timestamp_col = timestamp_col
        self.logger = logging.getLogger(__name__)
    
    def create_safe_targets(
        self, 
        df: pd.DataFrame, 
        price_col: str,
        horizons: List[str] = ['1h', '24h', '7d', '30d']
    ) -> pd.DataFrame:
        """Create targets with proper time shifts to prevent look-ahead bias"""
        
        result_df = df.copy()
        
        # Ensure data is sorted by timestamp
        result_df = result_df.sort_values(self.timestamp_col)
        
        for horizon in horizons:
            # Parse horizon
            horizon_value, horizon_unit = self._parse_horizon(horizon)
            
            # Calculate shift periods based on data frequency
            freq = self._detect_frequency(result_df[self.timestamp_col])
            shift_periods = self._calculate_shift_periods(horizon_value, horizon_unit, freq)
            
            # Create target with proper shift
            target_col = f'target_{horizon}'
            
            # Calculate return: (future_price - current_price) / current_price
            future_price = result_df[price_col].shift(-shift_periods)
            current_price = result_df[price_col]
            
            result_df[target_col] = (future_price - current_price) / current_price
            
            # Also create confidence column based on data availability
            conf_col = f'conf_{horizon}'
            result_df[conf_col] = (~future_price.isna()).astype(float)
            
            self.logger.info(f"Created {target_col} with {shift_periods} period shift")
        
        return result_df
    
    def create_safe_features(
        self, 
        df: pd.DataFrame, 
        price_col: str,
        volume_col: str = None
    ) -> pd.DataFrame:
        """Create features with proper temporal alignment"""
        
        result_df = df.copy()
        
        # Technical indicators (properly lagged)
        result_df['returns_1h'] = result_df[price_col].pct_change(1)
        result_df['returns_24h'] = result_df[price_col].pct_change(24)
        
        # Rolling features (with proper min_periods)
        result_df['sma_24h'] = result_df[price_col].rolling(24, min_periods=24).mean()
        result_df['volatility_24h'] = result_df['returns_1h'].rolling(24, min_periods=24).std()
        
        # RSI (14-period)
        result_df['rsi_14'] = self._calculate_rsi(result_df[price_col], 14)
        
        # Volume features (if volume column provided)
        if volume_col and volume_col in df.columns:
            result_df['volume_sma_24h'] = result_df[volume_col].rolling(24, min_periods=24).mean()
            result_df['volume_ratio'] = result_df[volume_col] / result_df['volume_sma_24h']
        
        # Lag features to prevent any potential leakage
        feature_cols = [col for col in result_df.columns 
                       if col not in [self.timestamp_col, price_col, volume_col]]
        
        for col in feature_cols:
            if col.startswith(('target_', 'conf_')):
                continue
            
            # Lag all features by 1 period to ensure no same-period contamination
            result_df[col] = result_df[col].shift(1)
        
        return result_df
    
    def _parse_horizon(self, horizon: str) -> Tuple[int, str]:
        """Parse horizon string like '1h', '24h', '7d' into value and unit"""
        
        import re
        match = re.match(r'(\d+)([hdw])', horizon.lower())
        
        if not match:
            raise ValueError(f"Invalid horizon format: {horizon}. Use format like '1h', '24h', '7d'")
        
        return int(match.group(1)), match.group(2)
    
    def _detect_frequency(self, timestamps: pd.Series) -> str:
        """Detect data frequency from timestamps"""
        
        timestamps = pd.to_datetime(timestamps)
        time_diffs = timestamps.diff().dropna()
        
        if len(time_diffs) == 0:
            return 'H'  # Default to hourly
        
        median_diff = time_diffs.median()
        
        if median_diff <= pd.Timedelta(minutes=5):
            return 'T'  # Minute
        elif median_diff <= pd.Timedelta(hours=1):
            return 'H'  # Hourly
        elif median_diff <= pd.Timedelta(days=1):
            return 'D'  # Daily
        else:
            return 'W'  # Weekly
    
    def _calculate_shift_periods(self, horizon_value: int, horizon_unit: str, freq: str) -> int:
        """Calculate number of periods to shift based on horizon and data frequency"""
        
        # Convert horizon to minutes
        unit_to_minutes = {'h': 60, 'd': 1440, 'w': 10080}
        horizon_minutes = horizon_value * unit_to_minutes[horizon_unit]
        
        # Convert frequency to minutes
        freq_to_minutes = {'T': 1, 'H': 60, 'D': 1440, 'W': 10080}
        freq_minutes = freq_to_minutes.get(freq, 60)
        
        # Calculate shift periods
        shift_periods = horizon_minutes // freq_minutes
        
        return max(1, shift_periods)
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI with proper temporal alignment"""
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

def validate_ml_dataset(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    price_col: str = 'price',
    target_cols: List[str] = None,
    fix_violations: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive validation of ML dataset for temporal integrity
    
    Args:
        df: Dataset to validate
        timestamp_col: Name of timestamp column
        price_col: Name of price column
        target_cols: List of target column names
        fix_violations: Whether to attempt automatic fixes
    
    Returns:
        Dictionary with validation results and optionally fixed dataset
    """
    
    validator = TemporalIntegrityValidator()
    
    # Run validation
    validation_result = validator.validate_dataset(
        df, timestamp_col, target_cols
    )
    
    result = {
        'is_valid': validation_result.is_valid,
        'critical_violations': validation_result.critical_count,
        'warning_violations': validation_result.warning_count,
        'violations': [
            {
                'type': v.violation_type,
                'severity': v.severity,
                'column': v.column_name,
                'description': v.description,
                'recommendation': v.recommendation
            }
            for v in validation_result.violations
        ],
        'recommendations': validation_result.recommendations
    }
    
    # Attempt fixes if requested
    if fix_violations and not validation_result.is_valid:
        try:
            builder = TemporalDataBuilder(timestamp_col)
            
            # Fix targets if price column is available
            if price_col in df.columns:
                fixed_df = builder.create_safe_targets(df, price_col)
                fixed_df = builder.create_safe_features(fixed_df, price_col)
                
                # Re-validate fixed dataset
                fixed_validation = validator.validate_dataset(fixed_df, timestamp_col)
                
                result['fixed_dataset'] = fixed_df
                result['fixed_is_valid'] = fixed_validation.is_valid
                result['fixes_applied'] = True
            else:
                result['fixes_applied'] = False
                result['fix_error'] = f"Price column '{price_col}' not found for automatic fixes"
                
        except Exception as e:
            result['fixes_applied'] = False
            result['fix_error'] = str(e)
    
    return result