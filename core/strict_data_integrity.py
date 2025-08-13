#!/usr/bin/env python3
"""
Strict Data Integrity System
Zero-tolerance enforcement for synthetic/fallback data in production
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class DataSource(Enum):
    """Data source classification"""
    AUTHENTIC = "authentic"         # Real data from API/exchange
    SYNTHETIC = "synthetic"         # Generated/simulated data
    FALLBACK = "fallback"          # Default values when API fails
    INTERPOLATED = "interpolated"   # Filled missing values
    UNKNOWN = "unknown"            # Source not verified

@dataclass
class DataIntegrityViolation:
    """Data integrity violation record"""
    violation_type: str  # 'synthetic_data', 'fallback_used', 'nan_values', 'interpolated_data'
    severity: str  # 'critical', 'warning', 'info'
    column_name: str
    row_indices: List[int]
    violation_count: int
    data_source: DataSource
    description: str
    recommended_action: str

@dataclass
class DataIntegrityReport:
    """Comprehensive data integrity report"""
    is_production_ready: bool
    violations: List[DataIntegrityViolation]
    critical_violations: int
    warning_violations: int
    authentic_data_percentage: float
    recommended_actions: List[str]
    safe_for_training: bool
    safe_for_prediction: bool

class StrictDataIntegrityEnforcer:
    """Enforces zero-tolerance data integrity for production systems"""
    
    def __init__(self, production_mode: bool = True):
        self.production_mode = production_mode
        self.logger = logging.getLogger(__name__)
        
        # Zero-tolerance thresholds for production
        self.production_thresholds = {
            'max_nan_percentage': 0.0,      # 0% NaN values allowed
            'max_synthetic_percentage': 0.0, # 0% synthetic data allowed
            'max_fallback_percentage': 0.0,  # 0% fallback data allowed
            'min_authentic_percentage': 100.0, # 100% authentic data required
        }
        
        # More lenient thresholds for development/testing
        self.development_thresholds = {
            'max_nan_percentage': 5.0,
            'max_synthetic_percentage': 10.0,
            'max_fallback_percentage': 5.0,
            'min_authentic_percentage': 85.0,
        }
        
        self.current_thresholds = self.production_thresholds if production_mode else self.development_thresholds
    
    def validate_data_integrity(
        self, 
        df: pd.DataFrame,
        data_sources: Dict[str, DataSource] = None,
        column_metadata: Dict[str, Dict] = None
    ) -> DataIntegrityReport:
        """Comprehensive data integrity validation"""
        
        violations = []
        
        # Initialize data sources if not provided
        if data_sources is None:
            data_sources = {col: DataSource.UNKNOWN for col in df.columns}
        
        # Validation 1: Check for NaN values
        violations.extend(self._check_nan_values(df))
        
        # Validation 2: Check for synthetic data markers
        violations.extend(self._check_synthetic_data(df, data_sources))
        
        # Validation 3: Check for fallback data patterns
        violations.extend(self._check_fallback_patterns(df, data_sources))
        
        # Validation 4: Check for interpolated data
        violations.extend(self._check_interpolated_data(df, column_metadata))
        
        # Validation 5: Check data source authenticity
        violations.extend(self._check_data_source_authenticity(df, data_sources))
        
        # Validation 6: Check for unrealistic data patterns
        violations.extend(self._check_unrealistic_patterns(df))
        
        # Calculate metrics
        critical_violations = sum(1 for v in violations if v.severity == 'critical')
        warning_violations = sum(1 for v in violations if v.severity == 'warning')
        
        authentic_percentage = self._calculate_authentic_percentage(df, data_sources)
        
        # Determine production readiness
        is_production_ready = (
            critical_violations == 0 and
            authentic_percentage >= self.current_thresholds['min_authentic_percentage']
        )
        
        safe_for_training = is_production_ready
        safe_for_prediction = is_production_ready
        
        # Generate recommendations
        recommendations = self._generate_integrity_recommendations(violations, authentic_percentage)
        
        return DataIntegrityReport(
            is_production_ready=is_production_ready,
            violations=violations,
            critical_violations=critical_violations,
            warning_violations=warning_violations,
            authentic_data_percentage=authentic_percentage,
            recommended_actions=recommendations,
            safe_for_training=safe_for_training,
            safe_for_prediction=safe_for_prediction
        )
    
    def _check_nan_values(self, df: pd.DataFrame) -> List[DataIntegrityViolation]:
        """Check for NaN values in critical columns"""
        
        violations = []
        
        for column in df.columns:
            nan_mask = df[column].isna()
            nan_count = nan_mask.sum()
            
            if nan_count > 0:
                nan_percentage = (nan_count / len(df)) * 100
                severity = 'critical' if self.production_mode else 'warning'
                
                if nan_percentage > self.current_thresholds['max_nan_percentage']:
                    violations.append(DataIntegrityViolation(
                        violation_type='nan_values',
                        severity=severity,
                        column_name=column,
                        row_indices=df.index[nan_mask].tolist()[:100],  # First 100 indices
                        violation_count=nan_count,
                        data_source=DataSource.UNKNOWN,
                        description=f"Column '{column}' contains {nan_count} NaN values ({nan_percentage:.2f}%)",
                        recommended_action=f"Remove rows with NaN in '{column}' or exclude column from production"
                    ))
        
        return violations
    
    def _check_synthetic_data(self, df: pd.DataFrame, data_sources: Dict[str, DataSource]) -> List[DataIntegrityViolation]:
        """Check for synthetic data markers"""
        
        violations = []
        
        # Check data source markers
        for column, source in data_sources.items():
            if source == DataSource.SYNTHETIC:
                violations.append(DataIntegrityViolation(
                    violation_type='synthetic_data',
                    severity='critical',
                    column_name=column,
                    row_indices=list(range(len(df))),
                    violation_count=len(df),
                    data_source=source,
                    description=f"Column '{column}' is marked as synthetic data",
                    recommended_action=f"Replace synthetic data in '{column}' with authentic API data"
                ))
        
        # Check for synthetic data patterns
        for column in df.columns:
            if column in df.columns and df[column].dtype in ['float64', 'int64']:
                # Pattern 1: Perfect sequences (likely synthetic)
                if len(df) > 5:
                    values = df[column].dropna()
                    if len(values) > 5:
                        # Check for perfect arithmetic sequences
                        diffs = values.diff().dropna()
                        if len(diffs) > 3 and diffs.std() < 1e-10:
                            violations.append(DataIntegrityViolation(
                                violation_type='synthetic_data',
                                severity='warning',
                                column_name=column,
                                row_indices=[],
                                violation_count=len(values),
                                data_source=DataSource.SYNTHETIC,
                                description=f"Column '{column}' shows perfect arithmetic sequence (likely synthetic)",
                                recommended_action=f"Verify data source for '{column}' - replace if synthetic"
                            ))
                
                # Pattern 2: Suspiciously round numbers
                if df[column].dtype == 'float64':
                    values = df[column].dropna()
                    if len(values) > 10:
                        # Check for too many round numbers
                        round_values = values[values == values.round()]
                        round_percentage = len(round_values) / len(values) * 100
                        
                        if round_percentage > 80:  # More than 80% round numbers
                            violations.append(DataIntegrityViolation(
                                violation_type='synthetic_data',
                                severity='warning',
                                column_name=column,
                                row_indices=[],
                                violation_count=len(round_values),
                                data_source=DataSource.SYNTHETIC,
                                description=f"Column '{column}' has {round_percentage:.1f}% round numbers (possibly synthetic)",
                                recommended_action=f"Verify authenticity of '{column}' data"
                            ))
        
        return violations
    
    def _check_fallback_patterns(self, df: pd.DataFrame, data_sources: Dict[str, DataSource]) -> List[DataIntegrityViolation]:
        """Check for fallback data patterns"""
        
        violations = []
        
        # Check data source markers
        for column, source in data_sources.items():
            if source == DataSource.FALLBACK:
                violations.append(DataIntegrityViolation(
                    violation_type='fallback_used',
                    severity='critical',
                    column_name=column,
                    row_indices=list(range(len(df))),
                    violation_count=len(df),
                    data_source=source,
                    description=f"Column '{column}' uses fallback data",
                    recommended_action=f"Obtain authentic data for '{column}' or exclude from production"
                ))
        
        # Check for common fallback patterns
        for column in df.columns:
            if column in df.columns:
                values = df[column].dropna()
                
                if len(values) > 10:
                    # Pattern 1: Too many identical values (common fallback pattern)
                    value_counts = values.value_counts()
                    most_common_count = value_counts.iloc[0] if len(value_counts) > 0 else 0
                    most_common_percentage = most_common_count / len(values) * 100
                    
                    if most_common_percentage > 50:  # More than 50% identical values
                        violations.append(DataIntegrityViolation(
                            violation_type='fallback_used',
                            severity='warning',
                            column_name=column,
                            row_indices=[],
                            violation_count=most_common_count,
                            data_source=DataSource.FALLBACK,
                            description=f"Column '{column}' has {most_common_percentage:.1f}% identical values (possible fallback)",
                            recommended_action=f"Verify data diversity for '{column}'"
                        ))
                    
                    # Pattern 2: Common fallback values
                    common_fallbacks = [0, 1, -1, 999, 9999, 0.5]
                    for fallback_val in common_fallbacks:
                        if fallback_val in values.values:
                            fallback_count = (values == fallback_val).sum()
                            fallback_percentage = fallback_count / len(values) * 100
                            
                            if fallback_percentage > 30:  # More than 30% fallback values
                                violations.append(DataIntegrityViolation(
                                    violation_type='fallback_used',
                                    severity='warning',
                                    column_name=column,
                                    row_indices=[],
                                    violation_count=fallback_count,
                                    data_source=DataSource.FALLBACK,
                                    description=f"Column '{column}' has {fallback_percentage:.1f}% fallback values ({fallback_val})",
                                    recommended_action=f"Replace fallback values in '{column}' with authentic data"
                                ))
        
        return violations
    
    def _check_interpolated_data(self, df: pd.DataFrame, column_metadata: Dict[str, Dict] = None) -> List[DataIntegrityViolation]:
        """Check for interpolated/forward-filled data"""
        
        violations = []
        
        if column_metadata is None:
            column_metadata = {}
        
        for column in df.columns:
            # Check metadata for interpolation markers
            col_meta = column_metadata.get(column, {})
            if col_meta.get('is_interpolated', False) or col_meta.get('is_forward_filled', False):
                violations.append(DataIntegrityViolation(
                    violation_type='interpolated_data',
                    severity='critical' if self.production_mode else 'warning',
                    column_name=column,
                    row_indices=[],
                    violation_count=len(df),
                    data_source=DataSource.INTERPOLATED,
                    description=f"Column '{column}' contains interpolated data",
                    recommended_action=f"Remove interpolated data from '{column}' or exclude from production"
                ))
            
            # Check for interpolation patterns
            if df[column].dtype in ['float64', 'int64'] and len(df) > 5:
                values = df[column].dropna()
                
                if len(values) > 5:
                    # Check for linear interpolation patterns
                    # Look for sequences with constant second derivative (linear interpolation)
                    if len(values) >= 10:
                        second_diffs = values.diff().diff().dropna()
                        if len(second_diffs) > 5:
                            # If many second differences are very small, likely interpolated
                            small_second_diffs = (second_diffs.abs() < 1e-10).sum()
                            interpolation_percentage = small_second_diffs / len(second_diffs) * 100
                            
                            if interpolation_percentage > 70:  # More than 70% constant second derivative
                                violations.append(DataIntegrityViolation(
                                    violation_type='interpolated_data',
                                    severity='warning',
                                    column_name=column,
                                    row_indices=[],
                                    violation_count=small_second_diffs,
                                    data_source=DataSource.INTERPOLATED,
                                    description=f"Column '{column}' shows linear interpolation pattern ({interpolation_percentage:.1f}%)",
                                    recommended_action=f"Verify if '{column}' contains interpolated values"
                                ))
        
        return violations
    
    def _check_data_source_authenticity(self, df: pd.DataFrame, data_sources: Dict[str, DataSource]) -> List[DataIntegrityViolation]:
        """Check overall data source authenticity"""
        
        violations = []
        
        # Count data sources
        source_counts = {}
        for source in data_sources.values():
            source_counts[source] = source_counts.get(source, 0) + 1
        
        total_columns = len(data_sources)
        
        # Check authentic data percentage
        authentic_count = source_counts.get(DataSource.AUTHENTIC, 0)
        authentic_percentage = (authentic_count / total_columns * 100) if total_columns > 0 else 0
        
        if authentic_percentage < self.current_thresholds['min_authentic_percentage']:
            violations.append(DataIntegrityViolation(
                violation_type='insufficient_authentic_data',
                severity='critical',
                column_name='ALL_COLUMNS',
                row_indices=[],
                violation_count=total_columns - authentic_count,
                data_source=DataSource.UNKNOWN,
                description=f"Only {authentic_percentage:.1f}% of data is authentic (requires {self.current_thresholds['min_authentic_percentage']:.1f}%)",
                recommended_action="Increase authentic data sources or exclude non-authentic columns"
            ))
        
        # Check for unknown data sources
        unknown_count = source_counts.get(DataSource.UNKNOWN, 0)
        if unknown_count > 0:
            unknown_percentage = (unknown_count / total_columns) * 100
            violations.append(DataIntegrityViolation(
                violation_type='unknown_data_source',
                severity='warning',
                column_name='MULTIPLE_COLUMNS',
                row_indices=[],
                violation_count=unknown_count,
                data_source=DataSource.UNKNOWN,
                description=f"{unknown_percentage:.1f}% of columns have unknown data sources",
                recommended_action="Verify and document data sources for all columns"
            ))
        
        return violations
    
    def _check_unrealistic_patterns(self, df: pd.DataFrame) -> List[DataIntegrityViolation]:
        """Check for unrealistic data patterns that suggest synthetic data"""
        
        violations = []
        
        for column in df.columns:
            if df[column].dtype in ['float64', 'int64'] and len(df) > 10:
                values = df[column].dropna()
                
                if len(values) > 10:
                    # Check for unrealistically perfect distributions
                    try:
                        # Check if values are too perfectly distributed
                        if len(np.unique(values)) > 5:
                            # Calculate coefficient of variation
                            cv = values.std() / values.mean() if values.mean() != 0 else float('inf')
                            
                            # Real market data typically has CV between 0.1 and 2.0
                            if cv < 0.05:  # Too little variation
                                violations.append(DataIntegrityViolation(
                                    violation_type='unrealistic_pattern',
                                    severity='warning',
                                    column_name=column,
                                    row_indices=[],
                                    violation_count=len(values),
                                    data_source=DataSource.SYNTHETIC,
                                    description=f"Column '{column}' has unrealistically low variation (CV={cv:.3f})",
                                    recommended_action=f"Verify authenticity of '{column}' - real data should have more variation"
                                ))
                    except Exception:
                        pass  # Skip if calculation fails
        
        return violations
    
    def _calculate_authentic_percentage(self, df: pd.DataFrame, data_sources: Dict[str, DataSource]) -> float:
        """Calculate percentage of authentic data"""
        
        if not data_sources:
            return 0.0
        
        authentic_count = sum(1 for source in data_sources.values() if source == DataSource.AUTHENTIC)
        total_count = len(data_sources)
        
        return (authentic_count / total_count * 100) if total_count > 0 else 0.0
    
    def _generate_integrity_recommendations(self, violations: List[DataIntegrityViolation], authentic_percentage: float) -> List[str]:
        """Generate actionable recommendations for data integrity issues"""
        
        recommendations = []
        
        # Critical issues first
        critical_violations = [v for v in violations if v.severity == 'critical']
        
        if critical_violations:
            recommendations.append("CRITICAL: Address all critical data integrity violations before production deployment")
            
            # Specific recommendations by violation type
            synthetic_violations = [v for v in critical_violations if v.violation_type == 'synthetic_data']
            if synthetic_violations:
                recommendations.append("• Replace all synthetic data with authentic API data from exchanges")
            
            fallback_violations = [v for v in critical_violations if v.violation_type == 'fallback_used']
            if fallback_violations:
                recommendations.append("• Eliminate all fallback data - obtain authentic data or exclude columns")
            
            nan_violations = [v for v in critical_violations if v.violation_type == 'nan_values']
            if nan_violations:
                recommendations.append("• Remove all rows with NaN values or exclude incomplete columns")
            
            interpolated_violations = [v for v in critical_violations if v.violation_type == 'interpolated_data']
            if interpolated_violations:
                recommendations.append("• Remove interpolated data points - use only original authentic values")
        
        # Authenticity recommendations
        if authentic_percentage < 100.0:
            recommendations.append(f"• Increase authentic data percentage from {authentic_percentage:.1f}% to 100%")
            recommendations.append("• Verify all data sources and mark them appropriately")
            recommendations.append("• Implement strict API data collection without fallbacks")
        
        # Production mode recommendations
        if self.production_mode and violations:
            recommendations.extend([
                "PRODUCTION MODE: Zero-tolerance enforcement active",
                "• Block training/prediction until all violations resolved",
                "• Implement real-time data integrity monitoring",
                "• Set up alerts for any non-authentic data detection"
            ])
        
        # General best practices
        if violations:
            recommendations.extend([
                "Best Practices:",
                "• Use only direct exchange API data",
                "• Implement data source tracking and validation",
                "• Add real-time data integrity checks to ML pipeline",
                "• Create automated alerts for data quality issues"
            ])
        
        return recommendations
    
    def enforce_production_compliance(self, df: pd.DataFrame, data_sources: Dict[str, DataSource] = None) -> Tuple[pd.DataFrame, DataIntegrityReport]:
        """Enforce strict production compliance - blocks non-authentic data"""
        
        # Validate data integrity
        integrity_report = self.validate_data_integrity(df, data_sources)
        
        if not integrity_report.is_production_ready:
            # In production mode, completely block non-compliant data
            if self.production_mode:
                self.logger.error(f"PRODUCTION BLOCKED: {integrity_report.critical_violations} critical data integrity violations")
                raise ValueError(f"Data integrity violations detected: {integrity_report.critical_violations} critical issues. Production blocked.")
            else:
                self.logger.warning(f"Data integrity issues detected: {integrity_report.critical_violations} critical, {integrity_report.warning_violations} warnings")
        
        # Filter out non-authentic data if not in strict mode
        if not self.production_mode and data_sources:
            authentic_columns = [col for col, source in data_sources.items() if source == DataSource.AUTHENTIC]
            
            if authentic_columns and len(authentic_columns) < len(df.columns):
                self.logger.info(f"Filtering to {len(authentic_columns)} authentic columns out of {len(df.columns)} total")
                filtered_df = df[authentic_columns].copy()
                
                # Remove rows with any NaN values in authentic data
                filtered_df = filtered_df.dropna()
                
                return filtered_df, integrity_report
        
        return df, integrity_report

def create_data_integrity_enforcer(production_mode: bool = True) -> StrictDataIntegrityEnforcer:
    """Create strict data integrity enforcer"""
    return StrictDataIntegrityEnforcer(production_mode=production_mode)

def validate_production_data(df: pd.DataFrame, data_sources: Dict[str, DataSource] = None) -> DataIntegrityReport:
    """Quick production data validation"""
    enforcer = StrictDataIntegrityEnforcer(production_mode=True)
    return enforcer.validate_data_integrity(df, data_sources)