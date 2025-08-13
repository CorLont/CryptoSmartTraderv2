#!/usr/bin/env python3
"""
ML Feature Engineering Monitor with Leakage Detection and SHAP Analysis
Implements automatic feature validation, drift detection, and importance tracking
"""

import numpy as np
import pandas as pd
import shap
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp, wasserstein_distance
import json
from pathlib import Path

from ..core.logging_manager import get_logger

class LeakageType(str, Enum):
    """Types of feature leakage"""
    LOOK_AHEAD_BIAS = "look_ahead_bias"
    PERFECT_CORRELATION = "perfect_correlation"
    TARGET_LEAKAGE = "target_leakage"
    TEMPORAL_LEAKAGE = "temporal_leakage"
    GROUP_LEAKAGE = "group_leakage"

class DriftType(str, Enum):
    """Types of feature drift"""
    DISTRIBUTION_SHIFT = "distribution_shift"
    CORRELATION_CHANGE = "correlation_change"
    IMPORTANCE_DRIFT = "importance_drift"
    STATISTICAL_DRIFT = "statistical_drift"

@dataclass
class LeakageViolation:
    """Feature leakage violation record"""
    feature_name: str
    leakage_type: LeakageType
    severity: float  # 0.0 to 1.0
    description: str
    detection_method: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DriftAlert:
    """Feature drift alert"""
    feature_name: str
    drift_type: DriftType
    drift_magnitude: float
    baseline_stats: Dict[str, float]
    current_stats: Dict[str, float]
    p_value: float
    action_required: bool
    timestamp: datetime

@dataclass
class FeatureImportanceReport:
    """SHAP-based feature importance analysis"""
    feature_name: str
    shap_importance: float
    shap_values: List[float]
    stability_score: float
    interaction_effects: Dict[str, float]
    temporal_consistency: float
    recommendation: str  # 'keep', 'monitor', 'remove'

class FeatureLeakageDetector:
    """Advanced feature leakage detection"""

    def __init__(self):
        self.logger = get_logger()
        self.leakage_thresholds = {
            'perfect_correlation': 0.95,
            'temporal_correlation': 0.8,
            'target_correlation': 0.9,
            'mutual_info_threshold': 0.8
        }

    def detect_leakage(
        self,
        features_df: pd.DataFrame,
        target: np.ndarray,
        feature_timestamps: Optional[List[datetime]] = None,
        target_timestamps: Optional[List[datetime]] = None
    ) -> List[LeakageViolation]:
        """Comprehensive feature leakage detection"""

        violations = []

        # 1. Perfect correlation detection
        violations.extend(self._detect_perfect_correlations(features_df, target))

        # 2. Target leakage detection
        violations.extend(self._detect_target_leakage(features_df, target))

        # 3. Temporal leakage detection
        if feature_timestamps and target_timestamps:
            violations.extend(self._detect_temporal_leakage(
                features_df, target, feature_timestamps, target_timestamps
            ))

        # 4. Look-ahead bias detection
        violations.extend(self._detect_lookahead_bias(features_df, target))

        # 5. Group leakage detection
        violations.extend(self._detect_group_leakage(features_df, target))

        # Log leakage summary
        if violations:
            self.logger.warning(
                f"Feature leakage detected: {len(violations)} violations",
                extra={
                    'total_violations': len(violations),
                    'violation_types': [v.leakage_type.value for v in violations],
                    'severe_violations': len([v for v in violations if v.severity > 0.8])
                }
            )

        return violations

    def _detect_perfect_correlations(self, features_df: pd.DataFrame, target: np.ndarray) -> List[LeakageViolation]:
        """Detect perfect or near-perfect correlations with target"""
        violations = []

        for column in features_df.columns:
            try:
                feature_values = features_df[column].values

                # Skip non-numeric features
                if not np.issubdtype(feature_values.dtype, np.number):
                    continue

                # Calculate correlation with target
                correlation = np.corrcoef(feature_values, target)[0, 1]

                if abs(correlation) > self.leakage_thresholds['perfect_correlation']:
                    violations.append(LeakageViolation(
                        feature_name=column,
                        leakage_type=LeakageType.PERFECT_CORRELATION,
                        severity=abs(correlation),
                        description=f"Perfect correlation with target: {correlation:.3f}",
                        detection_method="pearson_correlation",
                        timestamp=datetime.now(),
                        metadata={'correlation': correlation}
                    ))

            except Exception as e:
                self.logger.warning(f"Correlation check failed for {column}: {e}")

        return violations

    def _detect_target_leakage(self, features_df: pd.DataFrame, target: np.ndarray) -> List[LeakageViolation]:
        """Detect features that contain target information"""
        violations = []

        # Calculate mutual information between features and target
        try:
            for column in features_df.columns:
                feature_values = features_df[column].values

                if not np.issubdtype(feature_values.dtype, np.number):
                    continue

                # Handle missing values
                mask = ~(np.isnan(feature_values) | np.isnan(target))
                if np.sum(mask) < len(target) * 0.5:  # Too many missing values
                    continue

                clean_features = feature_values[mask].reshape(-1, 1)
                clean_target = target[mask]

                # Calculate mutual information
                mi_score = mutual_info_regression(clean_features, clean_target)[0]

                # Normalize by entropy of target
                target_entropy = -np.sum(clean_target * np.log2(clean_target + 1e-10))
                normalized_mi = mi_score / (target_entropy + 1e-10) if target_entropy > 0 else 0

                if normalized_mi > self.leakage_thresholds['mutual_info_threshold']:
                    violations.append(LeakageViolation(
                        feature_name=column,
                        leakage_type=LeakageType.TARGET_LEAKAGE,
                        severity=normalized_mi,
                        description=f"High mutual information with target: {normalized_mi:.3f}",
                        detection_method="mutual_information",
                        timestamp=datetime.now(),
                        metadata={'mutual_info': mi_score, 'normalized_mi': normalized_mi}
                    ))

        except Exception as e:
            self.logger.error(f"Target leakage detection failed: {e}")

        return violations

    def _detect_temporal_leakage(
        self,
        features_df: pd.DataFrame,
        target: np.ndarray,
        feature_timestamps: List[datetime],
        target_timestamps: List[datetime]
    ) -> List[LeakageViolation]:
        """Detect temporal leakage (features using future information)"""
        violations = []

        try:
            # Check if any features have timestamps after target timestamps
            for i, (feat_time, target_time) in enumerate(zip(feature_timestamps, target_timestamps)):
                if feat_time > target_time:
                    # Find which features have future information
                    for column in features_df.columns:
                        if not pd.isna(features_df.iloc[i][column]):
                            violations.append(LeakageViolation(
                                feature_name=column,
                                leakage_type=LeakageType.TEMPORAL_LEAKAGE,
                                severity=0.9,  # High severity for temporal violations
                                description=f"Feature uses future information: {feat_time} > {target_time}",
                                detection_method="timestamp_comparison",
                                timestamp=datetime.now(),
                                metadata={
                                    'feature_timestamp': feat_time.isoformat(),
                                    'target_timestamp': target_time.isoformat(),
                                    'time_diff_minutes': (feat_time - target_time).total_seconds() / 60
                                }
                            ))
                            break  # Only report once per timestamp

        except Exception as e:
            self.logger.error(f"Temporal leakage detection failed: {e}")

        return violations

    def _detect_lookahead_bias(self, features_df: pd.DataFrame, target: np.ndarray) -> List[LeakageViolation]:
        """Detect look-ahead bias in feature construction"""
        violations = []

        # Check for features that might use future information in their calculation
        suspicious_patterns = [
            'future_', 'next_', 'lead_', 'forward_', 'ahead_'
        ]

        for column in features_df.columns:
            column_lower = column.lower()

            for pattern in suspicious_patterns:
                if pattern in column_lower:
                    violations.append(LeakageViolation(
                        feature_name=column,
                        leakage_type=LeakageType.LOOK_AHEAD_BIAS,
                        severity=0.7,
                        description=f"Suspicious feature name suggests look-ahead bias: {column}",
                        detection_method="name_pattern_matching",
                        timestamp=datetime.now(),
                        metadata={'suspicious_pattern': pattern}
                    ))

        return violations

    def _detect_group_leakage(self, features_df: pd.DataFrame, target: np.ndarray) -> List[LeakageViolation]:
        """Detect group leakage (features calculated across entire dataset)"""
        violations = []

        try:
            # Look for features that have suspiciously low variance (might be global statistics)
            for column in features_df.columns:
                feature_values = features_df[column].values

                if not np.issubdtype(feature_values.dtype, np.number):
                    continue

                # Calculate coefficient of variation
                mean_val = np.mean(feature_values)
                std_val = np.std(feature_values)

                if mean_val != 0:
                    cv = std_val / abs(mean_val)

                    # Very low coefficient of variation might indicate global statistics
                    if cv < 0.01 and std_val > 0:
                        violations.append(LeakageViolation(
                            feature_name=column,
                            leakage_type=LeakageType.GROUP_LEAKAGE,
                            severity=0.6,
                            description=f"Suspiciously low variance (CV={cv:.4f}) suggests group leakage",
                            detection_method="coefficient_of_variation",
                            timestamp=datetime.now(),
                            metadata={'coefficient_of_variation': cv, 'std': std_val, 'mean': mean_val}
                        ))

        except Exception as e:
            self.logger.error(f"Group leakage detection failed: {e}")

        return violations

class FeatureDriftMonitor:
    """Feature drift detection and monitoring"""

    def __init__(self):
        self.logger = get_logger()
        self.baseline_stats = {}
        self.drift_thresholds = {
            'ks_test_p_value': 0.05,
            'wasserstein_threshold': 0.3,
            'correlation_change': 0.2,
            'importance_change': 0.3
        }

    def establish_baseline(self, features_df: pd.DataFrame, target: np.ndarray):
        """Establish baseline statistics for drift detection"""

        self.baseline_stats = {}

        for column in features_df.columns:
            try:
                feature_values = features_df[column].values

                if not np.issubdtype(feature_values.dtype, np.number):
                    continue

                # Calculate baseline statistics
                self.baseline_stats[column] = {
                    'mean': np.mean(feature_values),
                    'std': np.std(feature_values),
                    'median': np.median(feature_values),
                    'q25': np.percentile(feature_values, 25),
                    'q75': np.percentile(feature_values, 75),
                    'min': np.min(feature_values),
                    'max': np.max(feature_values),
                    'correlation_with_target': np.corrcoef(feature_values, target)[0, 1] if len(target) == len(feature_values) else 0,
                    'timestamp': datetime.now().isoformat()
                }

            except Exception as e:
                self.logger.warning(f"Baseline calculation failed for {column}: {e}")

        self.logger.info(
            f"Baseline established for {len(self.baseline_stats)} features",
            extra={'features_count': len(self.baseline_stats)}
        )

    def detect_drift(self, features_df: pd.DataFrame, target: np.ndarray) -> List[DriftAlert]:
        """Detect feature drift compared to baseline"""

        if not self.baseline_stats:
            self.logger.warning("No baseline established for drift detection")
            return []

        alerts = []

        for column in features_df.columns:
            if column not in self.baseline_stats:
                continue

            try:
                feature_values = features_df[column].values

                if not np.issubdtype(feature_values.dtype, np.number):
                    continue

                # Calculate current statistics
                current_stats = {
                    'mean': np.mean(feature_values),
                    'std': np.std(feature_values),
                    'median': np.median(feature_values),
                    'correlation_with_target': np.corrcoef(feature_values, target)[0, 1] if len(target) == len(feature_values) else 0
                }

                baseline = self.baseline_stats[column]

                # Statistical drift tests
                drift_detected = False
                p_value = 1.0
                drift_magnitude = 0.0

                # Kolmogorov-Smirnov test for distribution shift
                try:
                    # Generate baseline samples (approximation)
                    baseline_samples = np.random.normal(0, 1)
                    ks_stat, p_value = ks_2samp(baseline_samples, feature_values)

                    if p_value < self.drift_thresholds['ks_test_p_value']:
                        drift_detected = True
                        drift_magnitude = ks_stat

                except Exception:
                    pass

                # Wasserstein distance for distribution comparison
                try:
                    baseline_samples = np.random.normal(0, 1)
                    wasserstein_dist = wasserstein_distance(baseline_samples, feature_values)

                    # Normalize by baseline standard deviation
                    normalized_wasserstein = wasserstein_dist / (baseline['std'] + 1e-10)

                    if normalized_wasserstein > self.drift_thresholds['wasserstein_threshold']:
                        drift_detected = True
                        drift_magnitude = max(drift_magnitude, normalized_wasserstein)

                except Exception:
                    pass

                # Correlation drift
                correlation_change = abs(
                    current_stats['correlation_with_target'] - baseline['correlation_with_target']
                )

                if correlation_change > self.drift_thresholds['correlation_change']:
                    drift_detected = True
                    drift_magnitude = max(drift_magnitude, correlation_change)

                if drift_detected:
                    alerts.append(DriftAlert(
                        feature_name=column,
                        drift_type=DriftType.DISTRIBUTION_SHIFT,
                        drift_magnitude=drift_magnitude,
                        baseline_stats=baseline,
                        current_stats=current_stats,
                        p_value=p_value,
                        action_required=drift_magnitude > 0.5,
                        timestamp=datetime.now())

            except Exception as e:
                self.logger.warning(f"Drift detection failed for {column}: {e}")

        if alerts:
            self.logger.warning(
                f"Feature drift detected: {len(alerts)} features",
                extra={
                    'drift_alerts': len(alerts),
                    'high_priority_alerts': len([a for a in alerts if a.action_required])
                }
            )

        return alerts

class SHAPFeatureAnalyzer:
    """SHAP-based feature importance and interaction analysis"""

    def __init__(self):
        self.logger = get_logger()
        self.explainer = None
        self.feature_importance_history = []

    def analyze_feature_importance(
        self,
        model: Any,
        features_df: pd.DataFrame,
        background_samples: int = 100
    ) -> List[FeatureImportanceReport]:
        """Analyze feature importance using SHAP values"""

        reports = []

        try:
            # Create SHAP explainer
            if hasattr(model, 'predict'):
                # Use a sample for background
                background = features_df.sample(
                    min(background_samples, len(features_df)),
                    random_state=42
                )

                # Try different explainer types
                try:
                    self.explainer = shap.TreeExplainer(model)
                except Exception:
                    try:
                        self.explainer = shap.LinearExplainer(model, background)
                    except Exception:
                        try:
                            self.explainer = shap.KernelExplainer(model.predict, background)
                        except Exception as e:
                            self.logger.error(f"Failed to create SHAP explainer: {e}")
                            return reports

                # Calculate SHAP values
                shap_values = self.explainer.shap_values(features_df)

                # Handle different SHAP value formats
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]  # Take first class for multi-class

                # Create reports for each feature
                for i, feature_name in enumerate(features_df.columns):
                    feature_shap_values = shap_values[:, i] if len(shap_values.shape) > 1 else shap_values

                    # Calculate importance metrics
                    mean_abs_shap = np.mean(np.abs(feature_shap_values))
                    stability_score = 1.0 - (np.std(feature_shap_values) / (mean_abs_shap + 1e-10))

                    # Calculate interaction effects (simplified)
                    interaction_effects = {}
                    for j, other_feature in enumerate(features_df.columns):
                        if i != j:
                            correlation = np.corrcoef(feature_shap_values, shap_values[:, j])[0, 1]
                            if abs(correlation) > 0.3:  # Significant interaction
                                interaction_effects[other_feature] = correlation

                    # Determine recommendation
                    if mean_abs_shap > np.percentile(np.abs(shap_values), 75):
                        recommendation = 'keep'
                    elif mean_abs_shap > np.percentile(np.abs(shap_values), 25):
                        recommendation = 'monitor'
                    else:
                        recommendation = 'remove'

                    reports.append(FeatureImportanceReport(
                        feature_name=feature_name,
                        shap_importance=mean_abs_shap,
                        shap_values=feature_shap_values.tolist()[:100],  # Limit size
                        stability_score=stability_score,
                        interaction_effects=interaction_effects,
                        temporal_consistency=0.8,  # Placeholder
                        recommendation=recommendation
                    ))

        except Exception as e:
            self.logger.error(f"SHAP analysis failed: {e}")

        return reports

    def track_importance_over_time(self, reports: List[FeatureImportanceReport]):
        """Track feature importance changes over time"""

        timestamp = datetime.now()

        importance_snapshot = {
            'timestamp': timestamp.isoformat(),
            'feature_importance': {
                report.feature_name: {
                    'shap_importance': report.shap_importance,
                    'stability_score': report.stability_score,
                    'recommendation': report.recommendation
                }
                for report in reports
            }
        }

        self.feature_importance_history.append(importance_snapshot)

        # Keep last 100 snapshots
        if len(self.feature_importance_history) > 100:
            self.feature_importance_history = self.feature_importance_history[-100:]

        self.logger.info(
            f"Feature importance tracked: {len(reports)} features",
            extra={
                'timestamp': timestamp.isoformat(),
                'features_analyzed': len(reports),
                'high_importance_features': len([r for r in reports if r.recommendation == 'keep'])
            }
        )

class MLFeatureMonitor:
    """Main ML feature monitoring engine"""

    def __init__(self):
        self.logger = get_logger()

        # Component monitors
        self.leakage_detector = FeatureLeakageDetector()
        self.drift_monitor = FeatureDriftMonitor()
        self.shap_analyzer = SHAPFeatureAnalyzer()

        # Monitoring state
        self.monitoring_active = True
        self.last_analysis = None
        self.feature_quality_scores = {}

    def comprehensive_feature_analysis(
        self,
        features_df: pd.DataFrame,
        target: np.ndarray,
        model: Optional[Any] = None,
        feature_timestamps: Optional[List[datetime]] = None,
        target_timestamps: Optional[List[datetime]] = None
    ) -> Dict[str, Any]:
        """Comprehensive feature analysis with leakage, drift, and importance"""

        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'total_features': len(features_df.columns),
            'leakage_violations': [],
            'drift_alerts': [],
            'feature_importance': [],
            'recommendations': {},
            'quality_scores': {}
        }

        try:
            # 1. Leakage detection
            self.logger.info("Running feature leakage detection...")
            leakage_violations = self.leakage_detector.detect_leakage(
                features_df, target, feature_timestamps, target_timestamps
            )
            analysis_results['leakage_violations'] = [
                {
                    'feature': v.feature_name,
                    'type': v.leakage_type.value,
                    'severity': v.severity,
                    'description': v.description
                }
                for v in leakage_violations
            ]

            # 2. Drift detection
            self.logger.info("Running feature drift detection...")
            if not self.drift_monitor.baseline_stats:
                self.drift_monitor.establish_baseline(features_df, target)

            drift_alerts = self.drift_monitor.detect_drift(features_df, target)
            analysis_results['drift_alerts'] = [
                {
                    'feature': alert.feature_name,
                    'drift_type': alert.drift_type.value,
                    'magnitude': alert.drift_magnitude,
                    'p_value': alert.p_value,
                    'action_required': alert.action_required
                }
                for alert in drift_alerts
            ]

            # 3. SHAP importance analysis
            if model is not None:
                self.logger.info("Running SHAP feature importance analysis...")
                importance_reports = self.shap_analyzer.analyze_feature_importance(
                    model, features_df
                )
                analysis_results['feature_importance'] = [
                    {
                        'feature': report.feature_name,
                        'importance': report.shap_importance,
                        'stability': report.stability_score,
                        'recommendation': report.recommendation,
                        'interactions': report.interaction_effects
                    }
                    for report in importance_reports
                ]

                # Track importance over time
                self.shap_analyzer.track_importance_over_time(importance_reports)

            # 4. Generate recommendations
            analysis_results['recommendations'] = self._generate_feature_recommendations(
                leakage_violations, drift_alerts, analysis_results.get('feature_importance', [])

            # 5. Calculate quality scores
            analysis_results['quality_scores'] = self._calculate_feature_quality_scores(
                features_df.columns, leakage_violations, drift_alerts,
                analysis_results.get('feature_importance', [])

            self.last_analysis = analysis_results

            # Log summary
            self.logger.info(
                "Feature analysis completed",
                extra={
                    'total_features': analysis_results['total_features'],
                    'leakage_violations': len(analysis_results['leakage_violations']),
                    'drift_alerts': len(analysis_results['drift_alerts']),
                    'features_to_remove': len([f for f, r in analysis_results['recommendations'].items() if r == 'remove']),
                    'features_to_monitor': len([f for f, r in analysis_results['recommendations'].items() if r == 'monitor'])
                }
            )

        except Exception as e:
            self.logger.error(f"Feature analysis failed: {e}")
            analysis_results['error'] = str(e)

        return analysis_results

    def _generate_feature_recommendations(
        self,
        leakage_violations: List[LeakageViolation],
        drift_alerts: List[DriftAlert],
        importance_reports: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Generate actionable feature recommendations"""

        recommendations = {}

        # Features with severe leakage should be removed
        for violation in leakage_violations:
            if violation.severity > 0.8:
                recommendations[violation.feature_name] = 'remove'
            elif violation.severity > 0.5:
                recommendations[violation.feature_name] = 'investigate'

        # Features with high drift need monitoring
        for alert in drift_alerts:
            if alert.action_required:
                if alert.feature_name not in recommendations:
                    recommendations[alert.feature_name] = 'monitor'

        # Apply importance-based recommendations
        for report in importance_reports:
            feature_name = report['feature']
            if feature_name not in recommendations:
                recommendations[feature_name] = report['recommendation']

        return recommendations

    def _calculate_feature_quality_scores(
        self,
        feature_names: List[str],
        leakage_violations: List[LeakageViolation],
        drift_alerts: List[DriftAlert],
        importance_reports: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate overall quality scores for features"""

        quality_scores = {}

        # Initialize with base score
        for feature in feature_names:
            quality_scores[feature] = 1.0

        # Penalize for leakage
        for violation in leakage_violations:
            if violation.feature_name in quality_scores:
                penalty = violation.severity * 0.8  # Up to 80% penalty
                quality_scores[violation.feature_name] -= penalty

        # Penalize for drift
        for alert in drift_alerts:
            if alert.feature_name in quality_scores:
                penalty = min(alert.drift_magnitude * 0.3, 0.5)  # Up to 50% penalty
                quality_scores[alert.feature_name] -= penalty

        # Boost for high importance
        for report in importance_reports:
            feature_name = report['feature']
            if feature_name in quality_scores and report['recommendation'] == 'keep':
                boost = report['importance'] * 0.2  # Up to 20% boost
                quality_scores[feature_name] += boost

        # Clamp scores to [0, 1]
        for feature in quality_scores:
            quality_scores[feature] = max(0.0, min(1.0, quality_scores[feature]))

        return quality_scores

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get comprehensive monitoring summary"""

        summary = {
            'monitoring_status': 'active' if self.monitoring_active else 'inactive',
            'last_analysis': self.last_analysis['timestamp'] if self.last_analysis else None,
            'total_features_monitored': len(self.feature_quality_scores),
            'feature_quality_distribution': {},
            'recent_issues': {}
        }

        if self.feature_quality_scores:
            scores = list(self.feature_quality_scores.values())
            summary['feature_quality_distribution'] = {
                'high_quality': len([s for s in scores if s > 0.8]),
                'medium_quality': len([s for s in scores if 0.5 <= s <= 0.8]),
                'low_quality': len([s for s in scores if s < 0.5])
            }

        if self.last_analysis:
            summary['recent_issues'] = {
                'leakage_violations': len(self.last_analysis.get('leakage_violations', [])),
                'drift_alerts': len(self.last_analysis.get('drift_alerts', [])),
                'features_needing_attention': len([
                    f for f, r in self.last_analysis.get('recommendations', {}).items()
                    if r in ['remove', 'investigate', 'monitor']
                ])
            }

        return summary

# Global instance
_feature_monitor = None

def get_feature_monitor() -> MLFeatureMonitor:
    """Get global feature monitor instance"""
    global _feature_monitor
    if _feature_monitor is None:
        _feature_monitor = MLFeatureMonitor()
    return _feature_monitor
