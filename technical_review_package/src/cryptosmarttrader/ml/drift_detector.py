"""
Drift Detection System for CryptoSmartTrader
Advanced statistical and data drift detection with automatic rollback capabilities.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import warnings
from pathlib import Path

# Statistical libraries
try:
    from scipy import stats
    from scipy.spatial.distance import jensenshannon
except ImportError:
    warnings.warn("SciPy not available, some drift detection features may be limited")
    stats = None
    jensenshannon = None

from .model_registry import ModelRegistry, ModelVersion


class DriftType(Enum):
    """Types of drift that can be detected."""

    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"
    STATISTICAL_DRIFT = "statistical_drift"
    DISTRIBUTION_DRIFT = "distribution_drift"


class DriftSeverity(Enum):
    """Severity levels for drift detection."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """Drift detection alert with comprehensive information."""

    drift_type: DriftType
    severity: DriftSeverity
    detection_time: datetime
    affected_features: List[str]
    drift_scores: Dict[str, float]
    statistical_tests: Dict[str, Dict[str, float]]

    # Rollback information
    rollback_recommended: bool
    rollback_confidence: float
    rollback_reasons: List[str]

    # Metadata
    baseline_period: Tuple[datetime, datetime]
    current_period: Tuple[datetime, datetime]
    sample_sizes: Dict[str, int]
    confidence_intervals: Dict[str, Tuple[float, float]]

    # Action recommendations
    recommended_actions: List[str]
    urgency_level: int  # 1-5 scale


@dataclass
class DriftConfig:
    """Configuration for drift detection system."""

    # Detection thresholds
    data_drift_threshold: float = 0.3
    concept_drift_threshold: float = 0.25
    performance_drift_threshold: float = 0.15
    prediction_drift_threshold: float = 0.2

    # Statistical test settings
    significance_level: float = 0.05
    min_sample_size: int = 100
    baseline_window_days: int = 30
    detection_window_days: int = 7

    # Monitoring frequency
    check_frequency_hours: int = 6
    continuous_monitoring: bool = True

    # Rollback settings
    auto_rollback_threshold: float = 0.8
    rollback_confirmation_required: bool = True
    max_rollback_attempts: int = 3

    # Feature-specific settings
    numerical_drift_method: str = "ks_test"  # ks_test, psi, js_divergence
    categorical_drift_method: str = "chi2_test"

    # Performance monitoring
    performance_metrics: List[str] = None
    performance_window_size: int = 100


class DriftDetector:
    """
    Enterprise drift detection system with statistical testing and automatic rollback.

    Features:
    - Multiple drift detection methods (KS test, PSI, JS divergence, Chi-square)
    - Real-time monitoring of data, concept, and performance drift
    - Statistical significance testing with confidence intervals
    - Automatic rollback recommendations and execution
    - Feature-level drift analysis with importance weighting
    - Comprehensive alerting and reporting system
    - Historical drift tracking and trend analysis
    """

    def __init__(self, model_registry: ModelRegistry, config: DriftConfig = None):
        self.model_registry = model_registry
        self.config = config or DriftConfig()

        # Drift monitoring state
        self.baseline_data: Dict[str, pd.DataFrame] = {}
        self.drift_history: List[DriftAlert] = []
        self.last_check_time: Optional[datetime] = None

        # Performance tracking
        self.performance_baseline: Dict[str, List[float]] = {}
        self.current_performance: Dict[str, List[float]] = {}

        # Rollback tracking
        self.rollback_history: List[Dict[str, Any]] = []
        self.rollback_attempts: int = 0

        # Feature importance for drift weighting
        self.feature_weights: Dict[str, float] = {}

        self.logger = logging.getLogger(__name__)
        self.logger.info("DriftDetector initialized with advanced statistical methods")

    def set_baseline(
        self,
        model_id: str,
        version: str,
        baseline_data: pd.DataFrame,
        performance_data: Optional[Dict[str, List[float]]] = None,
    ):
        """
        Set baseline data and performance for drift detection.

        Args:
            model_id: Model identifier
            version: Model version
            baseline_data: Historical data to use as baseline
            performance_data: Historical performance metrics
        """

        baseline_key = f"{model_id}_{version}"

        # Store baseline data
        self.baseline_data[baseline_key] = baseline_data.copy()

        # Store performance baseline
        if performance_data:
            self.performance_baseline[baseline_key] = performance_data

        # Calculate feature weights from model
        self._update_feature_weights(model_id, version, baseline_data)

        self.logger.info(f"Baseline set for {model_id} {version}: {len(baseline_data)} samples")

    def detect_drift(
        self,
        model_id: str,
        version: str,
        current_data: pd.DataFrame,
        current_performance: Optional[Dict[str, List[float]]] = None,
    ) -> List[DriftAlert]:
        """
        Comprehensive drift detection across multiple dimensions.

        Args:
            model_id: Model identifier
            version: Model version
            current_data: Current data for comparison
            current_performance: Current performance metrics

        Returns:
            List of DriftAlert objects for detected drift
        """

        baseline_key = f"{model_id}_{version}"

        if baseline_key not in self.baseline_data:
            self.logger.warning(f"No baseline data found for {model_id} {version}")
            return []

        baseline_data = self.baseline_data[baseline_key]
        detection_time = datetime.utcnow()

        alerts = []

        # Data drift detection
        data_drift_alert = self._detect_data_drift(
            baseline_data, current_data, model_id, version, detection_time
        )
        if data_drift_alert:
            alerts.append(data_drift_alert)

        # Performance drift detection
        if current_performance and baseline_key in self.performance_baseline:
            perf_drift_alert = self._detect_performance_drift(
                self.performance_baseline[baseline_key],
                current_performance,
                model_id,
                version,
                detection_time,
            )
            if perf_drift_alert:
                alerts.append(perf_drift_alert)

        # Prediction drift detection (if model predictions available)
        pred_drift_alert = self._detect_prediction_drift(
            baseline_data, current_data, model_id, version, detection_time
        )
        if pred_drift_alert:
            alerts.append(pred_drift_alert)

        # Store alerts
        self.drift_history.extend(alerts)
        self.last_check_time = detection_time

        # Evaluate rollback recommendations
        for alert in alerts:
            self._evaluate_rollback_recommendation(alert, model_id, version)

        if alerts:
            self.logger.warning(f"Drift detected for {model_id} {version}: {len(alerts)} alerts")

        return alerts

    def monitor_continuous_drift(
        self,
        model_id: str,
        version: str,
        data_stream: pd.DataFrame,
        performance_stream: Optional[Dict[str, List[float]]] = None,
    ) -> bool:
        """
        Continuous monitoring of drift with automatic alerts.

        Args:
            model_id: Model identifier
            version: Model version
            data_stream: Streaming data for monitoring
            performance_stream: Streaming performance data

        Returns:
            True if critical drift detected requiring immediate action
        """

        if not self.config.continuous_monitoring:
            return False

        # Check if monitoring is due
        if (
            self.last_check_time
            and (datetime.utcnow() - self.last_check_time).total_seconds()
            < self.config.check_frequency_hours * 3600
        ):
            return False

        # Detect drift
        alerts = self.detect_drift(model_id, version, data_stream, performance_stream)

        # Check for critical drift
        critical_alerts = [a for a in alerts if a.severity == DriftSeverity.CRITICAL]

        if critical_alerts:
            self.logger.critical(f"Critical drift detected for {model_id} {version}")

            # Auto-rollback if configured
            if any(
                a.rollback_recommended
                and a.rollback_confidence > self.config.auto_rollback_threshold
                for a in critical_alerts
            ):
                self._execute_automatic_rollback(model_id, version, critical_alerts)

            return True

        return False

    def _detect_data_drift(
        self,
        baseline_data: pd.DataFrame,
        current_data: pd.DataFrame,
        model_id: str,
        version: str,
        detection_time: datetime,
    ) -> Optional[DriftAlert]:
        """Detect data drift using statistical tests."""

        if len(current_data) < self.config.min_sample_size:
            return None

        drift_scores = {}
        statistical_tests = {}
        affected_features = []

        # Align features
        common_features = list(set(baseline_data.columns) & set(current_data.columns))

        for feature in common_features:
            baseline_values = baseline_data[feature].dropna()
            current_values = current_data[feature].dropna()

            if len(baseline_values) == 0 or len(current_values) == 0:
                continue

            # Determine feature type
            if pd.api.types.is_numeric_dtype(baseline_values):
                drift_score, test_results = self._test_numerical_drift(
                    baseline_values, current_values, feature
                )
            else:
                drift_score, test_results = self._test_categorical_drift(
                    baseline_values, current_values, feature
                )

            drift_scores[feature] = drift_score
            statistical_tests[feature] = test_results

            # Check if feature has significant drift
            if drift_score > self.config.data_drift_threshold:
                affected_features.append(feature)

        if not drift_scores:
            return None

        # Calculate overall drift score (weighted by feature importance)
        weighted_scores = []
        for feature, score in drift_scores.items():
            weight = self.feature_weights.get(feature, 1.0)
            weighted_scores.append(score * weight)

        overall_drift = np.mean(weighted_scores)

        # Determine severity
        if overall_drift > 0.7:
            severity = DriftSeverity.CRITICAL
        elif overall_drift > 0.5:
            severity = DriftSeverity.HIGH
        elif overall_drift > 0.3:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW

        # Skip alert if drift is low
        if severity == DriftSeverity.LOW and overall_drift < self.config.data_drift_threshold:
            return None

        return DriftAlert(
            drift_type=DriftType.DATA_DRIFT,
            severity=severity,
            detection_time=detection_time,
            affected_features=affected_features,
            drift_scores=drift_scores,
            statistical_tests=statistical_tests,
            rollback_recommended=severity in [DriftSeverity.HIGH, DriftSeverity.CRITICAL],
            rollback_confidence=min(0.95, overall_drift),
            rollback_reasons=self._generate_rollback_reasons(overall_drift, affected_features),
            baseline_period=(baseline_data.index.min(), baseline_data.index.max()),
            current_period=(current_data.index.min(), current_data.index.max()),
            sample_sizes={"baseline": len(baseline_data), "current": len(current_data)},
            confidence_intervals=self._calculate_confidence_intervals(drift_scores),
            recommended_actions=self._generate_drift_actions(severity, affected_features),
            urgency_level=self._calculate_urgency_level(severity, overall_drift),
        )

    def _test_numerical_drift(
        self, baseline_values: pd.Series, current_values: pd.Series, feature_name: str
    ) -> Tuple[float, Dict[str, float]]:
        """Test numerical feature drift using multiple methods."""

        test_results = {}

        # Kolmogorov-Smirnov test
        if stats:
            try:
                ks_stat, ks_pvalue = stats.ks_2samp(baseline_values, current_values)
                test_results["ks_statistic"] = ks_stat
                test_results["ks_pvalue"] = ks_pvalue
                ks_drift = ks_stat  # KS statistic as drift score
            except Exception:
                ks_drift = 0.0
                test_results["ks_statistic"] = 0.0
                test_results["ks_pvalue"] = 1.0
        else:
            ks_drift = 0.0

        # Population Stability Index (PSI)
        try:
            psi_score = self._calculate_psi(baseline_values, current_values)
            test_results["psi_score"] = psi_score
            psi_drift = min(1.0, psi_score / 0.25)  # Normalize PSI
        except Exception:
            psi_drift = 0.0
            test_results["psi_score"] = 0.0

        # Jensen-Shannon divergence (if available)
        if jensenshannon:
            try:
                # Create histograms
                bins = np.linspace(
                    min(baseline_values.min(), current_values.min()),
                    max(baseline_values.max(), current_values.max()),
                    20,
                )

                hist_baseline, _ = np.histogram(baseline_values, bins=bins, density=True)
                hist_current, _ = np.histogram(current_values, bins=bins, density=True)

                # Add small epsilon to avoid zero probabilities
                hist_baseline = hist_baseline + 1e-10
                hist_current = hist_current + 1e-10

                # Normalize
                hist_baseline = hist_baseline / hist_baseline.sum()
                hist_current = hist_current / hist_current.sum()

                js_divergence = jensenshannon(hist_baseline, hist_current)
                test_results["js_divergence"] = js_divergence
                js_drift = js_divergence
            except Exception:
                js_drift = 0.0
                test_results["js_divergence"] = 0.0
        else:
            js_drift = 0.0

        # Simple statistical measures
        baseline_mean = baseline_values.mean()
        current_mean = current_values.mean()
        baseline_std = baseline_values.std()
        current_std = current_values.std()

        mean_drift = abs(current_mean - baseline_mean) / (baseline_std + 1e-8)
        std_drift = abs(current_std - baseline_std) / (baseline_std + 1e-8)

        test_results["mean_drift"] = mean_drift
        test_results["std_drift"] = std_drift
        test_results["baseline_mean"] = baseline_mean
        test_results["current_mean"] = current_mean
        test_results["baseline_std"] = baseline_std
        test_results["current_std"] = current_std

        # Combine drift scores
        if self.config.numerical_drift_method == "ks_test":
            primary_drift = ks_drift
        elif self.config.numerical_drift_method == "psi":
            primary_drift = psi_drift
        elif self.config.numerical_drift_method == "js_divergence":
            primary_drift = js_drift
        else:
            primary_drift = np.mean([ks_drift, psi_drift, js_drift])

        # Weight with statistical measures
        combined_drift = 0.7 * primary_drift + 0.2 * mean_drift + 0.1 * std_drift

        return min(1.0, combined_drift), test_results

    def _test_categorical_drift(
        self, baseline_values: pd.Series, current_values: pd.Series, feature_name: str
    ) -> Tuple[float, Dict[str, float]]:
        """Test categorical feature drift using Chi-square test."""

        test_results = {}

        # Get value counts
        baseline_counts = baseline_values.value_counts()
        current_counts = current_values.value_counts()

        # Align categories
        all_categories = set(baseline_counts.index) | set(current_counts.index)

        baseline_aligned = pd.Series(
            [baseline_counts.get(cat, 0) for cat in all_categories], index=all_categories
        )
        current_aligned = pd.Series(
            [current_counts.get(cat, 0) for cat in all_categories], index=all_categories
        )

        # Chi-square test
        if stats and len(all_categories) > 1:
            try:
                # Expected counts
                total_baseline = baseline_aligned.sum()
                total_current = current_aligned.sum()

                if total_baseline > 0 and total_current > 0:
                    expected_baseline = baseline_aligned / total_baseline
                    expected_current = current_aligned / total_current

                    # Chi-square statistic
                    chi2_stat = np.sum(
                        ((expected_current - expected_baseline) ** 2) / (expected_baseline + 1e-8)
                    )

                    test_results["chi2_statistic"] = chi2_stat
                    chi2_drift = min(1.0, chi2_stat / len(all_categories))
                else:
                    chi2_drift = 0.0
            except Exception:
                chi2_drift = 0.0
        else:
            chi2_drift = 0.0

        # Simple distribution comparison
        baseline_props = baseline_aligned / baseline_aligned.sum()
        current_props = current_aligned / current_aligned.sum()

        prop_diff = np.sum(np.abs(current_props - baseline_props)) / 2  # Total variation distance
        test_results["total_variation_distance"] = prop_diff

        # Missing categories
        missing_in_current = set(baseline_counts.index) - set(current_counts.index)
        new_in_current = set(current_counts.index) - set(baseline_counts.index)

        test_results["missing_categories"] = len(missing_in_current)
        test_results["new_categories"] = len(new_in_current)

        # Category drift score
        category_drift = (len(missing_in_current) + len(new_in_current)) / len(all_categories)

        # Combined categorical drift
        combined_drift = 0.6 * chi2_drift + 0.3 * prop_diff + 0.1 * category_drift

        return min(1.0, combined_drift), test_results

    def _detect_performance_drift(
        self,
        baseline_performance: Dict[str, List[float]],
        current_performance: Dict[str, List[float]],
        model_id: str,
        version: str,
        detection_time: datetime,
    ) -> Optional[DriftAlert]:
        """Detect performance drift using statistical tests."""

        drift_scores = {}
        statistical_tests = {}
        affected_metrics = []

        for metric_name in baseline_performance:
            if metric_name not in current_performance:
                continue

            baseline_values = baseline_performance[metric_name]
            current_values = current_performance[metric_name]

            if len(baseline_values) < 5 or len(current_values) < 5:
                continue

            # T-test for performance change
            if stats:
                try:
                    t_stat, p_value = stats.ttest_ind(baseline_values, current_values)
                    statistical_tests[metric_name] = {"t_statistic": t_stat, "p_value": p_value}

                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(
                        (
                            (len(baseline_values) - 1) * np.var(baseline_values)
                            + (len(current_values) - 1) * np.var(current_values)
                        )
                        / (len(baseline_values) + len(current_values) - 2)
                    )

                    cohens_d = (np.mean(current_values) - np.mean(baseline_values)) / pooled_std
                    statistical_tests[metric_name]["cohens_d"] = cohens_d

                    # Drift score based on effect size and significance
                    if p_value < self.config.significance_level:
                        drift_score = min(1.0, abs(cohens_d) / 2.0)  # Normalize effect size
                    else:
                        drift_score = 0.0

                except Exception:
                    drift_score = 0.0
            else:
                # Simple percentage change
                baseline_mean = np.mean(baseline_values)
                current_mean = np.mean(current_values)

                if baseline_mean != 0:
                    pct_change = abs(current_mean - baseline_mean) / abs(baseline_mean)
                    drift_score = min(1.0, pct_change)
                else:
                    drift_score = 0.0

            drift_scores[metric_name] = drift_score

            if drift_score > self.config.performance_drift_threshold:
                affected_metrics.append(metric_name)

        if not drift_scores:
            return None

        overall_drift = np.mean(list(drift_scores.values()))

        # Determine severity
        if overall_drift > 0.5:
            severity = DriftSeverity.CRITICAL
        elif overall_drift > 0.3:
            severity = DriftSeverity.HIGH
        elif overall_drift > 0.15:
            severity = DriftSeverity.MEDIUM
        else:
            severity = DriftSeverity.LOW

        if severity == DriftSeverity.LOW:
            return None

        return DriftAlert(
            drift_type=DriftType.PERFORMANCE_DRIFT,
            severity=severity,
            detection_time=detection_time,
            affected_features=affected_metrics,
            drift_scores=drift_scores,
            statistical_tests=statistical_tests,
            rollback_recommended=severity == DriftSeverity.CRITICAL,
            rollback_confidence=min(0.9, overall_drift * 1.5),
            rollback_reasons=[f"Performance degradation in {', '.join(affected_metrics)}"],
            baseline_period=(
                detection_time - timedelta(days=self.config.baseline_window_days),
                detection_time - timedelta(days=self.config.detection_window_days),
            ),
            current_period=(
                detection_time - timedelta(days=self.config.detection_window_days),
                detection_time,
            ),
            sample_sizes={
                "baseline": sum(len(v) for v in baseline_performance.values()),
                "current": sum(len(v) for v in current_performance.values()),
            },
            confidence_intervals=self._calculate_confidence_intervals(drift_scores),
            recommended_actions=self._generate_performance_actions(severity, affected_metrics),
            urgency_level=self._calculate_urgency_level(severity, overall_drift),
        )

    def _detect_prediction_drift(
        self,
        baseline_data: pd.DataFrame,
        current_data: pd.DataFrame,
        model_id: str,
        version: str,
        detection_time: datetime,
    ) -> Optional[DriftAlert]:
        """Detect prediction drift by comparing model outputs."""

        try:
            # Load model
            model, model_version = self.model_registry.get_production_model()
            if not model:
                return None

            # Generate predictions on both datasets
            feature_columns = model_version.dataset_info.features

            # Ensure we have the required features
            if not all(col in baseline_data.columns for col in feature_columns):
                return None
            if not all(col in current_data.columns for col in feature_columns):
                return None

            baseline_features = baseline_data[feature_columns].dropna()
            current_features = current_data[feature_columns].dropna()

            if len(baseline_features) < 50 or len(current_features) < 50:
                return None

            # Sample for performance
            baseline_sample = baseline_features.sample(
                min(1000, len(baseline_features)), random_state=42
            )
            current_sample = current_features.sample(
                min(1000, len(current_features)), random_state=42
            )

            # Generate predictions
            baseline_preds = model.predict(baseline_sample)
            current_preds = model.predict(current_sample)

            # Compare prediction distributions
            if hasattr(model, "predict_proba"):
                baseline_proba = model.predict_proba(baseline_sample)
                current_proba = model.predict_proba(current_sample)

                # Compare probability distributions
                if baseline_proba.shape[1] == 2:  # Binary classification
                    baseline_pos_proba = baseline_proba[:, 1]
                    current_pos_proba = current_proba[:, 1]

                    # KS test on probabilities
                    if stats:
                        ks_stat, ks_pvalue = stats.ks_2samp(baseline_pos_proba, current_pos_proba)
                        prob_drift = ks_stat
                    else:
                        prob_drift = abs(np.mean(current_pos_proba) - np.mean(baseline_pos_proba))
                else:
                    prob_drift = 0.0
            else:
                prob_drift = 0.0

            # Compare prediction frequencies
            baseline_pred_freq = pd.Series(baseline_preds).value_counts(normalize=True)
            current_pred_freq = pd.Series(current_preds).value_counts(normalize=True)

            all_labels = set(baseline_pred_freq.index) | set(current_pred_freq.index)
            freq_diff = 0.0

            for label in all_labels:
                baseline_freq = baseline_pred_freq.get(label, 0)
                current_freq = current_pred_freq.get(label, 0)
                freq_diff += abs(current_freq - baseline_freq)

            freq_drift = freq_diff / 2  # Total variation distance

            # Combined prediction drift
            prediction_drift = 0.6 * prob_drift + 0.4 * freq_drift

            drift_scores = {
                "probability_drift": prob_drift,
                "frequency_drift": freq_drift,
                "overall_prediction_drift": prediction_drift,
            }

            statistical_tests = {
                "prediction_stats": {
                    "baseline_mean_prob": np.mean(baseline_pos_proba)
                    if "baseline_pos_proba" in locals()
                    else 0,
                    "current_mean_prob": np.mean(current_pos_proba)
                    if "current_pos_proba" in locals()
                    else 0,
                    "frequency_difference": freq_drift,
                }
            }

            # Determine severity
            if prediction_drift > 0.4:
                severity = DriftSeverity.HIGH
            elif prediction_drift > 0.25:
                severity = DriftSeverity.MEDIUM
            elif prediction_drift > 0.1:
                severity = DriftSeverity.LOW
            else:
                return None

            return DriftAlert(
                drift_type=DriftType.PREDICTION_DRIFT,
                severity=severity,
                detection_time=detection_time,
                affected_features=["model_predictions"],
                drift_scores=drift_scores,
                statistical_tests=statistical_tests,
                rollback_recommended=severity == DriftSeverity.HIGH,
                rollback_confidence=min(0.85, prediction_drift * 2),
                rollback_reasons=["Significant change in model prediction patterns"],
                baseline_period=(baseline_data.index.min(), baseline_data.index.max()),
                current_period=(current_data.index.min(), current_data.index.max()),
                sample_sizes={"baseline": len(baseline_sample), "current": len(current_sample)},
                confidence_intervals=self._calculate_confidence_intervals(drift_scores),
                recommended_actions=self._generate_prediction_actions(severity),
                urgency_level=self._calculate_urgency_level(severity, prediction_drift),
            )

        except Exception as e:
            self.logger.warning(f"Prediction drift detection failed: {e}")
            return None

    def _calculate_psi(self, baseline: pd.Series, current: pd.Series, buckets: int = 10) -> float:
        """Calculate Population Stability Index (PSI)."""

        try:
            # Create bins based on baseline quantiles
            quantiles = np.linspace(0, 1, buckets + 1)
            bins = baseline.quantile(quantiles).unique()

            if len(bins) < 2:
                return 0.0

            # Bin both distributions
            baseline_binned = pd.cut(baseline, bins=bins, include_lowest=True, duplicates="drop")
            current_binned = pd.cut(current, bins=bins, include_lowest=True, duplicates="drop")

            # Calculate proportions
            baseline_props = baseline_binned.value_counts(normalize=True, sort=False)
            current_props = current_binned.value_counts(normalize=True, sort=False)

            # Align and fill missing categories
            baseline_props = baseline_props.reindex(
                baseline_props.index.union(current_props.index), fill_value=0.001
            )
            current_props = current_props.reindex(baseline_props.index, fill_value=0.001)

            # Calculate PSI
            psi = np.sum((current_props - baseline_props) * np.log(current_props / baseline_props))

            return abs(psi)

        except Exception:
            return 0.0

    def _update_feature_weights(self, model_id: str, version: str, data: pd.DataFrame):
        """Update feature importance weights for drift detection."""

        try:
            model_version = self.model_registry.registry[model_id][version]

            # Use stored feature importance if available
            if hasattr(model_version, "feature_importance"):
                importance_dict = model_version.feature_importance
                if importance_dict:
                    # Normalize importance scores
                    total_importance = sum(importance_dict.values())
                    if total_importance > 0:
                        self.feature_weights.update(
                            {
                                feature: importance / total_importance
                                for feature, importance in importance_dict.items()
                            }
                        )
                        return

            # Fallback: uniform weights
            features = [
                col for col in data.columns if col != model_version.dataset_info.target_column
            ]
            uniform_weight = 1.0 / len(features) if features else 1.0
            self.feature_weights.update({feature: uniform_weight for feature in features})

        except Exception as e:
            self.logger.warning(f"Failed to update feature weights: {e}")
            # Use uniform weights as fallback
            uniform_weight = 1.0 / len(data.columns)
            self.feature_weights.update({col: uniform_weight for col in data.columns})

    def _calculate_confidence_intervals(
        self, drift_scores: Dict[str, float]
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for drift scores."""

        confidence_intervals = {}

        for feature, score in drift_scores.items():
            # Simple confidence interval based on score magnitude
            margin = score * 0.1  # 10% margin
            lower_bound = max(0.0, score - margin)
            upper_bound = min(1.0, score + margin)

            confidence_intervals[feature] = (lower_bound, upper_bound)

        return confidence_intervals

    def _generate_rollback_reasons(
        self, drift_score: float, affected_features: List[str]
    ) -> List[str]:
        """Generate specific reasons for rollback recommendation."""

        reasons = []

        if drift_score > 0.7:
            reasons.append("Severe data distribution shift detected")
        elif drift_score > 0.5:
            reasons.append("Significant data drift affecting model reliability")

        if len(affected_features) > len(self.feature_weights) * 0.5:
            reasons.append("Majority of features showing drift")
        elif any(self.feature_weights.get(f, 0) > 0.1 for f in affected_features):
            reasons.append("High-importance features affected by drift")

        return reasons if reasons else ["Data quality concerns"]

    def _generate_drift_actions(
        self, severity: DriftSeverity, affected_features: List[str]
    ) -> List[str]:
        """Generate recommended actions for drift alerts."""

        actions = []

        if severity == DriftSeverity.CRITICAL:
            actions.extend(
                [
                    "Immediately investigate data sources",
                    "Consider emergency model rollback",
                    "Notify data engineering team",
                ]
            )
        elif severity == DriftSeverity.HIGH:
            actions.extend(
                [
                    "Schedule urgent data quality review",
                    "Prepare for potential model retraining",
                    "Increase monitoring frequency",
                ]
            )
        elif severity == DriftSeverity.MEDIUM:
            actions.extend(
                [
                    "Monitor affected features closely",
                    "Investigate data pipeline changes",
                    "Plan model validation tests",
                ]
            )

        if len(affected_features) > 5:
            actions.append("Conduct comprehensive feature analysis")
        else:
            actions.append(f"Focus investigation on: {', '.join(affected_features[:3])}")

        return actions

    def _generate_performance_actions(
        self, severity: DriftSeverity, affected_metrics: List[str]
    ) -> List[str]:
        """Generate recommended actions for performance drift."""

        actions = []

        if severity == DriftSeverity.CRITICAL:
            actions.extend(
                [
                    "Execute immediate model rollback",
                    "Investigate performance degradation cause",
                    "Implement emergency monitoring",
                ]
            )
        elif severity == DriftSeverity.HIGH:
            actions.extend(
                [
                    "Reduce model confidence thresholds",
                    "Increase position size limits",
                    "Schedule model retraining",
                ]
            )

        for metric in affected_metrics:
            actions.append(f"Deep dive analysis on {metric} degradation")

        return actions

    def _generate_prediction_actions(self, severity: DriftSeverity) -> List[str]:
        """Generate recommended actions for prediction drift."""

        actions = [
            "Validate model predictions against recent data",
            "Check for changes in feature preprocessing",
            "Review model calibration",
        ]

        if severity == DriftSeverity.HIGH:
            actions.extend(
                [
                    "Consider prediction confidence adjustments",
                    "Implement prediction monitoring dashboard",
                ]
            )

        return actions

    def _calculate_urgency_level(self, severity: DriftSeverity, drift_score: float) -> int:
        """Calculate urgency level (1-5) based on severity and score."""

        base_urgency = {
            DriftSeverity.LOW: 1,
            DriftSeverity.MEDIUM: 2,
            DriftSeverity.HIGH: 4,
            DriftSeverity.CRITICAL: 5,
        }

        urgency = base_urgency.get(severity, 3)

        # Adjust based on drift score magnitude
        if drift_score > 0.8:
            urgency = min(5, urgency + 1)
        elif drift_score < 0.2:
            urgency = max(1, urgency - 1)

        return urgency

    def _evaluate_rollback_recommendation(self, alert: DriftAlert, model_id: str, version: str):
        """Evaluate and enhance rollback recommendation."""

        # Consider rollback history
        recent_rollbacks = len(
            [
                r
                for r in self.rollback_history
                if (datetime.utcnow() - datetime.fromisoformat(r["timestamp"])).days < 7
            ]
        )

        if recent_rollbacks >= self.config.max_rollback_attempts:
            alert.rollback_recommended = False
            alert.rollback_reasons.append("Maximum rollback attempts reached")
            return

        # Consider model age and stability
        try:
            model_version = self.model_registry.registry[model_id][version]
            model_age_days = (datetime.utcnow() - model_version.created_at).days

            if model_age_days < 1:  # Very new model
                alert.rollback_confidence *= 0.8  # Reduce confidence
                alert.rollback_reasons.append("Model is very new, drift may be normal")

        except Exception:
            pass

        # Final rollback decision
        if (
            alert.rollback_confidence > self.config.auto_rollback_threshold
            and alert.severity == DriftSeverity.CRITICAL
        ):
            alert.rollback_recommended = True

    def _execute_automatic_rollback(self, model_id: str, version: str, alerts: List[DriftAlert]):
        """Execute automatic model rollback."""

        if self.rollback_attempts >= self.config.max_rollback_attempts:
            self.logger.error("Maximum rollback attempts reached, manual intervention required")
            return

        self.logger.critical(f"Executing automatic rollback for {model_id} {version}")

        try:
            # Execute rollback
            rollback_reason = (
                f"Automatic rollback due to critical drift: {[a.drift_type.value for a in alerts]}"
            )
            success = self.model_registry.rollback_model(reason=rollback_reason)

            if success:
                # Record rollback
                rollback_record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_id": model_id,
                    "version": version,
                    "reason": rollback_reason,
                    "alerts": [
                        {"type": a.drift_type.value, "severity": a.severity.value} for a in alerts
                    ],
                    "success": True,
                }

                self.rollback_history.append(rollback_record)
                self.rollback_attempts += 1

                self.logger.info(f"Automatic rollback completed successfully")
            else:
                self.logger.error("Automatic rollback failed")

        except Exception as e:
            self.logger.error(f"Automatic rollback failed with exception: {e}")

    def get_drift_summary(self) -> Dict[str, Any]:
        """Get comprehensive drift detection summary."""

        if not self.drift_history:
            return {"message": "No drift history available"}

        # Aggregate by drift type
        drift_by_type = {}
        for alert in self.drift_history:
            drift_type = alert.drift_type.value
            if drift_type not in drift_by_type:
                drift_by_type[drift_type] = []
            drift_by_type[drift_type].append(alert)

        # Recent alerts
        recent_alerts = [
            a for a in self.drift_history if (datetime.utcnow() - a.detection_time).days < 7
        ]

        # Feature drift frequency
        feature_drift_count = {}
        for alert in self.drift_history:
            for feature in alert.affected_features:
                feature_drift_count[feature] = feature_drift_count.get(feature, 0) + 1

        summary = {
            "total_alerts": len(self.drift_history),
            "recent_alerts": len(recent_alerts),
            "drift_by_type": {
                drift_type: {
                    "count": len(alerts),
                    "avg_severity": np.mean(
                        [
                            4
                            if a.severity == DriftSeverity.CRITICAL
                            else 3
                            if a.severity == DriftSeverity.HIGH
                            else 2
                            if a.severity == DriftSeverity.MEDIUM
                            else 1
                            for a in alerts
                        ]
                    ),
                }
                for drift_type, alerts in drift_by_type.items()
            },
            "most_affected_features": sorted(
                feature_drift_count.items(), key=lambda x: x[1], reverse=True
            )[:10],
            "rollback_stats": {
                "total_rollbacks": len(self.rollback_history),
                "successful_rollbacks": len(
                    [r for r in self.rollback_history if r.get("success", False)]
                ),
                "recent_rollbacks": len(
                    [
                        r
                        for r in self.rollback_history
                        if (datetime.utcnow() - datetime.fromisoformat(r["timestamp"])).days < 7
                    ]
                ),
            },
            "monitoring_status": {
                "continuous_monitoring": self.config.continuous_monitoring,
                "last_check": self.last_check_time.isoformat() if self.last_check_time else None,
                "check_frequency_hours": self.config.check_frequency_hours,
            },
        }

        return summary


def create_drift_detector(
    model_registry: ModelRegistry, config: Optional[DriftConfig] = None
) -> DriftDetector:
    """Create drift detector instance."""
    return DriftDetector(model_registry, config)
