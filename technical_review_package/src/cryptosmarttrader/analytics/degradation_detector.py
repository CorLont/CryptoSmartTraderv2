"""
Degradation Detector

Advanced system for detecting performance degradation with
statistical significance testing and automated remediation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from scipy import stats
from collections import deque

logger = logging.getLogger(__name__)


class DegradationType(Enum):
    """Types of performance degradation"""

    ALPHA_DECAY = "alpha_decay"
    SLIPPAGE_DRIFT = "slippage_drift"
    EXECUTION_QUALITY_DROP = "execution_quality_drop"
    CORRELATION_BREAK = "correlation_break"
    VOLATILITY_ANOMALY = "volatility_anomaly"
    REGIME_MISMATCH = "regime_mismatch"
    FEATURE_DRIFT = "feature_drift"
    MODEL_OVERFIT = "model_overfit"


class DriftSeverity(Enum):
    """Severity levels for drift detection"""

    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class DriftMetrics:
    """Comprehensive drift detection metrics"""

    timestamp: datetime

    # Statistical metrics
    ks_statistic: float  # Kolmogorov-Smirnov test
    ks_p_value: float
    psi_score: float  # Population Stability Index
    js_divergence: float  # Jensen-Shannon divergence

    # Performance metrics
    performance_shift: float  # Performance change in bps
    confidence_interval: Tuple[float, float]
    statistical_significance: bool

    # Feature drift metrics
    feature_drift_scores: Dict[str, float] = field(default_factory=dict)
    most_drifted_feature: Optional[str] = None

    # Context
    baseline_period: str = ""
    comparison_period: str = ""
    sample_size: int = 0


@dataclass
class DegradationAlert:
    """Degradation detection alert"""

    alert_id: str
    timestamp: datetime
    degradation_type: DegradationType
    severity: DriftSeverity

    # Metrics
    drift_metrics: DriftMetrics
    impact_estimate_bps: float
    confidence_score: float

    # Details
    message: str
    affected_components: List[str] = field(default_factory=list)
    root_cause_analysis: str = ""

    # Remediation
    recommended_actions: List[str] = field(default_factory=list)
    auto_remediation_available: bool = False

    # Status
    acknowledged: bool = False
    remediated: bool = False
    false_positive: bool = False


class DegradationDetector:
    """
    Advanced degradation detection system with statistical testing
    """

    def __init__(
        self,
        baseline_window_hours: int = 168,  # 1 week baseline
        detection_window_hours: int = 24,  # 1 day detection window
        significance_level: float = 0.05,  # 5% significance level
        min_sample_size: int = 100,
    ):
        self.baseline_window = timedelta(hours=baseline_window_hours)
        self.detection_window = timedelta(hours=detection_window_hours)
        self.significance_level = significance_level
        self.min_sample_size = min_sample_size

        # Data buffers for continuous monitoring
        self.performance_buffer = deque(maxlen=10080)  # 1 week at 1-minute intervals
        self.feature_buffers = {}

        # Detection thresholds
        self.thresholds = {
            "psi_mild": 0.1,
            "psi_moderate": 0.25,
            "psi_severe": 0.5,
            "performance_shift_bps": 10,
            "js_divergence_threshold": 0.1,
            "correlation_threshold": 0.7,
        }

        # Alert management
        self.degradation_alerts = []
        self.baseline_stats = {}

    def update_performance_data(
        self,
        timestamp: datetime,
        performance_metrics: Dict[str, float],
        feature_values: Dict[str, float] = None,
    ) -> Optional[DegradationAlert]:
        """Update performance data and check for degradation"""
        try:
            # Store performance data
            self.performance_buffer.append((timestamp, performance_metrics))

            # Store feature data if available
            if feature_values:
                for feature, value in feature_values.items():
                    if feature not in self.feature_buffers:
                        self.feature_buffers[feature] = deque(maxlen=10080)
                    self.feature_buffers[feature].append((timestamp, value))

            # Check if we have enough data for detection
            if len(self.performance_buffer) < self.min_sample_size:
                return None

            # Perform degradation detection
            alert = self._detect_degradation(timestamp)

            if alert:
                self.degradation_alerts.append(alert)
                logger.warning(
                    f"Degradation detected: {alert.degradation_type.value} - {alert.severity.value}"
                )

            return alert

        except Exception as e:
            logger.error(f"Performance data update failed: {e}")
            return None

    def _detect_degradation(self, current_time: datetime) -> Optional[DegradationAlert]:
        """Perform comprehensive degradation detection"""
        try:
            # Define time windows
            baseline_start = current_time - self.baseline_window - self.detection_window
            baseline_end = current_time - self.detection_window
            detection_start = current_time - self.detection_window
            detection_end = current_time

            # Extract baseline and current data
            baseline_data = self._extract_window_data(baseline_start, baseline_end)
            current_data = self._extract_window_data(detection_start, detection_end)

            if (
                len(baseline_data) < self.min_sample_size
                or len(current_data) < self.min_sample_size
            ):
                return None

            # Perform statistical tests
            drift_metrics = self._calculate_drift_metrics(baseline_data, current_data)

            # Determine degradation type and severity
            degradation_type, severity = self._classify_degradation(drift_metrics)

            if severity == DriftSeverity.NONE:
                return None

            # Create alert
            alert = self._create_degradation_alert(
                degradation_type, severity, drift_metrics, current_time
            )

            return alert

        except Exception as e:
            logger.error(f"Degradation detection failed: {e}")
            return None

    def _extract_window_data(
        self, start_time: datetime, end_time: datetime
    ) -> List[Dict[str, float]]:
        """Extract performance data for a time window"""
        window_data = []

        for timestamp, metrics in self.performance_buffer:
            if start_time <= timestamp <= end_time:
                window_data.append(metrics)

        return window_data

    def _calculate_drift_metrics(
        self, baseline_data: List[Dict[str, float]], current_data: List[Dict[str, float]]
    ) -> DriftMetrics:
        """Calculate comprehensive drift detection metrics"""
        try:
            # Extract key performance metrics
            baseline_returns = [d.get("return_bps", 0) for d in baseline_data]
            current_returns = [d.get("return_bps", 0) for d in current_data]

            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.ks_2samp(baseline_returns, current_returns)

            # Population Stability Index (PSI)
            psi_score = self._calculate_psi(baseline_returns, current_returns)

            # Jensen-Shannon divergence
            js_div = self._calculate_js_divergence(baseline_returns, current_returns)

            # Performance shift
            baseline_mean = np.mean(baseline_returns)
            current_mean = np.mean(current_returns)
            performance_shift = current_mean - baseline_mean

            # Confidence interval for the shift
            pooled_std = np.sqrt((np.var(baseline_returns) + np.var(current_returns)) / 2)
            margin_of_error = 1.96 * pooled_std / np.sqrt(len(current_returns))
            confidence_interval = (
                performance_shift - margin_of_error,
                performance_shift + margin_of_error,
            )

            # Statistical significance
            t_stat, t_p = stats.ttest_ind(current_returns, baseline_returns)
            statistical_significance = t_p < self.significance_level

            # Feature drift analysis
            feature_drift_scores = {}
            most_drifted_feature = None
            max_drift_score = 0

            # Analyze feature drift if feature data is available
            for feature_name, buffer in self.feature_buffers.items():
                if len(buffer) > self.min_sample_size:
                    # Extract feature values for the time windows
                    baseline_features = []
                    current_features = []

                    baseline_start = datetime.now() - self.baseline_window - self.detection_window
                    baseline_end = datetime.now() - self.detection_window
                    detection_start = datetime.now() - self.detection_window

                    for timestamp, value in buffer:
                        if baseline_start <= timestamp <= baseline_end:
                            baseline_features.append(value)
                        elif timestamp >= detection_start:
                            current_features.append(value)

                    if len(baseline_features) > 10 and len(current_features) > 10:
                        feature_psi = self._calculate_psi(baseline_features, current_features)
                        feature_drift_scores[feature_name] = feature_psi

                        if feature_psi > max_drift_score:
                            max_drift_score = feature_psi
                            most_drifted_feature = feature_name

            return DriftMetrics(
                timestamp=datetime.now(),
                ks_statistic=ks_stat,
                ks_p_value=ks_p,
                psi_score=psi_score,
                js_divergence=js_div,
                performance_shift=performance_shift,
                confidence_interval=confidence_interval,
                statistical_significance=statistical_significance,
                feature_drift_scores=feature_drift_scores,
                most_drifted_feature=most_drifted_feature,
                baseline_period=f"{len(baseline_data)} samples",
                comparison_period=f"{len(current_data)} samples",
                sample_size=len(current_data),
            )

        except Exception as e:
            logger.error(f"Drift metrics calculation failed: {e}")
            return DriftMetrics(
                timestamp=datetime.now(),
                ks_statistic=0,
                ks_p_value=1,
                psi_score=0,
                js_divergence=0,
                performance_shift=0,
                confidence_interval=(0, 0),
                statistical_significance=False,
            )

    def _calculate_psi(self, baseline: List[float], current: List[float]) -> float:
        """Calculate Population Stability Index"""
        try:
            # Create bins based on baseline distribution
            baseline_array = np.array(baseline)
            current_array = np.array(current)

            # Use quantile-based binning
            bins = np.quantile(baseline_array, np.linspace(0, 1, 11))  # 10 bins
            bins[0] -= 1e-6  # Adjust first bin to include minimum
            bins[-1] += 1e-6  # Adjust last bin to include maximum

            # Calculate distributions
            baseline_hist, _ = np.histogram(baseline_array, bins=bins)
            current_hist, _ = np.histogram(current_array, bins=bins)

            # Convert to proportions
            baseline_prop = baseline_hist / len(baseline_array)
            current_prop = current_hist / len(current_array)

            # Avoid division by zero
            baseline_prop = np.where(baseline_prop == 0, 0.0001, baseline_prop)
            current_prop = np.where(current_prop == 0, 0.0001, current_prop)

            # Calculate PSI
            psi = np.sum((current_prop - baseline_prop) * np.log(current_prop / baseline_prop))

            return psi

        except Exception as e:
            logger.error(f"PSI calculation failed: {e}")
            return 0.0

    def _calculate_js_divergence(self, baseline: List[float], current: List[float]) -> float:
        """Calculate Jensen-Shannon divergence"""
        try:
            # Create histograms
            combined = baseline + current
            bins = np.histogram_bin_edges(combined, bins=50)

            baseline_hist, _ = np.histogram(baseline, bins=bins, density=True)
            current_hist, _ = np.histogram(current, bins=bins, density=True)

            # Normalize to create probability distributions
            baseline_prob = baseline_hist / np.sum(baseline_hist)
            current_prob = current_hist / np.sum(current_hist)

            # Avoid log(0)
            baseline_prob = np.where(baseline_prob == 0, 1e-10, baseline_prob)
            current_prob = np.where(current_prob == 0, 1e-10, current_prob)

            # Calculate M (average distribution)
            M = 0.5 * (baseline_prob + current_prob)

            # Calculate KL divergences
            kl_baseline = np.sum(baseline_prob * np.log(baseline_prob / M))
            kl_current = np.sum(current_prob * np.log(current_prob / M))

            # Jensen-Shannon divergence
            js_div = 0.5 * kl_baseline + 0.5 * kl_current

            return js_div

        except Exception as e:
            logger.error(f"JS divergence calculation failed: {e}")
            return 0.0

    def _classify_degradation(
        self, drift_metrics: DriftMetrics
    ) -> Tuple[DegradationType, DriftSeverity]:
        """Classify degradation type and severity"""
        try:
            # Determine primary degradation type
            degradation_type = DegradationType.ALPHA_DECAY  # Default

            # Performance-based classification
            if abs(drift_metrics.performance_shift) > self.thresholds["performance_shift_bps"]:
                if drift_metrics.performance_shift < 0:
                    degradation_type = DegradationType.ALPHA_DECAY
                else:
                    # Unexpected performance improvement might indicate overfitting
                    degradation_type = DegradationType.MODEL_OVERFIT

            # Feature drift classification
            if drift_metrics.most_drifted_feature and drift_metrics.feature_drift_scores:
                max_feature_drift = max(drift_metrics.feature_drift_scores.values())
                if max_feature_drift > 0.25:
                    degradation_type = DegradationType.FEATURE_DRIFT

            # Distribution-based classification
            if drift_metrics.js_divergence > self.thresholds["js_divergence_threshold"]:
                if drift_metrics.psi_score > 0.5:
                    degradation_type = DegradationType.REGIME_MISMATCH
                else:
                    degradation_type = DegradationType.VOLATILITY_ANOMALY

            # Determine severity based on PSI score
            psi = drift_metrics.psi_score

            if psi < self.thresholds["psi_mild"]:
                severity = DriftSeverity.NONE
            elif psi < self.thresholds["psi_moderate"]:
                severity = DriftSeverity.MILD
            elif psi < self.thresholds["psi_severe"]:
                severity = DriftSeverity.MODERATE
            else:
                severity = DriftSeverity.SEVERE

            # Upgrade severity if statistically significant and large impact
            if drift_metrics.statistical_significance and abs(drift_metrics.performance_shift) > 20:
                if severity == DriftSeverity.SEVERE:
                    severity = DriftSeverity.CRITICAL
                elif severity in [DriftSeverity.MILD, DriftSeverity.MODERATE]:
                    severity = DriftSeverity.SEVERE

            return degradation_type, severity

        except Exception as e:
            logger.error(f"Degradation classification failed: {e}")
            return DegradationType.ALPHA_DECAY, DriftSeverity.MILD

    def _create_degradation_alert(
        self,
        degradation_type: DegradationType,
        severity: DriftSeverity,
        drift_metrics: DriftMetrics,
        timestamp: datetime,
    ) -> DegradationAlert:
        """Create comprehensive degradation alert"""

        # Generate alert ID
        alert_id = f"{degradation_type.value}_{int(timestamp.timestamp())}"

        # Calculate impact estimate
        impact_estimate = abs(drift_metrics.performance_shift)

        # Calculate confidence score
        confidence_score = min(
            1.0,
            (1 - drift_metrics.ks_p_value) * 0.5
            + min(drift_metrics.psi_score / 0.5, 1.0) * 0.3
            + min(drift_metrics.js_divergence / 0.2, 1.0) * 0.2,
        )

        # Generate message
        message = self._generate_alert_message(degradation_type, severity, drift_metrics)

        # Root cause analysis
        root_cause = self._analyze_root_cause(degradation_type, drift_metrics)

        # Recommended actions
        recommendations = self._get_remediation_recommendations(degradation_type, severity)

        # Auto-remediation availability
        auto_remediation = degradation_type in [
            DegradationType.SLIPPAGE_DRIFT,
            DegradationType.EXECUTION_QUALITY_DROP,
        ] and severity in [DriftSeverity.MILD, DriftSeverity.MODERATE]

        return DegradationAlert(
            alert_id=alert_id,
            timestamp=timestamp,
            degradation_type=degradation_type,
            severity=severity,
            drift_metrics=drift_metrics,
            impact_estimate_bps=impact_estimate,
            confidence_score=confidence_score,
            message=message,
            affected_components=self._identify_affected_components(degradation_type),
            root_cause_analysis=root_cause,
            recommended_actions=recommendations,
            auto_remediation_available=auto_remediation,
        )

    def _generate_alert_message(
        self,
        degradation_type: DegradationType,
        severity: DriftSeverity,
        drift_metrics: DriftMetrics,
    ) -> str:
        """Generate human-readable alert message"""

        severity_text = severity.value.upper()
        shift_text = f"{drift_metrics.performance_shift:+.1f} bps"
        psi_text = f"PSI: {drift_metrics.psi_score:.3f}"

        base_messages = {
            DegradationType.ALPHA_DECAY: f"Alpha generation degraded by {abs(drift_metrics.performance_shift):.1f} bps",
            DegradationType.SLIPPAGE_DRIFT: f"Execution slippage increased significantly",
            DegradationType.EXECUTION_QUALITY_DROP: f"Execution quality declined",
            DegradationType.FEATURE_DRIFT: f"Feature drift detected in {drift_metrics.most_drifted_feature}",
            DegradationType.REGIME_MISMATCH: f"Market regime mismatch detected",
            DegradationType.VOLATILITY_ANOMALY: f"Volatility pattern anomaly detected",
            DegradationType.MODEL_OVERFIT: f"Potential model overfitting detected",
            DegradationType.CORRELATION_BREAK: f"Correlation structure breakdown detected",
        }

        base_message = base_messages.get(degradation_type, "Performance degradation detected")

        return f"[{severity_text}] {base_message} ({shift_text}, {psi_text})"

    def _analyze_root_cause(
        self, degradation_type: DegradationType, drift_metrics: DriftMetrics
    ) -> str:
        """Perform automated root cause analysis"""

        root_causes = {
            DegradationType.ALPHA_DECAY: "Model predictions losing edge due to market evolution or feature staleness",
            DegradationType.SLIPPAGE_DRIFT: "Execution conditions deteriorated - check liquidity and timing",
            DegradationType.EXECUTION_QUALITY_DROP: "Order routing or execution algorithm performance declined",
            DegradationType.FEATURE_DRIFT: f"Input feature '{drift_metrics.most_drifted_feature}' distribution changed significantly",
            DegradationType.REGIME_MISMATCH: "Market regime changed - model trained on different conditions",
            DegradationType.VOLATILITY_ANOMALY: "Volatility patterns outside normal ranges - check risk management",
            DegradationType.MODEL_OVERFIT: "Model showing signs of overfitting - validation performance diverging",
            DegradationType.CORRELATION_BREAK: "Asset correlations changed - portfolio assumptions invalid",
        }

        base_cause = root_causes.get(degradation_type, "Unknown degradation cause")

        # Add statistical context
        if drift_metrics.statistical_significance:
            base_cause += f" (Statistically significant with p={drift_metrics.ks_p_value:.4f})"

        return base_cause

    def _get_remediation_recommendations(
        self, degradation_type: DegradationType, severity: DriftSeverity
    ) -> List[str]:
        """Get automated remediation recommendations"""

        base_recommendations = {
            DegradationType.ALPHA_DECAY: [
                "Retrain model with recent data",
                "Review feature engineering pipeline",
                "Check for regime changes in market conditions",
                "Consider ensemble methods or model averaging",
            ],
            DegradationType.SLIPPAGE_DRIFT: [
                "Review execution timing and order routing",
                "Check liquidity provider relationships",
                "Analyze market microstructure changes",
                "Consider order size optimization",
            ],
            DegradationType.EXECUTION_QUALITY_DROP: [
                "Audit execution algorithms and parameters",
                "Check system latency and connectivity",
                "Review order type selection logic",
                "Consider execution venue diversification",
            ],
            DegradationType.FEATURE_DRIFT: [
                "Investigate data source changes",
                "Update feature calculation methods",
                "Implement adaptive feature scaling",
                "Consider feature selection review",
            ],
            DegradationType.REGIME_MISMATCH: [
                "Update regime detection parameters",
                "Retrain models with regime-aware features",
                "Implement adaptive model switching",
                "Review risk management parameters",
            ],
        }

        recommendations = base_recommendations.get(
            degradation_type,
            [
                "Investigate performance degradation",
                "Review model and execution parameters",
                "Consider system diagnostic checks",
            ],
        )

        # Add severity-specific recommendations
        if severity in [DriftSeverity.SEVERE, DriftSeverity.CRITICAL]:
            recommendations.insert(0, "Consider temporary trading halt for investigation")
            recommendations.append("Implement emergency risk reduction protocols")

        return recommendations

    def _identify_affected_components(self, degradation_type: DegradationType) -> List[str]:
        """Identify system components affected by degradation"""

        component_mapping = {
            DegradationType.ALPHA_DECAY: [
                "prediction_models",
                "feature_pipeline",
                "signal_generation",
            ],
            DegradationType.SLIPPAGE_DRIFT: [
                "execution_engine",
                "order_routing",
                "market_interface",
            ],
            DegradationType.EXECUTION_QUALITY_DROP: [
                "execution_engine",
                "order_management",
                "venue_selection",
            ],
            DegradationType.FEATURE_DRIFT: [
                "data_pipeline",
                "feature_engineering",
                "data_validation",
            ],
            DegradationType.REGIME_MISMATCH: [
                "regime_detection",
                "model_selection",
                "risk_management",
            ],
            DegradationType.VOLATILITY_ANOMALY: [
                "risk_management",
                "position_sizing",
                "volatility_models",
            ],
            DegradationType.MODEL_OVERFIT: [
                "model_training",
                "validation_system",
                "hyperparameter_tuning",
            ],
            DegradationType.CORRELATION_BREAK: [
                "portfolio_optimization",
                "risk_models",
                "correlation_tracking",
            ],
        }

        return component_mapping.get(degradation_type, ["unknown_component"])

    def get_degradation_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """Get degradation detection summary"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days_back)
            recent_alerts = [
                alert for alert in self.degradation_alerts if alert.timestamp >= cutoff_time
            ]

            if not recent_alerts:
                return {"status": "no_degradation", "period_days": days_back}

            # Analyze alert patterns
            degradation_frequency = {}
            severity_distribution = {}

            for alert in recent_alerts:
                deg_type = alert.degradation_type.value
                degradation_frequency[deg_type] = degradation_frequency.get(deg_type, 0) + 1

                severity = alert.severity.value
                severity_distribution[severity] = severity_distribution.get(severity, 0) + 1

            # Calculate average impact
            avg_impact = np.mean([alert.impact_estimate_bps for alert in recent_alerts])

            # Calculate remediation rate
            remediated_count = len([alert for alert in recent_alerts if alert.remediated])
            remediation_rate = remediated_count / len(recent_alerts) if recent_alerts else 0

            return {
                "period_days": days_back,
                "total_alerts": len(recent_alerts),
                "degradation_frequency": degradation_frequency,
                "severity_distribution": severity_distribution,
                "average_impact_bps": avg_impact,
                "remediation_rate": remediation_rate,
                "top_degradation_types": sorted(
                    degradation_frequency.items(), key=lambda x: x[1], reverse=True
                )[:3],
                "current_status": self._get_current_status(),
            }

        except Exception as e:
            logger.error(f"Degradation summary failed: {e}")
            return {"status": "error", "error": str(e)}

    def _get_current_status(self) -> Dict[str, Any]:
        """Get current system degradation status"""
        active_alerts = [
            alert
            for alert in self.degradation_alerts
            if not alert.remediated and not alert.false_positive
        ]

        if not active_alerts:
            return {"status": "healthy", "active_alerts": 0}

        # Determine overall status based on worst active alert
        worst_severity = max(alert.severity for alert in active_alerts)

        status_mapping = {
            DriftSeverity.MILD: "monitoring",
            DriftSeverity.MODERATE: "degraded",
            DriftSeverity.SEVERE: "impaired",
            DriftSeverity.CRITICAL: "critical",
        }

        return {
            "status": status_mapping.get(worst_severity, "unknown"),
            "active_alerts": len(active_alerts),
            "worst_severity": worst_severity.value,
            "requires_attention": worst_severity in [DriftSeverity.SEVERE, DriftSeverity.CRITICAL],
        }
