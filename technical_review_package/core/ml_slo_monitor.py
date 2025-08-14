#!/usr/bin/env python3
"""
ML SLO (Service Level Objective) Monitor
Implements formal SLO monitoring with automatic retraining triggers and rollback capabilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from core.logging_manager import get_logger


class SLOStatus(str, Enum):
    """SLO compliance status"""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    BREACH = "breach"


class ActionType(str, Enum):
    """Available remediation actions"""

    MONITOR = "monitor"
    RETRAIN = "retrain"
    ROLLBACK = "rollback"
    ALERT = "alert"
    DISABLE_MODEL = "disable_model"


@dataclass
class SLOMetric:
    """Individual SLO metric definition"""

    name: str
    description: str
    target_value: float
    warning_threshold: float
    critical_threshold: float
    measurement_window_hours: int
    minimum_samples: int
    metric_type: str  # 'lower_is_better', 'higher_is_better', 'range'
    enabled: bool = True


@dataclass
class SLOViolation:
    """SLO violation record"""

    metric_name: str
    current_value: float
    target_value: float
    threshold_breached: str  # 'warning', 'critical', 'breach'
    severity: float
    violation_duration: timedelta
    samples_count: int
    timestamp: datetime
    horizon: str
    model_version: Optional[str] = None


@dataclass
class PerformanceSnapshot:
    """Performance measurement snapshot"""

    timestamp: datetime
    horizon: str
    model_version: str
    metrics: Dict[str, float]
    sample_count: int
    predictions: List[float]
    actuals: List[float]


class MLSLODefinitions:
    """Standard SLO definitions for ML models"""

    @staticmethod
    def get_default_slos() -> Dict[str, SLOMetric]:
        """Get default SLO metrics for crypto trading models"""

        return {
            # Accuracy metrics
            "mae_1h": SLOMetric(
                name="mae_1h",
                description="Mean Absolute Error for 1-hour predictions",
                target_value=0.02,  # 2% error target
                warning_threshold=0.025,  # 2.5% warning
                critical_threshold=0.035,  # 3.5% critical
                measurement_window_hours=24,
                minimum_samples=50,
                metric_type="lower_is_better",
            ),
            "mae_24h": SLOMetric(
                name="mae_24h",
                description="Mean Absolute Error for 24-hour predictions",
                target_value=0.05,  # 5% error target
                warning_threshold=0.07,  # 7% warning
                critical_threshold=0.10,  # 10% critical
                measurement_window_hours=168,  # 7 days
                minimum_samples=30,
                metric_type="lower_is_better",
            ),
            "mape_1h": SLOMetric(
                name="mape_1h",
                description="Mean Absolute Percentage Error for 1-hour predictions",
                target_value=3.0,  # 3% MAPE target
                warning_threshold=4.0,  # 4% warning
                critical_threshold=6.0,  # 6% critical
                measurement_window_hours=24,
                minimum_samples=50,
                metric_type="lower_is_better",
            ),
            "mape_24h": SLOMetric(
                name="mape_24h",
                description="Mean Absolute Percentage Error for 24-hour predictions",
                target_value=8.0,  # 8% MAPE target
                warning_threshold=12.0,  # 12% warning
                critical_threshold=18.0,  # 18% critical
                measurement_window_hours=168,
                minimum_samples=30,
                metric_type="lower_is_better",
            ),
            # Directional accuracy
            "directional_accuracy_1h": SLOMetric(
                name="directional_accuracy_1h",
                description="Directional accuracy for 1-hour predictions",
                target_value=0.65,  # 65% directional accuracy
                warning_threshold=0.60,  # 60% warning
                critical_threshold=0.55,  # 55% critical
                measurement_window_hours=24,
                minimum_samples=50,
                metric_type="higher_is_better",
            ),
            "directional_accuracy_24h": SLOMetric(
                name="directional_accuracy_24h",
                description="Directional accuracy for 24-hour predictions",
                target_value=0.60,  # 60% directional accuracy
                warning_threshold=0.55,  # 55% warning
                critical_threshold=0.50,  # 50% critical
                measurement_window_hours=168,
                minimum_samples=30,
                metric_type="higher_is_better",
            ),
            # Precision at top-K (for ranking predictions)
            "precision_at_10": SLOMetric(
                name="precision_at_10",
                description="Precision in top 10 predicted performers",
                target_value=0.40,  # 40% of top 10 should be actual top performers
                warning_threshold=0.30,  # 30% warning
                critical_threshold=0.20,  # 20% critical
                measurement_window_hours=168,
                minimum_samples=10,
                metric_type="higher_is_better",
            ),
            "precision_at_50": SLOMetric(
                name="precision_at_50",
                description="Precision in top 50 predicted performers",
                target_value=0.25,  # 25% of top 50 should be actual top performers
                warning_threshold=0.18,  # 18% warning
                critical_threshold=0.12,  # 12% critical
                measurement_window_hours=168,
                minimum_samples=20,
                metric_type="higher_is_better",
            ),
            # Coverage and reliability
            "prediction_coverage_95": SLOMetric(
                name="prediction_coverage_95",
                description="95% prediction interval coverage",
                target_value=0.95,  # 95% coverage target
                warning_threshold=0.90,  # 90% warning
                critical_threshold=0.85,  # 85% critical
                measurement_window_hours=168,
                minimum_samples=50,
                metric_type="higher_is_better",
            ),
            "model_confidence": SLOMetric(
                name="model_confidence",
                description="Average model confidence score",
                target_value=0.70,  # 70% confidence target
                warning_threshold=0.60,  # 60% warning
                critical_threshold=0.40,  # 40% critical
                measurement_window_hours=24,
                minimum_samples=20,
                metric_type="higher_is_better",
            ),
            # Latency metrics
            "prediction_latency_p95": SLOMetric(
                name="prediction_latency_p95",
                description="95th percentile prediction latency (ms)",
                target_value=500.0,  # 500ms target
                warning_threshold=1000.0,  # 1s warning
                critical_threshold=2000.0,  # 2s critical
                measurement_window_hours=1,
                minimum_samples=10,
                metric_type="lower_is_better",
            ),
            # Data quality
            "feature_drift_score": SLOMetric(
                name="feature_drift_score",
                description="Average feature drift score",
                target_value=0.10,  # 10% drift target
                warning_threshold=0.20,  # 20% warning
                critical_threshold=0.35,  # 35% critical
                measurement_window_hours=24,
                minimum_samples=5,
                metric_type="lower_is_better",
            ),
        }


class SLOMonitor:
    """Main SLO monitoring and enforcement engine"""

    def __init__(self):
        self.logger = get_logger()

        # SLO definitions
        self.slo_metrics = MLSLODefinitions.get_default_slos()

        # Performance tracking
        self.performance_history: List[PerformanceSnapshot] = []
        self.violation_history: List[SLOViolation] = []

        # Current status
        self.current_slo_status = {}
        self.last_evaluation = None

        # Configuration
        self.evaluation_interval_minutes = 15  # Check SLOs every 15 minutes
        self.max_history_days = 30
        self.auto_remediation_enabled = True

        # Remediation state
        self.recent_actions = []
        self.model_versions = {}  # Track model versions for rollback

        self.logger.info(
            "SLO Monitor initialized",
            extra={
                "slo_metrics_count": len(self.slo_metrics),
                "auto_remediation": self.auto_remediation_enabled,
                "evaluation_interval_minutes": self.evaluation_interval_minutes,
            },
        )

    def record_performance(
        self,
        horizon: str,
        model_version: str,
        predictions: List[float],
        actuals: List[float],
        latencies: Optional[List[float]] = None,
        confidence_scores: Optional[List[float]] = None,
        prediction_intervals: Optional[List[Tuple[float, float]]] = None,
    ):
        """Record model performance for SLO evaluation"""

        if len(predictions) != len(actuals) or len(predictions) == 0:
            self.logger.warning("Invalid performance data provided")
            return

        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(
            predictions, actuals, latencies, confidence_scores, prediction_intervals
        )

        # Create performance snapshot
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now(),
            horizon=horizon,
            model_version=model_version,
            metrics=metrics,
            sample_count=len(predictions),
            predictions=predictions[:100],  # Store limited history
            actuals=actuals[:100],
        )

        # Store performance
        self.performance_history.append(snapshot)

        # Cleanup old performance data
        cutoff_date = datetime.now() - timedelta(days=self.max_history_days)
        self.performance_history = [
            s for s in self.performance_history if s.timestamp > cutoff_date
        ]

        self.logger.info(
            f"Performance recorded for {horizon}",
            extra={
                "horizon": horizon,
                "model_version": model_version,
                "sample_count": len(predictions),
                "metrics": {k: round(v, 4) for k, v in metrics.items()},
            },
        )

        # Trigger SLO evaluation
        self.evaluate_slos(horizon)

    def _calculate_performance_metrics(
        self,
        predictions: List[float],
        actuals: List[float],
        latencies: Optional[List[float]] = None,
        confidence_scores: Optional[List[float]] = None,
        prediction_intervals: Optional[List[Tuple[float, float]]] = None,
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""

        pred_array = np.array(predictions)
        actual_array = np.array(actuals)

        metrics = {}

        # Basic accuracy metrics
        mae = np.mean(np.abs(pred_array - actual_array))
        mse = np.mean((pred_array - actual_array) ** 2)
        rmse = np.sqrt(mse)

        # MAPE (with protection against division by zero)
        mape = np.mean(np.abs((actual_array - pred_array) / (actual_array + 1e-10))) * 100

        metrics.update({"mae": mae, "mse": mse, "rmse": rmse, "mape": mape})

        # Directional accuracy
        pred_direction = np.sign(pred_array)
        actual_direction = np.sign(actual_array)
        directional_accuracy = np.mean(pred_direction == actual_direction)
        metrics["directional_accuracy"] = directional_accuracy

        # Correlation
        if len(predictions) > 2:
            correlation = np.corrcoef(pred_array, actual_array)[0, 1]
            metrics["correlation"] = correlation if not np.isnan(correlation) else 0.0

        # Precision at K (simplified)
        if len(predictions) >= 10:
            # Sort by predictions, check if top K match actual top K
            pred_rankings = np.argsort(pred_array)[::-1]  # Descending order
            actual_rankings = np.argsort(actual_array)[::-1]

            # Top 10 precision
            top_10_pred = set(pred_rankings[:10])
            top_10_actual = set(actual_rankings[:10])
            precision_at_10 = len(top_10_pred.intersection(top_10_actual)) / 10.0
            metrics["precision_at_10"] = precision_at_10

            # Top 50 precision (if enough samples)
            if len(predictions) >= 50:
                top_50_pred = set(pred_rankings[:50])
                top_50_actual = set(actual_rankings[:50])
                precision_at_50 = len(top_50_pred.intersection(top_50_actual)) / 50.0
                metrics["precision_at_50"] = precision_at_50

        # Confidence metrics
        if confidence_scores:
            metrics["model_confidence"] = np.mean(confidence_scores)

        # Coverage metrics
        if prediction_intervals:
            covered = 0
            for i, (lower, upper) in enumerate(prediction_intervals):
                if i < len(actuals) and lower <= actuals[i] <= upper:
                    covered += 1
            coverage = covered / len(prediction_intervals)
            metrics["prediction_coverage_95"] = coverage

        # Latency metrics
        if latencies:
            metrics["prediction_latency_p95"] = np.percentile(latencies, 95)
            metrics["prediction_latency_mean"] = np.mean(latencies)

        return metrics

    def evaluate_slos(self, horizon: str = None) -> Dict[str, SLOStatus]:
        """Evaluate SLO compliance and trigger remediation if needed"""

        evaluation_time = datetime.now()
        horizon_filter = horizon

        slo_results = {}
        violations = []

        for metric_name, slo_metric in self.slo_metrics.items():
            if not slo_metric.enabled:
                continue

            # Filter by horizon if metric is horizon-specific
            if horizon_filter and (metric_name.endswith("_1h") or metric_name.endswith("_24h")):
                metric_horizon = "1h" if metric_name.endswith("_1h") else "24h"
                if metric_horizon != horizon_filter:
                    continue

            # Get relevant performance data
            window_start = evaluation_time - timedelta(hours=slo_metric.measurement_window_hours)
            relevant_snapshots = [
                s for s in self.performance_history if s.timestamp >= window_start
            ]

            # Filter by horizon if specified
            if metric_name.endswith("_1h"):
                relevant_snapshots = [s for s in relevant_snapshots if s.horizon == "1h"]
            elif metric_name.endswith("_24h"):
                relevant_snapshots = [s for s in relevant_snapshots if s.horizon == "24h"]

            # Check if we have enough samples
            if len(relevant_snapshots) < slo_metric.minimum_samples:
                slo_results[metric_name] = SLOStatus.WARNING
                self.logger.warning(
                    f"Insufficient samples for SLO {metric_name}",
                    extra={
                        "metric": metric_name,
                        "samples": len(relevant_snapshots),
                        "required": slo_metric.minimum_samples,
                    },
                )
                continue

            # Extract metric values
            metric_values = []
            for snapshot in relevant_snapshots:
                # Map SLO metric name to snapshot metric name
                snapshot_metric_name = self._map_slo_to_snapshot_metric(metric_name)
                if snapshot_metric_name in snapshot.metrics:
                    metric_values.append(snapshot.metrics[snapshot_metric_name])

            if not metric_values:
                continue

            # Calculate current metric value (typically mean)
            current_value = np.mean(metric_values)

            # Evaluate SLO compliance
            status = self._evaluate_metric_compliance(slo_metric, current_value)
            slo_results[metric_name] = status

            # Record violations
            if status in [SLOStatus.CRITICAL, SLOStatus.BREACH]:
                violation = SLOViolation(
                    metric_name=metric_name,
                    current_value=current_value,
                    target_value=slo_metric.target_value,
                    threshold_breached=status.value,
                    severity=self._calculate_violation_severity(slo_metric, current_value),
                    violation_duration=timedelta(hours=slo_metric.measurement_window_hours),
                    samples_count=len(metric_values),
                    timestamp=evaluation_time,
                    horizon=horizon_filter or "all",
                )
                violations.append(violation)

        # Store results
        self.current_slo_status = slo_results
        self.last_evaluation = evaluation_time
        self.violation_history.extend(violations)

        # Trigger remediation for violations
        if violations and self.auto_remediation_enabled:
            self._trigger_remediation(violations, horizon_filter)

        # Log SLO evaluation results
        self.logger.info(
            f"SLO evaluation completed for {horizon_filter or 'all horizons'}",
            extra={
                "horizon": horizon_filter,
                "total_metrics": len(slo_results),
                "healthy": len([s for s in slo_results.values() if s == SLOStatus.HEALTHY]),
                "warning": len([s for s in slo_results.values() if s == SLOStatus.WARNING]),
                "critical": len([s for s in slo_results.values() if s == SLOStatus.CRITICAL]),
                "breach": len([s for s in slo_results.values() if s == SLOStatus.BREACH]),
                "violations": len(violations),
            },
        )

        return slo_results

    def _map_slo_to_snapshot_metric(self, slo_metric_name: str) -> str:
        """Map SLO metric names to performance snapshot metric names"""

        mapping = {
            "mae_1h": "mae",
            "mae_24h": "mae",
            "mape_1h": "mape",
            "mape_24h": "mape",
            "directional_accuracy_1h": "directional_accuracy",
            "directional_accuracy_24h": "directional_accuracy",
            "precision_at_10": "precision_at_10",
            "precision_at_50": "precision_at_50",
            "prediction_coverage_95": "prediction_coverage_95",
            "model_confidence": "model_confidence",
            "prediction_latency_p95": "prediction_latency_p95",
        }

        return mapping.get(slo_metric_name, slo_metric_name)

    def _evaluate_metric_compliance(self, slo_metric: SLOMetric, current_value: float) -> SLOStatus:
        """Evaluate individual metric compliance"""

        if slo_metric.metric_type == "lower_is_better":
            if current_value <= slo_metric.target_value:
                return SLOStatus.HEALTHY
            elif current_value <= slo_metric.warning_threshold:
                return SLOStatus.WARNING
            elif current_value <= slo_metric.critical_threshold:
                return SLOStatus.CRITICAL
            else:
                return SLOStatus.BREACH

        elif slo_metric.metric_type == "higher_is_better":
            if current_value >= slo_metric.target_value:
                return SLOStatus.HEALTHY
            elif current_value >= slo_metric.warning_threshold:
                return SLOStatus.WARNING
            elif current_value >= slo_metric.critical_threshold:
                return SLOStatus.CRITICAL
            else:
                return SLOStatus.BREACH

        return SLOStatus.WARNING  # Default fallback

    def _calculate_violation_severity(self, slo_metric: SLOMetric, current_value: float) -> float:
        """Calculate violation severity score (0.0 to 1.0)"""

        if slo_metric.metric_type == "lower_is_better":
            if current_value <= slo_metric.critical_threshold:
                return 0.5  # Warning level
            else:
                # Scale beyond critical threshold
                max_expected = slo_metric.critical_threshold * 2
                severity = min(
                    1.0,
                    (current_value - slo_metric.critical_threshold)
                    / (max_expected - slo_metric.critical_threshold),
                )
                return 0.5 + severity * 0.5

        elif slo_metric.metric_type == "higher_is_better":
            if current_value >= slo_metric.critical_threshold:
                return 0.5  # Warning level
            else:
                # Scale below critical threshold
                min_expected = slo_metric.critical_threshold * 0.5
                severity = min(
                    1.0,
                    (slo_metric.critical_threshold - current_value)
                    / (slo_metric.critical_threshold - min_expected),
                )
                return 0.5 + severity * 0.5

        return 0.5

    def _trigger_remediation(self, violations: List[SLOViolation], horizon: Optional[str]):
        """Trigger appropriate remediation actions for violations"""

        # Group violations by severity
        critical_violations = [v for v in violations if v.severity > 0.8]
        moderate_violations = [v for v in violations if 0.5 < v.severity <= 0.8]

        actions_taken = []

        # Handle critical violations
        if critical_violations:
            # Check for retraining trigger conditions
            accuracy_violations = [
                v
                for v in critical_violations
                if v.metric_name in ["mae_1h", "mae_24h", "mape_1h", "mape_24h"]
            ]

            if accuracy_violations:
                action = self._trigger_retraining(horizon, "critical_accuracy_violation")
                if action:
                    actions_taken.append(action)

            # Check for rollback conditions
            severe_violations = [v for v in critical_violations if v.severity > 0.9]
            if len(severe_violations) >= 2:  # Multiple severe violations
                action = self._trigger_rollback(horizon, "multiple_severe_violations")
                if action:
                    actions_taken.append(action)

        # Handle moderate violations
        if moderate_violations and not actions_taken:
            # Trigger monitoring/alerting
            action = self._trigger_monitoring_alert(moderate_violations, horizon)
            if action:
                actions_taken.append(action)

        # Record actions
        self.recent_actions.extend(actions_taken)

        # Keep recent actions history manageable
        if len(self.recent_actions) > 100:
            self.recent_actions = self.recent_actions[-100:]

        if actions_taken:
            self.logger.warning(
                f"Remediation actions triggered for {len(violations)} violations",
                extra={
                    "violations": len(violations),
                    "critical_violations": len(critical_violations),
                    "actions_taken": len(actions_taken),
                    "action_types": [a["type"] for a in actions_taken],
                },
            )

    def _trigger_retraining(self, horizon: Optional[str], reason: str) -> Optional[Dict[str, Any]]:
        """Trigger model retraining"""

        # Check if retraining was already triggered recently
        recent_retrains = [
            a
            for a in self.recent_actions
            if a.get("type") == ActionType.RETRAIN.value
            and (datetime.now() - datetime.fromisoformat(a["timestamp"])).total_seconds()
            < 3600  # 1 hour
        ]

        if recent_retrains:
            self.logger.info("Retraining recently triggered, skipping")
            return None

        action = {
            "type": ActionType.RETRAIN.value,
            "reason": reason,
            "horizon": horizon,
            "timestamp": datetime.now().isoformat(),
            "status": "triggered",
        }

        self.logger.critical(
            f"RETRAINING TRIGGERED: {reason}",
            extra={"reason": reason, "horizon": horizon, "action": "model_retrain_required"},
        )

        return action

    def _trigger_rollback(self, horizon: Optional[str], reason: str) -> Optional[Dict[str, Any]]:
        """Trigger model rollback to previous version"""

        # Check if rollback is possible
        if horizon not in self.model_versions or len(self.model_versions[horizon]) < 2:
            self.logger.warning(f"Cannot rollback {horizon} - insufficient model versions")
            return None

        action = {
            "type": ActionType.ROLLBACK.value,
            "reason": reason,
            "horizon": horizon,
            "timestamp": datetime.now().isoformat(),
            "status": "triggered",
            "rollback_to": self.model_versions[horizon][-2],  # Previous version
        }

        self.logger.critical(
            f"MODEL ROLLBACK TRIGGERED: {reason}",
            extra={
                "reason": reason,
                "horizon": horizon,
                "action": "model_rollback_required",
                "rollback_to": action["rollback_to"],
            },
        )

        return action

    def _trigger_monitoring_alert(
        self, violations: List[SLOViolation], horizon: Optional[str]
    ) -> Dict[str, Any]:
        """Trigger monitoring alert for moderate violations"""

        action = {
            "type": ActionType.ALERT.value,
            "reason": "moderate_slo_violations",
            "horizon": horizon,
            "timestamp": datetime.now().isoformat(),
            "violations": [
                {
                    "metric": v.metric_name,
                    "severity": v.severity,
                    "current_value": v.current_value,
                    "target_value": v.target_value,
                }
                for v in violations
            ],
        }

        self.logger.warning(
            f"SLO monitoring alert: {len(violations)} moderate violations",
            extra={
                "violations_count": len(violations),
                "horizon": horizon,
                "action": "monitoring_alert",
            },
        )

        return action

    def register_model_version(self, horizon: str, model_version: str):
        """Register new model version for rollback capability"""

        if horizon not in self.model_versions:
            self.model_versions[horizon] = []

        self.model_versions[horizon].append(model_version)

        # Keep last 10 versions
        if len(self.model_versions[horizon]) > 10:
            self.model_versions[horizon] = self.model_versions[horizon][-10:]

        self.logger.info(
            f"Model version registered for {horizon}",
            extra={
                "horizon": horizon,
                "model_version": model_version,
                "total_versions": len(self.model_versions[horizon]),
            },
        )

    def get_slo_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive SLO status for dashboard"""

        dashboard_data = {
            "timestamp": datetime.now().isoformat(),
            "overall_health": self._calculate_overall_health(),
            "slo_status": {},
            "recent_violations": [],
            "performance_trend": {},
            "remediation_actions": self.recent_actions[-10:],  # Last 10 actions
            "coverage": {},
        }

        # Current SLO status
        for metric_name, status in self.current_slo_status.items():
            slo_metric = self.slo_metrics[metric_name]
            dashboard_data["slo_status"][metric_name] = {
                "status": status.value,
                "description": slo_metric.description,
                "target": slo_metric.target_value,
                "warning_threshold": slo_metric.warning_threshold,
                "critical_threshold": slo_metric.critical_threshold,
            }

        # Recent violations
        recent_violations = sorted(
            self.violation_history[-20:], key=lambda x: x.timestamp, reverse=True
        )

        dashboard_data["recent_violations"] = [
            {
                "metric": v.metric_name,
                "severity": v.severity,
                "timestamp": v.timestamp.isoformat(),
                "current_value": v.current_value,
                "target_value": v.target_value,
                "horizon": v.horizon,
            }
            for v in recent_violations
        ]

        # Performance trends
        for horizon in ["1h", "24h"]:
            recent_performance = [s for s in self.performance_history[-50:] if s.horizon == horizon]

            if recent_performance:
                mae_values = [s.metrics.get("mae", 0) for s in recent_performance]
                dashboard_data["performance_trend"][f"mae_{horizon}"] = {
                    "values": mae_values[-10:],  # Last 10 values
                    "trend": "improving"
                    if len(mae_values) > 1 and mae_values[-1] < mae_values[-2]
                    else "degrading",
                }

        # Coverage stats
        total_metrics = len(self.slo_metrics)
        healthy_metrics = len(
            [s for s in self.current_slo_status.values() if s == SLOStatus.HEALTHY]
        )

        dashboard_data["coverage"] = {
            "total_metrics": total_metrics,
            "healthy_metrics": healthy_metrics,
            "health_percentage": (healthy_metrics / total_metrics * 100)
            if total_metrics > 0
            else 0,
            "last_evaluation": self.last_evaluation.isoformat() if self.last_evaluation else None,
        }

        return dashboard_data

    def _calculate_overall_health(self) -> str:
        """Calculate overall system health based on SLO status"""

        if not self.current_slo_status:
            return "unknown"

        status_counts = {}
        for status in self.current_slo_status.values():
            status_counts[status] = status_counts.get(status, 0) + 1

        total = len(self.current_slo_status)

        # Health calculation logic
        if status_counts.get(SLOStatus.BREACH, 0) > 0:
            return "critical"
        elif status_counts.get(SLOStatus.CRITICAL, 0) > total * 0.2:  # More than 20% critical
            return "critical"
        elif status_counts.get(SLOStatus.WARNING, 0) > total * 0.5:  # More than 50% warning
            return "warning"
        elif status_counts.get(SLOStatus.HEALTHY, 0) > total * 0.8:  # More than 80% healthy
            return "healthy"
        else:
            return "warning"


# Global instance
_slo_monitor = None


def get_slo_monitor() -> SLOMonitor:
    """Get global SLO monitor instance"""
    global _slo_monitor
    if _slo_monitor is None:
        _slo_monitor = SLOMonitor()
    return _slo_monitor
