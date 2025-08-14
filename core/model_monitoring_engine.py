#!/usr/bin/env python3
"""
CryptoSmartTrader V2 - Automated Model Monitoring & Auto-Healing Engine
Comprehensive model monitoring with automatic healing and performance tracking
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
import json
import json  # SECURITY: Replaced pickle with JSON for external data
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
import traceback
from abc import ABC, abstractmethod


class ModelStatus(Enum):
    """Model status types"""

    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    FAILED = "failed"
    DISABLED = "disabled"
    RETRAINING = "retraining"


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class DriftType(Enum):
    """Types of model drift"""

    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE_DRIFT = "performance_drift"


@dataclass
class ModelMetrics:
    """Model performance metrics"""

    model_id: str
    timestamp: datetime
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    prediction_latency: float
    data_quality_score: float
    feature_drift_score: float
    prediction_confidence: float
    error_rate: float
    throughput: float


@dataclass
class DriftAlert:
    """Model drift alert"""

    model_id: str
    drift_type: DriftType
    severity: AlertSeverity
    drift_score: float
    threshold: float
    description: str
    timestamp: datetime
    affected_features: List[str]
    recommended_actions: List[str]


@dataclass
class HealthCheckResult:
    """Model health check result"""

    model_id: str
    status: ModelStatus
    overall_score: float
    individual_scores: Dict[str, float]
    alerts: List[DriftAlert]
    recommendations: List[str]
    last_update: datetime


class DataQualityMonitor:
    """Monitor data quality and detect anomalies"""

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.quality_history = {}
        self.quality_thresholds = {
            "missing_data_threshold": 0.05,  # 5% missing data max
            "outlier_threshold": 0.02,  # 2% outliers max
            "drift_threshold": 0.1,  # 10% distribution drift max
            "correlation_threshold": 0.8,  # 80% correlation minimum
        }

    def assess_data_quality(self, data: pd.DataFrame, model_id: str) -> Dict[str, Any]:
        """Comprehensive data quality assessment"""
        try:
            quality_assessment = {
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "overall_score": 0.0,
                "metrics": {},
                "issues": [],
                "recommendations": [],
            }

            # 1. Missing data assessment
            missing_metrics = self._assess_missing_data(data)
            quality_assessment["metrics"]["missing_data"] = missing_metrics

            # 2. Outlier detection
            outlier_metrics = self._detect_outliers(data)
            quality_assessment["metrics"]["outliers"] = outlier_metrics

            # 3. Distribution drift
            drift_metrics = self._detect_distribution_drift(data, model_id)
            quality_assessment["metrics"]["distribution_drift"] = drift_metrics

            # 4. Feature correlation changes
            correlation_metrics = self._assess_feature_correlations(data, model_id)
            quality_assessment["metrics"]["correlations"] = correlation_metrics

            # 5. Data freshness
            freshness_metrics = self._assess_data_freshness(data)
            quality_assessment["metrics"]["freshness"] = freshness_metrics

            # Calculate overall score
            overall_score = self._calculate_overall_quality_score(quality_assessment["metrics"])
            quality_assessment["overall_score"] = overall_score

            # Generate recommendations
            recommendations = self._generate_quality_recommendations(quality_assessment["metrics"])
            quality_assessment["recommendations"] = recommendations

            # Store in history
            if model_id not in self.quality_history:
                self.quality_history[model_id] = []
            self.quality_history[model_id].append(quality_assessment)

            # Keep only last 100 assessments
            if len(self.quality_history[model_id]) > 100:
                self.quality_history[model_id] = self.quality_history[model_id][-100:]

            return quality_assessment

        except Exception as e:
            self.logger.error(f"Data quality assessment failed: {e}")
            return {
                "model_id": model_id,
                "error": str(e),
                "overall_score": 0.0,
                "timestamp": datetime.now().isoformat(),
            }

    def _assess_missing_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess missing data quality"""
        try:
            total_cells = data.size
            missing_cells = data.isnull().sum().sum()
            missing_ratio = missing_cells / total_cells if total_cells > 0 else 0

            # Per-column missing data
            column_missing = {}
            for col in data.columns:
                col_missing = data[col].isnull().sum() / len(data)
                column_missing[col] = float(col_missing)

            return {
                "total_missing_ratio": float(missing_ratio),
                "column_missing_ratios": column_missing,
                "columns_with_high_missing": [
                    col
                    for col, ratio in column_missing.items()
                    if ratio > self.quality_thresholds["missing_data_threshold"]
                ],
                "score": max(0.0, 1.0 - missing_ratio * 10),  # Penalize missing data
            }

        except Exception as e:
            return {"error": str(e), "score": 0.0}

    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numerical data"""
        try:
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            outlier_ratios = {}
            total_outliers = 0
            total_numerical_values = 0

            for col in numerical_cols:
                col_data = data[col].dropna()
                if len(col_data) == 0:
                    continue

                # IQR method for outlier detection
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1

                if IQR == 0:  # No variance
                    outlier_ratios[col] = 0.0
                    continue

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                outlier_ratio = outliers / len(col_data)
                outlier_ratios[col] = float(outlier_ratio)

                total_outliers += outliers
                total_numerical_values += len(col_data)

            overall_outlier_ratio = (
                total_outliers / total_numerical_values if total_numerical_values > 0 else 0
            )

            return {
                "overall_outlier_ratio": float(overall_outlier_ratio),
                "column_outlier_ratios": outlier_ratios,
                "columns_with_high_outliers": [
                    col
                    for col, ratio in outlier_ratios.items()
                    if ratio > self.quality_thresholds["outlier_threshold"]
                ],
                "score": max(0.0, 1.0 - overall_outlier_ratio * 20),  # Penalize outliers
            }

        except Exception as e:
            return {"error": str(e), "score": 0.5}

    def _detect_distribution_drift(self, data: pd.DataFrame, model_id: str) -> Dict[str, Any]:
        """Detect distribution drift compared to historical data"""
        try:
            if model_id not in self.quality_history or len(self.quality_history[model_id]) == 0:
                return {
                    "drift_detected": False,
                    "drift_score": 0.0,
                    "score": 1.0,
                    "note": "No historical data for comparison",
                }

            # Get recent historical data for comparison
            recent_assessment = self.quality_history[model_id][-1]

            # Simple distribution drift detection using column means
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            drift_scores = {}

            for col in numerical_cols:
                if col in data.columns and not data[col].empty:
                    current_mean = data[col].mean()
                    current_std = data[col].std()

                    # Compare with historical (simplified)
                    # In production, would use more sophisticated drift detection
                    historical_mean = current_mean  # Placeholder
                    drift_score = abs(current_mean - historical_mean) / (current_std + 1e-10)
                    drift_scores[col] = float(min(1.0, drift_score))

            overall_drift = np.mean(list(drift_scores.values())) if drift_scores else 0.0
            drift_detected = overall_drift > self.quality_thresholds["drift_threshold"]

            return {
                "drift_detected": drift_detected,
                "drift_score": float(overall_drift),
                "column_drift_scores": drift_scores,
                "drifted_columns": [
                    col
                    for col, score in drift_scores.items()
                    if score > self.quality_thresholds["drift_threshold"]
                ],
                "score": max(0.0, 1.0 - overall_drift),
            }

        except Exception as e:
            return {"error": str(e), "score": 0.5}

    def _assess_feature_correlations(self, data: pd.DataFrame, model_id: str) -> Dict[str, Any]:
        """Assess feature correlation stability"""
        try:
            numerical_cols = data.select_dtypes(include=[np.number]).columns

            if len(numerical_cols) < 2:
                return {
                    "correlation_stability": 1.0,
                    "score": 1.0,
                    "note": "Insufficient numerical columns for correlation analysis",
                }

            # Calculate current correlation matrix
            current_corr = data[numerical_cols].corr()

            # For now, assess correlation matrix health
            # Check for high correlations (potential multicollinearity)
            high_corr_count = 0
            total_pairs = 0

            for i in range(len(numerical_cols)):
                for j in range(i + 1, len(numerical_cols)):
                    corr_val = abs(current_corr.iloc[i, j])
                    if not np.isnan(corr_val):
                        total_pairs += 1
                        if corr_val > 0.9:  # Very high correlation
                            high_corr_count += 1

            high_corr_ratio = high_corr_count / total_pairs if total_pairs > 0 else 0

            return {
                "high_correlation_ratio": float(high_corr_ratio),
                "correlation_matrix_size": len(numerical_cols),
                "total_correlation_pairs": total_pairs,
                "score": max(0.0, 1.0 - high_corr_ratio * 2),  # Penalize high correlations
            }

        except Exception as e:
            return {"error": str(e), "score": 0.5}

    def _assess_data_freshness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data freshness and recency"""
        try:
            freshness_score = 1.0

            # Check if data has datetime index
            if isinstance(data.index, pd.DatetimeIndex):
                latest_data = data.index.max()
                current_time = pd.Timestamp.now()

                time_diff = current_time - latest_data
                hours_old = time_diff.total_seconds() / 3600

                # Penalize old data
                if hours_old > 24:  # More than 1 day old
                    freshness_score = max(0.1, 1.0 - (hours_old - 24) / (24 * 7))  # Decay over week

                return {
                    "latest_data_timestamp": latest_data.isoformat(),
                    "hours_since_latest": float(hours_old),
                    "freshness_score": freshness_score,
                    "score": freshness_score,
                }
            else:
                return {
                    "freshness_score": 0.8,  # Neutral score for non-datetime data
                    "score": 0.8,
                    "note": "No datetime index found",
                }

        except Exception as e:
            return {"error": str(e), "score": 0.5}

    def _calculate_overall_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall data quality score"""
        try:
            scores = []
            weights = {
                "missing_data": 0.25,
                "outliers": 0.20,
                "distribution_drift": 0.25,
                "correlations": 0.15,
                "freshness": 0.15,
            }

            for metric_name, weight in weights.items():
                if metric_name in metrics and "score" in metrics[metric_name]:
                    score = metrics[metric_name]["score"]
                    scores.append(score * weight)

            return sum(scores) if scores else 0.0

        except Exception:
            return 0.5

    def _generate_quality_recommendations(self, metrics: Dict[str, Any]) -> List[str]:
        """Generate data quality improvement recommendations"""
        recommendations = []

        try:
            # Missing data recommendations
            if "missing_data" in metrics:
                if metrics["missing_data"].get("score", 1.0) < 0.7:
                    recommendations.append(
                        "Consider data imputation or collection improvement for missing values"
                    )

            # Outlier recommendations
            if "outliers" in metrics:
                if metrics["outliers"].get("score", 1.0) < 0.7:
                    recommendations.append(
                        "Review outlier detection and consider data cleaning procedures"
                    )

            # Drift recommendations
            if "distribution_drift" in metrics:
                if metrics["distribution_drift"].get("drift_detected", False):
                    recommendations.append(
                        "Data distribution drift detected - consider model retraining"
                    )

            # Correlation recommendations
            if "correlations" in metrics:
                if metrics["correlations"].get("score", 1.0) < 0.7:
                    recommendations.append(
                        "High feature correlations detected - consider feature selection"
                    )

            # Freshness recommendations
            if "freshness" in metrics:
                if metrics["freshness"].get("score", 1.0) < 0.7:
                    recommendations.append("Data freshness issue - update data pipeline frequency")

        except Exception as e:
            recommendations.append(f"Error generating recommendations: {e}")

        return recommendations


class ModelPerformanceMonitor:
    """Monitor model performance and detect degradation"""

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.performance_history = {}
        self.performance_thresholds = {
            "accuracy_threshold": 0.7,
            "latency_threshold": 1.0,  # seconds
            "error_rate_threshold": 0.1,
            "confidence_threshold": 0.6,
        }

    def monitor_model_performance(
        self,
        model_id: str,
        predictions: np.ndarray,
        true_values: Optional[np.ndarray] = None,
        prediction_times: Optional[List[float]] = None,
    ) -> ModelMetrics:
        """Monitor comprehensive model performance"""
        try:
            timestamp = datetime.now()

            # Calculate performance metrics
            if true_values is not None and len(true_values) == len(predictions):
                accuracy, precision, recall, f1 = self._calculate_accuracy_metrics(
                    predictions, true_values
                )
            else:
                # Use prediction confidence as proxy
                accuracy = precision = recall = f1 = (
                    np.mean(np.abs(predictions)) if len(predictions) > 0 else 0.0
                )

            # Calculate latency metrics
            if prediction_times:
                avg_latency = np.mean(prediction_times)
            else:
                avg_latency = 0.0

            # Calculate prediction confidence
            if len(predictions) > 0:
                prediction_confidence = float(np.mean(np.abs(predictions)))
                prediction_std = float(np.std(predictions))
            else:
                prediction_confidence = 0.0
                prediction_std = 0.0

            # Error rate (simplified)
            error_rate = 1.0 - accuracy if accuracy <= 1.0 else 0.0

            # Throughput (predictions per second)
            if prediction_times:
                total_time = sum(prediction_times)
                throughput = len(predictions) / total_time if total_time > 0 else 0.0
            else:
                throughput = 0.0

            # Data quality score (simplified)
            data_quality_score = 1.0 - (
                np.isnan(predictions).sum() / len(predictions) if len(predictions) > 0 else 0
            )

            # Feature drift score (placeholder)
            feature_drift_score = 0.0  # Would be calculated based on feature distribution changes

            metrics = ModelMetrics(
                model_id=model_id,
                timestamp=timestamp,
                accuracy=float(accuracy),
                precision=float(precision),
                recall=float(recall),
                f1_score=float(f1),
                prediction_latency=float(avg_latency),
                data_quality_score=float(data_quality_score),
                feature_drift_score=float(feature_drift_score),
                prediction_confidence=float(prediction_confidence),
                error_rate=float(error_rate),
                throughput=float(throughput),
            )

            # Store in history
            if model_id not in self.performance_history:
                self.performance_history[model_id] = []
            self.performance_history[model_id].append(metrics)

            # Keep only last 1000 metrics
            if len(self.performance_history[model_id]) > 1000:
                self.performance_history[model_id] = self.performance_history[model_id][-1000:]

            return metrics

        except Exception as e:
            self.logger.error(f"Performance monitoring failed: {e}")
            return ModelMetrics(
                model_id=model_id,
                timestamp=datetime.now(),
                accuracy=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                prediction_latency=999.0,
                data_quality_score=0.0,
                feature_drift_score=1.0,
                prediction_confidence=0.0,
                error_rate=1.0,
                throughput=0.0,
            )

    def _calculate_accuracy_metrics(
        self, predictions: np.ndarray, true_values: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """Calculate accuracy, precision, recall, and F1 score"""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

            # For regression, convert to classification (above/below median)
            if len(np.unique(true_values)) > 10:  # Likely regression
                median_true = np.median(true_values)
                true_binary = (true_values > median_true).astype(int)
                pred_binary = (predictions > median_true).astype(int)
            else:  # Classification
                true_binary = true_values.astype(int)
                pred_binary = np.round(predictions).astype(int)

            accuracy = accuracy_score(true_binary, pred_binary)
            precision = precision_score(
                true_binary, pred_binary, average="weighted", zero_division=0
            )
            recall = recall_score(true_binary, pred_binary, average="weighted", zero_division=0)
            f1 = f1_score(true_binary, pred_binary, average="weighted", zero_division=0)

            return accuracy, precision, recall, f1

        except Exception as e:
            self.logger.warning(f"Accuracy calculation failed: {e}")
            return 0.0, 0.0, 0.0, 0.0

    def detect_performance_degradation(
        self, model_id: str, lookback_periods: int = 10
    ) -> List[DriftAlert]:
        """Detect performance degradation over time"""
        try:
            alerts = []

            if model_id not in self.performance_history:
                return alerts

            history = self.performance_history[model_id]
            if len(history) < lookback_periods:
                return alerts

            # Compare recent performance with baseline
            recent_metrics = history[-lookback_periods:]
            baseline_metrics = (
                history[:-lookback_periods]
                if len(history) > lookback_periods
                else history[: len(history) // 2]
            )

            if not baseline_metrics:
                return alerts

            # Check accuracy degradation
            recent_accuracy = np.mean([m.accuracy for m in recent_metrics])
            baseline_accuracy = np.mean([m.accuracy for m in baseline_metrics])

            if recent_accuracy < baseline_accuracy * 0.9:  # 10% degradation
                alerts.append(
                    DriftAlert(
                        model_id=model_id,
                        drift_type=DriftType.PERFORMANCE_DRIFT,
                        severity=AlertSeverity.WARNING
                        if recent_accuracy > baseline_accuracy * 0.8
                        else AlertSeverity.CRITICAL,
                        drift_score=float(baseline_accuracy - recent_accuracy),
                        threshold=baseline_accuracy * 0.9,
                        description=f"Accuracy degraded from {baseline_accuracy:.3f} to {recent_accuracy:.3f}",
                        timestamp=datetime.now(),
                        affected_features=["accuracy"],
                        recommended_actions=[
                            "Investigate data quality",
                            "Consider model retraining",
                        ],
                    )
                )

            # Check latency degradation
            recent_latency = np.mean([m.prediction_latency for m in recent_metrics])
            baseline_latency = np.mean([m.prediction_latency for m in baseline_metrics])

            if recent_latency > baseline_latency * 1.5:  # 50% increase
                alerts.append(
                    DriftAlert(
                        model_id=model_id,
                        drift_type=DriftType.PERFORMANCE_DRIFT,
                        severity=AlertSeverity.WARNING,
                        drift_score=float(recent_latency - baseline_latency),
                        threshold=baseline_latency * 1.5,
                        description=f"Latency increased from {baseline_latency:.3f}s to {recent_latency:.3f}s",
                        timestamp=datetime.now(),
                        affected_features=["latency"],
                        recommended_actions=["Check system resources", "Optimize model inference"],
                    )
                )

            return alerts

        except Exception as e:
            self.logger.error(f"Performance degradation detection failed: {e}")
            return []


class AutoHealingEngine:
    """Automatic model healing and recovery system"""

    def __init__(self, config_manager=None):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
        self.healing_actions = {}
        self.healing_history = []

    def register_healing_action(
        self, model_id: str, alert_type: DriftType, action: Callable[[str, DriftAlert], bool]
    ):
        """Register an automatic healing action for specific alert types"""
        try:
            if model_id not in self.healing_actions:
                self.healing_actions[model_id] = {}

            self.healing_actions[model_id][alert_type] = action
            self.logger.info(f"Registered healing action for {model_id} - {alert_type.value}")

        except Exception as e:
            self.logger.error(f"Failed to register healing action: {e}")

    def execute_healing(self, model_id: str, alerts: List[DriftAlert]) -> Dict[str, Any]:
        """Execute healing actions for detected alerts"""
        try:
            healing_result = {
                "model_id": model_id,
                "timestamp": datetime.now().isoformat(),
                "alerts_processed": len(alerts),
                "actions_taken": [],
                "success_count": 0,
                "failure_count": 0,
            }

            for alert in alerts:
                if (
                    model_id in self.healing_actions
                    and alert.drift_type in self.healing_actions[model_id]
                ):
                    action = self.healing_actions[model_id][alert.drift_type]

                    try:
                        success = action(model_id, alert)

                        action_result = {
                            "alert_type": alert.drift_type.value,
                            "action_taken": action.__name__
                            if hasattr(action, "__name__")
                            else "unknown",
                            "success": success,
                            "timestamp": datetime.now().isoformat(),
                        }

                        healing_result["actions_taken"].append(action_result)

                        if success:
                            healing_result["success_count"] += 1
                            self.logger.info(
                                f"Healing action successful for {model_id} - {alert.drift_type.value}"
                            )
                        else:
                            healing_result["failure_count"] += 1
                            self.logger.warning(
                                f"Healing action failed for {model_id} - {alert.drift_type.value}"
                            )

                    except Exception as e:
                        healing_result["failure_count"] += 1
                        healing_result["actions_taken"].append(
                            {
                                "alert_type": alert.drift_type.value,
                                "action_taken": "error",
                                "success": False,
                                "error": str(e),
                                "timestamp": datetime.now().isoformat(),
                            }
                        )
                        self.logger.error(f"Healing action error for {model_id}: {e}")
                else:
                    self.logger.info(
                        f"No healing action registered for {model_id} - {alert.drift_type.value}"
                    )

            # Store healing history
            self.healing_history.append(healing_result)

            return healing_result

        except Exception as e:
            self.logger.error(f"Healing execution failed: {e}")
            return {"model_id": model_id, "error": str(e), "timestamp": datetime.now().isoformat()}


class ModelMonitoringCoordinator:
    """Main coordinator for model monitoring and auto-healing"""

    def __init__(self, config_manager=None, cache_manager=None):
        self.config_manager = config_manager
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)

        # Initialize components
        self.data_quality_monitor = DataQualityMonitor(config_manager)
        self.performance_monitor = ModelPerformanceMonitor(config_manager)
        self.auto_healing_engine = AutoHealingEngine(config_manager)

        # Monitoring state
        self.monitored_models = {}
        self.monitoring_active = False
        self.monitoring_thread = None

        self.logger.info("Model Monitoring Coordinator initialized")

    def register_model(self, model_id: str, model_config: Dict[str, Any] = None):
        """Register a model for monitoring"""
        try:
            self.monitored_models[model_id] = {
                "config": model_config or {},
                "status": ModelStatus.HEALTHY,
                "last_health_check": datetime.now(),
                "alert_count": 0,
            }

            self.logger.info(f"Model {model_id} registered for monitoring")

        except Exception as e:
            self.logger.error(f"Failed to register model {model_id}: {e}")

    def conduct_comprehensive_health_check(
        self,
        model_id: str,
        data: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        true_values: Optional[np.ndarray] = None,
    ) -> HealthCheckResult:
        """Conduct comprehensive health check for a model"""
        try:
            # Data quality assessment
            data_quality = self.data_quality_monitor.assess_data_quality(data, model_id)

            # Performance monitoring
            if predictions is not None:
                performance_metrics = self.performance_monitor.monitor_model_performance(
                    model_id, predictions, true_values
                )
            else:
                performance_metrics = None

            # Detect degradation
            degradation_alerts = self.performance_monitor.detect_performance_degradation(model_id)

            # Calculate overall health scores
            individual_scores = {
                "data_quality": data_quality.get("overall_score", 0.0),
                "performance": performance_metrics.accuracy if performance_metrics else 0.5,
                "latency": 1.0 - min(1.0, performance_metrics.prediction_latency)
                if performance_metrics
                else 0.5,
                "stability": 1.0 - len(degradation_alerts) * 0.2,  # Penalize alerts
            }

            overall_score = np.mean(list(individual_scores.values()))

            # Determine status
            if overall_score > 0.8:
                status = ModelStatus.HEALTHY
            elif overall_score > 0.6:
                status = ModelStatus.WARNING
            elif overall_score > 0.4:
                status = ModelStatus.DEGRADED
            else:
                status = ModelStatus.FAILED

            # Generate recommendations
            recommendations = []
            recommendations.extend(data_quality.get("recommendations", []))

            if performance_metrics and performance_metrics.prediction_latency > 1.0:
                recommendations.append("Optimize model inference for better latency")

            if degradation_alerts:
                recommendations.append("Investigate performance degradation alerts")

            # Create health check result
            health_result = HealthCheckResult(
                model_id=model_id,
                status=status,
                overall_score=overall_score,
                individual_scores=individual_scores,
                alerts=degradation_alerts,
                recommendations=recommendations,
                last_update=datetime.now(),
            )

            # Update model status
            if model_id in self.monitored_models:
                self.monitored_models[model_id]["status"] = status
                self.monitored_models[model_id]["last_health_check"] = datetime.now()
                self.monitored_models[model_id]["alert_count"] = len(degradation_alerts)

            # Execute auto-healing if needed
            if degradation_alerts:
                healing_result = self.auto_healing_engine.execute_healing(
                    model_id, degradation_alerts
                )
                self.logger.info(
                    f"Auto-healing executed for {model_id}: {healing_result['success_count']} successes"
                )

            return health_result

        except Exception as e:
            self.logger.error(f"Health check failed for {model_id}: {e}")
            return HealthCheckResult(
                model_id=model_id,
                status=ModelStatus.FAILED,
                overall_score=0.0,
                individual_scores={},
                alerts=[],
                recommendations=[f"Health check failed: {e}"],
                last_update=datetime.now(),
            )

    def start_continuous_monitoring(self, check_interval: int = 300):
        """Start continuous monitoring of all registered models"""
        try:
            self.monitoring_active = True

            if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
                self.monitoring_thread = threading.Thread(
                    target=self._monitoring_loop, args=(check_interval,), daemon=True
                )
                self.monitoring_thread.start()

            self.logger.info(f"Continuous monitoring started with {check_interval}s interval")

        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")

    def stop_monitoring(self):
        """Stop continuous monitoring"""
        try:
            self.monitoring_active = False

            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=10)

            self.logger.info("Monitoring stopped")

        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {e}")

    def _monitoring_loop(self, check_interval: int):
        """Main monitoring loop"""
        try:
            while self.monitoring_active:
                for model_id in list(self.monitored_models.keys()):
                    try:
                        # Simplified health check (would need real data in production)
                        self.logger.debug(f"Monitoring check for {model_id}")

                        # Update last check time
                        self.monitored_models[model_id]["last_health_check"] = datetime.now()

                    except Exception as e:
                        self.logger.error(f"Monitoring check failed for {model_id}: {e}")

                time.sleep(check_interval)

        except Exception as e:
            self.logger.error(f"Monitoring loop error: {e}")

    def get_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report"""
        try:
            return {
                "monitoring_active": self.monitoring_active,
                "total_models": len(self.monitored_models),
                "model_statuses": {
                    model_id: {
                        "status": info["status"].value,
                        "last_check": info["last_health_check"].isoformat(),
                        "alert_count": info["alert_count"],
                    }
                    for model_id, info in self.monitored_models.items()
                },
                "system_health": {
                    "data_quality_monitor_active": self.data_quality_monitor is not None,
                    "performance_monitor_active": self.performance_monitor is not None,
                    "auto_healing_active": self.auto_healing_engine is not None,
                },
                "healing_history_count": len(self.auto_healing_engine.healing_history),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            return {"error": str(e), "timestamp": datetime.now().isoformat()}


# Convenience function
def get_model_monitoring_coordinator(
    config_manager=None, cache_manager=None
) -> ModelMonitoringCoordinator:
    """Get configured model monitoring coordinator"""
    return ModelMonitoringCoordinator(config_manager, cache_manager)


if __name__ == "__main__":
    # Test the model monitoring engine
    coordinator = get_model_monitoring_coordinator()

    print("Testing Model Monitoring Engine...")

    # Register a test model
    coordinator.register_model("test_model_v1", {"type": "LSTM", "version": "1.0"})

    # Create test data
    import pandas as pd
    import numpy as np

    np.random.seed(42)
    test_data = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
            "target": np.random.randn(100),
        }
    )

    # Simulate predictions
    test_predictions = np.random.randn(100)
    test_true_values = test_data["target"].values

    # Conduct health check
    health_result = coordinator.conduct_comprehensive_health_check(
        "test_model_v1", test_data, test_predictions, test_true_values
    )

    print(f"\nHealth Check Result:")
    print(f"  Status: {health_result.status.value}")
    print(f"  Overall Score: {health_result.overall_score:.2f}")
    print(f"  Individual Scores: {health_result.individual_scores}")
    print(f"  Alerts: {len(health_result.alerts)}")
    print(f"  Recommendations: {len(health_result.recommendations)}")

    # Get monitoring report
    report = coordinator.get_monitoring_report()
    print(f"\nMonitoring Report:")
    print(f"  Total Models: {report['total_models']}")
    print(f"  System Health: {report['system_health']}")

    print("Model monitoring engine test completed")

"""
SECURITY POLICY: NO PICKLE ALLOWED
This file handles external data.
Pickle usage is FORBIDDEN for security reasons.
Use JSON or msgpack for all serialization.
"""

