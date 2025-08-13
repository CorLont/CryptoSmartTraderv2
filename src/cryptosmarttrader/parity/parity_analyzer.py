"""
Backtest-Live Parity Analyzer for CryptoSmartTrader
Comprehensive analysis and monitoring of backtest vs live performance.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import logging
import warnings


class ParityStatus(Enum):
    """Parity status classifications."""

    EXCELLENT = "excellent"  # < 5 bps tracking error
    GOOD = "good"  # 5-10 bps tracking error
    ACCEPTABLE = "acceptable"  # 10-20 bps tracking error
    WARNING = "warning"  # 20-50 bps tracking error
    CRITICAL = "critical"  # > 50 bps tracking error


@dataclass
class ParityMetrics:
    """Comprehensive parity metrics."""

    period_start: datetime
    period_end: datetime
    backtest_return: float
    live_return: float
    tracking_error_bps: float
    correlation: float
    hit_rate: float
    information_ratio: float
    max_deviation_bps: float
    avg_deviation_bps: float
    status: ParityStatus
    confidence_score: float
    component_attribution: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class DriftDetection:
    """Model/data drift detection results."""

    drift_detected: bool
    drift_type: str
    drift_magnitude: float
    drift_confidence: float
    detection_time: datetime
    affected_components: List[str]
    recommended_actions: List[str]
    metadata: Dict[str, Any]


class ParityAnalyzer:
    """
    Enterprise backtest-live parity analyzer with drift detection.

    Features:
    - Daily parity monitoring and reporting
    - Component-level attribution analysis
    - Statistical drift detection
    - Automatic degradation triggers
    - Performance quality scoring
    - Real-time alerting integration
    """

    def __init__(
        self,
        tracking_error_threshold_bps: float = 20.0,
        drift_detection_window: int = 30,
        min_observations: int = 10,
        confidence_threshold: float = 0.7,
    ):
        self.tracking_error_threshold = tracking_error_threshold_bps
        self.drift_detection_window = drift_detection_window
        self.min_observations = min_observations
        self.confidence_threshold = confidence_threshold

        # Parity history tracking
        self.parity_history: List[ParityMetrics] = []
        self.component_history: Dict[str, List[float]] = {}

        # Drift detection state
        self.drift_alerts: List[DriftDetection] = []
        self.last_drift_check: Optional[datetime] = None

        # Performance baselines
        self.baseline_metrics: Dict[str, float] = {}

        self.logger = logging.getLogger(__name__)
        self.logger.info("ParityAnalyzer initialized with enterprise monitoring")

    def analyze_parity(
        self,
        backtest_data: Dict[str, Any],
        live_data: Dict[str, Any],
        period_start: datetime,
        period_end: datetime,
        execution_results: Optional[List[Any]] = None,
    ) -> ParityMetrics:
        """
        Perform comprehensive parity analysis between backtest and live performance.

        Args:
            backtest_data: Backtest performance data
            live_data: Live trading performance data
            period_start: Analysis period start
            period_end: Analysis period end
            execution_results: Optional execution results for attribution

        Returns:
            ParityMetrics with detailed analysis
        """

        # Extract returns
        backtest_return = self._extract_return(backtest_data)
        live_return = self._extract_return(live_data)

        # Calculate tracking error
        tracking_error_bps = abs(backtest_return - live_return) * 10000

        # Calculate correlation if we have time series data
        correlation = self._calculate_correlation(backtest_data, live_data)

        # Calculate hit rate (directional accuracy)
        hit_rate = self._calculate_hit_rate(backtest_data, live_data)

        # Calculate information ratio
        info_ratio = self._calculate_information_ratio(backtest_data, live_data)

        # Component attribution analysis
        component_attribution = self._analyze_component_attribution(
            backtest_data, live_data, execution_results
        )

        # Calculate deviation statistics
        max_deviation_bps, avg_deviation_bps = self._calculate_deviation_stats(
            backtest_data, live_data
        )

        # Determine parity status
        status = self._determine_parity_status(tracking_error_bps, correlation, hit_rate)

        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            backtest_data, live_data, tracking_error_bps, correlation
        )

        # Create parity metrics
        parity_metrics = ParityMetrics(
            period_start=period_start,
            period_end=period_end,
            backtest_return=backtest_return,
            live_return=live_return,
            tracking_error_bps=tracking_error_bps,
            correlation=correlation,
            hit_rate=hit_rate,
            information_ratio=info_ratio,
            max_deviation_bps=max_deviation_bps,
            avg_deviation_bps=avg_deviation_bps,
            status=status,
            confidence_score=confidence_score,
            component_attribution=component_attribution,
            metadata={
                "observation_count": self._get_observation_count(backtest_data, live_data),
                "analysis_timestamp": datetime.utcnow(),
                "data_quality_score": self._assess_data_quality(backtest_data, live_data),
            },
        )

        # Update history
        self.parity_history.append(parity_metrics)
        self._update_component_history(component_attribution)

        # Check for drift
        drift_detection = self._check_for_drift(parity_metrics)
        if drift_detection.drift_detected:
            self.drift_alerts.append(drift_detection)

        return parity_metrics

    def _extract_return(self, data: Dict[str, Any]) -> float:
        """Extract return from performance data."""
        # Try multiple possible keys for return data
        return_keys = ["return", "total_return", "pnl", "performance", "profit_loss"]

        for key in return_keys:
            if key in data:
                value = data[key]
                if isinstance(value, (list, pd.Series)):
                    return float(value[-1]) if len(value) > 0 else 0.0
                else:
                    return float(value)

        # Fallback: calculate from price data if available
        if "prices" in data or "price_series" in data:
            prices = data.get("prices", data.get("price_series", []))
            if len(prices) >= 2:
                return (prices[-1] / prices[0]) - 1.0

        self.logger.warning("Could not extract return from data, using 0.0")
        return 0.0

    def _calculate_correlation(
        self, backtest_data: Dict[str, Any], live_data: Dict[str, Any]
    ) -> float:
        """Calculate correlation between backtest and live returns."""

        # Try to extract time series data
        backtest_series = self._extract_time_series(backtest_data)
        live_series = self._extract_time_series(live_data)

        if backtest_series is None or live_series is None:
            return 0.8  # Default assumption

        # Align series by timestamp if possible
        if len(backtest_series) != len(live_series):
            min_length = min(len(backtest_series), len(live_series))
            backtest_series = backtest_series[-min_length:]
            live_series = live_series[-min_length:]

        if len(backtest_series) < 3:
            return 0.8  # Not enough data

        try:
            correlation = np.corrcoef(backtest_series, live_series)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.8
        except Exception:
            return 0.8

    def _extract_time_series(self, data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract time series data for correlation calculation."""

        series_keys = ["returns", "return_series", "pnl_series", "prices", "price_series"]

        for key in series_keys:
            if key in data:
                series = data[key]
                if isinstance(series, (list, np.ndarray)):
                    return np.array(series)
                elif isinstance(series, pd.Series):
                    return series.values

        return None

    def _calculate_hit_rate(
        self, backtest_data: Dict[str, Any], live_data: Dict[str, Any]
    ) -> float:
        """Calculate directional hit rate between backtest and live."""

        backtest_series = self._extract_time_series(backtest_data)
        live_series = self._extract_time_series(live_data)

        if backtest_series is None or live_series is None:
            return 0.75  # Default assumption

        # Calculate returns if we have prices
        if len(backtest_series) > 1 and len(live_series) > 1:
            backtest_returns = np.diff(backtest_series) / backtest_series[:-1]
            live_returns = np.diff(live_series) / live_series[:-1]

            # Align lengths
            min_length = min(len(backtest_returns), len(live_returns))
            backtest_returns = backtest_returns[-min_length:]
            live_returns = live_returns[-min_length:]

            if min_length > 0:
                # Calculate directional agreement
                same_direction = np.sign(backtest_returns) == np.sign(live_returns)
                hit_rate = np.mean(same_direction)
                return float(hit_rate)

        return 0.75

    def _calculate_information_ratio(
        self, backtest_data: Dict[str, Any], live_data: Dict[str, Any]
    ) -> float:
        """Calculate information ratio (excess return / tracking error)."""

        backtest_return = self._extract_return(backtest_data)
        live_return = self._extract_return(live_data)

        excess_return = live_return - backtest_return

        # Estimate tracking error from time series if available
        backtest_series = self._extract_time_series(backtest_data)
        live_series = self._extract_time_series(live_data)

        if backtest_series is not None and live_series is not None:
            min_length = min(len(backtest_series), len(live_series))
            if min_length > 1:
                backtest_rets = (
                    np.diff(backtest_series[-min_length:]) / backtest_series[-min_length:-1]
                )
                live_rets = np.diff(live_series[-min_length:]) / live_series[-min_length:-1]

                tracking_error = np.std(live_rets - backtest_rets)

                if tracking_error > 0:
                    return excess_return / tracking_error

        # Fallback calculation
        tracking_error_daily = abs(excess_return) / np.sqrt(252)  # Assume daily data

        if tracking_error_daily > 0:
            return excess_return / tracking_error_daily
        else:
            return 0.0

    def _analyze_component_attribution(
        self,
        backtest_data: Dict[str, Any],
        live_data: Dict[str, Any],
        execution_results: Optional[List[Any]],
    ) -> Dict[str, float]:
        """Analyze component-level attribution of tracking error."""

        attribution = {
            "execution_costs": 0.0,
            "timing_differences": 0.0,
            "data_differences": 0.0,
            "model_differences": 0.0,
            "slippage_impact": 0.0,
            "fee_impact": 0.0,
            "other": 0.0,
        }

        # Analyze execution costs if available
        if execution_results:
            total_fees = sum(getattr(result, "total_fees", 0) for result in execution_results)
            total_slippage = sum(getattr(result, "slippage_bps", 0) for result in execution_results)

            if execution_results:
                attribution["fee_impact"] = (
                    total_fees / len(execution_results) * 10000
                )  # Convert to bps
                attribution["slippage_impact"] = total_slippage / len(execution_results)
                attribution["execution_costs"] = (
                    attribution["fee_impact"] + attribution["slippage_impact"]
                )

        # Estimate other components based on available data
        backtest_return = self._extract_return(backtest_data)
        live_return = self._extract_return(live_data)
        total_difference_bps = abs(backtest_return - live_return) * 10000

        explained_difference = attribution["execution_costs"]
        unexplained_difference = max(0, total_difference_bps - explained_difference)

        # Distribute unexplained difference across components
        attribution["timing_differences"] = unexplained_difference * 0.3
        attribution["data_differences"] = unexplained_difference * 0.2
        attribution["model_differences"] = unexplained_difference * 0.3
        attribution["other"] = unexplained_difference * 0.2

        return attribution

    def _calculate_deviation_stats(
        self, backtest_data: Dict[str, Any], live_data: Dict[str, Any]
    ) -> Tuple[float, float]:
        """Calculate deviation statistics between backtest and live."""

        backtest_series = self._extract_time_series(backtest_data)
        live_series = self._extract_time_series(live_data)

        if backtest_series is None or live_series is None:
            # Use simple return difference
            backtest_return = self._extract_return(backtest_data)
            live_return = self._extract_return(live_data)
            deviation_bps = abs(backtest_return - live_return) * 10000
            return deviation_bps, deviation_bps

        # Calculate time series deviations
        min_length = min(len(backtest_series), len(live_series))
        backtest_aligned = backtest_series[-min_length:]
        live_aligned = live_series[-min_length:]

        # Calculate percentage deviations
        deviations = np.abs((live_aligned - backtest_aligned) / backtest_aligned) * 10000

        max_deviation_bps = float(np.max(deviations))
        avg_deviation_bps = float(np.mean(deviations))

        return max_deviation_bps, avg_deviation_bps

    def _determine_parity_status(
        self, tracking_error_bps: float, correlation: float, hit_rate: float
    ) -> ParityStatus:
        """Determine overall parity status based on metrics."""

        # Primary classification based on tracking error
        if tracking_error_bps < 5.0:
            primary_status = ParityStatus.EXCELLENT
        elif tracking_error_bps < 10.0:
            primary_status = ParityStatus.GOOD
        elif tracking_error_bps < 20.0:
            primary_status = ParityStatus.ACCEPTABLE
        elif tracking_error_bps < 50.0:
            primary_status = ParityStatus.WARNING
        else:
            primary_status = ParityStatus.CRITICAL

        # Adjust based on correlation and hit rate
        quality_score = (correlation + hit_rate) / 2.0

        if quality_score < 0.5 and primary_status in [ParityStatus.EXCELLENT, ParityStatus.GOOD]:
            # Downgrade if correlation/hit rate is poor
            primary_status = ParityStatus.ACCEPTABLE
        elif quality_score > 0.8 and primary_status == ParityStatus.WARNING:
            # Upgrade if correlation/hit rate is excellent
            primary_status = ParityStatus.ACCEPTABLE

        return primary_status

    def _calculate_confidence_score(
        self,
        backtest_data: Dict[str, Any],
        live_data: Dict[str, Any],
        tracking_error_bps: float,
        correlation: float,
    ) -> float:
        """Calculate confidence score for parity analysis."""

        confidence = 0.5  # Base confidence

        # Data quality component
        data_quality = self._assess_data_quality(backtest_data, live_data)
        confidence += data_quality * 0.3

        # Observation count component
        obs_count = self._get_observation_count(backtest_data, live_data)
        obs_confidence = min(1.0, obs_count / 30.0)  # Full confidence with 30+ observations
        confidence += obs_confidence * 0.2

        # Tracking error component (inverse relationship)
        te_confidence = max(
            0, 1.0 - (tracking_error_bps / 100.0)  # Full confidence at 0 bps, zero at 100 bps
        confidence += te_confidence * 0.3

        # Correlation component
        confidence += correlation * 0.2

        return max(0.0, min(1.0, confidence))

    def _assess_data_quality(
        self, backtest_data: Dict[str, Any], live_data: Dict[str, Any]
    ) -> float:
        """Assess quality of input data."""

        quality_score = 0.5  # Base quality

        # Check for required fields
        required_fields = ["return", "total_return", "pnl", "performance"]
        has_backtest_return = any(field in backtest_data for field in required_fields)
        has_live_return = any(field in live_data for field in required_fields)

        if has_backtest_return and has_live_return:
            quality_score += 0.3

        # Check for time series data
        has_backtest_series = self._extract_time_series(backtest_data) is not None
        has_live_series = self._extract_time_series(live_data) is not None

        if has_backtest_series and has_live_series:
            quality_score += 0.2

        return min(1.0, quality_score)

    def _get_observation_count(
        self, backtest_data: Dict[str, Any], live_data: Dict[str, Any]
    ) -> int:
        """Get number of observations available for analysis."""

        backtest_series = self._extract_time_series(backtest_data)
        live_series = self._extract_time_series(live_data)

        if backtest_series is not None and live_series is not None:
            return min(len(backtest_series), len(live_series))
        else:
            return 1  # Single observation

    def _update_component_history(self, component_attribution: Dict[str, float]):
        """Update component attribution history."""

        for component, value in component_attribution.items():
            if component not in self.component_history:
                self.component_history[component] = []

            self.component_history[component].append(value)

            # Keep only recent history
            if len(self.component_history[component]) > 100:
                self.component_history[component] = self.component_history[component][-100:]

    def _check_for_drift(self, parity_metrics: ParityMetrics) -> DriftDetection:
        """Check for model or data drift based on parity degradation."""

        drift_detected = False
        drift_type = "none"
        drift_magnitude = 0.0
        drift_confidence = 0.0
        affected_components = []
        recommended_actions = []

        # Check if we have sufficient history
        if len(self.parity_history) < self.min_observations:
            return DriftDetection(
                drift_detected=False,
                drift_type="insufficient_data",
                drift_magnitude=0.0,
                drift_confidence=0.0,
                detection_time=datetime.utcnow(),
                affected_components=[],
                recommended_actions=["Collect more data"],
                metadata={"history_length": len(self.parity_history)},
            )

        # Get recent tracking errors
        recent_errors = [
            m.tracking_error_bps for m in self.parity_history[-self.drift_detection_window :]
        ]
        baseline_errors = [
            m.tracking_error_bps for m in self.parity_history[: -self.drift_detection_window]
        ]

        if len(baseline_errors) < self.min_observations:
            baseline_errors = recent_errors[: len(recent_errors) // 2]
            recent_errors = recent_errors[len(recent_errors) // 2 :]

        # Statistical drift detection
        if len(baseline_errors) > 0 and len(recent_errors) > 0:
            baseline_mean = np.mean(baseline_errors)
            recent_mean = np.mean(recent_errors)

            # Check for significant increase in tracking error
            if recent_mean > baseline_mean * 1.5 and recent_mean > self.tracking_error_threshold:
                drift_detected = True
                drift_type = "tracking_error_degradation"
                drift_magnitude = (recent_mean - baseline_mean) / baseline_mean

                # Calculate confidence using t-test approximation
                pooled_std = np.sqrt((np.var(baseline_errors) + np.var(recent_errors)) / 2)
                if pooled_std > 0:
                    t_stat = abs(recent_mean - baseline_mean) / pooled_std
                    drift_confidence = min(0.95, max(0.5, t_stat / 3.0))
                else:
                    drift_confidence = 0.7

        # Component-level drift detection
        for component in self.component_history:
            if len(self.component_history[component]) >= self.min_observations:
                recent_values = self.component_history[component][-10:]
                baseline_values = self.component_history[component][:-10]

                if len(baseline_values) > 0:
                    recent_avg = np.mean(recent_values)
                    baseline_avg = np.mean(baseline_values)

                    if recent_avg > baseline_avg * 2.0:  # Component doubled
                        affected_components.append(component)

        # Generate recommendations
        if drift_detected:
            recommended_actions.append("Investigate model performance")
            recommended_actions.append("Check data quality and feeds")

            if parity_metrics.correlation < 0.5:
                recommended_actions.append("Verify signal generation consistency")

            if "execution_costs" in affected_components:
                recommended_actions.append("Review execution strategy")

            if "data_differences" in affected_components:
                recommended_actions.append("Audit data sources and alignment")

            if drift_magnitude > 0.5:
                recommended_actions.append("Consider model retraining")
                recommended_actions.append("Implement temporary position size reduction")

        return DriftDetection(
            drift_detected=drift_detected,
            drift_type=drift_type,
            drift_magnitude=drift_magnitude,
            drift_confidence=drift_confidence,
            detection_time=datetime.utcnow(),
            affected_components=affected_components,
            recommended_actions=recommended_actions,
            metadata={
                "baseline_tracking_error": np.mean(baseline_errors) if baseline_errors else 0.0,
                "recent_tracking_error": np.mean(recent_errors) if recent_errors else 0.0,
                "analysis_window": self.drift_detection_window,
                "threshold_bps": self.tracking_error_threshold,
            },
        )

    def generate_daily_report(self, date: Optional[datetime] = None) -> Dict[str, Any]:
        """Generate comprehensive daily parity report."""

        if date is None:
            date = datetime.utcnow().date()

        # Filter metrics for the specified date
        daily_metrics = [
            m for m in self.parity_history if m.period_start.date() <= date <= m.period_end.date()
        ]

        if not daily_metrics:
            return {"error": f"No parity data available for {date}"}

        latest_metrics = daily_metrics[-1]

        # Calculate trends
        if len(self.parity_history) >= 7:
            weekly_trend = self._calculate_trend(self.parity_history[-7:])
        else:
            weekly_trend = {"direction": "stable", "magnitude": 0.0}

        # Active alerts
        active_drift_alerts = [
            alert
            for alert in self.drift_alerts
            if alert.detection_time.date() == date and alert.drift_detected
        ]

        # Component analysis
        component_summary = {}
        for component in latest_metrics.component_attribution:
            if component in self.component_history:
                values = self.component_history[component]
                component_summary[component] = {
                    "current": latest_metrics.component_attribution[component],
                    "average": np.mean(values),
                    "trend": "increasing"
                    if len(values) > 1 and values[-1] > values[-2]
                    else "stable",
                }

        report = {
            "date": date.isoformat(),
            "generated_at": datetime.utcnow().isoformat(),
            "overall_status": latest_metrics.status.value,
            "summary": {
                "tracking_error_bps": latest_metrics.tracking_error_bps,
                "correlation": latest_metrics.correlation,
                "hit_rate": latest_metrics.hit_rate,
                "confidence_score": latest_metrics.confidence_score,
            },
            "performance": {
                "backtest_return": latest_metrics.backtest_return,
                "live_return": latest_metrics.live_return,
                "excess_return": latest_metrics.live_return - latest_metrics.backtest_return,
                "information_ratio": latest_metrics.information_ratio,
            },
            "trends": {
                "weekly_direction": weekly_trend["direction"],
                "weekly_magnitude": weekly_trend["magnitude"],
                "status_changes": len(
                    [m for m in self.parity_history[-7:] if m.status != latest_metrics.status]
                ),
            },
            "component_attribution": component_summary,
            "drift_alerts": [
                {
                    "type": alert.drift_type,
                    "magnitude": alert.drift_magnitude,
                    "confidence": alert.drift_confidence,
                    "affected_components": alert.affected_components,
                    "recommendations": alert.recommended_actions,
                }
                for alert in active_drift_alerts
            ],
            "recommendations": self._generate_recommendations(
                latest_metrics, weekly_trend, active_drift_alerts
            ),
            "data_quality": {
                "observation_count": latest_metrics.metadata.get("observation_count", 0),
                "data_quality_score": latest_metrics.metadata.get("data_quality_score", 0.0),
                "confidence_threshold_met": latest_metrics.confidence_score
                >= self.confidence_threshold,
            },
        }

        return report

    def _calculate_trend(self, metrics_list: List[ParityMetrics]) -> Dict[str, Any]:
        """Calculate trend direction and magnitude."""

        tracking_errors = [m.tracking_error_bps for m in metrics_list]

        if len(tracking_errors) < 2:
            return {"direction": "stable", "magnitude": 0.0}

        # Simple linear trend
        x = np.arange(len(tracking_errors))
        slope = np.polyfit(x, tracking_errors, 1)[0]

        magnitude = abs(slope)

        if slope > 1.0:  # Tracking error increasing
            direction = "deteriorating"
        elif slope < -1.0:  # Tracking error decreasing
            direction = "improving"
        else:
            direction = "stable"

        return {"direction": direction, "magnitude": magnitude}

    def _generate_recommendations(
        self,
        latest_metrics: ParityMetrics,
        weekly_trend: Dict[str, Any],
        drift_alerts: List[DriftDetection],
    ) -> List[str]:
        """Generate actionable recommendations."""

        recommendations = []

        # Status-based recommendations
        if latest_metrics.status == ParityStatus.CRITICAL:
            recommendations.append("URGENT: Disable live trading until parity is restored")
            recommendations.append("Conduct immediate investigation of model and data pipeline")
        elif latest_metrics.status == ParityStatus.WARNING:
            recommendations.append("Reduce position sizes by 50% until parity improves")
            recommendations.append("Increase monitoring frequency to hourly")

        # Trend-based recommendations
        if weekly_trend["direction"] == "deteriorating":
            recommendations.append("Investigate cause of degrading parity trend")
            if weekly_trend["magnitude"] > 5.0:
                recommendations.append("Consider model retraining or data source audit")

        # Correlation-based recommendations
        if latest_metrics.correlation < 0.3:
            recommendations.append("Review signal generation consistency between backtest and live")
            recommendations.append("Verify data feed alignment and timing")

        # Hit rate recommendations
        if latest_metrics.hit_rate < 0.4:
            recommendations.append("Analyze directional prediction accuracy")
            recommendations.append("Consider recalibrating prediction thresholds")

        # Component-specific recommendations
        max_component = max(latest_metrics.component_attribution.items(), key=lambda x: x[1])
        if max_component[1] > 20.0:  # Component > 20 bps
            if max_component[0] == "execution_costs":
                recommendations.append("Optimize execution strategy to reduce costs")
            elif max_component[0] == "slippage_impact":
                recommendations.append("Review order sizing and timing strategies")
            elif max_component[0] == "data_differences":
                recommendations.append("Audit data sources for consistency")

        # Drift-based recommendations
        for alert in drift_alerts:
            recommendations.extend(alert.recommended_actions)

        # Confidence-based recommendations
        if latest_metrics.confidence_score < self.confidence_threshold:
            recommendations.append("Increase data collection period for better analysis confidence")

        return list(set(recommendations))  # Remove duplicates

    def should_degrade_trading(self) -> Tuple[bool, str, List[str]]:
        """Determine if trading should be degraded based on parity analysis."""

        if not self.parity_history:
            return False, "No parity history available", []

        latest_metrics = self.parity_history[-1]

        # Critical status triggers immediate degradation
        if latest_metrics.status == ParityStatus.CRITICAL:
            return True, "Critical parity status", ["Disable live trading"]

        # Multiple warning periods trigger degradation
        recent_warnings = sum(
            1 for m in self.parity_history[-5:] if m.status == ParityStatus.WARNING
        )
        if recent_warnings >= 3:
            return (
                True,
                "Persistent parity warnings",
                ["Reduce position sizes", "Increase monitoring"],
            )

        # Drift detection triggers degradation
        recent_drift_alerts = [
            a
            for a in self.drift_alerts
            if (datetime.utcnow() - a.detection_time).days < 1 and a.drift_detected
        ]
        if recent_drift_alerts:
            high_confidence_drifts = [a for a in recent_drift_alerts if a.drift_confidence > 0.8]
            if high_confidence_drifts:
                return True, "High-confidence drift detected", ["Investigate model", "Reduce risk"]

        # Low confidence with poor metrics
        if (
            latest_metrics.confidence_score < 0.3
            and latest_metrics.correlation < 0.3
            and latest_metrics.hit_rate < 0.3
        ):
            return True, "Multiple poor quality metrics", ["Verify data pipeline", "Check model"]

        return False, "Parity within acceptable bounds", []


def create_parity_analyzer(
    tracking_error_threshold_bps: float = 20.0, drift_detection_window: int = 30
) -> ParityAnalyzer:
    """Create parity analyzer with specified thresholds."""
    return ParityAnalyzer(tracking_error_threshold_bps, drift_detection_window)
