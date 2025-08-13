#!/usr/bin/env python3
"""
Signal Quality Validator - Multi-Horizon Performance Validation
Implements enterprise-grade signal quality metrics with out-of-sample testing
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
from scipy import stats
import sklearn.metrics as metrics

from ..core.logging_manager import get_logger


class SignalHorizon(str, Enum):
    """Signal prediction horizons"""

    H1 = "1H"
    H24 = "24H"
    D7 = "7D"
    D30 = "30D"


class ValidationStatus(str, Enum):
    """Signal quality validation status"""

    PASSED = "passed"
    FAILED = "failed"
    INSUFFICIENT_DATA = "insufficient_data"
    ERROR = "error"


@dataclass
class SignalQualityConfig:
    """Configuration for signal quality validation"""

    # Precision metrics
    minimum_precision_at_k: float = 0.60  # 60% precision@K
    top_k_candidates: int = 10  # Top-K for precision calculation

    # Hit rate requirements
    minimum_hit_rate: float = 0.55  # 55% hit rate
    minimum_confidence_threshold: float = 0.80  # 80% confidence threshold

    # Calibration requirements
    maximum_mae_ratio: float = 0.25  # MAE ≤ 0.25 × median(|pred|)
    calibration_confidence_threshold: float = 0.80  # 80% confidence
    calibration_success_rate: float = 0.70  # 70% success rate

    # Strategy performance requirements
    minimum_sharpe_ratio: float = 1.0  # Sharpe ≥ 1.0
    maximum_drawdown: float = 0.15  # Max 15% drawdown

    # Transaction costs
    transaction_cost_percent: float = 0.0025  # 0.25% transaction costs
    slippage_percent: float = 0.001  # 0.1% slippage

    # Data requirements
    minimum_out_of_sample_days: int = 28  # 4 weeks minimum
    target_out_of_sample_days: int = 56  # 8 weeks target


@dataclass
class HorizonSignalMetrics:
    """Signal quality metrics for specific horizon"""

    horizon: SignalHorizon
    total_signals: int
    high_confidence_signals: int

    # Precision metrics
    precision_at_k: float
    top_k_used: int

    # Hit rate metrics
    hit_rate: float
    total_predictions: int
    successful_predictions: int

    # Calibration metrics
    mae_actual: float
    mae_threshold: float
    mae_ratio: float
    median_prediction_magnitude: float

    # Calibration test
    calibration_confidence_threshold: float
    calibration_predictions: int
    calibration_successes: int
    calibration_success_rate: float

    # Strategy performance
    sharpe_ratio: float
    max_drawdown: float
    total_return: float
    volatility: float

    # Status
    validation_status: ValidationStatus
    failed_criteria: List[str]
    passed_criteria: List[str]


@dataclass
class SignalQualityReport:
    """Comprehensive signal quality validation report"""

    validation_id: str
    timestamp: datetime
    out_of_sample_period_days: int
    out_of_sample_start: datetime
    out_of_sample_end: datetime

    # Per-horizon metrics
    horizon_metrics: Dict[SignalHorizon, HorizonSignalMetrics]

    # Overall validation
    overall_status: ValidationStatus
    horizons_passed: int
    horizons_failed: int

    # Summary statistics
    average_precision_at_k: float
    average_hit_rate: float
    average_sharpe_ratio: float
    worst_max_drawdown: float

    # Recommendations
    recommendations: List[str]
    critical_issues: List[str]


class SignalQualityValidator:
    """Multi-horizon signal quality validator"""

    def __init__(self, config: Optional[SignalQualityConfig] = None):
        self.config = config or SignalQualityConfig()
        self.logger = get_logger()

    async def validate_signal_quality(
        self, historical_signals: Dict[str, Any], market_data: Dict[str, Any]
    ) -> SignalQualityReport:
        """Validate signal quality across all horizons"""

        validation_id = f"signal_quality_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        validation_start = datetime.now()

        self.logger.info(f"Starting signal quality validation: {validation_id}")

        try:
            # Determine out-of-sample period
            out_of_sample_end = validation_start
            out_of_sample_start = out_of_sample_end - timedelta(
                days=self.config.target_out_of_sample_days
            )
            out_of_sample_days = (out_of_sample_end - out_of_sample_start).days

            # Validate each horizon
            horizon_metrics = {}
            horizons_passed = 0
            horizons_failed = 0

            for horizon in SignalHorizon:
                self.logger.info(f"Validating signals for horizon: {horizon.value}")

                horizon_signals = self._extract_horizon_signals(
                    historical_signals, horizon, out_of_sample_start, out_of_sample_end
                )

                if not horizon_signals:
                    self.logger.warning(f"No signals found for horizon {horizon.value}")
                    horizon_metrics[horizon] = self._create_empty_horizon_metrics(horizon)
                    horizons_failed += 1
                    continue

                # Calculate metrics for this horizon
                metrics_result = await self._calculate_horizon_metrics(
                    horizon, horizon_signals, market_data
                )

                horizon_metrics[horizon] = metrics_result

                if metrics_result.validation_status == ValidationStatus.PASSED:
                    horizons_passed += 1
                else:
                    horizons_failed += 1

            # Calculate overall statistics
            overall_stats = self._calculate_overall_statistics(horizon_metrics)

            # Determine overall status
            overall_status = (
                ValidationStatus.PASSED if horizons_failed == 0 else ValidationStatus.FAILED
            )
            if not horizon_metrics:
                overall_status = ValidationStatus.INSUFFICIENT_DATA

            # Generate recommendations
            recommendations, critical_issues = self._generate_recommendations(horizon_metrics)

            # Create validation report
            report = SignalQualityReport(
                validation_id=validation_id,
                timestamp=validation_start,
                out_of_sample_period_days=out_of_sample_days,
                out_of_sample_start=out_of_sample_start,
                out_of_sample_end=out_of_sample_end,
                horizon_metrics=horizon_metrics,
                overall_status=overall_status,
                horizons_passed=horizons_passed,
                horizons_failed=horizons_failed,
                average_precision_at_k=overall_stats["average_precision_at_k"],
                average_hit_rate=overall_stats["average_hit_rate"],
                average_sharpe_ratio=overall_stats["average_sharpe_ratio"],
                worst_max_drawdown=overall_stats["worst_max_drawdown"],
                recommendations=recommendations,
                critical_issues=critical_issues,
            )

            # Store validation report
            await self._store_validation_report(report)

            self.logger.info(
                f"Signal quality validation completed: {validation_id}",
                extra={
                    "validation_id": validation_id,
                    "overall_status": overall_status.value,
                    "horizons_passed": horizons_passed,
                    "horizons_failed": horizons_failed,
                    "out_of_sample_days": out_of_sample_days,
                },
            )

            return report

        except Exception as e:
            self.logger.error(f"Signal quality validation failed: {e}")

            return SignalQualityReport(
                validation_id=validation_id,
                timestamp=validation_start,
                out_of_sample_period_days=0,
                out_of_sample_start=validation_start,
                out_of_sample_end=validation_start,
                horizon_metrics={},
                overall_status=ValidationStatus.ERROR,
                horizons_passed=0,
                horizons_failed=len(SignalHorizon),
                average_precision_at_k=0.0,
                average_hit_rate=0.0,
                average_sharpe_ratio=0.0,
                worst_max_drawdown=1.0,
                recommendations=[],
                critical_issues=[f"Validation error: {e}"],
            )

    def _extract_horizon_signals(
        self,
        historical_signals: Dict[str, Any],
        horizon: SignalHorizon,
        start_date: datetime,
        end_date: datetime,
    ) -> List[Dict[str, Any]]:
        """Extract signals for specific horizon and time period"""

        try:
            # Extract signals for this horizon within the out-of-sample period
            horizon_signals = []

            for signal_id, signal_data in historical_signals.items():
                signal_timestamp = datetime.fromisoformat(signal_data.get("timestamp", ""))
                signal_horizon = signal_data.get("horizon", "")

                if signal_horizon == horizon.value and start_date <= signal_timestamp <= end_date:
                    horizon_signals.append(
                        {
                            "signal_id": signal_id,
                            "timestamp": signal_timestamp,
                            "symbol": signal_data.get("symbol", ""),
                            "predicted_return": signal_data.get("predicted_return", 0.0),
                            "confidence": signal_data.get("confidence", 0.0),
                            "actual_return": signal_data.get("actual_return"),  # May be None
                            "realized": signal_data.get("realized", False),
                            "metadata": signal_data.get("metadata", {}),
                        }
                    )

            return horizon_signals

        except Exception as e:
            self.logger.error(f"Failed to extract horizon signals for {horizon.value}: {e}")
            return []

    async def _calculate_horizon_metrics(
        self, horizon: SignalHorizon, signals: List[Dict[str, Any]], market_data: Dict[str, Any]
    ) -> HorizonSignalMetrics:
        """Calculate comprehensive metrics for horizon"""

        try:
            # Filter for high-confidence signals
            high_conf_signals = [
                s for s in signals if s["confidence"] >= self.config.minimum_confidence_threshold
            ]

            # Calculate precision@K
            precision_at_k, top_k_used = self._calculate_precision_at_k(signals)

            # Calculate hit rate
            hit_rate, total_preds, successful_preds = self._calculate_hit_rate(high_conf_signals)

            # Calculate MAE and calibration
            mae_metrics = self._calculate_mae_metrics(signals)

            # Calculate calibration test
            calibration_metrics = self._calculate_calibration_metrics(signals)

            # Calculate strategy performance
            strategy_metrics = await self._calculate_strategy_performance(signals, market_data)

            # Determine validation status
            failed_criteria = []
            passed_criteria = []

            # Check precision@K
            if precision_at_k >= self.config.minimum_precision_at_k:
                passed_criteria.append("precision_at_k")
            else:
                failed_criteria.append("precision_at_k")

            # Check hit rate
            if hit_rate >= self.config.minimum_hit_rate:
                passed_criteria.append("hit_rate")
            else:
                failed_criteria.append("hit_rate")

            # Check MAE calibration
            if mae_metrics["mae_ratio"] <= self.config.maximum_mae_ratio:
                passed_criteria.append("mae_calibration")
            else:
                failed_criteria.append("mae_calibration")

            # Check calibration test
            if calibration_metrics["success_rate"] >= self.config.calibration_success_rate:
                passed_criteria.append("calibration_test")
            else:
                failed_criteria.append("calibration_test")

            # Check Sharpe ratio
            if strategy_metrics["sharpe_ratio"] >= self.config.minimum_sharpe_ratio:
                passed_criteria.append("sharpe_ratio")
            else:
                failed_criteria.append("sharpe_ratio")

            # Check max drawdown
            if strategy_metrics["max_drawdown"] <= self.config.maximum_drawdown:
                passed_criteria.append("max_drawdown")
            else:
                failed_criteria.append("max_drawdown")

            # Determine overall status
            status = (
                ValidationStatus.PASSED if len(failed_criteria) == 0 else ValidationStatus.FAILED
            )

            if len(signals) < 10:  # Minimum signal count
                status = ValidationStatus.INSUFFICIENT_DATA
                failed_criteria.append("insufficient_signals")

            return HorizonSignalMetrics(
                horizon=horizon,
                total_signals=len(signals),
                high_confidence_signals=len(high_conf_signals),
                precision_at_k=precision_at_k,
                top_k_used=top_k_used,
                hit_rate=hit_rate,
                total_predictions=total_preds,
                successful_predictions=successful_preds,
                mae_actual=mae_metrics["mae_actual"],
                mae_threshold=mae_metrics["mae_threshold"],
                mae_ratio=mae_metrics["mae_ratio"],
                median_prediction_magnitude=mae_metrics["median_magnitude"],
                calibration_confidence_threshold=self.config.calibration_confidence_threshold,
                calibration_predictions=calibration_metrics["total_predictions"],
                calibration_successes=calibration_metrics["successes"],
                calibration_success_rate=calibration_metrics["success_rate"],
                sharpe_ratio=strategy_metrics["sharpe_ratio"],
                max_drawdown=strategy_metrics["max_drawdown"],
                total_return=strategy_metrics["total_return"],
                volatility=strategy_metrics["volatility"],
                validation_status=status,
                failed_criteria=failed_criteria,
                passed_criteria=passed_criteria,
            )

        except Exception as e:
            self.logger.error(f"Failed to calculate metrics for {horizon.value}: {e}")

            return self._create_empty_horizon_metrics(horizon, error=str(e))

    def _calculate_precision_at_k(self, signals: List[Dict[str, Any]]) -> Tuple[float, int]:
        """Calculate Precision@K metric"""

        try:
            # Sort signals by confidence (descending)
            sorted_signals = sorted(signals, key=lambda x: x["confidence"], reverse=True)

            # Take top-K
            top_k = min(self.config.top_k_candidates, len(sorted_signals))
            top_k_signals = sorted_signals[:top_k]

            if not top_k_signals:
                return 0.0, 0

            # Count successful predictions in top-K
            successful_count = 0
            for signal in top_k_signals:
                actual_return = signal.get("actual_return")
                predicted_return = signal.get("predicted_return", 0)

                if actual_return is not None and predicted_return > 0:
                    # Success if actual return is positive when predicted positive
                    if actual_return > 0:
                        successful_count += 1

            precision = successful_count / top_k if top_k > 0 else 0.0

            return precision, top_k

        except Exception as e:
            self.logger.error(f"Precision@K calculation failed: {e}")
            return 0.0, 0

    def _calculate_hit_rate(self, signals: List[Dict[str, Any]]) -> Tuple[float, int, int]:
        """Calculate hit rate for high-confidence signals"""

        try:
            total_predictions = 0
            successful_predictions = 0

            for signal in signals:
                predicted_return = signal.get("predicted_return", 0)
                actual_return = signal.get("actual_return")

                if predicted_return > 0 and actual_return is not None:
                    total_predictions += 1

                    if actual_return > 0:
                        successful_predictions += 1

            hit_rate = successful_predictions / total_predictions if total_predictions > 0 else 0.0

            return hit_rate, total_predictions, successful_predictions

        except Exception as e:
            self.logger.error(f"Hit rate calculation failed: {e}")
            return 0.0, 0, 0

    def _calculate_mae_metrics(self, signals: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate MAE calibration metrics"""

        try:
            predictions = []
            actuals = []

            for signal in signals:
                predicted_return = signal.get("predicted_return", 0)
                actual_return = signal.get("actual_return")

                if actual_return is not None:
                    predictions.append(predicted_return)
                    actuals.append(actual_return)

            if not predictions:
                return {
                    "mae_actual": float("inf"),
                    "mae_threshold": 0.0,
                    "mae_ratio": float("inf"),
                    "median_magnitude": 0.0,
                }

            # Calculate MAE
            mae_actual = np.mean(np.abs(np.array(predictions) - np.array(actuals)))

            # Calculate median prediction magnitude
            median_magnitude = np.median(np.abs(predictions))

            # Calculate threshold and ratio
            mae_threshold = self.config.maximum_mae_ratio * median_magnitude
            mae_ratio = mae_actual / median_magnitude if median_magnitude > 0 else float("inf")

            return {
                "mae_actual": mae_actual,
                "mae_threshold": mae_threshold,
                "mae_ratio": mae_ratio,
                "median_magnitude": median_magnitude,
            }

        except Exception as e:
            self.logger.error(f"MAE calculation failed: {e}")
            return {
                "mae_actual": float("inf"),
                "mae_threshold": 0.0,
                "mae_ratio": float("inf"),
                "median_magnitude": 0.0,
            }

    def _calculate_calibration_metrics(self, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate calibration test metrics"""

        try:
            # Filter for high-confidence predictions
            high_conf_signals = [
                s
                for s in signals
                if s["confidence"] >= self.config.calibration_confidence_threshold
            ]

            total_predictions = 0
            successes = 0

            for signal in high_conf_signals:
                predicted_return = signal.get("predicted_return", 0)
                actual_return = signal.get("actual_return")

                if predicted_return > 0 and actual_return is not None:
                    total_predictions += 1

                    if actual_return > 0:
                        successes += 1

            success_rate = successes / total_predictions if total_predictions > 0 else 0.0

            return {
                "total_predictions": total_predictions,
                "successes": successes,
                "success_rate": success_rate,
            }

        except Exception as e:
            self.logger.error(f"Calibration calculation failed: {e}")
            return {"total_predictions": 0, "successes": 0, "success_rate": 0.0}

    async def _calculate_strategy_performance(
        self, signals: List[Dict[str, Any]], market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate strategy performance with transaction costs"""

        try:
            # REMOVED: Mock data pattern not allowed in production
            returns = []
            portfolio_value = 1.0
            portfolio_history = [portfolio_value]

            for signal in sorted(signals, key=lambda x: x["timestamp"]):
                predicted_return = signal.get("predicted_return", 0)
                actual_return = signal.get("actual_return")
                confidence = signal.get("confidence", 0)

                if (
                    predicted_return > 0
                    and confidence >= self.config.minimum_confidence_threshold
                    and actual_return is not None
                ):
                    # Apply transaction costs and slippage
                    total_cost = self.config.transaction_cost_percent + self.config.slippage_percent
                    net_return = actual_return - (2 * total_cost)  # Entry + exit costs

                    returns.append(net_return)
                    portfolio_value *= 1 + net_return
                    portfolio_history.append(portfolio_value)

            if not returns:
                return {
                    "sharpe_ratio": 0.0,
                    "max_drawdown": 1.0,
                    "total_return": 0.0,
                    "volatility": 0.0,
                }

            # Calculate performance metrics
            mean_return = np.mean(returns)
            volatility = np.std(returns)

            # Sharpe ratio (annualized)
            sharpe_ratio = (mean_return / volatility) * np.sqrt(252) if volatility > 0 else 0.0

            # Max drawdown
            portfolio_peaks = np.maximum.accumulate(portfolio_history)
            drawdowns = (portfolio_peaks - portfolio_history) / portfolio_peaks
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0.0

            # Total return
            total_return = portfolio_value - 1.0

            return {
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "total_return": total_return,
                "volatility": volatility,
            }

        except Exception as e:
            self.logger.error(f"Strategy performance calculation failed: {e}")
            return {
                "sharpe_ratio": 0.0,
                "max_drawdown": 1.0,
                "total_return": 0.0,
                "volatility": 0.0,
            }

    def _create_empty_horizon_metrics(
        self, horizon: SignalHorizon, error: str = ""
    ) -> HorizonSignalMetrics:
        """Create empty metrics for horizon with insufficient data"""

        failed_criteria = ["insufficient_data"]
        if error:
            failed_criteria.append("calculation_error")

        return HorizonSignalMetrics(
            horizon=horizon,
            total_signals=0,
            high_confidence_signals=0,
            precision_at_k=0.0,
            top_k_used=0,
            hit_rate=0.0,
            total_predictions=0,
            successful_predictions=0,
            mae_actual=float("inf"),
            mae_threshold=0.0,
            mae_ratio=float("inf"),
            median_prediction_magnitude=0.0,
            calibration_confidence_threshold=self.config.calibration_confidence_threshold,
            calibration_predictions=0,
            calibration_successes=0,
            calibration_success_rate=0.0,
            sharpe_ratio=0.0,
            max_drawdown=1.0,
            total_return=0.0,
            volatility=0.0,
            validation_status=ValidationStatus.INSUFFICIENT_DATA,
            failed_criteria=failed_criteria,
            passed_criteria=[],
        )

    def _calculate_overall_statistics(
        self, horizon_metrics: Dict[SignalHorizon, HorizonSignalMetrics]
    ) -> Dict[str, float]:
        """Calculate overall statistics across all horizons"""

        if not horizon_metrics:
            return {
                "average_precision_at_k": 0.0,
                "average_hit_rate": 0.0,
                "average_sharpe_ratio": 0.0,
                "worst_max_drawdown": 1.0,
            }

        valid_metrics = [
            m
            for m in horizon_metrics.values()
            if m.validation_status != ValidationStatus.INSUFFICIENT_DATA
        ]

        if not valid_metrics:
            return {
                "average_precision_at_k": 0.0,
                "average_hit_rate": 0.0,
                "average_sharpe_ratio": 0.0,
                "worst_max_drawdown": 1.0,
            }

        return {
            "average_precision_at_k": np.mean([m.precision_at_k for m in valid_metrics]),
            "average_hit_rate": np.mean([m.hit_rate for m in valid_metrics]),
            "average_sharpe_ratio": np.mean([m.sharpe_ratio for m in valid_metrics]),
            "worst_max_drawdown": max([m.max_drawdown for m in valid_metrics]),
        }

    def _generate_recommendations(
        self, horizon_metrics: Dict[SignalHorizon, HorizonSignalMetrics]
    ) -> Tuple[List[str], List[str]]:
        """Generate recommendations and critical issues"""

        recommendations = []
        critical_issues = []

        # Analyze each horizon
        for horizon, metrics in horizon_metrics.items():
            if metrics.validation_status == ValidationStatus.FAILED:
                if "precision_at_k" in metrics.failed_criteria:
                    critical_issues.append(
                        f"{horizon.value}: Precision@{metrics.top_k_used} = {metrics.precision_at_k:.1%} < {self.config.minimum_precision_at_k:.1%}"
                    )
                    recommendations.append(
                        f"Improve {horizon.value} model accuracy - current precision too low"
                    )

                if "hit_rate" in metrics.failed_criteria:
                    critical_issues.append(
                        f"{horizon.value}: Hit rate = {metrics.hit_rate:.1%} < {self.config.minimum_hit_rate:.1%}"
                    )
                    recommendations.append(
                        f"Review {horizon.value} confidence thresholds - too many false positives"
                    )

                if "mae_calibration" in metrics.failed_criteria:
                    critical_issues.append(
                        f"{horizon.value}: MAE ratio = {metrics.mae_ratio:.2f} > {self.config.maximum_mae_ratio:.2f}"
                    )
                    recommendations.append(
                        f"Recalibrate {horizon.value} predictions - poor magnitude estimation"
                    )

                if "sharpe_ratio" in metrics.failed_criteria:
                    critical_issues.append(
                        f"{horizon.value}: Sharpe = {metrics.sharpe_ratio:.2f} < {self.config.minimum_sharpe_ratio:.2f}"
                    )
                    recommendations.append(f"Improve {horizon.value} risk-adjusted returns")

                if "max_drawdown" in metrics.failed_criteria:
                    critical_issues.append(
                        f"{horizon.value}: Max drawdown = {metrics.max_drawdown:.1%} > {self.config.maximum_drawdown:.1%}"
                    )
                    recommendations.append(f"Implement better risk management for {horizon.value}")

        # Overall recommendations
        failed_horizons = sum(
            1 for m in horizon_metrics.values() if m.validation_status == ValidationStatus.FAILED
        )

        if failed_horizons > len(horizon_metrics) / 2:
            recommendations.append("Consider fundamental model architecture review")
            recommendations.append("Increase out-of-sample validation period")

        if not recommendations:
            recommendations.append("All signal quality targets met - ready for live trading")

        return recommendations, critical_issues

    async def _store_validation_report(self, report: SignalQualityReport):
        """Store validation report to disk"""

        try:
            # Create directories
            reports_dir = Path("data/signal_quality_reports")
            reports_dir.mkdir(parents=True, exist_ok=True)

            # Prepare report data
            report_data = {
                "validation_id": report.validation_id,
                "timestamp": report.timestamp.isoformat(),
                "out_of_sample_period": {
                    "days": report.out_of_sample_period_days,
                    "start": report.out_of_sample_start.isoformat(),
                    "end": report.out_of_sample_end.isoformat(),
                },
                "overall_status": report.overall_status.value,
                "summary": {
                    "horizons_passed": report.horizons_passed,
                    "horizons_failed": report.horizons_failed,
                    "average_precision_at_k": report.average_precision_at_k,
                    "average_hit_rate": report.average_hit_rate,
                    "average_sharpe_ratio": report.average_sharpe_ratio,
                    "worst_max_drawdown": report.worst_max_drawdown,
                },
                "horizon_metrics": {
                    horizon.value: {
                        "validation_status": metrics.validation_status.value,
                        "total_signals": metrics.total_signals,
                        "high_confidence_signals": metrics.high_confidence_signals,
                        "precision_at_k": metrics.precision_at_k,
                        "hit_rate": metrics.hit_rate,
                        "mae_ratio": metrics.mae_ratio,
                        "calibration_success_rate": metrics.calibration_success_rate,
                        "sharpe_ratio": metrics.sharpe_ratio,
                        "max_drawdown": metrics.max_drawdown,
                        "failed_criteria": metrics.failed_criteria,
                        "passed_criteria": metrics.passed_criteria,
                    }
                    for horizon, metrics in report.horizon_metrics.items()
                },
                "recommendations": report.recommendations,
                "critical_issues": report.critical_issues,
            }

            # Write to file
            report_file = reports_dir / f"signal_quality_report_{report.validation_id}.json"

            with open(report_file, "w") as f:
                json.dump(report_data, f, indent=2)

            # Also store as latest
            latest_file = reports_dir / "latest_signal_quality_report.json"
            with open(latest_file, "w") as f:
                json.dump(report_data, f, indent=2)

            self.logger.info(f"Signal quality report stored: {report_file}")

        except Exception as e:
            self.logger.error(f"Failed to store signal quality report: {e}")


# Global instance
_signal_quality_validator = None


def get_signal_quality_validator(
    config: Optional[SignalQualityConfig] = None,
) -> SignalQualityValidator:
    """Get global signal quality validator instance"""
    global _signal_quality_validator
    if _signal_quality_validator is None:
        _signal_quality_validator = SignalQualityValidator(config)
    return _signal_quality_validator
