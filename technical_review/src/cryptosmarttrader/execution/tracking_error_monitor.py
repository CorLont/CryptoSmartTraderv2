"""
Tracking Error Monitor for Backtest-Live Parity

Comprehensive tracking error analysis between paper trading
and live execution with detailed attribution and reporting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from pathlib import Path
from scipy import stats

from .execution_simulator import ExecutionResult, Fill, OrderSide

logger = logging.getLogger(__name__)

class TrackingErrorComponent(Enum):
    """Components of tracking error"""
    SLIPPAGE = "slippage"
    FEES = "fees"
    TIMING = "timing"
    PARTIAL_FILLS = "partial_fills"
    MARKET_IMPACT = "market_impact"
    LATENCY = "latency"
    OTHER = "other"

@dataclass
class TradeComparison:
    """Comparison between paper and live trade"""
    trade_id: str
    symbol: str
    side: OrderSide
    size: float
    timestamp: datetime

    # Paper trading results
    paper_price: float
    paper_fees: float
    paper_total: float

    # Live execution results
    live_price: float
    live_fees: float
    live_total: float
    live_slippage_bps: float
    live_execution_time_ms: float

    # Tracking error breakdown
    price_difference: float
    fee_difference: float
    total_difference: float
    tracking_error_bps: float

    # Attribution
    error_components: Dict[TrackingErrorComponent, float] = field(default_factory=dict)

@dataclass
class TrackingErrorReport:
    """Comprehensive tracking error report"""
    period_start: datetime
    period_end: datetime
    total_trades: int

    # Overall metrics
    total_tracking_error_bps: float
    mean_tracking_error_bps: float
    std_tracking_error_bps: float
    max_tracking_error_bps: float

    # Statistical tests
    tracking_error_significance: float  # p-value
    tracking_error_distribution: str    # normal, skewed, etc.

    # Component attribution
    component_attribution: Dict[TrackingErrorComponent, float]

    # Performance metrics
    paper_total_pnl: float
    live_total_pnl: float
    pnl_difference: float

    # Quality metrics
    trades_within_threshold: int
    threshold_compliance_rate: float

class TrackingErrorMonitor:
    """
    Advanced tracking error monitoring and analysis system
    """

    def __init__(self,
                 max_tracking_error_bps: float = 20.0,
                 significance_level: float = 0.05):

        self.max_tracking_error_bps = max_tracking_error_bps
        self.significance_level = significance_level

        # Trade tracking
        self.trade_comparisons: List[TradeComparison] = []
        self.paper_trades: Dict[str, Dict[str, Any]] = {}
        self.live_executions: Dict[str, ExecutionResult] = {}

        # Rolling statistics
        self.rolling_window_days = 7
        self.daily_tracking_errors: List[Tuple[datetime, float]] = []

        # Alert thresholds
        self.alert_thresholds = {
            'single_trade_bps': 50.0,      # Single trade exceeds 50 bps
            'daily_average_bps': 25.0,     # Daily average exceeds 25 bps
            'weekly_average_bps': 20.0,    # Weekly average exceeds 20 bps
            'compliance_rate': 0.85        # Less than 85% within threshold
        }

        # Attribution weights for component analysis
        self.component_weights = {
            TrackingErrorComponent.SLIPPAGE: 0.4,
            TrackingErrorComponent.FEES: 0.2,
            TrackingErrorComponent.TIMING: 0.15,
            TrackingErrorComponent.PARTIAL_FILLS: 0.1,
            TrackingErrorComponent.MARKET_IMPACT: 0.1,
            TrackingErrorComponent.LATENCY: 0.05
        }

    def record_paper_trade(self,
                          trade_id: str,
                          symbol: str,
                          side: OrderSide,
                          size: float,
                          execution_price: float,
                          fees: float = 0.0,
                          timestamp: Optional[datetime] = None):
        """Record paper trading execution"""

        if timestamp is None:
            timestamp = datetime.now()

        self.paper_trades[trade_id] = {
            'symbol': symbol,
            'side': side,
            'size': size,
            'price': execution_price,
            'fees': fees,
            'total': size * execution_price + fees,
            'timestamp': timestamp
        }

        logger.debug(f"Recorded paper trade {trade_id}: {symbol} {side.value} {size} @ {execution_price}")

    def record_live_execution(self,
                            trade_id: str,
                            execution_result: ExecutionResult):
        """Record live execution result"""

        self.live_executions[trade_id] = execution_result

        # Create comparison if paper trade exists
        if trade_id in self.paper_trades:
            self._create_trade_comparison(trade_id)

        logger.debug(f"Recorded live execution {trade_id}: {execution_result.symbol}")

    def _create_trade_comparison(self, trade_id: str):
        """Create detailed trade comparison"""

        try:
            paper_trade = self.paper_trades[trade_id]
            live_execution = self.live_executions[trade_id]

            # Calculate differences
            price_diff = live_execution.average_price - paper_trade['price']
            fee_diff = live_execution.total_fees - paper_trade['fees']
            total_diff = (live_execution.average_price * live_execution.filled_size +
                         live_execution.total_fees) - paper_trade['total']

            # Calculate tracking error in basis points
            if paper_trade['total'] > 0:
                tracking_error_bps = (total_diff / paper_trade['total']) * 10000
            else:
                tracking_error_bps = 0.0

            # Create comparison
            comparison = TradeComparison(
                trade_id=trade_id,
                symbol=paper_trade['symbol'],
                side=paper_trade['side'],
                size=paper_trade['size'],
                timestamp=paper_trade['timestamp'],
                paper_price=paper_trade['price'],
                paper_fees=paper_trade['fees'],
                paper_total=paper_trade['total'],
                live_price=live_execution.average_price,
                live_fees=live_execution.total_fees,
                live_total=live_execution.average_price * live_execution.filled_size + live_execution.total_fees,
                live_slippage_bps=live_execution.slippage_bps,
                live_execution_time_ms=live_execution.execution_time_ms,
                price_difference=price_diff,
                fee_difference=fee_diff,
                total_difference=total_diff,
                tracking_error_bps=tracking_error_bps
            )

            # Perform component attribution
            comparison.error_components = self._attribute_tracking_error(
                comparison, live_execution
            )

            self.trade_comparisons.append(comparison)

            # Check alerts
            self._check_tracking_error_alerts(comparison)

            logger.info(f"Trade comparison created: {trade_id} TE={tracking_error_bps:.1f}bps")

        except Exception as e:
            logger.error(f"Failed to create trade comparison for {trade_id}: {e}")

    def _attribute_tracking_error(self,
                                comparison: TradeComparison,
                                live_execution: ExecutionResult) -> Dict[TrackingErrorComponent, float]:
        """Attribute tracking error to components"""

        components = {}
        total_error_bps = abs(comparison.tracking_error_bps)

        # Fee component
        if comparison.paper_total > 0:
            fee_component = abs(comparison.fee_difference) / comparison.paper_total * 10000
            components[TrackingErrorComponent.FEES] = fee_component
        else:
            components[TrackingErrorComponent.FEES] = 0.0

        # Slippage component
        components[TrackingErrorComponent.SLIPPAGE] = live_execution.slippage_bps

        # Market impact component
        components[TrackingErrorComponent.MARKET_IMPACT] = live_execution.market_impact_bps

        # Timing component (from execution delay)
        timing_component = min(5.0, live_execution.execution_time_ms / 100)  # Rough estimate
        components[TrackingErrorComponent.TIMING] = timing_component

        # Partial fill component
        if live_execution.partial_fill:
            partial_fill_component = (1 - live_execution.fill_rate) * 10  # Rough estimate
            components[TrackingErrorComponent.PARTIAL_FILLS] = partial_fill_component
        else:
            components[TrackingErrorComponent.PARTIAL_FILLS] = 0.0

        # Latency component
        latency_component = min(2.0, live_execution.execution_time_ms / 200)
        components[TrackingErrorComponent.LATENCY] = latency_component

        # Other component (residual)
        attributed_total = sum(components.values())
        if total_error_bps > attributed_total:
            components[TrackingErrorComponent.OTHER] = total_error_bps - attributed_total
        else:
            components[TrackingErrorComponent.OTHER] = 0.0

        return components

    def _check_tracking_error_alerts(self, comparison: TradeComparison):
        """Check if tracking error exceeds alert thresholds"""

        abs_error = abs(comparison.tracking_error_bps)

        # Single trade alert
        if abs_error > self.alert_thresholds['single_trade_bps']:
            logger.warning(f"High tracking error alert: {comparison.trade_id} "
                          f"{abs_error:.1f}bps > {self.alert_thresholds['single_trade_bps']}bps")

        # Daily average alert
        self._check_daily_average_alert()

    def _check_daily_average_alert(self):
        """Check daily average tracking error"""

        today = datetime.now().date()
        today_comparisons = [
            comp for comp in self.trade_comparisons
            if comp.timestamp.date() == today
        ]

        if len(today_comparisons) >= 5:  # Minimum trades for meaningful average
            avg_error = np.mean([abs(comp.tracking_error_bps) for comp in today_comparisons])

            if avg_error > self.alert_thresholds['daily_average_bps']:
                logger.warning(f"High daily average tracking error: {avg_error:.1f}bps")

    def calculate_tracking_error_statistics(self,
                                          days_back: int = 7) -> Dict[str, Any]:
        """Calculate comprehensive tracking error statistics"""

        cutoff_time = datetime.now() - timedelta(days=days_back)
        recent_comparisons = [
            comp for comp in self.trade_comparisons
            if comp.timestamp >= cutoff_time
        ]

        if not recent_comparisons:
            return {"status": "no_data", "period_days": days_back}

        # Extract tracking errors
        tracking_errors = [comp.tracking_error_bps for comp in recent_comparisons]
        abs_tracking_errors = [abs(te) for te in tracking_errors]

        # Basic statistics
        stats_dict = {
            "period_days": days_back,
            "total_trades": len(recent_comparisons),
            "mean_tracking_error_bps": np.mean(tracking_errors),
            "mean_abs_tracking_error_bps": np.mean(abs_tracking_errors),
            "std_tracking_error_bps": np.std(tracking_errors),
            "max_abs_tracking_error_bps": np.max(abs_tracking_errors),
            "min_tracking_error_bps": np.min(tracking_errors),
            "max_tracking_error_bps": np.max(tracking_errors)
        }

        # Percentiles
        stats_dict.update({
            "p50_abs_tracking_error_bps": np.percentile(abs_tracking_errors, 50),
            "p75_abs_tracking_error_bps": np.percentile(abs_tracking_errors, 75),
            "p90_abs_tracking_error_bps": np.percentile(abs_tracking_errors, 90),
            "p95_abs_tracking_error_bps": np.percentile(abs_tracking_errors, 95),
            "p99_abs_tracking_error_bps": np.percentile(abs_tracking_errors, 99)
        })

        # Threshold compliance
        within_threshold = sum(1 for te in abs_tracking_errors
                              if te <= self.max_tracking_error_bps)
        compliance_rate = within_threshold / len(abs_tracking_errors)

        stats_dict.update({
            "threshold_bps": self.max_tracking_error_bps,
            "trades_within_threshold": within_threshold,
            "compliance_rate": compliance_rate
        })

        # Statistical tests
        if len(tracking_errors) >= 10:
            # Normality test
            _, normality_p = stats.shapiro(tracking_errors)
            stats_dict["normality_test_p_value"] = normality_p
            stats_dict["is_normal_distribution"] = normality_p > 0.05

            # Test if mean tracking error is significantly different from zero
            _, ttest_p = stats.ttest_1samp(tracking_errors, 0)
            stats_dict["mean_zero_test_p_value"] = ttest_p
            stats_dict["mean_significantly_nonzero"] = ttest_p < self.significance_level

        return stats_dict

    def get_component_attribution(self, days_back: int = 7) -> Dict[str, Any]:
        """Get tracking error component attribution analysis"""

        cutoff_time = datetime.now() - timedelta(days=days_back)
        recent_comparisons = [
            comp for comp in self.trade_comparisons
            if comp.timestamp >= cutoff_time and comp.error_components
        ]

        if not recent_comparisons:
            return {"status": "no_data"}

        # Aggregate component contributions
        component_totals = {}
        for comp in recent_comparisons:
            for component_type, value in comp.error_components.items():
                if component_type not in component_totals:
                    component_totals[component_type] = []
                component_totals[component_type].append(value)

        # Calculate statistics for each component
        component_stats = {}
        total_attributed = 0

        for component_type, values in component_totals.items():
            mean_value = np.mean(values)
            component_stats[component_type.value] = {
                "mean_bps": mean_value,
                "std_bps": np.std(values),
                "max_bps": np.max(values),
                "contribution_percentage": 0  # Will calculate after getting total
            }
            total_attributed += mean_value

        # Calculate contribution percentages
        for component_type in component_stats:
            if total_attributed > 0:
                contribution_pct = (component_stats[component_type]["mean_bps"] /
                                  total_attributed) * 100
                component_stats[component_type]["contribution_percentage"] = contribution_pct

        return {
            "period_days": days_back,
            "total_trades": len(recent_comparisons),
            "total_attributed_error_bps": total_attributed,
            "component_breakdown": component_stats
        }

    def generate_tracking_error_report(self, days_back: int = 7) -> TrackingErrorReport:
        """Generate comprehensive tracking error report"""

        cutoff_time = datetime.now() - timedelta(days=days_back)
        recent_comparisons = [
            comp for comp in self.trade_comparisons
            if comp.timestamp >= cutoff_time
        ]

        if not recent_comparisons:
            # Return empty report
            return TrackingErrorReport(
                period_start=cutoff_time,
                period_end=datetime.now(),
                total_trades=0,
                total_tracking_error_bps=0.0,
                mean_tracking_error_bps=0.0,
                std_tracking_error_bps=0.0,
                max_tracking_error_bps=0.0,
                tracking_error_significance=1.0,
                tracking_error_distribution="unknown",
                component_attribution={},
                paper_total_pnl=0.0,
                live_total_pnl=0.0,
                pnl_difference=0.0,
                trades_within_threshold=0,
                threshold_compliance_rate=0.0
            )

        # Calculate basic metrics
        tracking_errors = [comp.tracking_error_bps for comp in recent_comparisons]
        abs_tracking_errors = [abs(te) for te in tracking_errors]

        # PnL calculations
        paper_pnl = sum(comp.paper_total for comp in recent_comparisons)
        live_pnl = sum(comp.live_total for comp in recent_comparisons)

        # Component attribution
        component_attribution = {}
        for comp in recent_comparisons:
            for component_type, value in comp.error_components.items():
                if component_type not in component_attribution:
                    component_attribution[component_type] = []
                component_attribution[component_type].append(value)

        # Average component contributions
        avg_component_attribution = {}
        for component_type, values in component_attribution.items():
            avg_component_attribution[component_type] = np.mean(values)

        # Statistical tests
        tracking_error_significance = 1.0
        tracking_error_distribution = "unknown"

        if len(tracking_errors) >= 10:
            _, ttest_p = stats.ttest_1samp(tracking_errors, 0)
            tracking_error_significance = ttest_p

            _, normality_p = stats.shapiro(tracking_errors)
            if normality_p > 0.05:
                tracking_error_distribution = "normal"
            else:
                skewness = stats.skew(tracking_errors)
                if abs(skewness) > 1:
                    tracking_error_distribution = "highly_skewed"
                else:
                    tracking_error_distribution = "moderately_skewed"

        # Threshold compliance
        within_threshold = sum(1 for te in abs_tracking_errors
                              if te <= self.max_tracking_error_bps)
        compliance_rate = within_threshold / len(abs_tracking_errors)

        return TrackingErrorReport(
            period_start=cutoff_time,
            period_end=datetime.now(),
            total_trades=len(recent_comparisons),
            total_tracking_error_bps=sum(abs_tracking_errors),
            mean_tracking_error_bps=np.mean(tracking_errors),
            std_tracking_error_bps=np.std(tracking_errors),
            max_tracking_error_bps=np.max(abs_tracking_errors),
            tracking_error_significance=tracking_error_significance,
            tracking_error_distribution=tracking_error_distribution,
            component_attribution=avg_component_attribution,
            paper_total_pnl=paper_pnl,
            live_total_pnl=live_pnl,
            pnl_difference=live_pnl - paper_pnl,
            trades_within_threshold=within_threshold,
            threshold_compliance_rate=compliance_rate
        )

    def export_detailed_report(self,
                              filepath: str,
                              days_back: int = 7,
                              include_individual_trades: bool = True):
        """Export detailed tracking error report to file"""

        try:
            report = self.generate_tracking_error_report(days_back)
            stats = self.calculate_tracking_error_statistics(days_back)
            attribution = self.get_component_attribution(days_back)

            export_data = {
                "report_timestamp": datetime.now().isoformat(),
                "period_days": days_back,
                "summary_report": {
                    "period_start": report.period_start.isoformat(),
                    "period_end": report.period_end.isoformat(),
                    "total_trades": report.total_trades,
                    "mean_tracking_error_bps": report.mean_tracking_error_bps,
                    "std_tracking_error_bps": report.std_tracking_error_bps,
                    "max_tracking_error_bps": report.max_tracking_error_bps,
                    "compliance_rate": report.threshold_compliance_rate,
                    "pnl_difference": report.pnl_difference
                },
                "detailed_statistics": stats,
                "component_attribution": attribution
            }

            # Include individual trades if requested
            if include_individual_trades:
                cutoff_time = datetime.now() - timedelta(days=days_back)
                recent_comparisons = [
                    comp for comp in self.trade_comparisons
                    if comp.timestamp >= cutoff_time
                ]

                trade_details = []
                for comp in recent_comparisons:
                    trade_details.append({
                        "trade_id": comp.trade_id,
                        "symbol": comp.symbol,
                        "side": comp.side.value,
                        "size": comp.size,
                        "timestamp": comp.timestamp.isoformat(),
                        "paper_price": comp.paper_price,
                        "live_price": comp.live_price,
                        "tracking_error_bps": comp.tracking_error_bps,
                        "components": {k.value: v for k, v in comp.error_components.items()}
                    })

                export_data["individual_trades"] = trade_details

            # Write to file
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

            logger.info(f"Tracking error report exported to {filepath}")

        except Exception as e:
            logger.error(f"Failed to export tracking error report: {e}")

    def get_live_dashboard_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics for dashboard display"""

        # Last 24 hours
        stats_24h = self.calculate_tracking_error_statistics(days_back=1)

        # Last 7 days
        stats_7d = self.calculate_tracking_error_statistics(days_back=7)

        # Component attribution (7 days)
        attribution = self.get_component_attribution(days_back=7)

        # Recent alerts
        recent_alerts = []
        cutoff_time = datetime.now() - timedelta(hours=24)
        for comp in self.trade_comparisons:
            if (comp.timestamp >= cutoff_time and
                abs(comp.tracking_error_bps) > self.alert_thresholds['single_trade_bps']):
                recent_alerts.append({
                    "trade_id": comp.trade_id,
                    "symbol": comp.symbol,
                    "tracking_error_bps": comp.tracking_error_bps,
                    "timestamp": comp.timestamp.isoformat()
                })

        return {
            "current_status": {
                "total_tracked_trades": len(self.trade_comparisons),
                "monitoring_active": True,
                "last_update": datetime.now().isoformat()
            },
            "performance_24h": stats_24h,
            "performance_7d": stats_7d,
            "component_attribution": attribution,
            "recent_alerts": recent_alerts[-10:],  # Last 10 alerts
            "thresholds": self.alert_thresholds
        }
