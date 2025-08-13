"""
Daily Parity Reporter for CryptoSmartTrader
Comprehensive daily tracking error monitoring and auto-disable on drift.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd

from ..core.structured_logger import get_logger
from .parity_analyzer import ParityAnalyzer, ParityMetrics, ParityStatus, DriftDetection
from .execution_simulator import ExecutionSimulator, ExecutionResult
from ..analysis.backtest_parity import BacktestParityAnalyzer


class SystemAction(Enum):
    """System actions based on parity analysis."""
    CONTINUE = "continue"           # Normal operation
    WARNING = "warning"             # Issue warning
    REDUCE_SIZE = "reduce_size"     # Reduce position sizes
    DISABLE_TRADING = "disable"     # Disable trading
    EMERGENCY_STOP = "emergency"    # Emergency stop all operations


@dataclass
class DailyParityReport:
    """Daily parity report structure."""
    date: str
    tracking_error_bps: float
    correlation: float
    hit_rate: float
    execution_quality_score: float
    parity_status: ParityStatus
    system_action: SystemAction
    component_attribution: Dict[str, float]
    drift_alerts: List[DriftDetection]
    execution_statistics: Dict[str, float]
    recommendations: List[str]
    next_check_time: datetime
    metadata: Dict[str, Any]


@dataclass
class ParityConfiguration:
    """Configuration for parity monitoring."""
    # Tracking error thresholds (bps)
    warning_threshold_bps: float = 20.0
    critical_threshold_bps: float = 50.0
    emergency_threshold_bps: float = 100.0
    
    # Correlation thresholds
    min_correlation_warning: float = 0.7
    min_correlation_critical: float = 0.5
    
    # Hit rate thresholds
    min_hit_rate_warning: float = 0.6
    min_hit_rate_critical: float = 0.4
    
    # Drift detection
    max_consecutive_warnings: int = 3
    max_consecutive_critical: int = 2
    drift_lookback_days: int = 7
    
    # Auto-disable settings
    auto_disable_on_drift: bool = True
    auto_reduce_size_threshold: float = 30.0
    emergency_stop_threshold: float = 80.0
    
    # Reporting
    daily_report_time: str = "08:00"  # UTC time for daily reports
    slack_notifications: bool = True
    email_notifications: bool = True


class DailyParityReporter:
    """
    Enterprise daily parity monitoring and reporting system.
    
    Features:
    - Automated daily tracking error analysis
    - Drift detection with configurable thresholds
    - Automatic trading disable on significant drift
    - Component-level attribution reporting
    - Slack/email notifications
    - Historical parity trend analysis
    - Emergency stop capabilities
    """
    
    def __init__(self, config: Optional[ParityConfiguration] = None):
        self.config = config or ParityConfiguration()
        self.logger = get_logger("daily_parity_reporter")
        
        # Core analyzers
        self.parity_analyzer = ParityAnalyzer(
            tracking_error_threshold_bps=self.config.warning_threshold_bps
        )
        self.backtest_analyzer = BacktestParityAnalyzer(
            target_tracking_error_bps=self.config.warning_threshold_bps
        )
        self.execution_simulator = ExecutionSimulator()
        
        # Report storage
        self.reports_path = Path("data/parity_reports")
        self.reports_path.mkdir(parents=True, exist_ok=True)
        
        # State tracking
        self.consecutive_warnings = 0
        self.consecutive_critical = 0
        self.trading_disabled = False
        self.last_report_date: Optional[datetime] = None
        
        # Historical data
        self.daily_reports: List[DailyParityReport] = []
        self.load_historical_reports()
        
        # Performance tracking
        self.tracking_error_history: List[float] = []
        self.correlation_history: List[float] = []
        
        self.logger.info("Daily Parity Reporter initialized", 
                        config=asdict(self.config))
    
    async def generate_daily_report(self, 
                                  backtest_data: Dict[str, Any],
                                  live_data: Dict[str, Any],
                                  execution_results: Optional[List[ExecutionResult]] = None,
                                  force_date: Optional[datetime] = None) -> DailyParityReport:
        """
        Generate comprehensive daily parity report.
        
        Args:
            backtest_data: Backtest performance data
            live_data: Live trading performance data
            execution_results: Optional execution results
            force_date: Force specific date (for testing)
            
        Returns:
            DailyParityReport with full analysis
        """
        report_date = force_date or datetime.utcnow()
        period_start = report_date - timedelta(days=1)
        period_end = report_date
        
        self.logger.info("Generating daily parity report", 
                        date=report_date.strftime("%Y-%m-%d"))
        
        # Core parity analysis
        parity_metrics = self.parity_analyzer.analyze_parity(
            backtest_data=backtest_data,
            live_data=live_data,
            period_start=period_start,
            period_end=period_end,
            execution_results=execution_results
        )
        
        # Execution quality analysis
        execution_stats = self._analyze_execution_quality(execution_results)
        
        # Drift detection
        drift_alerts = await self._perform_drift_analysis(parity_metrics)
        
        # Determine system action
        system_action = self._determine_system_action(parity_metrics, drift_alerts)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(parity_metrics, drift_alerts, system_action)
        
        # Create daily report
        daily_report = DailyParityReport(
            date=report_date.strftime("%Y-%m-%d"),
            tracking_error_bps=parity_metrics.tracking_error_bps,
            correlation=parity_metrics.correlation,
            hit_rate=parity_metrics.hit_rate,
            execution_quality_score=parity_metrics.confidence_score,
            parity_status=parity_metrics.status,
            system_action=system_action,
            component_attribution=parity_metrics.component_attribution,
            drift_alerts=drift_alerts,
            execution_statistics=execution_stats,
            recommendations=recommendations,
            next_check_time=report_date + timedelta(days=1),
            metadata={
                'report_generation_time': datetime.utcnow().isoformat(),
                'data_quality_score': parity_metrics.metadata.get('data_quality_score', 0.8),
                'observation_count': parity_metrics.metadata.get('observation_count', 0),
                'consecutive_warnings': self.consecutive_warnings,
                'consecutive_critical': self.consecutive_critical,
                'trading_disabled': self.trading_disabled
            }
        )
        
        # Update state
        self._update_state(daily_report)
        
        # Save report
        await self._save_report(daily_report)
        
        # Execute system actions
        await self._execute_system_action(system_action, daily_report)
        
        # Send notifications
        await self._send_notifications(daily_report)
        
        self.logger.info("Daily parity report generated", 
                        tracking_error_bps=parity_metrics.tracking_error_bps,
                        system_action=system_action.value,
                        parity_status=parity_metrics.status.value)
        
        return daily_report
    
    def _analyze_execution_quality(self, 
                                 execution_results: Optional[List[ExecutionResult]]) -> Dict[str, float]:
        """Analyze execution quality metrics."""
        if not execution_results:
            return {
                'avg_slippage_bps': 0.0,
                'avg_fees_bps': 0.0,
                'fill_rate': 0.0,
                'avg_latency_ms': 0.0,
                'execution_count': 0
            }
        
        total_slippage = sum(result.slippage_bps for result in execution_results)
        total_fees = sum(result.total_fees for result in execution_results)
        total_volume = sum(result.executed_quantity * result.avg_fill_price for result in execution_results)
        filled_orders = sum(1 for result in execution_results if result.executed_quantity > 0)
        total_latency = sum(result.latency_ms for result in execution_results)
        
        return {
            'avg_slippage_bps': total_slippage / len(execution_results),
            'avg_fees_bps': (total_fees / total_volume * 10000) if total_volume > 0 else 0.0,
            'fill_rate': filled_orders / len(execution_results),
            'avg_latency_ms': total_latency / len(execution_results),
            'execution_count': len(execution_results)
        }
    
    async def _perform_drift_analysis(self, parity_metrics: ParityMetrics) -> List[DriftDetection]:
        """Perform comprehensive drift analysis."""
        drift_alerts = []
        
        # Update historical tracking
        self.tracking_error_history.append(parity_metrics.tracking_error_bps)
        self.correlation_history.append(parity_metrics.correlation)
        
        # Keep only recent history
        max_history = self.config.drift_lookback_days
        self.tracking_error_history = self.tracking_error_history[-max_history:]
        self.correlation_history = self.correlation_history[-max_history:]
        
        # Statistical drift detection
        if len(self.tracking_error_history) >= 3:
            # Tracking error drift
            recent_te = np.mean(self.tracking_error_history[-3:])
            baseline_te = np.mean(self.tracking_error_history[:-3]) if len(self.tracking_error_history) > 3 else recent_te
            
            te_drift_pct = ((recent_te - baseline_te) / baseline_te * 100) if baseline_te > 0 else 0
            
            if abs(te_drift_pct) > 50:  # 50% drift threshold
                drift_alerts.append(DriftDetection(
                    drift_detected=True,
                    drift_type="tracking_error_drift",
                    drift_magnitude=float(te_drift_pct),
                    drift_confidence=0.8,
                    detection_time=datetime.utcnow(),
                    affected_components=["execution", "timing"],
                    recommended_actions=["investigate_execution", "check_market_regime"],
                    metadata={'recent_te': recent_te, 'baseline_te': baseline_te}
                ))
        
        # Correlation drift detection
        if len(self.correlation_history) >= 3:
            recent_corr = np.mean(self.correlation_history[-3:])
            baseline_corr = np.mean(self.correlation_history[:-3]) if len(self.correlation_history) > 3 else recent_corr
            
            corr_drift = baseline_corr - recent_corr
            
            if corr_drift > 0.2:  # 20% correlation drop
                drift_alerts.append(DriftDetection(
                    drift_detected=True,
                    drift_type="correlation_drift",
                    drift_magnitude=float(corr_drift),
                    drift_confidence=0.9,
                    detection_time=datetime.utcnow(),
                    affected_components=["model", "data"],
                    recommended_actions=["retrain_models", "validate_data_sources"],
                    metadata={'recent_corr': recent_corr, 'baseline_corr': baseline_corr}
                ))
        
        return drift_alerts
    
    def _determine_system_action(self, 
                               parity_metrics: ParityMetrics, 
                               drift_alerts: List[DriftDetection]) -> SystemAction:
        """Determine appropriate system action based on analysis."""
        
        # Emergency conditions
        if (parity_metrics.tracking_error_bps > self.config.emergency_threshold_bps or
            parity_metrics.correlation < 0.3 or
            parity_metrics.hit_rate < 0.3):
            return SystemAction.EMERGENCY_STOP
        
        # Critical conditions
        if (parity_metrics.tracking_error_bps > self.config.critical_threshold_bps or
            parity_metrics.correlation < self.config.min_correlation_critical or
            parity_metrics.hit_rate < self.config.min_hit_rate_critical or
            len([d for d in drift_alerts if d.drift_confidence > 0.8]) > 0):
            
            self.consecutive_critical += 1
            if self.consecutive_critical >= self.config.max_consecutive_critical:
                return SystemAction.DISABLE_TRADING
            else:
                return SystemAction.REDUCE_SIZE
        
        # Warning conditions
        if (parity_metrics.tracking_error_bps > self.config.warning_threshold_bps or
            parity_metrics.correlation < self.config.min_correlation_warning or
            parity_metrics.hit_rate < self.config.min_hit_rate_warning or
            len(drift_alerts) > 0):
            
            self.consecutive_warnings += 1
            if self.consecutive_warnings >= self.config.max_consecutive_warnings:
                return SystemAction.REDUCE_SIZE
            else:
                return SystemAction.WARNING
        
        # Reset counters on good performance
        self.consecutive_warnings = 0
        self.consecutive_critical = 0
        
        return SystemAction.CONTINUE
    
    def _generate_recommendations(self, 
                                parity_metrics: ParityMetrics,
                                drift_alerts: List[DriftDetection],
                                system_action: SystemAction) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Tracking error recommendations
        if parity_metrics.tracking_error_bps > self.config.warning_threshold_bps:
            recommendations.append(f"High tracking error ({parity_metrics.tracking_error_bps:.1f} bps). Investigate execution costs and timing differences.")
        
        # Correlation recommendations
        if parity_metrics.correlation < self.config.min_correlation_warning:
            recommendations.append(f"Low correlation ({parity_metrics.correlation:.2f}). Check model performance and data quality.")
        
        # Hit rate recommendations
        if parity_metrics.hit_rate < self.config.min_hit_rate_warning:
            recommendations.append(f"Low hit rate ({parity_metrics.hit_rate:.2f}). Review signal generation and market regime detection.")
        
        # Component-specific recommendations
        attribution = parity_metrics.component_attribution
        if attribution.get('execution_costs', 0) > 20:
            recommendations.append("High execution costs detected. Optimize order routing and timing.")
        
        if attribution.get('timing_differences', 0) > 15:
            recommendations.append("Significant timing differences. Check latency and order processing delays.")
        
        # Drift-specific recommendations
        for alert in drift_alerts:
            recommendations.extend(alert.recommended_actions)
        
        # System action recommendations
        if system_action == SystemAction.REDUCE_SIZE:
            recommendations.append("Reducing position sizes by 50% until parity improves.")
        elif system_action == SystemAction.DISABLE_TRADING:
            recommendations.append("Trading disabled due to poor parity. Manual intervention required.")
        elif system_action == SystemAction.EMERGENCY_STOP:
            recommendations.append("EMERGENCY STOP: Severe parity breakdown detected. Immediate investigation required.")
        
        return recommendations
    
    def _update_state(self, report: DailyParityReport) -> None:
        """Update internal state based on report."""
        self.daily_reports.append(report)
        self.last_report_date = datetime.fromisoformat(report.date)
        
        # Update trading status
        if report.system_action in [SystemAction.DISABLE_TRADING, SystemAction.EMERGENCY_STOP]:
            self.trading_disabled = True
        elif report.system_action == SystemAction.CONTINUE and self.consecutive_warnings == 0:
            self.trading_disabled = False
        
        # Keep only recent reports
        self.daily_reports = self.daily_reports[-30:]  # Keep 30 days
    
    async def _save_report(self, report: DailyParityReport) -> None:
        """Save report to persistent storage."""
        report_file = self.reports_path / f"parity_report_{report.date}.json"
        
        # Convert dataclasses to dict for JSON serialization
        report_dict = asdict(report)
        
        # Handle datetime serialization
        report_dict['next_check_time'] = report.next_check_time.isoformat()
        
        # Handle drift alerts
        drift_alerts_serialized = []
        for alert in report.drift_alerts:
            alert_dict = asdict(alert)
            alert_dict['detection_time'] = alert.detection_time.isoformat()
            drift_alerts_serialized.append(alert_dict)
        
        report_dict['drift_alerts'] = drift_alerts_serialized
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
                
            self.logger.info("Parity report saved", file=str(report_file))
        except Exception as e:
            self.logger.error(f"Failed to save parity report: {e}")
    
    async def _execute_system_action(self, 
                                   action: SystemAction, 
                                   report: DailyParityReport) -> None:
        """Execute system actions based on parity analysis."""
        
        if action == SystemAction.CONTINUE:
            return
        
        self.logger.warning(f"Executing system action: {action.value}", 
                          tracking_error=report.tracking_error_bps,
                          correlation=report.correlation)
        
        # Create action file for system to read
        action_file = Path("data/system_actions/parity_action.json")
        action_file.parent.mkdir(parents=True, exist_ok=True)
        
        action_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action.value,
            'reason': 'parity_drift_detection',
            'tracking_error_bps': report.tracking_error_bps,
            'correlation': report.correlation,
            'recommendations': report.recommendations,
            'expires_at': (datetime.utcnow() + timedelta(hours=24)).isoformat()
        }
        
        try:
            with open(action_file, 'w') as f:
                json.dump(action_data, f, indent=2)
                
            self.logger.info("System action file created", action=action.value)
        except Exception as e:
            self.logger.error(f"Failed to create system action file: {e}")
    
    async def _send_notifications(self, report: DailyParityReport) -> None:
        """Send notifications based on report status."""
        
        # Only send notifications for warnings and above
        if report.system_action == SystemAction.CONTINUE:
            return
        
        message = self._format_notification_message(report)
        
        # Log notification (actual Slack/email would be implemented here)
        self.logger.warning("Parity notification", 
                          notification_message=message,
                          action=report.system_action.value)
        
        # TODO: Implement actual Slack/email notifications
        # if self.config.slack_notifications:
        #     await self._send_slack_notification(message)
        # if self.config.email_notifications:
        #     await self._send_email_notification(message)
    
    def _format_notification_message(self, report: DailyParityReport) -> str:
        """Format notification message."""
        status_emoji = {
            SystemAction.WARNING: "⚠️",
            SystemAction.REDUCE_SIZE: "🔄",
            SystemAction.DISABLE_TRADING: "🛑",
            SystemAction.EMERGENCY_STOP: "🚨"
        }
        
        emoji = status_emoji.get(report.system_action, "📊")
        
        message = f"""
{emoji} CryptoSmartTrader Parity Alert

Date: {report.date}
Tracking Error: {report.tracking_error_bps:.1f} bps
Correlation: {report.correlation:.2f}
Hit Rate: {report.hit_rate:.2f}
Status: {report.parity_status.value.upper()}
Action: {report.system_action.value.upper()}

Drift Alerts: {len(report.drift_alerts)}
Recommendations: {len(report.recommendations)}

Top Recommendations:
{chr(10).join(f"• {rec}" for rec in report.recommendations[:3])}
        """.strip()
        
        return message
    
    def load_historical_reports(self) -> None:
        """Load historical reports from storage."""
        try:
            for report_file in self.reports_path.glob("parity_report_*.json"):
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                
                # TODO: Implement proper deserialization of reports
                # For now, just track basic metrics
                if 'tracking_error_bps' in report_data:
                    self.tracking_error_history.append(report_data['tracking_error_bps'])
                if 'correlation' in report_data:
                    self.correlation_history.append(report_data['correlation'])
                    
        except Exception as e:
            self.logger.warning(f"Failed to load historical reports: {e}")
    
    def get_parity_trend(self, days: int = 7) -> Dict[str, Any]:
        """Get parity trend analysis for specified days."""
        recent_reports = self.daily_reports[-days:] if len(self.daily_reports) >= days else self.daily_reports
        
        if not recent_reports:
            return {'error': 'No historical data available'}
        
        tracking_errors = [report.tracking_error_bps for report in recent_reports]
        correlations = [report.correlation for report in recent_reports]
        
        return {
            'period_days': len(recent_reports),
            'avg_tracking_error_bps': np.mean(tracking_errors),
            'max_tracking_error_bps': np.max(tracking_errors),
            'avg_correlation': np.mean(correlations),
            'min_correlation': np.min(correlations),
            'trend_direction': 'improving' if tracking_errors[-1] < tracking_errors[0] else 'degrading',
            'stability_score': 1.0 - (np.std(tracking_errors) / np.mean(tracking_errors)) if np.mean(tracking_errors) > 0 else 0.0
        }
    
    def is_trading_enabled(self) -> bool:
        """Check if trading is currently enabled based on parity status."""
        return not self.trading_disabled
    
    def force_enable_trading(self, reason: str) -> None:
        """Force enable trading (manual override)."""
        self.trading_disabled = False
        self.consecutive_warnings = 0
        self.consecutive_critical = 0
        
        self.logger.warning("Trading force-enabled", reason=reason)
        
        # Create override file
        override_file = Path("data/system_actions/trading_override.json")
        override_file.parent.mkdir(parents=True, exist_ok=True)
        
        override_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': 'force_enable_trading',
            'reason': reason,
            'authorized_by': 'manual_override'
        }
        
        with open(override_file, 'w') as f:
            json.dump(override_data, f, indent=2)