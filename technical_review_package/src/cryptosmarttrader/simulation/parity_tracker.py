"""
Backtest-Live Parity Tracking System
Monitors tracking error between backtest and live execution
Provides daily reporting and auto-disable functionality
"""

import time
import json
import statistics
import os
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ParityStatus(Enum):
    """Parity monitoring status"""
    ACTIVE = "active"
    WARNING = "warning"
    CRITICAL = "critical"
    DISABLED = "disabled"


class DriftType(Enum):
    """Types of performance drift"""
    EXECUTION_SLIPPAGE = "execution_slippage"
    TIMING_DRIFT = "timing_drift"
    FEE_IMPACT = "fee_impact"
    PARTIAL_FILLS = "partial_fills"
    LATENCY_IMPACT = "latency_impact"
    MARKET_IMPACT = "market_impact"


@dataclass
class TradeExecution:
    """Individual trade execution record"""
    trade_id: str
    symbol: str
    side: str
    size: float
    
    # Backtest execution
    backtest_price: float
    backtest_timestamp: float
    backtest_fees: float = 0.0
    
    # Live execution
    live_price: Optional[float] = None
    live_timestamp: Optional[float] = None
    live_fees: float = 0.0
    live_slippage: float = 0.0
    live_latency_ms: float = 0.0
    
    # Calculated differences
    price_diff_bps: Optional[float] = None
    timing_diff_ms: Optional[float] = None
    fee_diff_bps: Optional[float] = None
    
    def calculate_differences(self):
        """Calculate differences between backtest and live execution"""
        if self.live_price is not None and self.backtest_price > 0:
            # Price difference in basis points
            price_diff = (self.live_price - self.backtest_price) / self.backtest_price
            self.price_diff_bps = price_diff * 10000
            
            # Timing difference in milliseconds
            if self.live_timestamp is not None:
                self.timing_diff_ms = (self.live_timestamp - self.backtest_timestamp) * 1000
            
            # Fee difference in basis points
            trade_value = self.size * self.backtest_price
            if trade_value > 0:
                fee_diff = (self.live_fees - self.backtest_fees) / trade_value
                self.fee_diff_bps = fee_diff * 10000
    
    @property
    def is_complete(self) -> bool:
        """Check if both backtest and live data are available"""
        return self.live_price is not None and self.live_timestamp is not None


@dataclass
class DailyParityReport:
    """Daily parity tracking report"""
    date: datetime
    strategy_id: str
    
    # Trade counts
    total_trades: int = 0
    completed_trades: int = 0
    missing_live_trades: int = 0
    
    # Tracking error metrics (in basis points)
    tracking_error_bps: float = 0.0
    mean_price_diff_bps: float = 0.0
    std_price_diff_bps: float = 0.0
    max_price_diff_bps: float = 0.0
    
    # Component analysis
    slippage_impact_bps: float = 0.0
    fee_impact_bps: float = 0.0
    timing_impact_bps: float = 0.0
    market_impact_bps: float = 0.0
    
    # Latency metrics
    avg_execution_latency_ms: float = 0.0
    max_execution_latency_ms: float = 0.0
    
    # Performance impact
    total_pnl_diff_bps: float = 0.0
    cumulative_drift_bps: float = 0.0
    
    # Status and alerts
    parity_status: ParityStatus = ParityStatus.ACTIVE
    drift_violations: List[DriftType] = field(default_factory=list)
    auto_disable_triggered: bool = False


@dataclass
class ParityThresholds:
    """Configurable thresholds for parity monitoring"""
    
    # Daily tracking error thresholds (basis points)
    warning_threshold_bps: float = 20.0      # 20 bps daily tracking error warning
    critical_threshold_bps: float = 50.0     # 50 bps daily tracking error critical
    disable_threshold_bps: float = 100.0     # 100 bps auto-disable threshold
    
    # Component-specific thresholds
    max_slippage_bps: float = 30.0           # 30 bps max slippage
    max_fee_impact_bps: float = 15.0         # 15 bps max fee impact
    max_timing_impact_bps: float = 10.0      # 10 bps max timing impact
    max_latency_ms: float = 1000.0           # 1 second max execution latency
    
    # Drift detection
    max_cumulative_drift_bps: float = 200.0  # 200 bps max cumulative drift
    drift_window_days: int = 7               # 7-day drift monitoring window
    min_trades_for_analysis: int = 10        # Minimum trades for valid analysis


class ParityTracker:
    """
    Tracks backtest-live parity and calculates tracking error
    Provides automatic monitoring and alerting
    """
    
    def __init__(self, strategy_id: str, thresholds: Optional[ParityThresholds] = None):
        self.strategy_id = strategy_id
        self.thresholds = thresholds or ParityThresholds()
        
        # Trade tracking
        self.pending_trades: Dict[str, TradeExecution] = {}
        self.completed_trades: List[TradeExecution] = []
        self.daily_reports: List[DailyParityReport] = []
        
        # Status tracking
        self.current_status = ParityStatus.ACTIVE
        self.is_disabled = False
        self.disable_reason = ""
        
        # Cumulative metrics
        self.cumulative_drift_bps = 0.0
        self.total_tracking_error_bps = 0.0
        
        self._lock = threading.Lock()
    
    def record_backtest_execution(
        self,
        trade_id: str,
        symbol: str,
        side: str,
        size: float,
        price: float,
        timestamp: float,
        fees: float = 0.0
    ):
        """Record backtest execution for comparison"""
        
        with self._lock:
            execution = TradeExecution(
                trade_id=trade_id,
                symbol=symbol,
                side=side,
                size=size,
                backtest_price=price,
                backtest_timestamp=timestamp,
                backtest_fees=fees
            )
            
            self.pending_trades[trade_id] = execution
            logger.debug(f"Recorded backtest execution: {trade_id} {symbol} {size}@{price}")
    
    def record_live_execution(
        self,
        trade_id: str,
        price: float,
        timestamp: float,
        fees: float = 0.0,
        slippage: float = 0.0,
        latency_ms: float = 0.0
    ):
        """Record live execution for comparison"""
        
        with self._lock:
            if trade_id not in self.pending_trades:
                logger.warning(f"Live execution recorded for unknown trade: {trade_id}")
                return
            
            execution = self.pending_trades[trade_id]
            execution.live_price = price
            execution.live_timestamp = timestamp
            execution.live_fees = fees
            execution.live_slippage = slippage
            execution.live_latency_ms = latency_ms
            
            # Calculate differences
            execution.calculate_differences()
            
            # Move to completed trades
            self.completed_trades.append(execution)
            del self.pending_trades[trade_id]
            
            logger.debug(f"Recorded live execution: {trade_id} diff={execution.price_diff_bps:.1f} bps")
            
            # Update cumulative tracking
            if execution.price_diff_bps is not None:
                self.total_tracking_error_bps += abs(execution.price_diff_bps)
    
    def calculate_daily_tracking_error(self, date: Optional[datetime] = None) -> float:
        """Calculate tracking error for specific date (in basis points)"""
        
        if date is None:
            date = datetime.now().date()
        
        with self._lock:
            # Get trades for the date
            daily_trades = [
                trade for trade in self.completed_trades
                if (datetime.fromtimestamp(trade.backtest_timestamp).date() == date and
                    trade.is_complete and
                    trade.price_diff_bps is not None)
            ]
            
            if not daily_trades:
                return 0.0
            
            # Calculate RMS tracking error
            price_diffs = [trade.price_diff_bps for trade in daily_trades]
            
            if len(price_diffs) >= 2:
                # Use standard deviation as tracking error
                tracking_error = statistics.stdev(price_diffs)
            else:
                # Use absolute difference for single trade
                tracking_error = abs(price_diffs[0])
            
            return tracking_error
    
    def generate_daily_report(self, date: Optional[datetime] = None) -> DailyParityReport:
        """Generate comprehensive daily parity report"""
        
        if date is None:
            date = datetime.now().date()
        
        with self._lock:
            # Get trades for the date
            daily_trades = [
                trade for trade in self.completed_trades
                if datetime.fromtimestamp(trade.backtest_timestamp).date() == date
            ]
            
            completed_daily_trades = [trade for trade in daily_trades if trade.is_complete]
            
            # Calculate basic metrics
            total_trades = len(daily_trades)
            completed_trades = len(completed_daily_trades)
            missing_live_trades = total_trades - completed_trades
            
            if not completed_daily_trades:
                return DailyParityReport(
                    date=datetime.combine(date, datetime.min.time()),
                    strategy_id=self.strategy_id,
                    total_trades=total_trades,
                    missing_live_trades=missing_live_trades
                )
            
            # Price difference analysis
            price_diffs = [trade.price_diff_bps for trade in completed_daily_trades if trade.price_diff_bps is not None]
            
            if price_diffs:
                mean_price_diff = statistics.mean(price_diffs)
                std_price_diff = statistics.stdev(price_diffs) if len(price_diffs) > 1 else 0.0
                max_price_diff = max(abs(diff) for diff in price_diffs)
                tracking_error = std_price_diff
            else:
                mean_price_diff = std_price_diff = max_price_diff = tracking_error = 0.0
            
            # Component analysis
            slippage_impact = statistics.mean([trade.live_slippage * 10000 for trade in completed_daily_trades])
            
            fee_diffs = [trade.fee_diff_bps for trade in completed_daily_trades if trade.fee_diff_bps is not None]
            fee_impact = statistics.mean(fee_diffs) if fee_diffs else 0.0
            
            timing_diffs = [trade.timing_diff_ms for trade in completed_daily_trades if trade.timing_diff_ms is not None]
            timing_impact = statistics.mean(timing_diffs) / 1000 * 10 if timing_diffs else 0.0  # Rough estimate
            
            # Latency analysis
            latencies = [trade.live_latency_ms for trade in completed_daily_trades]
            avg_latency = statistics.mean(latencies) if latencies else 0.0
            max_latency = max(latencies) if latencies else 0.0
            
            # Calculate total P&L impact
            total_pnl_diff = sum(price_diffs) if price_diffs else 0.0
            
            # Create report
            report = DailyParityReport(
                date=datetime.combine(date, datetime.min.time()),
                strategy_id=self.strategy_id,
                total_trades=total_trades,
                completed_trades=completed_trades,
                missing_live_trades=missing_live_trades,
                tracking_error_bps=tracking_error,
                mean_price_diff_bps=mean_price_diff,
                std_price_diff_bps=std_price_diff,
                max_price_diff_bps=max_price_diff,
                slippage_impact_bps=slippage_impact,
                fee_impact_bps=fee_impact,
                timing_impact_bps=timing_impact,
                avg_execution_latency_ms=avg_latency,
                max_execution_latency_ms=max_latency,
                total_pnl_diff_bps=total_pnl_diff,
                cumulative_drift_bps=self.cumulative_drift_bps
            )
            
            # Determine status and violations
            report.parity_status, report.drift_violations = self._assess_parity_status(report)
            
            # Check for auto-disable
            if tracking_error >= self.thresholds.disable_threshold_bps:
                report.auto_disable_triggered = True
                self._trigger_auto_disable(f"Daily tracking error {tracking_error:.1f} bps exceeds threshold {self.thresholds.disable_threshold_bps:.1f} bps")
            
            # Store report
            self.daily_reports.append(report)
            
            # Update cumulative drift
            self.cumulative_drift_bps += abs(total_pnl_diff)
            
            return report
    
    def _assess_parity_status(self, report: DailyParityReport) -> Tuple[ParityStatus, List[DriftType]]:
        """Assess parity status and identify drift violations"""
        
        violations = []
        
        # Check component thresholds
        if report.slippage_impact_bps > self.thresholds.max_slippage_bps:
            violations.append(DriftType.EXECUTION_SLIPPAGE)
        
        if abs(report.fee_impact_bps) > self.thresholds.max_fee_impact_bps:
            violations.append(DriftType.FEE_IMPACT)
        
        if abs(report.timing_impact_bps) > self.thresholds.max_timing_impact_bps:
            violations.append(DriftType.TIMING_DRIFT)
        
        if report.max_execution_latency_ms > self.thresholds.max_latency_ms:
            violations.append(DriftType.LATENCY_IMPACT)
        
        # Determine overall status
        if report.tracking_error_bps >= self.thresholds.critical_threshold_bps:
            status = ParityStatus.CRITICAL
        elif report.tracking_error_bps >= self.thresholds.warning_threshold_bps:
            status = ParityStatus.WARNING
        elif violations:
            status = ParityStatus.WARNING
        else:
            status = ParityStatus.ACTIVE
        
        return status, violations
    
    def _trigger_auto_disable(self, reason: str):
        """Trigger automatic disable of live trading"""
        
        if not self.is_disabled:
            self.is_disabled = True
            self.disable_reason = reason
            self.current_status = ParityStatus.DISABLED
            
            logger.critical(f"Auto-disable triggered for {self.strategy_id}: {reason}")
            
            # Save disable state
            self._save_disable_state()
            
            # Send alerts (implement actual alerting)
            self._send_disable_alert(reason)
    
    def _save_disable_state(self):
        """Save disable state to persistent storage"""
        
        try:
            disable_data = {
                'strategy_id': self.strategy_id,
                'disabled': self.is_disabled,
                'disable_reason': self.disable_reason,
                'disable_timestamp': datetime.now().isoformat(),
                'cumulative_drift_bps': self.cumulative_drift_bps
            }
            
            os.makedirs('data/parity', exist_ok=True)
            with open(f'data/parity/{self.strategy_id}_disable_state.json', 'w') as f:
                json.dump(disable_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save disable state: {e}")
    
    def _send_disable_alert(self, reason: str):
        """Send disable alert (implement actual alerting)"""
        
        try:
            alert_msg = f"CRITICAL: Auto-disable triggered for {self.strategy_id}\n"
            alert_msg += f"Reason: {reason}\n"
            alert_msg += f"Cumulative drift: {self.cumulative_drift_bps:.1f} bps\n"
            alert_msg += f"Timestamp: {datetime.now().isoformat()}\n"
            
            # Log to emergency file
            os.makedirs('logs', exist_ok=True)
            with open('logs/parity_disable_alerts.log', 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] {alert_msg}\n")
            
        except Exception as e:
            logger.error(f"Failed to send disable alert: {e}")
    
    def get_parity_summary(self) -> Dict:
        """Get comprehensive parity tracking summary"""
        
        with self._lock:
            # Recent tracking error (last 7 days)
            recent_reports = [
                report for report in self.daily_reports
                if (datetime.now() - report.date).days <= 7
            ]
            
            if recent_reports:
                avg_tracking_error = statistics.mean([r.tracking_error_bps for r in recent_reports])
                max_tracking_error = max([r.tracking_error_bps for r in recent_reports])
            else:
                avg_tracking_error = max_tracking_error = 0.0
            
            return {
                'strategy_id': self.strategy_id,
                'current_status': self.current_status.value,
                'is_disabled': self.is_disabled,
                'disable_reason': self.disable_reason,
                'pending_trades': len(self.pending_trades),
                'completed_trades': len(self.completed_trades),
                'total_tracking_error_bps': self.total_tracking_error_bps,
                'cumulative_drift_bps': self.cumulative_drift_bps,
                'recent_avg_tracking_error_bps': avg_tracking_error,
                'recent_max_tracking_error_bps': max_tracking_error,
                'daily_reports_count': len(self.daily_reports),
                'thresholds': {
                    'warning_bps': self.thresholds.warning_threshold_bps,
                    'critical_bps': self.thresholds.critical_threshold_bps,
                    'disable_bps': self.thresholds.disable_threshold_bps
                }
            }


# Global parity trackers
_parity_trackers: Dict[str, ParityTracker] = {}
_tracker_lock = threading.Lock()


def get_parity_tracker(strategy_id: str, thresholds: Optional[ParityThresholds] = None) -> ParityTracker:
    """Get parity tracker for strategy"""
    global _parity_trackers
    
    if strategy_id not in _parity_trackers:
        with _tracker_lock:
            if strategy_id not in _parity_trackers:
                _parity_trackers[strategy_id] = ParityTracker(strategy_id, thresholds)
    
    return _parity_trackers[strategy_id]


def calculate_tracking_error(strategy_id: str, date: Optional[datetime] = None) -> float:
    """Convenience function for tracking error calculation"""
    tracker = get_parity_tracker(strategy_id)
    return tracker.calculate_daily_tracking_error(date)