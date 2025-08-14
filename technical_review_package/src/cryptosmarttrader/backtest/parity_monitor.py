"""
Backtest-Live Parity Monitor
Daily tracking error monitoring met auto-disable bij drift
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .execution_simulator import ExecutionSimulator, OrderResult

logger = logging.getLogger(__name__)


class ParityStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DISABLED = "disabled"


@dataclass
class TradeRecord:
    """Individual trade record for parity comparison"""
    timestamp: float
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: Optional[float] = None
    backtest_pnl: float = 0.0
    live_pnl: float = 0.0
    execution_cost: float = 0.0
    slippage_bps: float = 0.0
    strategy_id: str = "default"


@dataclass
class DailyParityReport:
    """Daily parity analysis report"""
    date: str
    total_trades: int
    backtest_pnl: float
    live_pnl: float
    tracking_error_bps: float
    execution_cost_bps: float
    slippage_cost_bps: float
    fee_cost_bps: float
    parity_status: ParityStatus
    component_attribution: Dict[str, float]
    recommendations: List[str]


@dataclass
class ParityConfig:
    """Parity monitoring configuration"""
    warning_threshold_bps: float = 20.0  # 20 bps warning
    critical_threshold_bps: float = 50.0  # 50 bps critical
    auto_disable_threshold_bps: float = 100.0  # 100 bps auto-disable
    lookback_days: int = 7  # 7-day rolling window
    min_trades_for_analysis: int = 10  # Minimum trades for meaningful analysis


class BacktestLiveParityMonitor:
    """
    Advanced backtest-live parity monitoring met drift detection
    Tracks execution costs, slippage, fees en auto-disable functionality
    """
    
    def __init__(self, config: Optional[ParityConfig] = None):
        self.config = config or ParityConfig()
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.trade_records: List[TradeRecord] = []
        self.daily_reports: List[DailyParityReport] = []
        
        # State tracking
        self.current_status = ParityStatus.HEALTHY
        self.auto_disabled = False
        self.disable_timestamp: Optional[float] = None
        self.disable_reason: Optional[str] = None
        
        # Performance tracking
        self.cumulative_tracking_error = 0.0
        self.consecutive_bad_days = 0
        self.total_execution_cost = 0.0
        
        # Integration
        self.execution_simulator: Optional[ExecutionSimulator] = None
    
    def set_execution_simulator(self, simulator: ExecutionSimulator):
        """Set execution simulator for cost analysis"""
        self.execution_simulator = simulator
    
    def record_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        backtest_entry_price: float,
        live_execution_result: OrderResult,
        strategy_id: str = "default"
    ) -> TradeRecord:
        """
        Record trade for parity comparison
        
        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            size: Trade size
            backtest_entry_price: Expected price from backtest
            live_execution_result: Actual execution result
            strategy_id: Strategy identifier
            
        Returns:
            TradeRecord with parity analysis
        """
        
        # Calculate execution costs
        execution_cost = live_execution_result.total_fee
        slippage_bps = live_execution_result.slippage_bps
        
        # Calculate PnL difference (simplified - would need exit prices for complete analysis)
        price_diff = live_execution_result.average_price - backtest_entry_price
        price_diff_bps = (price_diff / backtest_entry_price) * 10000
        
        # Adjust for side (buy vs sell)
        if side == "sell":
            price_diff_bps = -price_diff_bps
        
        trade_record = TradeRecord(
            timestamp=time.time(),
            symbol=symbol,
            side=side,
            size=size,
            entry_price=live_execution_result.average_price,
            backtest_pnl=0.0,  # Would be calculated with strategy logic
            live_pnl=0.0,     # Would be calculated with actual fills
            execution_cost=execution_cost,
            slippage_bps=slippage_bps,
            strategy_id=strategy_id
        )
        
        self.trade_records.append(trade_record)
        
        # Check for immediate parity violations
        self._check_immediate_violations(trade_record)
        
        self.logger.info(
            f"📊 Recorded trade: {symbol} {side} {size:.2f} "
            f"(slippage: {slippage_bps:.1f} bps, cost: ${execution_cost:.2f})"
        )
        
        return trade_record
    
    def _check_immediate_violations(self, trade: TradeRecord):
        """Check for immediate parity violations"""
        
        # Check if single trade exceeds critical threshold
        if trade.slippage_bps > self.config.critical_threshold_bps:
            self.logger.warning(
                f"⚠️  Critical slippage detected: {trade.slippage_bps:.1f} bps > "
                f"{self.config.critical_threshold_bps:.1f} bps threshold"
            )
            
            # Auto-disable if exceeds auto-disable threshold
            if trade.slippage_bps > self.config.auto_disable_threshold_bps:
                self._trigger_auto_disable(
                    f"Single trade slippage exceeded auto-disable threshold: "
                    f"{trade.slippage_bps:.1f} bps"
                )
    
    def generate_daily_report(self, target_date: Optional[datetime] = None) -> DailyParityReport:
        """
        Generate comprehensive daily parity report
        
        Args:
            target_date: Date for report (default: today)
            
        Returns:
            DailyParityReport with detailed analysis
        """
        
        if target_date is None:
            target_date = datetime.now()
        
        # Filter trades for target date
        start_timestamp = target_date.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        end_timestamp = start_timestamp + 86400  # 24 hours
        
        daily_trades = [
            trade for trade in self.trade_records
            if start_timestamp <= trade.timestamp < end_timestamp
        ]
        
        if len(daily_trades) < self.config.min_trades_for_analysis:
            return DailyParityReport(
                date=target_date.strftime("%Y-%m-%d"),
                total_trades=len(daily_trades),
                backtest_pnl=0.0,
                live_pnl=0.0,
                tracking_error_bps=0.0,
                execution_cost_bps=0.0,
                slippage_cost_bps=0.0,
                fee_cost_bps=0.0,
                parity_status=ParityStatus.HEALTHY,
                component_attribution={},
                recommendations=["Insufficient trades for meaningful analysis"]
            )
        
        # Calculate aggregate metrics
        total_notional = sum(trade.size * trade.entry_price for trade in daily_trades)
        total_execution_cost = sum(trade.execution_cost for trade in daily_trades)
        total_slippage = sum(trade.slippage_bps * trade.size * trade.entry_price / 10000 for trade in daily_trades)
        
        # Calculate basis points
        execution_cost_bps = (total_execution_cost / total_notional) * 10000 if total_notional > 0 else 0
        slippage_cost_bps = (total_slippage / total_notional) * 10000 if total_notional > 0 else 0
        
        # Calculate tracking error (simplified)
        avg_slippage_bps = np.mean([trade.slippage_bps for trade in daily_trades])
        tracking_error_bps = avg_slippage_bps + execution_cost_bps
        
        # Component attribution
        component_attribution = {
            "slippage": slippage_cost_bps,
            "fees": execution_cost_bps - slippage_cost_bps,  # Approximate
            "timing": 0.0,  # Would need more detailed analysis
            "partial_fills": 0.0  # Would need fill analysis
        }
        
        # Determine status
        parity_status = self._determine_parity_status(tracking_error_bps)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(daily_trades, tracking_error_bps)
        
        report = DailyParityReport(
            date=target_date.strftime("%Y-%m-%d"),
            total_trades=len(daily_trades),
            backtest_pnl=sum(trade.backtest_pnl for trade in daily_trades),
            live_pnl=sum(trade.live_pnl for trade in daily_trades),
            tracking_error_bps=tracking_error_bps,
            execution_cost_bps=execution_cost_bps,
            slippage_cost_bps=slippage_cost_bps,
            fee_cost_bps=execution_cost_bps - slippage_cost_bps,
            parity_status=parity_status,
            component_attribution=component_attribution,
            recommendations=recommendations
        )
        
        self.daily_reports.append(report)
        
        # Check for auto-disable conditions
        self._check_auto_disable_conditions(report)
        
        self.logger.info(
            f"📊 Daily parity report ({target_date.strftime('%Y-%m-%d')}): "
            f"{len(daily_trades)} trades, {tracking_error_bps:.1f} bps tracking error, "
            f"status: {parity_status.value}"
        )
        
        return report
    
    def _determine_parity_status(self, tracking_error_bps: float) -> ParityStatus:
        """Determine parity status based on tracking error"""
        
        if self.auto_disabled:
            return ParityStatus.DISABLED
        elif tracking_error_bps >= self.config.critical_threshold_bps:
            return ParityStatus.CRITICAL
        elif tracking_error_bps >= self.config.warning_threshold_bps:
            return ParityStatus.WARNING
        else:
            return ParityStatus.HEALTHY
    
    def _generate_recommendations(
        self, 
        daily_trades: List[TradeRecord], 
        tracking_error_bps: float
    ) -> List[str]:
        """Generate actionable recommendations"""
        
        recommendations = []
        
        # High tracking error recommendations
        if tracking_error_bps > self.config.warning_threshold_bps:
            recommendations.append(
                f"Tracking error {tracking_error_bps:.1f} bps exceeds warning threshold"
            )
            
            # Analyze root causes
            avg_slippage = np.mean([trade.slippage_bps for trade in daily_trades])
            if avg_slippage > 15:
                recommendations.append("Consider using limit orders to reduce slippage")
            
            # Check for timing issues
            execution_times = []
            if self.execution_simulator:
                recent_executions = self.execution_simulator.execution_history[-len(daily_trades):]
                execution_times = [result.total_latency_ms for result in recent_executions]
                
                if execution_times and np.mean(execution_times) > 200:
                    recommendations.append("High execution latency detected - check network/exchange performance")
        
        # Volume-based recommendations
        large_trades = [trade for trade in daily_trades if trade.size * trade.entry_price > 50000]
        if len(large_trades) > len(daily_trades) * 0.3:
            recommendations.append("High proportion of large trades - consider order splitting")
        
        # Strategy-specific recommendations
        strategy_performance = {}
        for trade in daily_trades:
            if trade.strategy_id not in strategy_performance:
                strategy_performance[trade.strategy_id] = []
            strategy_performance[trade.strategy_id].append(trade.slippage_bps)
        
        for strategy_id, slippages in strategy_performance.items():
            avg_strategy_slippage = np.mean(slippages)
            if avg_strategy_slippage > 20:
                recommendations.append(
                    f"Strategy {strategy_id} showing high slippage ({avg_strategy_slippage:.1f} bps)"
                )
        
        return recommendations
    
    def _check_auto_disable_conditions(self, report: DailyParityReport):
        """Check if auto-disable should be triggered"""
        
        if self.auto_disabled:
            return
        
        # Single day critical threshold
        if report.tracking_error_bps >= self.config.auto_disable_threshold_bps:
            self._trigger_auto_disable(
                f"Daily tracking error {report.tracking_error_bps:.1f} bps exceeds "
                f"auto-disable threshold {self.config.auto_disable_threshold_bps:.1f} bps"
            )
            return
        
        # Multiple consecutive bad days
        if report.parity_status == ParityStatus.CRITICAL:
            self.consecutive_bad_days += 1
        else:
            self.consecutive_bad_days = 0
        
        if self.consecutive_bad_days >= 3:
            self._trigger_auto_disable(
                f"3 consecutive days of critical parity status"
            )
        
        # Rolling window analysis
        recent_reports = self.daily_reports[-self.config.lookback_days:]
        if len(recent_reports) >= self.config.lookback_days:
            avg_tracking_error = np.mean([r.tracking_error_bps for r in recent_reports])
            if avg_tracking_error > self.config.critical_threshold_bps * 0.8:  # 80% of critical
                self._trigger_auto_disable(
                    f"Rolling {self.config.lookback_days}-day average tracking error "
                    f"{avg_tracking_error:.1f} bps exceeds sustained threshold"
                )
    
    def _trigger_auto_disable(self, reason: str):
        """Trigger auto-disable functionality"""
        
        self.auto_disabled = True
        self.disable_timestamp = time.time()
        self.disable_reason = reason
        self.current_status = ParityStatus.DISABLED
        
        self.logger.critical(f"🚨 AUTO-DISABLE TRIGGERED: {reason}")
        
        # Here you would integrate with your trading system to halt operations
        # For example: send alerts, stop new orders, close positions, etc.
    
    def manual_enable(self, operator_id: str, reason: str) -> bool:
        """
        Manually re-enable after auto-disable
        
        Args:
            operator_id: ID of operator enabling system
            reason: Reason for re-enabling
            
        Returns:
            True if successful, False if conditions not met
        """
        
        if not self.auto_disabled:
            self.logger.warning("System not auto-disabled - cannot manually enable")
            return False
        
        # Reset state
        self.auto_disabled = False
        self.disable_timestamp = None
        self.disable_reason = None
        self.current_status = ParityStatus.HEALTHY
        self.consecutive_bad_days = 0
        
        self.logger.info(
            f"✅ System manually re-enabled by {operator_id}: {reason}"
        )
        
        return True
    
    def get_parity_summary(self) -> Dict[str, Any]:
        """Get comprehensive parity monitoring summary"""
        
        if not self.daily_reports:
            return {"error": "No daily reports available"}
        
        recent_reports = self.daily_reports[-7:]  # Last 7 days
        
        return {
            "current_status": {
                "status": self.current_status.value,
                "auto_disabled": self.auto_disabled,
                "disable_reason": self.disable_reason,
                "disable_timestamp": self.disable_timestamp
            },
            "recent_performance": {
                "average_tracking_error_bps": np.mean([r.tracking_error_bps for r in recent_reports]),
                "max_tracking_error_bps": max([r.tracking_error_bps for r in recent_reports]),
                "days_above_warning": len([r for r in recent_reports if r.parity_status in [ParityStatus.WARNING, ParityStatus.CRITICAL]]),
                "consecutive_bad_days": self.consecutive_bad_days
            },
            "cost_breakdown": {
                "average_execution_cost_bps": np.mean([r.execution_cost_bps for r in recent_reports]),
                "average_slippage_cost_bps": np.mean([r.slippage_cost_bps for r in recent_reports]),
                "total_trades_analyzed": sum([r.total_trades for r in recent_reports])
            },
            "thresholds": {
                "warning_threshold_bps": self.config.warning_threshold_bps,
                "critical_threshold_bps": self.config.critical_threshold_bps,
                "auto_disable_threshold_bps": self.config.auto_disable_threshold_bps
            },
            "latest_report": recent_reports[-1].__dict__ if recent_reports else None
        }
    
    def export_parity_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Export parity data for external analysis"""
        
        # Filter trade records by date range
        start_ts = start_date.timestamp()
        end_ts = end_date.timestamp()
        
        filtered_trades = [
            trade for trade in self.trade_records
            if start_ts <= trade.timestamp <= end_ts
        ]
        
        # Convert to DataFrame
        data = []
        for trade in filtered_trades:
            data.append({
                "timestamp": trade.timestamp,
                "date": datetime.fromtimestamp(trade.timestamp).strftime("%Y-%m-%d"),
                "symbol": trade.symbol,
                "side": trade.side,
                "size": trade.size,
                "entry_price": trade.entry_price,
                "execution_cost": trade.execution_cost,
                "slippage_bps": trade.slippage_bps,
                "strategy_id": trade.strategy_id
            })
        
        return pd.DataFrame(data)


# Global parity monitor instance
_global_parity_monitor: Optional[BacktestLiveParityMonitor] = None


def get_global_parity_monitor() -> BacktestLiveParityMonitor:
    """Get or create global parity monitor"""
    global _global_parity_monitor
    if _global_parity_monitor is None:
        _global_parity_monitor = BacktestLiveParityMonitor()
        logger.info("✅ Global BacktestLiveParityMonitor initialized")
    return _global_parity_monitor


def reset_global_parity_monitor():
    """Reset global parity monitor (for testing)"""
    global _global_parity_monitor
    _global_parity_monitor = None