"""
Integrated Backtest-Live Parity System
Combineert execution simulation en parity monitoring voor complete pipeline
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import time
import pandas as pd

from .execution_simulator import (
    ExecutionSimulator, OrderRequest, OrderResult, MarketMicrostructure, ExchangeConfig
)
from .parity_monitor import (
    BacktestLiveParityMonitor, TradeRecord, DailyParityReport, ParityStatus, ParityConfig
)
from ..risk.central_risk_guard import CentralRiskGuard, TradingOperation, RiskDecision

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    """Backtest trade for parity comparison"""
    timestamp: float
    symbol: str
    side: str
    size: float
    entry_price: float
    exit_price: Optional[float] = None
    expected_pnl: float = 0.0
    strategy_id: str = "default"


@dataclass
class ParitySystemConfig:
    """Integrated parity system configuration"""
    execution_config: ExchangeConfig
    parity_config: ParityConfig
    enable_auto_disable: bool = True
    enable_cost_attribution: bool = True
    daily_report_time: str = "23:59"  # UTC time for daily reports


class IntegratedParitySystem:
    """
    Complete backtest-live parity system met:
    - Realistic execution simulation
    - Daily tracking error monitoring  
    - Component cost attribution
    - Auto-disable functionality
    """
    
    def __init__(self, config: Optional[ParitySystemConfig] = None):
        if config is None:
            config = ParitySystemConfig(
                execution_config=ExchangeConfig("kraken"),
                parity_config=ParityConfig()
            )
        
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.execution_simulator = ExecutionSimulator(config.execution_config)
        self.parity_monitor = BacktestLiveParityMonitor(config.parity_config)
        self.risk_guard: Optional[CentralRiskGuard] = None
        
        # Link components
        self.parity_monitor.set_execution_simulator(self.execution_simulator)
        
        # State tracking
        self.backtest_trades: List[BacktestTrade] = []
        self.live_executions: List[OrderResult] = []
        self.market_data_cache: Dict[str, MarketMicrostructure] = {}
        
        # Performance tracking
        self.daily_tracking_errors: List[float] = []
        self.cost_attribution_history: List[Dict[str, float]] = []
        
        self.logger.info("✅ Integrated Parity System initialized")
    
    def set_risk_guard(self, risk_guard: CentralRiskGuard):
        """Set risk guard for validation"""
        self.risk_guard = risk_guard
    
    def update_market_data(self, symbol: str, market_data: MarketMicrostructure):
        """Update market microstructure data"""
        self.market_data_cache[symbol] = market_data
        self.execution_simulator.update_market_data(market_data)
    
    def record_backtest_trade(
        self,
        timestamp: float,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        exit_price: Optional[float] = None,
        expected_pnl: float = 0.0,
        strategy_id: str = "default"
    ) -> BacktestTrade:
        """
        Record backtest trade for future parity comparison
        
        Args:
            timestamp: Trade timestamp
            symbol: Trading symbol
            side: "buy" or "sell"
            size: Trade size
            entry_price: Backtest entry price
            exit_price: Backtest exit price (if available)
            expected_pnl: Expected PnL from backtest
            strategy_id: Strategy identifier
            
        Returns:
            BacktestTrade record
        """
        
        backtest_trade = BacktestTrade(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            exit_price=exit_price,
            expected_pnl=expected_pnl,
            strategy_id=strategy_id
        )
        
        self.backtest_trades.append(backtest_trade)
        
        self.logger.info(
            f"📊 Recorded backtest trade: {symbol} {side} {size:.2f} @ {entry_price:.2f}"
        )
        
        return backtest_trade
    
    def execute_live_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "limit",
        limit_price: Optional[float] = None,
        strategy_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Execute live trade met complete parity tracking
        
        Args:
            symbol: Trading symbol
            side: "buy" or "sell" 
            size: Trade size
            order_type: "market" or "limit"
            limit_price: Limit price (for limit orders)
            strategy_id: Strategy identifier
            
        Returns:
            Complete execution result with parity analysis
        """
        
        # Validate with risk guard if available
        if self.risk_guard:
            trading_operation = TradingOperation(
                operation_type="entry",
                symbol=symbol,
                side=side,
                size_usd=size * (limit_price or 50000.0),  # Approximate USD value
                current_price=limit_price or 50000.0,
                strategy_id=strategy_id
            )
            
            risk_evaluation = self.risk_guard.evaluate_operation(trading_operation)
            
            if risk_evaluation.decision == RiskDecision.REJECT:
                return {
                    "success": False,
                    "error": f"Risk guard rejected: {'; '.join(risk_evaluation.reasons)}",
                    "risk_evaluation": risk_evaluation
                }
            elif risk_evaluation.decision == RiskDecision.REDUCE_SIZE:
                # Adjust size based on risk guard recommendation
                original_size = size
                size = risk_evaluation.approved_size_usd / (limit_price or 50000.0)
                self.logger.info(
                    f"🛡️  Risk guard reduced size: {original_size:.2f} → {size:.2f}"
                )
        
        # Check if system is auto-disabled
        if self.parity_monitor.auto_disabled:
            return {
                "success": False,
                "error": f"System auto-disabled: {self.parity_monitor.disable_reason}",
                "auto_disabled": True
            }
        
        # Get market data
        market_data = self.market_data_cache.get(symbol)
        if not market_data:
            return {
                "success": False,
                "error": f"No market data available for {symbol}"
            }
        
        # Create order request
        order_request = OrderRequest(
            order_id=f"live_{int(time.time() * 1000)}_{strategy_id}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            size=size,
            limit_price=limit_price,
            strategy_id=strategy_id
        )
        
        # Execute through simulator
        execution_result = self.execution_simulator.simulate_order_execution(
            order_request, market_data
        )
        
        # Find corresponding backtest trade
        backtest_trade = self._find_matching_backtest_trade(
            symbol, side, size, strategy_id
        )
        
        # Record for parity monitoring
        if backtest_trade:
            trade_record = self.parity_monitor.record_trade(
                symbol=symbol,
                side=side,
                size=size,
                backtest_entry_price=backtest_trade.entry_price,
                live_execution_result=execution_result,
                strategy_id=strategy_id
            )
        else:
            # No matching backtest trade - still record for analysis
            trade_record = self.parity_monitor.record_trade(
                symbol=symbol,
                side=side,
                size=size,
                backtest_entry_price=limit_price or market_data.last_price,
                live_execution_result=execution_result,
                strategy_id=strategy_id
            )
        
        # Store live execution
        self.live_executions.append(execution_result)
        
        # Calculate parity metrics
        parity_metrics = self._calculate_trade_parity_metrics(
            trade_record, backtest_trade, execution_result
        )
        
        return {
            "success": execution_result.status.value in ["filled", "partial"],
            "execution_result": execution_result,
            "trade_record": trade_record,
            "backtest_trade": backtest_trade,
            "parity_metrics": parity_metrics,
            "system_status": self.parity_monitor.current_status.value
        }
    
    def _find_matching_backtest_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        strategy_id: str,
        tolerance: float = 0.05  # 5% size tolerance
    ) -> Optional[BacktestTrade]:
        """Find matching backtest trade for parity comparison"""
        
        # Look for recent backtest trades that match criteria
        recent_time_threshold = time.time() - 3600  # Last hour
        
        candidates = [
            trade for trade in self.backtest_trades
            if (trade.symbol == symbol and
                trade.side == side and
                trade.strategy_id == strategy_id and
                trade.timestamp > recent_time_threshold and
                abs(trade.size - size) / trade.size <= tolerance)
        ]
        
        if candidates:
            # Return most recent matching trade
            return max(candidates, key=lambda t: t.timestamp)
        
        return None
    
    def _calculate_trade_parity_metrics(
        self,
        trade_record: TradeRecord,
        backtest_trade: Optional[BacktestTrade],
        execution_result: OrderResult
    ) -> Dict[str, Any]:
        """Calculate detailed parity metrics for single trade"""
        
        metrics = {
            "has_backtest_comparison": backtest_trade is not None,
            "execution_quality_score": execution_result.execution_quality_score,
            "slippage_bps": execution_result.slippage_bps,
            "total_execution_cost": execution_result.total_fee,
            "latency_ms": execution_result.total_latency_ms
        }
        
        if backtest_trade:
            # Price difference analysis
            price_diff = execution_result.average_price - backtest_trade.entry_price
            price_diff_bps = (price_diff / backtest_trade.entry_price) * 10000
            
            # Adjust for trade side
            if trade_record.side == "sell":
                price_diff_bps = -price_diff_bps
            
            metrics.update({
                "backtest_entry_price": backtest_trade.entry_price,
                "live_execution_price": execution_result.average_price,
                "price_difference_bps": abs(price_diff_bps),
                "expected_vs_actual_cost_bps": abs(price_diff_bps) + execution_result.slippage_bps
            })
        
        return metrics
    
    def generate_daily_parity_report(self) -> DailyParityReport:
        """Generate daily parity report with enhanced analysis"""
        
        daily_report = self.parity_monitor.generate_daily_report()
        
        # Add system-level analysis
        if daily_report.tracking_error_bps > 0:
            self.daily_tracking_errors.append(daily_report.tracking_error_bps)
            
            # Keep rolling window
            if len(self.daily_tracking_errors) > 30:  # 30-day window
                self.daily_tracking_errors = self.daily_tracking_errors[-30:]
        
        # Enhanced cost attribution
        if self.config.enable_cost_attribution:
            cost_attribution = self._perform_detailed_cost_attribution(daily_report)
            self.cost_attribution_history.append(cost_attribution)
        
        self.logger.info(
            f"📈 Daily parity report generated: {daily_report.tracking_error_bps:.1f} bps "
            f"tracking error, status: {daily_report.parity_status.value}"
        )
        
        return daily_report
    
    def _perform_detailed_cost_attribution(self, daily_report: DailyParityReport) -> Dict[str, float]:
        """Perform detailed cost attribution analysis"""
        
        # Get today's executions
        today_start = time.time() - 86400  # 24 hours ago
        recent_executions = [
            exec_result for exec_result in self.live_executions
            if any(fill.timestamp > today_start for fill in exec_result.fills)
        ]
        
        if not recent_executions:
            return {}
        
        # Calculate component costs
        total_value = sum(
            exec_result.filled_size * exec_result.average_price 
            for exec_result in recent_executions
        )
        
        if total_value == 0:
            return {}
        
        # Slippage cost
        slippage_cost = sum(
            (exec_result.slippage_bps / 10000) * exec_result.filled_size * exec_result.average_price
            for exec_result in recent_executions
        )
        
        # Fee cost
        fee_cost = sum(exec_result.total_fee for exec_result in recent_executions)
        
        # Latency cost (estimated impact)
        avg_latency = sum(exec_result.total_latency_ms for exec_result in recent_executions) / len(recent_executions)
        latency_cost_bps = min(5, max(0, (avg_latency - 50) / 20))  # Rough estimate
        latency_cost = (latency_cost_bps / 10000) * total_value
        
        # Partial fill cost (opportunity cost)
        partial_fills = [
            exec_result for exec_result in recent_executions
            if exec_result.filled_size < exec_result.requested_size * 0.99
        ]
        partial_fill_cost = len(partial_fills) * 0.1 * (total_value / len(recent_executions))  # Rough estimate
        
        cost_attribution = {
            "slippage_cost_bps": (slippage_cost / total_value) * 10000,
            "fee_cost_bps": (fee_cost / total_value) * 10000,
            "latency_cost_bps": latency_cost_bps,
            "partial_fill_cost_bps": (partial_fill_cost / total_value) * 10000,
            "total_cost_bps": daily_report.tracking_error_bps
        }
        
        return cost_attribution
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary"""
        
        parity_summary = self.parity_monitor.get_parity_summary()
        execution_stats = self.execution_simulator.get_execution_statistics()
        
        # Calculate trend analysis
        trend_analysis = {}
        if len(self.daily_tracking_errors) >= 7:
            recent_errors = self.daily_tracking_errors[-7:]
            trend_analysis = {
                "7_day_avg_tracking_error": sum(recent_errors) / len(recent_errors),
                "tracking_error_trend": "improving" if recent_errors[-1] < recent_errors[0] else "deteriorating",
                "days_above_warning": len([e for e in recent_errors if e > self.config.parity_config.warning_threshold_bps])
            }
        
        return {
            "system_status": {
                "operational": not self.parity_monitor.auto_disabled,
                "auto_disabled": self.parity_monitor.auto_disabled,
                "current_parity_status": self.parity_monitor.current_status.value
            },
            "parity_monitoring": parity_summary,
            "execution_simulation": execution_stats,
            "trend_analysis": trend_analysis,
            "cost_attribution": {
                "latest": self.cost_attribution_history[-1] if self.cost_attribution_history else {},
                "historical_count": len(self.cost_attribution_history)
            },
            "data_coverage": {
                "backtest_trades": len(self.backtest_trades),
                "live_executions": len(self.live_executions),
                "daily_reports": len(self.parity_monitor.daily_reports)
            }
        }
    
    def export_comprehensive_analysis(
        self, 
        start_date: str, 
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Export comprehensive parity analysis data"""
        
        from datetime import datetime
        
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Export trade-level data
        trade_data = self.parity_monitor.export_parity_data(start_dt, end_dt)
        
        # Export daily reports
        daily_data = []
        for report in self.parity_monitor.daily_reports:
            report_date = datetime.strptime(report.date, "%Y-%m-%d")
            if start_dt <= report_date <= end_dt:
                daily_data.append({
                    "date": report.date,
                    "total_trades": report.total_trades,
                    "tracking_error_bps": report.tracking_error_bps,
                    "execution_cost_bps": report.execution_cost_bps,
                    "slippage_cost_bps": report.slippage_cost_bps,
                    "parity_status": report.parity_status.value
                })
        
        # Export execution statistics
        execution_data = []
        for exec_result in self.live_executions:
            exec_date = datetime.fromtimestamp(exec_result.fills[0].timestamp if exec_result.fills else time.time())
            if start_dt <= exec_date <= end_dt:
                execution_data.append({
                    "timestamp": exec_result.fills[0].timestamp if exec_result.fills else 0,
                    "order_id": exec_result.order_id,
                    "symbol": exec_result.symbol,
                    "side": exec_result.side,
                    "requested_size": exec_result.requested_size,
                    "filled_size": exec_result.filled_size,
                    "average_price": exec_result.average_price,
                    "slippage_bps": exec_result.slippage_bps,
                    "total_fee": exec_result.total_fee,
                    "latency_ms": exec_result.total_latency_ms,
                    "execution_quality": exec_result.execution_quality_score
                })
        
        return {
            "trade_level_data": trade_data,
            "daily_reports": pd.DataFrame(daily_data),
            "execution_details": pd.DataFrame(execution_data)
        }


# Global integrated parity system instance
_global_parity_system: Optional[IntegratedParitySystem] = None


def get_global_parity_system() -> IntegratedParitySystem:
    """Get or create global integrated parity system"""
    global _global_parity_system
    if _global_parity_system is None:
        _global_parity_system = IntegratedParitySystem()
        logger.info("✅ Global IntegratedParitySystem initialized")
    return _global_parity_system


def reset_global_parity_system():
    """Reset global parity system (for testing)"""
    global _global_parity_system
    _global_parity_system = None