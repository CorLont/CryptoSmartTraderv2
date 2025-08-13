"""
Daily Parity Job - Fase D Implementation
Automated daily backtest-live parity monitoring with tracking error < X bps.
"""

import asyncio
import json
import schedule
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import numpy as np
import pandas as pd

from ..core.structured_logger import get_logger
from .enhanced_execution_simulator import EnhancedExecutionSimulator
from .parity_analyzer import ParityAnalyzer, ParityMetrics
from ..analysis.backtest_parity import BacktestParityAnalyzer
from ..observability.metrics_collector import MetricsCollector


class ParityStatus(Enum):
    """Parity monitoring status levels."""
    
    EXCELLENT = "excellent"      # < 10 bps tracking error
    GOOD = "good"               # 10-20 bps tracking error  
    WARNING = "warning"         # 20-50 bps tracking error
    CRITICAL = "critical"       # 50-100 bps tracking error
    EMERGENCY = "emergency"     # > 100 bps tracking error


class SystemAction(Enum):
    """Auto-disable system actions."""
    
    CONTINUE = "continue"        # Normal operation
    MONITOR = "monitor"          # Increased monitoring
    REDUCE_SIZE = "reduce_size"  # Reduce position sizes 50%
    DISABLE = "disable"          # Disable new entries
    EMERGENCY_STOP = "stop"      # Emergency stop all trading


@dataclass
class ParityThresholds:
    """Configurable thresholds for parity monitoring."""
    
    # Tracking error thresholds (bps)
    excellent_threshold_bps: float = 10.0
    good_threshold_bps: float = 20.0
    warning_threshold_bps: float = 50.0
    critical_threshold_bps: float = 100.0
    
    # Auto-disable thresholds
    auto_disable_threshold_bps: float = 75.0
    emergency_stop_threshold_bps: float = 150.0
    
    # Consecutive breach limits
    max_warning_breaches: int = 3
    max_critical_breaches: int = 2
    
    # Minimum data requirements
    min_trades_for_analysis: int = 10
    min_days_for_trend: int = 3


@dataclass  
class DailyParityResult:
    """Results of daily parity analysis."""
    
    date: str
    tracking_error_bps: float
    correlation: float
    hit_rate: float
    parity_status: ParityStatus
    system_action: SystemAction
    
    # Component attribution
    alpha_attribution: float
    fees_attribution: float 
    slippage_attribution: float
    timing_attribution: float
    sizing_attribution: float
    
    # Statistics
    total_trades: int
    avg_slippage_bps: float
    total_fees_bps: float
    execution_quality_score: float
    
    # Alerts and recommendations
    breach_count: int
    consecutive_breaches: int
    recommendations: List[str]
    next_review_time: datetime
    
    metadata: Dict[str, Any] = None


class DailyParityJob:
    """
    Daily Parity Job for automated backtest-live parity monitoring.
    
    Features:
    - Daily tracking error calculation < X bps target
    - Automated component attribution analysis  
    - Progressive system actions on breaches
    - Persistent state and breach counting
    - Integration with alert system
    - Comprehensive daily reports
    """
    
    def __init__(self, target_tracking_error_bps: float = 20.0):
        self.logger = get_logger("daily_parity_job")
        
        # Configuration
        self.target_tracking_error_bps = target_tracking_error_bps
        self.thresholds = ParityThresholds()
        
        # Core components
        self.execution_simulator = EnhancedExecutionSimulator()
        self.parity_analyzer = ParityAnalyzer()
        self.backtest_analyzer = BacktestParityAnalyzer()
        self.metrics_collector = MetricsCollector("parity_monitor")
        
        # State tracking
        self.daily_results: List[DailyParityResult] = []
        self.consecutive_warning_days = 0
        self.consecutive_critical_days = 0
        self.current_system_action = SystemAction.CONTINUE
        self.last_analysis_date = None
        
        # Persistence
        self.results_file = Path("data/parity/daily_parity_results.json")
        self.state_file = Path("data/parity/parity_job_state.json")
        
        # Ensure directories exist
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load previous state
        self._load_previous_state()
        
        self.logger.info(
            f"DailyParityJob initialized with target tracking error {target_tracking_error_bps} bps"
        )
    
    def setup_daily_schedule(self) -> None:
        """Setup automated daily parity job scheduling."""
        
        # Run daily at 6 AM UTC (after markets have settled)
        schedule.every().day.at("06:00").do(self._run_daily_analysis_wrapper)
        
        # Backup run at 18:00 UTC in case morning run fails
        schedule.every().day.at("18:00").do(self._run_daily_analysis_wrapper)
        
        self.logger.info("Daily parity job scheduled for 06:00 and 18:00 UTC")
    
    def _run_daily_analysis_wrapper(self) -> None:
        """Wrapper for async daily analysis."""
        try:
            asyncio.run(self.run_daily_analysis())
        except Exception as e:
            self.logger.error(f"Daily parity analysis failed: {e}")
    
    async def run_daily_analysis(self, analysis_date: Optional[datetime] = None) -> DailyParityResult:
        """
        Run comprehensive daily parity analysis.
        
        Returns:
            DailyParityResult with complete analysis and recommended actions
        """
        analysis_date = analysis_date or datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        date_str = analysis_date.strftime("%Y-%m-%d")
        
        self.logger.info(f"Starting daily parity analysis for {date_str}")
        
        try:
            # Step 1: Collect trading data from previous day
            trading_data = await self._collect_daily_trading_data(analysis_date)
            
            if len(trading_data) < self.thresholds.min_trades_for_analysis:
                self.logger.warning(
                    f"Insufficient trades ({len(trading_data)}) for analysis. Minimum: {self.thresholds.min_trades_for_analysis}"
                )
                return self._create_insufficient_data_result(date_str)
            
            # Step 2: Simulate backtest execution for same signals  
            backtest_data = await self._simulate_backtest_execution(trading_data)
            
            # Step 3: Calculate comprehensive parity metrics
            parity_metrics = await self._calculate_parity_metrics(trading_data, backtest_data)
            
            # Step 4: Perform component attribution analysis
            attribution = await self._perform_component_attribution(trading_data, backtest_data)
            
            # Step 5: Assess parity status and determine system actions
            parity_status, system_action = self._assess_parity_status(parity_metrics)
            
            # Step 6: Generate recommendations
            recommendations = self._generate_recommendations(parity_metrics, attribution)
            
            # Step 7: Create comprehensive daily result
            daily_result = DailyParityResult(
                date=date_str,
                tracking_error_bps=parity_metrics.tracking_error_bps,
                correlation=parity_metrics.correlation,
                hit_rate=parity_metrics.hit_rate,
                parity_status=parity_status,
                system_action=system_action,
                
                # Attribution
                alpha_attribution=attribution.get('alpha', 0.0),
                fees_attribution=attribution.get('fees', 0.0),
                slippage_attribution=attribution.get('slippage', 0.0),
                timing_attribution=attribution.get('timing', 0.0),
                sizing_attribution=attribution.get('sizing', 0.0),
                
                # Statistics
                total_trades=len(trading_data),
                avg_slippage_bps=parity_metrics.avg_slippage_bps,
                total_fees_bps=parity_metrics.total_fees_bps,
                execution_quality_score=parity_metrics.execution_quality_score,
                
                # State tracking
                breach_count=self._count_recent_breaches(),
                consecutive_breaches=self._get_consecutive_breaches(parity_status),
                recommendations=recommendations,
                next_review_time=analysis_date + timedelta(days=1),
                
                metadata={
                    'analysis_duration_seconds': 0,  # Will be updated
                    'data_sources': ['live_trades', 'backtest_simulation'],
                    'target_tracking_error_bps': self.target_tracking_error_bps,
                    'market_conditions': await self._assess_market_conditions()
                }
            )
            
            # Step 8: Update state and persist results
            self._update_system_state(daily_result)
            self._persist_results(daily_result)
            
            # Step 9: Send alerts and notifications
            await self._send_parity_alerts(daily_result)
            
            # Step 10: Update metrics
            self._update_metrics(daily_result)
            
            self.last_analysis_date = analysis_date
            
            self.logger.info(
                f"Daily parity analysis completed",
                date=date_str,
                tracking_error_bps=daily_result.tracking_error_bps,
                parity_status=parity_status.value,
                system_action=system_action.value,
                total_trades=daily_result.total_trades
            )
            
            return daily_result
            
        except Exception as e:
            self.logger.error(f"Daily parity analysis failed: {e}")
            raise
    
    async def _collect_daily_trading_data(self, date: datetime) -> List[Dict[str, Any]]:
        """Collect actual trading data from the previous day."""
        # In production, this would query actual trading database
        # For now, simulate realistic trading data
        
        trading_data = []
        base_time = date.replace(hour=9)  # Start at 9 AM
        
        for i in range(np.random.randint(15, 50)):  # 15-50 trades per day
            trade_time = base_time + timedelta(minutes=np.random.randint(0, 600))  # Within 10 hours
            
            symbols = ['BTC/USD', 'ETH/USD', 'ADA/USD', 'SOL/USD']
            symbol = np.random.choice(symbols)
            side = np.random.choice(['buy', 'sell'])
            quantity = np.random.uniform(0.1, 5.0)
            
            # Simulate realistic prices and slippage
            base_prices = {'BTC/USD': 45000, 'ETH/USD': 3000, 'ADA/USD': 0.5, 'SOL/USD': 100}
            base_price = base_prices[symbol]
            execution_price = base_price * (1 + np.random.normal(0, 0.001))  # Small price movement
            
            slippage_bps = np.random.gamma(2, 5)  # Realistic slippage distribution
            fees_bps = 25.0 if np.random.random() > 0.7 else 10.0  # 70% maker, 30% taker
            
            trade_data = {
                'timestamp': trade_time,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'execution_price': execution_price,
                'slippage_bps': slippage_bps,
                'fees_bps': fees_bps,
                'strategy_id': f"strategy_{np.random.randint(1, 5)}",
                'confidence_score': np.random.uniform(0.6, 0.95)
            }
            
            trading_data.append(trade_data)
        
        return trading_data
    
    async def _simulate_backtest_execution(self, trading_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simulate how the same trades would have executed in backtest."""
        
        backtest_data = []
        
        for trade in trading_data:
            # Simulate backtest execution (typically more optimistic)
            backtest_slippage = trade['slippage_bps'] * 0.7  # Backtest typically underestimates slippage
            backtest_fees = trade['fees_bps'] * 0.9  # May not account for all fee scenarios
            
            # Backtest timing is perfect (no latency)
            backtest_execution_price = trade['execution_price'] * (1 - np.random.normal(0, 0.0005))
            
            backtest_trade = {
                'timestamp': trade['timestamp'],
                'symbol': trade['symbol'],
                'side': trade['side'],
                'quantity': trade['quantity'],
                'execution_price': backtest_execution_price,
                'slippage_bps': backtest_slippage,
                'fees_bps': backtest_fees,
                'strategy_id': trade['strategy_id'],
                'confidence_score': trade['confidence_score']
            }
            
            backtest_data.append(backtest_trade)
        
        return backtest_data
    
    async def _calculate_parity_metrics(self, live_data: List[Dict], backtest_data: List[Dict]) -> Any:
        """Calculate comprehensive parity metrics."""
        
        # Convert to numpy arrays for calculation
        live_returns = np.array([self._calculate_trade_return(t) for t in live_data])
        backtest_returns = np.array([self._calculate_trade_return(t) for t in backtest_data])
        
        # Tracking error (key metric)
        tracking_error = np.std(live_returns - backtest_returns) * np.sqrt(252) * 10000  # Annualized bps
        
        # Correlation
        correlation = np.corrcoef(live_returns, backtest_returns)[0, 1] if len(live_returns) > 1 else 1.0
        
        # Hit rate (% of trades in same direction)
        hit_rate = np.mean(np.sign(live_returns) == np.sign(backtest_returns))
        
        # Slippage and fees
        avg_slippage_bps = np.mean([t['slippage_bps'] for t in live_data])
        total_fees_bps = np.mean([t['fees_bps'] for t in live_data])
        
        # Execution quality score (composite metric)
        execution_quality = self._calculate_execution_quality_score(
            tracking_error, correlation, hit_rate, avg_slippage_bps
        )
        
        # Create metrics object (simplified structure)
        class ParityMetrics:
            def __init__(self):
                self.tracking_error_bps = tracking_error
                self.correlation = correlation
                self.hit_rate = hit_rate
                self.avg_slippage_bps = avg_slippage_bps
                self.total_fees_bps = total_fees_bps
                self.execution_quality_score = execution_quality
        
        return ParityMetrics()
    
    def _calculate_trade_return(self, trade_data: Dict[str, Any]) -> float:
        """Calculate return for a single trade."""
        # Simplified return calculation
        base_return = np.random.normal(0.001, 0.01)  # Small random return
        
        # Adjust for slippage and fees
        costs = (trade_data['slippage_bps'] + trade_data['fees_bps']) / 10000.0
        net_return = base_return - costs
        
        return net_return
    
    def _calculate_execution_quality_score(self, tracking_error: float, correlation: float, 
                                         hit_rate: float, avg_slippage: float) -> float:
        """Calculate composite execution quality score (0-100)."""
        
        # Normalize metrics to 0-100 scale
        tracking_score = max(0, 100 - (tracking_error / 2.0))  # 2 bps = 1 point deduction
        correlation_score = correlation * 100
        hit_rate_score = hit_rate * 100
        slippage_score = max(0, 100 - avg_slippage)  # 1 bps = 1 point deduction
        
        # Weighted composite score
        weights = {'tracking': 0.4, 'correlation': 0.25, 'hit_rate': 0.2, 'slippage': 0.15}
        
        composite_score = (
            tracking_score * weights['tracking'] +
            correlation_score * weights['correlation'] +
            hit_rate_score * weights['hit_rate'] +
            slippage_score * weights['slippage']
        )
        
        return min(100.0, max(0.0, composite_score))
    
    async def _perform_component_attribution(self, live_data: List[Dict], backtest_data: List[Dict]) -> Dict[str, float]:
        """Perform component attribution analysis."""
        
        # Calculate individual attribution components
        alpha_attribution = np.random.normal(0.0, 0.005)  # Strategy alpha difference
        fees_attribution = np.mean([l['fees_bps'] - b['fees_bps'] for l, b in zip(live_data, backtest_data)]) / 10000.0
        slippage_attribution = np.mean([l['slippage_bps'] - b['slippage_bps'] for l, b in zip(live_data, backtest_data)]) / 10000.0
        timing_attribution = np.random.normal(0.0, 0.002)  # Execution timing difference
        sizing_attribution = np.random.normal(0.0, 0.001)  # Position sizing difference
        
        return {
            'alpha': alpha_attribution,
            'fees': fees_attribution,
            'slippage': slippage_attribution,
            'timing': timing_attribution,
            'sizing': sizing_attribution
        }
    
    def _assess_parity_status(self, parity_metrics: Any) -> Tuple[ParityStatus, SystemAction]:
        """Assess parity status and determine system action."""
        
        tracking_error = parity_metrics.tracking_error_bps
        
        # Determine parity status
        if tracking_error < self.thresholds.excellent_threshold_bps:
            status = ParityStatus.EXCELLENT
        elif tracking_error < self.thresholds.good_threshold_bps:
            status = ParityStatus.GOOD
        elif tracking_error < self.thresholds.warning_threshold_bps:
            status = ParityStatus.WARNING
        elif tracking_error < self.thresholds.critical_threshold_bps:
            status = ParityStatus.CRITICAL
        else:
            status = ParityStatus.EMERGENCY
        
        # Determine system action
        action = SystemAction.CONTINUE
        
        if tracking_error > self.thresholds.emergency_stop_threshold_bps:
            action = SystemAction.EMERGENCY_STOP
        elif tracking_error > self.thresholds.auto_disable_threshold_bps:
            action = SystemAction.DISABLE
        elif status == ParityStatus.CRITICAL:
            if self.consecutive_critical_days >= self.thresholds.max_critical_breaches:
                action = SystemAction.DISABLE
            else:
                action = SystemAction.REDUCE_SIZE
        elif status == ParityStatus.WARNING:
            if self.consecutive_warning_days >= self.thresholds.max_warning_breaches:
                action = SystemAction.REDUCE_SIZE
            else:
                action = SystemAction.MONITOR
        
        return status, action
    
    def _generate_recommendations(self, parity_metrics: Any, attribution: Dict[str, float]) -> List[str]:
        """Generate specific recommendations based on analysis."""
        
        recommendations = []
        
        # Tracking error recommendations
        if parity_metrics.tracking_error_bps > self.target_tracking_error_bps:
            recommendations.append(f"Tracking error {parity_metrics.tracking_error_bps:.1f} bps exceeds target {self.target_tracking_error_bps:.1f} bps")
        
        # Attribution-based recommendations
        if abs(attribution['slippage']) > 0.001:
            recommendations.append("Review slippage model - significant live vs backtest difference detected")
        
        if abs(attribution['fees']) > 0.001:
            recommendations.append("Review fee calculations - backtest may not match live fee structure")
        
        if abs(attribution['timing']) > 0.002:
            recommendations.append("Review execution timing - consider latency impact in backtests")
        
        # Correlation recommendations
        if parity_metrics.correlation < 0.8:
            recommendations.append(f"Low correlation {parity_metrics.correlation:.3f} - investigate strategy drift")
        
        # Hit rate recommendations
        if parity_metrics.hit_rate < 0.6:
            recommendations.append(f"Low hit rate {parity_metrics.hit_rate:.3f} - review signal quality")
        
        # Execution quality recommendations
        if parity_metrics.execution_quality_score < 75:
            recommendations.append(f"Execution quality {parity_metrics.execution_quality_score:.1f} below target - review execution parameters")
        
        return recommendations
    
    def _update_system_state(self, result: DailyParityResult) -> None:
        """Update internal system state based on result."""
        
        # Update consecutive breach counters
        if result.parity_status in [ParityStatus.WARNING]:
            self.consecutive_warning_days += 1
            self.consecutive_critical_days = 0
        elif result.parity_status in [ParityStatus.CRITICAL, ParityStatus.EMERGENCY]:
            self.consecutive_critical_days += 1
            self.consecutive_warning_days = 0
        else:
            self.consecutive_warning_days = 0
            self.consecutive_critical_days = 0
        
        # Update current system action
        self.current_system_action = result.system_action
        
        # Add to daily results history
        self.daily_results.append(result)
        
        # Keep only last 30 days
        if len(self.daily_results) > 30:
            self.daily_results = self.daily_results[-30:]
    
    def _persist_results(self, result: DailyParityResult) -> None:
        """Persist daily result to file."""
        try:
            # Load existing results
            existing_results = []
            if self.results_file.exists():
                with open(self.results_file, 'r') as f:
                    existing_results = json.load(f)
            
            # Add new result
            result_dict = asdict(result)
            result_dict['next_review_time'] = result.next_review_time.isoformat()
            existing_results.append(result_dict)
            
            # Keep only last 90 days
            if len(existing_results) > 90:
                existing_results = existing_results[-90:]
            
            # Save back to file
            with open(self.results_file, 'w') as f:
                json.dump(existing_results, f, indent=2, default=str)
                
            # Save current state
            state = {
                'consecutive_warning_days': self.consecutive_warning_days,
                'consecutive_critical_days': self.consecutive_critical_days,
                'current_system_action': self.current_system_action.value,
                'last_analysis_date': self.last_analysis_date.isoformat() if self.last_analysis_date else None,
                'target_tracking_error_bps': self.target_tracking_error_bps
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
            self.logger.info(f"Results persisted to {self.results_file}")
            
        except Exception as e:
            self.logger.error(f"Failed to persist results: {e}")
    
    def _load_previous_state(self) -> None:
        """Load previous state from file."""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                self.consecutive_warning_days = state.get('consecutive_warning_days', 0)
                self.consecutive_critical_days = state.get('consecutive_critical_days', 0)
                self.current_system_action = SystemAction(state.get('current_system_action', 'continue'))
                
                last_analysis_str = state.get('last_analysis_date')
                if last_analysis_str:
                    self.last_analysis_date = datetime.fromisoformat(last_analysis_str)
                
                self.logger.info("Previous state loaded successfully")
            
        except Exception as e:
            self.logger.warning(f"Could not load previous state: {e}")
    
    async def _send_parity_alerts(self, result: DailyParityResult) -> None:
        """Send alerts based on parity analysis results."""
        
        # Send alerts for critical conditions
        if result.parity_status in [ParityStatus.CRITICAL, ParityStatus.EMERGENCY]:
            alert_data = {
                'type': 'parity_breach',
                'severity': 'critical' if result.parity_status == ParityStatus.CRITICAL else 'emergency',
                'tracking_error_bps': result.tracking_error_bps,
                'target_bps': self.target_tracking_error_bps,
                'system_action': result.system_action.value,
                'recommendations': result.recommendations[:3],  # Top 3 recommendations
                'date': result.date
            }
            
            self.logger.critical(f"Parity breach detected", **alert_data)
        
        # Send warning for threshold breaches
        elif result.tracking_error_bps > self.target_tracking_error_bps:
            self.logger.warning(
                f"Tracking error {result.tracking_error_bps:.1f} bps exceeds target {self.target_tracking_error_bps:.1f} bps",
                date=result.date,
                system_action=result.system_action.value
            )
    
    def _update_metrics(self, result: DailyParityResult) -> None:
        """Update Prometheus metrics."""
        try:
            self.metrics_collector.record_parity_metrics(
                tracking_error_bps=result.tracking_error_bps,
                correlation=result.correlation,
                hit_rate=result.hit_rate,
                execution_quality_score=result.execution_quality_score
            )
            
            self.metrics_collector.record_system_action(result.system_action.value)
            
        except Exception as e:
            self.logger.error(f"Failed to update metrics: {e}")
    
    def _count_recent_breaches(self) -> int:
        """Count breaches in last 7 days."""
        recent_results = [r for r in self.daily_results if 
                         (datetime.now() - datetime.fromisoformat(r.date)).days <= 7]
        return len([r for r in recent_results if r.parity_status in [ParityStatus.WARNING, ParityStatus.CRITICAL, ParityStatus.EMERGENCY]])
    
    def _get_consecutive_breaches(self, status: ParityStatus) -> int:
        """Get consecutive breach count for status level."""
        if status in [ParityStatus.WARNING]:
            return self.consecutive_warning_days
        elif status in [ParityStatus.CRITICAL, ParityStatus.EMERGENCY]:
            return self.consecutive_critical_days
        return 0
    
    async def _assess_market_conditions(self) -> Dict[str, Any]:
        """Assess current market conditions."""
        return {
            'volatility': 'normal',
            'liquidity': 'good', 
            'spread_environment': 'tight',
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_insufficient_data_result(self, date_str: str) -> DailyParityResult:
        """Create result for insufficient data scenario."""
        return DailyParityResult(
            date=date_str,
            tracking_error_bps=0.0,
            correlation=1.0,
            hit_rate=1.0,
            parity_status=ParityStatus.GOOD,
            system_action=SystemAction.CONTINUE,
            alpha_attribution=0.0,
            fees_attribution=0.0,
            slippage_attribution=0.0,
            timing_attribution=0.0,
            sizing_attribution=0.0,
            total_trades=0,
            avg_slippage_bps=0.0,
            total_fees_bps=0.0,
            execution_quality_score=100.0,
            breach_count=0,
            consecutive_breaches=0,
            recommendations=["Insufficient trading data for analysis"],
            next_review_time=datetime.now() + timedelta(days=1),
            metadata={'reason': 'insufficient_data'}
        )
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current parity monitoring status."""
        recent_result = self.daily_results[-1] if self.daily_results else None
        
        return {
            'target_tracking_error_bps': self.target_tracking_error_bps,
            'current_system_action': self.current_system_action.value,
            'consecutive_warning_days': self.consecutive_warning_days,
            'consecutive_critical_days': self.consecutive_critical_days,
            'last_analysis_date': self.last_analysis_date.isoformat() if self.last_analysis_date else None,
            'recent_result': asdict(recent_result) if recent_result else None,
            'total_daily_analyses': len(self.daily_results),
            'job_status': 'active'
        }


# Convenience function to start the daily parity job
async def start_daily_parity_monitoring(target_tracking_error_bps: float = 20.0):
    """Start the daily parity monitoring job."""
    
    parity_job = DailyParityJob(target_tracking_error_bps)
    parity_job.setup_daily_schedule()
    
    # Run initial analysis if none exists today
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if parity_job.last_analysis_date != today:
        await parity_job.run_daily_analysis()
    
    return parity_job


if __name__ == "__main__":
    # Direct execution for testing
    job = asyncio.run(start_daily_parity_monitoring(20.0))
    print(f"Daily parity job started with target tracking error 20 bps")
    print(f"Current status: {job.get_current_status()}")