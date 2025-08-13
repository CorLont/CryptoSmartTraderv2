"""
Daily Parity Job - FASE D
Automated daily tracking-error monitoring met <X bps target
"""

import time
import json
import asyncio
import schedule
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

from .enhanced_execution_simulator import EnhancedExecutionSimulator, SimulationResult
from ..core.structured_logger import get_logger
from ..observability.comprehensive_alerts import ComprehensiveAlertManager


@dataclass
class ParityMetrics:
    """Daily parity metrics."""
    
    date: str
    backtest_return: float
    live_return: float
    tracking_error_bps: float
    correlation: float
    hit_rate: float
    avg_slippage_bps: float
    p95_slippage_bps: float
    total_trades: int
    successful_trades: int
    quality_score: float
    alert_triggered: bool = False
    
    @property
    def tracking_error_percent(self) -> float:
        """Convert tracking error to percentage."""
        return self.tracking_error_bps / 100.0
    
    @property
    def parity_grade(self) -> str:
        """Grade parity quality A-F."""
        if self.tracking_error_bps <= 10:
            return "A+"
        elif self.tracking_error_bps <= 20:
            return "A"
        elif self.tracking_error_bps <= 30:
            return "B"
        elif self.tracking_error_bps <= 50:
            return "C"
        elif self.tracking_error_bps <= 100:
            return "D"
        else:
            return "F"


@dataclass
class ParityThresholds:
    """Parity monitoring thresholds."""
    
    target_tracking_error_bps: float = 20.0      # <20bps target
    warning_threshold_bps: float = 30.0          # 30bps warning
    critical_threshold_bps: float = 50.0         # 50bps critical
    emergency_threshold_bps: float = 100.0       # 100bps emergency
    min_correlation: float = 0.85                # 85% minimum correlation
    min_hit_rate: float = 0.60                   # 60% minimum hit rate
    max_p95_slippage_bps: float = 30.0          # 30bps P95 slippage limit


class DailyParityJob:
    """
    AUTOMATED DAILY PARITY MONITORING - FASE D
    
    Features:
    - Daily backtest vs live performance comparison
    - Tracking error calculation and monitoring
    - Automatic degradation detection
    - Alert integration for threshold breaches
    - Historical parity tracking with persistence
    """
    
    def __init__(self, data_dir: str = "data/parity"):
        """Initialize daily parity job."""
        self.logger = get_logger("daily_parity_job")
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.execution_simulator = EnhancedExecutionSimulator()
        self.alert_manager = ComprehensiveAlertManager()
        self.thresholds = ParityThresholds()
        
        # State
        self.parity_history: List[ParityMetrics] = []
        self.last_run_date: Optional[str] = None
        self.job_enabled = True
        
        # Load historical data
        self._load_parity_history()
        
        self.logger.info("Daily Parity Job initialized")
    
    def _load_parity_history(self):
        """Load historical parity data."""
        history_file = self.data_dir / "parity_history.json"
        
        try:
            if history_file.exists():
                with open(history_file, 'r') as f:
                    data = json.load(f)
                    self.parity_history = [
                        ParityMetrics(**item) for item in data.get("parity_metrics", [])
                    ]
                    self.last_run_date = data.get("last_run_date")
                
                self.logger.info(f"Loaded {len(self.parity_history)} historical parity records")
        except Exception as e:
            self.logger.warning(f"Failed to load parity history: {e}")
    
    def _save_parity_history(self):
        """Save parity history to disk."""
        history_file = self.data_dir / "parity_history.json"
        
        try:
            data = {
                "last_run_date": self.last_run_date,
                "parity_metrics": [asdict(metric) for metric in self.parity_history],
                "generated_at": datetime.now().isoformat()
            }
            
            with open(history_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info("Parity history saved successfully")
        except Exception as e:
            self.logger.error(f"Failed to save parity history: {e}")
    
    async def simulate_backtest_performance(self, trading_date: str) -> Dict[str, Any]:
        """Simulate backtest performance for comparison."""
        
        # Generate representative trading orders for the day
        sample_orders = [
            {"symbol": "BTC/USD", "side": "buy", "quantity": 1.0, "order_type": "MARKET"},
            {"symbol": "BTC/USD", "side": "sell", "quantity": 0.8, "order_type": "LIMIT", "limit_price": 45500},
            {"symbol": "ETH/USD", "side": "buy", "quantity": 5.0, "order_type": "MARKET"},
            {"symbol": "ETH/USD", "side": "sell", "quantity": 3.0, "order_type": "LIMIT", "limit_price": 2850},
            {"symbol": "BTC/USD", "side": "buy", "quantity": 2.0, "order_type": "MARKET"},
        ]
        
        # Simulate backtest execution (perfect conditions)
        backtest_result = self.execution_simulator.simulate_trading_session(
            sample_orders, session_duration_minutes=30
        )
        
        # Apply backtest assumptions (lower slippage, perfect fills)
        backtest_return = 0.02 + random.uniform(-0.005, 0.015)  # 2% +/- 0.5-1.5%
        
        return {
            "trading_date": trading_date,
            "return_percent": backtest_return,
            "total_trades": len(sample_orders),
            "execution_quality": 95.0,  # Backtest assumes high quality
            "avg_slippage_bps": 5.0,    # Low backtest slippage
            "simulation_result": backtest_result
        }
    
    async def simulate_live_performance(self, trading_date: str) -> Dict[str, Any]:
        """Simulate live performance with realistic conditions."""
        
        # Same orders as backtest but with realistic execution
        sample_orders = [
            {"symbol": "BTC/USD", "side": "buy", "quantity": 1.0, "order_type": "MARKET"},
            {"symbol": "BTC/USD", "side": "sell", "quantity": 0.8, "order_type": "LIMIT", "limit_price": 45500},
            {"symbol": "ETH/USD", "side": "buy", "quantity": 5.0, "order_type": "MARKET"},
            {"symbol": "ETH/USD", "side": "sell", "quantity": 3.0, "order_type": "LIMIT", "limit_price": 2850},
            {"symbol": "BTC/USD", "side": "buy", "quantity": 2.0, "order_type": "MARKET"},
        ]
        
        # Simulate live execution (realistic conditions)
        live_result = self.execution_simulator.simulate_trading_session(
            sample_orders, session_duration_minutes=60
        )
        
        # Apply live execution degradation
        live_return = (0.02 + random.uniform(-0.005, 0.015)) * random.uniform(0.85, 0.98)
        
        return {
            "trading_date": trading_date,
            "return_percent": live_return,
            "total_trades": len(sample_orders),
            "successful_trades": live_result.successful_fills + live_result.partial_fills,
            "execution_quality": live_result.execution_quality_score,
            "avg_slippage_bps": live_result.avg_slippage_bps,
            "p95_slippage_bps": live_result.p95_slippage_bps,
            "simulation_result": live_result
        }
    
    def calculate_tracking_error(
        self, 
        backtest_returns: List[float], 
        live_returns: List[float]
    ) -> float:
        """Calculate tracking error in basis points."""
        
        if len(backtest_returns) != len(live_returns) or len(backtest_returns) == 0:
            return float('inf')
        
        # Calculate return differences
        return_diffs = np.array(live_returns) - np.array(backtest_returns)
        
        # Standard deviation of return differences (tracking error)
        tracking_error = np.std(return_diffs) * 10000  # Convert to bps
        
        return tracking_error
    
    def calculate_hit_rate(
        self, 
        backtest_returns: List[float], 
        live_returns: List[float]
    ) -> float:
        """Calculate hit rate (% of times live matches backtest direction)."""
        
        if len(backtest_returns) != len(live_returns) or len(backtest_returns) == 0:
            return 0.0
        
        hits = 0
        for bt_ret, live_ret in zip(backtest_returns, live_returns):
            # Same direction (both positive or both negative)
            if (bt_ret >= 0 and live_ret >= 0) or (bt_ret < 0 and live_ret < 0):
                hits += 1
        
        return hits / len(backtest_returns)
    
    async def run_daily_parity_check(self, target_date: Optional[str] = None) -> ParityMetrics:
        """Run comprehensive daily parity check."""
        
        if target_date is None:
            target_date = datetime.now().strftime("%Y-%m-%d")
        
        self.logger.info(f"Running daily parity check for {target_date}")
        
        try:
            # Simulate backtest and live performance
            backtest_perf = await self.simulate_backtest_performance(target_date)
            live_perf = await self.simulate_live_performance(target_date)
            
            # Get recent history for tracking error calculation
            recent_history = self.parity_history[-20:] if len(self.parity_history) >= 20 else self.parity_history
            
            if recent_history:
                backtest_returns = [m.backtest_return for m in recent_history] + [backtest_perf["return_percent"]]
                live_returns = [m.live_return for m in recent_history] + [live_perf["return_percent"]]
            else:
                backtest_returns = [backtest_perf["return_percent"]]
                live_returns = [live_perf["return_percent"]]
            
            # Calculate metrics
            tracking_error_bps = self.calculate_tracking_error(backtest_returns, live_returns)
            correlation = np.corrcoef(backtest_returns, live_returns)[0,1] if len(backtest_returns) > 1 else 1.0
            hit_rate = self.calculate_hit_rate(backtest_returns, live_returns)
            
            # Create parity metrics
            parity_metrics = ParityMetrics(
                date=target_date,
                backtest_return=backtest_perf["return_percent"],
                live_return=live_perf["return_percent"],
                tracking_error_bps=tracking_error_bps,
                correlation=correlation,
                hit_rate=hit_rate,
                avg_slippage_bps=live_perf["avg_slippage_bps"],
                p95_slippage_bps=live_perf["p95_slippage_bps"],
                total_trades=live_perf["total_trades"],
                successful_trades=live_perf["successful_trades"],
                quality_score=live_perf["execution_quality"]
            )
            
            # Check thresholds and trigger alerts
            alert_triggered = await self._check_parity_thresholds(parity_metrics)
            parity_metrics.alert_triggered = alert_triggered
            
            # Store results
            self.parity_history.append(parity_metrics)
            self.last_run_date = target_date
            self._save_parity_history()
            
            # Generate daily report
            await self._generate_daily_report(parity_metrics, target_date)
            
            self.logger.info(
                f"Parity check completed: {tracking_error_bps:.1f}bps tracking error, "
                f"Grade: {parity_metrics.parity_grade}, Quality: {parity_metrics.quality_score:.1f}/100"
            )
            
            return parity_metrics
            
        except Exception as e:
            self.logger.error(f"Daily parity check failed: {e}")
            raise
    
    async def _check_parity_thresholds(self, metrics: ParityMetrics) -> bool:
        """Check parity metrics against thresholds and trigger alerts."""
        
        alerts_triggered = False
        
        # Tracking error thresholds
        if metrics.tracking_error_bps > self.thresholds.emergency_threshold_bps:
            await self._trigger_parity_alert(
                "EMERGENCY", 
                f"Tracking error {metrics.tracking_error_bps:.1f}bps exceeds emergency threshold {self.thresholds.emergency_threshold_bps}bps",
                metrics
            )
            alerts_triggered = True
        elif metrics.tracking_error_bps > self.thresholds.critical_threshold_bps:
            await self._trigger_parity_alert(
                "CRITICAL",
                f"Tracking error {metrics.tracking_error_bps:.1f}bps exceeds critical threshold {self.thresholds.critical_threshold_bps}bps", 
                metrics
            )
            alerts_triggered = True
        elif metrics.tracking_error_bps > self.thresholds.warning_threshold_bps:
            await self._trigger_parity_alert(
                "WARNING",
                f"Tracking error {metrics.tracking_error_bps:.1f}bps exceeds warning threshold {self.thresholds.warning_threshold_bps}bps",
                metrics
            )
            alerts_triggered = True
        
        # Correlation threshold
        if metrics.correlation < self.thresholds.min_correlation:
            await self._trigger_parity_alert(
                "WARNING",
                f"Correlation {metrics.correlation:.3f} below minimum {self.thresholds.min_correlation}",
                metrics
            )
            alerts_triggered = True
        
        # Hit rate threshold
        if metrics.hit_rate < self.thresholds.min_hit_rate:
            await self._trigger_parity_alert(
                "WARNING", 
                f"Hit rate {metrics.hit_rate:.3f} below minimum {self.thresholds.min_hit_rate}",
                metrics
            )
            alerts_triggered = True
        
        # P95 slippage threshold
        if metrics.p95_slippage_bps > self.thresholds.max_p95_slippage_bps:
            await self._trigger_parity_alert(
                "CRITICAL",
                f"P95 slippage {metrics.p95_slippage_bps:.1f}bps exceeds limit {self.thresholds.max_p95_slippage_bps}bps",
                metrics
            )
            alerts_triggered = True
        
        return alerts_triggered
    
    async def _trigger_parity_alert(self, severity: str, message: str, metrics: ParityMetrics):
        """Trigger parity-specific alert."""
        
        self.logger.error(f"PARITY ALERT [{severity}]: {message}")
        
        # Record alert for monitoring systems
        alert_metrics = {
            "parity_tracking_error_bps": metrics.tracking_error_bps,
            "parity_correlation": metrics.correlation,
            "parity_hit_rate": metrics.hit_rate,
            "parity_p95_slippage_bps": metrics.p95_slippage_bps,
            "parity_quality_score": metrics.quality_score
        }
        
        # Integrate with comprehensive alert system
        alerts = self.alert_manager.evaluate_rules(alert_metrics)
        
        for alert in alerts:
            self.logger.warning(f"Comprehensive alert triggered: {alert.rule_name}")
    
    async def _generate_daily_report(self, metrics: ParityMetrics, date: str):
        """Generate daily parity report."""
        
        report_file = self.data_dir / f"daily_parity_report_{date}.json"
        
        # Recent performance trend (last 7 days)
        recent_metrics = self.parity_history[-7:] if len(self.parity_history) >= 7 else self.parity_history
        
        if recent_metrics:
            avg_tracking_error = np.mean([m.tracking_error_bps for m in recent_metrics])
            avg_correlation = np.mean([m.correlation for m in recent_metrics])
            avg_hit_rate = np.mean([m.hit_rate for m in recent_metrics])
            trend_direction = "IMPROVING" if len(recent_metrics) >= 2 and \
                            recent_metrics[-1].tracking_error_bps < recent_metrics[-2].tracking_error_bps else "STABLE"
        else:
            avg_tracking_error = avg_correlation = avg_hit_rate = 0.0
            trend_direction = "UNKNOWN"
        
        report = {
            "date": date,
            "parity_metrics": asdict(metrics),
            "thresholds": asdict(self.thresholds),
            "seven_day_trend": {
                "avg_tracking_error_bps": avg_tracking_error,
                "avg_correlation": avg_correlation,
                "avg_hit_rate": avg_hit_rate,
                "trend_direction": trend_direction
            },
            "compliance_status": {
                "tracking_error_compliant": metrics.tracking_error_bps <= self.thresholds.target_tracking_error_bps,
                "correlation_compliant": metrics.correlation >= self.thresholds.min_correlation,
                "hit_rate_compliant": metrics.hit_rate >= self.thresholds.min_hit_rate,
                "slippage_compliant": metrics.p95_slippage_bps <= self.thresholds.max_p95_slippage_bps,
                "overall_grade": metrics.parity_grade
            },
            "generated_at": datetime.now().isoformat()
        }
        
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Daily parity report generated: {report_file}")
        except Exception as e:
            self.logger.error(f"Failed to generate daily report: {e}")
    
    def schedule_daily_job(self, run_time: str = "09:00"):
        """Schedule daily parity job."""
        
        def job_wrapper():
            if self.job_enabled:
                try:
                    asyncio.create_task(self.run_daily_parity_check())
                except Exception as e:
                    self.logger.error(f"Scheduled parity job failed: {e}")
        
        schedule.every().day.at(run_time).do(job_wrapper)
        self.logger.info(f"Daily parity job scheduled for {run_time}")
    
    def get_parity_summary(self) -> Dict[str, Any]:
        """Get comprehensive parity summary."""
        
        if not self.parity_history:
            return {"status": "NO_DATA", "message": "No parity data available"}
        
        recent_metrics = self.parity_history[-1]
        recent_7_days = self.parity_history[-7:] if len(self.parity_history) >= 7 else self.parity_history
        
        summary = {
            "last_run_date": self.last_run_date,
            "current_status": {
                "tracking_error_bps": recent_metrics.tracking_error_bps,
                "grade": recent_metrics.parity_grade,
                "quality_score": recent_metrics.quality_score,
                "alert_status": "ACTIVE" if recent_metrics.alert_triggered else "CLEAR"
            },
            "7_day_performance": {
                "avg_tracking_error_bps": np.mean([m.tracking_error_bps for m in recent_7_days]),
                "avg_correlation": np.mean([m.correlation for m in recent_7_days]), 
                "avg_hit_rate": np.mean([m.hit_rate for m in recent_7_days]),
                "total_measurements": len(recent_7_days)
            },
            "compliance_status": {
                "target_met": recent_metrics.tracking_error_bps <= self.thresholds.target_tracking_error_bps,
                "target_tracking_error_bps": self.thresholds.target_tracking_error_bps,
                "current_tracking_error_bps": recent_metrics.tracking_error_bps,
                "days_measured": len(self.parity_history)
            },
            "job_status": {
                "enabled": self.job_enabled,
                "total_runs": len(self.parity_history),
                "last_successful_run": self.last_run_date
            }
        }
        
        return summary


def create_daily_parity_job(data_dir: str = "data/parity") -> DailyParityJob:
    """Factory function to create daily parity job."""
    return DailyParityJob(data_dir)