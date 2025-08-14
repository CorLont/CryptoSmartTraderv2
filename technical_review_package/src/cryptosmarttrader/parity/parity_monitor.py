"""
Parity Monitor Service for CryptoSmartTrader
Background service for continuous backtest-live parity monitoring.
"""

import asyncio
import logging
import signal
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path

from ..core.structured_logger import get_logger
from .daily_parity_reporter import DailyParityReporter, ParityConfiguration
from ..analysis.backtest_parity import BacktestParityAnalyzer


class ParityMonitorService:
    """
    Background service for continuous parity monitoring.

    Features:
    - Continuous backtest-live parity monitoring
    - Automated daily reporting
    - Real-time drift detection
    - Trading disable/enable automation
    - Health status monitoring
    """

    def __init__(self, config: Optional[ParityConfiguration] = None):
        self.config = config or ParityConfiguration()
        self.logger = get_logger("parity_monitor")

        # Core components
        self.daily_reporter = DailyParityReporter(self.config)
        self.backtest_analyzer = BacktestParityAnalyzer()

        # Service state
        self.running = False
        self.last_check = None
        self.health_status = "initializing"

        # Mock data for demonstration
        self.mock_data_enabled = True

        self.logger.info("Parity Monitor Service initialized")

    async def start(self) -> None:
        """Start the parity monitoring service."""
        self.running = True
        self.health_status = "running"

        self.logger.info("Starting Parity Monitor Service")

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        # Main monitoring loop
        try:
            await self._monitoring_loop()
        except Exception as e:
            self.logger.error(f"Parity monitoring failed: {e}")
            self.health_status = "error"
        finally:
            self.health_status = "stopped"
            self.logger.info("Parity Monitor Service stopped")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
        self.running = False

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                current_time = datetime.utcnow()

                # Check if it's time for daily report
                if self._should_generate_daily_report(current_time):
                    await self._generate_daily_report(current_time)

                # Continuous monitoring (every 15 minutes)
                if self._should_perform_continuous_check(current_time):
                    await self._perform_continuous_monitoring()

                # Update health status
                self._update_health_status()

                # Sleep for 60 seconds before next check
                await asyncio.sleep(60)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Continue after error

    def _should_generate_daily_report(self, current_time: datetime) -> bool:
        """Check if it's time to generate daily report."""
        # Parse daily report time (e.g., "08:00")
        report_hour, report_minute = map(int, self.config.daily_report_time.split(":"))

        # Check if we've crossed the daily report time
        if (
            current_time.hour == report_hour
            and current_time.minute >= report_minute
            and (
                self.daily_reporter.last_report_date is None
                or self.daily_reporter.last_report_date.date() < current_time.date()
            )
        ):
            return True

        return False

    def _should_perform_continuous_check(self, current_time: datetime) -> bool:
        """Check if it's time for continuous monitoring."""
        if self.last_check is None:
            return True

        # Perform check every 15 minutes
        return (current_time - self.last_check) >= timedelta(minutes=15)

    async def _generate_daily_report(self, current_time: datetime) -> None:
        """Generate daily parity report."""
        self.logger.info("Generating daily parity report")

        try:
            # Get mock or real data
            backtest_data, live_data = await self._get_performance_data()

            # Generate report
            report = await self.daily_reporter.generate_daily_report(
                backtest_data=backtest_data, live_data=live_data, force_date=current_time
            )

            self.logger.info(
                "Daily parity report generated successfully",
                tracking_error_bps=report.tracking_error_bps,
                system_action=report.system_action.value,
            )

        except Exception as e:
            self.logger.error(f"Failed to generate daily report: {e}")

    async def _perform_continuous_monitoring(self) -> None:
        """Perform continuous parity monitoring."""
        self.last_check = datetime.utcnow()

        try:
            # Get recent performance data
            backtest_data, live_data = await self._get_performance_data()

            # Quick parity check
            parity_metrics = self.daily_reporter.parity_analyzer.analyze_parity(
                backtest_data=backtest_data,
                live_data=live_data,
                period_start=datetime.utcnow() - timedelta(hours=1),
                period_end=datetime.utcnow(),
            )

            # Check for immediate action required
            if parity_metrics.tracking_error_bps > self.config.emergency_threshold_bps:
                self.logger.critical(
                    "Emergency threshold exceeded",
                    tracking_error_bps=parity_metrics.tracking_error_bps,
                )

                # Create emergency report
                from .daily_parity_reporter import DailyParityReport, SystemAction

                emergency_report = DailyParityReport(
                    date=datetime.utcnow().strftime("%Y-%m-%d"),
                    tracking_error_bps=parity_metrics.tracking_error_bps,
                    correlation=parity_metrics.correlation,
                    hit_rate=parity_metrics.hit_rate,
                    execution_quality_score=0.0,
                    parity_status=parity_metrics.status,
                    system_action=SystemAction.EMERGENCY_STOP,
                    component_attribution={},
                    drift_alerts=[],
                    execution_statistics={},
                    recommendations=["Emergency threshold exceeded"],
                    next_check_time=datetime.utcnow(),
                    metadata={},
                )

                # Trigger emergency action
                await self.daily_reporter._execute_system_action(
                    action=SystemAction.EMERGENCY_STOP, report=emergency_report
                )

        except Exception as e:
            self.logger.error(f"Continuous monitoring failed: {e}")

    async def _get_performance_data(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Get backtest and live performance data."""

        if self.mock_data_enabled:
            # Generate mock data for demonstration
            return self._generate_mock_data()
        else:
            # TODO: Implement real data retrieval
            # This would connect to your actual trading system
            return self._generate_mock_data()  # Fallback to mock for now

    def _generate_mock_data(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Generate mock performance data for testing."""
        import random
        import numpy as np

        # Mock backtest data
        base_return = 0.02  # 2% daily return
        backtest_data = {
            "return": base_return + random.uniform(-0.005, 0.005),
            "returns": np.random.normal(0.0002, 0.01, 100),  # Hourly returns
            "prices": np.cumsum(np.random.normal(0.0002, 0.01, 100)) + 50000,
            "sharpe_ratio": 1.5 + random.uniform(-0.2, 0.2),
            "max_drawdown": -0.03 + random.uniform(-0.01, 0.01),
            "trade_count": random.randint(80, 120),
        }

        # Mock live data (with some tracking error)
        tracking_error_bps = random.uniform(5, 25)  # 5-25 bps tracking error
        live_return = base_return + (tracking_error_bps / 10000) * random.choice([-1, 1])

        live_data = {
            "return": live_return,
            "returns": backtest_data["returns"] + np.random.normal(0, 0.002, 100),  # Add noise
            "prices": backtest_data["prices"] + np.random.normal(0, 100, 100),  # Add price noise
            "sharpe_ratio": backtest_data["sharpe_ratio"] - 0.1 + random.uniform(-0.1, 0.1),
            "max_drawdown": backtest_data["max_drawdown"] - 0.005 + random.uniform(-0.005, 0.005),
            "trade_count": backtest_data["trade_count"] + random.randint(-10, 10),
        }

        return backtest_data, live_data

    def _update_health_status(self) -> None:
        """Update service health status."""
        try:
            # Check if trading is disabled
            if self.daily_reporter.trading_disabled:
                self.health_status = "trading_disabled"
            elif self.daily_reporter.consecutive_critical > 0:
                self.health_status = "critical"
            elif self.daily_reporter.consecutive_warnings > 0:
                self.health_status = "warning"
            else:
                self.health_status = "healthy"

        except Exception:
            self.health_status = "error"

    def get_status(self) -> Dict[str, Any]:
        """Get current service status."""
        return {
            "running": self.running,
            "health_status": self.health_status,
            "trading_enabled": self.daily_reporter.is_trading_enabled(),
            "last_check": self.last_check.isoformat() if self.last_check else None,
            "last_report_date": self.daily_reporter.last_report_date.isoformat()
            if self.daily_reporter.last_report_date
            else None,
            "consecutive_warnings": self.daily_reporter.consecutive_warnings,
            "consecutive_critical": self.daily_reporter.consecutive_critical,
            "config": {
                "warning_threshold_bps": self.config.warning_threshold_bps,
                "critical_threshold_bps": self.config.critical_threshold_bps,
                "auto_disable_enabled": self.config.auto_disable_on_drift,
            },
        }


async def main():
    """Main entry point for parity monitor service."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Create and start service
    service = ParityMonitorService()
    await service.start()


if __name__ == "__main__":
    asyncio.run(main())
