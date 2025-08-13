"""
Comprehensive tests for Risk Management System
Tests all hard blockers, kill switches, and risk scenarios.
"""

import asyncio
import pytest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.cryptosmarttrader.risk.risk_guard import (
    RiskGuard,
    RiskLevel,
    TradingMode,
    RiskLimits,
    RiskMonitor,
)


class TestRiskGuard:
    """Test enterprise risk management system."""

    @pytest.fixture
    def risk_guard(self, tmp_path):
        """Create RiskGuard instance for testing."""
        config_path = tmp_path / "test_risk_config.json"
        guard = RiskGuard(config_path=config_path)

        # Set test portfolio state
        guard.daily_start_equity = 100000.0
        guard.current_equity = 100000.0
        guard.peak_equity = 100000.0

        return guard

    @pytest.fixture
    def mock_portfolio_state(self):
        """Mock portfolio state data."""
        return {
            "equity": 95000.0,  # 5% loss
            "positions": {
                "BTC-USD": 0.015,  # 1.5% position
                "ETH-USD": 0.020,  # 2% position
                "ADA-USD": 0.010,  # 1% position
            },
            "asset_exposures": {"BTC": 0.015, "ETH": 0.020, "ADA": 0.010},
            "cluster_exposures": {"large_cap": 0.035, "defi": 0.010},
        }

    def test_daily_loss_limits(self, risk_guard):
        """Test daily loss limit triggers."""
        # Test warning level (3% loss)
        risk_guard.update_portfolio_state(
            equity=97000.0,  # 3% loss
            positions={"BTC-USD": 0.01},
        )
        assert risk_guard.current_risk_level == RiskLevel.WARNING
        assert risk_guard.trading_mode == TradingMode.CONSERVATIVE

        # Test critical level (5% loss)
        risk_guard.update_portfolio_state(
            equity=95000.0,  # 5% loss
            positions={"BTC-USD": 0.01},
        )
        assert risk_guard.current_risk_level == RiskLevel.CRITICAL
        assert risk_guard.trading_mode == TradingMode.DEFENSIVE

        # Test emergency level (8% loss) - triggers kill switch
        risk_guard.update_portfolio_state(
            equity=92000.0,  # 8% loss
            positions={"BTC-USD": 0.01},
        )
        assert risk_guard.current_risk_level == RiskLevel.EMERGENCY
        assert risk_guard.kill_switch_active is True
        assert risk_guard.trading_mode == TradingMode.SHUTDOWN
        assert not risk_guard.is_trading_allowed()

    def test_drawdown_limits(self, risk_guard):
        """Test maximum drawdown triggers."""
        # Set peak equity higher to simulate drawdown
        risk_guard.peak_equity = 110000.0

        # Test warning drawdown (5%)
        risk_guard.update_portfolio_state(
            equity=104500.0,  # 5% drawdown from peak
            positions={"ETH-USD": 0.01},
        )
        assert risk_guard.current_risk_level == RiskLevel.WARNING

        # Test critical drawdown (10%)
        risk_guard.update_portfolio_state(
            equity=99000.0,  # 10% drawdown from peak
            positions={"ETH-USD": 0.01},
        )
        assert risk_guard.current_risk_level == RiskLevel.CRITICAL

        # Test emergency drawdown (15%) - triggers kill switch
        risk_guard.update_portfolio_state(
            equity=93500.0,  # 15% drawdown from peak
            positions={"ETH-USD": 0.01},
        )
        assert risk_guard.current_risk_level == RiskLevel.EMERGENCY
        assert risk_guard.kill_switch_active is True

    def test_position_size_limits(self, risk_guard):
        """Test position size hard blockers."""
        # Test oversized position (3% when limit is 2%)
        risk_guard.update_portfolio_state(
            equity=100000.0,
            positions={"BTC-USD": 0.03},  # 3% position exceeds 2% limit
        )
        assert risk_guard.current_risk_level == RiskLevel.CRITICAL

        # Test too many positions
        large_position_dict = {
            f"COIN{i}-USD": 0.01 for i in range(60)
        }  # 60 positions when limit is 50
        risk_guard.update_portfolio_state(equity=100000.0, positions=large_position_dict)
        assert risk_guard.current_risk_level == RiskLevel.WARNING

    def test_exposure_limits(self, risk_guard):
        """Test asset and cluster exposure limits."""
        # Test asset exposure limit (6% when limit is 5%)
        risk_guard.update_portfolio_state(
            equity=100000.0,
            positions={"BTC-USD": 0.01},
            asset_exposures={"BTC": 0.06},  # 6% asset exposure exceeds 5% limit
        )
        assert risk_guard.current_risk_level == RiskLevel.CRITICAL

        # Test cluster exposure limit (25% when limit is 20%)
        risk_guard.update_portfolio_state(
            equity=100000.0,
            positions={"BTC-USD": 0.01},
            cluster_exposures={"large_cap": 0.25},  # 25% cluster exposure exceeds 20% limit
        )
        assert risk_guard.current_risk_level == RiskLevel.WARNING

    def test_data_gap_kill_switch(self, risk_guard):
        """Test data gap detection triggers kill switch."""
        # Simulate data gap longer than 5 minutes
        old_timestamp = datetime.now() - timedelta(minutes=10)
        risk_guard.update_data_quality(old_timestamp, True, 100.0)

        # Check that kill switch is activated
        assert risk_guard.kill_switch_active is True
        assert not risk_guard.is_trading_allowed()

        # Verify event logging
        kill_switch_events = [e for e in risk_guard.risk_events if "kill_switch" in e.event_type]
        assert len(kill_switch_events) > 0
        assert "Data gap" in kill_switch_events[-1].description

    def test_api_reliability_kill_switch(self, risk_guard):
        """Test API reliability monitoring."""
        # Simulate low API success rate
        for i in range(20):
            success = i < 15  # 75% success rate (below 90% threshold)
            risk_guard.update_data_quality(datetime.now(), success, 100.0)

        # Check that kill switch is activated due to low API success rate
        assert risk_guard.kill_switch_active is True

        # Verify specific kill switch reason
        kill_switch_events = [e for e in risk_guard.risk_events if "kill_switch" in e.event_type]
        assert any("API success rate" in e.description for e in kill_switch_events)

    def test_latency_kill_switch(self, risk_guard):
        """Test high latency detection."""
        # Simulate high latency measurements
        high_latency = 8000.0  # 8 seconds (above 5 second limit)
        for _ in range(10):
            risk_guard.update_data_quality(datetime.now(), True, high_latency)

        # Check that kill switch is activated
        assert risk_guard.kill_switch_active is True

        # Verify latency-specific kill switch
        kill_switch_events = [e for e in risk_guard.risk_events if "kill_switch" in e.event_type]
        assert any("latency" in e.description.lower() for e in kill_switch_events)

    def test_manual_kill_switch(self, risk_guard):
        """Test manual kill switch activation and reset."""
        # Test manual activation
        risk_guard.manual_kill_switch("Emergency manual stop")
        assert risk_guard.kill_switch_active is True
        assert risk_guard.trading_mode == TradingMode.SHUTDOWN

        # Test manual reset
        risk_guard.reset_kill_switch("Manual intervention complete")
        assert risk_guard.kill_switch_active is False
        assert risk_guard.trading_mode == TradingMode.ACTIVE
        assert risk_guard.is_trading_allowed()

    def test_risk_status_reporting(self, risk_guard, mock_portfolio_state):
        """Test comprehensive risk status reporting."""
        risk_guard.update_portfolio_state(**mock_portfolio_state)

        status = risk_guard.get_risk_status()

        # Verify all required fields are present
        required_fields = [
            "risk_level",
            "trading_mode",
            "kill_switch_active",
            "daily_pnl",
            "total_drawdown",
            "current_equity",
            "position_count",
            "last_update",
            "limits",
            "recent_events",
        ]

        for field in required_fields:
            assert field in status

        # Verify data types and values
        assert isinstance(status["risk_level"], str)
        assert isinstance(status["kill_switch_active"], bool)
        assert isinstance(status["position_count"], int)
        assert status["position_count"] == len(mock_portfolio_state["positions"])

    def test_daily_reset(self, risk_guard):
        """Test daily tracking reset functionality."""
        # Set some tracking data
        risk_guard.current_equity = 95000.0
        risk_guard.api_success_count = 50
        risk_guard.api_total_count = 60

        # Reset daily tracking
        risk_guard.reset_daily_tracking()

        # Verify reset
        assert risk_guard.daily_start_equity == 95000.0
        assert risk_guard.daily_pnl == 0.0
        assert risk_guard.api_success_count == 0
        assert risk_guard.api_total_count == 0

    def test_risk_limits_persistence(self, risk_guard, tmp_path):
        """Test risk limits save and load functionality."""
        # Modify limits
        risk_guard.limits.daily_loss_critical = 0.04  # Change from default 0.05
        risk_guard.limits.max_position_size = 0.025  # Change from default 0.02

        # Save limits
        risk_guard.save_limits()

        # Create new instance and verify limits are loaded
        new_guard = RiskGuard(config_path=risk_guard.config_path)
        assert new_guard.limits.daily_loss_critical == 0.04
        assert new_guard.limits.max_position_size == 0.025

    def test_progressive_risk_escalation(self, risk_guard):
        """Test progressive risk level escalation."""
        # Start with normal state
        assert risk_guard.current_risk_level == RiskLevel.NORMAL
        assert risk_guard.trading_mode == TradingMode.ACTIVE

        # Move to warning (3% loss)
        risk_guard.update_portfolio_state(97000.0, {"BTC-USD": 0.01})
        assert risk_guard.current_risk_level == RiskLevel.WARNING
        assert risk_guard.trading_mode == TradingMode.CONSERVATIVE

        # Escalate to critical (5% loss)
        risk_guard.update_portfolio_state(95000.0, {"BTC-USD": 0.01})
        assert risk_guard.current_risk_level == RiskLevel.CRITICAL
        assert risk_guard.trading_mode == TradingMode.DEFENSIVE

        # Final escalation to emergency (8% loss)
        risk_guard.update_portfolio_state(92000.0, {"BTC-USD": 0.01})
        assert risk_guard.current_risk_level == RiskLevel.EMERGENCY
        assert risk_guard.trading_mode == TradingMode.SHUTDOWN
        assert risk_guard.kill_switch_active is True


class TestRiskMonitor:
    """Test risk monitoring service."""

    @pytest.fixture
    def risk_monitor(self, tmp_path):
        """Create RiskMonitor for testing."""
        risk_guard = RiskGuard(config_path=tmp_path / "test_config.json")
        return RiskMonitor(risk_guard)

    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, risk_monitor):
        """Test monitoring start and stop."""
        assert not risk_monitor.monitoring_active

        # Start monitoring
        await risk_monitor.start_monitoring(interval_seconds=1)
        assert risk_monitor.monitoring_active
        assert risk_monitor._monitoring_task is not None

        # Stop monitoring
        await risk_monitor.stop_monitoring()
        assert not risk_monitor.monitoring_active

    @pytest.mark.asyncio
    async def test_stale_data_detection(self, risk_monitor):
        """Test detection of stale portfolio data."""
        # Set old update timestamp
        risk_monitor.risk_guard.last_update = datetime.now() - timedelta(minutes=10)

        # Start monitoring for a short period
        await risk_monitor.start_monitoring(interval_seconds=0.1)
        await asyncio.sleep(0.2)  # Let monitoring run
        await risk_monitor.stop_monitoring()

        # Check that kill switch was activated due to stale data
        assert risk_monitor.risk_guard.kill_switch_active is True

        # Verify stale data kill switch event
        events = risk_monitor.risk_guard.risk_events
        stale_events = [e for e in events if "stale" in e.description.lower()]
        assert len(stale_events) > 0


class TestRiskScenarios:
    """Test comprehensive risk scenarios and edge cases."""

    @pytest.fixture
    def risk_system(self, tmp_path):
        """Complete risk system for scenario testing."""
        risk_guard = RiskGuard(config_path=tmp_path / "scenario_config.json")
        risk_monitor = RiskMonitor(risk_guard)
        return risk_guard, risk_monitor

    def test_flash_crash_scenario(self, risk_system):
        """Test flash crash scenario with rapid portfolio decline."""
        risk_guard, _ = risk_system

        # Simulate flash crash: 10% drop in 1 minute
        risk_guard.peak_equity = 100000.0

        # Rapid decline
        risk_guard.update_portfolio_state(95000.0, {"BTC-USD": 0.02})  # 5% down
        assert risk_guard.current_risk_level == RiskLevel.CRITICAL

        risk_guard.update_portfolio_state(90000.0, {"BTC-USD": 0.02})  # 10% down
        assert risk_guard.current_risk_level == RiskLevel.EMERGENCY
        assert risk_guard.kill_switch_active is True

        # Verify multiple risk events logged
        assert len(risk_guard.risk_events) >= 2
        assert any("critical" in e.event_type for e in risk_guard.risk_events)
        assert any("emergency" in e.event_type for e in risk_guard.risk_events)

    def test_data_feed_failure_scenario(self, risk_system):
        """Test complete data feed failure scenario."""
        risk_guard, _ = risk_system

        # Simulate complete API failure
        for _ in range(20):
            risk_guard.update_data_quality(
                datetime.now(), False, 10000.0
            )  # All failures, high latency

        # Both API reliability and latency should trigger kill switch
        assert risk_guard.kill_switch_active is True

        # Verify multiple kill switch triggers
        kill_events = [e for e in risk_guard.risk_events if "kill_switch" in e.event_type]
        assert len(kill_events) >= 1

    def test_concentration_risk_scenario(self, risk_system):
        """Test concentration risk with large positions."""
        risk_guard, _ = risk_system

        # Simulate high concentration in single asset
        risk_guard.update_portfolio_state(
            equity=100000.0,
            positions={"BTC-USD": 0.04},  # 4% position (above 2% limit)
            asset_exposures={"BTC": 0.08},  # 8% asset exposure (above 5% limit)
            cluster_exposures={"crypto": 0.30},  # 30% cluster exposure (above 20% limit)
        )

        # Should trigger critical risk due to position and asset exposure
        assert risk_guard.current_risk_level == RiskLevel.CRITICAL

        # Verify multiple risk events
        events = risk_guard.risk_events
        assert any("position_size" in e.event_type for e in events)
        assert any("asset_exposure" in e.event_type for e in events)

    @pytest.mark.asyncio
    async def test_recovery_scenario(self, risk_system):
        """Test system recovery after kill switch activation."""
        risk_guard, risk_monitor = risk_system

        # Trigger kill switch with emergency loss
        risk_guard.update_portfolio_state(92000.0, {"BTC-USD": 0.01})  # 8% loss
        assert risk_guard.kill_switch_active is True

        # Simulate recovery
        risk_guard.reset_kill_switch("Manual recovery after review")
        assert not risk_guard.kill_switch_active
        assert risk_guard.is_trading_allowed()

        # Verify recovery event logged
        recovery_events = [e for e in risk_guard.risk_events if "reset" in e.event_type]
        assert len(recovery_events) > 0
        assert "recovery" in recovery_events[-1].description.lower()


# Utility functions for testing
def simulate_trading_day(risk_guard: RiskGuard, scenarios: list):
    """Simulate a full trading day with various scenarios."""
    risk_guard.reset_daily_tracking()

    for hour, scenario in enumerate(scenarios):
        equity = scenario.get("equity", 100000.0)
        positions = scenario.get("positions", {})

        risk_guard.update_portfolio_state(equity, positions)

        # Log hourly status
        status = risk_guard.get_risk_status()
        print(
            f"Hour {hour}: Risk Level {status['risk_level']}, "
            f"PnL {status['daily_pnl']:.2%}, "
            f"Kill Switch: {status['kill_switch_active']}"
        )

        if risk_guard.kill_switch_active:
            print(f"⚠️ Trading halted at hour {hour}")
            break

    return risk_guard.get_risk_status()


if __name__ == "__main__":
    # Run basic functionality test
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test basic risk guard functionality
        guard = RiskGuard(config_path=Path(tmp_dir) / "test_config.json")

        print("🧪 Testing Risk Management System")
        print("=" * 50)

        # Test normal operation
        guard.update_portfolio_state(100000.0, {"BTC-USD": 0.01})
        print(f"Normal: {guard.get_risk_status()['risk_level']}")

        # Test warning level
        guard.update_portfolio_state(97000.0, {"BTC-USD": 0.01})
        print(f"Warning: {guard.get_risk_status()['risk_level']}")

        # Test critical level
        guard.update_portfolio_state(95000.0, {"BTC-USD": 0.01})
        print(f"Critical: {guard.get_risk_status()['risk_level']}")

        # Test kill switch
        guard.update_portfolio_state(92000.0, {"BTC-USD": 0.01})
        print(f"Emergency: {guard.get_risk_status()['risk_level']}")
        print(f"Kill Switch Active: {guard.kill_switch_active}")
        print(f"Trading Allowed: {guard.is_trading_allowed()}")

        print("\n✅ Risk Management System Test Complete")
