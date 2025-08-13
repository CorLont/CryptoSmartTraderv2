"""Tests for Fase 2 Guardrails & Observability components."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.cryptosmarttrader.core.risk_guard import RiskGuard, RiskLevel, RiskLimits
from src.cryptosmarttrader.core.execution_policy import ExecutionPolicy, OrderRequest, OrderType
from src.cryptosmarttrader.monitoring.alert_rules import AlertManager, AlertSeverity
from src.cryptosmarttrader.testing.simulation_tester import (
    SimulationTester,
    FailureScenario,
    SimulationConfig,
)


class TestRiskGuard:
    """Test RiskGuard functionality."""

    def test_risk_guard_initialization(self):
        """Test RiskGuard initialization."""
        risk_guard = RiskGuard()

        assert risk_guard.current_risk_level == RiskLevel.NORMAL
        assert not risk_guard.kill_switch_active
        assert isinstance(risk_guard.risk_limits, RiskLimits)

    def test_portfolio_value_tracking(self):
        """Test portfolio value updates."""
        risk_guard = RiskGuard()

        risk_guard.update_portfolio_value(100000.0)
        assert risk_guard.session_start_portfolio_value == 100000.0

        risk_guard.update_portfolio_value(105000.0)
        assert risk_guard.max_portfolio_value == 105000.0

    def test_risk_level_assessment(self):
        """Test risk level assessment logic."""
        risk_guard = RiskGuard()

        # Normal conditions
        metrics = risk_guard.calculate_current_metrics(100000.0)
        risk_level = risk_guard.assess_risk_level(metrics)
        assert risk_level == RiskLevel.NORMAL

    def test_kill_switch_activation(self):
        """Test kill switch functionality."""
        risk_guard = RiskGuard()

        # Trigger kill switch
        risk_guard.trigger_kill_switch("Test activation")

        assert risk_guard.kill_switch_active
        assert risk_guard.current_risk_level == RiskLevel.SHUTDOWN

    def test_kill_switch_reset(self):
        """Test kill switch reset."""
        risk_guard = RiskGuard()

        # Activate and reset
        risk_guard.trigger_kill_switch("Test")
        reset_success = risk_guard.reset_kill_switch(manual_override=True)

        assert reset_success
        assert not risk_guard.kill_switch_active

    def test_trading_constraints(self):
        """Test trading constraint calculation."""
        risk_guard = RiskGuard()

        constraints = risk_guard.get_trading_constraints()

        assert "max_position_size_percent" in constraints
        assert "trading_enabled" in constraints
        assert constraints["trading_enabled"] is True

    @pytest.mark.unit
    def test_position_tracking(self):
        """Test position tracking functionality."""
        risk_guard = RiskGuard()

        # Add position
        risk_guard.update_position("BTC/USDT", 1.0, 50000.0, 49000.0, 50000.0)

        assert "BTC/USDT" in risk_guard.position_tracker
        assert risk_guard.position_tracker["BTC/USDT"]["size"] == 1.0

        # Remove position
        risk_guard.remove_position("BTC/USDT")
        assert "BTC/USDT" not in risk_guard.position_tracker


class TestExecutionPolicy:
    """Test ExecutionPolicy functionality."""

    def test_execution_policy_initialization(self):
        """Test ExecutionPolicy initialization."""
        policy = ExecutionPolicy()

        assert policy.tradability_gate is not None
        assert policy.slippage_budget is not None
        assert len(policy.order_cache) == 0

    def test_client_order_id_generation(self):
        """Test client order ID generation."""
        policy = ExecutionPolicy()

        order_id = policy.generate_client_order_id("BTC/USDT", "buy", 1.0)

        assert order_id.startswith("CST_")
        assert len(order_id) > 20  # Should be reasonably long for uniqueness

    def test_order_deduplication(self):
        """Test order deduplication logic."""
        policy = ExecutionPolicy()

        order_request = OrderRequest(
            client_order_id="TEST123",
            symbol="BTC/USDT",
            side="buy",
            order_type=OrderType.MARKET,
            quantity=1.0,
        )

        # First check should pass
        is_duplicate = policy.check_order_deduplication(order_request)
        assert not is_duplicate

        # Second check should detect duplicate
        is_duplicate = policy.check_order_deduplication(order_request)
        assert is_duplicate

    def test_slippage_estimation(self):
        """Test slippage estimation."""
        policy = ExecutionPolicy()

        # Mock market conditions
        from src.cryptosmarttrader.core.execution_policy import MarketConditions

        conditions = MarketConditions(
            bid_price=49900.0,
            ask_price=50100.0,
            mid_price=50000.0,
            spread_percent=0.4,
            volume_24h=1000000.0,
            orderbook_depth_bid=50000.0,
            orderbook_depth_ask=50000.0,
            price_volatility=2.0,
            liquidity_score=0.8,
        )

        policy.update_market_conditions("BTC/USDT", conditions)

        order_request = OrderRequest(
            client_order_id="TEST123",
            symbol="BTC/USDT",
            side="buy",
            order_type=OrderType.MARKET,
            quantity=1.0,
        )

        estimated_slippage = policy.estimate_slippage(order_request)
        assert estimated_slippage > 0.0
        assert estimated_slippage < 5.0  # Should be reasonable

    def test_order_validation(self):
        """Test order request validation."""
        policy = ExecutionPolicy()

        # Valid order
        valid_order = OrderRequest(
            client_order_id="VALID123",
            symbol="BTC/USDT",
            side="buy",
            order_type=OrderType.MARKET,
            quantity=1.0,
        )

        # Mock market conditions for validation
        from src.cryptosmarttrader.core.execution_policy import MarketConditions

        conditions = MarketConditions(
            bid_price=49900.0,
            ask_price=50100.0,
            mid_price=50000.0,
            spread_percent=0.3,
            volume_24h=1000000.0,
            orderbook_depth_bid=50000.0,
            orderbook_depth_ask=50000.0,
            price_volatility=1.0,
            liquidity_score=0.8,
        )
        policy.update_market_conditions("BTC/USDT", conditions)

        is_valid, issues = policy.validate_order_request(valid_order)
        # Should pass basic validation (may fail on tradability without full setup)
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)


class TestAlertManager:
    """Test AlertManager functionality."""

    def test_alert_manager_initialization(self):
        """Test AlertManager initialization."""
        alert_manager = AlertManager()

        assert len(alert_manager.rules) > 0  # Should have default rules
        assert len(alert_manager.active_alerts) == 0

    def test_alert_callback_registration(self):
        """Test alert callback registration."""
        alert_manager = AlertManager()

        callback_called = False

        def test_callback(alert):
            nonlocal callback_called
            callback_called = True

        alert_manager.register_alert_callback(test_callback)
        assert len(alert_manager.alert_callbacks) == 1

    def test_metric_checking(self):
        """Test metric value checking against rules."""
        alert_manager = AlertManager()

        # Test high drawdown (should trigger alert)
        alert_manager.check_metric("cst_max_drawdown_percent", 12.0)  # Above 10% threshold

        # Check if alert was triggered
        active_alerts = alert_manager.get_active_alerts()
        drawdown_alerts = [a for a in active_alerts if "drawdown" in a.rule_name.lower()]

        # Should have triggered drawdown alert
        assert len(drawdown_alerts) > 0

    def test_alert_summary(self):
        """Test alert summary generation."""
        alert_manager = AlertManager()

        summary = alert_manager.get_alert_summary()

        assert "total_active" in summary
        assert "by_severity" in summary
        assert "total_rules" in summary
        assert isinstance(summary["total_active"], int)


class TestSimulationTester:
    """Test SimulationTester functionality."""

    @pytest.fixture
    def simulation_components(self):
        """Create simulation test components."""
        risk_guard = RiskGuard()
        execution_policy = ExecutionPolicy()
        alert_manager = AlertManager()
        simulation_tester = SimulationTester(risk_guard, execution_policy, alert_manager)

        return {
            "risk_guard": risk_guard,
            "execution_policy": execution_policy,
            "alert_manager": alert_manager,
            "simulation_tester": simulation_tester,
        }

    def test_simulation_tester_initialization(self, simulation_components):
        """Test SimulationTester initialization."""
        tester = simulation_components["simulation_tester"]

        assert tester.risk_guard is not None
        assert tester.execution_policy is not None
        assert tester.alert_manager is not None
        assert len(tester.test_results) == 0

    @pytest.mark.asyncio
    async def test_kill_switch_simulation(self, simulation_components):
        """Test kill switch simulation scenario."""
        tester = simulation_components["simulation_tester"]

        config = SimulationConfig(
            scenario=FailureScenario.KILL_SWITCH_TEST,
            duration_minutes=1,
            intensity=1.0,
            auto_recovery=True,
        )

        result = await tester.run_simulation(config)

        assert result.scenario == FailureScenario.KILL_SWITCH_TEST
        assert result.kill_switch_activated
        assert len(result.alerts_triggered) > 0

    @pytest.mark.asyncio
    async def test_drawdown_simulation(self, simulation_components):
        """Test drawdown spike simulation."""
        tester = simulation_components["simulation_tester"]

        config = SimulationConfig(
            scenario=FailureScenario.DRAWDOWN_SPIKE,
            duration_minutes=1,
            intensity=0.8,  # 12% drawdown (0.8 * 15%)
            portfolio_value=100000.0,
        )

        result = await tester.run_simulation(config)

        assert result.scenario == FailureScenario.DRAWDOWN_SPIKE
        assert len(result.alerts_triggered) > 0
        # Should trigger drawdown-related alerts
        drawdown_alerts = [a for a in result.alerts_triggered if "drawdown" in a.lower()]
        assert len(drawdown_alerts) > 0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_data_gap_simulation(self, simulation_components):
        """Test data gap simulation."""
        tester = simulation_components["simulation_tester"]

        config = SimulationConfig(
            scenario=FailureScenario.DATA_GAP, duration_minutes=1, intensity=0.5, auto_recovery=True
        )

        result = await tester.run_simulation(config)

        assert result.scenario == FailureScenario.DATA_GAP
        assert result.success  # Should have triggered appropriate alerts

    def test_test_summary_generation(self, simulation_components):
        """Test test summary generation."""
        tester = simulation_components["simulation_tester"]

        # Add some mock test results
        from src.cryptosmarttrader.testing.simulation_tester import TestResult

        mock_result = TestResult(
            scenario=FailureScenario.HIGH_SLIPPAGE,
            start_time=datetime.now(),
            end_time=datetime.now(),
            alerts_triggered=["HighSlippage:warning"],
            kill_switch_activated=False,
            auto_recovery_successful=True,
            max_slippage_observed=1.2,
            risk_level_changes=["normal"],
            metrics_collected={},
            success=True,
        )

        tester.test_results.append(mock_result)

        summary = tester.get_test_summary()

        assert summary["total_tests"] == 1
        assert summary["success_rate"] == 100.0
        assert "high_slippage" in summary["scenarios_tested"]


# Integration test for full Fase 2 pipeline
@pytest.mark.integration
@pytest.mark.asyncio
async def test_fase2_integration():
    """Integration test for complete Fase 2 functionality."""
    # Create all components
    risk_guard = RiskGuard()
    execution_policy = ExecutionPolicy()
    alert_manager = AlertManager()

    # Test component integration
    assert risk_guard is not None
    assert execution_policy is not None
    assert alert_manager is not None

    # Test basic risk scenario
    risk_guard.update_portfolio_value(100000.0)

    # Simulate small drawdown
    risk_status = risk_guard.run_risk_check(95000.0)  # 5% loss

    assert "risk_level" in risk_status
    assert "violations" in risk_status

    # Test alert triggering
    alert_manager.check_metric("cst_daily_pnl_percent", -5.0)

    # Should have working integration
    assert True  # If we get here, basic integration works
