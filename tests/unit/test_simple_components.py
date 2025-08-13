#!/usr/bin/env python3
"""
Simple component tests for clean build validation
Tests core functionality without complex dependencies
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.cryptosmarttrader.risk.risk_guard import RiskGuard, RiskLevel, RiskLimits
from src.cryptosmarttrader.execution.execution_policy import ExecutionPolicy
from src.cryptosmarttrader.observability.unified_metrics import UnifiedMetrics


class TestRiskGuard:
    """Test RiskGuard basic functionality"""

    def test_risk_guard_creation(self):
        """Test RiskGuard can be created with defaults"""
        risk_guard = RiskGuard()
        assert risk_guard is not None
        assert hasattr(risk_guard, "limits")

    def test_risk_levels_enum(self):
        """Test RiskLevel enum values"""
        assert RiskLevel.NORMAL.value == "normal"
        assert RiskLevel.EMERGENCY.value == "emergency"
        assert RiskLevel.SHUTDOWN.value == "shutdown"

    def test_risk_limits_creation(self):
        """Test RiskLimits with default values"""
        limits = RiskLimits()
        assert limits.max_daily_loss_percent == 5.0
        assert limits.max_drawdown_percent == 10.0
        assert limits.max_position_size_percent == 2.0


class TestExecutionPolicy:
    """Test ExecutionPolicy basic functionality"""

    def test_execution_policy_creation(self):
        """Test ExecutionPolicy can be created"""
        policy = ExecutionPolicy()
        assert policy is not None

    def test_execution_policy_has_methods(self):
        """Test ExecutionPolicy has required methods"""
        policy = ExecutionPolicy()
        assert hasattr(policy, "check_order_validity")
        assert hasattr(policy, "generate_client_order_id")


class TestUnifiedMetrics:
    """Test UnifiedMetrics basic functionality"""

    def test_metrics_creation(self):
        """Test UnifiedMetrics can be created"""
        metrics = UnifiedMetrics("test_service")
        assert metrics is not None
        assert metrics.service_name == "test_service"

    def test_metrics_summary(self):
        """Test metrics summary generation"""
        metrics = UnifiedMetrics("test")
        summary = metrics.get_metrics_summary()

        assert "service_name" in summary
        assert "alert_rules_count" in summary
        assert summary["service_name"] == "test"

    def test_trading_metrics(self):
        """Test basic trading metrics recording"""
        metrics = UnifiedMetrics("test")

        # Should not raise exceptions
        metrics.record_order("filled", "BTC/USD", "buy")
        metrics.record_slippage("BTC/USD", "market", 25.0)
        metrics.update_drawdown(3.5)
        metrics.record_signal("test_agent", 0.85)

        # Basic validation
        summary = metrics.get_metrics_summary()
        assert summary["service_name"] == "test"


class TestPackageImports:
    """Test package imports work correctly"""

    def test_core_imports(self):
        """Test core package imports work"""
        from src.cryptosmarttrader import RiskGuard
        from src.cryptosmarttrader import ExecutionPolicy

        assert RiskGuard is not None
        assert ExecutionPolicy is not None

    def test_no_experimental_imports(self):
        """Test that experimental modules don't interfere"""
        # Should be able to import without issues
        from src.cryptosmarttrader.core.structured_logger import get_logger

        logger = get_logger("test")
        assert logger is not None


# Simple integration test
def test_basic_system_integration():
    """Test basic system components work together"""

    # Create components
    risk_guard = RiskGuard()
    execution_policy = ExecutionPolicy()
    metrics = UnifiedMetrics("integration_test")

    # Basic interactions
    metrics.record_order("pending", "BTC/USD", "buy")
    metrics.update_drawdown(2.5)

    # Should complete without errors
    summary = metrics.get_metrics_summary()
    assert summary["service_name"] == "integration_test"
    assert summary["alert_rules_count"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
