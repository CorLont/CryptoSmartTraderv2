#!/usr/bin/env python3
"""
Quick Risk Management Test - Streamlined testing of core safety features
"""

import sys

sys.path.append(".")

from src.cryptosmarttrader.risk.risk_guard import RiskGuard, RiskMetrics, RiskLevel, TradingMode
from src.cryptosmarttrader.execution.execution_policy import ExecutionPolicy
from datetime import datetime


def test_risk_scenarios():
    """Test critical risk management scenarios"""

    print("🛡️ CRITICAL RISK MANAGEMENT TESTING")
    print("=" * 45)

    # Initialize systems
    risk_guard = RiskGuard()
    execution_policy = ExecutionPolicy()

    test_results = []

    # Test 1: Daily Loss Limit (6% loss should trigger emergency)
    print("\n🧪 Test 1: Daily Loss Limit (6% loss)")

    loss_metrics = RiskMetrics(
        daily_pnl=-6000,  # $6k loss on $100k portfolio
        daily_pnl_percent=-6.0,  # 6% daily loss
        max_drawdown=6000,
        max_drawdown_percent=6.0,
        total_exposure=50000,
        position_count=3,
        largest_position_percent=1.5,
        correlation_risk=0.2,
        data_quality_score=0.95,
        last_signal_age_minutes=2,
    )

    risk_level = risk_guard.assess_risk_level(loss_metrics)
    emergency_triggered = risk_level in [RiskLevel.EMERGENCY, RiskLevel.SHUTDOWN]

    result1 = {
        "name": "Daily Loss Limit",
        "daily_loss": -6.0,
        "risk_level": risk_level.value,
        "emergency_triggered": emergency_triggered,
        "passed": emergency_triggered,  # Should trigger emergency at 6% loss
    }
    test_results.append(result1)

    status = "✅ PASSED" if result1["passed"] else "❌ FAILED"
    print(f"Daily loss: {loss_metrics.daily_pnl_percent}%")
    print(f"Risk level: {risk_level.value}")
    print(f"Emergency triggered: {emergency_triggered}")
    print(f"Result: {status}")

    # Test 2: Max Drawdown (12% drawdown should trigger shutdown)
    print("\n🧪 Test 2: Max Drawdown (12% drawdown)")

    drawdown_metrics = RiskMetrics(
        daily_pnl=-2000,
        daily_pnl_percent=-2.0,
        max_drawdown=12000,  # $12k drawdown
        max_drawdown_percent=12.0,  # 12% max drawdown
        total_exposure=60000,
        position_count=4,
        largest_position_percent=1.8,
        correlation_risk=0.25,
        data_quality_score=0.92,
        last_signal_age_minutes=3,
    )

    risk_level = risk_guard.assess_risk_level(drawdown_metrics)
    shutdown_triggered = risk_level == RiskLevel.SHUTDOWN

    result2 = {
        "name": "Max Drawdown",
        "max_drawdown": 12.0,
        "risk_level": risk_level.value,
        "shutdown_triggered": shutdown_triggered,
        "passed": shutdown_triggered,  # Should trigger shutdown at 12% drawdown
    }
    test_results.append(result2)

    status = "✅ PASSED" if result2["passed"] else "❌ FAILED"
    print(f"Max drawdown: {drawdown_metrics.max_drawdown_percent}%")
    print(f"Risk level: {risk_level.value}")
    print(f"Shutdown triggered: {shutdown_triggered}")
    print(f"Result: {status}")

    # Test 3: Position Size Limits
    print("\n🧪 Test 3: Position Size Enforcement")

    # Test normal position (1.5% - should pass)
    normal_position = {
        "symbol": "BTC/USD",
        "size_usd": 1500,
        "portfolio_value": 100000,
        "side": "buy",
    }

    normal_allowed = execution_policy.validate_order_size(
        normal_position["size_usd"], normal_position["portfolio_value"]
    )

    # Test oversized position (3% - should fail)
    oversized_position = {
        "symbol": "ETH/USD",
        "size_usd": 3000,
        "portfolio_value": 100000,
        "side": "buy",
    }

    oversized_blocked = not execution_policy.validate_order_size(
        oversized_position["size_usd"], oversized_position["portfolio_value"]
    )

    result3 = {
        "name": "Position Size Limits",
        "normal_position_pct": 1.5,
        "normal_allowed": normal_allowed,
        "oversized_position_pct": 3.0,
        "oversized_blocked": oversized_blocked,
        "passed": normal_allowed and oversized_blocked,
    }
    test_results.append(result3)

    status = "✅ PASSED" if result3["passed"] else "❌ FAILED"
    print(f"Normal position (1.5%): {'Allowed' if normal_allowed else 'Blocked'}")
    print(f"Oversized position (3.0%): {'Blocked' if oversized_blocked else 'Allowed'}")
    print(f"Result: {status}")

    # Test 4: Data Quality Circuit Breaker
    print("\n🧪 Test 4: Data Quality Circuit Breaker")

    poor_data_metrics = RiskMetrics(
        daily_pnl=500,
        daily_pnl_percent=0.5,
        max_drawdown=1000,
        max_drawdown_percent=1.0,
        total_exposure=30000,
        position_count=2,
        largest_position_percent=1.2,
        correlation_risk=0.15,
        data_quality_score=0.65,  # Poor data quality (below 0.7 threshold)
        last_signal_age_minutes=45,  # Stale signals (above 30min threshold)

    risk_level = risk_guard.assess_risk_level(poor_data_metrics)
    quality_protection = risk_level != RiskLevel.NORMAL  # Should escalate due to poor data

    result4 = {
        "name": "Data Quality Protection",
        "data_quality_score": 0.65,
        "signal_age_minutes": 45,
        "risk_level": risk_level.value,
        "protection_triggered": quality_protection,
        "passed": quality_protection,
    }
    test_results.append(result4)

    status = "✅ PASSED" if result4["passed"] else "❌ FAILED"
    print(f"Data quality: {poor_data_metrics.data_quality_score:.1%}")
    print(f"Signal age: {poor_data_metrics.last_signal_age_minutes} minutes")
    print(f"Risk escalation: {quality_protection}")
    print(f"Result: {status}")

    # Summary
    print("\n📊 RISK TESTING SUMMARY:")
    print("-" * 30)

    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results if result["passed"])
    pass_rate = (passed_tests / total_tests) * 100

    for result in test_results:
        status = "✅" if result["passed"] else "❌"
        print(f"{status} {result['name']}")

    print(f"\nPass Rate: {passed_tests}/{total_tests} ({pass_rate:.0f}%)")

    if pass_rate >= 90:
        print("✅ RISK MANAGEMENT: PRODUCTION READY")
        return True
    elif pass_rate >= 75:
        print("⚠️ RISK MANAGEMENT: NEEDS ATTENTION")
        return False
    else:
        print("❌ RISK MANAGEMENT: NOT READY")
        return False


if __name__ == "__main__":
    success = test_risk_scenarios()
    print(f"\n🎯 OVERALL RESULT: {'SUCCESS' if success else 'NEEDS IMPROVEMENT'}")
