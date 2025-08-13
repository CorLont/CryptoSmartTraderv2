"""
Comprehensive tests for central risk guard system
Tests all risk limits and kill switch functionality
"""

import time
import threading
from datetime import datetime, timedelta
from src.cryptosmarttrader.risk.central_risk_guard import (
    CentralRiskGuard, RiskLimits, PositionInfo, DataGap,
    RiskType, RiskLevel, KillSwitch
)
from src.cryptosmarttrader.risk.risk_integration import (
    RiskIntegratedExecutionPolicy, RiskAwarePortfolioManager
)


def test_central_risk_guard():
    """Test central risk guard functionality"""
    
    print("🛡️ Testing Central Risk Guard")
    print("=" * 40)
    
    # Setup
    limits = RiskLimits(
        max_day_loss_usd=5000.0,
        max_drawdown_percent=3.0,
        max_total_exposure_usd=50000.0,
        max_total_positions=5
    )
    
    risk_guard = CentralRiskGuard(limits)
    
    # Test 1: Normal trade approval
    print("\n1. Testing normal trade approval...")
    result1 = risk_guard.check_trade_risk("BTC/USD", 1000.0, "test_strategy")
    print(f"   Result: {'SAFE' if result1.is_safe else 'BLOCKED'}")
    print(f"   Risk score: {result1.risk_score:.2f}")
    
    assert result1.is_safe, "Normal trade should be approved"
    print("   ✅ Normal trade approved")
    
    # Test 2: Day loss limit
    print("\n2. Testing day loss limit...")
    risk_guard.update_daily_pnl(-6000.0)  # Exceeds limit
    result2 = risk_guard.check_trade_risk("ETH/USD", 1000.0, "test_strategy")
    print(f"   Result: {'SAFE' if result2.is_safe else 'BLOCKED'}")
    print(f"   Kill switch triggered: {risk_guard.kill_switch.is_triggered()}")
    
    assert risk_guard.kill_switch.is_triggered(), "Kill switch should be triggered"
    print("   ✅ Kill switch triggered on day loss")
    
    # Reset for further tests
    risk_guard.kill_switch.reset("test")
    risk_guard.update_daily_pnl(0.0)
    
    # Test 3: Exposure limit
    print("\n3. Testing exposure limit...")
    result3 = risk_guard.check_trade_risk("BTC/USD", 60000.0, "test_strategy")  # Exceeds limit
    print(f"   Result: {'SAFE' if result3.is_safe else 'BLOCKED'}")
    print(f"   Violations: {len(result3.violations)}")
    
    exposure_violations = [v for v in result3.violations if v.risk_type == RiskType.MAX_EXPOSURE]
    assert len(exposure_violations) > 0, "Should detect exposure violation"
    print("   ✅ Exposure limit enforced")
    
    # Test 4: Position limit
    print("\n4. Testing position limit...")
    
    # Add positions up to limit
    for i in range(5):
        position = PositionInfo(
            symbol=f"SYMBOL{i}",
            size_usd=1000.0,
            entry_price=100.0,
            current_price=101.0,
            unrealized_pnl=10.0,
            timestamp=time.time(),
            strategy_id="test_strategy"
        )
        risk_guard.update_position(position)
    
    result4 = risk_guard.check_trade_risk("NEWSYMBOL", 1000.0, "test_strategy")
    print(f"   Result: {'SAFE' if result4.is_safe else 'BLOCKED'}")
    print(f"   Current positions: {len(risk_guard.positions)}")
    
    position_violations = [v for v in result4.violations if v.risk_type == RiskType.MAX_POSITIONS]
    assert len(position_violations) > 0, "Should detect position limit violation"
    print("   ✅ Position limit enforced")
    
    # Test 5: Data gap reporting
    print("\n5. Testing data gap reporting...")
    
    gap = DataGap(
        symbol="BTC/USD",
        gap_start=datetime.now() - timedelta(minutes=10),
        gap_end=datetime.now() - timedelta(minutes=2),
        gap_minutes=8.0,  # Exceeds default 5 min limit
        data_type="price"
    )
    
    risk_guard.report_data_gap(gap)
    result5 = risk_guard.check_trade_risk("BTC/USD", 1000.0, "test_strategy")
    
    gap_violations = [v for v in result5.violations if v.risk_type == RiskType.DATA_GAP]
    assert len(gap_violations) > 0, "Should detect data gap violation"
    print("   ✅ Data gap detection working")
    
    # Test 6: Drawdown limit
    print("\n6. Testing drawdown limit...")
    
    risk_guard.peak_equity = 100000.0
    risk_guard.current_equity = 96000.0  # 4% drawdown, exceeds 3% limit
    
    result6 = risk_guard.check_trade_risk("TEST", 1000.0, "test_strategy")
    print(f"   Current drawdown: {((100000 - 96000) / 100000) * 100:.1f}%")
    print(f"   Kill switch triggered: {risk_guard.kill_switch.is_triggered()}")
    
    assert risk_guard.kill_switch.is_triggered(), "Kill switch should trigger on drawdown"
    print("   ✅ Drawdown limit enforced")
    
    # Test 7: Risk summary
    print("\n7. Testing risk summary...")
    
    summary = risk_guard.get_risk_summary()
    print(f"   Kill switch status: {summary['kill_switch']['status']}")
    print(f"   Total positions: {summary['current']['total_positions']}")
    print(f"   Total exposure: ${summary['current']['total_exposure_usd']:,.0f}")
    
    assert 'kill_switch' in summary, "Summary should include kill switch status"
    assert 'current' in summary, "Summary should include current metrics"
    print("   ✅ Risk summary complete")
    
    print("\n🎯 All central risk guard tests passed!")
    return True


def test_risk_integration():
    """Test risk integration with execution policy"""
    
    print("\n🔗 Testing Risk Integration")
    print("=" * 30)
    
    # Test integrated execution policy
    from src.cryptosmarttrader.execution.execution_discipline import (
        OrderRequest, MarketConditions, OrderSide
    )
    
    integrated_policy = RiskIntegratedExecutionPolicy()
    
    market = MarketConditions(
        spread_bps=10.0,
        bid_depth_usd=100000.0,
        ask_depth_usd=100000.0,
        volume_1m_usd=500000.0,
        last_price=50000.0,
        bid_price=49995.0,
        ask_price=50005.0,
        timestamp=time.time()
    )
    
    # Normal order should pass both execution and risk checks
    order = OrderRequest(
        symbol="BTC/USD",
        side=OrderSide.BUY,
        size=0.1,
        limit_price=50000.0,
        strategy_id="integration_test"
    )
    
    result = integrated_policy.decide(order, market)
    print(f"   Integrated policy result: {result.decision.value}")
    print(f"   Risk guard included: {'risk_guard' in result.gate_results}")
    
    assert 'risk_guard' in result.gate_results, "Should include risk guard results"
    print("   ✅ Risk integration working")
    
    # Test portfolio manager
    portfolio_mgr = RiskAwarePortfolioManager()
    
    position = PositionInfo(
        symbol="BTC/USD",
        size_usd=5000.0,
        entry_price=50000.0,
        current_price=50500.0,
        unrealized_pnl=50.0,
        timestamp=time.time(),
        strategy_id="integration_test"
    )
    
    portfolio_mgr.add_position(position)
    summary = portfolio_mgr.get_portfolio_summary()
    
    print(f"   Portfolio positions: {summary['total_positions']}")
    print(f"   Portfolio value: ${summary['total_value_usd']:,.0f}")
    
    assert summary['total_positions'] == 1, "Should track position"
    print("   ✅ Portfolio integration working")
    
    print("\n🎯 All integration tests passed!")
    return True


if __name__ == "__main__":
    print("🧪 Running Risk Guard Test Suite")
    print("=" * 50)
    
    try:
        test_central_risk_guard()
        test_risk_integration()
        
        print("\n🎉 ALL RISK GUARD TESTS PASSED!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
