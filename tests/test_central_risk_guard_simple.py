#!/usr/bin/env python3
"""
Simple test for central risk guard without dependencies
Validates risk limits and kill switch functionality
"""

import sys
import os
import time
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, 'src')

try:
    from cryptosmarttrader.risk.central_risk_guard import (
        CentralRiskGuard, RiskLimits, PositionInfo, DataGap,
        RiskType, RiskLevel
    )
    
    def test_central_risk_guard():
        """Test central risk guard functionality"""
        
        print("🛡️ Testing Central RiskGuard System")
        print("=" * 45)
        
        # Setup with test limits
        limits = RiskLimits(
            max_day_loss_usd=5000.0,
            max_drawdown_percent=3.0,
            max_total_exposure_usd=50000.0,
            max_total_positions=5,
            max_data_gap_minutes=5
        )
        
        risk_guard = CentralRiskGuard(limits)
        
        # Test 1: Normal trade approval
        print("\n1. Testing normal trade approval...")
        result1 = risk_guard.check_trade_risk("BTC/USD", 1000.0, "test_strategy")
        print(f"   Result: {'SAFE' if result1.is_safe else 'BLOCKED'}")
        print(f"   Risk score: {result1.risk_score:.2f}")
        print(f"   Violations: {len(result1.violations)}")
        
        assert result1.is_safe, "Normal trade should be approved"
        print("   ✅ Normal trade approved")
        
        # Test 2: Day loss limit breach
        print("\n2. Testing day loss limit breach...")
        risk_guard.update_daily_pnl(-6000.0)  # Exceeds $5k limit
        
        result2 = risk_guard.check_trade_risk("ETH/USD", 1000.0, "test_strategy")
        print(f"   Daily PnL: ${risk_guard.daily_pnl:,.0f}")
        print(f"   Result: {'SAFE' if result2.is_safe else 'BLOCKED'}")
        print(f"   Kill switch triggered: {risk_guard.kill_switch.is_triggered()}")
        
        assert risk_guard.kill_switch.is_triggered(), "Kill switch should be triggered"
        print("   ✅ Kill switch triggered on day loss limit")
        
        # Reset for further tests
        risk_guard.kill_switch.reset("test_user")
        risk_guard.update_daily_pnl(0.0)
        print("   🔄 Reset kill switch for further testing")
        
        # Test 3: Exposure limit check
        print("\n3. Testing exposure limit...")
        result3 = risk_guard.check_trade_risk("BTC/USD", 60000.0, "test_strategy")  # Exceeds $50k limit
        print(f"   Trade size: $60,000 (limit: $50,000)")
        print(f"   Result: {'SAFE' if result3.is_safe else 'BLOCKED'}")
        print(f"   Violations: {len(result3.violations)}")
        
        exposure_violations = [v for v in result3.violations if v.risk_type == RiskType.MAX_EXPOSURE]
        assert len(exposure_violations) > 0, "Should detect exposure violation"
        
        if exposure_violations:
            violation = exposure_violations[0]
            print(f"   Violation: {violation.description}")
            print(f"   Violation %: {violation.violation_percent:.1f}%")
        
        print("   ✅ Exposure limit enforced")
        
        # Test 4: Position count limit
        print("\n4. Testing position count limit...")
        
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
        
        print(f"   Added {len(risk_guard.positions)} positions")
        
        # Try to add one more (should hit limit)
        result4 = risk_guard.check_trade_risk("NEWSYMBOL", 1000.0, "test_strategy")
        print(f"   Result for new position: {'SAFE' if result4.is_safe else 'BLOCKED'}")
        print(f"   Current positions: {len(risk_guard.positions)}")
        
        position_violations = [v for v in result4.violations if v.risk_type == RiskType.MAX_POSITIONS]
        assert len(position_violations) > 0, "Should detect position limit violation"
        print("   ✅ Position count limit enforced")
        
        # Test 5: Drawdown limit
        print("\n5. Testing drawdown limit...")
        
        risk_guard.peak_equity = 100000.0
        risk_guard.current_equity = 96000.0  # 4% drawdown, exceeds 3% limit
        
        drawdown_pct = (risk_guard.peak_equity - risk_guard.current_equity) / risk_guard.peak_equity * 100
        print(f"   Peak equity: ${risk_guard.peak_equity:,.0f}")
        print(f"   Current equity: ${risk_guard.current_equity:,.0f}")
        print(f"   Drawdown: {drawdown_pct:.1f}% (limit: {limits.max_drawdown_percent:.1f}%)")
        
        result5 = risk_guard.check_trade_risk("DRAWDOWN_TEST", 1000.0, "test_strategy")
        print(f"   Kill switch triggered: {risk_guard.kill_switch.is_triggered()}")
        
        assert risk_guard.kill_switch.is_triggered(), "Kill switch should trigger on drawdown"
        print("   ✅ Drawdown limit enforced with kill switch")
        
        # Reset for gap test
        risk_guard.kill_switch.reset("test_user")
        
        # Test 6: Data gap detection
        print("\n6. Testing data gap detection...")
        
        gap = DataGap(
            symbol="BTC/USD",
            gap_start=datetime.now() - timedelta(minutes=10),
            gap_end=datetime.now() - timedelta(minutes=2),
            gap_minutes=8.0,  # Exceeds 5 min limit
            data_type="price"
        )
        
        risk_guard.report_data_gap(gap)
        print(f"   Reported gap: {gap.gap_minutes:.1f} min (limit: {limits.max_data_gap_minutes} min)")
        
        result6 = risk_guard.check_trade_risk("BTC/USD", 1000.0, "test_strategy")
        gap_violations = [v for v in result6.violations if v.risk_type == RiskType.DATA_GAP]
        
        assert len(gap_violations) > 0, "Should detect data gap violation"
        print(f"   Gap violations detected: {len(gap_violations)}")
        print("   ✅ Data gap detection working")
        
        # Test 7: Risk summary
        print("\n7. Testing risk summary...")
        
        summary = risk_guard.get_risk_summary()
        print(f"   Kill switch status: {summary['kill_switch']['status']}")
        print(f"   Total positions: {summary['current']['total_positions']}")
        print(f"   Total exposure: ${summary['current']['total_exposure_usd']:,.0f}")
        print(f"   Current equity: ${summary['current']['equity']:,.0f}")
        print(f"   Data gaps: {summary['data_gaps']}")
        
        # Utilization percentages
        util = summary['utilization']
        print(f"   Utilization:")
        print(f"     Exposure: {util['exposure_pct']:.1f}%")
        print(f"     Positions: {util['positions_pct']:.1f}%")
        print(f"     Drawdown: {util['drawdown_pct']:.1f}%")
        
        assert 'kill_switch' in summary, "Summary should include kill switch status"
        assert 'current' in summary, "Summary should include current metrics"
        assert 'utilization' in summary, "Summary should include utilization"
        print("   ✅ Risk summary complete and comprehensive")
        
        # Test 8: Kill switch status and reset
        print("\n8. Testing kill switch functionality...")
        
        # Trigger emergency halt
        from cryptosmarttrader.risk.central_risk_guard import trigger_emergency_halt
        trigger_emergency_halt("Manual test halt")
        
        print(f"   Manual halt triggered")
        print(f"   Trading halted: {risk_guard.kill_switch.is_triggered()}")
        
        status = risk_guard.kill_switch.get_status()
        print(f"   Status: {status['status']}")
        print(f"   Reason: {status['reason']}")
        
        assert risk_guard.kill_switch.is_triggered(), "Manual halt should trigger kill switch"
        
        # Reset kill switch
        risk_guard.kill_switch.reset("test_admin")
        print(f"   Kill switch reset")
        print(f"   Trading resumed: {not risk_guard.kill_switch.is_triggered()}")
        
        assert not risk_guard.kill_switch.is_triggered(), "Kill switch should be reset"
        print("   ✅ Kill switch functionality working")
        
        print(f"\n📊 Final Risk Summary:")
        final_summary = risk_guard.get_risk_summary()
        print(f"   System status: {final_summary['kill_switch']['status']}")
        print(f"   Risk checks performed: Multiple comprehensive tests")
        print(f"   All limits enforced: ✅")
        print(f"   Kill switch functional: ✅")
        print(f"   Data gap detection: ✅")
        print(f"   Position tracking: ✅")
        
        print(f"\n🎯 All central risk guard tests passed!")
        return True
        
    if __name__ == "__main__":
        test_central_risk_guard()
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure central_risk_guard.py is properly created")
except Exception as e:
    print(f"❌ Test failed: {e}")
    raise