#!/usr/bin/env python3
"""
Risk Management Scenario Testing
Comprehensive testing of all risk scenarios and hard blockers.
"""

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cryptosmarttrader.risk.risk_guard import RiskGuard, RiskLevel, TradingMode, RiskMonitor


class RiskScenarioTester:
    """Comprehensive risk scenario testing framework."""
    
    def __init__(self):
        self.test_results = []
        self.passed_tests = 0
        self.failed_tests = 0
    
    def run_all_scenarios(self):
        """Run comprehensive risk testing scenarios."""
        print("🧪 COMPREHENSIVE RISK MANAGEMENT TESTING")
        print("=" * 60)
        
        # Test scenarios
        scenarios = [
            ("Daily Loss Limits", self.test_daily_loss_scenarios),
            ("Drawdown Limits", self.test_drawdown_scenarios),
            ("Position Size Limits", self.test_position_size_scenarios),
            ("Data Quality Kill Switch", self.test_data_quality_scenarios),
            ("API Reliability", self.test_api_reliability_scenarios),
            ("Concentration Risk", self.test_concentration_scenarios),
            ("Flash Crash Response", self.test_flash_crash_scenario),
            ("Recovery Procedures", self.test_recovery_scenarios),
            ("Edge Cases", self.test_edge_cases),
            ("Performance Under Load", self.test_performance_scenarios)
        ]
        
        for scenario_name, test_func in scenarios:
            print(f"\n🔬 Testing: {scenario_name}")
            print("-" * 40)
            
            try:
                result = test_func()
                if result:
                    print(f"✅ {scenario_name}: PASSED")
                    self.passed_tests += 1
                else:
                    print(f"❌ {scenario_name}: FAILED")
                    self.failed_tests += 1
                    
                self.test_results.append({
                    'scenario': scenario_name,
                    'passed': result,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"💥 {scenario_name}: ERROR - {e}")
                self.failed_tests += 1
                self.test_results.append({
                    'scenario': scenario_name,
                    'passed': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        self._print_summary()
        return self.passed_tests / (self.passed_tests + self.failed_tests) if (self.passed_tests + self.failed_tests) > 0 else 0
    
    def test_daily_loss_scenarios(self) -> bool:
        """Test daily loss limit scenarios."""
        guard = RiskGuard()
        guard.daily_start_equity = 100000.0
        
        test_cases = [
            # (equity, expected_risk_level, expected_kill_switch)
            (97000.0, RiskLevel.WARNING, False),     # 3% loss - warning
            (95000.0, RiskLevel.CRITICAL, False),    # 5% loss - critical
            (92000.0, RiskLevel.EMERGENCY, True),    # 8% loss - emergency/kill switch
        ]
        
        for equity, expected_risk, expected_kill in test_cases:
            guard = RiskGuard()  # Reset for each test
            guard.daily_start_equity = 100000.0
            guard.update_portfolio_state(equity, {'BTC-USD': 0.01})
            
            if guard.current_risk_level != expected_risk:
                print(f"   ❌ Daily loss {equity}: Expected {expected_risk}, got {guard.current_risk_level}")
                return False
            
            if guard.kill_switch_active != expected_kill:
                print(f"   ❌ Kill switch {equity}: Expected {expected_kill}, got {guard.kill_switch_active}")
                return False
            
            print(f"   ✅ Daily loss ${equity}: {expected_risk.value} + kill_switch={expected_kill}")
        
        return True
    
    def test_drawdown_scenarios(self) -> bool:
        """Test maximum drawdown scenarios."""
        guard = RiskGuard()
        guard.peak_equity = 110000.0  # Set peak higher to test drawdown
        
        test_cases = [
            # (equity, drawdown_pct, expected_risk_level, expected_kill_switch)
            (104500.0, 5.0, RiskLevel.WARNING, False),    # 5% drawdown - warning
            (99000.0, 10.0, RiskLevel.CRITICAL, False),   # 10% drawdown - critical
            (93500.0, 15.0, RiskLevel.EMERGENCY, True),   # 15% drawdown - emergency/kill switch
        ]
        
        for equity, dd_pct, expected_risk, expected_kill in test_cases:
            guard = RiskGuard()  # Reset for each test
            guard.peak_equity = 110000.0
            guard.daily_start_equity = 110000.0
            guard.update_portfolio_state(equity, {'ETH-USD': 0.01})
            
            actual_dd = (guard.peak_equity - equity) / guard.peak_equity
            
            if abs(actual_dd - dd_pct/100) > 0.001:  # Allow small floating point difference
                print(f"   ❌ Drawdown calculation error: Expected {dd_pct}%, got {actual_dd:.1%}")
                return False
            
            if guard.current_risk_level != expected_risk:
                print(f"   ❌ Drawdown {dd_pct}%: Expected {expected_risk}, got {guard.current_risk_level}")
                return False
            
            if guard.kill_switch_active != expected_kill:
                print(f"   ❌ Kill switch {dd_pct}%: Expected {expected_kill}, got {guard.kill_switch_active}")
                return False
            
            print(f"   ✅ Drawdown {dd_pct}%: {expected_risk.value} + kill_switch={expected_kill}")
        
        return True
    
    def test_position_size_scenarios(self) -> bool:
        """Test position size limit scenarios."""
        guard = RiskGuard()
        
        # Test oversized position (3% when limit is 2%)
        guard.update_portfolio_state(
            100000.0,
            {'BTC-USD': 0.03}  # 3% position exceeds limit
        )
        
        if guard.current_risk_level != RiskLevel.CRITICAL:
            print(f"   ❌ Oversized position: Expected CRITICAL, got {guard.current_risk_level}")
            return False
        
        print("   ✅ Oversized position detection: CRITICAL")
        
        # Test too many positions
        large_positions = {f'COIN{i}-USD': 0.001 for i in range(60)}  # 60 positions
        guard = RiskGuard()  # Reset
        guard.update_portfolio_state(100000.0, large_positions)
        
        if guard.current_risk_level != RiskLevel.WARNING:
            print(f"   ❌ Too many positions: Expected WARNING, got {guard.current_risk_level}")
            return False
        
        print("   ✅ Too many positions detection: WARNING")
        return True
    
    def test_data_quality_scenarios(self) -> bool:
        """Test data quality kill switch scenarios."""
        guard = RiskGuard()
        
        # Test data gap (10 minutes when limit is 5 minutes)
        old_timestamp = datetime.now() - timedelta(minutes=10)
        guard.update_data_quality(old_timestamp, True, 100.0)
        
        if not guard.kill_switch_active:
            print("   ❌ Data gap kill switch not activated")
            return False
        
        print("   ✅ Data gap kill switch: ACTIVATED")
        
        # Test high latency
        guard = RiskGuard()  # Reset
        for _ in range(10):
            guard.update_data_quality(datetime.now(), True, 8000.0)  # 8 second latency
        
        if not guard.kill_switch_active:
            print("   ❌ High latency kill switch not activated")
            return False
        
        print("   ✅ High latency kill switch: ACTIVATED")
        return True
    
    def test_api_reliability_scenarios(self) -> bool:
        """Test API reliability monitoring."""
        guard = RiskGuard()
        
        # Simulate 75% API success rate (below 90% threshold)
        for i in range(20):
            success = i < 15  # 15/20 = 75% success rate
            guard.update_data_quality(datetime.now(), success, 100.0)
        
        if not guard.kill_switch_active:
            print(f"   ❌ API reliability kill switch not activated (success rate: {guard.api_success_count/guard.api_total_count:.1%})")
            return False
        
        print(f"   ✅ API reliability kill switch: ACTIVATED (success rate: {guard.api_success_count/guard.api_total_count:.1%})")
        return True
    
    def test_concentration_scenarios(self) -> bool:
        """Test concentration risk scenarios."""
        guard = RiskGuard()
        
        # Test high asset concentration
        guard.update_portfolio_state(
            100000.0,
            {'BTC-USD': 0.02},
            asset_exposures={'BTC': 0.08},  # 8% exposure exceeds 5% limit
            cluster_exposures={'large_cap': 0.08}
        )
        
        if guard.current_risk_level != RiskLevel.CRITICAL:
            print(f"   ❌ Asset concentration: Expected CRITICAL, got {guard.current_risk_level}")
            return False
        
        print("   ✅ Asset concentration detection: CRITICAL")
        
        # Test cluster concentration
        guard = RiskGuard()  # Reset
        guard.update_portfolio_state(
            100000.0,
            {'ETH-USD': 0.01},
            asset_exposures={'ETH': 0.04},
            cluster_exposures={'defi': 0.25}  # 25% cluster exposure exceeds 20% limit
        )
        
        if guard.current_risk_level != RiskLevel.WARNING:
            print(f"   ❌ Cluster concentration: Expected WARNING, got {guard.current_risk_level}")
            return False
        
        print("   ✅ Cluster concentration detection: WARNING")
        return True
    
    def test_flash_crash_scenario(self) -> bool:
        """Test flash crash scenario."""
        guard = RiskGuard()
        guard.peak_equity = 100000.0
        guard.daily_start_equity = 100000.0
        
        # Simulate rapid 15% decline in portfolio
        decline_steps = [98000, 95000, 92000, 88000, 85000]  # Progressive decline
        
        for i, equity in enumerate(decline_steps):
            guard.update_portfolio_state(equity, {'BTC-USD': 0.02})
            
            loss_pct = (guard.daily_start_equity - equity) / guard.daily_start_equity
            dd_pct = (guard.peak_equity - equity) / guard.peak_equity
            
            print(f"   Step {i+1}: ${equity} (loss: {loss_pct:.1%}, dd: {dd_pct:.1%}) -> {guard.current_risk_level.value}")
            
            # Should trigger kill switch by step 3 (8% daily loss or 12% drawdown)
            if i >= 2 and not guard.kill_switch_active:
                print(f"   ❌ Kill switch should be active by step {i+1}")
                return False
        
        if not guard.kill_switch_active:
            print("   ❌ Flash crash kill switch not activated")
            return False
        
        print("   ✅ Flash crash kill switch: ACTIVATED")
        return True
    
    def test_recovery_scenarios(self) -> bool:
        """Test recovery procedures."""
        guard = RiskGuard()
        
        # Trigger kill switch
        guard.manual_kill_switch("Test activation")
        
        if not guard.kill_switch_active:
            print("   ❌ Manual kill switch activation failed")
            return False
        
        print("   ✅ Manual kill switch activation: SUCCESS")
        
        # Test reset
        guard.reset_kill_switch("Test recovery")
        
        if guard.kill_switch_active:
            print("   ❌ Kill switch reset failed")
            return False
        
        if not guard.is_trading_allowed():
            print("   ❌ Trading not allowed after reset")
            return False
        
        print("   ✅ Kill switch reset: SUCCESS")
        print("   ✅ Trading resumed: SUCCESS")
        return True
    
    def test_edge_cases(self) -> bool:
        """Test edge cases and boundary conditions."""
        guard = RiskGuard()
        
        # Test exactly at thresholds
        test_cases = [
            (97000.0, RiskLevel.WARNING),   # Exactly 3% loss
            (95000.0, RiskLevel.CRITICAL),  # Exactly 5% loss
        ]
        
        for equity, expected_level in test_cases:
            guard = RiskGuard()  # Reset
            guard.daily_start_equity = 100000.0
            guard.update_portfolio_state(equity, {'BTC-USD': 0.01})
            
            if guard.current_risk_level != expected_level:
                print(f"   ❌ Threshold boundary ${equity}: Expected {expected_level}, got {guard.current_risk_level}")
                return False
        
        print("   ✅ Threshold boundaries: CORRECT")
        
        # Test empty portfolio
        guard = RiskGuard()  # Reset
        guard.update_portfolio_state(100000.0, {})
        
        if guard.current_risk_level != RiskLevel.NORMAL:
            print(f"   ❌ Empty portfolio: Expected NORMAL, got {guard.current_risk_level}")
            return False
        
        print("   ✅ Empty portfolio handling: CORRECT")
        return True
    
    def test_performance_scenarios(self) -> bool:
        """Test performance under load."""
        guard = RiskGuard()
        
        import time
        start_time = time.time()
        
        # Simulate 1000 rapid updates
        for i in range(1000):
            equity = 100000.0 - (i * 10)  # Gradual decline
            positions = {f'COIN{i%10}-USD': 0.001}  # Rotating positions
            guard.update_portfolio_state(equity, positions)
        
        elapsed = time.time() - start_time
        updates_per_second = 1000 / elapsed
        
        print(f"   ✅ Performance: {updates_per_second:.0f} updates/second")
        
        # Should handle at least 100 updates per second
        if updates_per_second < 100:
            print(f"   ❌ Performance too slow: {updates_per_second:.0f} updates/second")
            return False
        
        print("   ✅ Performance test: PASSED")
        return True
    
    def _print_summary(self):
        """Print comprehensive test summary."""
        print("\n" + "=" * 60)
        print("📊 RISK MANAGEMENT TEST SUMMARY")
        print("=" * 60)
        
        total_tests = self.passed_tests + self.failed_tests
        success_rate = (self.passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 95:
            print("🏆 EXCELLENT - Enterprise risk management fully validated")
        elif success_rate >= 85:
            print("✅ GOOD - Most risk scenarios handled correctly")
        else:
            print("⚠️ NEEDS IMPROVEMENT - Risk management gaps detected")
        
        # Save detailed results
        results_file = Path("test_results_risk_management.json")
        with open(results_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_tests': total_tests,
                    'passed': self.passed_tests,
                    'failed': self.failed_tests,
                    'success_rate': success_rate,
                    'timestamp': datetime.now().isoformat()
                },
                'detailed_results': self.test_results
            }, f, indent=2)
        
        print(f"📄 Detailed results saved to: {results_file}")


async def test_risk_monitoring():
    """Test risk monitoring service."""
    print("\n🔄 Testing Risk Monitoring Service")
    print("-" * 40)
    
    guard = RiskGuard()
    monitor = RiskMonitor(guard)
    
    # Test monitoring lifecycle
    await monitor.start_monitoring(interval_seconds=0.1)
    print("   ✅ Monitoring started")
    
    # Let it run briefly
    await asyncio.sleep(0.3)
    
    # Test stale data detection
    guard.last_update = datetime.now() - timedelta(minutes=10)
    await asyncio.sleep(0.2)
    
    await monitor.stop_monitoring()
    print("   ✅ Monitoring stopped")
    
    # Check if stale data was detected
    if guard.kill_switch_active:
        print("   ✅ Stale data detection: WORKING")
        return True
    else:
        print("   ❌ Stale data detection: FAILED")
        return False


def main():
    """Run comprehensive risk management testing."""
    print("🚀 STARTING COMPREHENSIVE RISK MANAGEMENT TESTING")
    print("=" * 60)
    
    # Run synchronous tests
    tester = RiskScenarioTester()
    sync_success_rate = tester.run_all_scenarios()
    
    # Run asynchronous tests
    print("\n" + "=" * 60)
    print("🔄 TESTING ASYNCHRONOUS COMPONENTS")
    print("=" * 60)
    
    async def run_async_tests():
        monitor_test = await test_risk_monitoring()
        return monitor_test
    
    async_success = asyncio.run(run_async_tests())
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 FINAL RISK MANAGEMENT VALIDATION")
    print("=" * 60)
    
    overall_success = sync_success_rate >= 0.95 and async_success
    
    print(f"Synchronous Tests: {sync_success_rate:.1%} success rate")
    print(f"Asynchronous Tests: {'PASSED' if async_success else 'FAILED'}")
    print(f"Overall Result: {'🏆 ENTERPRISE READY' if overall_success else '⚠️ NEEDS IMPROVEMENT'}")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)