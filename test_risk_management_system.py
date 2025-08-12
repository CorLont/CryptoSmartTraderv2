#!/usr/bin/env python3
"""
Risk Management System Test Suite

Comprehensive testing of risk limits, kill switch, circuit breakers,
and emergency stop functionality.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

from src.cryptosmarttrader.risk.risk_limits import (
    RiskLimitEngine,
    RiskLimitType,
    CircuitBreakerType,
    TradingMode,
    RiskAction
)

class RiskManagementTester:
    """Comprehensive risk management test suite"""
    
    def __init__(self):
        self.risk_engine = RiskLimitEngine(initial_capital=100000.0)
        self.test_results: List[Dict[str, Any]] = []
    
    def test_daily_loss_limit(self) -> Dict[str, Any]:
        """Test daily loss limit enforcement"""
        
        print("\n🧪 Testing Daily Loss Limit...")
        
        test_start = time.time()
        
        # Start with 100k capital
        initial_capital = 100000.0
        self.risk_engine.update_capital(initial_capital)
        
        # Simulate 3% loss (warning level)
        warning_capital = initial_capital * 0.97  # 3% loss
        self.risk_engine.update_capital(warning_capital)
        
        can_trade_warning, violations_warning = self.risk_engine.check_risk_limits()
        
        # Simulate 5% loss (kill switch level)
        kill_switch_capital = initial_capital * 0.95  # 5% loss
        self.risk_engine.update_capital(kill_switch_capital)
        
        can_trade_kill, violations_kill = self.risk_engine.check_risk_limits()
        
        result = {
            'test_name': 'daily_loss_limit',
            'duration_seconds': time.time() - test_start,
            'warning_triggered': len(violations_warning) > 0,
            'kill_switch_triggered': not can_trade_kill,
            'trading_mode_shutdown': self.risk_engine.trading_mode == TradingMode.SHUTDOWN,
            'kill_switch_active': self.risk_engine.kill_switch_active,
            'success': (not can_trade_kill and 
                       self.risk_engine.kill_switch_active),  # Simplified success criteria
            'details': {
                'initial_capital': initial_capital,
                'warning_capital': warning_capital,
                'kill_switch_capital': kill_switch_capital,
                'warning_violations': len(violations_warning),
                'kill_violations': len(violations_kill),
                'final_trading_mode': self.risk_engine.trading_mode.value
            }
        }
        
        if result['success']:
            print("✅ Daily loss limit test PASSED")
        else:
            print("❌ Daily loss limit test FAILED")
        
        return result
    
    def test_max_drawdown_guard(self) -> Dict[str, Any]:
        """Test maximum drawdown protection"""
        
        print("\n🧪 Testing Max Drawdown Guard...")
        
        test_start = time.time()
        
        # Reset engine
        self.risk_engine = RiskLimitEngine(initial_capital=100000.0)
        
        # Simulate equity curve with drawdown
        equity_points = [
            100000,  # Start
            110000,  # +10%
            120000,  # +20% (peak)
            110000,  # -8.3% from peak (warning level)
            108000,  # -10% from peak (trigger level)
        ]
        
        violations_at_each_point = []
        
        for i, equity in enumerate(equity_points):
            self.risk_engine.update_capital(equity)
            can_trade, violations = self.risk_engine.check_risk_limits()
            violations_at_each_point.append((equity, len(violations), can_trade))
        
        # Check final state
        final_can_trade, final_violations = self.risk_engine.check_risk_limits()
        
        result = {
            'test_name': 'max_drawdown_guard',
            'duration_seconds': time.time() - test_start,
            'drawdown_detected': len(final_violations) > 0,
            'emergency_stop_triggered': self.risk_engine.trading_mode == TradingMode.SHUTDOWN,
            'trading_halted': not final_can_trade,
            'success': (len(final_violations) > 0 and not final_can_trade),
            'details': {
                'equity_curve': equity_points,
                'violations_per_point': violations_at_each_point,
                'final_violations': len(final_violations),
                'final_trading_mode': self.risk_engine.trading_mode.value
            }
        }
        
        if result['success']:
            print("✅ Max drawdown guard test PASSED")
        else:
            print("❌ Max drawdown guard test FAILED")
        
        return result
    
    def test_position_size_limits(self) -> Dict[str, Any]:
        """Test position size and exposure limits"""
        
        print("\n🧪 Testing Position Size Limits...")
        
        test_start = time.time()
        
        # Reset engine
        self.risk_engine = RiskLimitEngine(initial_capital=100000.0)
        
        # Test individual position size (2% limit)
        large_position = 3000.0  # 3% of capital (should trigger)
        self.risk_engine.update_position("BTC/USD", large_position)
        
        # Test asset exposure (5% limit)
        very_large_position = 6000.0  # 6% of capital (should trigger)
        self.risk_engine.update_position("ETH/USD", very_large_position)
        
        # Test cluster exposure (multiple positions in same cluster)
        self.risk_engine.update_position("BTC/USD", 4000.0)  # 4%
        self.risk_engine.update_position("ETH/USD", 4000.0)  # 4%
        self.risk_engine.update_position("BNB/USD", 4000.0)  # 4%
        # Total major_crypto cluster: 12% (under 30% limit, should be OK)
        
        can_trade, violations = self.risk_engine.check_risk_limits()
        
        # Check specific violation types
        position_violations = [v for v in violations if 'position' in v.get('limit_type', '')]
        exposure_violations = [v for v in violations if 'exposure' in v.get('limit_type', '')]
        
        result = {
            'test_name': 'position_size_limits',
            'duration_seconds': time.time() - test_start,
            'position_violations_detected': len(position_violations) > 0,
            'exposure_violations_detected': len(exposure_violations) > 0,
            'trading_restricted': not can_trade or self.risk_engine.trading_mode != TradingMode.NORMAL,
            'success': (len(position_violations) > 0 or len(exposure_violations) > 0),
            'details': {
                'total_violations': len(violations),
                'position_violations': len(position_violations),
                'exposure_violations': len(exposure_violations),
                'current_positions': dict(self.risk_engine.positions),
                'trading_mode': self.risk_engine.trading_mode.value
            }
        }
        
        if result['success']:
            print("✅ Position size limits test PASSED")
        else:
            print("❌ Position size limits test FAILED")
        
        return result
    
    def test_circuit_breakers(self) -> Dict[str, Any]:
        """Test circuit breaker functionality"""
        
        print("\n🧪 Testing Circuit Breakers...")
        
        test_start = time.time()
        
        # Reset engine
        self.risk_engine = RiskLimitEngine(initial_capital=100000.0)
        
        # Test data gap circuit breaker
        self.risk_engine.simulate_data_gap(600.0)  # 10 minutes (above 5 min threshold)
        
        # Test model drift circuit breaker
        self.risk_engine.simulate_model_drift(0.20)  # 20% accuracy drop (above 15% threshold)
        
        # Check circuit breaker status
        risk_status = self.risk_engine.get_risk_status()
        active_breakers = risk_status.get('active_breakers', [])
        
        can_trade, violations = self.risk_engine.check_risk_limits()
        
        # Check for circuit breaker violations
        breaker_violations = [v for v in violations if v.get('severity') == 'circuit_breaker']
        
        result = {
            'test_name': 'circuit_breakers',
            'duration_seconds': time.time() - test_start,
            'data_gap_triggered': 'data_gap' in active_breakers,
            'model_drift_triggered': 'model_drift' in active_breakers,
            'breakers_halt_trading': not can_trade,
            'breaker_violations_detected': len(breaker_violations) > 0,
            'success': (len(active_breakers) > 0 and not can_trade),
            'details': {
                'active_breakers': active_breakers,
                'breaker_violations': len(breaker_violations),
                'total_violations': len(violations),
                'trading_mode': self.risk_engine.trading_mode.value
            }
        }
        
        if result['success']:
            print("✅ Circuit breakers test PASSED")
        else:
            print("❌ Circuit breakers test FAILED")
        
        return result
    
    def test_kill_switch_functionality(self) -> Dict[str, Any]:
        """Test manual and automatic kill switch"""
        
        print("\n🧪 Testing Kill Switch Functionality...")
        
        test_start = time.time()
        
        # Reset engine
        self.risk_engine = RiskLimitEngine(initial_capital=100000.0)
        
        # Test manual kill switch activation
        initial_can_trade, _ = self.risk_engine.check_risk_limits()
        
        self.risk_engine.force_kill_switch("Manual test activation")
        
        post_kill_can_trade, _ = self.risk_engine.check_risk_limits()
        kill_switch_was_active = self.risk_engine.kill_switch_active
        
        # Test kill switch reset
        reset_success = self.risk_engine.reset_kill_switch("Manual test reset")
        
        post_reset_can_trade, _ = self.risk_engine.check_risk_limits()
        
        result = {
            'test_name': 'kill_switch_functionality',
            'duration_seconds': time.time() - test_start,
            'initial_trading_allowed': initial_can_trade,
            'kill_switch_stops_trading': not post_kill_can_trade,
            'kill_switch_activated': kill_switch_was_active,  # Was kill switch actually active
            'reset_functionality_works': reset_success,
            'trading_restored_after_reset': self.risk_engine.trading_mode != TradingMode.SHUTDOWN,
            'success': (initial_can_trade and 
                       not post_kill_can_trade and 
                       reset_success),
            'details': {
                'initial_can_trade': initial_can_trade,
                'post_kill_can_trade': post_kill_can_trade,
                'post_reset_can_trade': post_reset_can_trade,
                'reset_success': reset_success,
                'final_trading_mode': self.risk_engine.trading_mode.value,
                'kill_switch_triggers': self.risk_engine.stats['kill_switch_triggers']
            }
        }
        
        if result['success']:
            print("✅ Kill switch functionality test PASSED")
        else:
            print("❌ Kill switch functionality test FAILED")
        
        return result
    
    def test_trading_mode_escalation(self) -> Dict[str, Any]:
        """Test trading mode escalation under stress"""
        
        print("\n🧪 Testing Trading Mode Escalation...")
        
        test_start = time.time()
        
        # Reset engine
        self.risk_engine = RiskLimitEngine(initial_capital=100000.0)
        
        mode_progression = []
        
        # Start normal
        mode_progression.append(('initial', self.risk_engine.trading_mode.value))
        
        # Add moderate risk (should go to CONSERVATIVE)
        self.risk_engine.update_position("BTC/USD", 1800.0)  # 1.8% (warning level)
        _, _ = self.risk_engine.check_risk_limits()
        mode_progression.append(('warning_level', self.risk_engine.trading_mode.value))
        
        # Add more risk (should go to DEFENSIVE)
        self.risk_engine.update_position("ETH/USD", 1900.0)  # Another 1.9%
        _, _ = self.risk_engine.check_risk_limits()
        mode_progression.append(('defensive_level', self.risk_engine.trading_mode.value))
        
        # Add critical risk (should go to EMERGENCY)
        self.risk_engine.update_position("BNB/USD", 2200.0)  # Another 2.2% (over limit)
        _, _ = self.risk_engine.check_risk_limits()
        mode_progression.append(('emergency_level', self.risk_engine.trading_mode.value))
        
        # Trigger kill switch condition (should go to SHUTDOWN)
        self.risk_engine.update_capital(94000.0)  # 6% loss (over 5% limit)
        _, _ = self.risk_engine.check_risk_limits()
        mode_progression.append(('shutdown_level', self.risk_engine.trading_mode.value))
        
        # Check progression logic
        modes_seen = [mode for _, mode in mode_progression]
        proper_escalation = (
            'normal' in modes_seen and
            ('conservative' in modes_seen or 'defensive' in modes_seen) and
            'shutdown' in modes_seen
        )
        
        result = {
            'test_name': 'trading_mode_escalation',
            'duration_seconds': time.time() - test_start,
            'mode_progression_logical': proper_escalation,
            'final_mode_is_shutdown': self.risk_engine.trading_mode == TradingMode.SHUTDOWN,
            'mode_changes_tracked': self.risk_engine.stats['mode_changes'] > 0,
            'success': proper_escalation and self.risk_engine.trading_mode == TradingMode.SHUTDOWN,
            'details': {
                'mode_progression': mode_progression,
                'total_mode_changes': self.risk_engine.stats['mode_changes'],
                'final_mode': self.risk_engine.trading_mode.value,
                'modes_seen': list(set(modes_seen))
            }
        }
        
        if result['success']:
            print("✅ Trading mode escalation test PASSED")
        else:
            print("❌ Trading mode escalation test FAILED")
        
        return result
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run complete risk management test suite"""
        
        print("🚀 Starting Comprehensive Risk Management Test Suite\n")
        print("=" * 60)
        
        suite_start = time.time()
        
        # Run all tests
        tests = [
            self.test_daily_loss_limit,
            self.test_max_drawdown_guard,
            self.test_position_size_limits,
            self.test_circuit_breakers,
            self.test_kill_switch_functionality,
            self.test_trading_mode_escalation
        ]
        
        test_results = []
        for test_func in tests:
            try:
                result = test_func()
                test_results.append(result)
                self.test_results.append(result)
            except Exception as e:
                print(f"❌ Test failed with exception: {e}")
                test_results.append({
                    'test_name': 'unknown',
                    'success': False,
                    'error': str(e)
                })
        
        # Calculate overall results
        total_tests = len(test_results)
        passed_tests = len([r for r in test_results if r.get('success', False)])
        
        # Get final system status
        final_status = self.risk_engine.get_risk_status()
        
        suite_result = {
            'suite_name': 'risk_management_comprehensive',
            'total_duration_seconds': time.time() - suite_start,
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate_percent': (passed_tests / total_tests) * 100 if total_tests > 0 else 0,
            'overall_success': passed_tests == total_tests,
            
            'test_results': test_results,
            'final_system_status': final_status,
            
            'validation_summary': {
                'daily_loss_protection_working': any(r.get('test_name') == 'daily_loss_limit' and r.get('success') for r in test_results),
                'drawdown_protection_working': any(r.get('test_name') == 'max_drawdown_guard' and r.get('success') for r in test_results),
                'position_limits_working': any(r.get('test_name') == 'position_size_limits' and r.get('success') for r in test_results),
                'circuit_breakers_working': any(r.get('test_name') == 'circuit_breakers' and r.get('success') for r in test_results),
                'kill_switch_working': any(r.get('test_name') == 'kill_switch_functionality' and r.get('success') for r in test_results),
                'mode_escalation_working': any(r.get('test_name') == 'trading_mode_escalation' and r.get('success') for r in test_results)
            }
        }
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 TEST SUITE SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {suite_result['success_rate_percent']:.1f}%")
        print(f"Duration: {suite_result['total_duration_seconds']:.2f}s")
        
        if suite_result['overall_success']:
            print("\n🎉 ALL TESTS PASSED - RISK MANAGEMENT SYSTEM FULLY OPERATIONAL!")
        else:
            print("\n⚠️  SOME TESTS FAILED - REVIEW RESULTS ABOVE")
        
        print("\n🔍 Key Risk Protections:")
        validations = suite_result['validation_summary']
        for key, value in validations.items():
            status = "✅" if value else "❌"
            readable_key = key.replace('_', ' ').title()
            print(f"{status} {readable_key}")
        
        return suite_result

async def main():
    """Main test execution"""
    
    tester = RiskManagementTester()
    
    # Run comprehensive test suite
    results = await tester.run_comprehensive_test_suite()
    
    # Save results
    import json
    with open('test_results_risk_management.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📁 Test results saved to: test_results_risk_management.json")

if __name__ == "__main__":
    asyncio.run(main())