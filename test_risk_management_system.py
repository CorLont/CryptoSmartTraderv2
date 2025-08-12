"""
Risk Management System Test

Test risk limits, circuit breakers, and kill switch functionality
with comprehensive safety validation.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from cryptosmarttrader.risk.risk_limits import (
    RiskLimitManager, RiskLimit, RiskLimitType, RiskStatus, ActionType
)
from cryptosmarttrader.risk.circuit_breaker import (
    CircuitBreakerManager, CircuitBreakerConfig, CircuitBreakerType, BreakerState
)
from cryptosmarttrader.risk.kill_switch import (
    KillSwitchManager, KillSwitchTrigger, KillSwitchState
)

import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RiskManagementTester:
    """
    Comprehensive test suite for risk management systems
    """
    
    def __init__(self):
        self.test_results = {}
        
        # Initialize components
        self.risk_manager = RiskLimitManager(initial_portfolio_value=100000.0)
        self.circuit_breaker_manager = CircuitBreakerManager()
        self.kill_switch = KillSwitchManager(self.risk_manager, self.circuit_breaker_manager)
        
        # Test callbacks
        self.positions_closed = 0
        self.orders_cancelled = 0
        
        # Setup test callbacks
        self.kill_switch.set_position_close_callback(self._test_close_positions)
        self.kill_switch.set_order_cancel_callback(self._test_cancel_orders)
    
    def _test_close_positions(self) -> bool:
        """Test callback for closing positions"""
        self.positions_closed += 1
        logger.info("Test: Positions closed via callback")
        return True
    
    def _test_cancel_orders(self) -> bool:
        """Test callback for cancelling orders"""
        self.orders_cancelled += 1
        logger.info("Test: Orders cancelled via callback")
        return True
    
    def test_risk_limits_system(self) -> bool:
        """Test risk limits and violation detection"""
        
        logger.info("Testing risk limits system...")
        
        try:
            # Setup test portfolio
            initial_value = 100000.0
            self.risk_manager.current_portfolio_value = initial_value
            self.risk_manager.high_water_mark = initial_value
            
            # Setup cluster allocations
            self.risk_manager.set_cluster_allocation("large_cap", ["BTC/USD", "ETH/USD"])
            self.risk_manager.set_cluster_allocation("meme", ["DOGE/USD", "SHIB/USD"])
            
            # Test 1: Daily loss limit
            logger.info("Testing daily loss limit...")
            
            # Simulate 4% daily loss (should trigger warning)
            loss_amount = initial_value * 0.04
            self.risk_manager.update_portfolio_value(initial_value - loss_amount)
            
            daily_loss_status, daily_loss_value = self.risk_manager.check_daily_loss_limit()
            daily_loss_test = (daily_loss_status == RiskStatus.WARNING and daily_loss_value > 0.03)
            
            logger.info(f"Daily loss: {daily_loss_value:.1%} - Status: {daily_loss_status.value}")
            
            # Test 2: Position size limits
            logger.info("Testing position size limits...")
            
            # Add position that exceeds 5% limit
            large_position_value = initial_value * 0.06  # 6% position
            self.risk_manager.update_position("BTC/USD", large_position_value)
            
            position_violations = self.risk_manager.check_position_size_limits()
            position_test = len(position_violations) > 0 and position_violations[0][1] != RiskStatus.NORMAL
            
            logger.info(f"Position violations: {len(position_violations)}")
            if position_violations:
                symbol, status, pct = position_violations[0]
                logger.info(f"  {symbol}: {pct:.1%} - {status.value}")
            
            # Test 3: Cluster exposure limits
            logger.info("Testing cluster exposure limits...")
            
            # Add positions to large_cap cluster
            self.risk_manager.update_position("ETH/USD", initial_value * 0.10)  # 10%
            # Total large_cap now: 16% (6% BTC + 10% ETH) > 15% limit
            
            cluster_violations = self.risk_manager.check_cluster_exposure_limits()
            cluster_test = len(cluster_violations) > 0
            
            logger.info(f"Cluster violations: {len(cluster_violations)}")
            for cluster, status, pct in cluster_violations:
                logger.info(f"  {cluster}: {pct:.1%} - {status.value}")
            
            # Test 4: Total exposure limit
            logger.info("Testing total exposure limit...")
            
            total_exposure_status, total_exposure_value = self.risk_manager.check_total_exposure_limit()
            total_exposure_test = total_exposure_value > 0.15  # Should be 16%
            
            logger.info(f"Total exposure: {total_exposure_value:.1%} - {total_exposure_status.value}")
            
            # Test 5: Comprehensive violation check
            logger.info("Testing comprehensive violation check...")
            
            violations = self.risk_manager.check_all_limits()
            comprehensive_test = len(violations) >= 3  # Should have multiple violations
            
            logger.info(f"Total violations detected: {len(violations)}")
            for violation in violations:
                logger.info(f"  {violation.limit_name}: {violation.current_value:.3f} > {violation.threshold_value:.3f}")
            
            # Test 6: Risk summary
            risk_summary = self.risk_manager.get_risk_summary()
            summary_test = (
                risk_summary["overall_risk_status"] in ["warning", "critical", "breach"] and
                risk_summary["active_violations"] > 0
            )
            
            logger.info(f"Overall risk status: {risk_summary['overall_risk_status']}")
            logger.info(f"Active violations: {risk_summary['active_violations']}")
            
            overall_success = (daily_loss_test and position_test and cluster_test and 
                             total_exposure_test and comprehensive_test and summary_test)
            
            self.test_results['risk_limits'] = {
                'success': overall_success,
                'daily_loss_test': daily_loss_test,
                'position_test': position_test,
                'cluster_test': cluster_test,
                'total_exposure_test': total_exposure_test,
                'comprehensive_test': comprehensive_test,
                'summary_test': summary_test,
                'violations_detected': len(violations)
            }
            
            return overall_success
            
        except Exception as e:
            logger.error(f"Risk limits test failed: {e}")
            return False
    
    def test_circuit_breakers_system(self) -> bool:
        """Test circuit breaker functionality"""
        
        logger.info("Testing circuit breakers system...")
        
        try:
            # Test 1: Data gap detection
            logger.info("Testing data gap circuit breaker...")
            
            # Don't update data for a while to trigger gap
            old_timestamp = datetime.now() - timedelta(minutes=3)
            self.circuit_breaker_manager.last_data_timestamp = old_timestamp
            
            # Manually check data gaps
            self.circuit_breaker_manager._check_data_gaps()
            
            data_gap_breaker = self.circuit_breaker_manager.breakers.get("data_gap_breaker")
            data_gap_test = data_gap_breaker and data_gap_breaker.state == BreakerState.OPEN
            
            logger.info(f"Data gap breaker state: {data_gap_breaker.state.value if data_gap_breaker else 'None'}")
            
            # Test 2: Latency spike detection
            logger.info("Testing latency spike circuit breaker...")
            
            # Record high latency events
            for _ in range(4):  # Need 3 failures to trigger
                self.circuit_breaker_manager.record_api_latency(3000.0)  # 3 seconds
                time.sleep(0.1)
            
            latency_breaker = self.circuit_breaker_manager.breakers.get("latency_spike_breaker")
            latency_test = latency_breaker and latency_breaker.state == BreakerState.OPEN
            
            logger.info(f"Latency breaker state: {latency_breaker.state.value if latency_breaker else 'None'}")
            
            # Test 3: Prediction anomaly detection
            logger.info("Testing prediction anomaly circuit breaker...")
            
            # Record normal predictions first to establish baseline
            for _ in range(10):
                self.circuit_breaker_manager.record_prediction(np.random.normal(0.1, 0.02))
            
            # Then record anomalous predictions
            for _ in range(4):
                self.circuit_breaker_manager.record_prediction(0.8)  # Very high prediction
                time.sleep(0.1)
            
            anomaly_breaker = self.circuit_breaker_manager.breakers.get("prediction_anomaly_breaker")
            anomaly_test = anomaly_breaker and anomaly_breaker.state == BreakerState.OPEN
            
            logger.info(f"Anomaly breaker state: {anomaly_breaker.state.value if anomaly_breaker else 'None'}")
            
            # Test 4: Execution failure detection
            logger.info("Testing execution failure circuit breaker...")
            
            # Record execution failures
            for _ in range(6):  # Need 5 failures to trigger
                self.circuit_breaker_manager.record_execution_result(False)
                time.sleep(0.1)
            
            execution_breaker = self.circuit_breaker_manager.breakers.get("execution_failure_breaker")
            execution_test = execution_breaker and execution_breaker.state == BreakerState.OPEN
            
            logger.info(f"Execution breaker state: {execution_breaker.state.value if execution_breaker else 'None'}")
            
            # Test 5: Trading allowed check
            trading_allowed, reason = self.circuit_breaker_manager.is_trading_allowed()
            trading_test = not trading_allowed  # Should be blocked due to open breakers
            
            logger.info(f"Trading allowed: {trading_allowed} - {reason}")
            
            # Test 6: System status
            system_status = self.circuit_breaker_manager.get_system_status()
            status_test = len(system_status.get("open_breakers", [])) > 0
            
            logger.info(f"Open breakers: {system_status.get('open_breakers', [])}")
            
            overall_success = (data_gap_test and latency_test and anomaly_test and 
                             execution_test and trading_test and status_test)
            
            self.test_results['circuit_breakers'] = {
                'success': overall_success,
                'data_gap_test': data_gap_test,
                'latency_test': latency_test,
                'anomaly_test': anomaly_test,
                'execution_test': execution_test,
                'trading_test': trading_test,
                'status_test': status_test,
                'open_breakers': len(system_status.get("open_breakers", []))
            }
            
            return overall_success
            
        except Exception as e:
            logger.error(f"Circuit breakers test failed: {e}")
            return False
    
    def test_kill_switch_system(self) -> bool:
        """Test kill switch functionality"""
        
        logger.info("Testing kill switch system...")
        
        try:
            # Test 1: Initial state
            initial_status = self.kill_switch.get_status()
            initial_test = initial_status["state"] == "armed"
            
            logger.info(f"Initial kill switch state: {initial_status['state']}")
            
            # Test 2: Manual trigger
            logger.info("Testing manual kill switch trigger...")
            
            manual_trigger_success = self.kill_switch.manual_trigger(
                "Test trigger",
                "EMERGENCY_OVERRIDE_2024"
            )
            
            manual_test = manual_trigger_success and self.kill_switch.state == KillSwitchState.TRIGGERED
            
            logger.info(f"Manual trigger success: {manual_trigger_success}")
            logger.info(f"Kill switch state after trigger: {self.kill_switch.state.value}")
            
            # Test 3: Emergency actions executed
            emergency_test = self.positions_closed > 0 and self.orders_cancelled > 0
            
            logger.info(f"Positions closed: {self.positions_closed}")
            logger.info(f"Orders cancelled: {self.orders_cancelled}")
            
            # Test 4: Trading blocked
            risk_trading_allowed, risk_reason = self.risk_manager.is_trading_allowed()
            circuit_trading_allowed, circuit_reason = self.circuit_breaker_manager.is_trading_allowed()
            
            trading_blocked_test = not risk_trading_allowed and not circuit_trading_allowed
            
            logger.info(f"Risk manager trading: {risk_trading_allowed} - {risk_reason}")
            logger.info(f"Circuit breaker trading: {circuit_trading_allowed} - {circuit_reason}")
            
            # Test 5: Status reporting
            triggered_status = self.kill_switch.get_status()
            status_test = (
                triggered_status["state"] == "triggered" and
                triggered_status["last_trigger"] is not None
            )
            
            logger.info(f"Kill switch status: {triggered_status['state']}")
            logger.info(f"Last trigger: {triggered_status['last_trigger'] is not None}")
            
            # Test 6: Reset functionality
            logger.info("Testing kill switch reset...")
            
            reset_success = self.kill_switch.reset_kill_switch(
                "EMERGENCY_OVERRIDE_2024",
                "Test reset"
            )
            
            reset_test = reset_success and self.kill_switch.state == KillSwitchState.ARMED
            
            logger.info(f"Reset success: {reset_success}")
            logger.info(f"Kill switch state after reset: {self.kill_switch.state.value}")
            
            # Test 7: Test functionality
            test_success = self.kill_switch.test_kill_switch("EMERGENCY_OVERRIDE_2024")
            functionality_test = test_success
            
            logger.info(f"Kill switch test success: {test_success}")
            
            overall_success = (initial_test and manual_test and emergency_test and 
                             trading_blocked_test and status_test and reset_test and 
                             functionality_test)
            
            self.test_results['kill_switch'] = {
                'success': overall_success,
                'initial_test': initial_test,
                'manual_test': manual_test,
                'emergency_test': emergency_test,
                'trading_blocked_test': trading_blocked_test,
                'status_test': status_test,
                'reset_test': reset_test,
                'functionality_test': functionality_test,
                'positions_closed': self.positions_closed,
                'orders_cancelled': self.orders_cancelled
            }
            
            return overall_success
            
        except Exception as e:
            logger.error(f"Kill switch test failed: {e}")
            return False
    
    def test_integrated_risk_scenarios(self) -> bool:
        """Test integrated risk management scenarios"""
        
        logger.info("Testing integrated risk scenarios...")
        
        try:
            # Reset systems for integration test
            self.risk_manager.emergency_stop_active = False
            self.risk_manager.kill_switch_triggered = False
            self.circuit_breaker_manager.trading_enabled = True
            self.kill_switch.state = KillSwitchState.ARMED
            
            # Scenario 1: Cascade failure
            logger.info("Testing cascade failure scenario...")
            
            # Trigger multiple risk conditions
            self.risk_manager.update_portfolio_value(85000)  # 15% drawdown + 15k daily loss
            self.circuit_breaker_manager.record_api_latency(5000)  # High latency
            
            # Check if kill switch would be triggered by monitoring
            time.sleep(1)  # Let monitoring detect issues
            
            # Manually check trigger conditions
            self.kill_switch._check_trigger_conditions()
            
            cascade_test = (
                self.kill_switch.state == KillSwitchState.TRIGGERED or
                not self.circuit_breaker_manager.trading_enabled
            )
            
            logger.info(f"Cascade failure handled: {cascade_test}")
            
            # Scenario 2: Gradual degradation
            logger.info("Testing gradual degradation scenario...")
            
            # Reset for gradual test
            if self.kill_switch.state == KillSwitchState.TRIGGERED:
                self.kill_switch.reset_kill_switch("EMERGENCY_OVERRIDE_2024", "Test reset")
            
            # Gradual position size increases
            self.risk_manager.update_position("BTC/USD", 30000)  # 3%
            violations_3pct = len(self.risk_manager.check_all_limits())
            
            self.risk_manager.update_position("BTC/USD", 40000)  # 4%
            violations_4pct = len(self.risk_manager.check_all_limits())
            
            self.risk_manager.update_position("BTC/USD", 60000)  # 6%
            violations_6pct = len(self.risk_manager.check_all_limits())
            
            gradual_test = violations_6pct > violations_4pct > violations_3pct
            
            logger.info(f"Violation progression: {violations_3pct} → {violations_4pct} → {violations_6pct}")
            
            # Scenario 3: Recovery testing
            logger.info("Testing recovery scenario...")
            
            # Reduce positions back to safe levels
            self.risk_manager.update_position("BTC/USD", 25000)  # 2.5%
            self.risk_manager.update_portfolio_value(98000)  # Minor loss
            
            # Check if violations cleared
            recovery_violations = self.risk_manager.check_all_limits()
            recovery_test = len(recovery_violations) < violations_6pct
            
            logger.info(f"Recovery violations: {len(recovery_violations)} (was {violations_6pct})")
            
            overall_success = cascade_test and gradual_test and recovery_test
            
            self.test_results['integrated_scenarios'] = {
                'success': overall_success,
                'cascade_test': cascade_test,
                'gradual_test': gradual_test,
                'recovery_test': recovery_test,
                'violation_progression': [violations_3pct, violations_4pct, violations_6pct],
                'recovery_violations': len(recovery_violations)
            }
            
            return overall_success
            
        except Exception as e:
            logger.error(f"Integrated scenarios test failed: {e}")
            return False
    
    def run_comprehensive_tests(self):
        """Run all risk management tests"""
        
        logger.info("=" * 60)
        logger.info("🧪 RISK MANAGEMENT SYSTEM TESTS")
        logger.info("=" * 60)
        
        tests = [
            ("Risk Limits System", self.test_risk_limits_system),
            ("Circuit Breakers System", self.test_circuit_breakers_system),
            ("Kill Switch System", self.test_kill_switch_system),
            ("Integrated Risk Scenarios", self.test_integrated_risk_scenarios)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n📋 {test_name}")
            try:
                success = test_func()
                if success:
                    logger.info(f"✅ {test_name} - PASSED")
                    passed_tests += 1
                else:
                    logger.error(f"❌ {test_name} - FAILED")
            except Exception as e:
                logger.error(f"💥 {test_name} failed with exception: {e}")
        
        # Final results
        logger.info("\n" + "=" * 60)
        logger.info("🏁 RISK MANAGEMENT TEST RESULTS")
        logger.info("=" * 60)
        
        for test_name, _ in tests:
            test_key = test_name.lower().replace(' ', '_')
            result = "✅ PASSED" if self.test_results.get(test_key, {}).get('success', False) else "❌ FAILED"
            logger.info(f"{test_name:<35} {result}")
        
        logger.info("=" * 60)
        logger.info(f"OVERALL: {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")
        
        if passed_tests == total_tests:
            logger.info("🎉 ALL TESTS PASSED - RISK MANAGEMENT READY")
        else:
            logger.warning("⚠️ SOME TESTS FAILED - REVIEW REQUIRED")
        
        # Key metrics summary
        logger.info("\n📊 KEY RISK METRICS:")
        
        if 'risk_limits' in self.test_results:
            risk = self.test_results['risk_limits']
            logger.info(f"• Risk violations detected: {risk.get('violations_detected', 0)}")
        
        if 'circuit_breakers' in self.test_results:
            circuits = self.test_results['circuit_breakers']
            logger.info(f"• Circuit breakers triggered: {circuits.get('open_breakers', 0)}")
        
        if 'kill_switch' in self.test_results:
            kill = self.test_results['kill_switch']
            logger.info(f"• Emergency actions: {kill.get('positions_closed', 0)} positions closed")
        
        return passed_tests == total_tests

def main():
    """Run risk management system tests"""
    
    tester = RiskManagementTester()
    success = tester.run_comprehensive_tests()
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)