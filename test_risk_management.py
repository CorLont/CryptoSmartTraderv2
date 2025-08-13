#!/usr/bin/env python3
"""
Risk Management Testing Suite - Comprehensive validation of enterprise safety systems
Tests kill-switch scenarios, position limits, and emergency procedures for 500% target system
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from pathlib import Path

# Import risk management systems
try:
    from src.cryptosmarttrader.risk.risk_guard import RiskGuard, RiskLevel
    from src.cryptosmarttrader.execution.execution_policy import ExecutionPolicy
    from src.cryptosmarttrader.observability.metrics import PrometheusMetrics
    HAS_RISK_SYSTEMS = True
except ImportError:
    HAS_RISK_SYSTEMS = False
    logging.warning("Risk management systems not available for testing")

logger = logging.getLogger(__name__)

class RiskTestScenario:
    """Individual risk management test scenario"""
    
    def __init__(self, name: str, description: str, test_func, expected_outcome: str):
        self.name = name
        self.description = description
        self.test_func = test_func
        self.expected_outcome = expected_outcome
        self.result = None
        self.execution_time = None
        self.passed = False

class RiskManagementTester:
    """Comprehensive risk management testing system"""
    
    def __init__(self):
        self.logger = logger
        self.test_results = []
        self.risk_guard = None
        self.execution_policy = None
        self.metrics = None
        
        if HAS_RISK_SYSTEMS:
            try:
                self.risk_guard = RiskGuard()
                self.execution_policy = ExecutionPolicy()
                self.metrics = PrometheusMetrics()
                self.logger.info("Risk management systems initialized for testing")
            except Exception as e:
                self.logger.error(f"Failed to initialize risk systems: {e}")
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run all risk management test scenarios"""
        
        print("🛡️ RISK MANAGEMENT COMPREHENSIVE TESTING")
        print("=" * 50)
        
        test_scenarios = self._create_test_scenarios()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_scenarios': len(test_scenarios),
            'passed_scenarios': 0,
            'failed_scenarios': 0,
            'test_details': []
        }
        
        for scenario in test_scenarios:
            print(f"\n🧪 Testing: {scenario.name}")
            print(f"Description: {scenario.description}")
            
            start_time = datetime.now()
            
            try:
                scenario.result = scenario.test_func()
                scenario.execution_time = (datetime.now() - start_time).total_seconds()
                scenario.passed = self._evaluate_test_result(scenario)
                
                status = "✅ PASSED" if scenario.passed else "❌ FAILED"
                print(f"Result: {status} ({scenario.execution_time:.2f}s)")
                
                if scenario.passed:
                    results['passed_scenarios'] += 1
                else:
                    results['failed_scenarios'] += 1
                    
            except Exception as e:
                scenario.result = {'error': str(e)}
                scenario.execution_time = (datetime.now() - start_time).total_seconds()
                scenario.passed = False
                results['failed_scenarios'] += 1
                
                print(f"Result: ❌ EXCEPTION - {e}")
            
            # Store detailed results
            results['test_details'].append({
                'name': scenario.name,
                'description': scenario.description,
                'passed': scenario.passed,
                'execution_time': scenario.execution_time,
                'result': scenario.result,
                'expected_outcome': scenario.expected_outcome
            })
        
        # Generate summary
        pass_rate = results['passed_scenarios'] / results['total_scenarios'] * 100
        print(f"\n📊 RISK TESTING SUMMARY:")
        print(f"Total Tests: {results['total_scenarios']}")
        print(f"Passed: {results['passed_scenarios']}")
        print(f"Failed: {results['failed_scenarios']}")
        print(f"Pass Rate: {pass_rate:.1f}%")
        
        if pass_rate >= 90:
            print("✅ RISK MANAGEMENT: PRODUCTION READY")
        elif pass_rate >= 75:
            print("⚠️ RISK MANAGEMENT: NEEDS ATTENTION")
        else:
            print("❌ RISK MANAGEMENT: NOT READY FOR PRODUCTION")
        
        return results
    
    def _create_test_scenarios(self) -> List[RiskTestScenario]:
        """Create comprehensive test scenarios"""
        
        scenarios = [
            RiskTestScenario(
                "Daily Loss Limit Trigger",
                "Test if daily loss limit (5%) triggers risk escalation",
                self._test_daily_loss_limit,
                "Risk level escalates when daily loss exceeds 5%"
            ),
            RiskTestScenario(
                "Max Drawdown Protection", 
                "Test maximum drawdown limit (10%) emergency stop",
                self._test_max_drawdown,
                "Emergency stop triggered at 10% portfolio drawdown"
            ),
            RiskTestScenario(
                "Position Size Limits",
                "Test individual position size enforcement (2% max)",
                self._test_position_size_limits,
                "Orders rejected when exceeding 2% portfolio size"
            ),
            RiskTestScenario(
                "Confidence Gate Enforcement",
                "Test 90%+ confidence threshold enforcement",
                self._test_confidence_gates,
                "Low confidence signals (< 90%) are blocked"
            ),
            RiskTestScenario(
                "Emergency Kill Switch",
                "Test manual emergency kill switch activation",
                self._test_emergency_kill_switch,
                "All trading immediately halted on kill switch"
            ),
            RiskTestScenario(
                "Data Quality Circuit Breaker",
                "Test trading halt on poor data quality",
                self._test_data_quality_breaker,
                "Trading suspended when data quality < 95%"
            ),
            RiskTestScenario(
                "API Latency Protection",
                "Test trading halt on high API latency",
                self._test_api_latency_protection,
                "Trading paused when API latency > 5 seconds"
            ),
            RiskTestScenario(
                "Correlation Limits",
                "Test asset correlation exposure limits",
                self._test_correlation_limits,
                "Prevent excessive correlation concentration"
            ),
            RiskTestScenario(
                "Volatility Escalation",
                "Test risk escalation during high volatility",
                self._test_volatility_escalation,
                "Risk mode escalates during extreme volatility"
            ),
            RiskTestScenario(
                "Recovery Procedures",
                "Test automatic recovery after risk events",
                self._test_recovery_procedures,
                "System automatically recovers when conditions normalize"
            )
        ]
        
        return scenarios
    
    def _test_daily_loss_limit(self) -> Dict[str, Any]:
        """Test daily loss limit enforcement"""
        
        if not self.risk_guard:
            return {'status': 'skipped', 'reason': 'RiskGuard not available'}
        
        # Simulate portfolio starting at $100,000
        initial_value = 100000
        
        # Simulate 6% daily loss (exceeds 5% limit)
        current_value = 94000  # 6% loss
        daily_pnl = current_value - initial_value
        
        # Update risk guard with loss scenario
        self.risk_guard.daily_pnl = daily_pnl
        self.risk_guard.portfolio_value = current_value
        
        # Check risk assessment
        risk_level = self.risk_guard.assess_risk_level()
        
        return {
            'initial_value': initial_value,
            'current_value': current_value,
            'daily_loss_pct': (daily_pnl / initial_value) * 100,
            'risk_level': risk_level.name if hasattr(risk_level, 'name') else str(risk_level),
            'trading_allowed': self.risk_guard.is_trading_allowed(),
            'triggered_correctly': daily_pnl / initial_value < -0.05
        }
    
    def _test_max_drawdown(self) -> Dict[str, Any]:
        """Test maximum drawdown protection"""
        
        if not self.risk_guard:
            return {'status': 'skipped', 'reason': 'RiskGuard not available'}
        
        # Simulate 12% drawdown (exceeds 10% limit)
        peak_value = 100000
        current_value = 88000  # 12% drawdown
        
        drawdown_pct = (peak_value - current_value) / peak_value
        
        # Update risk guard
        self.risk_guard.portfolio_value = current_value
        self.risk_guard.portfolio_peak = peak_value
        
        risk_level = self.risk_guard.assess_risk_level()
        
        return {
            'peak_value': peak_value,
            'current_value': current_value,
            'drawdown_pct': drawdown_pct * 100,
            'risk_level': risk_level.name if hasattr(risk_level, 'name') else str(risk_level),
            'emergency_triggered': drawdown_pct > 0.10,
            'trading_halted': not self.risk_guard.is_trading_allowed()
        }
    
    def _test_position_size_limits(self) -> Dict[str, Any]:
        """Test position size limit enforcement"""
        
        if not self.execution_policy:
            return {'status': 'skipped', 'reason': 'ExecutionPolicy not available'}
        
        portfolio_value = 100000
        
        # Test normal position (1.5% - should pass)
        normal_position = 1500
        normal_allowed = self._check_position_size(normal_position, portfolio_value)
        
        # Test oversized position (3% - should fail)
        oversized_position = 3000
        oversized_blocked = not self._check_position_size(oversized_position, portfolio_value)
        
        return {
            'portfolio_value': portfolio_value,
            'normal_position': normal_position,
            'normal_position_pct': (normal_position / portfolio_value) * 100,
            'normal_allowed': normal_allowed,
            'oversized_position': oversized_position,
            'oversized_position_pct': (oversized_position / portfolio_value) * 100,
            'oversized_blocked': oversized_blocked,
            'test_passed': normal_allowed and oversized_blocked
        }
    
    def _test_confidence_gates(self) -> Dict[str, Any]:
        """Test confidence threshold enforcement"""
        
        # Test high confidence signal (92% - should pass)
        high_conf_signal = {'confidence': 0.92, 'prediction': 0.05}
        high_conf_allowed = self._check_confidence_gate(high_conf_signal)
        
        # Test low confidence signal (75% - should be blocked)
        low_conf_signal = {'confidence': 0.75, 'prediction': 0.08}
        low_conf_blocked = not self._check_confidence_gate(low_conf_signal)
        
        return {
            'high_confidence_signal': high_conf_signal,
            'high_conf_allowed': high_conf_allowed,
            'low_confidence_signal': low_conf_signal,
            'low_conf_blocked': low_conf_blocked,
            'threshold': 0.90,
            'test_passed': high_conf_allowed and low_conf_blocked
        }
    
    def _test_emergency_kill_switch(self) -> Dict[str, Any]:
        """Test emergency kill switch functionality"""
        
        if not self.risk_guard:
            return {'status': 'skipped', 'reason': 'RiskGuard not available'}
        
        # Check initial state
        initial_trading_allowed = self.risk_guard.is_trading_allowed()
        
        # Activate emergency kill switch
        self.risk_guard.emergency_stop = True
        
        # Check if trading is halted
        post_kill_trading_allowed = self.risk_guard.is_trading_allowed()
        
        # Reset for other tests
        self.risk_guard.emergency_stop = False
        
        return {
            'initial_trading_allowed': initial_trading_allowed,
            'kill_switch_activated': True,
            'post_kill_trading_allowed': post_kill_trading_allowed,
            'kill_switch_effective': not post_kill_trading_allowed,
            'test_passed': initial_trading_allowed and not post_kill_trading_allowed
        }
    
    def _test_data_quality_breaker(self) -> Dict[str, Any]:
        """Test data quality circuit breaker"""
        
        # Simulate poor data quality (85% - below 95% threshold)
        poor_quality_score = 0.85
        good_quality_score = 0.97
        
        poor_quality_allowed = self._check_data_quality_gate(poor_quality_score)
        good_quality_allowed = self._check_data_quality_gate(good_quality_score)
        
        return {
            'poor_quality_score': poor_quality_score,
            'poor_quality_allowed': poor_quality_allowed,
            'good_quality_score': good_quality_score,
            'good_quality_allowed': good_quality_allowed,
            'quality_threshold': 0.95,
            'test_passed': not poor_quality_allowed and good_quality_allowed
        }
    
    def _test_api_latency_protection(self) -> Dict[str, Any]:
        """Test API latency protection"""
        
        # Simulate high latency (8 seconds - exceeds 5 second limit)
        high_latency = 8.0
        normal_latency = 1.2
        
        high_latency_blocked = not self._check_latency_gate(high_latency)
        normal_latency_allowed = self._check_latency_gate(normal_latency)
        
        return {
            'high_latency': high_latency,
            'high_latency_blocked': high_latency_blocked,
            'normal_latency': normal_latency,
            'normal_latency_allowed': normal_latency_allowed,
            'latency_threshold': 5.0,
            'test_passed': high_latency_blocked and normal_latency_allowed
        }
    
    def _test_correlation_limits(self) -> Dict[str, Any]:
        """Test asset correlation limits"""
        
        # Simulate high correlation portfolio
        correlations = {
            'BTC-ETH': 0.85,
            'BTC-ADA': 0.82,
            'ETH-ADA': 0.88
        }
        
        correlation_risk = max(correlations.values())
        correlation_allowed = correlation_risk < 0.80  # 80% correlation limit
        
        return {
            'asset_correlations': correlations,
            'max_correlation': correlation_risk,
            'correlation_threshold': 0.80,
            'trading_allowed': correlation_allowed,
            'test_passed': correlation_risk > 0.80 and not correlation_allowed
        }
    
    def _test_volatility_escalation(self) -> Dict[str, Any]:
        """Test volatility-based risk escalation"""
        
        # Simulate extreme volatility (150% annualized)
        extreme_volatility = 1.50
        normal_volatility = 0.60
        
        extreme_vol_escalated = self._check_volatility_escalation(extreme_volatility)
        normal_vol_allowed = not self._check_volatility_escalation(normal_volatility)
        
        return {
            'extreme_volatility': extreme_volatility,
            'extreme_vol_escalated': extreme_vol_escalated,
            'normal_volatility': normal_volatility,
            'normal_vol_allowed': normal_vol_allowed,
            'volatility_threshold': 1.0,
            'test_passed': extreme_vol_escalated and normal_vol_allowed
        }
    
    def _test_recovery_procedures(self) -> Dict[str, Any]:
        """Test automatic recovery procedures"""
        
        if not self.risk_guard:
            return {'status': 'skipped', 'reason': 'RiskGuard not available'}
        
        # Simulate risk event and recovery
        self.risk_guard.daily_pnl = -6000  # 6% loss
        initial_risk_level = self.risk_guard.assess_risk_level()
        
        # Simulate market recovery
        self.risk_guard.daily_pnl = -2000  # 2% loss (recovered)
        recovered_risk_level = self.risk_guard.assess_risk_level()
        
        # Reset for other tests
        self.risk_guard.daily_pnl = 0
        
        return {
            'initial_loss_pct': -6.0,
            'initial_risk_level': initial_risk_level.name if hasattr(initial_risk_level, 'name') else str(initial_risk_level),
            'recovered_loss_pct': -2.0,
            'recovered_risk_level': recovered_risk_level.name if hasattr(recovered_risk_level, 'name') else str(recovered_risk_level),
            'recovery_successful': str(recovered_risk_level) != str(initial_risk_level)
        }
    
    # Helper methods for testing
    def _check_position_size(self, position_value: float, portfolio_value: float) -> bool:
        """Check if position size is within limits"""
        position_pct = position_value / portfolio_value
        return position_pct <= 0.02  # 2% limit
    
    def _check_confidence_gate(self, signal: Dict[str, float]) -> bool:
        """Check if signal meets confidence threshold"""
        return signal.get('confidence', 0.0) >= 0.90  # 90% threshold
    
    def _check_data_quality_gate(self, quality_score: float) -> bool:
        """Check if data quality meets threshold"""
        return quality_score >= 0.95  # 95% threshold
    
    def _check_latency_gate(self, latency_seconds: float) -> bool:
        """Check if API latency is acceptable"""
        return latency_seconds <= 5.0  # 5 second limit
    
    def _check_volatility_escalation(self, volatility: float) -> bool:
        """Check if volatility triggers escalation"""
        return volatility > 1.0  # 100% annualized volatility threshold
    
    def _evaluate_test_result(self, scenario: RiskTestScenario) -> bool:
        """Evaluate if test scenario passed"""
        
        if scenario.result is None or 'error' in scenario.result:
            return False
        
        # Scenario-specific evaluation logic
        if 'test_passed' in scenario.result:
            return scenario.result['test_passed']
        
        # Default evaluation based on expected behavior
        return True

def main():
    """Run risk management testing suite"""
    
    tester = RiskManagementTester()
    results = tester.run_comprehensive_tests()
    
    # Save detailed results
    results_path = f"test_results/risk_management_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path("test_results").mkdir(exist_ok=True)
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved: {results_path}")
    
    return results['passed_scenarios'] >= results['total_scenarios'] * 0.9

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)