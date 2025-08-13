"""
Fase C Simulation Tester - Comprehensive testing of guardrails and observability
Tests ExecutionPolicy, RiskGuard, and Alert system under various breach scenarios.
"""

import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from ..execution.mandatory_order_pipeline import MandatoryOrderPipeline, OrderRequest, OrderStatus
from ..execution.execution_policy import OrderType, TimeInForce
from ..risk.centralized_risk_guard_enforcer import CentralizedRiskGuardEnforcer, TradingOperation
from ..observability.alert_rules import AlertManager
from ..observability.metrics_collector import MetricsCollector, AlertSeverity
from ..core.structured_logger import get_logger


@dataclass
class SimulationScenario:
    """Simulation scenario configuration."""
    name: str
    description: str
    duration_minutes: int
    portfolio_start_value: float
    daily_loss_target_percent: float
    drawdown_target_percent: float
    order_frequency_seconds: int
    slippage_stress_factor: float = 1.0
    api_error_rate: float = 0.0
    data_gap_minutes: int = 0


@dataclass
class SimulationResult:
    """Simulation test result."""
    scenario: SimulationScenario
    total_orders_attempted: int
    orders_executed: int
    orders_blocked_policy: int
    orders_blocked_risk: int
    alerts_triggered: Dict[str, int]
    kill_switch_triggers: int
    p95_slippage_bps: float
    max_daily_loss_percent: float
    max_drawdown_percent: float
    test_passed: bool
    failure_reasons: List[str]
    execution_time_seconds: float
    timestamp: datetime


class FaseCSimulationTester:
    """
    Comprehensive simulation tester for Fase C guardrails and observability.
    
    Tests:
    1. ExecutionPolicy slippage budget enforcement
    2. RiskGuard kill-switch on day-loss/drawdown breaches
    3. Alert system (HighErrorRate, DrawdownTooHigh, NoSignals 30m)
    4. P95 slippage ≤ budget validation
    5. Block/alert behavior under stress conditions
    
    Validates that system properly blocks unsafe operations and triggers appropriate alerts.
    """
    
    def __init__(self):
        self.logger = get_logger("fase_c_simulation_tester")
        
        # Initialize test components
        self.order_pipeline = MandatoryOrderPipeline()
        self.risk_enforcer = CentralizedRiskGuardEnforcer()
        self.metrics_collector = MetricsCollector("simulation_tester")
        self.alert_manager = AlertManager(self.metrics_collector)
        
        # Test scenarios
        self.test_scenarios = self._create_test_scenarios()
        
        # Results tracking
        self.simulation_results: List[SimulationResult] = []
        self.overall_test_results = {
            'scenarios_passed': 0,
            'scenarios_failed': 0,
            'critical_failures': [],
            'test_start_time': None,
            'test_end_time': None
        }
        
        self.logger.info("FaseCSimulationTester initialized - Ready to test guardrails")
    
    def _create_test_scenarios(self) -> List[SimulationScenario]:
        """Create comprehensive test scenarios."""
        scenarios = [
            # Scenario 1: Normal operations - should pass
            SimulationScenario(
                name="normal_operations",
                description="Normal trading operations with good conditions",
                duration_minutes=5,
                portfolio_start_value=1000000.0,
                daily_loss_target_percent=1.0,  # Stay within limits
                drawdown_target_percent=3.0,    # Stay within limits
                order_frequency_seconds=10,
                slippage_stress_factor=0.5,     # Low slippage
                api_error_rate=0.02,            # Low error rate
                data_gap_minutes=0
            ),
            
            # Scenario 2: High slippage stress - should trigger slippage alerts
            SimulationScenario(
                name="high_slippage_stress",
                description="High slippage conditions to test budget enforcement",
                duration_minutes=3,
                portfolio_start_value=1000000.0,
                daily_loss_target_percent=2.0,
                drawdown_target_percent=4.0,
                order_frequency_seconds=5,
                slippage_stress_factor=3.0,     # High slippage
                api_error_rate=0.05,
                data_gap_minutes=0
            ),
            
            # Scenario 3: Daily loss breach - should trigger kill-switch
            SimulationScenario(
                name="daily_loss_breach",
                description="Daily loss exceeds limits - should trigger kill-switch",
                duration_minutes=4,
                portfolio_start_value=1000000.0,
                daily_loss_target_percent=6.0,  # Exceeds 5% limit
                drawdown_target_percent=3.0,
                order_frequency_seconds=8,
                slippage_stress_factor=1.0,
                api_error_rate=0.03,
                data_gap_minutes=0
            ),
            
            # Scenario 4: Drawdown breach - should trigger kill-switch
            SimulationScenario(
                name="drawdown_breach", 
                description="Drawdown exceeds limits - should trigger kill-switch",
                duration_minutes=4,
                portfolio_start_value=1000000.0,
                daily_loss_target_percent=3.0,
                drawdown_target_percent=12.0,   # Exceeds 10% limit
                order_frequency_seconds=6,
                slippage_stress_factor=1.0,
                api_error_rate=0.03,
                data_gap_minutes=0
            ),
            
            # Scenario 5: High API error rate - should trigger error alerts
            SimulationScenario(
                name="high_api_errors",
                description="High API error rate to test error alerting",
                duration_minutes=3,
                portfolio_start_value=1000000.0,
                daily_loss_target_percent=2.0,
                drawdown_target_percent=4.0,
                order_frequency_seconds=5,
                slippage_stress_factor=1.0,
                api_error_rate=0.25,            # High error rate
                data_gap_minutes=0
            ),
            
            # Scenario 6: Data gap - should trigger NoSignals alert
            SimulationScenario(
                name="data_gap_scenario",
                description="Data gap to test NoSignals alert",
                duration_minutes=6,
                portfolio_start_value=1000000.0,
                daily_loss_target_percent=1.0,
                drawdown_target_percent=2.0,
                order_frequency_seconds=10,
                slippage_stress_factor=1.0,
                api_error_rate=0.02,
                data_gap_minutes=35             # Exceeds 30 min threshold
            )
        ]
        
        return scenarios
    
    async def run_all_simulations(self) -> Dict[str, Any]:
        """Run all simulation scenarios and return comprehensive results."""
        self.overall_test_results['test_start_time'] = datetime.now()
        self.logger.info("Starting Fase C comprehensive simulation testing")
        
        for scenario in self.test_scenarios:
            self.logger.info(f"Running scenario: {scenario.name}")
            
            try:
                result = await self.run_scenario_simulation(scenario)
                self.simulation_results.append(result)
                
                if result.test_passed:
                    self.overall_test_results['scenarios_passed'] += 1
                    self.logger.info(f"✅ Scenario {scenario.name} PASSED")
                else:
                    self.overall_test_results['scenarios_failed'] += 1
                    self.overall_test_results['critical_failures'].extend(result.failure_reasons)
                    self.logger.warning(f"❌ Scenario {scenario.name} FAILED: {result.failure_reasons}")
                
                # Brief pause between scenarios
                await asyncio.sleep(2)
                
            except Exception as e:
                self.overall_test_results['scenarios_failed'] += 1
                self.overall_test_results['critical_failures'].append(f"Scenario {scenario.name} crashed: {str(e)}")
                self.logger.error(f"Scenario {scenario.name} failed with exception: {e}")
        
        self.overall_test_results['test_end_time'] = datetime.now()
        
        # Generate final report
        return self._generate_final_report()
    
    async def run_scenario_simulation(self, scenario: SimulationScenario) -> SimulationResult:
        """Run single scenario simulation."""
        start_time = time.time()
        
        # Reset systems for clean test
        await self._reset_test_environment()
        
        # Initialize scenario state
        current_portfolio_value = scenario.portfolio_start_value
        max_portfolio_value = scenario.portfolio_start_value
        daily_start_value = scenario.portfolio_start_value
        
        orders_attempted = 0
        orders_executed = 0
        orders_blocked_policy = 0
        orders_blocked_risk = 0
        slippage_history = []
        
        # Simulate data gap if required
        if scenario.data_gap_minutes > 0:
            self.metrics_collector.record_signal_received(
                "test", "data_gap_start", 
                timestamp=datetime.now() - timedelta(minutes=scenario.data_gap_minutes)
            )
        
        # Run simulation for specified duration
        simulation_end_time = datetime.now() + timedelta(minutes=scenario.duration_minutes)
        
        while datetime.now() < simulation_end_time:
            # Generate test order
            order_request = self._generate_test_order(scenario)
            orders_attempted += 1
            
            # Simulate API errors
            if random.random() < scenario.api_error_rate:
                self.metrics_collector.record_operation_error("api_call", "timeout_error", "exchange")
                await asyncio.sleep(scenario.order_frequency_seconds)
                continue
            
            try:
                # Execute through mandatory pipeline
                result = await self.order_pipeline.execute_order(order_request)
                
                if result.status == OrderStatus.FILLED:
                    orders_executed += 1
                    
                    # Apply P&L impact
                    pnl_impact = self._calculate_pnl_impact(scenario, result)
                    current_portfolio_value += pnl_impact
                    
                    # Track slippage
                    slippage_bps = result.slippage_percent * 100 * scenario.slippage_stress_factor
                    slippage_history.append(slippage_bps)
                    
                    # Update max portfolio value for drawdown calculation
                    max_portfolio_value = max(max_portfolio_value, current_portfolio_value)
                    
                    self.metrics_collector.record_order_filled(
                        result.symbol if hasattr(result, 'symbol') else 'BTC/USD',
                        result.filled_quantity,
                        result.slippage_percent
                    )
                
                elif "ExecutionPolicy" in (result.error_message or ""):
                    orders_blocked_policy += 1
                elif "RiskGuard" in (result.error_message or ""):
                    orders_blocked_risk += 1
                
            except Exception as e:
                self.logger.error(f"Order execution failed: {e}")
                orders_blocked_risk += 1  # Count as risk block
            
            # Record current portfolio metrics
            daily_loss_percent = (current_portfolio_value - daily_start_value) / daily_start_value * 100
            drawdown_percent = (max_portfolio_value - current_portfolio_value) / max_portfolio_value * 100
            
            # Update metrics for alert evaluation
            self._update_simulation_metrics(scenario, current_portfolio_value, daily_loss_percent, drawdown_percent)
            
            # Evaluate alerts
            metrics_summary = self.metrics_collector.get_metrics_summary()
            self.alert_manager.evaluate_rules(metrics_summary)
            
            await asyncio.sleep(scenario.order_frequency_seconds)
        
        # Collect final results
        execution_time = time.time() - start_time
        alerts_triggered = self._count_triggered_alerts()
        
        # Calculate metrics
        p95_slippage = np.percentile(slippage_history, 95) if len(slippage_history) >= 5 else 0.0
        final_daily_loss = (current_portfolio_value - daily_start_value) / daily_start_value * 100
        final_drawdown = (max_portfolio_value - current_portfolio_value) / max_portfolio_value * 100
        
        # Assess test result
        test_passed, failure_reasons = self._assess_scenario_result(
            scenario, alerts_triggered, p95_slippage, final_daily_loss, final_drawdown, 
            orders_blocked_policy, orders_blocked_risk
        )
        
        return SimulationResult(
            scenario=scenario,
            total_orders_attempted=orders_attempted,
            orders_executed=orders_executed,
            orders_blocked_policy=orders_blocked_policy,
            orders_blocked_risk=orders_blocked_risk,
            alerts_triggered=alerts_triggered,
            kill_switch_triggers=self.risk_enforcer.kill_switch_triggers,
            p95_slippage_bps=p95_slippage,
            max_daily_loss_percent=final_daily_loss,
            max_drawdown_percent=final_drawdown,
            test_passed=test_passed,
            failure_reasons=failure_reasons,
            execution_time_seconds=execution_time,
            timestamp=datetime.now()
        )
    
    def _generate_test_order(self, scenario: SimulationScenario) -> OrderRequest:
        """Generate test order for scenario."""
        symbols = ['BTC/USD', 'ETH/USD', 'ADA/USD', 'SOL/USD']
        sides = ['buy', 'sell']
        
        return OrderRequest(
            client_order_id="",  # Will be generated by pipeline
            symbol=random.choice(symbols),
            side=random.choice(sides),
            order_type=OrderType.MARKET,
            quantity=random.uniform(0.1, 2.0),
            price=None,  # Market order
            time_in_force=TimeInForce.IOC,
            confidence_score=random.uniform(0.6, 0.9),
            strategy_id="simulation_test",
            max_slippage_percent=0.003  # 30 bps budget
        )
    
    def _calculate_pnl_impact(self, scenario: SimulationScenario, result) -> float:
        """Calculate P&L impact based on scenario targets."""
        # Calculate impact to reach target loss/drawdown levels
        target_daily_loss = scenario.daily_loss_target_percent / 100
        target_drawdown = scenario.drawdown_target_percent / 100
        
        # Use larger of the two targets as basis for loss simulation
        target_loss_factor = max(abs(target_daily_loss), abs(target_drawdown))
        
        # Apply gradual loss over the simulation period
        loss_per_order = -scenario.portfolio_start_value * target_loss_factor / max(scenario.duration_minutes * 60 // scenario.order_frequency_seconds, 1)
        
        # Add some randomness
        return loss_per_order * random.uniform(0.5, 1.5)
    
    def _update_simulation_metrics(self, scenario: SimulationScenario, portfolio_value: float, daily_loss_percent: float, drawdown_percent: float):
        """Update metrics for alert evaluation."""
        # Record portfolio metrics
        self.metrics_collector.record_portfolio_value(portfolio_value)
        self.metrics_collector.record_daily_pnl(daily_loss_percent)
        self.metrics_collector.record_drawdown(drawdown_percent)
        
        # Record system metrics
        self.metrics_collector.record_cpu_usage(random.uniform(20, 80))
        self.metrics_collector.record_memory_usage(random.uniform(40, 90))
    
    def _count_triggered_alerts(self) -> Dict[str, int]:
        """Count alerts triggered during simulation."""
        alert_counts = {}
        
        for rule_name, alert_state in self.alert_manager.alert_states.items():
            if alert_state.trigger_count > 0:
                alert_counts[rule_name] = alert_state.trigger_count
        
        return alert_counts
    
    def _assess_scenario_result(self, scenario: SimulationScenario, alerts_triggered: Dict[str, int], 
                              p95_slippage: float, daily_loss: float, drawdown: float,
                              blocked_policy: int, blocked_risk: int) -> Tuple[bool, List[str]]:
        """Assess if scenario test passed."""
        failure_reasons = []
        
        # Check scenario-specific expectations
        if scenario.name == "normal_operations":
            # Should have minimal blocks and alerts
            if blocked_policy + blocked_risk > 2:
                failure_reasons.append(f"Too many blocks ({blocked_policy + blocked_risk}) for normal operations")
            
            if len(alerts_triggered) > 2:
                failure_reasons.append(f"Too many alerts ({len(alerts_triggered)}) for normal operations")
        
        elif scenario.name == "high_slippage_stress":
            # Should trigger slippage alerts
            if "HighSlippage" not in alerts_triggered:
                failure_reasons.append("HighSlippage alert not triggered despite high slippage conditions")
            
            if p95_slippage <= 50:  # Should exceed budget
                failure_reasons.append(f"P95 slippage {p95_slippage:.1f} bps too low for stress test")
        
        elif scenario.name in ["daily_loss_breach", "drawdown_breach"]:
            # Should trigger kill-switch
            if self.risk_enforcer.kill_switch_triggers == 0:
                failure_reasons.append("Kill-switch not triggered despite breach conditions")
            
            if "DrawdownTooHigh" not in alerts_triggered and scenario.drawdown_target_percent > 5:
                failure_reasons.append("DrawdownTooHigh alert not triggered")
        
        elif scenario.name == "high_api_errors":
            # Should trigger API error alerts
            if "HighAPIErrorRate" not in alerts_triggered:
                failure_reasons.append("HighAPIErrorRate alert not triggered despite high error rate")
        
        elif scenario.name == "data_gap_scenario":
            # Should trigger NoSignals alert
            if "NoSignals" not in alerts_triggered:
                failure_reasons.append("NoSignals alert not triggered despite data gap")
        
        # General validations
        if p95_slippage > 100:  # 100 bps emergency threshold
            expected_blocks = blocked_policy > 0
            if not expected_blocks:
                failure_reasons.append(f"No policy blocks despite excessive P95 slippage {p95_slippage:.1f} bps")
        
        return len(failure_reasons) == 0, failure_reasons
    
    async def _reset_test_environment(self):
        """Reset test environment for clean scenario start."""
        # Reset order pipeline
        self.order_pipeline.processed_orders.clear()
        self.order_pipeline.slippage_history.clear()
        
        # Reset risk enforcer (but not kill-switch for continuity)
        self.risk_enforcer.pending_operations.clear()
        
        # Reset alerts
        self.alert_manager.alert_states.clear()
        
    def _generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final test report."""
        total_scenarios = len(self.test_scenarios)
        passed_scenarios = self.overall_test_results['scenarios_passed']
        failed_scenarios = self.overall_test_results['scenarios_failed']
        
        # Calculate overall pass rate
        pass_rate = (passed_scenarios / total_scenarios * 100) if total_scenarios > 0 else 0
        
        # Aggregate statistics
        total_orders = sum(r.total_orders_attempted for r in self.simulation_results)
        total_executed = sum(r.orders_executed for r in self.simulation_results)
        total_blocked = sum(r.orders_blocked_policy + r.orders_blocked_risk for r in self.simulation_results)
        
        all_alerts = {}
        for result in self.simulation_results:
            for alert_name, count in result.alerts_triggered.items():
                all_alerts[alert_name] = all_alerts.get(alert_name, 0) + count
        
        # Determine overall test status
        overall_passed = pass_rate >= 85  # Require 85% pass rate
        
        report = {
            'test_summary': {
                'overall_passed': overall_passed,
                'pass_rate_percent': pass_rate,
                'scenarios_total': total_scenarios,
                'scenarios_passed': passed_scenarios,
                'scenarios_failed': failed_scenarios,
                'critical_failures': self.overall_test_results['critical_failures'],
                'test_duration_seconds': (self.overall_test_results['test_end_time'] - self.overall_test_results['test_start_time']).total_seconds()
            },
            'execution_statistics': {
                'total_orders_attempted': total_orders,
                'total_orders_executed': total_executed,
                'total_orders_blocked': total_blocked,
                'execution_rate_percent': (total_executed / max(total_orders, 1)) * 100,
                'block_rate_percent': (total_blocked / max(total_orders, 1)) * 100
            },
            'alert_statistics': {
                'unique_alerts_triggered': len(all_alerts),
                'total_alert_instances': sum(all_alerts.values()),
                'alerts_by_type': all_alerts
            },
            'guardrail_validation': {
                'execution_policy_active': any(r.orders_blocked_policy > 0 for r in self.simulation_results),
                'risk_guard_active': any(r.orders_blocked_risk > 0 for r in self.simulation_results),
                'kill_switch_functional': any(r.kill_switch_triggers > 0 for r in self.simulation_results),
                'p95_slippage_monitoring': any(r.p95_slippage_bps > 30 for r in self.simulation_results),
                'alert_system_responsive': len(all_alerts) > 0
            },
            'scenario_details': [
                {
                    'name': r.scenario.name,
                    'passed': r.test_passed,
                    'orders_attempted': r.total_orders_attempted,
                    'orders_executed': r.orders_executed,
                    'blocks_policy': r.orders_blocked_policy,
                    'blocks_risk': r.orders_blocked_risk,
                    'alerts_triggered': len(r.alerts_triggered),
                    'p95_slippage_bps': r.p95_slippage_bps,
                    'max_daily_loss_percent': r.max_daily_loss_percent,
                    'failure_reasons': r.failure_reasons
                }
                for r in self.simulation_results
            ],
            'recommendations': self._generate_recommendations(overall_passed),
            'timestamp': datetime.now()
        }
        
        return report
    
    def _generate_recommendations(self, overall_passed: bool) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if not overall_passed:
            recommendations.append("CRITICAL: Fix failing test scenarios before production deployment")
        
        # Check specific failure patterns
        critical_failures = self.overall_test_results['critical_failures']
        
        if any("kill-switch" in failure.lower() for failure in critical_failures):
            recommendations.append("Review kill-switch trigger logic - may be too sensitive or not sensitive enough")
        
        if any("slippage" in failure.lower() for failure in critical_failures):
            recommendations.append("Adjust slippage budget parameters or improve slippage estimation")
        
        if any("alert" in failure.lower() for failure in critical_failures):
            recommendations.append("Review alert threshold configuration and rule logic")
        
        # Check execution rates
        avg_execution_rate = np.mean([
            r.orders_executed / max(r.total_orders_attempted, 1) * 100 
            for r in self.simulation_results
        ])
        
        if avg_execution_rate < 60:
            recommendations.append(f"Low execution rate ({avg_execution_rate:.1f}%) - review blocking thresholds")
        elif avg_execution_rate > 95:
            recommendations.append(f"Very high execution rate ({avg_execution_rate:.1f}%) - verify guardrails are active")
        
        if overall_passed:
            recommendations.append("✅ All guardrails functioning correctly - ready for production deployment")
            recommendations.append("Continue monitoring alert thresholds and adjust based on live trading patterns")
        
        return recommendations


# Test runner function
async def run_fase_c_testing():
    """Convenience function to run Fase C testing."""
    tester = FaseCSimulationTester()
    results = await tester.run_all_simulations()
    return results


if __name__ == "__main__":
    # Direct test execution
    results = asyncio.run(run_fase_c_testing())
    print(f"\n🎯 FASE C SIMULATION TEST RESULTS:")
    print(f"Overall Passed: {'✅ YES' if results['test_summary']['overall_passed'] else '❌ NO'}")
    print(f"Pass Rate: {results['test_summary']['pass_rate_percent']:.1f}%")
    print(f"Scenarios: {results['test_summary']['scenarios_passed']}/{results['test_summary']['scenarios_total']}")
    print(f"Orders Executed: {results['execution_statistics']['total_orders_executed']}/{results['execution_statistics']['total_orders_attempted']}")
    print(f"Alerts Triggered: {results['alert_statistics']['total_alert_instances']}")
    
    if not results['test_summary']['overall_passed']:
        print(f"\n❌ FAILURES:")
        for failure in results['test_summary']['critical_failures']:
            print(f"  • {failure}")