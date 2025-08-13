"""Simulation testing system for forced error scenarios and guardrail validation."""

import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from ..core.structured_logger import get_logger
from ..core.risk_guard import RiskGuard, RiskLevel
from ..core.execution_policy import ExecutionPolicy, OrderRequest, OrderType, TimeInForce
from ..monitoring.prometheus_metrics import get_metrics
from ..monitoring.alert_rules import AlertManager, AlertSeverity


class FailureScenario(Enum):
    """Types of failure scenarios to simulate."""
    DATA_GAP = "data_gap"
    HIGH_SLIPPAGE = "high_slippage"
    ORDER_FAILURES = "order_failures"
    DRAWDOWN_SPIKE = "drawdown_spike"
    API_TIMEOUTS = "api_timeouts"
    RISK_LIMIT_BREACH = "risk_limit_breach"
    KILL_SWITCH_TEST = "kill_switch_test"
    MEMORY_LEAK = "memory_leak"
    CPU_SPIKE = "cpu_spike"
    PREDICTION_DEGRADATION = "prediction_degradation"


@dataclass
class SimulationConfig:
    """Configuration for simulation testing."""
    scenario: FailureScenario
    duration_minutes: int = 5
    intensity: float = 1.0  # 0.0 to 1.0
    portfolio_value: float = 100000.0
    position_size: float = 1000.0
    auto_recovery: bool = True
    recovery_delay_minutes: int = 2


@dataclass
class TestResult:
    """Result of a simulation test."""
    scenario: FailureScenario
    start_time: datetime
    end_time: datetime
    alerts_triggered: List[str]
    kill_switch_activated: bool
    auto_recovery_successful: bool
    max_slippage_observed: float
    risk_level_changes: List[str]
    metrics_collected: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None


class SimulationTester:
    """Comprehensive simulation testing for guardrails and observability."""
    
    def __init__(self, risk_guard: RiskGuard, execution_policy: ExecutionPolicy,
                 alert_manager: AlertManager):
        """Initialize simulation tester."""
        self.logger = get_logger("simulation_tester")
        self.risk_guard = risk_guard
        self.execution_policy = execution_policy
        self.alert_manager = alert_manager
        self.metrics = get_metrics()
        
        # Test state
        self.active_simulation: Optional[FailureScenario] = None
        self.simulation_start: Optional[datetime] = None
        self.original_values: Dict[str, Any] = {}
        self.test_results: List[TestResult] = []
        
        # Alert callback registration
        self.triggered_alerts: List[str] = []
        self.alert_manager.register_alert_callback(self._capture_alert)
        
        # Persistence
        self.data_path = Path("data/simulation_tests")
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Simulation tester initialized")
    
    def _capture_alert(self, alert) -> None:
        """Capture alerts triggered during simulation."""
        alert_info = f"{alert.rule_name}:{alert.severity.value}"
        self.triggered_alerts.append(alert_info)
        self.logger.info(f"Alert captured during simulation: {alert_info}")
    
    async def run_simulation(self, config: SimulationConfig) -> TestResult:
        """Run a comprehensive simulation test."""
        self.logger.info(f"Starting simulation: {config.scenario.value}",
                        duration=config.duration_minutes,
                        intensity=config.intensity)
        
        start_time = datetime.now()
        self.active_simulation = config.scenario
        self.simulation_start = start_time
        self.triggered_alerts = []
        
        # Store original values for recovery
        self._store_original_values(config)
        
        try:
            # Execute scenario-specific simulation
            await self._execute_scenario(config)
            
            # Monitor for specified duration
            await self._monitor_simulation(config)
            
            # Check results
            result = self._evaluate_results(config, start_time)
            
            # Auto-recovery if enabled
            if config.auto_recovery:
                await self._perform_recovery(config)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            return TestResult(
                scenario=config.scenario,
                start_time=start_time,
                end_time=datetime.now(),
                alerts_triggered=[],
                kill_switch_activated=False,
                auto_recovery_successful=False,
                max_slippage_observed=0.0,
                risk_level_changes=[],
                metrics_collected={},
                success=False,
                error_message=str(e)
            )
        finally:
            self.active_simulation = None
            self.simulation_start = None
    
    def _store_original_values(self, config: SimulationConfig) -> None:
        """Store original system values for recovery."""
        self.original_values = {
            'portfolio_value': config.portfolio_value,
            'risk_level': self.risk_guard.current_risk_level.value,
            'trading_mode': self.risk_guard.trading_mode.value,
            'kill_switch_active': self.risk_guard.kill_switch_active
        }
    
    async def _execute_scenario(self, config: SimulationConfig) -> None:
        """Execute specific failure scenario."""
        scenario_map = {
            FailureScenario.DATA_GAP: self._simulate_data_gap,
            FailureScenario.HIGH_SLIPPAGE: self._simulate_high_slippage,
            FailureScenario.ORDER_FAILURES: self._simulate_order_failures,
            FailureScenario.DRAWDOWN_SPIKE: self._simulate_drawdown_spike,
            FailureScenario.API_TIMEOUTS: self._simulate_api_timeouts,
            FailureScenario.RISK_LIMIT_BREACH: self._simulate_risk_limit_breach,
            FailureScenario.KILL_SWITCH_TEST: self._simulate_kill_switch_test,
            FailureScenario.MEMORY_LEAK: self._simulate_memory_leak,
            FailureScenario.CPU_SPIKE: self._simulate_cpu_spike,
            FailureScenario.PREDICTION_DEGRADATION: self._simulate_prediction_degradation
        }
        
        await scenario_map[config.scenario](config)
    
    async def _simulate_data_gap(self, config: SimulationConfig) -> None:
        """Simulate data feed gap."""
        self.logger.info("Simulating data gap scenario")
        
        # Stop updating data sources
        gap_minutes = config.intensity * 30  # Up to 30 minutes
        
        # Update metrics to show data gap
        self.metrics.update_data_source("kraken", 0.0)  # Zero quality score
        self.metrics.data_source_last_update.labels(source="kraken").set(
            time.time() - (gap_minutes * 60)
        )
        
        # Trigger data quality alert
        self.alert_manager.check_metric("cst_data_quality_score", 0.0)
        self.alert_manager.check_metric("cst_data_gap_minutes", gap_minutes)
    
    async def _simulate_high_slippage(self, config: SimulationConfig) -> None:
        """Simulate high slippage conditions."""
        self.logger.info("Simulating high slippage scenario")
        
        # Force high slippage in execution policy
        high_slippage = config.intensity * 2.0  # Up to 2% slippage
        
        # Create test order to trigger slippage
        test_order = OrderRequest(
            client_order_id="SIM_SLIPPAGE_TEST",
            symbol="BTC/USDT",
            side="buy",
            order_type=OrderType.MARKET,
            quantity=config.position_size / 50000,  # Assume BTC price
            confidence_score=0.8
        )
        
        # Record high slippage metric
        self.metrics.record_order("simulation", "BTC/USDT", "buy", "filled", 
                                 0.5, high_slippage)
        
        # Check against alert thresholds
        self.alert_manager.check_metric("cst_average_slippage_percent", high_slippage)
    
    async def _simulate_order_failures(self, config: SimulationConfig) -> None:
        """Simulate order execution failures."""
        self.logger.info("Simulating order failure scenario")
        
        failure_rate = config.intensity * 0.5  # Up to 50% failure rate
        
        # Simulate multiple failed orders
        for i in range(10):
            self.metrics.record_order("simulation", "ETH/USDT", "sell", "rejected", 
                                     5.0, 0.0)
            await asyncio.sleep(0.1)
        
        # Update error rate metrics
        self.metrics.update_error_rate("order_execution", failure_rate * 100)
        self.alert_manager.check_metric("cst_order_error_rate", failure_rate)
    
    async def _simulate_drawdown_spike(self, config: SimulationConfig) -> None:
        """Simulate sudden portfolio drawdown."""
        self.logger.info("Simulating drawdown spike scenario")
        
        # Calculate drawdown percentage
        drawdown_percent = config.intensity * 15.0  # Up to 15% drawdown
        
        # Update portfolio value to reflect drawdown
        current_value = config.portfolio_value * (1 - drawdown_percent / 100)
        self.risk_guard.update_portfolio_value(current_value)
        
        # Run risk check to trigger alerts
        risk_status = self.risk_guard.run_risk_check(current_value)
        
        # Update metrics
        self.metrics.update_portfolio_metrics(
            current_value, 
            -drawdown_percent,  # Daily PnL
            drawdown_percent    # Max drawdown
        )
        
        # Check alert thresholds
        self.alert_manager.check_metric("cst_max_drawdown_percent", drawdown_percent)
        self.alert_manager.check_metric("cst_daily_pnl_percent", -drawdown_percent)
    
    async def _simulate_api_timeouts(self, config: SimulationConfig) -> None:
        """Simulate API timeout conditions."""
        self.logger.info("Simulating API timeout scenario")
        
        timeout_duration = config.intensity * 20.0  # Up to 20 seconds
        
        # Record slow API responses
        self.metrics.record_api_request("kraken", "ticker", "timeout", timeout_duration)
        self.metrics.record_api_request("binance", "orderbook", "error", timeout_duration)
        
        # Check alert thresholds
        self.alert_manager.check_metric("cst_api_response_time_seconds", timeout_duration)
    
    async def _simulate_risk_limit_breach(self, config: SimulationConfig) -> None:
        """Simulate risk limit breaches."""
        self.logger.info("Simulating risk limit breach scenario")
        
        # Force risk limit violations
        large_position_percent = config.intensity * 5.0  # Up to 5% position size
        
        # Add large position
        self.risk_guard.update_position("BTC/USDT", 
                                       config.position_size, 
                                       config.position_size * large_position_percent / 100,
                                       50000, 50000)
        
        # Run risk check
        risk_status = self.risk_guard.run_risk_check(config.portfolio_value)
    
    async def _simulate_kill_switch_test(self, config: SimulationConfig) -> None:
        """Test kill switch activation and recovery."""
        self.logger.info("Testing kill switch activation")
        
        # Manually trigger kill switch
        self.risk_guard.trigger_kill_switch("Simulation test trigger", auto_trigger=False)
        
        # Update metrics
        self.metrics.set_kill_switch_status(True)
        self.alert_manager.check_metric("cst_kill_switch_active", 1.0)
    
    async def _simulate_memory_leak(self, config: SimulationConfig) -> None:
        """Simulate memory usage spike."""
        self.logger.info("Simulating memory usage spike")
        
        memory_percent = 70 + (config.intensity * 25)  # 70-95% memory usage
        memory_bytes = int(memory_percent * 8_000_000_000 / 100)  # Assume 8GB system
        
        self.metrics.update_system_resources("trading_system", memory_bytes, 50.0)
        self.alert_manager.check_metric("cst_memory_usage_percent", memory_percent)
    
    async def _simulate_cpu_spike(self, config: SimulationConfig) -> None:
        """Simulate CPU usage spike."""
        self.logger.info("Simulating CPU usage spike")
        
        cpu_percent = 80 + (config.intensity * 15)  # 80-95% CPU usage
        
        self.metrics.update_system_resources("trading_system", 4_000_000_000, cpu_percent)
        self.alert_manager.check_metric("cst_cpu_usage_percent", cpu_percent)
    
    async def _simulate_prediction_degradation(self, config: SimulationConfig) -> None:
        """Simulate prediction accuracy degradation."""
        self.logger.info("Simulating prediction accuracy degradation")
        
        accuracy = 80 - (config.intensity * 30)  # Down to 50% accuracy
        
        self.metrics.update_prediction_accuracy("random_forest", "1h", accuracy)
        self.alert_manager.check_metric("cst_prediction_accuracy_percent", accuracy)
    
    async def _monitor_simulation(self, config: SimulationConfig) -> None:
        """Monitor simulation for specified duration."""
        monitor_duration = config.duration_minutes * 60  # Convert to seconds
        end_time = time.time() + monitor_duration
        
        while time.time() < end_time:
            # Collect metrics during simulation
            if self.active_simulation:
                # Check if kill switch was activated
                if self.risk_guard.kill_switch_active:
                    self.logger.info("Kill switch activated during simulation")
                    break
                
                # Update agent status (simulate some agents going down)
                if config.scenario in [FailureScenario.MEMORY_LEAK, FailureScenario.CPU_SPIKE]:
                    self.metrics.set_agent_status("technical_analyzer", False)
                    self.alert_manager.check_metric("cst_agent_status", 0.0)
            
            await asyncio.sleep(5)  # Check every 5 seconds
    
    def _evaluate_results(self, config: SimulationConfig, start_time: datetime) -> TestResult:
        """Evaluate simulation test results."""
        end_time = datetime.now()
        
        # Calculate max slippage observed
        max_slippage = 0.0
        if config.scenario == FailureScenario.HIGH_SLIPPAGE:
            max_slippage = config.intensity * 2.0
        
        # Track risk level changes
        risk_changes = [self.risk_guard.current_risk_level.value]
        
        # Collect final metrics
        metrics_collected = {
            'portfolio_value': self.original_values.get('portfolio_value', 0),
            'final_risk_level': self.risk_guard.current_risk_level.value,
            'kill_switch_activated': self.risk_guard.kill_switch_active,
            'alerts_count': len(self.triggered_alerts),
            'duration_minutes': (end_time - start_time).total_seconds() / 60
        }
        
        # Determine success criteria
        success = self._evaluate_success_criteria(config)
        
        result = TestResult(
            scenario=config.scenario,
            start_time=start_time,
            end_time=end_time,
            alerts_triggered=self.triggered_alerts.copy(),
            kill_switch_activated=self.risk_guard.kill_switch_active,
            auto_recovery_successful=False,  # Will be updated in recovery
            max_slippage_observed=max_slippage,
            risk_level_changes=risk_changes,
            metrics_collected=metrics_collected,
            success=success
        )
        
        self.test_results.append(result)
        return result
    
    def _evaluate_success_criteria(self, config: SimulationConfig) -> bool:
        """Evaluate if simulation met success criteria."""
        # Basic success criteria: appropriate alerts were triggered
        expected_alerts = {
            FailureScenario.DATA_GAP: ["DataGapDetected", "LowDataQuality"],
            FailureScenario.HIGH_SLIPPAGE: ["HighSlippage"],
            FailureScenario.ORDER_FAILURES: ["HighOrderErrorRate"],
            FailureScenario.DRAWDOWN_SPIKE: ["DrawdownTooHigh", "DailyLossLimit"],
            FailureScenario.KILL_SWITCH_TEST: ["KillSwitchActivated"],
            FailureScenario.MEMORY_LEAK: ["HighMemoryUsage"],
            FailureScenario.CPU_SPIKE: ["HighCPUUsage"],
            FailureScenario.PREDICTION_DEGRADATION: ["LowPredictionAccuracy"]
        }
        
        expected = expected_alerts.get(config.scenario, [])
        triggered_rule_names = [alert.split(':')[0] for alert in self.triggered_alerts]
        
        # Check if at least one expected alert was triggered
        return any(rule in triggered_rule_names for rule in expected)
    
    async def _perform_recovery(self, config: SimulationConfig) -> bool:
        """Perform auto-recovery after simulation."""
        self.logger.info("Performing auto-recovery after simulation")
        
        # Wait for recovery delay
        if config.recovery_delay_minutes > 0:
            await asyncio.sleep(config.recovery_delay_minutes * 60)
        
        try:
            # Reset kill switch if activated
            if self.risk_guard.kill_switch_active:
                recovery_success = self.risk_guard.reset_kill_switch(manual_override=True)
                if not recovery_success:
                    self.logger.warning("Failed to reset kill switch during recovery")
                    return False
            
            # Restore original portfolio value
            if 'portfolio_value' in self.original_values:
                self.risk_guard.update_portfolio_value(self.original_values['portfolio_value'])
            
            # Reset metrics to normal values
            self.metrics.update_data_source("kraken", 1.0)  # Restore data quality
            self.metrics.set_kill_switch_status(False)
            
            # Clear positions added during simulation
            self.risk_guard.remove_position("BTC/USDT")
            
            self.logger.info("Auto-recovery completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Auto-recovery failed: {e}")
            return False
    
    def get_test_summary(self) -> Dict[str, Any]:
        """Get summary of all simulation tests."""
        if not self.test_results:
            return {"total_tests": 0, "success_rate": 0.0}
        
        successful_tests = sum(1 for result in self.test_results if result.success)
        
        return {
            "total_tests": len(self.test_results),
            "successful_tests": successful_tests,
            "success_rate": successful_tests / len(self.test_results) * 100,
            "scenarios_tested": list(set(r.scenario.value for r in self.test_results)),
            "kill_switch_activations": sum(1 for r in self.test_results if r.kill_switch_activated),
            "total_alerts_triggered": sum(len(r.alerts_triggered) for r in self.test_results)
        }
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive test suite covering all scenarios."""
        self.logger.info("Starting comprehensive simulation test suite")
        
        test_configs = [
            SimulationConfig(FailureScenario.DATA_GAP, duration_minutes=3, intensity=0.8),
            SimulationConfig(FailureScenario.HIGH_SLIPPAGE, duration_minutes=2, intensity=0.6),
            SimulationConfig(FailureScenario.DRAWDOWN_SPIKE, duration_minutes=1, intensity=0.7),
            SimulationConfig(FailureScenario.KILL_SWITCH_TEST, duration_minutes=1, intensity=1.0),
            SimulationConfig(FailureScenario.ORDER_FAILURES, duration_minutes=2, intensity=0.4),
        ]
        
        results = []
        for config in test_configs:
            try:
                result = await self.run_simulation(config)
                results.append(result)
                
                # Wait between tests
                await asyncio.sleep(30)
                
            except Exception as e:
                self.logger.error(f"Test suite failed on {config.scenario.value}: {e}")
        
        summary = self.get_test_summary()
        
        self.logger.info("Comprehensive test suite completed",
                        success_rate=summary['success_rate'],
                        total_tests=summary['total_tests'])
        
        return summary


def create_simulation_tester(risk_guard: RiskGuard, execution_policy: ExecutionPolicy,
                           alert_manager: AlertManager) -> SimulationTester:
    """Factory function to create SimulationTester instance."""
    return SimulationTester(risk_guard, execution_policy, alert_manager)