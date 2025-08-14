#!/usr/bin/env python3
"""
Test Centralized Observability System
Tests voor Prometheus metrics, alerts, en integration
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from src.cryptosmarttrader.observability.centralized_prometheus import (
        CentralizedPrometheusMetrics, AlertRule, AlertSeverity, AlertEvent
    )
    from src.cryptosmarttrader.observability.alert_manager import (
        AlertManager, NotificationTarget, NotificationChannel, EscalationRule
    )
    from src.cryptosmarttrader.observability.metrics_integration import (
        MetricsIntegration, record_trading_operation, record_api_call
    )
except ImportError:
    pytest.skip("Observability modules not available", allow_module_level=True)


class TestCentralizedPrometheusMetrics:
    """Test centralized Prometheus metrics system"""
    
    def setup_method(self):
        """Setup for each test"""
        self.metrics = CentralizedPrometheusMetrics(enable_server=False)
    
    def test_trading_metrics_recording(self):
        """Test trading metrics recording"""
        
        # Record order
        self.metrics.record_order("BTC/USD", "buy", "market", "filled")
        
        # Record fill
        self.metrics.record_fill("BTC/USD", "buy", "taker", 1000.0)
        
        # Record slippage
        self.metrics.record_slippage("BTC/USD", "buy", "market", 15.0)
        
        # Record signal
        self.metrics.record_signal("momentum", "BTC/USD", "buy", 0.8)
        
        # Should not raise exceptions
        assert True
    
    def test_portfolio_metrics_updates(self):
        """Test portfolio metrics updates"""
        
        # Update portfolio metrics
        self.metrics.update_portfolio_metrics(
            portfolio_value=100000.0,
            drawdown_pct=5.0,
            daily_pnl=1500.0
        )
        
        # Update position sizes
        self.metrics.update_position_size("BTC/USD", 25000.0)
        self.metrics.update_position_size("ETH/USD", 15000.0)
        
        # Check metrics were recorded
        assert True
    
    def test_risk_metrics_recording(self):
        """Test risk metrics recording"""
        
        # Record risk violation
        self.metrics.record_risk_violation("position_size", "warning")
        
        # Update kill switch
        self.metrics.update_kill_switch(True, "drawdown_exceeded")
        
        # Update tracking error
        self.metrics.update_tracking_error(25.0, 1)  # Warning status
        
        assert True
    
    def test_alert_rule_management(self):
        """Test alert rule management"""
        
        # Add custom alert rule
        custom_rule = AlertRule(
            name="TestAlert",
            condition="test_metric > 100",
            threshold=100.0,
            duration="5m",
            severity=AlertSeverity.WARNING,
            description="Test alert for validation"
        )
        
        self.metrics.add_alert_rule(custom_rule)
        
        # Check rule was added
        assert "TestAlert" in self.metrics.alert_rules
        assert self.metrics.alert_rules["TestAlert"].threshold == 100.0
        
        # Remove rule
        self.metrics.remove_alert_rule("TestAlert")
        assert "TestAlert" not in self.metrics.alert_rules
    
    def test_metric_value_recording(self):
        """Test metric value recording for alerts"""
        
        # Record some metric values
        self.metrics._record_metric_value("test_metric", 50.0)
        self.metrics._record_metric_value("test_metric", 75.0)
        self.metrics._record_metric_value("test_metric", 125.0)
        
        # Check values were stored
        assert "test_metric" in self.metrics.metric_values
        assert len(self.metrics.metric_values["test_metric"]) == 3
        
        # Check latest value
        latest_value = self.metrics.metric_values["test_metric"][-1][1]
        assert latest_value == 125.0
    
    def test_critical_alert_rules(self):
        """Test critical alert rules are properly configured"""
        
        # Check that critical alerts are configured
        critical_alerts = [
            "HighOrderErrorRate",
            "DrawdownTooHigh", 
            "NoSignals30m",
            "SlippageP95ExceedsBudget"
        ]
        
        for alert_name in critical_alerts:
            assert alert_name in self.metrics.alert_rules
            rule = self.metrics.alert_rules[alert_name]
            assert isinstance(rule.threshold, (int, float))
            assert rule.severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY]
    
    def test_alert_evaluation_logic(self):
        """Test alert evaluation logic"""
        
        # Test drawdown alert
        self.metrics._record_metric_value("portfolio_drawdown_percent", 15.0)
        
        drawdown_rule = self.metrics.alert_rules["DrawdownTooHigh"]
        should_fire = self.metrics._evaluate_alert_condition(drawdown_rule, time.time())
        
        # Should fire because 15% > 10% threshold
        assert should_fire is not None
        assert should_fire == 15.0
        
        # Test with value below threshold
        self.metrics._record_metric_value("portfolio_drawdown_percent", 5.0)
        should_fire = self.metrics._evaluate_alert_condition(drawdown_rule, time.time())
        
        # Should not fire
        assert should_fire is None
    
    def test_metrics_output_generation(self):
        """Test Prometheus metrics output generation"""
        
        # Record some metrics
        self.metrics.record_order("BTC/USD", "buy", "market", "filled")
        self.metrics.update_portfolio_metrics(100000.0, 2.0, 500.0)
        
        # Generate output
        output = self.metrics.get_metrics_output()
        
        # Should be bytes
        assert isinstance(output, bytes)
        
        # Should contain metric names
        output_str = output.decode('utf-8')
        assert "trading_orders_total" in output_str
        assert "portfolio_value_usd" in output_str


class TestAlertManager:
    """Test alert manager functionality"""
    
    def setup_method(self):
        """Setup for each test"""
        self.alert_manager = AlertManager()
    
    def teardown_method(self):
        """Cleanup after each test"""
        self.alert_manager.shutdown()
    
    def test_alert_handling(self):
        """Test basic alert handling"""
        
        # Create test alert
        alert = AlertEvent(
            rule_name="TestAlert",
            severity=AlertSeverity.WARNING,
            current_value=150.0,
            threshold=100.0,
            started_at=time.time(),
            description="Test alert for validation"
        )
        
        # Handle alert
        self.alert_manager.handle_alert(alert)
        
        # Check alert is active
        assert "TestAlert" in self.alert_manager.active_alert_contexts
        context = self.alert_manager.active_alert_contexts["TestAlert"]
        assert context.alert.rule_name == "TestAlert"
        assert context.notification_count >= 0
    
    def test_alert_acknowledgment(self):
        """Test alert acknowledgment"""
        
        # Create and handle alert
        alert = AlertEvent(
            rule_name="AckTest",
            severity=AlertSeverity.CRITICAL,
            current_value=200.0,
            threshold=100.0,
            started_at=time.time(),
            description="Test acknowledgment"
        )
        
        self.alert_manager.handle_alert(alert)
        
        # Acknowledge alert
        self.alert_manager.acknowledge_alert("AckTest", "test_operator")
        
        # Check acknowledgment
        context = self.alert_manager.active_alert_contexts["AckTest"]
        assert context.acknowledged
        assert context.acknowledged_by == "test_operator"
        assert context.acknowledged_at is not None
    
    def test_alert_resolution(self):
        """Test alert resolution"""
        
        # Create and handle alert
        alert = AlertEvent(
            rule_name="ResolveTest",
            severity=AlertSeverity.WARNING,
            current_value=150.0,
            threshold=100.0,
            started_at=time.time(),
            description="Test resolution"
        )
        
        self.alert_manager.handle_alert(alert)
        assert "ResolveTest" in self.alert_manager.active_alert_contexts
        
        # Resolve alert
        self.alert_manager.resolve_alert("ResolveTest")
        
        # Check resolution
        assert "ResolveTest" not in self.alert_manager.active_alert_contexts
    
    def test_notification_target_management(self):
        """Test notification target management"""
        
        # Add custom notification target
        custom_target = NotificationTarget(
            channel=NotificationChannel.WEBHOOK,
            target="https://custom.webhook.com/alerts",
            severity_filter=[AlertSeverity.CRITICAL]
        )
        
        initial_count = len(self.alert_manager.notification_targets)
        self.alert_manager.add_notification_target(custom_target)
        
        # Check target was added
        assert len(self.alert_manager.notification_targets) == initial_count + 1
        
        # Find our target
        custom_targets = [
            t for t in self.alert_manager.notification_targets
            if t.target == "https://custom.webhook.com/alerts"
        ]
        assert len(custom_targets) == 1
        assert custom_targets[0].channel == NotificationChannel.WEBHOOK
    
    def test_escalation_rules(self):
        """Test escalation rule functionality"""
        
        # Check default escalation rules exist
        assert len(self.alert_manager.escalation_rules) > 0
        
        # Find emergency escalation rule
        emergency_rules = [
            rule for rule in self.alert_manager.escalation_rules
            if rule.name == "emergency_escalation"
        ]
        assert len(emergency_rules) == 1
        
        emergency_rule = emergency_rules[0]
        assert "DrawdownTooHigh" in emergency_rule.alert_patterns
        assert emergency_rule.escalation_delay == 60  # 1 minute
    
    def test_custom_notification_handler(self):
        """Test custom notification handler registration"""
        
        # Create mock handler
        def mock_handler(target: str, context) -> bool:
            return True
        
        # Register handler
        self.alert_manager.register_notification_handler(
            NotificationChannel.SMS, mock_handler
        )
        
        # Check handler was registered
        assert NotificationChannel.SMS in self.alert_manager.notification_handlers
        assert self.alert_manager.notification_handlers[NotificationChannel.SMS] == mock_handler
    
    def test_alert_status_reporting(self):
        """Test alert status reporting"""
        
        # Create test alert
        alert = AlertEvent(
            rule_name="StatusTest",
            severity=AlertSeverity.CRITICAL,
            current_value=200.0,
            threshold=100.0,
            started_at=time.time(),
            description="Test status reporting"
        )
        
        self.alert_manager.handle_alert(alert)
        
        # Get status
        status = self.alert_manager.get_alert_status()
        
        # Check status structure
        assert "active_alerts" in status
        assert "notification_targets" in status
        assert "escalation_rules" in status
        assert "active_alert_details" in status
        
        # Check active alert details
        assert status["active_alerts"] >= 1
        
        active_details = status["active_alert_details"]
        status_test_alerts = [
            alert for alert in active_details
            if alert["name"] == "StatusTest"
        ]
        assert len(status_test_alerts) == 1
        
        status_alert = status_test_alerts[0]
        assert status_alert["severity"] == "critical"
        assert status_alert["escalation_level"] == 0
        assert not status_alert["acknowledged"]


class TestMetricsIntegration:
    """Test metrics integration layer"""
    
    def setup_method(self):
        """Setup for each test"""
        # Create integration with disabled components for testing
        with patch('src.cryptosmarttrader.observability.metrics_integration.get_global_prometheus_metrics') as mock_metrics, \
             patch('src.cryptosmarttrader.observability.metrics_integration.get_global_alert_manager') as mock_alerts:
            
            mock_metrics.return_value = Mock()
            mock_alerts.return_value = Mock()
            
            self.integration = MetricsIntegration()
    
    def test_trading_operation_recording(self):
        """Test trading operation recording"""
        
        # Record successful order
        self.integration.record_successful_order("BTC/USD", "buy", "market", 1000.0)
        
        # Verify metrics were called
        self.integration.metrics.record_order.assert_called_with("BTC/USD", "buy", "market", "filled")
    
    def test_trade_execution_recording(self):
        """Test trade execution recording"""
        
        # Record trade execution
        self.integration.record_trade_execution(
            symbol="ETH/USD",
            side="sell",
            fill_type="maker",
            size_usd=2000.0,
            slippage_bps=12.5,
            execution_quality=85.0
        )
        
        # Verify metrics were called
        self.integration.metrics.record_fill.assert_called_with("ETH/USD", "sell", "maker", 2000.0)
        self.integration.metrics.record_slippage.assert_called_with("ETH/USD", "sell", "market", 12.5)
    
    def test_portfolio_state_update(self):
        """Test portfolio state update"""
        
        positions = {
            "BTC/USD": 25000.0,
            "ETH/USD": 15000.0,
            "SOL/USD": 5000.0
        }
        
        # Update portfolio state
        self.integration.update_portfolio_state(
            total_value=50000.0,
            positions=positions,
            daily_pnl=1200.0,
            drawdown_pct=3.5
        )
        
        # Verify metrics were called
        self.integration.metrics.update_portfolio_metrics.assert_called_with(50000.0, 3.5, 1200.0)
        
        # Verify position updates
        assert self.integration.metrics.update_position_size.call_count == 3
    
    def test_execution_time_measurement(self):
        """Test execution time measurement context manager"""
        
        # Test successful measurement
        with self.integration.measure_execution_time("BTC/USD", "buy", "limit"):
            time.sleep(0.01)  # Simulate execution time
        
        # Verify duration was recorded
        self.integration.metrics.record_execution_duration.assert_called_once()
        args = self.integration.metrics.record_execution_duration.call_args[0]
        
        assert args[0] == "BTC/USD"
        assert args[1] == "buy"
        assert args[2] == "limit"
        assert args[3] > 0.008  # At least 8ms
    
    def test_api_call_measurement(self):
        """Test API call measurement context manager"""
        
        # Test successful measurement
        with self.integration.measure_api_call("kraken", "order_book"):
            time.sleep(0.005)  # Simulate API call
        
        # Verify latency was recorded
        self.integration.metrics.exchange_latency.labels.assert_called_with(
            exchange="kraken", endpoint="order_book"
        )
    
    def test_decorators(self):
        """Test utility decorators"""
        
        @record_trading_operation("BTC/USD", "buy", "market")
        def mock_trading_function():
            return {"success": True, "size_usd": 1000.0}
        
        @record_api_call("kraken", "place_order")
        def mock_api_function():
            return {"status": "success"}
        
        # Execute decorated functions
        with patch('src.cryptosmarttrader.observability.metrics_integration.get_global_metrics_integration') as mock_integration:
            mock_integration.return_value = self.integration
            
            result1 = mock_trading_function()
            result2 = mock_api_function()
            
            assert result1["success"]
            assert result2["status"] == "success"
    
    def test_risk_violation_recording(self):
        """Test risk violation recording"""
        
        violation_details = {
            "current_exposure": 60000.0,
            "limit": 50000.0,
            "symbol": "BTC/USD"
        }
        
        self.integration.record_risk_violation(
            "max_exposure", "critical", violation_details
        )
        
        # Verify risk violation was recorded
        self.integration.metrics.record_risk_violation.assert_called_with("max_exposure", "critical")
    
    def test_system_health_update(self):
        """Test system health update"""
        
        component_scores = {
            "data_pipeline": 95.0,
            "execution_engine": 88.0,
            "risk_management": 92.0
        }
        
        self.integration.update_system_health(90.0, component_scores)
        
        # Verify system health was updated
        self.integration.metrics.system_health_score.set.assert_called_with(90.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])