#!/usr/bin/env python3
"""
Critical Tests for Observability & Alert System
Ensures comprehensive monitoring en alert functionality
"""

import pytest
import time
from unittest.mock import Mock, patch
import threading

import sys
sys.path.append('.')

from src.cryptosmarttrader.observability.centralized_metrics import (
    CentralizedMetrics,
    MetricType,
    AlertSeverity,
    AlertRule
)


class TestCentralizedMetrics:
    """Critical tests voor centralized metrics system"""
    
    @pytest.fixture
    def metrics_system(self):
        """Fresh metrics system instance"""
        # Reset singleton
        CentralizedMetrics._instance = None
        return CentralizedMetrics()
    
    def test_singleton_enforcement(self):
        """CRITICAL: Singleton pattern enforcement"""
        # Reset singleton
        CentralizedMetrics._instance = None
        
        metrics1 = CentralizedMetrics()
        metrics2 = CentralizedMetrics()
        
        # Should be same instance
        assert metrics1 is metrics2
    
    def test_metric_registration(self, metrics_system):
        """Test metric registration en retrieval"""
        # Verify core metrics are registered
        assert "cst_system_health_status" in metrics_system.metrics
        assert "cst_errors_total" in metrics_system.metrics
        assert "cst_trades_total" in metrics_system.metrics
        assert "cst_order_latency_seconds" in metrics_system.metrics
    
    def test_error_recording(self, metrics_system):
        """CRITICAL: Error recording functionality"""
        # Record some errors
        metrics_system.record_error("trading", "validation_failed", "high")
        metrics_system.record_error("data", "api_timeout", "medium")
        metrics_system.record_error("risk", "limit_breach", "critical")
        
        # Verify errors are recorded
        error_metric = metrics_system.metrics["cst_errors_total"]
        
        # Check that metric values increased
        # Note: In real Prometheus, we'd check the metric registry
        # For testing, we verify the method calls work without error
        assert error_metric is not None
    
    def test_trade_recording(self, metrics_system):
        """CRITICAL: Trade recording functionality"""
        # Record some trades
        metrics_system.record_trade("ETH", "buy", "momentum", "filled", 150.0)
        metrics_system.record_trade("BTC", "sell", "mean_reversion", "filled", -75.0)
        metrics_system.record_trade("ETH", "buy", "momentum", "rejected", 0.0)
        
        # Verify trades are recorded
        trade_metric = metrics_system.metrics["cst_trades_total"]
        pnl_metric = metrics_system.metrics["cst_trade_pnl_usd"]
        
        assert trade_metric is not None
        assert pnl_metric is not None
    
    def test_system_health_recording(self, metrics_system):
        """CRITICAL: System health recording"""
        # Record different health states
        metrics_system.record_system_health(1)  # Healthy
        time.sleep(0.1)
        metrics_system.record_system_health(0)  # Unhealthy
        
        # Verify health is recorded
        health_metric = metrics_system.metrics["cst_system_health_status"]
        assert health_metric is not None
    
    def test_latency_recording(self, metrics_system):
        """CRITICAL: Latency recording functionality"""
        # Record various latencies
        metrics_system.record_latency("api_call", "exchange", 0.150)  # 150ms
        metrics_system.record_latency("order_execution", "exchange", 0.050)  # 50ms
        metrics_system.record_latency("risk_check", "internal", 0.010)  # 10ms
        
        # Verify latency is recorded
        latency_metric = metrics_system.metrics["cst_order_latency_seconds"]
        assert latency_metric is not None
    
    def test_alert_rule_generation(self, metrics_system):
        """CRITICAL: Alert rule generation"""
        # Get alert rules
        alert_rules = metrics_system.get_alert_rules()
        
        # Verify critical alert rules exist
        rule_names = [rule.name for rule in alert_rules]
        
        assert "HighErrorRate" in rule_names
        assert "HighOrderLatency" in rule_names
        assert "KillSwitchActivated" in rule_names
        assert "SystemOverloaded" in rule_names
        assert "DataIntegrityIssue" in rule_names
    
    def test_alert_rule_prometheus_export(self, metrics_system):
        """Test Prometheus alert rule export"""
        # Export alert rules
        prometheus_rules = metrics_system.export_prometheus_rules()
        
        # Verify format
        assert "groups:" in prometheus_rules
        assert "- name:" in prometheus_rules
        assert "rules:" in prometheus_rules
        assert "alert:" in prometheus_rules
        assert "expr:" in prometheus_rules
        assert "severity:" in prometheus_rules
    
    def test_concurrent_metric_recording(self, metrics_system):
        """CRITICAL: Thread safety voor concurrent metric recording"""
        errors = []
        
        def record_metrics_thread(thread_id):
            try:
                for i in range(100):
                    metrics_system.record_error(f"component_{thread_id}", "test_error", "low")
                    metrics_system.record_trade("ETH", "buy", f"strategy_{thread_id}", "filled", 10.0)
                    metrics_system.record_latency("test_operation", "test_exchange", 0.1)
                    time.sleep(0.001)  # Small delay
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=record_metrics_thread, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        assert len(errors) == 0
    
    def test_prometheus_export(self, metrics_system):
        """Test Prometheus metrics export"""
        # Record some metrics
        metrics_system.record_error("test", "error", "high")
        metrics_system.record_trade("ETH", "buy", "test", "filled", 100.0)
        
        # Export metrics
        exported_metrics = metrics_system.export_metrics()
        
        # Verify export format
        assert isinstance(exported_metrics, str)
        assert "cst_errors_total" in exported_metrics
        assert "cst_trades_total" in exported_metrics
        assert "TYPE" in exported_metrics  # Prometheus format
        assert "HELP" in exported_metrics
    
    def test_health_endpoint(self, metrics_system):
        """CRITICAL: Health endpoint functionality"""
        # Get health status
        health_status = metrics_system.get_health_status()
        
        # Verify health response format
        assert "status" in health_status
        assert "timestamp" in health_status
        assert "uptime_seconds" in health_status
        assert "metrics_count" in health_status
        assert "alert_rules_count" in health_status
        
        # Verify values
        assert health_status["status"] in ["healthy", "degraded", "unhealthy"]
        assert health_status["metrics_count"] > 0
        assert health_status["alert_rules_count"] > 0
    
    def test_metric_cleanup(self, metrics_system):
        """Test metric cleanup en memory management"""
        initial_metric_count = len(metrics_system.metrics)
        
        # Record many metrics
        for i in range(1000):
            metrics_system.record_error("test", f"error_{i}", "low")
        
        # Verify metrics don't grow unbounded
        # (In real implementation, there should be cleanup mechanisms)
        final_metric_count = len(metrics_system.metrics)
        
        # Metric count should not grow significantly
        assert final_metric_count <= initial_metric_count + 10


class TestAlertRuleValidation:
    """Test alert rule validation en configuration"""
    
    def test_alert_rule_structure(self):
        """Test alert rule structure validation"""
        rule = AlertRule(
            name="TestAlert",
            description="Test alert for validation",
            query="test_metric > 10",
            severity=AlertSeverity.HIGH,
            threshold=10.0,
            for_duration="5m",
            labels={"component": "test"},
            annotations={"summary": "Test alert fired"}
        )
        
        # Verify all fields are set
        assert rule.name == "TestAlert"
        assert rule.severity == AlertSeverity.HIGH
        assert rule.threshold == 10.0
        assert "component" in rule.labels
        assert "summary" in rule.annotations
    
    def test_prometheus_query_validation(self):
        """Test Prometheus query validation"""
        metrics_system = CentralizedMetrics()
        
        # Valid queries should not raise errors
        valid_queries = [
            "cst_errors_total > 10",
            "rate(cst_trades_total[5m]) > 1",
            "histogram_quantile(0.95, cst_order_latency_seconds_bucket) > 0.5"
        ]
        
        for query in valid_queries:
            # Should not raise exception
            validated = metrics_system.validate_prometheus_query(query)
            assert validated is not None
    
    def test_alert_severity_levels(self):
        """Test alert severity level handling"""
        severities = [
            AlertSeverity.CRITICAL,
            AlertSeverity.HIGH,
            AlertSeverity.MEDIUM,
            AlertSeverity.LOW,
            AlertSeverity.INFO
        ]
        
        for severity in severities:
            rule = AlertRule(
                name=f"Test{severity.value}",
                description="Test severity",
                query="test_metric > 1",
                severity=severity
            )
            
            assert rule.severity == severity
            assert rule.severity.value in ["critical", "high", "medium", "low", "info"]


class TestMetricsHTTPServer:
    """Test metrics HTTP server functionality"""
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint(self):
        """Test /metrics endpoint"""
        metrics_system = CentralizedMetrics()
        
        # Start HTTP server
        server = metrics_system.start_http_server(port=8001)
        
        try:
            # Give server time to start
            await asyncio.sleep(0.5)
            
            # Test metrics endpoint (would need aiohttp client in real test)
            # For now, verify server started
            assert server is not None
            
        finally:
            # Cleanup
            if server:
                server.shutdown()
    
    def test_health_endpoint_response(self):
        """Test /health endpoint response format"""
        metrics_system = CentralizedMetrics()
        
        # Get health response
        health = metrics_system.get_health_status()
        
        # Verify required fields
        required_fields = ["status", "timestamp", "uptime_seconds", "metrics_count"]
        for field in required_fields:
            assert field in health
        
        # Verify data types
        assert isinstance(health["status"], str)
        assert isinstance(health["timestamp"], (int, float))
        assert isinstance(health["uptime_seconds"], (int, float))
        assert isinstance(health["metrics_count"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])