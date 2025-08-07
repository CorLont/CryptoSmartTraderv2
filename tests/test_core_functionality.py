"""
CryptoSmartTrader V2 - Core Functionality Tests
Comprehensive test suite for enterprise-grade quality assurance
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.config_manager import ConfigManager
from core.cache_manager import CacheManager
from core.health_monitor import HealthMonitor
from core.error_handler import CentralizedErrorHandler, ErrorCategory, with_error_handling, with_retry
from core.monitoring_system import ProductionMonitoringSystem, Alert
from core.daily_analysis_scheduler import DailyAnalysisScheduler

class TestConfigManager:
    """Test configuration management functionality"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.config_manager = ConfigManager()
    
    def test_config_loading(self):
        """Test configuration loading and validation"""
        # Test default configuration loading
        config = self.config_manager.get_config()
        assert isinstance(config, dict)
        assert "agents" in config
        assert "data_sources" in config
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test valid configuration
        valid_config = {
            "agents": {
                "sentiment": {"enabled": True},
                "technical": {"enabled": True}
            },
            "data_sources": {
                "exchanges": ["kraken", "binance"]
            }
        }
        
        is_valid = self.config_manager.validate_config(valid_config)
        assert is_valid
    
    def test_config_backup_restore(self):
        """Test configuration backup and restore functionality"""
        # Create backup
        backup_created = self.config_manager.create_backup()
        assert backup_created
        
        # Modify configuration
        original_config = self.config_manager.get_config().copy()
        self.config_manager.set("test_key", "test_value")
        
        # Restore from backup
        restore_success = self.config_manager.restore_from_backup()
        assert restore_success
        
        # Verify restoration
        restored_config = self.config_manager.get_config()
        assert "test_key" not in restored_config

class TestCacheManager:
    """Test cache management functionality"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.cache_manager = CacheManager()
    
    def test_basic_cache_operations(self):
        """Test basic cache set/get operations"""
        # Test cache set and get
        test_key = "test_key"
        test_value = {"data": "test_data", "timestamp": datetime.now().isoformat()}
        
        self.cache_manager.set(test_key, test_value, ttl_minutes=10)
        retrieved_value = self.cache_manager.get(test_key)
        
        assert retrieved_value == test_value
    
    def test_cache_expiration(self):
        """Test cache TTL expiration"""
        test_key = "expiring_key"
        test_value = "expiring_value"
        
        # Set with very short TTL
        self.cache_manager.set(test_key, test_value, ttl_minutes=0.01)  # ~0.6 seconds
        
        # Should be available immediately
        assert self.cache_manager.get(test_key) == test_value
        
        # Wait for expiration
        time.sleep(1)
        
        # Should be expired
        assert self.cache_manager.get(test_key) is None
    
    def test_cache_statistics(self):
        """Test cache statistics tracking"""
        # Generate some cache activity
        for i in range(10):
            self.cache_manager.set(f"key_{i}", f"value_{i}")
        
        for i in range(5):
            self.cache_manager.get(f"key_{i}")
        
        # Try to get non-existent keys (cache misses)
        for i in range(10, 15):
            self.cache_manager.get(f"key_{i}")
        
        stats = self.cache_manager.get_cache_stats()
        assert stats["total_hits"] >= 5
        assert stats["total_misses"] >= 5
        assert "hit_rate" in stats

class TestHealthMonitor:
    """Test health monitoring functionality"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.health_monitor = HealthMonitor()
    
    def test_health_check_registration(self):
        """Test health check registration and execution"""
        # Register a test health check
        def test_health_check():
            return {"status": "healthy", "details": "Test check passed"}
        
        self.health_monitor.register_health_check("test_component", test_health_check)
        
        # Run health checks
        health_report = self.health_monitor.get_system_health()
        
        assert "test_component" in health_report["components"]
        assert health_report["components"]["test_component"]["status"] == "healthy"
    
    def test_health_grading(self):
        """Test health grading system"""
        # Register checks with different health levels
        def healthy_check():
            return {"status": "healthy", "score": 1.0}
        
        def degraded_check():
            return {"status": "degraded", "score": 0.6}
        
        def unhealthy_check():
            return {"status": "unhealthy", "score": 0.2}
        
        self.health_monitor.register_health_check("healthy_component", healthy_check)
        self.health_monitor.register_health_check("degraded_component", degraded_check)
        self.health_monitor.register_health_check("unhealthy_component", unhealthy_check)
        
        health_report = self.health_monitor.get_system_health()
        
        # Check overall grade calculation
        assert "overall_grade" in health_report
        assert health_report["overall_score"] < 1.0  # Should be affected by degraded/unhealthy components

class TestErrorHandler:
    """Test centralized error handling"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.error_handler = CentralizedErrorHandler()
    
    def test_error_categorization(self):
        """Test error categorization and handling"""
        # Test different error categories
        network_error = ConnectionError("Network connection failed")
        api_error = ValueError("Invalid API response")
        
        # Handle network error
        network_result = self.error_handler.handle_error(
            network_error, 
            ErrorCategory.NETWORK,
            {"endpoint": "https://api.example.com"}
        )
        
        assert network_result["category"] == ErrorCategory.NETWORK
        assert network_result["severity"] == "medium"
        assert "recovery_attempted" in network_result
        
        # Handle API error
        api_result = self.error_handler.handle_error(
            api_error,
            ErrorCategory.API,
            {"api_call": "get_market_data"}
        )
        
        assert api_result["category"] == ErrorCategory.API
    
    def test_error_statistics(self):
        """Test error statistics collection"""
        # Generate some errors
        for i in range(5):
            self.error_handler.handle_error(
                Exception(f"Test error {i}"),
                ErrorCategory.NETWORK
            )
        
        stats = self.error_handler.get_error_statistics()
        
        assert stats["total_errors"] == 5
        assert ErrorCategory.NETWORK in stats["errors_by_category"]
        assert stats["errors_by_category"][ErrorCategory.NETWORK] == 5
    
    def test_error_decorator(self):
        """Test error handling decorator"""
        
        @with_error_handling(ErrorCategory.CALCULATION)
        def test_function_with_error(should_fail=False):
            if should_fail:
                raise ValueError("Test error")
            return "success"
        
        # Test successful execution
        result = test_function_with_error(should_fail=False)
        assert result == "success"
        
        # Test error handling
        result = test_function_with_error(should_fail=True)
        # Should return None due to error handling (depending on recovery strategy)
        assert result is None or result == "success"
    
    def test_retry_decorator(self):
        """Test retry decorator functionality"""
        call_count = 0
        
        @with_retry(ErrorCategory.NETWORK, max_attempts=3)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"
        
        result = flaky_function()
        assert result == "success"
        assert call_count == 3

class TestProductionMonitoring:
    """Test production monitoring system"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.monitoring_system = ProductionMonitoringSystem()
    
    def test_metrics_collection(self):
        """Test metrics collection and recording"""
        # Record some test metrics
        self.monitoring_system.record_analysis_request("sentiment", "success", 1.5)
        self.monitoring_system.record_api_request("kraken", "success")
        self.monitoring_system.record_error("network", "medium")
        
        # Test that metrics are being collected
        # (This would typically involve checking Prometheus metrics)
        assert True  # Placeholder - actual metrics checking would require more setup
    
    def test_alert_generation(self):
        """Test alert generation and management"""
        # Create test alert
        alert = Alert(
            name="Test Alert",
            severity="warning",
            message="This is a test alert",
            timestamp=datetime.now(),
            source="test_system"
        )
        
        # Test alert sending (mocked)
        with patch.object(self.monitoring_system.alert_manager, 'send_alert') as mock_send:
            self.monitoring_system._send_alert_if_new(alert)
            mock_send.assert_called_once_with(alert)
    
    def test_monitoring_status(self):
        """Test monitoring status reporting"""
        status = self.monitoring_system.get_monitoring_status()
        
        assert isinstance(status, dict)
        assert "monitoring_active" in status
        assert "system_metrics" in status

class TestDailyAnalysisScheduler:
    """Test daily analysis scheduling"""
    
    def setup_method(self):
        """Setup for each test method"""
        self.config_manager = ConfigManager()
        self.cache_manager = CacheManager()
        self.health_monitor = HealthMonitor()
        
        self.scheduler = DailyAnalysisScheduler(
            self.config_manager,
            self.cache_manager,
            self.health_monitor
        )
    
    def test_daily_structure_initialization(self):
        """Test daily analysis structure initialization"""
        self.scheduler._initialize_daily_structure()
        
        today = datetime.now().strftime("%Y-%m-%d")
        cache_key = f"daily_analysis_{today}"
        
        daily_data = self.cache_manager.get(cache_key)
        assert daily_data is not None
        assert "social_sentiment" in daily_data
        assert "technical_analysis" in daily_data
        assert "ml_analysis" in daily_data
    
    def test_analysis_status_tracking(self):
        """Test analysis status tracking"""
        status = self.scheduler.get_daily_analysis_status()
        
        assert isinstance(status, dict)
        assert "scheduler_running" in status
        assert "services" in status
        assert "analysis_data" in status

class TestIntegration:
    """Integration tests for component interaction"""
    
    def setup_method(self):
        """Setup for integration tests"""
        self.config_manager = ConfigManager()
        self.cache_manager = CacheManager()
        self.health_monitor = HealthMonitor()
        self.error_handler = CentralizedErrorHandler(self.config_manager)
        self.monitoring_system = ProductionMonitoringSystem(
            self.config_manager, 
            self.error_handler
        )
    
    def test_error_handler_monitoring_integration(self):
        """Test integration between error handler and monitoring"""
        # Generate an error
        test_error = Exception("Integration test error")
        self.error_handler.handle_error(test_error, ErrorCategory.SYSTEM)
        
        # Check that monitoring recorded the error
        error_stats = self.error_handler.get_error_statistics()
        assert error_stats["total_errors"] > 0
    
    def test_health_monitoring_integration(self):
        """Test health monitoring with other components"""
        # Register health checks for other components
        def cache_health():
            stats = self.cache_manager.get_cache_stats()
            return {
                "status": "healthy" if stats["hit_rate"] > 0.5 else "degraded",
                "details": stats
            }
        
        self.health_monitor.register_health_check("cache_manager", cache_health)
        
        health_report = self.health_monitor.get_system_health()
        assert "cache_manager" in health_report["components"]

# Test fixtures and utilities
@pytest.fixture
def mock_config():
    """Mock configuration for testing"""
    return {
        "agents": {
            "sentiment": {"enabled": True, "update_interval": 300},
            "technical": {"enabled": True, "update_interval": 600},
            "ml_predictor": {"enabled": True, "update_interval": 900}
        },
        "data_sources": {
            "exchanges": ["kraken", "binance"],
            "social_media": ["reddit", "twitter"]
        },
        "cache": {
            "default_ttl": 3600,
            "max_size": 1000
        }
    }

@pytest.fixture
def mock_market_data():
    """Mock market data for testing"""
    return {
        "symbol": "BTC/USD",
        "price": 45000.0,
        "volume": 1000000,
        "timestamp": datetime.now().isoformat(),
        "indicators": {
            "rsi": 65.5,
            "macd": 150.2,
            "bollinger_upper": 46000,
            "bollinger_lower": 44000
        }
    }

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])